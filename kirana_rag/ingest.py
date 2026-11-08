"""Extract PDF evidence with page, region, and method provenance."""

import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np
import pymupdf
from PIL import Image

from .evidence import Evidence
from .llm_provider import LLMProvider, ProviderError

CAPTION = re.compile(r"^(?:Table|Figure|Exhibit)\s+[\w.:-]+\b", re.I)
TABLE = re.compile(r"^Table\s+[\w.:-]+\b", re.I)
PAGE_NUMBER = re.compile(r"^(?:page\s+)?\d+(?:\s+of\s+\d+)?$", re.I)
Cell = tuple[str, float, float]


@dataclass
class Document:
    name: str
    digest: str
    pages: list[dict]
    evidence: list[Evidence]


def clean(text: str) -> str:
    return " ".join(text.split())


def bounds(rect: pymupdf.Rect | tuple) -> tuple[float, float, float, float]:
    return tuple(round(float(value), 1) for value in rect)


def chunks(text: str, size: int = 120, overlap: int = 20):
    words = text.split()
    for start in range(0, len(words), size - overlap):
        yield " ".join(words[start:start + size])
        if start + size >= len(words):
            return


class PageEvidence:
    def __init__(self, number: int):
        self.number = number
        self.items: list[Evidence] = []
        self.notes: list[str] = []

    def add(self, kind: str, section: str, label: str, text: str, box: tuple, *,
            group: str = "", method: str = "pdf_text"):
        if not text.strip():
            return
        identifier = f"p{self.number:03d}-{len(self.items) + 1:03d}"
        self.items.append(Evidence(
            identifier, self.number, kind, section, label, text.strip(),
            bounds(box), group, method,
        ))


def block_text(block: dict) -> str:
    return clean(" ".join(
        "".join(span["text"] for span in line["spans"])
        for line in block["lines"]
    ))


def margin_key(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower())


def native_blocks(page: pymupdf.Page) -> list[dict]:
    flags = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES
    return sorted(
        (block for block in page.get_text("dict", flags=flags)["blocks"]
         if block.get("lines") and block_text(block)),
        key=lambda block: (round(block["bbox"][1], 1), block["bbox"][0]),
    )


def in_margin(block: dict, height: float) -> bool:
    return block["bbox"][3] < height * 0.06 or block["bbox"][1] > height * 0.94


def table_cells(block: dict) -> list[Cell] | None:
    """Recognize a row when separate text columns share a baseline."""
    lines = []
    for line in block["lines"]:
        text = clean("".join(span["text"] for span in line["spans"]))
        if text:
            lines.append((line["bbox"], text))
    if len(lines) < 2:
        return None
    line_height = median(box[3] - box[1] for box, _ in lines)
    if block["bbox"][3] - block["bbox"][1] > 2.5 * line_height:
        return None
    columns: list[list[tuple]] = []
    for box, text in sorted(lines, key=lambda row: row[0][0]):
        if columns and box[0] < max(other[2] for other, _ in columns[-1]) - 1:
            columns[-1].append((box, text))
        else:
            columns.append([(box, text)])
    if len(columns) < 2:
        return None
    if not any(
        sum(any(abs(box[3] - candidate[3]) <= 0.55 * line_height
                for box, _ in column) for column in columns) >= 2
        for candidate, _ in lines
    ):
        return None
    return [
        (
            " ".join(text for _, text in sorted(col, key=lambda cell: cell[0][1])),
            min(box[0] for box, _ in col),
            max(box[2] for box, _ in col),
        )
        for col in columns
    ]


def table_header(cells: list[Cell] | None, first_row: list[Cell] | None, *,
                 captioned: bool) -> tuple[str, ...]:
    if not cells or not first_row:
        return ()
    labels = tuple(text for text, _, _ in cells)
    if len(cells) == len(first_row):
        return labels
    if not captioned or len(first_row) != len(cells) + 1 or first_row[0][2] >= cells[0][1]:
        return ()
    # A captioned table may leave the first column header empty. The remaining
    # headers must line up with their data cells by left edge, right edge, or center.
    for (_, left, right), (_, row_left, row_right) in zip(cells, first_row[1:]):
        if min(abs(left - row_left), abs(right - row_right),
               abs(left + right - row_left - row_right) / 2) > 4:
            return ()
    return ("", *labels)


def heading(block: dict, typical_size: float) -> bool:
    text = block_text(block)
    sizes = [span["size"] for line in block["lines"] for span in line["spans"]]
    return len(text) <= 110 and len(text.split()) <= 14 and median(sizes) >= typical_size * 1.12


def native_evidence(blocks: list[dict], output: PageEvidence, section: str,
                    previous_table: tuple | None) -> tuple[str, tuple | None]:
    sizes = [span["size"] for block in blocks for line in block["lines"] for span in line["spans"]]
    typical = median(sizes) if sizes else 10
    rows = [table_cells(block) for block in blocks]
    pending_table = None
    last_table = None
    position = 0
    while position < len(blocks):
        block = blocks[position]
        text = block_text(block)
        following = rows[position + 1] if position + 1 < len(rows) else None
        header = table_header(rows[position], following, captioned=pending_table is not None)
        if TABLE.match(text):
            pending_table = text
            output.add("caption", section, text, text, block["bbox"])
        elif CAPTION.match(text):
            output.add("caption", section, text, text, block["bbox"])
        elif header:
            end = position + 1
            while end < len(rows) and rows[end] and len(rows[end]) == len(header):
                end += 1
            if not pending_table and previous_table and previous_table[0] == header:
                _, title, group = previous_table
            else:
                title = pending_table or f"{section} table".strip()
                group = f"table-{output.number}-{position}"
            row_texts = []
            for row_index in range(position + 1, end):
                values = [value for value, _, _ in rows[row_index]]
                row_text = " | ".join(
                    f"{key}: {value}" if key else value for key, value in zip(header, values)
                )
                output.add("table_row", section, f"{title} / {values[0]}", row_text,
                           blocks[row_index]["bbox"], group=group, method="pdf_layout")
                row_texts.append(" | ".join(values))
            table_box = pymupdf.Rect(block["bbox"]) | pymupdf.Rect(blocks[end - 1]["bbox"])
            output.add("table", section, title, "\n".join([" | ".join(header), *row_texts]),
                       table_box, group=group, method="pdf_layout")
            last_table = (header, title, group)
            pending_table = None
            position = end
            continue
        elif heading(block, typical):
            section = text
            pending_table = None
            output.add("heading", section, section, text, block["bbox"])
        else:
            kind = "erratum" if re.match(r"^(?:errata|erratum|corrections?)\b", section, re.I) else "prose"
            for part in chunks(text):
                output.add(kind, section, section, part, block["bbox"])
        position += 1
    return section, last_table


def ocr_rows(result: list) -> str:
    hits = []
    for points, text, confidence in result:
        if confidence < 0.55 or not text.strip():
            continue
        left = min(point[0] for point in points)
        top = min(point[1] for point in points)
        bottom = max(point[1] for point in points)
        hits.append(((top + bottom) / 2, left, bottom - top, text.strip()))
    if not hits:
        return ""
    tolerance = max(3, median(hit[2] for hit in hits) * 0.42)
    rows: list[list[tuple]] = []
    for hit in sorted(hits):
        if rows and abs(hit[0] - rows[-1][0][0]) <= tolerance:
            rows[-1].append(hit)
        else:
            rows.append([hit])
    return "\n".join(" | ".join(hit[3] for hit in sorted(row, key=lambda item: item[1]))
                     for row in rows)


def image_evidence(page: pymupdf.Page, blocks: list[dict], output: PageEvidence,
                   section: str, ocr, client: LLMProvider | None):
    image_boxes = {}
    for image in page.get_image_info():
        box = pymupdf.Rect(image["bbox"]) & page.rect
        if box.width >= 50 and box.height >= 40:
            image_boxes[bounds(box)] = box
    captions = [(block_text(block), pymupdf.Rect(block["bbox"])) for block in blocks
                if CAPTION.match(block_text(block)) and not TABLE.match(block_text(block))]
    for sequence, box in enumerate(image_boxes.values(), start=1):
        scan = box.get_area() / page.rect.get_area() > 0.60 or not blocks
        kind = "scan" if scan else "figure"
        region_section = "" if scan and not blocks else section
        label = f"{'Scanned region' if scan else 'Figure region'} {sequence}"
        nearby = [(min(abs(cap.y1 - box.y0), abs(cap.y0 - box.y1)), title)
                  for title, cap in captions if cap.x1 >= box.x0 and cap.x0 <= box.x1]
        if nearby and min(nearby)[0] < 30:
            label = min(nearby)[1]
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2.6, 2.6), clip=box)
        png = pixmap.tobytes("png")
        rgb = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
        result, _ = ocr(np.ascontiguousarray(rgb[:, :, ::-1]))
        transcription = ocr_rows(result or [])
        group = f"visual-{output.number}-{sequence}"
        if transcription:
            output.add(kind, region_section, label, transcription, box, group=group, method="local_ocr")
        if client is None:
            if not transcription:
                output.notes.append(f"{label}: no readable text detected by OCR.")
            continue
        try:
            visual = client.read_figure(png)
        except ProviderError as exc:
            output.notes.append(f"{label}: vision unavailable; retained local OCR if readable. {exc}")
            continue
        visual_label = label if not isinstance(visual.get("label"), str) else visual["label"].strip() or label
        context = visual.get("context")
        context = clean(context)[:260] if isinstance(context, str) else ""
        records = visual.get("records")
        if not isinstance(records, list) or not records:
            if not transcription:
                output.notes.append(f"{label}: no readable text or visual evidence detected.")
            continue
        for record in records[:16]:
            if not isinstance(record, dict):
                continue
            text = record.get("text")
            title = record.get("label")
            anchors = record.get("visible_labels")
            if not (isinstance(text, str) and isinstance(title, str) and
                    isinstance(anchors, list) and anchors and all(isinstance(x, str) for x in anchors)):
                continue
            source = " | ".join(anchor.strip() for anchor in anchors if anchor.strip())
            if source:
                record_kind = "visual_note" if record.get("role") == "note" else kind
                parts = [text, f"Visible labels: {source}"]
                if context:
                    parts.append(f"Visible header: {context}")
                output.add(record_kind, region_section or visual_label, f"{visual_label} / {title}",
                           "\n".join(parts), box, group=group,
                           method="vision")


def ingest(path: Path, client: LLMProvider | None = None) -> Document:
    from rapidocr_onnxruntime import RapidOCR

    pdf_bytes = path.read_bytes()
    digest = hashlib.sha256(pdf_bytes).hexdigest()
    pages = []
    all_evidence = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
        if pdf.needs_pass:
            raise ValueError("The PDF is password-protected.")
        layout = [native_blocks(page) for page in pdf]
        margin_counts: dict[str, int] = {}
        for page, blocks in zip(pdf, layout):
            keys = {margin_key(block_text(block)) for block in blocks
                    if in_margin(block, page.rect.height)}
            for key in keys:
                margin_counts[key] = margin_counts.get(key, 0) + 1
        ocr = None
        section = ""
        previous_table = None
        for page, blocks in zip(pdf, layout):
            blocks = [block for block in blocks if not (
                in_margin(block, page.rect.height) and (
                    margin_counts[margin_key(block_text(block))] > 1 or PAGE_NUMBER.fullmatch(block_text(block))
                )
            )]
            output = PageEvidence(page.number + 1)
            section, previous_table = native_evidence(blocks, output, section, previous_table)
            if page.get_image_info():
                if ocr is None:
                    ocr = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)
                image_evidence(page, blocks, output, section, ocr, client)
            if not output.items:
                output.notes.append("No indexed evidence on this page.")
            pages.append({"page": output.number, "evidence": len(output.items), "notes": output.notes})
            all_evidence.extend(output.items)
    if not all_evidence:
        raise ValueError("No readable evidence was extracted from the PDF.")
    return Document(path.name, digest, pages, all_evidence)
