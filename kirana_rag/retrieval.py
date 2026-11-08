"""A small typed evidence graph with exact cosine and BM25 retrieval."""

import json
import math
import os
import re
import sqlite3
import tempfile
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .evidence import Edge, Evidence
from .ingest import Document
from .openrouter import OpenRouter

TOKEN = re.compile(r"[^\W_]+(?:[-.][^\W_]+)*", re.UNICODE)
QUARTER = re.compile(r"\b(first|second|third|fourth)[ -]+quarter\b")
QUARTER_NUMBERS = {"first": "q1", "second": "q2", "third": "q3", "fourth": "q4"}
STOPWORDS = set(
    "a an and are as at be by can did do does for from had has have how in is it its of on or "
    "per report say says than that the their they this to was were what when where which who "
    "why will with would you".split()
)
IDENTIFIER = re.compile(r"\b[A-Z]{2,}-[A-Z0-9]{2,}\b")
MONTHS = {
    "jan": "january", "feb": "february", "mar": "march", "apr": "april",
    "jun": "june", "jul": "july", "aug": "august", "sep": "september",
    "sept": "september", "oct": "october", "nov": "november", "dec": "december",
}
EDGE_WEIGHT = {
    "has_erratum": 5, "amends": 5, "erratum_context": 3, "context_for_erratum": 3,
    "table_continues": 3, "figure_note": 3, "scan_transcript": 3, "shared_identifier": 3,
    "table_row": 2, "part_of_table": 2, "next_on_page": 1, "same_figure": 1,
    "in_section": 0, "section_evidence": 0,
}
CONTEXT_RELATIONS = {"has_erratum", "amends", "table_continues", "figure_note", "scan_transcript"}


def tokens(text: str) -> list[str]:
    terms = []
    text = text.casefold().replace("–", "-").replace("‑", "-")
    text = QUARTER.sub(lambda match: QUARTER_NUMBERS[match[1]], text)
    for word in TOKEN.findall(text):
        word = MONTHS.get(word, word)
        if word in STOPWORDS:
            continue
        if len(word) > 5 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 4 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
            word = word[:-1]
        terms.append(word)
    return terms


def build_edges(items: list[Evidence]) -> list[Edge]:
    edges: set[Edge] = set()

    def link(a: Evidence, b: Evidence, relation: str, reverse: str | None = None):
        if a.id != b.id:
            edges.add(Edge(a.id, b.id, relation))
            edges.add(Edge(b.id, a.id, reverse or relation))

    sections = {}
    by_group = defaultdict(list)
    by_identifier = defaultdict(list)
    previous = None
    for item in items:
        if item.kind == "heading":
            sections[item.section] = item
        elif item.section in sections:
            link(item, sections[item.section], "in_section", "section_evidence")
        if item.group:
            by_group[item.group].append(item)
        for identifier in set(IDENTIFIER.findall(item.text)):
            by_identifier[identifier].append(item)
        if previous and previous.page == item.page:
            link(previous, item, "next_on_page")
        previous = item
    for group, members in by_group.items():
        if group.startswith("table-"):
            tables = [item for item in members if item.kind == "table"]
            for first, second in zip(tables, tables[1:]):
                link(first, second, "table_continues")
            table_by_page = {table.page: table for table in tables}
            for member in members:
                if member.kind == "table_row" and member.page in table_by_page:
                    link(member, table_by_page[member.page], "part_of_table", "table_row")
        else:
            for first, second in zip(members, members[1:]):
                link(first, second, "same_figure")
            notes = [member for member in members if member.kind == "visual_note"]
            scans = [member for member in members if member.kind == "scan" and member.method == "local_ocr"]
            for member in members:
                if member.kind != "visual_note":
                    for note in notes:
                        link(member, note, "figure_note", "same_figure")
                if member.method == "openrouter_vision":
                    for scan in scans:
                        link(member, scan, "scan_transcript", "same_figure")
    for members in by_identifier.values():
        for first, second in zip(members, members[1:]):
            link(first, second, "shared_identifier")

    for erratum in (item for item in items if item.kind == "erratum"):
        relevant = set(tokens(erratum.text))
        opening = re.split(r"(?<=[.!?])\s+", erratum.text, maxsplit=1)[0]
        original_terms = set(tokens(opening))
        original_values = {term for term in original_terms if term.isdecimal() and not (
            len(term) == 4 and term.startswith(("19", "20"))
        )}
        candidates = defaultdict(list)
        for item in items:
            if item.kind not in {"prose", "table_row", "figure"}:
                continue
            overlap = relevant & set(tokens(item.text))
            named_section = len(item.section.split()) >= 2 and (
                item.section.casefold() in erratum.text.casefold()
            )
            if len(overlap & original_terms) >= 4 and overlap & original_values:
                candidates[(item.section, "amends", "has_erratum")].append((len(overlap), item))
            elif named_section and len(overlap) >= 4 and any(
                word.isdecimal() and len(word) < 4 for word in overlap
            ):
                candidates[(item.section, "erratum_context", "context_for_erratum")].append(
                    (len(overlap), item)
                )
        for (_, relation, reverse), matches in candidates.items():
            _, best = min(matches, key=lambda match: (-match[0], match[1].id))
            link(erratum, best, relation, reverse)
    return sorted(edges, key=lambda edge: (edge.source, edge.target, edge.relation))


class Index:
    def __init__(self, metadata: dict, items: list[Evidence], edges: list[Edge],
                 vectors: np.ndarray | None):
        self.metadata = metadata
        self.items = items
        self.by_id = {item.id: item for item in items}
        self.adjacency: dict[str, list[Edge]] = defaultdict(list)
        for edge in edges:
            self.adjacency[edge.source].append(edge)
        self.vectors = vectors
        self.term_counts = [Counter(tokens(item.searchable)) for item in items]
        self.lengths = [sum(counts.values()) for counts in self.term_counts]
        self.average_length = max(sum(self.lengths) / len(items), 1)
        frequencies = Counter(term for counts in self.term_counts for term in counts)
        self.idf = {
            term: math.log1p((len(items) - count + 0.5) / (count + 0.5))
            for term, count in frequencies.items()
        }

    def keyword(self, query: str) -> list[Evidence]:
        terms = set(tokens(query)) & self.idf.keys()
        scores = []
        for item, counts, length in zip(self.items, self.term_counts, self.lengths):
            length_weight = 1.5 * (0.25 + 0.75 * length / self.average_length)
            score = sum(
                self.idf[term] * counts[term] * 2.5 / (counts[term] + length_weight)
                for term in sorted(terms)
            )
            if score:
                scores.append((score, item))
        scores.sort(key=lambda result: (-result[0], result[1].id))
        return [item for _, item in scores]

    def search(self, query: str, *, k: int = 7, vector: np.ndarray | None = None) -> list[Evidence]:
        keyword = self.keyword(query)
        if vector is None:
            ranked = keyword
        else:
            if self.vectors is None or vector.shape != (self.vectors.shape[1],):
                raise ValueError("Query embedding is incompatible with the index. Re-ingest the PDF.")
            dense = np.argsort(-(self.vectors @ vector), kind="stable")[:25]
            fused = defaultdict(float)
            for ranking in ([item.id for item in keyword[:25]], [self.items[i].id for i in dense]):
                for rank, identifier in enumerate(ranking, start=1):
                    fused[identifier] += 1 / (60 + rank)
            fused_order = sorted(fused, key=lambda i: (-fused[i], i))
            anchors = [item.id for item in keyword[:2]]
            ranked = [self.by_id[i] for i in dict.fromkeys(anchors + fused_order)]
        selected = []
        groups: Counter = Counter()
        deferred = []
        for item in ranked:
            if item.group and groups[item.group] >= 2:
                deferred.append(item)
                continue
            selected.append(item)
            groups[item.group] += 1
            if len(selected) == k:
                break
        return (selected + deferred)[:k]

    def expand(self, query: str, seeds: list[str], *, seen: set[str] | None = None,
               limit: int = 4, essential_only: bool = False) -> list[tuple[Evidence, Edge]]:
        ignored = set(seeds) | (seen or set())
        query_terms = set(tokens(query))
        candidates = []
        for seed in seeds:
            for edge in self.adjacency.get(seed, ()):
                if edge.target in ignored or (essential_only and edge.relation not in CONTEXT_RELATIONS):
                    continue
                item = self.by_id[edge.target]
                overlap = query_terms.intersection(tokens(item.searchable))
                score = EDGE_WEIGHT[edge.relation] + sum(self.idf.get(term, 0) for term in sorted(overlap))
                candidates.append((score, item, edge))
        chosen = []
        for _, item, edge in sorted(candidates, key=lambda candidate: (
            -candidate[0], candidate[1].id, candidate[2].source, candidate[2].relation
        )):
            if item.id not in ignored:
                chosen.append((item, edge))
                ignored.add(item.id)
            if len(chosen) == limit:
                break
        return chosen


def write_index(path: Path, document: Document, client: OpenRouter | None) -> dict:
    edges = build_edges(document.evidence)
    vectors = client.embed([item.searchable for item in document.evidence]) if client else None
    metadata = {
        "schema": 1, "name": document.name, "sha256": document.digest, "pages": document.pages,
        "evidence_count": len(document.evidence), "edge_count": len(edges),
        "embedding_model": client.embed_model if client else None,
        "vision_model": client.vision_model if client else None,
        "embedding_dimensions": vectors.shape[1] if vectors is not None else 0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".sqlite", delete=False) as temporary:
        scratch = Path(temporary.name)
    try:
        with closing(sqlite3.connect(scratch)) as connection, connection:
            connection.executescript("""
                CREATE TABLE metadata (payload TEXT NOT NULL);
                CREATE TABLE evidence (id TEXT PRIMARY KEY, payload TEXT NOT NULL, vector BLOB);
                CREATE TABLE edge (source TEXT NOT NULL, target TEXT NOT NULL, relation TEXT NOT NULL);
                CREATE INDEX edge_source ON edge (source);
            """)
            connection.execute("INSERT INTO metadata VALUES (?)", (json.dumps(metadata),))
            connection.executemany("INSERT INTO evidence VALUES (?, ?, ?)", (
                (item.id, json.dumps(asdict(item)),
                 vectors[i].astype("<f4", copy=False).tobytes() if vectors is not None else None)
                for i, item in enumerate(document.evidence)
            ))
            connection.executemany("INSERT INTO edge VALUES (?, ?, ?)", (
                (edge.source, edge.target, edge.relation) for edge in edges
            ))
        os.replace(scratch, path)
    finally:
        scratch.unlink(missing_ok=True)
    return metadata


def read_index(path: Path) -> Index:
    if not path.is_file():
        raise ValueError(f"No index at {path}. Run ingest first.")
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        metadata = json.loads(connection.execute("SELECT payload FROM metadata").fetchone()[0])
        if metadata.get("schema") != 1:
            raise ValueError("Unsupported index version. Re-ingest the PDF.")
        rows = connection.execute("SELECT payload, vector FROM evidence ORDER BY id").fetchall()
        edges = [Edge(*row) for row in connection.execute("SELECT source, target, relation FROM edge")]
    vectors = np.stack([np.frombuffer(row[1], dtype="<f4") for row in rows]) if metadata[
        "embedding_dimensions"
    ] else None
    return Index(metadata, [Evidence(**json.loads(payload)) for payload, _ in rows], edges, vectors)
