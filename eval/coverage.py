"""Audit the indexed evidence for the supplied Kirana PDF locally."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from kirana_rag.retrieval import read_index

REPORT_SHA = "109f0c113b5d8184ff720b87d69d3d9c7c236cf3be4aff5cd02b50addea43605"

# Body fingerprints came from an independent pdftotext -layout extraction.
# Keep header expectations separate so a data row cannot pass as the header.
TABLES = (
    (4, 1, ("THB million", "2023", "2024", "2025"), 6,
     "a952f9d32b6c0c2ce4ddaddc7ca66a9319a056c5bf016591126763e7dbb00027"),
    (7, 2, ("Code", "Location", "Country", "Type", "Area (sqm)", "Utilisation 2025", "Opened"), 10,
     "25688d9f0277b5b744ea8b368d329687a80fe35c2de8588bf4efd507c6583cb0"),
    (8, 2, ("Code", "Location", "Country", "Type", "Area (sqm)", "Utilisation 2025", "Opened"), 13,
     "f1452dc2e93a59bcc412739c3b8d628f0531128654d9e012689405eae229c649"),
    (9, 3, ("Vehicle type", "Thailand", "Indonesia", "Vietnam", "Malaysia", "Total"), 4,
     "b984cf74a87bfaa5b1969a3ebc77fa2fc0c7c6388ec0eca3bed9bd28fa0c1516"),
    (9, 4, ("", "2023", "2024", "2025"), 3,
     "a3aa6cfba23298ba9a3cd095ce963bf052c0b495d2895031268ca56b239e65b8"),
    (10, 5, ("Country", "Employees 2024", "Employees 2025"), 5,
     "46035a281449a0f34b23df6aa04d4d105172a2a5001870c90d25ef4edb297489"),
)


def number(text: str) -> str:
    return rf"(?<![\d.]){re.escape(text)}(?![\d.])"


def check(path: Path) -> list[str]:
    index = read_index(path)
    if index.metadata["sha256"] != REPORT_SHA:
        raise ValueError("The coverage checks apply to the supplied Kirana report.")
    failures = []
    native_count = 0
    for page, table, header, count, expected in TABLES:
        title = f"Table {table} on page {page}"
        matches = [item for item in index.items if item.page == page and item.kind == "table"
                   and item.label.startswith(f"Table {table}.")]
        if len(matches) != 1:
            failures.append(f"{title}: expected one table.")
            continue
        item = matches[0]
        columns, *rows = [[cell.strip() for cell in line.split("|")]
                          for line in item.text.splitlines()]
        fingerprint = hashlib.sha256(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode())
        if tuple(columns) != header or len(rows) != count or fingerprint.hexdigest() != expected:
            failures.append(f"{title}: header or table cells differ from the PDF.")
            continue
        separate_rows = [row.text for row in index.items if row.page == page and
                         row.group == item.group and row.kind == "table_row"]
        expected_rows = [" | ".join(f"{key}: {value}" if key else value
                                    for key, value in zip(header, row)) for row in rows]
        if separate_rows != expected_rows:
            failures.append(f"{title}: individual rows lost data or column labels.")
            continue
        native_count += count
    if [page["page"] for page in index.metadata["pages"] if page["evidence"]] != list(range(1, 14)):
        failures.append("Expected indexed evidence on all 13 pages.")
    print(f"Native table rows matched: {native_count}/41")

    visual_records: dict[int, list[str]] = {}
    for item in index.items:
        if item.method == "vision":
            statement = item.text.partition("\nVisible labels:")[0]
            record = re.sub(r"(?<=\d),(?=\d{3}\b)", "", f"{item.label} | {statement}")
            visual_records.setdefault(item.page, []).append(record)

    def visual(page: int, name: str, *patterns: str):
        if not any(all(re.search(pattern, text, re.I) for pattern in patterns)
                   for text in visual_records.get(page, [])):
            failures.append(f"Page {page}: missing structured evidence for {name}.")

    for label, values in (
        ("Revenue", ("8420", "10.0%")), ("Parcels delivered", ("41.6",)),
        ("Vehicles in fleet", ("1240",)), ("Warehouses", ("37", "566000")),
        ("Customer NPS", ("52", "47", "2024")), ("Employees", ("6850",)),
    ):
        visual(3, label, re.escape(label), *(number(value) for value in values))
    for quarter, amount in zip(("Q1", "Q2", "Q3", "Q4"), ("1962", "2048", "2105", "2305")):
        visual(5, quarter, rf"\b{quarter}\b", number(amount), r"THB million")
    for country, share in (("Thailand", "46%"), ("Indonesia", "31%"), ("Vietnam", "14%"), ("Malaysia", "9%")):
        visual(5, country, country, number(share))
    months = (
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    )
    monthly = ("94.1", "94.6", "95.0", "94.8", "95.3", "95.6", "95.1", "95.4", "94.2", "91.8", "89.7", "92.9")
    for month, rate in zip(months, monthly):
        visual(6, month, rf"\b(?:{month}|{month[:3]})\b(?:\s+2025)?[\s:|=]+{number(rate)}\s*%")
    for start, end, rate in (
        ("Chiang Mai", "Lat Krabang", "27 trucks/day"), ("Lat Krabang", "Hat Yai", "33 trucks/day"),
        ("Hat Yai", "Shah Alam", "18 trucks/day"), ("Hai Phong", "Binh Duong", "41 trucks/day"),
        ("Laem Chabang", "Cikarang", "3 sailings per week"),
        ("Cikarang", "Semarang", "29 trucks/day"), ("Cikarang", "Surabaya", "64 trucks/day"),
    ):
        visual(7, f"{start} / {end}", start, end, number(rate))
    visual(7, "sea route label", "Laem Chabang", "Cikarang", "Tanjung Priok", "3 sailings per week")
    for year, intensity in zip(range(2021, 2026), ("3.05", "2.88", "2.71", "2.58", "2.40")):
        visual(9, f"{year} emissions intensity", number(str(year)), number(intensity), "CO2e per parcel")
    for label, amount, unit in (
        ("Ambient pallet storage", "420.00", "per pallet per month"),
        ("Cold-chain pallet storage", "780.00", "per pallet per month"),
        ("Frozen pallet storage", "1150.00", "per pallet per month"),
        ("Pick and pack", "18.50", "per order"),
        ("Line-haul FTL Bangkok", "21500.00", "per trip"),
        ("Last-mile parcel", "38.00", "per parcel"),
        ("Returns handling", "12.00", "per item"),
    ):
        visual(13, label, label, number(amount), unit, "THB")
    visual(13, "VAT and effective date", r"exclude\s+7%\s+VAT", "1 January 2026")
    visual(13, "Schedule C is absent", "Schedule C", "not included")
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path(".rag/index.sqlite3"))
    try:
        problems = check(parser.parse_args().index)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    if problems:
        raise SystemExit("\n".join(problems))
    print("Coverage verified: 13 pages, 6 KPI tiles, 25 chart points, 7 routes, and 7 tariff rows with both notes.")
