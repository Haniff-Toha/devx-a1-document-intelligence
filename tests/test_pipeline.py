import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageDraw, ImageFont

from kirana_rag.agent import Agent, validate_claims
from kirana_rag.evidence import Edge, Evidence
from kirana_rag.ingest import ingest
from kirana_rag.llm_provider import LLMProvider, ProviderError
from kirana_rag.retrieval import Index, build_edges, read_index, write_index

BOX = (10, 10, 100, 20)


def make_pdf(path: Path, name: str = "Northstar Logistics", solar: int = 43):
    """Create an independent input with native text and a raster-only second page."""
    image = Image.new("RGB", (1200, 1400), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=42)
    draw.text((75, 100), "Warehouse service tariff", font=font, fill="black")
    draw.text((75, 190), "Handling surcharge: THB 74 per order", font=font, fill="black")
    draw.text((75, 280), "All rates exclude VAT.", font=font, fill="black")
    png = io.BytesIO()
    image.save(png, format="PNG")
    with pymupdf.open() as pdf:
        first = pdf.new_page()
        first.insert_text((54, 64), name, fontsize=20)
        first.insert_text((54, 102), f"Solar supplied {solar}% of warehouse electricity.", fontsize=12)
        second = pdf.new_page()
        second.insert_image(second.rect, stream=png.getvalue())
        pdf.save(path)


def evidence(identifier, page, text, *, kind="prose", section="", group=""):
    return Evidence(identifier, page, kind, section, section, text, BOX, group)


class EvidenceTests(unittest.TestCase):
    def test_graph_distinguishes_amended_source_from_corroboration(self):
        original = evidence("a", 1, "In 2025, 96 electric trucks were added.", section="Chair Letter")
        signature = evidence("b", 1, "Chair Letter signed by the chair in 2025.", section="Chair Letter")
        corroboration = evidence("c", 2, "In 2025, 82 electric trucks were added.", section="Fleet Review")
        note = evidence("d", 3, (
            "The Chair Letter states that 96 electric trucks were added in 2025. "
            "The correct figure is 82, as reported in the Fleet Review section."
        ), kind="erratum", section="Errata")
        links = build_edges([original, signature, corroboration, note])
        self.assertIn(Edge("a", "d", "has_erratum"), links)
        self.assertIn(Edge("d", "c", "erratum_context"), links)
        self.assertNotIn(Edge("d", "b", "amends"), links)
        self.assertNotIn(Edge("d", "c", "amends"), links)

    def test_table_can_continue_across_pages(self):
        row = evidence("a", 1, "warehouse A", kind="table_row", group="table-1")
        first = evidence("b", 1, "first part", kind="table", group="table-1")
        next_page = evidence("c", 2, "second part", kind="table", group="table-1")
        self.assertIn(Edge("a", "b", "part_of_table"), build_edges([row, first, next_page]))
        self.assertIn(Edge("b", "c", "table_continues"), build_edges([row, first, next_page]))

    def test_hybrid_adds_vector_candidate_with_no_shared_keyword(self):
        items = [evidence("a", 1, "turnover 112"), evidence("b", 2, "income 115"),
                 evidence("c", 3, "staff count 38")]
        index = Index({}, items, [], np.array([[0., 1.], [1., 0.], [-1., 0.]]))
        self.assertEqual([item.id for item in index.search("turnover")], ["a"])
        self.assertEqual({item.id for item in index.search(
            "turnover", k=2, vector=np.array([1., 0.])
        )}, {"a", "b"})

    def test_quotes_must_be_in_context_and_contain_claimed_numbers(self):
        item = evidence("a", 1, "Solar supplied 43% of warehouse electricity.")
        valid = {"statement": "Solar supplied 43%.", "support": [
            {"evidence_id": "a", "quote": item.text}
        ]}
        self.assertTrue(validate_claims([valid], {"a": item})[0])
        self.assertFalse(validate_claims([valid], {})[0])
        self.assertFalse(validate_claims([{**valid, "statement": "Solar supplied 73%."}], {"a": item})[0])
        canonical = validate_claims([{**valid, "support": [
            {"evidence_id": "a", "quote": "Solar supplied 43% of fleet electricity."}
        ]}], {"a": item})[0]
        self.assertEqual(canonical[0]["citations"][0]["quote"], item.text)


class ScriptedModel:
    embed_model = "test"

    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.observed = []

    def structured(self, _instructions, content, _schema, **_):
        self.observed.append(json.loads(content))
        return next(self.decisions)


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.vector = np.array([1., 0.])
        items = [evidence(f"a{i}", 1, "rail schedule") for i in range(7)]
        items.append(evidence("b", 2, "The final depot was Harborside."))
        vectors = np.array([[1., 0.]] * 7 + [[-1., 0.]])
        self.index = Index({"name": "sample.pdf", "embedding_model": "test"}, items, [
            Edge("a0", "b", "next_on_page")
        ], vectors)

    def test_agent_calls_graph_tool_and_can_cite_discovered_evidence(self):
        client = ScriptedModel([
            {"action": "expand", "evidence_ids": ["a0"]},
            {"action": "answer", "claims": [{"statement": "The final depot was Harborside.",
             "support": [{"evidence_id": "b", "quote": "The final depot was Harborside."}]}]},
            {"verdict": "supported"},
        ])
        response = Agent(self.index, client).answer("Which rail destination?", vector=self.vector)
        self.assertEqual(response["status"], "answered")
        self.assertNotIn("b", response["initial_context"])
        self.assertEqual(response["claims"][0]["citations"][0]["page"], 2)
        self.assertEqual(response["trace"][2]["action"], "expand")

    def test_agent_rejects_unknown_graph_seed_and_stops(self):
        client = ScriptedModel([{"action": "expand", "evidence_ids": ["invented"]}] * 3)
        response = Agent(self.index, client).answer("Unknown depot?", vector=self.vector)
        self.assertEqual(response["status"], "refused")
        self.assertEqual(response["reason"], "validation_or_tool_budget_exhausted")
        self.assertEqual(len(client.observed), 3)

    def test_semantic_review_can_block_a_valid_quote_for_the_wrong_claim(self):
        client = ScriptedModel([
            {"action": "answer", "claims": [{"statement": "The depot is Southbank.",
             "support": [{"evidence_id": "a0", "quote": "rail schedule"}]}]},
            {"verdict": "incorrect"},
            {"action": "refuse", "claims": []},
        ])
        response = Agent(self.index, client).answer("Which rail destination?", vector=self.vector)
        self.assertEqual(response["status"], "refused")
        self.assertFalse(response["claims"])
        self.assertIn({"action": "review_claims", "verdict": "incorrect"}, response["trace"])


class ProviderTests(unittest.TestCase):
    def test_embedding_response_order_is_restored_and_normalized(self):
        client = LLMProvider("test", "test", "test", base_url="https://example.test/v1")
        client._post = lambda *_: {"data": [
            {"index": 1, "embedding": [0, 3]}, {"index": 0, "embedding": [2, 0]}
        ]}
        np.testing.assert_array_equal(client.embed(["first", "second"]), np.eye(2))

    def test_missing_embedding_rows_are_rejected(self):
        client = LLMProvider("test", "test", "test", base_url="https://example.test/v1")
        client._post = lambda *_: {"data": [{"index": 1, "embedding": [0, 3]}]}
        with self.assertRaises(ProviderError):
            client.embed(["first", "second"])


class IngestionTests(unittest.TestCase):
    def test_captioned_table_retains_year_headers_after_blank_first_cell(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "metrics.pdf"
            with pymupdf.open() as pdf:
                page = pdf.new_page()
                page.insert_text((54, 80), "Table 1. Annual metrics", fontsize=11)
                for y, label, columns in (
                    (105, "", ["2023", "2024", "2025"]),
                    (125, "Metric A", ["1,000", "1,200", "1,300"]),
                    (145, "Metric B", ["450", "500", "550"]),
                ):
                    if label:
                        page.insert_text((65, y), label, fontsize=9)
                    for x, text in zip((330, 410, 490), columns):
                        width = pymupdf.get_text_length(text, fontsize=9)
                        page.insert_text((x - width, y), text, fontsize=9)
                pdf.save(path)
            records = ingest(path).evidence
        self.assertEqual([item.text for item in records if item.kind == "table_row"], [
            "Metric A | 2023: 1,000 | 2024: 1,200 | 2025: 1,300",
            "Metric B | 2023: 450 | 2024: 500 | 2025: 550",
        ])

    def test_second_pdf_replaces_prior_index_and_retains_scan_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first, second, database = root / "first.pdf", root / "second.pdf", root / "index.sqlite3"
            make_pdf(first, name="Inland Logistics", solar=29)
            make_pdf(second)
            write_index(database, ingest(first), None)
            before = read_index(database)
            write_index(database, ingest(second), None)
            after = read_index(database)
            self.assertNotEqual(before.metadata["sha256"], after.metadata["sha256"])
            self.assertEqual(after.metadata["name"], "second.pdf")
            self.assertEqual(len(after.metadata["pages"]), 2)
            self.assertFalse(any("29%" in item.text for item in after.items))
            self.assertTrue(any("43%" in item.text and item.page == 1 for item in after.items))
            self.assertTrue(any("74" in item.text and item.page == 2 and item.method == "local_ocr"
                                for item in after.items))


if __name__ == "__main__":
    unittest.main()
