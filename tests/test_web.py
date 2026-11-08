import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf

from kirana_rag.evidence import Evidence
from kirana_rag.retrieval import Index
from kirana_rag.web import Workbench, highlight_region, render_markdown


class WebTests(unittest.TestCase):
    def test_highlight_aligns_with_rotated_page(self):
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=100, height=200)
            page.set_rotation(90)
            region = highlight_region(page, (10, 20, 30, 40))
        self.assertEqual(region, {"left": 0.8, "top": 0.1, "width": 0.1, "height": 0.2})

    def test_workbench_rejects_a_different_pdf_and_marks_the_cited_page(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.pdf"
            with pymupdf.open() as pdf:
                pdf.new_page(width=100, height=200)
                pdf.new_page(width=200, height=100)
                pdf.save(path)
            box = (20, 10, 60, 30)
            item = Evidence("row", 2, "prose", "", "Sample", "Sales total", box)
            index = Index({"sha256": "wrong"}, [item], [], None)
            with self.assertRaisesRegex(ValueError, "differs from the indexed source"):
                Workbench(path, index)
            index.metadata["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            workbench = Workbench(path, index)
            self.assertTrue(workbench.page_png(2).startswith(b"\x89PNG\r\n\x1a\n"))
            draft = {"status": "answered", "message": "", "claims": [
                {"statement": "Sales total", "citations": [{"page": 2, "box": box}]}
            ]}
            with patch("kirana_rag.web.OpenRouter.from_env"), patch("kirana_rag.web.Agent") as agent:
                agent.return_value.answer.return_value = draft
                response = workbench.ask("What were sales?")
            self.assertEqual(response["claims"][0]["citations"][0]["highlight"],
                             {"left": 0.1, "top": 0.1, "width": 0.2, "height": 0.2})

    def test_readme_renders_tables_and_escapes_raw_html(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "README.md"
            path.write_text("# Guide\n\n| Tool |\n| --- |\n| Chat |\n\n<script>bad()</script>",
                            encoding="utf-8")
            rendered = render_markdown(path)
        self.assertIn(b"<h1>Guide</h1>", rendered)
        self.assertIn(b"<table>", rendered)
        self.assertIn(b"&lt;script&gt;", rendered)
        self.assertNotIn(b"<script>", rendered)
