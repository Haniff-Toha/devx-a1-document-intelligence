import hashlib
import json
import socket
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path
from unittest.mock import Mock, patch

import pymupdf
import uvicorn

from kirana_rag.evidence import Evidence
from kirana_rag.retrieval import Index
from kirana_rag.visuals import catalog, requested_visuals
from kirana_rag.web import Workbench, create_app, highlight_region, render_markdown


class WebTests(unittest.TestCase):
    def test_visual_lookup_distinguishes_chart_inventory_from_chart_data(self):
        box = (10, 20, 180, 100)
        items = [
            Evidence("ocr", 1, "figure", "", "Figure region", "Figure 1. Revenue by quarter",
                     box, "visual-1-1", "local_ocr"),
            Evidence("chart", 1, "figure", "", "Revenue by quarter / Q4", "Q4 revenue",
                     box, "visual-1-1", "vision"),
            Evidence("kpi", 1, "figure", "", "Key performance indicators / NPS", "NPS",
                     box, "visual-1-2", "vision"),
        ]
        visuals = catalog(items)
        self.assertEqual([visual.source.id for visual in requested_visuals(
            "is there any chart in our pdf?", visuals)], ["chart"])
        self.assertEqual([visual.source.id for visual in requested_visuals(
            "is there anychart on the documents?", visuals)], ["chart"])
        self.assertEqual([visual.source.id for visual in requested_visuals(
            "Show Figure 1", visuals)], ["chart"])
        self.assertEqual(requested_visuals("Show the profit chart", visuals), [])
        self.assertIsNone(requested_visuals("What is Q4 revenue in the chart?", visuals))

    def test_visual_catalog_returns_a_cropped_source_without_model_calls(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.pdf"
            with pymupdf.open() as pdf:
                pdf.new_page(width=200, height=200)
                pdf.save(path)
            item = Evidence("chart", 1, "figure", "", "Revenue by month / June", "June revenue",
                            (20, 30, 100, 90), "visual-1-1", "vision")
            index = Index({"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, [item], [], None)
            workbench = Workbench(path, index)
            with patch("kirana_rag.web.LLMProvider.from_env", side_effect=AssertionError("unexpected model call")):
                result = workbench.ask("Show the chart")
            self.assertEqual(result["visuals"][0]["citation"]["highlight"],
                             {"left": 0.1, "top": 0.15, "width": 0.4, "height": 0.3})
            crop = pymupdf.Pixmap(workbench.visual_png("visual-1-1"))
            self.assertEqual((crop.width, crop.height), (200, 150))

    def test_stream_delivers_progress_before_a_verified_claim(self):
        ready = threading.Event()
        workbench = Mock()

        def ask(question, *, progress):
            progress("search")
            if not ready.wait(4):
                raise AssertionError("The client did not receive the progress event.")
            return {"status": "answered", "message": "", "claims": [
                {"statement": "Verified", "citations": []}
            ], "visuals": []}

        workbench.ask.side_effect = ask
        frontend = "http://127.0.0.1:17666"
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            server = uvicorn.Server(uvicorn.Config(create_app(workbench, frontend), log_level="critical"))
            worker = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
            worker.start()
            connection = HTTPConnection("127.0.0.1", port, timeout=4)
            try:
                connection.request("POST", "/api/ask/stream", json.dumps({"question": "Revenue?"}),
                                   {"Content-Type": "application/json", "Origin": frontend})
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(response.getheader("Access-Control-Allow-Origin"), frontend)
                self.assertEqual(json.loads(response.readline()), {"type": "progress", "stage": "search"})
                ready.set()
                events = [json.loads(line) for line in response.read().splitlines()]
                self.assertEqual([event["type"] for event in events], ["answer", "claim", "done"])
                connection.request("OPTIONS", "/api/ask/stream", headers={
                    "Origin": frontend, "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Content-Type",
                })
                preflight = connection.getresponse()
                self.assertEqual(preflight.getheader("Access-Control-Allow-Origin"), frontend)
                preflight.read()
                connection.request("POST", "/api/ask", json.dumps({"question": "Revenue?"}),
                                   {"Content-Type": "application/json", "Origin": "https://example.test"})
                blocked = connection.getresponse()
                self.assertEqual(blocked.status, 403)
                blocked.read()
            finally:
                ready.set()
                connection.close()
                server.should_exit = True
                worker.join(8)

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
            with patch("kirana_rag.web.LLMProvider.from_env"), patch("kirana_rag.web.Agent") as agent:
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
