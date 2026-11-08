"""Loopback-only chat and citation viewer for an already indexed PDF."""

import hashlib
import json
import re
from functools import lru_cache
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import pymupdf
from markdown_it import MarkdownIt

from .agent import Agent
from .openrouter import OpenRouter, OpenRouterError
from .retrieval import Index

ASSETS = Path(__file__).with_name("web_assets")
ROOT = ASSETS.parent.parent
DOCUMENTS = {
    f"/docs/{name}": ROOT / name for name in (
        "README.md", "CUSTOMER_SUMMARY.md", "eval/REPORT.md", "eval/questions.json", "eval/results.json"
    )
}
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/assets/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/assets/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
PAGE_ROUTE = re.compile(r"/api/pages/([1-9][0-9]{0,8})")
CSP = ("default-src 'self'; object-src 'none'; base-uri 'none'; "
       "frame-ancestors 'self'; form-action 'self'")


def render_markdown(path: Path) -> bytes:
    markdown = MarkdownIt("commonmark", {"html": False}).enable("table")

    def render_link(tokens, index, options, env):
        token = tokens[index]
        if (token.attrGet("href") or "").startswith(("http://", "https://")):
            token.attrSet("target", "_blank")
            token.attrSet("rel", "noopener noreferrer")
        return markdown.renderer.renderToken(tokens, index, options, env)

    markdown.renderer.rules["link_open"] = render_link
    body = markdown.render(path.read_text(encoding="utf-8"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(path.stem)} · Kirana Document Intelligence</title>
  <link rel="stylesheet" href="/assets/styles.css">
</head>
<body class="doc-body"><article class="doc-content">{body}</article></body>
</html>""".encode()


def highlight_region(page: pymupdf.Page, box: tuple | list) -> dict | None:
    area = page.rect
    region = (pymupdf.Rect(box) * page.rotation_matrix) & area
    if region.is_empty:
        return None
    return {
        "left": (region.x0 - area.x0) / area.width,
        "top": (region.y0 - area.y0) / area.height,
        "width": region.width / area.width,
        "height": region.height / area.height,
    }


class Workbench:
    def __init__(self, pdf: Path, index: Index):
        self.pdf = pdf.read_bytes()
        if hashlib.sha256(self.pdf).hexdigest() != index.metadata["sha256"]:
            raise ValueError("This PDF differs from the indexed source. Ingest it first, or choose its matching index.")
        self.index = index
        with pymupdf.open(stream=self.pdf, filetype="pdf") as document:
            self.pages = len(document)
        self.name = pdf.name

    @lru_cache(maxsize=6)
    def page_png(self, number: int) -> bytes:
        with pymupdf.open(stream=self.pdf, filetype="pdf") as document:
            page = document[number - 1]
            scale = min(2, 1600 / max(page.rect.width, page.rect.height))
            return page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).tobytes("png")

    def ask(self, question: str) -> dict:
        result = Agent(self.index, OpenRouter.from_env()).answer(question)
        with pymupdf.open(stream=self.pdf, filetype="pdf") as document:
            for claim in result["claims"]:
                for cite in claim["citations"]:
                    cite["highlight"] = highlight_region(document[cite["page"] - 1], cite["box"])
        return {"status": result["status"], "message": result["message"], "claims": result["claims"]}


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, content_type: str, status: int = 200, **headers: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        for name, value in headers.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, value: dict, status: int = 200):
        self._send(json.dumps(value).encode(), "application/json; charset=utf-8", status)

    def _local(self) -> bool:
        if self.headers.get("Host") != self.server.authority:
            self._json({"error": "Open the server at its 127.0.0.1 address."}, 403)
            return False
        return True

    def do_GET(self):
        if not self._local():
            return
        path = urlsplit(self.path).path
        workbench = self.server.workbench
        if path in STATIC:
            filename, content_type = STATIC[path]
            self._send((ASSETS / filename).read_bytes(), content_type)
        elif path == "/api/document":
            self._json({"name": workbench.name, "pages": workbench.pages})
        elif path == "/source.pdf":
            self._send(workbench.pdf, "application/pdf", **{"Content-Disposition": "inline"})
        elif match := PAGE_ROUTE.fullmatch(path):
            number = int(match[1])
            if number > workbench.pages:
                self._json({"error": "This page is outside the PDF."}, 404)
                return
            self._send(workbench.page_png(number), "image/png")
        elif path in DOCUMENTS:
            document = DOCUMENTS[path]
            if document.suffix == ".md":
                self._send(render_markdown(document), "text/html; charset=utf-8")
            else:
                self._send(document.read_bytes(), "application/json; charset=utf-8")
        else:
            self._json({"error": "Page not found."}, 404)

    def do_POST(self):
        if not self._local():
            return
        if urlsplit(self.path).path != "/api/ask":
            self._json({"error": "Page not found."}, 404)
            return
        if self.headers.get("Origin") != f"http://{self.server.authority}":
            self._json({"error": "Questions must come from the local chatbot."}, 403)
            return
        if self.headers.get_content_type() != "application/json":
            self._json({"error": "Send the question as JSON."}, 415)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 1 <= length <= 8000:
                raise ValueError
            data = json.loads(self.rfile.read(length))
            question = data.get("question") if isinstance(data, dict) else None
            if not isinstance(question, str) or not 1 <= len(question.strip()) <= 800:
                raise ValueError
        except (ValueError, UnicodeError):
            self._json({"error": "Enter a question between 1 and 800 characters."}, 400)
            return
        try:
            result = self.server.workbench.ask(question)
        except OpenRouterError as exc:
            self._json({"error": str(exc)}, 502)
            return
        except ValueError as exc:
            self._json({"error": str(exc)}, 409)
            return
        self._json(result)


class Server(ThreadingHTTPServer):
    def __init__(self, workbench: Workbench, port: int):
        super().__init__(("127.0.0.1", port), Handler)
        self.authority = f"127.0.0.1:{self.server_port}"
        self.workbench = workbench


def serve(pdf: Path, index: Index, port: int):
    if not 0 <= port <= 65535:
        raise ValueError("Use a port between 0 and 65535.")
    with Server(Workbench(pdf, index), port) as server:
        print(f"Kirana chatbot at http://{server.authority}", flush=True)
        server.serve_forever()
