"""Loopback-only chat and citation viewer for an already indexed PDF."""

import asyncio
import hashlib
import json
import logging
import os
from collections.abc import Callable
from functools import lru_cache
from html import escape
from pathlib import Path
from threading import Event
from urllib.parse import SplitResult, urlsplit

import pymupdf
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from markdown_it import MarkdownIt

from .agent import Agent
from .llm_provider import LLMProvider, ProviderError
from .retrieval import Index
from .visuals import Visual, catalog, mentions_visual, requested_visuals

ASSETS = Path(__file__).with_name("web_assets")
ROOT = ASSETS.parent.parent
DOCUMENTS = {
    f"/docs/{name}": ROOT / name for name in (
        "README.md", "CUSTOMER_SUMMARY.md", "eval/REPORT.md", "eval/questions.json", "eval/results.json"
    )
}
STATIC = {
    "index.html": "text/html",
    "styles.css": "text/css",
    "app.js": "text/javascript",
}
QUESTION_BODY = {"requestBody": {
    "required": True,
    "content": {"application/json": {"schema": {
        "type": "object", "required": ["question"],
        "properties": {"question": {"type": "string", "minLength": 1, "maxLength": 800}},
    }}},
}}
logger = logging.getLogger(__name__)


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
  <link rel="stylesheet" href="/styles.css">
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
        self.visuals = catalog(index.items)
        with pymupdf.open(stream=self.pdf, filetype="pdf") as document:
            self.pages = len(document)
        self.name = pdf.name

    @lru_cache(maxsize=6)
    def page_png(self, number: int) -> bytes:
        return self._render_png(number)

    @lru_cache(maxsize=8)
    def visual_png(self, identifier: str) -> bytes:
        source = self.visuals[identifier].source
        return self._render_png(source.page, source.box)

    def _render_png(self, number: int, box: tuple | None = None) -> bytes:
        with pymupdf.open(stream=self.pdf, filetype="pdf") as document:
            page = document[number - 1]
            clip = page.rect if box is None else (pymupdf.Rect(box) * page.rotation_matrix) & page.rect
            if clip.is_empty:
                raise ValueError("The figure region is empty.")
            scale = min(2.5, 1600 / max(clip.width, clip.height))
            return page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), clip=clip, alpha=False).tobytes("png")

    def ask(self, question: str, *, progress: Callable[[str], None] | None = None) -> dict:
        direct = requested_visuals(question.strip(), self.visuals)
        if direct is not None:
            if progress:
                progress("figures")
            if not direct:
                return {"status": "refused", "message": "I couldn't find a matching visual in the indexed PDF.",
                        "claims": [], "visuals": []}
            kind = direct[0].kind if all(visual.kind == direct[0].kind for visual in direct) else "visual"
            noun = kind if len(direct) == 1 else f"{kind}s"
            crops = self._visual_cards(direct)
            return {
                "status": "answered", "message": "", "catalog": True, "visuals": crops,
                "claims": [{"statement": f"I found {len(direct)} matching {noun} in the PDF.",
                            "citations": [crop["citation"] for crop in crops]}],
            }
        result = Agent(self.index, LLMProvider.from_env()).answer(question, progress=progress)
        selected = {}
        with pymupdf.open(stream=self.pdf, filetype="pdf") as document:
            for claim in result["claims"]:
                for cite in claim["citations"]:
                    cite["highlight"] = highlight_region(document[cite["page"] - 1], cite["box"])
                    if mentions_visual(question) and (item := self.index.by_id.get(cite.get("id"))):
                        if item.group in self.visuals:
                            selected[item.group] = self.visuals[item.group]
        return {"status": result["status"], "message": result["message"], "claims": result["claims"],
                "visuals": self._visual_cards(list(selected.values()))}

    def _visual_cards(self, visuals: list[Visual]) -> list[dict]:
        if not visuals:
            return []
        cards = []
        with pymupdf.open(stream=self.pdf, filetype="pdf") as document:
            for visual in visuals:
                item = visual.source
                label = f"Figure {visual.number} · {visual.title}" if visual.number else visual.title
                cards.append({
                    "id": item.group, "kind": visual.kind, "title": label, "page": item.page,
                    "image": f"/api/visuals/{item.group}",
                    "citation": {
                        "id": item.id, "page": item.page, "label": label, "method": item.method,
                        "box": item.box, "quote": "",
                        "highlight": highlight_region(document[item.page - 1], item.box),
                    },
                })
        return cards


async def read_question(request: Request, frontend_origin: str) -> str:
    origin = request.headers.get("origin")
    if origin not in {f"{request.url.scheme}://{request.url.netloc}", frontend_origin}:
        raise HTTPException(403, "Questions must come from the local chatbot.")
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        raise HTTPException(415, "Send the question as JSON.")
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > 8000:
            raise HTTPException(400, "Enter a question between 1 and 800 characters.")
        body.extend(chunk)
    try:
        data = json.loads(body)
    except (ValueError, UnicodeError):
        raise HTTPException(400, "Send a valid question as JSON.") from None
    question = data.get("question") if isinstance(data, dict) else None
    if not isinstance(question, str) or not 1 <= len(question.strip()) <= 800:
        raise HTTPException(400, "Enter a question between 1 and 800 characters.")
    return question.strip()


class StreamClosed(Exception):
    pass


async def answer_events(workbench: Workbench, question: str):
    loop = asyncio.get_running_loop()
    events = asyncio.Queue()
    closed = Event()

    def emit(event: str, **fields):
        if closed.is_set():
            raise StreamClosed
        loop.call_soon_threadsafe(events.put_nowait, {"type": event, **fields})

    def produce():
        try:
            result = workbench.ask(question, progress=lambda stage: emit("progress", stage=stage))
            emit("answer", status=result["status"], message=result["message"], catalog=result.get("catalog", False))
            for claim in result["claims"]:
                emit("claim", claim=claim)
            if result["visuals"]:
                emit("visuals", visuals=result["visuals"])
            emit("done")
        except StreamClosed:
            pass
        except (ProviderError, ValueError) as exc:
            if not closed.is_set():
                emit("error", message=str(exc))
        except Exception:
            logger.exception("Chat answer failed")
            if not closed.is_set():
                emit("error", message="The local server could not complete the answer.")
        finally:
            if not closed.is_set():
                loop.call_soon_threadsafe(events.put_nowait, None)

    worker = asyncio.create_task(asyncio.to_thread(produce))
    try:
        while (event := await events.get()) is not None:
            yield json.dumps(event).encode() + b"\n"
        await worker
    finally:
        closed.set()
        worker.cancel()


def create_app(workbench: Workbench, frontend_origin: str) -> FastAPI:
    app = FastAPI(title="Kirana Document Intelligence", docs_url=None, redoc_url=None,
                  openapi_url="/api/openapi.json")
    app.add_middleware(
        CORSMiddleware, allow_origins=[frontend_origin],
        allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])

    @app.middleware("http")
    async def response_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; object-src 'none'; base-uri 'none'; "
            f"frame-ancestors 'self' {frontend_origin}; form-action 'self'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(HTTPException)
    async def api_error(_request: Request, exc: HTTPException):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(ASSETS / "index.html", media_type=STATIC["index.html"])

    @app.get("/config.json", include_in_schema=False)
    def frontend_config():
        return {"apiBaseUrl": ""}

    @app.get("/api/document")
    def document():
        return {"name": workbench.name, "pages": workbench.pages}

    @app.get("/source.pdf", response_class=Response)
    def source_pdf():
        return Response(workbench.pdf, media_type="application/pdf",
                        headers={"Content-Disposition": "inline"})

    @app.get("/api/pages/{number}", response_class=Response)
    def page(number: int):
        if not 1 <= number <= workbench.pages:
            raise HTTPException(404, "This page is outside the PDF.")
        return Response(workbench.page_png(number), media_type="image/png")

    @app.get("/api/visuals/{identifier}", response_class=Response)
    def visual(identifier: str):
        if identifier not in workbench.visuals:
            raise HTTPException(404, "This visual is outside the indexed PDF.")
        try:
            return Response(workbench.visual_png(identifier), media_type="image/png")
        except ValueError:
            raise HTTPException(404, "This visual could not be rendered.") from None

    @app.get("/docs/{name:path}", include_in_schema=False)
    def docs(name: str):
        path = DOCUMENTS.get(f"/docs/{name}")
        if path is None:
            raise HTTPException(404, "Document not found.")
        if path.suffix == ".md":
            return HTMLResponse(render_markdown(path))
        return FileResponse(path, media_type="application/json")

    @app.post("/api/ask", openapi_extra=QUESTION_BODY)
    async def ask(request: Request):
        question = await read_question(request, frontend_origin)
        try:
            return await asyncio.to_thread(workbench.ask, question)
        except ProviderError as exc:
            raise HTTPException(502, str(exc)) from None
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None

    @app.post("/api/ask/stream", response_class=StreamingResponse, openapi_extra=QUESTION_BODY)
    async def stream(request: Request):
        question = await read_question(request, frontend_origin)
        return StreamingResponse(answer_events(workbench, question), media_type="application/x-ndjson")

    @app.get("/assets/{name}", include_in_schema=False)
    @app.get("/{name}", include_in_schema=False)
    def asset(name: str):
        if name not in STATIC:
            raise HTTPException(404, "Page not found.")
        return FileResponse(ASSETS / name, media_type=STATIC[name])

    return app


def local_url(name: str) -> SplitResult:
    try:
        parsed = urlsplit(os.getenv(name, "").strip().rstrip("/"))
        if (parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"}
                and parsed.port and not parsed.username and not parsed.password
                and not parsed.path and not parsed.query and not parsed.fragment):
            return parsed
    except ValueError:
        pass
    raise ValueError(f"Set {name} in .env to a local HTTP URL with an explicit port.")


def serve(pdf: Path, index: Index, port: int | None):
    backend = local_url("BACKEND_BASE_URL")
    frontend = local_url("FRONTEND_BASE_URL")
    port = backend.port if port is None else port
    if not 0 <= port <= 65535:
        raise ValueError("Use a port between 0 and 65535.")
    app = create_app(Workbench(pdf, index), frontend.geturl())
    uvicorn.run(app, host=backend.hostname, port=port, proxy_headers=False)


def serve_frontend():
    backend = local_url("BACKEND_BASE_URL")
    frontend = local_url("FRONTEND_BASE_URL")
    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])

    @app.get("/config.json")
    def frontend_config():
        return JSONResponse({"apiBaseUrl": backend.geturl()}, headers={"Cache-Control": "no-store"})

    app.mount("/", StaticFiles(directory=ASSETS, html=True))
    uvicorn.run(app, host=frontend.hostname, port=frontend.port, proxy_headers=False)
