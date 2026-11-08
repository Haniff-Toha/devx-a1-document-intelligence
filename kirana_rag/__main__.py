import argparse
import json
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import pymupdf

from .agent import Agent
from .ingest import ingest
from .openrouter import OpenRouter, OpenRouterError
from .retrieval import read_index, write_index

DEFAULT_INDEX = Path(".rag/index.sqlite3")


def print_json(value: object):
    print(json.dumps(value, ensure_ascii=True, indent=2))


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description="Ask citation-grounded questions about a PDF.")
    sub = cli.add_subparsers(dest="command", required=True)
    load = sub.add_parser("ingest", help="Process a PDF and atomically replace an index.")
    load.add_argument("pdf", type=Path)
    load.add_argument("--local-only", action="store_true", help="Native PDF + OCR; keyword search only.")
    ask = sub.add_parser("ask", help="Answer a question with the Graph RAG agent.")
    ask.add_argument("question")
    ask.add_argument("--trace", action="store_true", help="Show search and graph actions.")
    search = sub.add_parser("search", help="Return evidence without generating an answer.")
    search.add_argument("query")
    search.add_argument("--mode", choices=["keyword", "hybrid", "graph"], default="keyword")
    inspect = sub.add_parser("inspect", help="View provenance and evidence from the index.")
    view = inspect.add_mutually_exclusive_group()
    view.add_argument("--page", type=int)
    view.add_argument("--id")
    evaluate = sub.add_parser("evaluate", help="Run the published A1 benchmark against the current index.")
    evaluate.add_argument("--cases", type=Path, default=Path("eval/questions.json"))
    evaluate.add_argument("--out", type=Path, default=Path(".rag/evaluation.json"))
    for command in (load, ask, search, inspect, evaluate):
        command.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    for command in (load, ask, search, inspect):
        command.add_argument("--json", action="store_true", help="Emit structured JSON.")
    return cli


def print_evidence(items):
    for item in items:
        print(f"{item.id}  p. {item.page}  {item.label}  ({item.method})")
        print(f"  {item.text}")


def run(args):
    if args.command == "ingest":
        if args.pdf.resolve() == args.index.resolve():
            raise ValueError("Use separate paths for the input PDF and the index.")
        started = perf_counter()
        client = None if args.local_only else OpenRouter.from_env()
        document = ingest(args.pdf, client)
        metadata = write_index(args.index, document, client)
        metadata["seconds"] = round(perf_counter() - started, 2)
        if client:
            metadata["api"] = {"requests": client.calls, "reported_cost_usd": client.cost,
                               "responses_with_cost": client.cost_reported}
        if args.json:
            print_json(metadata)
        else:
            print(f"Indexed {len(document.pages)} pages, {metadata['evidence_count']} evidence records, "
                  f"{metadata['edge_count']} directed links in {metadata['seconds']}s.")
            for page in metadata["pages"]:
                for note in page["notes"]:
                    print(f"p. {page['page']}  {note}")
        return

    index = read_index(args.index)
    if args.command == "inspect":
        if args.id:
            if args.id not in index.by_id:
                raise ValueError(f"Unknown evidence ID: {args.id}")
            items = [index.by_id[args.id]]
        else:
            if args.page is not None and not 1 <= args.page <= len(index.metadata["pages"]):
                raise ValueError("Page is outside this document.")
            items = [item for item in index.items if item.page == args.page]
        if args.json or not (args.id or args.page):
            print_json({"document": index.metadata, "evidence": [asdict(item) for item in items],
                        "links": [asdict(edge) for item in items for edge in index.adjacency[item.id]]})
        else:
            print_evidence(items)
        return
    if args.command == "search":
        vector = None
        if args.mode != "keyword":
            client = OpenRouter.from_env()
            if client.embed_model != index.metadata["embedding_model"]:
                raise ValueError("Hybrid search requires an index built with the configured embedding model.")
            vector = client.embed([args.query])[0]
        items = index.search(args.query, vector=vector)
        links = index.expand(args.query, [item.id for item in items], essential_only=True) if (
            args.mode == "graph"
        ) else []
        items.extend(item for item, _ in links)
        if args.json:
            print_json({"evidence": [asdict(item) for item in items],
                        "links": [asdict(edge) for _, edge in links]})
        else:
            print_evidence(items)
        return

    client = OpenRouter.from_env()
    if args.command == "evaluate":
        from .evaluation import evaluate

        if args.out.resolve() == args.index.resolve():
            raise ValueError("Use separate paths for the index and evaluation results.")
        report = evaluate(index, client, args.cases, args.out)
        print_json(report["metrics"])
        print(f"Detailed results saved to {args.out}.")
        return
    response = Agent(index, client).answer(args.question)
    if args.json:
        print_json(response)
        return
    if response["status"] == "refused":
        print(response["message"])
    for claim in response["claims"]:
        references = list(dict.fromkeys(
            f"p. {cite['page']}, {cite['label']}" for cite in claim["citations"]
        ))
        print(f"{claim['statement']}  [{'; '.join(references)}]")
    if args.trace:
        print_json(response["trace"])


def main():
    try:
        run(parser().parse_args())
    except (OSError, ValueError, OpenRouterError, pymupdf.FileDataError, sqlite3.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
