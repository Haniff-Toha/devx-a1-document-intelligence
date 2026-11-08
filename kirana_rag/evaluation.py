"""Score evidence retrieval, answer text, and citation coverage independently."""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median
from time import perf_counter

from .agent import SEED_COUNT, Agent, seed_evidence
from .openrouter import OpenRouter, OpenRouterError
from .retrieval import Index


def contains_all(text: str, patterns: list[str]) -> bool:
    return all(re.search(pattern, text, re.I) for pattern in patterns)


def evidence_score(groups: list[dict], snippets: list[tuple[int, str]]) -> dict:
    return {
        "groups": len(groups),
        "pages_hit": sum(any(page in group["pages"] for page, _ in snippets) for group in groups),
        "evidence_hit": sum(any(
            page in group["pages"] and contains_all(text, group["patterns"])
            for page, text in snippets
        ) for group in groups),
    }


def result_score(case: dict, answer: dict) -> dict:
    if not case["evidence"]:
        correct = answer["status"] == "refused" and answer["reason"] == "insufficient_evidence"
        return {"correct": correct, "citations_complete": None, "grounded_correct": correct}
    claims = answer["claims"]
    correct = answer["status"] == "answered" and contains_all(
        "\n".join(claim["statement"] for claim in claims), case["answer_patterns"]
    )
    citations = [(citation["page"], citation["quote"])
                 for claim in claims for citation in claim["citations"]]
    scores = evidence_score(case["evidence"], citations)
    complete = scores["evidence_hit"] == scores["groups"]
    return {"correct": correct, "citations_complete": complete, "grounded_correct": correct and complete}


def evaluate(index: Index, client: OpenRouter, cases_path: Path, output: Path) -> dict:
    started = perf_counter()
    if output.resolve() == cases_path.resolve():
        raise ValueError("Use separate paths for evaluation questions and results.")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if not cases:
        raise ValueError("Evaluation needs at least one question.")
    Agent(index, client)  # Fail before the batch if the embedding model differs from the index.
    vectors = client.embed([case["question"] for case in cases])
    batch_seconds = round(perf_counter() - started, 2)

    def run(entry):
        case, vector = entry
        started_question = perf_counter()
        question = case["question"]
        case_client = OpenRouter(client.key, client.chat_model, client.embed_model)
        keyword = index.search(question, k=SEED_COUNT)
        hybrid, links = seed_evidence(index, question, vector)
        initial = hybrid + [item for item, _ in links]
        try:
            answer = Agent(index, case_client).answer(question, vector=vector)
        except OpenRouterError as exc:
            answer = {"status": "error", "reason": str(exc), "claims": [], "trace": [],
                      "evidence_ids": [item.id for item in initial]}
        stages = {
            "keyword@7": keyword, "hybrid@7": hybrid, "hybrid_with_graph": initial,
            "agent_context": [index.by_id[i] for i in answer["evidence_ids"]],
        }
        retrieval = {
            name: {
                "ids": [item.id for item in items],
                "pages": sorted({item.page for item in items}),
                **evidence_score(case["evidence"], [(item.page, item.searchable) for item in items]),
            }
            for name, items in stages.items()
        }
        return {
            "id": case["id"], "category": case["category"], "question": question,
            "expected": case["expected"], "retrieval": retrieval, "answer": answer,
            "checks": result_score(case, answer),
            "seconds": round(perf_counter() - started_question, 2),
            "api": {"requests": case_client.calls, "reported_cost_usd": case_client.cost,
                    "responses_with_cost": case_client.cost_reported},
        }

    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(run, zip(cases, vectors)))
    supported = [row for case, row in zip(cases, rows) if case["evidence"]]
    unanswerable = [row for case, row in zip(cases, rows) if not case["evidence"]]
    retrieval = {}
    for name in rows[0]["retrieval"]:
        values = [row["retrieval"][name] for row in supported]
        retrieval[name] = {
            "complete_pages": sum(value["pages_hit"] == value["groups"] for value in values),
            "complete_evidence": sum(value["evidence_hit"] == value["groups"] for value in values),
            "cases": len(supported),
            "evidence_groups_hit": sum(value["evidence_hit"] for value in values),
            "evidence_groups": sum(value["groups"] for value in values),
        }
    metrics = {
        "retrieval": retrieval,
        "answers": {
            "correct": sum(row["checks"]["correct"] for row in supported),
            "grounded_correct": sum(row["checks"]["grounded_correct"] for row in supported),
            "cases": len(supported),
            "correct_refusals": sum(row["checks"]["correct"] for row in unanswerable),
            "unanswerable_cases": len(unanswerable),
            "provider_errors": sum(row["answer"]["status"] == "error" for row in rows),
        },
        "runtime": {
            "seconds_wall": round(perf_counter() - started, 2),
            "seconds_batch_embeddings": batch_seconds,
            "seconds_per_question_median": median(row["seconds"] for row in rows),
        },
        "api": {
            "requests": {name: client.calls[name] + sum(row["api"]["requests"][name] for row in rows)
                         for name in client.calls},
            "reported_cost_usd": round(client.cost + sum(row["api"]["reported_cost_usd"] for row in rows), 8),
            "responses_with_cost": client.cost_reported + sum(row["api"]["responses_with_cost"] for row in rows),
        },
    }
    report = {
        "document": index.metadata["name"], "document_sha256": index.metadata["sha256"],
        "models": {"chat": client.chat_model, "embeddings": client.embed_model,
                   "vision": index.metadata["vision_model"]},
        "metrics": metrics, "cases": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
