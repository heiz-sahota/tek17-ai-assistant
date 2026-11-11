"""
Run the golden set through the real assistant and save every answer with its sources.

    python -m eval.run_eval --limit 5          # quick first test
    python -m eval.run_eval                    # all questions, about 5-8 minutes on a 15K TPM quota
    python -m eval.run_eval --only pricing --pause 4

Output: eval/results/run_<time>/
    answers.jsonl   one line per question: the answer, citations, retrieved and used chunks, trace
    run_meta.json   settings, deployment, prompt fingerprint, when it ran

This step costs tokens. Scoring (eval.score) and judging (eval.judge) read the
saved answers, so they never call the assistant again.
"""
import argparse
import hashlib
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from app import rag
from app.config import CHAT_DEPLOYMENT
from app.prompts import SYSTEM_PROMPT
from eval.golden import load_golden

RESULTS_DIR = Path(__file__).resolve().parent / "results"
EVAL_ROLES = ["both"]   # evals see every source; access control is tested in break_it


def chunk_dict(chunk: rag.RetrievedChunk, with_content: bool) -> dict:
    """Returns: dict, a chunk as JSON, with or without its text."""
    d = {"reference": chunk.reference, "title": chunk.title, "source": chunk.source, "score": chunk.score}
    return d | {"content": chunk.content} if with_content else d


def ask(question: str) -> tuple[rag.RagResult | None, str]:
    """
    Ask one question; wait and retry once if the quota is used up.

    Returns: (RagResult, "") on success, or (None, error message).
    """
    for attempt in range(2):
        try:
            return rag.answer_with_details(question, EVAL_ROLES), ""
        except rag.RateLimitedError:
            if attempt == 0:
                print("    rate limited, waiting 30 s")
                time.sleep(30)
        except (rag.RetrievalError, rag.GenerationError, ValueError) as e:
            return None, f"{type(e).__name__}: {e}"
    return None, "RateLimitedError: still rate limited after waiting"


def main() -> None:
    """
    Command-line entry point.

    Returns: None. Writes answers.jsonl and run_meta.json into a new results folder.
    """
    parser = argparse.ArgumentParser(description="Run the golden set through the assistant")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--only", choices=["tek17", "pricing"], help="one source only")
    parser.add_argument("--pause", type=float, default=6.0, help="seconds between questions (rate limits)")
    args = parser.parse_args()

    items = load_golden([args.only] if args.only else None)[:args.limit]
    run_dir = RESULTS_DIR / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True)
    meta = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "questions": len(items),
        "roles": EVAL_ROLES,
        "chat_deployment": CHAT_DEPLOYMENT,
        "prompt_sha": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:12],
        "settings": asdict(rag.SETTINGS),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{len(items)} questions -> {run_dir.relative_to(RESULTS_DIR.parent.parent)}")

    with (run_dir / "answers.jsonl").open("w", encoding="utf-8") as out:
        for i, item in enumerate(items, start=1):
            result, error = ask(item.question)
            record = item.to_dict() | {"error": error}
            if result:
                record |= {
                    "answer": result.response.answer,
                    "citations": result.response.citations,
                    "found_in_sources": result.response.found_in_sources,
                    "retrieved": [chunk_dict(c, with_content=False) for c in result.retrieved],
                    "used": [chunk_dict(c, with_content=True) for c in result.used],
                    "invented_citations": result.trace.invented_citations,
                    "outcome": result.trace.outcome,
                    "prompt_tokens": result.trace.prompt_tokens,
                    "completion_tokens": result.trace.completion_tokens,
                    "timings_ms": result.trace.timings_ms,
                    "request_id": result.trace.request_id,
                }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()                       # a crash halfway keeps everything so far
            status = error or result.trace.outcome
            print(f"  {i:>2}/{len(items)} {item.id:<4} {status:<22} {item.question[:60]}")
            if i < len(items):
                time.sleep(args.pause)

    print(f"\nSaved. Next: python -m eval.score {run_dir.name}")


if __name__ == "__main__":
    main()