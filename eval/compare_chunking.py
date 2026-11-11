"""
eval/compare_chunking.py: which chunking strategy retrieves best?
=================================================================

Runs every chunk set in data/chunks/ (recursive-500, fixed-500, recursive-200 ...)
against the golden set and prints hit rate at 1, 3 and 5 and MRR for each.

Run from the project root, after `python -m ingest.chunk --size 500 200 100`:
    python -m eval.compare_chunking
    python -m eval.compare_chunking --k 5 --show-misses

HOW IT WORKS
------------
1. Load the golden set: questions with the section that SHOULD be found.
   Questions whose answer isn't in the sources (expected_doc is null) are
   skipped here: there is no right chunk to find. They're for answer evals.
2. For each chunk set: embed every chunk with the Foundry `embed` deployment.
3. Embed each question, rank chunks by cosine similarity, keep the top k.
4. Find the rank of the first chunk from the expected section.
5. Hit rate and MRR per chunk set, plus each question's rank, so you can read
   the failures and not just the averages.

WHY VECTOR SEARCH IN MEMORY, NOT AZURE AI SEARCH
------------------------------------------------
To compare chunking, change ONE thing and keep everything else fixed. Plain
cosine similarity in numpy is the simplest fixed retriever: no index per
strategy, no hybrid or ranking settings mixed in, and it runs in seconds.
The production retriever (hybrid search in Azure AI Search) is evaluated
later, with the winning chunk set, by the main eval.

EMBEDDING CACHE
---------------
Vectors are saved in data/embeddings/cache.jsonl, keyed by a hash of the
text. Re-runs and chunk sets that share chunks don't pay to embed them again.
OpenAI vectors have length 1, so the dot product equals cosine similarity.

WHAT TO SAY ABOUT THE RESULTS
-----------------------------
"With the same questions and the same retriever, section-aware recursive
chunking at 500 tokens had a hit rate at 5 of X against Y for fixed 500.
Fixed-size chunks split tables and lost their headers. I read every miss
before deciding." Use YOUR numbers; 17 questions is small, so say so.
"""
import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from openai import RateLimitError

from app.config import EMBEDDING_DEPLOYMENT, embed
from eval.metrics import first_hit_rank, hit_rate_at_k, mrr

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_DIR = ROOT / "data" / "chunks"
CACHE_PATH = ROOT / "data" / "embeddings" / "cache.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"
GOLDEN_SETS = [ROOT / "eval" / "pricing_eval_questions.jsonl",
               ROOT / "eval" / "questions_tek17.jsonl"]
BATCH = 64


# ======================================================================
# Golden set
# ======================================================================

def load_questions() -> list[dict]:
    """Questions that have a section to find. Template rows are skipped."""
    questions = []
    for path in GOLDEN_SETS:
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            q = json.loads(line)
            has_target = q.get("expected_section") or (
                q.get("expected_reference") and "xx" not in q["expected_reference"])
            if has_target and not q["question"].startswith("Write your"):
                questions.append(q)
    return questions


def normalise(text: str) -> str:
    return " ".join(text.replace("TEK17", "").split()).lower()


def is_correct(question: dict, chunk: dict) -> bool:
    """
    Pricing questions name a document and a section title:
        expected_doc "03_Postal_Zone_Rules", expected_section "Rule Z-2. Island exceptions"
    TEK17 questions name a reference:
        expected_reference "§ 12-15"   matches chunk reference "TEK17 § 12-15"
    """
    if question.get("expected_reference"):
        return normalise(chunk["reference"]) == normalise(question["expected_reference"])
    return (chunk["reference"].startswith(question["expected_doc"])
            and normalise(chunk["title"]) == normalise(question["expected_section"]))


# ======================================================================
# Embeddings with a cache
# ======================================================================

def text_key(text: str) -> str:
    return hashlib.sha256(f"{EMBEDDING_DEPLOYMENT}|{text}".encode("utf-8")).hexdigest()


def load_cache() -> dict[str, list[float]]:
    cache = {}
    if CACHE_PATH.exists():
        for line in CACHE_PATH.open(encoding="utf-8"):
            row = json.loads(line)
            cache[row["key"]] = row["vector"]
    return cache


def embed_with_retry(texts: list[str], attempts: int = 5) -> list[list[float]]:
    """429 means the deployment's tokens-per-minute quota is used up: wait and retry."""
    for attempt in range(attempts):
        try:
            return embed(texts)
        except RateLimitError:
            wait = 2 ** attempt
            print(f"  rate limited, waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError("Still rate limited; raise the deployment's quota or try later")


def embed_all(texts: list[str], cache: dict) -> np.ndarray:
    """Vectors for every text, embedding only the ones not cached yet."""
    missing = list({text_key(t): t for t in texts if text_key(t) not in cache}.items())
    if missing:
        print(f"  embedding {len(missing)} new texts")
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CACHE_PATH.open("a", encoding="utf-8") as f:
            for i in range(0, len(missing), BATCH):
                batch = missing[i:i + BATCH]
                vectors = embed_with_retry([t for _, t in batch])
                for (key, _), vector in zip(batch, vectors):
                    cache[key] = vector
                    f.write(json.dumps({"key": key, "vector": vector}) + "\n")
    return np.array([cache[text_key(t)] for t in texts], dtype=np.float32)


# ======================================================================
# Comparison
# ======================================================================

def load_chunk_sets() -> dict[str, list[dict]]:
    sets = {}
    for folder in sorted(p for p in CHUNKS_DIR.iterdir() if p.is_dir()):
        chunks = []
        for path in sorted(folder.glob("*.jsonl")):
            chunks += [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        if chunks:
            sets[folder.name] = chunks
    return sets


def evaluate(chunks: list[dict], questions: list[dict], question_vectors: np.ndarray,
             cache: dict, k: int) -> list[int | None]:
    chunk_vectors = embed_all([c["content"] for c in chunks], cache)
    scores = question_vectors @ chunk_vectors.T          # cosine similarity, one row per question
    ranks = []
    for q, row in zip(questions, scores):
        top = np.argsort(-row)[:k]                       # indexes of the k best chunks
        ranks.append(first_hit_rank([chunks[i] for i in top], lambda c: is_correct(q, c)))
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare chunking strategies on the golden set")
    parser.add_argument("--k", type=int, default=5, help="how many chunks to retrieve")
    parser.add_argument("--show-misses", action="store_true", help="print each question's rank")
    args = parser.parse_args()

    questions = load_questions()
    chunk_sets = load_chunk_sets()
    if not questions or not chunk_sets:
        raise SystemExit("Need questions in eval/*.jsonl and chunks in data/chunks/. "
                         "Run python -m ingest.chunk --size 500 200 100 first.")
    print(f"{len(questions)} questions, {len(chunk_sets)} chunk sets, k = {args.k}")

    cache = load_cache()
    question_vectors = embed_all([q["question"] for q in questions], cache)

    rows, all_ranks = [], {}
    for name, chunks in chunk_sets.items():
        print(f"\n{name}: {len(chunks)} chunks")
        ranks = evaluate(chunks, questions, question_vectors, cache, args.k)
        all_ranks[name] = ranks
        rows.append({
            "chunk_set": name, "chunks": len(chunks),
            "hit@1": hit_rate_at_k(ranks, 1), "hit@3": hit_rate_at_k(ranks, 3),
            f"hit@{args.k}": hit_rate_at_k(ranks, args.k), "mrr": mrr(ranks, args.k),
        })

    rows.sort(key=lambda r: (r[f"hit@{args.k}"], r["mrr"]), reverse=True)
    print(f"\n{'chunk set':<16}{'chunks':>8}{'hit@1':>8}{'hit@3':>8}{f'hit@{args.k}':>8}{'MRR':>8}")
    for r in rows:
        print(f"{r['chunk_set']:<16}{r['chunks']:>8}{r['hit@1']:>8.2f}{r['hit@3']:>8.2f}"
              f"{r[f'hit@{args.k}']:>8.2f}{r['mrr']:>8.2f}")

    if args.show_misses:
        names = [r["chunk_set"] for r in rows]
        print(f"\nRank of the right section per question (- = not in top {args.k}):")
        print(f"{'id':<6}" + "".join(f"{n:>15}" for n in names) + "  question")
        for i, q in enumerate(questions):
            cells = "".join(f"{(all_ranks[n][i] or '-'):>15}" for n in names)
            print(f"{q['id']:<6}{cells}  {q['question'][:60]}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"chunking_{datetime.now():%Y%m%d_%H%M}.json"
    out.write_text(json.dumps({
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "embedding_deployment": EMBEDDING_DEPLOYMENT, "k": args.k, "retriever": "cosine, in memory",
        "questions": len(questions), "results": rows,
        "ranks": {n: dict(zip([q["id"] for q in questions], r)) for n, r in all_ranks.items()},
    }, indent=2), encoding="utf-8")
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()