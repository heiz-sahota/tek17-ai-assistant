"""
Compare search modes on the golden set, against the real Azure AI Search index.

    python -m eval.compare_search --limit 5          # the same 5 questions in every mode
    python -m eval.compare_search --show-misses      # all questions, with each one's rank

Modes (same index, same questions, only the search call changes):
    keyword   search_text only                         BM25
    vector    vector_queries only                      HNSW, cosine
    hybrid    search_text + vector_queries             fused with RRF
    semantic  hybrid + query_type="semantic"           reranked by the semantic ranker

Needs the index filled by ingest/embed_upload.py.
"""
import argparse

from azure.search.documents.models import VectorizedQuery

from app.config import embed, search_client
from eval.compare_chunking import is_correct, load_questions
from eval.metrics import first_hit_rank, hit_rate_at_k, mrr

MODES = ["keyword", "vector", "hybrid", "semantic"]


def search(question: str, vector: list[float], mode: str, k: int) -> list[dict]:
    """
    Run one search in the given mode.

    Returns: list[dict], the top k results with reference and title, best first.
    """
    kwargs = {"top": k, "select": ["reference", "title"]}
    kwargs["search_text"] = question if mode != "vector" else None
    if mode != "keyword":
        kwargs["vector_queries"] = [VectorizedQuery(vector=vector, k_nearest_neighbors=50,
                                                    fields="content_vector")]
    if mode == "semantic":
        kwargs["query_type"] = "semantic"
        kwargs["semantic_configuration_name"] = "default"
    return list(search_client().search(**kwargs))


def main() -> None:
    """
    Command-line entry point.

    Returns: None. Prints a results table, and each question's rank with --show-misses.
    """
    parser = argparse.ArgumentParser(description="Compare keyword, vector, hybrid and semantic search")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    parser.add_argument("--show-misses", action="store_true")
    args = parser.parse_args()

    questions = load_questions()
    vectors = embed([q["question"] for q in questions])     # embed each question once, reuse in every mode
    print(f"{len(questions)} questions, k = {args.k}")

    ranks = {}
    for mode in args.modes:
        ranks[mode] = [
            first_hit_rank(search(q["question"], v, mode, args.k), lambda r, q=q: is_correct(q, r))
            for q, v in zip(questions, vectors)
        ]

    print(f"\n{'mode':<10}{'hit@1':>8}{'hit@3':>8}{f'hit@{args.k}':>8}{'MRR':>8}")
    for mode, r in ranks.items():
        print(f"{mode:<10}{hit_rate_at_k(r, 1):>8.2f}{hit_rate_at_k(r, 3):>8.2f}"
              f"{hit_rate_at_k(r, args.k):>8.2f}{mrr(r, args.k):>8.2f}")

    if args.show_misses:
        print(f"\nRank of the right section (- = not in top {args.k}):")
        print(f"{'id':<6}" + "".join(f"{m:>10}" for m in ranks) + "  question")
        for i, q in enumerate(questions):
            print(f"{q['id']:<6}" + "".join(f"{(ranks[m][i] or '-'):>10}" for m in ranks)
                  + f"  {q['question']}")


if __name__ == "__main__":
    main()