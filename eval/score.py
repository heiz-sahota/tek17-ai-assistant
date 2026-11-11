"""
Score a saved run with the code checks. Free and instant: no model calls.

    python -m eval.score                      # the latest run
    python -m eval.score run_20261111_2115    # a specific run

Writes into the run folder:
    checks.jsonl    the check results per question
    summary.json    pass rates, retrieval metrics, tokens and latency
    report.md       the same as a readable report, with every failure listed
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from eval.checks import CHECKS, run_checks
from eval.metrics import hit_rate_at_k, mrr

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def latest_run() -> Path:
    """Returns: Path of the newest run_* folder."""
    runs = sorted(RESULTS_DIR.glob("run_*"))
    if not runs:
        raise SystemExit("No runs yet. Run: python -m eval.run_eval --limit 5")
    return runs[-1]


def load_records(run_dir: Path) -> list[dict]:
    """Returns: list[dict], one per line of answers.jsonl."""
    path = run_dir / "answers.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rate(values: list) -> tuple[float | None, int]:
    """
    Pass rate over the applicable results (None excluded).

    Returns: (rate or None, number of applicable questions).
    """
    applicable = [v for v in values if v is not None]
    return (sum(applicable) / len(applicable) if applicable else None), len(applicable)


def fmt(r: float | None, n: int) -> str:
    """Returns: str like "0.88 (15)", or "-" when nothing applied."""
    return "-" if r is None else f"{r:.2f} ({n})"


def summarise(records: list[dict], results: list[dict]) -> dict:
    """
    Pass rates per check: overall, by source and by type; retrieval metrics; cost.

    Returns: dict.
    """
    groups = {"overall": lambda r: "all", "source": lambda r: r["source"], "type": lambda r: r["type"]}
    summary = {}
    for group, key in groups.items():
        buckets = defaultdict(lambda: defaultdict(list))
        for rec, res in zip(records, results):
            for check in CHECKS:
                buckets[key(rec)][check].append(res[check])
        summary[group] = {name: {check: rate(values) for check, values in checks.items()}
                          for name, checks in buckets.items()}

    ranks = [res["rank"] for rec, res in zip(records, results) if rec["in_scope"] and not rec.get("error")]
    summary["retrieval"] = {"questions": len(ranks), "hit@1": hit_rate_at_k(ranks, 1),
                            "hit@3": hit_rate_at_k(ranks, 3), "hit@5": hit_rate_at_k(ranks, 5), "mrr": mrr(ranks)}

    ok = [r for r in records if not r.get("error")]
    summary["cost"] = {
        "answered_ok": len(ok), "errors": len(records) - len(ok),
        "avg_prompt_tokens": round(sum(r["prompt_tokens"] for r in ok) / len(ok)) if ok else 0,
        "avg_completion_tokens": round(sum(r["completion_tokens"] for r in ok) / len(ok)) if ok else 0,
        "avg_latency_ms": round(sum(r["timings_ms"].get("total", 0) for r in ok) / len(ok)) if ok else 0,
    }
    return summary


def report_md(run_dir: Path, records: list[dict], results: list[dict], summary: dict) -> str:
    """Returns: str, the Markdown report."""
    lines = [f"# Eval report: {run_dir.name}", ""]
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        lines += [f"Deployment `{meta['chat_deployment']}`, prompt `{meta['prompt_sha']}`, "
                  f"top k {meta['settings']['top_k']}, semantic ranker {meta['settings']['use_semantic_ranker']}.", ""]

    lines += ["## Code checks", "", "| Check | Overall | TEK17 | Pricing |", "|---|---|---|---|"]
    for check in CHECKS:
        row = [fmt(*summary["overall"]["all"][check])]
        row += [fmt(*summary["source"][s][check]) if s in summary["source"] else "-" for s in ("tek17", "pricing")]
        lines.append(f"| {check} | " + " | ".join(row) + " |")
    lines += ["", "Numbers in brackets: how many questions the check applied to.", ""]

    types = sorted(summary["type"])
    lines += ["## By question type", "", "| Check | " + " | ".join(types) + " |", "|---|" + "---|" * len(types)]
    for check in CHECKS:
        lines.append(f"| {check} | " + " | ".join(fmt(*summary["type"][t][check]) for t in types) + " |")

    r = summary["retrieval"]
    lines += ["", "## Retrieval, full pipeline", "",
              f"{r['questions']} in-scope questions: hit@1 {r['hit@1']:.2f}, hit@3 {r['hit@3']:.2f}, "
              f"hit@5 {r['hit@5']:.2f}, MRR {r['mrr']:.2f}.", ""]

    c = summary["cost"]
    lines += ["## Cost and latency", "",
              f"{c['answered_ok']} answered without errors, {c['errors']} errors. Average {c['avg_prompt_tokens']} prompt "
              f"tokens, {c['avg_completion_tokens']} completion tokens, {c['avg_latency_ms']} ms per question.", ""]

    lines += ["## Failures", ""]
    for rec, res in zip(records, results):
        failed = [ch for ch in CHECKS if res[ch] is False]
        if not failed and not rec.get("error"):
            continue
        lines += [f"### {rec['id']} ({rec['source']}, {rec['type']}): {', '.join(failed) or 'error'}",
                  f"- **Question:** {rec['question']}",
                  f"- **Reference:** {rec['reference_answer']}"]
        if rec.get("error"):
            lines.append(f"- **Error:** {rec['error']}")
        else:
            expected = rec.get("expected_reference") or f"{rec.get('expected_doc')} / {rec.get('expected_section')}"
            lines += [f"- **Answer:** {rec['answer']}",
                      f"- **Citations:** {', '.join(rec['citations']) or 'none'}",
                      f"- **Expected section:** {expected if rec['in_scope'] else 'none, out of scope'}"
                      f" · rank in retrieval: {res['rank'] or 'not retrieved'}",
                      f"- **Retrieved:** {', '.join(c['reference'] for c in rec['retrieved'])}"]
            if rec.get("invented_citations"):
                lines.append(f"- **Invented citations removed:** {', '.join(rec['invented_citations'])}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    """
    Command-line entry point.

    Returns: None. Prints the table and writes checks.jsonl, summary.json and report.md.
    """
    parser = argparse.ArgumentParser(description="Score a saved eval run with code checks")
    parser.add_argument("run", nargs="?", help="run folder name in eval/results; default the latest")
    args = parser.parse_args()

    run_dir = RESULTS_DIR / args.run if args.run else latest_run()
    records = load_records(run_dir)
    results = [run_checks(r) for r in records]
    summary = summarise(records, results)

    with (run_dir / "checks.jsonl").open("w", encoding="utf-8") as f:
        for rec, res in zip(records, results):
            f.write(json.dumps({"id": rec["id"], **res}) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "report.md").write_text(report_md(run_dir, records, results, summary), encoding="utf-8")

    print(f"{run_dir.name}: {len(records)} questions\n")
    print(f"{'check':<24}{'overall':>12}{'tek17':>12}{'pricing':>12}")
    for check in CHECKS:
        cells = [fmt(*summary["overall"]["all"][check])]
        cells += [fmt(*summary["source"][s][check]) if s in summary["source"] else "-" for s in ("tek17", "pricing")]
        print(f"{check:<24}" + "".join(f"{c:>12}" for c in cells))
    r = summary["retrieval"]
    print(f"\nretrieval: hit@1 {r['hit@1']:.2f}  hit@3 {r['hit@3']:.2f}  hit@5 {r['hit@5']:.2f}  MRR {r['mrr']:.2f}")
    c = summary["cost"]
    print(f"cost: {c['avg_prompt_tokens']} prompt + {c['avg_completion_tokens']} completion tokens, "
          f"{c['avg_latency_ms']} ms per question; {c['errors']} errors")
    print(f"\nReport: {(run_dir / 'report.md').relative_to(RESULTS_DIR.parent.parent)}")


if __name__ == "__main__":
    main()