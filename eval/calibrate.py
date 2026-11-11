"""
Calibrate the judge: compare its verdicts with your own grades.

    python -m eval.calibrate init --n 15      # writes human_grades.csv in the latest run: grade it yourself
    python -m eval.calibrate score            # agreement and Cohen's kappa per criterion

Grade with eval/rubric.md, WITHOUT looking at judgments.jsonl first, or you will
anchor on the judge's verdicts. Fill each criterion column with pass, fail or n/a.
"""
import argparse
import csv
import json
import random

from eval.judge_prompt import CRITERIA
from eval.score import RESULTS_DIR, latest_run, load_records


def cohens_kappa(a: list[str], b: list[str]) -> float | None:
    """
    Agreement between two graders, corrected for agreement by chance.
    1.0 = perfect, 0 = no better than chance. Above about 0.6 is good, above 0.8 very good.

        kappa = (observed agreement - expected agreement) / (1 - expected agreement)

    Returns: float, or None if there are no items.
    """
    if not a:
        return None
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    labels = set(a) | set(b)
    expected = sum((a.count(lab) / n) * (b.count(lab) / n) for lab in labels)
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return (observed - expected) / (1 - expected)


def init(run_dir, n: int) -> None:
    """
    Write human_grades.csv: a random sample of judged answers, with empty columns to grade.

    Returns: None.
    """
    judged = {json.loads(l)["id"] for l in (run_dir / "judgments.jsonl").read_text(encoding="utf-8").splitlines() if l}
    records = [r for r in load_records(run_dir) if r["id"] in judged]
    sample = random.Random(42).sample(records, min(n, len(records)))
    path = run_dir / "human_grades.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "reference_answer", "answer", "citations", "sources_sent"] + CRITERIA)
        for r in sample:
            w.writerow([r["id"], r["question"], r["reference_answer"], r["answer"], "; ".join(r["citations"]),
                        "; ".join(c["reference"] for c in r.get("used", []))] + [""] * len(CRITERIA))
    print(f"Wrote {path.relative_to(RESULTS_DIR.parent.parent)} with {len(sample)} answers. "
          f"Grade each criterion pass, fail or n/a using eval/rubric.md, then run: python -m eval.calibrate score")


def score(run_dir) -> None:
    """
    Compare your grades with the judge's, per criterion. Rows where either side is n/a are left out.

    Returns: None. Prints agreement, kappa and every disagreement.
    """
    judge = {}
    for line in (run_dir / "judgments.jsonl").read_text(encoding="utf-8").splitlines():
        j = json.loads(line)
        if "error" not in j:
            judge[j["id"]] = j
    with (run_dir / "human_grades.csv").open(encoding="utf-8") as f:
        human = [row for row in csv.DictReader(f) if row["id"] in judge]

    print(f"{len(human)} answers graded by both\n")
    print(f"{'criterion':<16}{'compared':>10}{'agreement':>11}{'kappa':>8}")
    disagreements = []
    for c in CRITERIA:
        pairs = [(row[c].strip().lower(), judge[row["id"]][c]["verdict"]) for row in human]
        pairs = [(h, j) for h, j in pairs if h in ("pass", "fail") and j in ("pass", "fail")]
        if not pairs:
            print(f"{c:<16}{0:>10}{'-':>11}{'-':>8}")
            continue
        h, j = zip(*pairs)
        agreement = sum(x == y for x, y in pairs) / len(pairs)
        k = cohens_kappa(list(h), list(j))
        print(f"{c:<16}{len(pairs):>10}{agreement:>11.2f}{(k if k is not None else float('nan')):>8.2f}")
        for row in human:
            hv, jv = row[c].strip().lower(), judge[row["id"]][c]["verdict"]
            if hv in ("pass", "fail") and jv in ("pass", "fail") and hv != jv:
                disagreements.append((row["id"], c, hv, jv, judge[row["id"]][c]["reasoning"]))

    if disagreements:
        print("\nDisagreements: read the judge's reasoning, then decide whether the rubric or your grade is unclear")
        for id_, c, hv, jv, why in disagreements:
            print(f"  {id_:<4} {c:<16} you: {hv:<5} judge: {jv:<5} {why[:110]}")


def main() -> None:
    """
    Command-line entry point.

    Returns: None.
    """
    parser = argparse.ArgumentParser(description="Calibrate the LLM judge against your own grades")
    parser.add_argument("action", choices=["init", "score"])
    parser.add_argument("run", nargs="?", help="run folder name in eval/results; default the latest")
    parser.add_argument("--n", type=int, default=15, help="answers to sample for hand grading")
    args = parser.parse_args()
    run_dir = RESULTS_DIR / args.run if args.run else latest_run()
    init(run_dir, args.n) if args.action == "init" else score(run_dir)


if __name__ == "__main__":
    main()