"""
LLM as a judge: grade saved answers with the rubric in eval/rubric.md.

    python -m eval.judge --limit 10           # latest run, first 10 answers
    python -m eval.judge run_20261111_2115    # a specific run, all answers

Uses JUDGE_DEPLOYMENT from .env if set (ideally a different, stronger model),
otherwise the chat deployment. Temperature 0, JSON output.

Writes into the run folder:
    judgments.jsonl   per answer: verdict and reasoning for each criterion
    judge_summary.json
"""
import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from openai import APIError, RateLimitError

from app.config import CHAT_DEPLOYMENT, openai_client
from eval.golden import load_golden
from eval.judge_prompt import CRITERIA, judge_messages
from eval.score import RESULTS_DIR, latest_run, load_records

JUDGE_DEPLOYMENT = os.environ.get("JUDGE_DEPLOYMENT", CHAT_DEPLOYMENT)
VERDICTS = {"pass", "fail", "n/a"}

# Reference answers are read fresh from the golden set, so fixing a reference
# doesn't require re-running the assistant. Falls back to the copy saved with the run.
REFERENCES = {item.id: item.reference_answer for item in load_golden()}


def judge_one(record: dict) -> dict:
    """
    Grade one saved answer. The judge sees exactly the chunks the assistant was given.

    Returns: dict with rubric_version and, per criterion, {"reasoning", "verdict"};
             or {"error": ...} if the judge failed or replied with invalid JSON.
    """
    version, messages = judge_messages(
        question=record["question"],
        question_type=record["type"],
        reference=REFERENCES.get(record["id"], record["reference_answer"])
                  or "Not covered by the sources: the assistant should say so.",
        sources=record.get("used", []),
        answer=record["answer"],
        citations=record["citations"],
        found_in_sources=record["found_in_sources"],
    )
    for attempt in range(2):
        try:
            completion = openai_client().chat.completions.create(
                model=JUDGE_DEPLOYMENT, messages=messages, temperature=0,
                max_tokens=900, response_format={"type": "json_object"})
            break
        except RateLimitError:
            if attempt == 1:
                return {"error": "rate limited"}
            time.sleep(30)
        except APIError as e:
            return {"error": f"{type(e).__name__}: {e}"}

    try:
        verdicts = json.loads(completion.choices[0].message.content)
    except json.JSONDecodeError:
        return {"error": "judge reply was not JSON"}
    for criterion in CRITERIA:
        v = verdicts.get(criterion, {})
        if not isinstance(v, dict) or v.get("verdict") not in VERDICTS:
            return {"error": f"missing or invalid verdict for {criterion}", "raw": verdicts}
    verdicts["rubric_version"] = version
    verdicts["judge_deployment"] = JUDGE_DEPLOYMENT
    verdicts["judge_tokens"] = completion.usage.total_tokens if completion.usage else None
    return verdicts


def summarise(judgments: list[dict]) -> dict:
    """
    Pass rate per criterion over answers where it applied (n/a excluded),
    and how many answers passed every applicable criterion.

    Returns: dict.
    """
    counts = defaultdict(lambda: {"pass": 0, "fail": 0, "n/a": 0})
    all_pass = graded = 0
    for j in judgments:
        if "error" in j:
            continue
        graded += 1
        verdicts = [j[c]["verdict"] for c in CRITERIA]
        for c, v in zip(CRITERIA, verdicts):
            counts[c][v] += 1
        all_pass += "fail" not in verdicts
    rates = {c: {**counts[c], "pass_rate": (counts[c]["pass"] / (counts[c]["pass"] + counts[c]["fail"])
                                            if counts[c]["pass"] + counts[c]["fail"] else None)}
             for c in CRITERIA}
    return {"graded": graded, "errors": len(judgments) - graded,
            "all_criteria_pass": all_pass, "criteria": rates}


def main() -> None:
    """
    Command-line entry point.

    Returns: None. Prints pass rates and writes judgments.jsonl and judge_summary.json.
    """
    parser = argparse.ArgumentParser(description="Grade saved answers with the LLM judge")
    parser.add_argument("run", nargs="?", help="run folder name in eval/results; default the latest")
    parser.add_argument("--limit", type=int, help="only the first N answers")
    parser.add_argument("--pause", type=float, default=10.0, help="seconds between judge calls (rate limits)")
    args = parser.parse_args()

    run_dir = RESULTS_DIR / args.run if args.run else latest_run()
    records = [r for r in load_records(run_dir) if not r.get("error")][:args.limit]
    print(f"Judging {len(records)} answers from {run_dir.name} with '{JUDGE_DEPLOYMENT}'")

    judgments = []
    with (run_dir / "judgments.jsonl").open("w", encoding="utf-8") as out:
        for i, rec in enumerate(records, start=1):
            j = {"id": rec["id"], **judge_one(rec)}
            judgments.append(j)
            out.write(json.dumps(j, ensure_ascii=False) + "\n")
            out.flush()
            line = j.get("error") or "  ".join(f"{c}={j[c]['verdict']}" for c in CRITERIA)
            print(f"  {i:>2}/{len(records)} {rec['id']:<4} {line}")
            if i < len(records):
                time.sleep(args.pause)

    summary = summarise(judgments)
    (run_dir / "judge_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n{summary['graded']} graded, {summary['errors']} errors, "
          f"{summary['all_criteria_pass']} passed every applicable criterion")
    for c, r in summary["criteria"].items():
        pr = "-" if r["pass_rate"] is None else f"{r['pass_rate']:.2f}"
        print(f"  {c:<16} pass rate {pr:>5}   (pass {r['pass']}, fail {r['fail']}, n/a {r['n/a']})")


if __name__ == "__main__":
    main()