"""
Code checks: deterministic, free, instant. One small function per check.

Each check takes one saved record from answers.jsonl and returns
    True   pass
    False  fail
    None   not applicable to this question

    retrieval_hit          RETRIEVAL   the expected section is among the retrieved chunks
    refusal_correct        BEHAVIOUR   answered when it should, said "not found" when it should
    cited_expected         CITATIONS   the expected section is among the answer's citations
    no_invented_citations  CITATIONS   the citation check removed nothing
    key_values_present     CONTENT     every number in the reference answer appears in the answer

What code can't check (grounded, correct in substance, complete) is the judge's job.
"""
import re

from eval.golden import matches

CHECKS = ["retrieval_hit", "refusal_correct", "cited_expected", "no_invented_citations", "key_values_present"]

_NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:[.,]\d+)?)(?![\w])")


def numbers(text: str) -> set[float]:
    """
    Every number in a text, as floats, so "1.6" equals "1.60" and "5,000" equals "5000".
    A comma followed by exactly three digits is a thousands separator; otherwise a decimal comma.

        "12.5 percent, from 1 January 2026"  -> {12.5, 1.0, 2026.0}

    Returns: set[float].
    """
    found = set()
    for raw in _NUMBER.findall(text):
        if re.fullmatch(r"\d{1,3}(?:,\d{3})+", raw):
            raw = raw.replace(",", "")
        found.add(float(raw.replace(",", ".")))
    return found


def expected_rank(record: dict) -> int | None:
    """
    1-based rank of the first retrieved chunk that is the expected section.

    Returns: int, or None if it wasn't retrieved or the question has no expected section.
    """
    if not record.get("in_scope"):
        return None
    for rank, chunk in enumerate(record.get("retrieved", []), start=1):
        if matches(record, chunk):
            return rank
    return None


def retrieval_hit(record: dict) -> bool | None:
    """RETRIEVAL. Returns: True if the expected section was retrieved at all (top k); None if out of scope."""
    if not record.get("in_scope"):
        return None
    return expected_rank(record) is not None


def refusal_correct(record: dict) -> bool | None:
    """
    BEHAVIOUR. In scope: must be answered. Out of scope: must be "not found".
    An out-of-scope question answered with "the sources say this isn't covered" fails
    here but can be right; the judge's refusal criterion decides those.

    Returns: bool, or None if the run had an error.
    """
    if record.get("error"):
        return None
    return bool(record["found_in_sources"]) == bool(record["in_scope"])


def cited_expected(record: dict) -> bool | None:
    """
    CITATIONS. The answer cites the expected section. Citations are references
    such as "03_Postal_Zone_Rules / Rule Z-2"; titles come from the chunks sent to the model.

    Returns: bool, or None if out of scope or not answered.
    """
    if not record.get("in_scope") or not record.get("found_in_sources"):
        return None
    titles = {c["reference"]: c.get("title", "") for c in record.get("used", []) + record.get("retrieved", [])}
    return any(matches(record, {"reference": cit, "title": titles.get(cit, "")}) for cit in record["citations"])


def no_invented_citations(record: dict) -> bool | None:
    """CITATIONS. Returns: True if the citation check removed nothing; None if the run had an error."""
    if record.get("error"):
        return None
    return not record.get("invented_citations")


def key_values_present(record: dict) -> bool | None:
    """
    CONTENT. Every number in the reference answer appears in the answer.
    Strict on purpose: "12.5 percent of the subtotal" needs 12.5; "1.2 m, more than 10.0 m" needs both.

    Returns: bool, or None if out of scope, not answered, or the reference has no numbers.
    """
    if not record.get("in_scope") or not record.get("found_in_sources"):
        return None
    wanted = numbers(record["reference_answer"])
    if not wanted:
        return None
    return wanted <= numbers(record["answer"])


def run_checks(record: dict) -> dict:
    """
    Every check on one record, plus the retrieval rank.

    Returns: dict, {"retrieval_hit": True, ..., "rank": 2}.
    """
    results = {name: globals()[name](record) for name in CHECKS}
    results["rank"] = expected_rank(record)
    return results