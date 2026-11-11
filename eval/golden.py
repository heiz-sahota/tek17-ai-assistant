"""
The golden set: both question files loaded into one shape.

    eval/questions_tek17.jsonl        TEK17 questions, expected_reference like "§ 12-15"
    eval/pricing_eval_questions.jsonl pricing questions, expected_doc + expected_section

Out-of-scope questions (type "out_of_scope", no expected section) are included:
they test that the assistant refuses.
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_FILES = {
    "tek17": EVAL_DIR / "questions_tek17.jsonl",
    "pricing": EVAL_DIR / "pricing_eval_questions.jsonl",
}


@dataclass
class GoldenItem:
    id: str
    source: str                    # "tek17" or "pricing"
    question: str
    type: str                      # fact, number, condition, yes_no, out_of_scope ...
    reference_answer: str
    expected_reference: str | None = None   # TEK17: "§ 12-15"
    expected_doc: str | None = None         # pricing: "03_Postal_Zone_Rules"
    expected_section: str | None = None     # pricing: "Rule Z-2. Island exceptions"

    @property
    def in_scope(self) -> bool:
        """True if there is a section the assistant should find and answer from."""
        return self.type != "out_of_scope" and bool(self.expected_reference or self.expected_section)

    def expected(self) -> dict:
        """The fields used to recognise the right chunk. Returns: dict."""
        return {"expected_reference": self.expected_reference, "expected_doc": self.expected_doc,
                "expected_section": self.expected_section}

    def to_dict(self) -> dict:
        """Returns: dict, all fields plus in_scope."""
        return asdict(self) | {"in_scope": self.in_scope}


def load_golden(sources: list[str] | None = None) -> list[GoldenItem]:
    """
    Read the golden set. Template rows ("Write your question...") are skipped.

    Returns: list[GoldenItem], TEK17 first, then pricing, in file order.
    """
    items = []
    for source, path in GOLDEN_FILES.items():
        if sources and source not in sources or not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["question"].startswith("Write your"):
                continue
            items.append(GoldenItem(
                id=row["id"], source=source, question=row["question"], type=row.get("type", "fact"),
                reference_answer=row.get("answer", ""),
                expected_reference=row.get("expected_reference"),
                expected_doc=row.get("expected_doc"), expected_section=row.get("expected_section"),
            ))
    return items


def normalise(text: str) -> str:
    """Lower-case, single spaces, without "TEK17", so "TEK17 § 12-15" equals "§ 12-15". Returns: str."""
    return " ".join(text.replace("TEK17", "").split()).lower()


def matches(expected: dict, chunk: dict) -> bool:
    """
    Is this chunk the one the question expects?
      TEK17:   chunk reference "TEK17 § 12-15" matches expected_reference "§ 12-15"
      pricing: chunk reference starts with expected_doc, and its title equals expected_section

    Returns: bool.
    """
    if expected.get("expected_reference"):
        return normalise(chunk["reference"]) == normalise(expected["expected_reference"])
    if expected.get("expected_doc") and expected.get("expected_section"):
        return (chunk["reference"].startswith(expected["expected_doc"])
                and normalise(chunk.get("title", "")) == normalise(expected["expected_section"]))
    return False