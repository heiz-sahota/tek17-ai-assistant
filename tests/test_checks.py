"""Unit tests for eval/checks.py and the kappa calculation. No Azure calls. Run: pytest -q"""
import pytest

from eval.calibrate import cohens_kappa
from eval.checks import (cited_expected, expected_rank, key_values_present, no_invented_citations, numbers,
                         refusal_correct, retrieval_hit)


def pricing_record(**overrides) -> dict:
    base = {
        "id": "p01", "source": "pricing", "type": "fact", "in_scope": True, "error": "",
        "question": "What is the fuel surcharge?",
        "reference_answer": "12.5 percent of the subtotal, from 1 January 2026",
        "expected_reference": None, "expected_doc": "01_Pricing_Model_Overview", "expected_section": "2. Price components",
        "answer": "The fuel surcharge is 12.5 percent of the subtotal, from 1 January 2026.",
        "citations": ["01_Pricing_Model_Overview / 2"], "found_in_sources": True, "invented_citations": [],
        "retrieved": [{"reference": "06_Pricing_Change_Log / Version 2026.1", "title": "Version 2026.1"},
                      {"reference": "01_Pricing_Model_Overview / 2", "title": "2. Price components"}],
        "used": [],
    }
    return base | overrides


OUT_OF_SCOPE = pricing_record(id="p18", type="out_of_scope", in_scope=False, expected_doc=None,
                              expected_section=None, reference_answer="Not in the documents",
                              answer="I couldn't find this in the sources I have access to.",
                              citations=[], found_in_sources=False)


# ---------- numbers ----------

@pytest.mark.parametrize("text, expected", [
    ("12.5 percent, from 1 January 2026", {12.5, 1.0, 2026.0}),
    ("divided by 5,000", {5000.0}),
    ("a multiplier of 1,60", {1.6}),
    ("1.60 for Zone D", {1.6}),
    ("Rule Z-2", {2.0}),
])
def test_numbers(text, expected):
    assert numbers(text) == expected


# ---------- retrieval ----------

def test_expected_rank_and_hit():
    assert expected_rank(pricing_record()) == 2
    assert retrieval_hit(pricing_record()) is True
    assert retrieval_hit(pricing_record(retrieved=[])) is False
    assert retrieval_hit(OUT_OF_SCOPE) is None


def test_tek17_reference_matching():
    rec = {"in_scope": True, "expected_reference": "§ 12-15",
           "retrieved": [{"reference": "TEK17 § 12-14", "title": "Stairs"}, {"reference": "TEK17 § 12-15", "title": "x"}]}
    assert expected_rank(rec) == 2


# ---------- behaviour ----------

def test_refusal_correct():
    assert refusal_correct(pricing_record()) is True
    assert refusal_correct(pricing_record(found_in_sources=False)) is False      # in scope, refused
    assert refusal_correct(OUT_OF_SCOPE) is True
    assert refusal_correct(OUT_OF_SCOPE | {"found_in_sources": True}) is False  # out of scope, answered
    assert refusal_correct(pricing_record(error="GenerationError")) is None


# ---------- citations ----------

def test_cited_expected():
    assert cited_expected(pricing_record()) is True
    assert cited_expected(pricing_record(citations=["06_Pricing_Change_Log / Version 2026.1"])) is False
    assert cited_expected(OUT_OF_SCOPE) is None


def test_no_invented_citations():
    assert no_invented_citations(pricing_record()) is True
    assert no_invented_citations(pricing_record(invented_citations=["TEK17 § 99-1"])) is False


# ---------- content ----------

def test_key_values_present():
    assert key_values_present(pricing_record()) is True
    assert key_values_present(pricing_record(answer="The fuel surcharge is 11 percent.")) is False
    assert key_values_present(pricing_record(reference_answer="Head of Pricing approves it")) is None
    assert key_values_present(OUT_OF_SCOPE) is None


# ---------- kappa ----------

def test_cohens_kappa():
    assert cohens_kappa(["pass", "fail", "pass"], ["pass", "fail", "pass"]) == 1.0
    a = ["pass"] * 8 + ["fail"] * 2
    b = ["pass"] * 7 + ["fail"] * 3
    # observed 0.9; expected 0.8*0.7 + 0.2*0.3 = 0.62; kappa = 0.28/0.38
    assert cohens_kappa(a, b) == pytest.approx(0.28 / 0.38)
    assert cohens_kappa([], []) is None