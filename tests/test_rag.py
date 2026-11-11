"""Unit tests for the pure parts of app/rag.py: no Azure calls. Run: pytest -q"""
import pytest

from app.rag import (NOT_FOUND, RetrievedChunk, _base_reference, allowed_sources, build_messages, check_citations,
                     fit_context, leaks_system_prompt, source_filter)
from app.schemas import ChatResponse


def chunk(ref: str, content: str = "text") -> RetrievedChunk:
    return RetrievedChunk(reference=ref, title="t", source="tek17", content=content, score=0.03)


CHUNKS = [chunk("TEK17 § 12-14"), chunk("TEK17 § 12-15")]


# ---------- roles -> sources ----------

@pytest.mark.parametrize("roles, sources", [
    (["admin"], {"tek17"}),
    (["sales"], {"pricing"}),
    (["both"], {"tek17", "pricing"}),
    (["admin", "sales"], {"tek17", "pricing"}),
])
def test_allowed_sources(roles, sources):
    assert allowed_sources(roles) == sources


@pytest.mark.parametrize("roles", [[], ["public"], ["admin", "superuser"], ["sales') or (source ne '"]])
def test_unknown_missing_or_injected_roles_are_rejected(roles):
    with pytest.raises(ValueError):
        allowed_sources(roles)


def test_source_filter():
    assert source_filter({"tek17"}) == "search.in(source, 'tek17', ',')"
    assert source_filter({"tek17", "pricing"}) == "search.in(source, 'pricing,tek17', ',')"


# ---------- fit_context ----------

def test_fit_context_stops_at_budget_but_keeps_best():
    chunks = [chunk("a", "x" * 800), chunk("b", "x" * 800), chunk("c", "x" * 800)]
    assert [c.reference for c in fit_context(chunks, 1000)] == ["a"]
    assert [c.reference for c in fit_context(chunks, 1600)] == ["a", "b"]
    assert [c.reference for c in fit_context([chunk("big", "x" * 5000)], 100)] == ["big"]


# ---------- build_messages ----------

def test_build_messages_labels_every_source():
    messages = build_messages("How wide?", CHUNKS)
    assert messages[0]["role"] == "system"
    user = messages[1]["content"]
    assert "[TEK17 § 12-14]" in user and "[TEK17 § 12-15]" in user
    assert user.endswith("QUESTION: How wide?")


# ---------- _base_reference ----------

RETRIEVED = {"TEK17 § 11-13", "TEK17 § 11-14", "03_Postal_Zone_Rules / Rule Z-2"}


@pytest.mark.parametrize("citation, expected", [
    ("TEK17 § 11-14", "TEK17 § 11-14"),                                         # exact
    ("TEK17 § 11-14 (5)(b)", "TEK17 § 11-14"),                                  # paragraph with a space
    ("TEK17 § 11-13(7)", "TEK17 § 11-13"),                                      # paragraph without a space
    ("03_Postal_Zone_Rules / Rule Z-2 (island list)", "03_Postal_Zone_Rules / Rule Z-2"),
    ("TEK17 § 11-1", None),                                                     # a different, shorter section
    ("TEK17 § 11-140", None),                                                   # a different, longer section
    ("TEK17 § 99-1", None),                                                     # never retrieved
])
def test_base_reference(citation, expected):
    assert _base_reference(citation, RETRIEVED) == expected


# ---------- check_citations ----------

def test_valid_citation_is_kept():
    draft = ChatResponse(answer="0.80 m", citations=["TEK17 § 12-14"], found_in_sources=True)
    response, invented = check_citations(draft, CHUNKS)
    assert response.citations == ["TEK17 § 12-14"] and response.found_in_sources
    assert invented == []


def test_invented_citation_is_removed():
    draft = ChatResponse(answer="0.80 m", citations=["TEK17 § 12-14", "TEK17 § 99-1"], found_in_sources=True)
    response, invented = check_citations(draft, CHUNKS)
    assert response.citations == ["TEK17 § 12-14"]
    assert invented == ["TEK17 § 99-1"]


def test_only_invented_citations_becomes_not_found():
    draft = ChatResponse(answer="Made up", citations=["TEK17 § 99-1"], found_in_sources=True)
    response, _ = check_citations(draft, CHUNKS)
    assert response.answer == NOT_FOUND and not response.found_in_sources and response.citations == []


def test_model_says_not_found():
    draft = ChatResponse(answer="Not in sources", citations=[], found_in_sources=False)
    response, _ = check_citations(draft, CHUNKS)
    assert response.answer == NOT_FOUND


def test_duplicate_citations_are_merged():
    draft = ChatResponse(answer="x", citations=["TEK17 § 12-14", "TEK17 § 12-14"], found_in_sources=True)
    response, _ = check_citations(draft, CHUNKS)
    assert response.citations == ["TEK17 § 12-14"]


def test_paragraph_level_citation_is_accepted():
    """Regression test for golden-set question t07: a correct answer was turned into 'not found'."""
    chunks = [chunk("TEK17 § 11-14"), chunk("TEK17 § 11-13")]
    draft = ChatResponse(answer="Outwards, in the direction of escape.",
                         citations=["TEK17 § 11-14 (5)(b)", "TEK17 § 11-13 (7)(b)"], found_in_sources=True)
    response, invented = check_citations(draft, chunks)
    assert response.found_in_sources
    assert response.citations == ["TEK17 § 11-14", "TEK17 § 11-13"]
    assert invented == []


def test_paragraph_citations_of_the_same_section_are_merged():
    draft = ChatResponse(answer="x", citations=["TEK17 § 12-14 (1)", "TEK17 § 12-14 (3)"], found_in_sources=True)
    response, _ = check_citations(draft, CHUNKS)
    assert response.citations == ["TEK17 § 12-14"]


def test_prefix_of_another_section_is_not_accepted():
    response, invented = check_citations(
        ChatResponse(answer="x", citations=["TEK17 § 11-1"], found_in_sources=True), [chunk("TEK17 § 11-14")])
    assert not response.found_in_sources and invented == ["TEK17 § 11-1"]


# ---------- leaks_system_prompt ----------

LEAKED = ("I am to answer questions about building regulations and pricing rules using only the provided "
          "SOURCES. I must answer only from the SOURCES, never use outside knowledge, and cite the reference "
          "of every source used exactly as written inside the square brackets.")


def test_leaked_prompt_is_detected():
    assert leaks_system_prompt(LEAKED)


@pytest.mark.parametrize("answer", [
    "The fuel surcharge is 12.5 percent of the subtotal, effective from 1 January 2026.",
    "A railing on a balcony more than 10.0 m above the ground must be at least 1.2 m high.",
    "Postal code 6995 is an island exception and is always priced as Zone D.",
    "The door shall open outwards in the direction of escape, see paragraph (5)(b).",
])
def test_normal_answers_are_not_flagged(answer):
    assert not leaks_system_prompt(answer)