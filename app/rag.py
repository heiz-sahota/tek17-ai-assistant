"""
The question flow: retrieve, augment, generate, verify.

    answer(question, roles)
      1. retrieve()         hybrid search (BM25 + vector, RRF), only in the user's sources
      2. fit_context()      keep the best chunks that fit the context budget
      3. build_messages()   system rules + labelled sources + question
      4. generate()         chat model, temperature 0, JSON, validated by Pydantic
      5. check_citations()  keep only citations that were retrieved, else "not found"
      6. leaks_system_prompt()  block answers that repeat the system prompt

Production concerns handled here:
  - settings from environment variables, not hard-coded
  - user roles mapped to the sources they may search (ROLE_SOURCES), and the
    search filtered to those sources on the server
  - Azure and OpenAI failures raised as RetrievalError / GenerationError,
    which the API turns into a 503
  - a timeout on the model call
  - one structured log line per question: timings, tokens, chunks, outcome

The pure functions (allowed_sources, source_filter, fit_context, build_messages, check_citations)
need no Azure access and are unit-tested in tests/test_rag.py.
"""
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field

from azure.core.exceptions import HttpResponseError
from azure.search.documents.models import VectorizedQuery
from openai import APIError, RateLimitError
from pydantic import ValidationError

from app.config import CHAT_DEPLOYMENT, embed, openai_client, search_client
from app.prompts import SYSTEM_PROMPT
from app.schemas import ChatResponse

log = logging.getLogger("rag")

NOT_FOUND = "I couldn't find this in the sources I have access to."
# Which sources each user role may search. This is the ONE place access is decided.
#   admin -> TEK17 regulations     sales -> Lyngpakke pricing     both -> everything
# Chunks carry source="tek17" or source="pricing" from ingestion.
ROLE_SOURCES: dict[str, set[str]] = {
    "admin": {"tek17"},
    "sales": {"pricing"},
    "both": {"tek17", "pricing"},
}


# ======================================================================
# Settings and types
# ======================================================================

@dataclass(frozen=True)
class RagSettings:
    """Tunable settings, read once from the environment."""
    top_k: int = 5                      # chunks sent to the model
    k_nearest: int = 50                 # candidates from the vector side before fusion
    use_semantic_ranker: bool = False   # turn on after compare_search shows it helps
    min_reranker_score: float = 1.0     # 0-4 scale; only used with the semantic ranker
    max_context_chars: int = 12_000     # about 3,000 tokens of sources
    model_timeout_s: float = 30.0
    max_answer_tokens: int = 400        # caps the answer, and the tokens Azure reserves against TPM

    @classmethod
    def from_env(cls) -> "RagSettings":
        """
        Returns: RagSettings, with RAG_* environment variables overriding the defaults.
        """
        return cls(
            top_k=int(os.environ.get("RAG_TOP_K", cls.top_k)),
            k_nearest=int(os.environ.get("RAG_K_NEAREST", cls.k_nearest)),
            use_semantic_ranker=os.environ.get("RAG_SEMANTIC_RANKER", "false").lower() == "true",
            min_reranker_score=float(os.environ.get("RAG_MIN_RERANKER_SCORE", cls.min_reranker_score)),
            max_context_chars=int(os.environ.get("RAG_MAX_CONTEXT_CHARS", cls.max_context_chars)),
            model_timeout_s=float(os.environ.get("RAG_MODEL_TIMEOUT_S", cls.model_timeout_s)),
            max_answer_tokens=int(os.environ.get("RAG_MAX_ANSWER_TOKENS", cls.max_answer_tokens)),
        )


SETTINGS = RagSettings.from_env()


@dataclass
class RetrievedChunk:
    """One search result."""
    reference: str
    title: str
    source: str
    content: str
    score: float


@dataclass
class Trace:
    """What happened for one question. Logged as one JSON line, for dashboards and alerts."""
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timings_ms: dict[str, int] = field(default_factory=dict)
    chunks_retrieved: int = 0
    chunks_used: int = 0
    references: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    invented_citations: list[str] = field(default_factory=list)
    found_in_sources: bool = False
    outcome: str = ""

    def timed(self, step: str, started: float) -> None:
        """Record how long a step took since `started` (from time.perf_counter()). Returns: None."""
        self.timings_ms[step] = round((time.perf_counter() - started) * 1000)


class RetrievalError(Exception):
    """Azure AI Search failed: down, forbidden, or wrong index."""


class GenerationError(Exception):
    """The chat model failed after the client's retries, or timed out."""


class RateLimitedError(Exception):
    """A deployment's tokens- or requests-per-minute quota is still used up after the client's retries."""


# ======================================================================
# 1. Retrieve
# ======================================================================

def allowed_sources(roles: list[str]) -> set[str]:
    """
    The sources a user may search: the union of what each of their roles allows.
    Unknown roles are rejected rather than ignored, so a typo or a forged role
    fails loudly instead of quietly giving no or wrong access.

        ["admin"]          -> {"tek17"}
        ["sales"]          -> {"pricing"}
        ["admin", "sales"] -> {"tek17", "pricing"}     same as ["both"]

    Returns: set[str], source names.
    Raises: ValueError if roles is empty or contains an unknown role.
    """
    unknown = set(roles) - ROLE_SOURCES.keys()
    if not roles or unknown:
        raise ValueError(f"Unknown or missing roles: {sorted(unknown) or 'none'}")
    return set().union(*(ROLE_SOURCES[r] for r in roles))


def source_filter(sources: set[str]) -> str:
    """
    OData filter that keeps only chunks from the allowed sources. It runs inside
    Azure AI Search, so other sources' chunks are never returned, ranked or sent
    to the model. Values come from ROLE_SOURCES, never from user input.

    Returns: str, e.g. "search.in(source, 'pricing,tek17', ',')".
    """
    return f"search.in(source, '{','.join(sorted(sources))}', ',')"


def retrieve(question: str, roles: list[str], settings: RagSettings = SETTINGS) -> list[RetrievedChunk]:
    """
    Hybrid search: BM25 on content + vector search on content_vector, fused with
    RRF by Azure, limited to the sources the user's roles allow. With the semantic ranker on,
    results are reordered and weak ones (below min_reranker_score) are dropped.

    Returns: list[RetrievedChunk], best first, at most settings.top_k.
    Raises: ValueError for unknown roles, RetrievalError if embedding or search fails.
    """
    try:
        vector = embed([question])[0]
        kwargs = {
            "search_text": question,
            "vector_queries": [VectorizedQuery(vector=vector, k_nearest_neighbors=settings.k_nearest,
                                               fields="content_vector")],
            "filter": source_filter(allowed_sources(roles)),
            "select": ["reference", "title", "source", "content"],
            "top": settings.top_k,
        }
        if settings.use_semantic_ranker:
            kwargs |= {"query_type": "semantic", "semantic_configuration_name": "default"}
        results = list(search_client().search(**kwargs))
    except RateLimitError as e:
        raise RateLimitedError(str(e)) from e
    except (HttpResponseError, APIError) as e:
        raise RetrievalError(str(e)) from e

    chunks = []
    for r in results:
        reranker = r.get("@search.reranker_score")
        if settings.use_semantic_ranker and reranker is not None and reranker < settings.min_reranker_score:
            continue
        chunks.append(RetrievedChunk(reference=r["reference"], title=r["title"], source=r["source"],
                                     content=r["content"], score=reranker or r["@search.score"]))
    return chunks


# ======================================================================
# 2-3. Augment
# ======================================================================

def fit_context(chunks: list[RetrievedChunk], max_chars: int) -> list[RetrievedChunk]:
    """
    Keep chunks in rank order until the context budget is used. The best chunk
    is always kept, so the model never gets an empty context by accident.

    Returns: list[RetrievedChunk], a prefix of `chunks`.
    """
    kept, used = [], 0
    for chunk in chunks:
        if kept and used + len(chunk.content) > max_chars:
            break
        kept.append(chunk)
        used += len(chunk.content)
    return kept


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict]:
    """
    The fixed part (system rules) comes first and the changing part last, which
    also lets the provider's prompt caching work when the system prompt is long.
    Each source is labelled with its exact reference, which the model must cite.

    Returns: list[dict], messages for the chat completions API.
    """
    sources = "\n\n".join(f"[{c.reference}]\n{c.content}" for c in chunks)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"SOURCES:\n{sources}\n\nQUESTION: {question}"},
    ]


# ======================================================================
# 4. Generate
# ======================================================================

def generate(messages: list[dict], trace: Trace, settings: RagSettings = SETTINGS) -> ChatResponse:
    """
    Chat model with temperature 0 and JSON mode, then Pydantic validation.
    A reply that isn't valid ChatResponse JSON is treated as "not found",
    never shown to the user. Token usage is recorded on the trace.

    Returns: ChatResponse, as written by the model (citations not yet checked).
    Raises: RateLimitedError if the quota is still used up after retries,
            GenerationError if the model call fails or times out.
    """
    try:
        completion = openai_client().with_options(timeout=settings.model_timeout_s).chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=messages,
            temperature=0,
            max_tokens=settings.max_answer_tokens,
            response_format={"type": "json_object"},
        )
    except RateLimitError as e:
        raise RateLimitedError(str(e)) from e
    except APIError as e:
        raise GenerationError(str(e)) from e

    if completion.usage:
        trace.prompt_tokens = completion.usage.prompt_tokens
        trace.completion_tokens = completion.usage.completion_tokens
    try:
        return ChatResponse.model_validate_json(completion.choices[0].message.content or "")
    except ValidationError:
        trace.outcome = "invalid_model_json"
        return ChatResponse(answer=NOT_FOUND, citations=[], found_in_sources=False)


# ======================================================================
# 5. Verify
# ======================================================================

def _base_reference(citation: str, retrieved: set[str]) -> str | None:
    """
    The retrieved reference a citation points to. Exact match, or the reference
    followed by a paragraph marker, such as "TEK17 § 11-14 (5)(b)" -> "TEK17 § 11-14".
    The space or "(" after the reference stops "§ 11-1" matching "§ 11-14".

    Returns: str, or None if the citation doesn't point to any retrieved reference.
    """
    if citation in retrieved:
        return citation
    for ref in retrieved:
        if citation.startswith(ref + " ") or citation.startswith(ref + "("):
            return ref
    return None


def check_citations(response: ChatResponse, chunks: list[RetrievedChunk]) -> tuple[ChatResponse, list[str]]:
    """
    Keep only citations that point to chunks actually sent to the model. A paragraph-level
    citation like "TEK17 § 11-14 (5)(b)" counts, and is stored as "TEK17 § 11-14". With no valid
    citation, or found_in_sources false, the answer becomes "not found".

    Returns: (ChatResponse with verified citations, list of invented citations that were removed).
    """
    retrieved = {c.reference for c in chunks}
    resolved = [(c, _base_reference(c, retrieved)) for c in response.citations]
    valid = list(dict.fromkeys(base for _, base in resolved if base))      # dedupe, keep order
    invented = [c for c, base in resolved if base is None]
    if not response.found_in_sources or not valid:
        return ChatResponse(answer=NOT_FOUND, citations=[], found_in_sources=False), invented
    return ChatResponse(answer=response.answer, citations=valid, found_in_sources=True), invented


def _shingles(text: str, n: int = 5) -> set[tuple[str, ...]]:
    """
    Every run of n consecutive words, lower-cased, without punctuation.

    Returns: set[tuple[str, ...]].
    """
    words = re.findall(r"[a-zæøå0-9_]+", text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


_PROMPT_SHINGLES = _shingles(SYSTEM_PROMPT)


def leaks_system_prompt(answer_text: str, threshold: int = 2) -> bool:
    """
    True if the answer repeats the system prompt: at least `threshold` runs of
    5 consecutive words that also appear in it. A prompt rule asks the model not
    to reveal its instructions; this check in code catches it when it does anyway.
    Normal answers about stairs or prices never share 5-word runs with the rules.

    Returns: bool.
    """
    return len(_shingles(answer_text) & _PROMPT_SHINGLES) >= threshold


# ======================================================================
# Entry point
# ======================================================================

@dataclass
class RagResult:
    """Everything one question produced: what evals need besides the answer itself."""
    response: ChatResponse
    retrieved: list[RetrievedChunk]   # what search returned, best first
    used: list[RetrievedChunk]        # what was actually sent to the model, after fit_context
    trace: Trace


def answer_with_details(question: str, roles: list[str], settings: RagSettings = SETTINGS) -> RagResult:
    """
    The whole question flow, with one structured log line at the end.
    Evals call this directly, because judging needs the chunks the model actually saw.

    Returns: RagResult with the response, the retrieved and used chunks, and the trace.
    Raises: ValueError for unknown roles, RetrievalError, GenerationError, RateLimitedError.
    """
    trace = Trace()
    retrieved: list[RetrievedChunk] = []
    chunks: list[RetrievedChunk] = []
    try:
        started = time.perf_counter()
        retrieved = retrieve(question, roles, settings)
        trace.timed("retrieve", started)
        trace.chunks_retrieved = len(retrieved)

        if not retrieved:
            trace.outcome = "no_chunks"
            return RagResult(ChatResponse(answer=NOT_FOUND, citations=[], found_in_sources=False), [], [], trace)

        chunks = fit_context(retrieved, settings.max_context_chars)
        trace.chunks_used = len(chunks)
        trace.references = [c.reference for c in chunks]

        started = time.perf_counter()
        draft = generate(build_messages(question, chunks), trace, settings)
        trace.timed("generate", started)

        response, trace.invented_citations = check_citations(draft, chunks)
        if response.found_in_sources and leaks_system_prompt(response.answer):
            trace.outcome = "blocked_prompt_leak"
            response = ChatResponse(answer=NOT_FOUND, citations=[], found_in_sources=False)
        trace.found_in_sources = response.found_in_sources
        trace.outcome = trace.outcome or ("answered" if response.found_in_sources else "not_found")
        return RagResult(response, retrieved, chunks, trace)
    except Exception as e:
        trace.outcome = f"error:{type(e).__name__}"
        raise
    finally:
        trace.timings_ms["total"] = sum(trace.timings_ms.values())
        log.info(json.dumps({"event": "rag_answer", **trace.__dict__}, ensure_ascii=False))


def answer(question: str, roles: list[str], settings: RagSettings = SETTINGS) -> ChatResponse:
    """
    The question flow as the API uses it: only the response.

    Returns: ChatResponse with a grounded answer and verified citations, or "not found".
    Raises: ValueError for unknown roles, RetrievalError, GenerationError, RateLimitedError.
    """
    return answer_with_details(question, roles, settings).response