"""
ingest/embed_upload.py: embed chunks and load them into Azure AI Search.
=======================================================================

Run from the project root:
    python -m ingest.embed_upload                              # recursive-500, both sources
    python -m ingest.embed_upload --chunks recursive-500 --recreate
    python -m ingest.embed_upload --skip-embed                 # upload text only, to test plumbing

Input:  data/chunks/<chunk set>/tek17.jsonl and pricing.jsonl   (from chunk.py)
Output: the index named AZURE_SEARCH_INDEX in .env, ready for hybrid search

Needs the role "Search Index Data Contributor" (upload) and
"Search Service Contributor" (create the index) on the search service,
or AZURE_SEARCH_API_KEY in .env.

THE STEPS
---------
  1. create_or_update_index()   schema: fields, HNSW vector field, semantic config
  2. load_chunks()              read the chunk set, check ids are unique
  3. embed_chunks()             1,536 numbers per chunk, batched, cached, retried
  4. upload()                   merge_or_upload_documents in batches, by stable id
  5. delete_stale()             remove chunks that no longer exist in the source
  6. verify()                   document count and one test hybrid query

PUSH MODEL
----------
Our code creates the vectors and pushes documents into the index. The managed
alternative is "integrated vectorization": an indexer + skillset reads files
from Blob Storage and calls the embedding model for you. Push gives full
control over parsing, chunking, batching, caching and errors, which this
project needs for § numbers and roles. Integrated is less code at scale.

THE INDEX IS A CONTRACT
-----------------------
The vector field's dimensions must equal the embedding model's output (1,536
for text-embedding-3-small). The analyzer decides how BM25 splits and stems
words. Neither can be changed in place: changing them means a new index and
re-uploading. In production, build "tek17-v2", run the eval on it, then point
the app's AZURE_SEARCH_INDEX at it.

IDEMPOTENT
----------
merge_or_upload_documents inserts new ids and replaces existing ones. Ids are
stable hashes from chunk.py, so running this twice gives the same index, not
twice as many documents. delete_stale() removes ids that are no longer
produced, such as a section that was removed from a document.

INTERVIEW LINE
--------------
"Ingestion ends with a push model. I define the schema myself: a key, a
searchable content field for BM25, a 1,536-dimension HNSW vector field with
cosine similarity, and filterable role and version fields for server-side
security filtering. Chunks are embedded in batches with a content-hash cache
and backoff on 429s, uploaded with merge-or-upload by stable id so re-runs
are idempotent, and stale chunks are deleted."
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery
from openai import RateLimitError

from app.config import (
    EMBEDDING_DEPLOYMENT,
    EMBEDDING_DIMENSIONS,
    SEARCH_INDEX,
    embed,
    index_client,
    search_client,
)

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_DIR = ROOT / "data" / "chunks"
CACHE_PATH = ROOT / "data" / "embeddings" / "cache.jsonl"   # shared with eval/compare_chunking.py
DEFAULT_CHUNK_SET = "recursive-500"

EMBED_BATCH = 64        # chunks per embeddings request: far fewer calls than one by one
UPLOAD_BATCH = 500      # documents per upload request (Azure allows up to 1,000 or 16 MB)
VECTOR_PROFILE = "vector-profile"
HNSW_CONFIG = "hnsw-config"
SEMANTIC_CONFIG = "default"
ANALYZER = "en.microsoft"   # English stemming: "stairs" matches "stair". "nb.microsoft" for Norwegian

INDEX_FIELDS = ["id", "content", "content_vector", "reference", "title", "source", "role", "version"]


# ======================================================================
# 1. Index schema
# ======================================================================

def build_index() -> SearchIndex:
    """
    Every field's settings are a decision:

      id              key, filterable                  the stable id from chunk.py
      content         searchable, English analyzer     BM25 keyword search runs here
      content_vector  1,536 floats, vector profile     vector search runs here; hidden,
                                                       so results don't carry 6 KB of numbers
      reference       searchable, filterable           cited; "§ 12-14" is findable by keyword
      title           searchable                       title field for the semantic ranker
      source, role,   filterable, facetable            server-side filters, e.g. role eq 'public'
      version

    Only filterable fields can be used in filter=, and only searchable fields
    are used by BM25. Marking everything searchable makes the index bigger and
    keyword results noisier.

    Returns: SearchIndex: the index definition (fields, vector search, semantic search), not yet sent to Azure.
    """
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name=ANALYZER),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            hidden=True,
            vector_search_dimensions=EMBEDDING_DIMENSIONS,
            vector_search_profile_name=VECTOR_PROFILE,
        ),
        SearchableField(name="reference", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="title", type=SearchFieldDataType.String, analyzer_name=ANALYZER),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="role", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="version", type=SearchFieldDataType.String, filterable=True, facetable=True),
    ]

    # HNSW: a graph where each vector links to its nearest neighbours.
    #   m               links per node: more = better recall, more memory
    #   ef_construction how carefully the graph is built: more = better graph, slower upload
    #   ef_search       how widely a query explores: more = better recall, slower queries
    #   metric          cosine, to match how OpenAI embeddings are compared
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(
            name=HNSW_CONFIG,
            parameters=HnswParameters(m=4, ef_construction=400, ef_search=500, metric="cosine"),
        )],
        profiles=[VectorSearchProfile(name=VECTOR_PROFILE, algorithm_configuration_name=HNSW_CONFIG)],
    )

    # Tells the semantic ranker which field is the title and which is the body,
    # so step 4 of the ranking funnel can be switched on with query_type="semantic".
    semantic_search = SemanticSearch(
        default_configuration_name=SEMANTIC_CONFIG,
        configurations=[SemanticConfiguration(
            name=SEMANTIC_CONFIG,
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="title"),
                content_fields=[SemanticField(field_name="content")],
                keywords_fields=[SemanticField(field_name="reference")],
            ),
        )],
    )

    return SearchIndex(name=SEARCH_INDEX, fields=fields,
                       vector_search=vector_search, semantic_search=semantic_search)


def create_or_update_index(recreate: bool = False) -> None:
    """
    recreate=True deletes the index first. Needed when a change can't be made
    in place: vector dimensions, analyzer, or a field's type. Without it,
    create_or_update_index() adds new fields and keeps existing documents.

    Returns: None. Prints the fields of the index Azure now has.
    """
    client = index_client()
    if recreate:
        try:
            client.delete_index(SEARCH_INDEX)
            print(f"Deleted index '{SEARCH_INDEX}'")
        except ResourceNotFoundError:
            pass
    index = client.create_or_update_index(build_index())
    print(f"Index '{index.name}' ready: {', '.join(f.name for f in index.fields)}")


# ======================================================================
# 2. Load chunks
# ======================================================================

def load_chunks(chunk_set: str) -> list[dict]:
    """
    Read every *.jsonl in data/chunks/<chunk_set>/.
    Duplicate ids are a hard error: the second would silently overwrite the first.

    Returns: list[dict]: one dict per chunk, with id, source, reference, title, version, role, content.
    """
    folder = CHUNKS_DIR / chunk_set
    if not folder.is_dir():
        raise SystemExit(f"{folder} not found. Run: python -m ingest.chunk --strategy recursive --size 500")

    chunks = []
    for path in sorted(folder.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            chunks += [json.loads(line) for line in f if line.strip()]
    if not chunks:
        raise SystemExit(f"No chunks in {folder}")

    ids = [c["id"] for c in chunks]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate chunk ids: fix chunk.py before uploading")
    print(f"Loaded {len(chunks)} chunks from {folder.relative_to(ROOT)}")
    return chunks


# ======================================================================
# 3. Embed, with cache, batching and retry
# ======================================================================

def text_key(text: str) -> str:
    """Cache key: hash of deployment + text. Same as eval/compare_chunking.py, so the cache is shared.

    Returns: str: a 64-character hex SHA-256 hash, used as the cache key.
    """
    return hashlib.sha256(f"{EMBEDDING_DEPLOYMENT}|{text}".encode("utf-8")).hexdigest()


def load_cache() -> dict[str, list[float]]:
    """data/embeddings/cache.jsonl as {key: vector}. Empty if it doesn't exist yet.

    Returns: dict[str, list[float]]: cache key -> vector of EMBEDDING_DIMENSIONS floats.
    """
    cache = {}
    if CACHE_PATH.exists():
        with CACHE_PATH.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    cache[row["key"]] = row["vector"]
    return cache


def embed_with_retry(texts: list[str], attempts: int = 5) -> list[list[float]]:
    """
    HTTP 429 (RateLimitError) means the deployment's tokens-per-minute quota is
    used up for now. Wait 1, 2, 4, 8 s (exponential backoff) and try again.
    Give up after the last attempt rather than loop forever.

    Returns: list[list[float]]: one vector per input text, in the same order.
    """
    for attempt in range(attempts):
        try:
            return embed(texts)
        except RateLimitError:
            if attempt == attempts - 1:
                raise
            wait = 2 ** attempt
            print(f"  rate limited, waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def embed_chunks(chunks: list[dict]) -> int:
    """
    Add chunk["content_vector"] to every chunk. Cached vectors are reused;
    only new or changed content is embedded, EMBED_BATCH at a time. New vectors
    are appended to the cache as each batch returns, so a crash loses nothing.
    Returns how many chunks needed a new embedding.

    Returns: int: how many chunks were newly embedded. Also adds "content_vector" to every chunk dict in place.
    """
    cache = load_cache()
    missing = [c for c in chunks if text_key(c["content"]) not in cache]

    if missing:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CACHE_PATH.open("a", encoding="utf-8") as f:
            for i in range(0, len(missing), EMBED_BATCH):
                batch = missing[i:i + EMBED_BATCH]
                vectors = embed_with_retry([c["content"] for c in batch])
                for chunk, vector in zip(batch, vectors):
                    key = text_key(chunk["content"])
                    cache[key] = vector
                    f.write(json.dumps({"key": key, "vector": vector}) + "\n")
                print(f"  embedded {min(i + EMBED_BATCH, len(missing))}/{len(missing)}")

    for chunk in chunks:
        vector = cache[text_key(chunk["content"])]
        if len(vector) != EMBEDDING_DIMENSIONS:
            raise SystemExit(f"{chunk['reference']}: vector has {len(vector)} numbers, "
                             f"index expects {EMBEDDING_DIMENSIONS}. Wrong deployment or dimensions?")
        chunk["content_vector"] = vector
    return len(missing)


# ======================================================================
# 4. Upload
# ======================================================================

def to_document(chunk: dict) -> dict:
    """Only the index's fields. An unknown field would make the whole batch fail.

    Returns: dict: the chunk limited to INDEX_FIELDS, ready to upload.
    """
    return {field: chunk[field] for field in INDEX_FIELDS if field in chunk}


def upload(chunks: list[dict]) -> None:
    """
    merge_or_upload_documents: new ids are inserted, existing ids replaced.
    Each result says whether that one document succeeded, so one bad document
    doesn't hide behind a successful call.

    Returns: None. Stops the script with SystemExit if any document failed.
    """
    client = search_client()
    failed = []
    for i in range(0, len(chunks), UPLOAD_BATCH):
        batch = [to_document(c) for c in chunks[i:i + UPLOAD_BATCH]]
        results = client.merge_or_upload_documents(documents=batch)
        failed += [r for r in results if not r.succeeded]
        print(f"  uploaded {min(i + UPLOAD_BATCH, len(chunks))}/{len(chunks)}")
    if failed:
        for r in failed[:5]:
            print(f"  FAILED {r.key}: {r.error_message}")
        raise SystemExit(f"{len(failed)} document(s) failed to upload")


# ======================================================================
# 5. Delete stale chunks
# ======================================================================

def delete_stale(current_ids: set[str]) -> int:
    """
    Ids in the index that this chunk set no longer produces belong to text that
    changed or was removed. If they stay, old wording keeps being found and cited.

    Returns: int: how many stale documents were deleted.
    """
    client = search_client()
    indexed = {doc["id"] for doc in client.search(search_text="*", select=["id"])}
    stale = sorted(indexed - current_ids)
    for i in range(0, len(stale), UPLOAD_BATCH):
        client.delete_documents(documents=[{"id": doc_id} for doc_id in stale[i:i + UPLOAD_BATCH]])
    return len(stale)


# ======================================================================
# 6. Verify
# ======================================================================

def verify(expected: int, question: str = "How wide must a stair inside a home be?") -> None:
    """
    The count is updated a few seconds after upload, so poll briefly.
    Then one hybrid query: BM25 on content + vector on content_vector, fused with
    RRF, filtered to public chunks. Expect TEK17 § 12-14 near the top.

    Returns: None. Prints the document count and the top 3 results of a test hybrid query.
    """
    client = search_client()
    count = 0
    for _ in range(10):
        count = client.get_document_count()
        if count == expected:
            break
        time.sleep(2)
    status = "OK" if count == expected else "MISMATCH, check again in a minute"
    print(f"\nDocuments in index: {count} (expected {expected}) {status}")

    results = client.search(
        search_text=question,
        vector_queries=[VectorizedQuery(vector=embed_with_retry([question])[0],
                                        k_nearest_neighbors=50, fields="content_vector")],
        filter="role eq 'public'",
        select=["reference", "title"],
        top=3,
    )
    print(f"Test hybrid query: {question}")
    for rank, r in enumerate(results, start=1):
        print(f"  {rank}. {r['reference']}  {r['title']}  score {r['@search.score']:.4f}")


def main() -> None:
    """Parse arguments and run the six steps in order.

    Returns: None.
    """
    parser = argparse.ArgumentParser(description="Embed chunks and upload them to Azure AI Search")
    parser.add_argument("--chunks", default=DEFAULT_CHUNK_SET, help="folder name in data/chunks/")
    parser.add_argument("--recreate", action="store_true", help="delete and rebuild the index first")
    parser.add_argument("--skip-embed", action="store_true",
                        help="upload without vectors, to test the plumbing")
    args = parser.parse_args()

    print(f"Index: {SEARCH_INDEX}   embedding deployment: {EMBEDDING_DEPLOYMENT}")
    create_or_update_index(args.recreate)
    chunks = load_chunks(args.chunks)

    if not args.skip_embed:
        new = embed_chunks(chunks)
        print(f"Embeddings: {new} new, {len(chunks) - new} from cache")

    upload(chunks)
    deleted = delete_stale({c["id"] for c in chunks})
    print(f"Deleted {deleted} stale document(s)")

    if not args.skip_embed:
        verify(len(chunks))


if __name__ == "__main__":
    main()