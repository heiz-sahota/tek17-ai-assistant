# Azure SDK reference for tek17-ai-assistant

Every Azure call this project makes: which file, what it's for, and an example.

The project doesn't use the **Azure Functions** service (serverless functions).
It uses **Azure SDK calls** to three services:

| Service | Python package | Used for |
|---|---|---|
| Microsoft Entra ID | `azure-identity`, `azure-core` | Signing in without keys |
| Azure OpenAI in Microsoft Foundry | `openai` | Embeddings and chat |
| Azure AI Search | `azure-search-documents` | The index: schema, upload, search |

All clients are created in **`app/config.py`**. The rest of the code calls
`embed()`, `openai_client()`, `search_client()` and `index_client()` and never
handles credentials itself.

---

## Contents

1. [Sign-in: azure-identity and azure-core](#1-sign-in-azure-identity-and-azure-core)
2. [Azure OpenAI in Foundry: openai](#2-azure-openai-in-foundry-openai)
3. [Azure AI Search clients](#3-azure-ai-search-clients)
4. [Index schema classes](#4-index-schema-classes)
5. [Index and document operations](#5-index-and-document-operations)
6. [Search queries](#6-search-queries)
7. [Errors](#7-errors)
8. [Roles each call needs](#8-roles-each-call-needs)
9. [Which file uses what](#9-which-file-uses-what)

---

## 1. Sign-in: azure-identity and azure-core

### `DefaultAzureCredential()`

**File:** `app/config.py` · `credential()`
**Purpose:** Signs in without keys. It tries several methods in order and uses the first that works:
- environment variables
- a **managed identity**, when running in Azure, such as Container Apps
- the **Azure CLI**, meaning your `az login` locally
- and a few more, such as VS Code

The same code therefore works on your laptop and in Azure.

```python
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
# Locally:  uses the account from `az login`
# In Azure: uses the Container App's managed identity
```

Check which account it will use locally:
```bash
az account show --query "{user:user.name, subscription:name}"
```

### `get_bearer_token_provider(credential, scope)`

**File:** `app/config.py` · `openai_client()`
**Purpose:** Turns the credential into a function that returns a fresh access token
whenever the OpenAI client needs one. Tokens expire after about an hour, and this renews them for you.
The scope says *which* service the token is for.

```python
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default",   # Azure OpenAI and Foundry
)
token_provider()   # -> "eyJ0eXAiOiJKV1Qi..."  (a bearer token string)
```

### `AzureKeyCredential(key)`

**File:** `app/config.py` · `_search_credential()`
**Purpose:** Fallback sign-in to Azure AI Search with an API key. It's used only when
`AZURE_SEARCH_API_KEY` is set in `.env`. Uploading needs an **admin** key; a query key can only search.

```python
from azure.core.credentials import AzureKeyCredential

credential = AzureKeyCredential(os.environ["AZURE_SEARCH_API_KEY"])
```

In this project:
```python
def _search_credential():
    key = os.environ.get("AZURE_SEARCH_API_KEY")
    return AzureKeyCredential(key) if key else credential()   # key if set, otherwise Entra ID
```

---

## 2. Azure OpenAI in Foundry: openai

### `AzureOpenAI(...)`

**File:** `app/config.py` · `openai_client()`
**Purpose:** The client for your Foundry model deployments (`embed` and `chat`).
`max_retries=5` makes it retry 429 (rate limit) and 5xx errors with exponential backoff.

```python
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint="https://<your-resource>.openai.azure.com/",
    api_version="2024-10-21",
    azure_ad_token_provider=token_provider,   # keyless; or api_key="..." as a fallback
    max_retries=5,
)
```

> `model=` in every call is the **deployment name** you chose in Foundry
> (`embed`, `chat`), not the model name (`text-embedding-3-small`).

### `client.embeddings.create(model, input)`

**File:** `app/config.py` · `embed()`
**Called from:**
- `app/rag.py`: embeds the question (B3)
- `ingest/embed_upload.py`: embeds the chunks (A4)
- `eval/compare_chunking.py` and `eval/compare_search.py`: the evals

**Purpose:** Turns text into vectors of 1,536 numbers. Pass a **list** to embed many texts in one call.

```python
result = client.embeddings.create(
    model="embed",
    input=["How wide must a stair be?", "Rule Z-2. Island exceptions"],
)
vectors = [d.embedding for d in result.data]
len(vectors)       # 2, one per input, same order
len(vectors[0])    # 1536
vectors[0][:3]     # [0.0123, -0.0456, 0.0089]
```

In this project:
```python
def embed(texts: list[str]) -> list[list[float]]:
    result = openai_client().embeddings.create(model=EMBEDDING_DEPLOYMENT, input=texts)
    return [d.embedding for d in result.data]
```

### `client.chat.completions.create(model, messages, temperature, response_format)`

**File:** `app/rag.py` · `generate()` (also `scripts/check_setup.py` as a smoke test)
**Purpose:** Writes the answer from the retrieved chunks.
- `temperature=0`: the most likely wording, so answers are consistent and repeatable for evals
- `response_format={"type": "json_object"}`: JSON mode, so the reply is always valid JSON

```python
completion = client.chat.completions.create(
    model="chat",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "SOURCES:\n[TEK17 § 12-14]\n...\n\nQUESTION: How wide must a stair be?"},
    ],
    temperature=0,
    response_format={"type": "json_object"},
)
text = completion.choices[0].message.content
# '{"answer": "At least 0.80 m ...", "citations": ["TEK17 § 12-14"], "found_in_sources": true}'

completion.usage.prompt_tokens       # tokens sent, e.g. 1450
completion.usage.completion_tokens   # tokens written, e.g. 60
```

The chat API is **stateless**: every call sends the system prompt, the sources and the question.

### `openai.RateLimitError`

**File:** `eval/compare_chunking.py` · `embed_with_retry()`
**Purpose:** The exception for HTTP 429, raised when the deployment's tokens-per-minute quota is used up.

```python
import time
from openai import RateLimitError

for attempt in range(5):
    try:
        vectors = embed(texts)
        break
    except RateLimitError:
        time.sleep(2 ** attempt)   # 1, 2, 4, 8, 16 s
```

The client's `max_retries=5` already does this. The manual loop is only in the eval script.

---

## 3. Azure AI Search clients

### `SearchClient(endpoint, index_name, credential)`

**File:** `app/config.py` · `search_client()`
**Purpose:** Works with **documents** in one index: search, upload, delete, count.

```python
from azure.search.documents import SearchClient

client = SearchClient(
    endpoint="https://<your-search>.search.windows.net",
    index_name="tek17",
    credential=DefaultAzureCredential(),
)
client.get_document_count()   # e.g. 57
```

### `SearchIndexClient(endpoint, credential)`

**File:** `app/config.py` · `index_client()`
**Purpose:** Works with **indexes**: create, update, delete and list the schema.

```python
from azure.search.documents.indexes import SearchIndexClient

index_client = SearchIndexClient("https://<your-search>.search.windows.net", DefaultAzureCredential())
[i.name for i in index_client.list_indexes()]   # ['tek17']
```

> SearchClient = the rows. SearchIndexClient = the table definition.

---

## 4. Index schema classes

**File:** `ingest/embed_upload.py` · `build_index()`
**Package:** `azure.search.documents.indexes.models`

| Class | Purpose |
|---|---|
| `SimpleField` | Not full-text searchable. Used for keys and filters |
| `SearchableField` | Full-text (BM25) searchable text, with an analyzer |
| `SearchField` | General field. Used for the vector field |
| `SearchFieldDataType` | Field types: `String`, `Single`, `Collection(...)` |
| `HnswAlgorithmConfiguration`, `HnswParameters` | How vectors are indexed and compared |
| `VectorSearch`, `VectorSearchProfile` | Connects a vector field to an algorithm |
| `SemanticSearch`, `SemanticConfiguration`, `SemanticPrioritizedFields`, `SemanticField` | Settings for the semantic ranker |
| `SearchIndex` | The whole index definition |

### `SimpleField`: key and filters

```python
SimpleField(name="id",   type=SearchFieldDataType.String, key=True, filterable=True)
SimpleField(name="role", type=SearchFieldDataType.String, filterable=True)
# Enables:  filter="role eq 'public'"
```

### `SearchableField`: keyword search

```python
SearchableField(name="content", analyzer_name="en.microsoft")
# BM25 runs here. The English analyzer stems words: "stairs" matches "stair".
# For Norwegian text: analyzer_name="nb.microsoft"
```

### `SearchField` with `Collection(Single)`: the vector field

```python
SearchField(
    name="content_vector",
    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),   # a list of floats
    searchable=True,
    hidden=True,                          # not returned in results (6 KB of numbers)
    vector_search_dimensions=1536,        # must equal the embedding model's output
    vector_search_profile_name="vector-profile",
)
```

### `HnswAlgorithmConfiguration`, `VectorSearchProfile`, `VectorSearch`

```python
vector_search = VectorSearch(
    algorithms=[HnswAlgorithmConfiguration(
        name="hnsw",
        parameters=HnswParameters(metric="cosine"),   # defaults: m=4, ef_construction=400, ef_search=500
    )],
    profiles=[VectorSearchProfile(name="vector-profile", algorithm_configuration_name="hnsw")],
)
```

HNSW is a graph where each vector links to its nearest neighbours, so a search
can jump towards close vectors instead of comparing all of them.
- `m`: links per node
- `ef_construction`: how carefully the graph is built
- `ef_search`: how widely each query explores

Higher values give better recall but use more memory and are slower.

### `SemanticSearch`: settings for the semantic ranker

```python
semantic_search = SemanticSearch(configurations=[SemanticConfiguration(
    name="default",
    prioritized_fields=SemanticPrioritizedFields(
        title_field=SemanticField(field_name="title"),
        content_fields=[SemanticField(field_name="content")],
    ),
)])
```

### `SearchIndex`: putting it together

```python
index = SearchIndex(name="tek17", fields=fields,
                    vector_search=vector_search, semantic_search=semantic_search)
```

---

## 5. Index and document operations

**File:** `ingest/embed_upload.py`

### `index_client.create_or_update_index(index)`

**Function:** `ensure_index()`
**Purpose:** Creates the index, or updates it in place, for example by adding a field. It
can't change vector dimensions, analyzers or field types: those need a delete and rebuild.

```python
index_client().create_or_update_index(build_index())
```

### `index_client.delete_index(name)`

**Function:** `ensure_index()`, with `--recreate`
**Purpose:** Deletes the index **and all its documents**.

```python
try:
    index_client().delete_index("tek17")
except ResourceNotFoundError:
    pass   # it didn't exist, which is fine
```

### `search_client.merge_or_upload_documents(documents)`

**Function:** `upload()`
**Purpose:** Inserts new documents and replaces existing ones with the same `id`. Running it twice is safe.
At most 1,000 documents or 16 MB per call. The project sends 500.

```python
results = search_client().merge_or_upload_documents(documents=[
    {"id": "tek17-7a50b49c...", "content": "Reference: TEK17 § 11-1 ...",
     "content_vector": [0.0123, ...], "reference": "TEK17 § 11-1",
     "title": "Safety in case of fire", "source": "tek17", "role": "public", "version": "July 2017"},
])
for r in results:
    print(r.key, r.succeeded, r.error_message)   # tek17-7a50b49c... True None
```

Each document gets its own result. One can fail while the others succeed, so always check `succeeded`.

Related calls, not used here:

| Call | Behaviour |
|---|---|
| `upload_documents` | Insert, or fully replace |
| `merge_documents` | Update only some fields of an existing document; fails if it doesn't exist |
| `merge_or_upload_documents` | Merge if it exists, upload if it doesn't (used here) |

### `search_client.delete_documents(documents)`

**Function:** `delete()`
**Purpose:** Removes documents by key. Only the `id` is needed.

```python
search_client().delete_documents(documents=[{"id": "pricing-3f9a..."}, {"id": "tek17-81bc..."}])
```

### `search_client.get_document_count()`

**Purpose:** The number of documents in the index. It updates a few seconds after an upload.

```python
search_client().get_document_count()   # 57
```

---

## 6. Search queries

### `search_client.search(...)`

The single search call. What you pass in decides the mode.

| Parameter | Meaning | Example |
|---|---|---|
| `search_text` | Keyword (BM25) query. `"*"` = everything, `None` = no keyword part | `"stair width"` |
| `vector_queries` | Vector part, a list of `VectorizedQuery` | see below |
| `filter` | OData filter, applied inside the search | `"role eq 'public'"` |
| `select` | Fields to return | `["reference", "title"]` |
| `top` | Number of results | `5` |
| `query_type` | `"semantic"` turns on the semantic ranker | `"semantic"` |
| `semantic_configuration_name` | Which semantic config to use | `"default"` |

Each result is a dict with your selected fields plus scores:
- `@search.score`
- `@search.reranker_score`, in semantic mode, from 0 to 4

### `VectorizedQuery(vector, k_nearest_neighbors, fields)`

**Files:** `app/rag.py`, `eval/compare_search.py`
**Purpose:** The vector part of a search: find the `k` nearest chunks to this vector.

```python
from azure.search.documents.models import VectorizedQuery

vq = VectorizedQuery(vector=embed(["How wide must a stair be?"])[0],
                     k_nearest_neighbors=50,          # candidates from the vector side
                     fields="content_vector")
```

### The four modes

**1. Keyword (BM25)**, in `eval/compare_search.py`
```python
search_client().search(search_text="balcony railing height", top=5)
```

**2. Vector**, in `eval/compare_search.py`
```python
search_client().search(search_text=None, vector_queries=[vq], top=5)
```

**3. Hybrid (BM25 + vector, merged with RRF)**: used by the app in `app/rag.py` · `retrieve()`
```python
results = search_client().search(
    search_text=question,
    vector_queries=[VectorizedQuery(vector=embed([question])[0], k_nearest_neighbors=50,
                                    fields="content_vector")],
    filter="search.in(role, 'public,pricing', ',')",
    select=["reference", "title", "source", "content"],
    top=5,
)
for r in results:
    print(r["reference"], round(r["@search.score"], 4))
# TEK17 § 12-14  0.0325
# TEK17 § 12-15  0.0310
```
Hybrid scores are small, around 0.03, because they're RRF scores: the sum of `1 / (60 + rank)`.

**4. Hybrid + semantic ranker**, in `eval/compare_search.py`
```python
search_client().search(search_text=question, vector_queries=[vq], top=5,
                       query_type="semantic", semantic_configuration_name="default")
# each result also has r["@search.reranker_score"], from 0 to 4
```

### List every id

**File:** `ingest/embed_upload.py` · `indexed_ids()`
```python
ids = {doc["id"] for doc in search_client().search(search_text="*", select=["id"])}
```
The results iterator fetches page after page for you.

### OData filters

```python
"role eq 'public'"                                   # equals
"role ne 'pricing'"                                  # not equal
"source eq 'tek17' and version eq 'July 2017'"       # and / or
"search.in(role, 'public,pricing', ',')"             # one of several values (used here)
```
A field must be marked `filterable=True` in the schema to be used in a filter.

---

## 7. Errors

**Package:** `azure.core.exceptions`

### `HttpResponseError`

**File:** `app/main.py` · `chat()`
**Purpose:** Any failed Azure call, such as 403 Forbidden or the service being down.
The API turns it into a clear 503 for the user and logs the details.

```python
from azure.core.exceptions import HttpResponseError

try:
    return rag.answer(request.question, roles)
except HttpResponseError as e:
    log.error("Search failed: %s", e.message)        # e.status_code -> 403
    raise HTTPException(status_code=503, detail="Search is unavailable, try again shortly")
```

### `ResourceNotFoundError`

**File:** `ingest/embed_upload.py` · `ensure_index()`
**Purpose:** HTTP 404, a subclass of `HttpResponseError`. Here it means
"the index doesn't exist", which is fine when deleting before a rebuild.

### Common status codes in this project

| Code | Meaning here | Fix |
|---|---|---|
| 401 | Not signed in, or the token expired | `az login` |
| 403 | Signed in, but missing a role | Add the role (section 8) and wait 5–10 minutes |
| 404 | Wrong endpoint, deployment name or index name | Check `.env` |
| 429 | Quota used up | Retries handle it; raise the quota in Foundry if it keeps happening |

---

## 8. Roles each call needs

| Who | Role | Covers |
|---|---|---|
| You (ingestion) | **Search Service Contributor** | `create_or_update_index`, `delete_index` |
| You (ingestion) | **Search Index Data Contributor** | `merge_or_upload_documents`, `delete_documents`, `search` |
| You and the app | **Cognitive Services OpenAI User**, or Foundry User | `embeddings.create`, `chat.completions.create` |
| The deployed API | **Search Index Data Reader** | `search` only: least privilege |

Also check: the search service's **Settings → Keys → API access control** must be
*Role-based access control* or *Both*, or every Entra ID sign-in is refused.

> Search Service Contributor manages the service and indexes, but cannot read
> or write documents. That needs a **data** role. This is the cause of a 403 on upload.

---

## 9. Which file uses what

| File | Azure calls |
|---|---|
| `app/config.py` | `DefaultAzureCredential`, `get_bearer_token_provider`, `AzureKeyCredential`, `AzureOpenAI`, `SearchClient`, `SearchIndexClient`, `embeddings.create` |
| `app/rag.py` | `embed()`, `search()` with `VectorizedQuery` (hybrid), `chat.completions.create` |
| `app/main.py` | `HttpResponseError` |
| `ingest/embed_upload.py` | Schema classes, `create_or_update_index`, `delete_index`, `merge_or_upload_documents`, `delete_documents`, `search("*")`, `embed()`, `ResourceNotFoundError` |
| `eval/compare_search.py` | `search()` in four modes, `VectorizedQuery`, `embed()` |
| `eval/compare_chunking.py` | `embed()`, `RateLimitError` |
| `scripts/check_setup.py` | `chat.completions.create` |
| `ingest/parse.py`, `ingest/chunk.py` | None: local files only |

---

## Interview summary

> "Three Azure SDKs: `azure-identity` for keyless sign-in with `DefaultAzureCredential`,
> the `openai` SDK's `AzureOpenAI` client for embeddings and chat on Foundry deployments,
> and `azure-search-documents`, with `SearchIndexClient` for the schema and `SearchClient`
> for upload and hybrid search. All clients are created in one config module, so the
> rest of the code never handles credentials. Locally it signs in with `az login`, and in
> Azure with a managed identity that has only the Search Index Data Reader and OpenAI
> User roles."