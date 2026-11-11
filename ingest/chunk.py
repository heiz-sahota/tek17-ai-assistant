"""
ingest/chunk.py: split parsed sections into chunks for the search index.
=======================================================================

Input:  data/sections/tek17.jsonl and data/sections/pricing.jsonl  (from parse.py)
Output: data/chunks/<strategy>-<size>/tek17.jsonl and pricing.jsonl, one Chunk per line

Run from the project root:
    python -m ingest.chunk                                   # recursive and fixed, 500 tokens
    python -m ingest.chunk --strategy both --size 500 200 100  # several sizes, for comparing
    python -m ingest.chunk --strategy recursive --size 500     # the one you'll index

Then compare them:  python -m eval.compare_chunking

Library: langchain-text-splitters (only the splitters, not all of LangChain)
    pip install langchain-text-splitters tiktoken

WHY CHUNK AT ALL
----------------
Search returns chunks, not documents, and the model only reads the top k.
- Too big: one vector has to represent many topics, so it matches none of them
  well, and the prompt fills with text that doesn't help.
- Too small: a rule is cut off from its condition ("...applies only to
  islands" lands in another chunk), so the answer is incomplete.
The golden set decides the size, not a rule of thumb.

TWO STRATEGIES, SO YOU CAN COMPARE THEM
---------------------------------------
1. "recursive" (structure-aware, the main one)
   Runs INSIDE each parsed section, so a chunk never mixes two sections and
   always keeps its section's reference for the citation.
   RecursiveCharacterTextSplitter tries the biggest natural break first:
       blank line "\\n\\n" -> TEK17 paragraph "\\n(" -> line "\\n" -> sentence ". " -> word " "
   Tables are kept whole. Every chunk starts with a Reference and Title line.

2. "fixed" (the baseline most tutorials use)
   Joins each WHOLE document and cuts it every N tokens with overlap, ignoring
   sections, sentences and tables. No context header. A chunk that spans two
   sections is labelled with the section most of its tokens come from.
   It's here to prove with numbers that structure-aware chunking is better,
   instead of assuming it.

TOKENS, NOT CHARACTERS
----------------------
Sizes are measured with tiktoken's cl100k_base, the tokenizer of
text-embedding-3-small. Embedding limits and cost are counted in tokens.

OVERLAP
-------
The last ~15% of one chunk is repeated at the start of the next, so a
sentence near a boundary is fully inside at least one chunk.

STABLE IDS
----------
id = source + a hash of (reference, chunk number, content). The same chunk
always gets the same id, so re-running ingestion overwrites instead of adding
duplicates. Azure AI Search keys allow only letters, digits, "_", "-" and "=".

INTERVIEW LINE
--------------
"I chunk recursively with LangChain's splitter, measured in tokens, inside
each parsed section, so chunks follow the document's structure and keep their
§ number for citations. Tables are never split. I compared it with fixed-size
chunking at several sizes on the same golden set and kept the best."
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.schemas import Chunk

DATA = Path(__file__).resolve().parent.parent / "data"
SECTIONS_DIR = DATA / "sections"
CHUNKS_DIR = DATA / "chunks"
SOURCES = ["tek17", "pricing"]

ENCODING = "cl100k_base"   # tokenizer used by text-embedding-3-small
DEFAULT_SIZE = 500         # tokens; a starting point, tuned with the eval
OVERLAP_RATIO = 0.15       # overlap = 15 percent of the chunk size

# Tried in order: the biggest natural break first, then smaller ones.
# "\n(" is TEK17's own paragraph marker, as in "(1) ..." and "(2) ...".
SEPARATORS = ["\n\n", "\n(", "\n", ". ", " ", ""]

_encoding = tiktoken.get_encoding(ENCODING)


def count_tokens(text: str) -> int:
    """How many tokens the embedding model will see for this text."""
    return len(_encoding.encode(text))


def overlap_for(size: int) -> int:
    return int(size * OVERLAP_RATIO)


# ======================================================================
# Shared helpers
# ======================================================================

def make_id(source: str, reference: str, index: int, content: str) -> str:
    """Stable, index-safe key: the same chunk always gets the same id."""
    digest = hashlib.sha256(f"{reference}|{index}|{content}".encode("utf-8")).hexdigest()
    return f"{source}-{digest[:24]}"


def context_header(section: dict) -> str:
    """
    Two short lines at the top of every recursive chunk:
        Reference: TEK17 § 12-15
        Title: Stairs
    The reference is written exactly as the model must cite it.
    """
    return f"Reference: {section['reference']}\nTitle: {section['title']}"


def to_chunk(section: dict, index: int, content: str) -> Chunk:
    """Build a validated Chunk with the section's metadata."""
    return Chunk(
        id=make_id(section["source"], section["reference"], index, content),
        source=section["source"],
        reference=section["reference"],
        title=section["title"],
        version=section["version"],
        role=section["role"],
        content=content,
    )


# ======================================================================
# Strategy 1: recursive, inside each section
# ======================================================================

def make_recursive_splitter(size: int) -> RecursiveCharacterTextSplitter:
    """from_tiktoken_encoder makes the splitter count tokens instead of characters."""
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=ENCODING,
        chunk_size=size,
        chunk_overlap=overlap_for(size),
        separators=SEPARATORS,
        keep_separator=True,   # keeps "(2)" at the start of its paragraph
    )


def is_table_line(line: str) -> bool:
    """parse.py writes table rows as cells joined with " | "."""
    return " | " in line


def split_text_and_tables(text: str) -> list[tuple[str, str]]:
    """
    Cut a section's text into ("text", ...) and ("table", ...) blocks, in order.
    Consecutive " | " lines form one table, header row first.

        "Intro.\\nWeight | Price\\n0–1 kg | 89 NOK\\nAfter."
        -> [("text", "Intro."), ("table", "Weight | Price\\n0–1 kg | 89 NOK"), ("text", "After.")]
    """
    blocks, buffer, kind = [], [], None
    for line in text.split("\n"):
        line_kind = "table" if is_table_line(line) else "text"
        if line_kind != kind and buffer:
            blocks.append((kind, "\n".join(buffer).strip()))
            buffer = []
        kind = line_kind
        buffer.append(line)
    if buffer:
        blocks.append((kind, "\n".join(buffer).strip()))
    return [(k, b) for k, b in blocks if b]


def split_big_table(table: str, max_tokens: int, count=count_tokens) -> list[str]:
    """
    A table too big on its own is split by rows, and EVERY piece repeats the
    header row. Values make no sense without their column names.
    """
    header, *rows = table.split("\n")
    pieces, current = [], [header]
    for row in rows:
        if len(current) > 1 and count("\n".join(current + [row])) > max_tokens:
            pieces.append("\n".join(current))
            current = [header]
        current.append(row)
    pieces.append("\n".join(current))
    return pieces


def pack(pieces: list[str], max_tokens: int, count=count_tokens) -> list[str]:
    """
    Join neighbouring small pieces back together, up to max_tokens.
    Without this, a short sentence before a table would become a tiny chunk
    with too little meaning to be found.
    """
    packed, current = [], ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if current and count(candidate) > max_tokens:
            packed.append(current)
            current = piece
        else:
            current = candidate
    if current:
        packed.append(current)
    return packed


def recursive_pieces(text: str, splitter: RecursiveCharacterTextSplitter, size: int) -> list[str]:
    """Text blocks go through the recursive splitter; tables stay whole."""
    if count_tokens(text) <= size:
        return [text]                   # most sections fit in one chunk
    pieces = []
    for kind, block in split_text_and_tables(text):
        if kind == "table":
            pieces += split_big_table(block, size)
        else:
            pieces += splitter.split_text(block)
    return pack(pieces, size)


def chunk_section(section: dict, splitter: RecursiveCharacterTextSplitter, size: int) -> list[Chunk]:
    """One section -> one or more Chunks, each starting with the context header."""
    header = context_header(section)
    return [to_chunk(section, i, f"{header}\n\n{piece.strip()}")
            for i, piece in enumerate(recursive_pieces(section["text"], splitter, size))]


def chunk_recursive(sections: list[dict], size: int) -> list[Chunk]:
    splitter = make_recursive_splitter(size)
    return [c for s in sections for c in chunk_section(s, splitter, size)]


# ======================================================================
# Strategy 2: fixed size over the whole document (the baseline)
# ======================================================================

def chunk_fixed(sections: list[dict], size: int) -> list[Chunk]:
    """
    For each document: encode all its sections into one token list, remembering
    which section each token came from. Then take windows of `size` tokens,
    stepping by size - overlap, and decode each window back to text.

    Each window is labelled with the section most of its tokens belong to,
    which is the citation a fixed-size system would show.
    """
    step = size - overlap_for(size)
    by_doc = defaultdict(list)
    for s in sections:
        by_doc[s["doc"]].append(s)

    chunks = []
    for doc_sections in by_doc.values():
        tokens, owner = [], []
        for i, s in enumerate(doc_sections):
            ids = _encoding.encode(f"{s['title']}\n{s['text']}\n\n")
            tokens += ids
            owner += [i] * len(ids)

        for n, start in enumerate(range(0, len(tokens), step)):
            window = tokens[start:start + size]
            main_section = doc_sections[Counter(owner[start:start + size]).most_common(1)[0][0]]
            text = _encoding.decode(window).strip()   # may cut a word or "æ" at the edges: that's the point
            if text:
                chunks.append(to_chunk(main_section, n, text))
            if start + size >= len(tokens):
                break
    return chunks


# ======================================================================
# Files
# ======================================================================

def read_sections(source: str) -> list[dict]:
    path = SECTIONS_DIR / f"{source}.jsonl"
    if not path.exists():
        print(f"Skipping {source}: {path} not found. Run python -m ingest.parse first.")
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")   # keeps § and æøå readable


def check_chunks(chunks: list[Chunk], size: int) -> list[str]:
    """Problems worth looking at before spending money on embeddings."""
    problems = []
    ids = [c.id for c in chunks]
    if len(ids) != len(set(ids)):
        problems.append("duplicate ids: re-uploads would overwrite the wrong chunk")
    for c in chunks:
        tokens = count_tokens(c.content)
        if tokens > size + 100:
            problems.append(f"{c.reference}: {tokens} tokens, larger than expected")
        if tokens < 25:
            problems.append(f"{c.reference}: only {tokens} tokens, probably too little to be found")
    return problems


def summary(chunks: list[Chunk]) -> str:
    if not chunks:
        return "0 chunks"
    sizes = sorted(count_tokens(c.content) for c in chunks)
    return (f"{len(chunks)} chunks, tokens min {sizes[0]}, "
            f"median {sizes[len(sizes) // 2]}, max {sizes[-1]}")


STRATEGIES = {"recursive": chunk_recursive, "fixed": chunk_fixed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk parsed sections")
    parser.add_argument("--strategy", choices=["recursive", "fixed", "both"], default="both")
    parser.add_argument("--size", type=int, nargs="+", default=[DEFAULT_SIZE],
                        help="one or more chunk sizes in tokens, e.g. --size 500 200 100")
    args = parser.parse_args()

    strategies = ["recursive", "fixed"] if args.strategy == "both" else [args.strategy]
    for strategy in strategies:
        for size in args.size:
            name = f"{strategy}-{size}"
            print(f"\n=== {name} (overlap {overlap_for(size)}) ===")
            for source in SOURCES:
                sections = read_sections(source)
                if not sections:
                    continue
                chunks = STRATEGIES[strategy](sections, size)
                out = CHUNKS_DIR / name / f"{source}.jsonl"
                write_chunks(chunks, out)
                print(f"{source}: {len(sections)} sections -> {summary(chunks)}  -> {out.relative_to(DATA.parent)}")
                for problem in check_chunks(chunks, size)[:5]:
                    print(f"  WARNING {problem}")



if __name__ == "__main__":
    main()