"""
ingest/parse.py: turn source files into sections.
=================================================

WHERE THIS FITS
---------------
Part A of the RAG flow, building the index:

    A1 Documents -> A2 PARSE (this file) -> A3 Chunk -> A4 Embed -> A5 Index

Parsing turns files into clean, labelled SECTIONS. It does not cut them into
chunks, create embeddings or talk to Azure. Keeping it separate means you can
open the output and check it by eye before you spend money embedding anything.

Input:  data/tek17/tek17_en.pdf and the 6 files in data/lyngpakke/
Output: data/sections/tek17.jsonl and data/sections/pricing.jsonl

WHAT A SECTION LOOKS LIKE
-------------------------
    {
      "source": "pricing",                              # which knowledge base: "tek17" or "pricing"
      "doc": "03_Postal_Zone_Rules",                    # which file it came from
      "reference": "03_Postal_Zone_Rules / Rule Z-2",   # what the answer will CITE
      "title": "Rule Z-2. Island exceptions",           # the heading
      "text": "Some postal codes are on islands ...",
      "version": "2026.1",                              # which version of the document
      "role": "pricing"                                 # who may see it ("public" for TEK17)
    }

Every field has a job later:
  - reference -> the citation, and the citation check in app/rag.py
  - role      -> the security filter in Azure AI Search (role eq 'public')
  - version   -> handling old and new values, e.g. Zone D 1.50 vs 1.60
  - source    -> searching one source or both, and reporting evaluation per source

KEY IDEAS
---------
1. Unstructured data: PDFs and Word files have no fields. Parsing REBUILDS the
   structure (sections, headings, tables) the author had in mind.
2. PDFs store text by position on the page, not as headings and paragraphs.
   Headers, footers and page numbers get mixed into the text and must be removed.
3. Word files DO keep structure: paragraph styles ("Heading 1") and real tables.
   Use it rather than guessing from the text.
4. The reference is created HERE, while you still know which section a piece of
   text belongs to. After chunking, that knowledge is gone.
5. Garbage in, garbage out: if a rule is split from its exception here, no
   embedding model or prompt can fix it later.

INTERVIEW LINE
--------------
"I parse documents into sections with a stable reference and metadata before
chunking. For PDFs I remove repeated headers and footers and split on the
section headings. For Word files I use the heading styles and keep tables whole
with their header row. I run code checks and read samples before embedding."

BEFORE FIRST RUN
----------------
1. Set TEK17_VERSION to the date on the PDF cover.
2. Print one TEK17 page and check TEK17_HEADING matches its headings, and add
   any repeated header or footer to TEK17_NOISE. (Lyngpakke is already tested.)

Usage: python -m ingest.parse
"""
import json
import random
import re
from collections import Counter
from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

DATA = Path(__file__).resolve().parent.parent / "data"   # works whatever folder you run from
TEK17_PDF = DATA / "tek17" / "tek17_en.pdf"
LYNGPAKKE_DIR = DATA / "lyngpakke"
OUT_DIR = DATA / "sections"

"""
SETTINGS
--------
VERSION: the date printed on the TEK17 PDF cover goes on every TEK17 section,
so an answer can say which version it used. Regulations change over time.

CHAPTERS: you only index two chapters. A smaller, well-evaluated set beats a
large set you can't check. Add more later without changing code.
"""
TEK17_VERSION = "July 2017"
TEK17_CHAPTERS = [11, 12]
LYNGPAKKE_VERSION = "2026.1"

"""
NOISE PATTERNS
--------------
Text repeated on every page: running headers, footers, page numbers. Left in,
it ends up inside chunks and pollutes search results.

Regex quick reference:
    \\.     a literal dot (a plain . means "any character")
    \\s+    one or more spaces or line breaks
    \\d+    one or more digits
    ^ $    start / end of a line, with re.MULTILINE

TEK17_NOISE: print two pages, find the text on both, and add a pattern for it.
Example for a footer "Page 12 of 140":  r"Page \\d+ of \\d+"
"""
TEK17_NOISE: list[str] = [
    r"(?m)^\s*Page \d+( of \d+)?\s*$",     # a line that is only "Page 12" or "Page 12 of 140"
    r"(?m)^\s*Chapter \d+\b.*$",            # chapter title lines, which would otherwise end up
                                             # at the end of the previous section. Check on the real PDF.
    r"(?m)^.*\.{5,}\s*\d+\s*$",             # table of contents lines: "Section 11-1 Safety ..... 27"
    r"(?m)^[IVX]+\.\s+[A-Z][A-Z ,]+\s*$",     # part headings in capitals: "II. E NTRANCES, SAFETY IN USE, ..."
]
LYNGPAKKE_NOISE = [
    r"Lyngpakke AS is a fictional company\.\s+This is sample data for a personal RAG project\.\s+Page\s+\d+",
]

"""
HEADING PATTERNS
----------------
A heading pattern finds where each section STARTS. Everything between one
heading and the next belongs to the first.

Groups ( ) capture the parts you need: group 1 the number, group 2 the title.

TEK17_HEADING accepts, at the start of a line, the title on the SAME line or the NEXT:
    "Section 12-9 Bathrooms and toilets"
    "Section 11-11"  followed by  "General requirements relating to escape and rescue"
pypdf also extracts most numbers with a space before the dash: "Section 11 -2".
The pattern allows spaces around the dash, and split_tek17_sections removes them.
In the DiBK English PDF headings can also be the second form. Found by printing
the real text: the first version of this pattern only found 12-9.
`(?:\n[ \t]*)?` is what allows one optional line break between number and title.
CHECK IT on the real text:
    for m in re.finditer(TEK17_HEADING, text, re.MULTILINE): print(m.group(0))

LYNGPAKKE_PDF_HEADING matches, at the start of a line:
    "Rule Z-2. Island exceptions", "2. Price components", "2.1 Chargeable weight"

Traps handled below:
  - Cross-references inside a paragraph ("see § 12-2") don't match, because the
    pattern is anchored with ^ to the start of a line.
  - The table of contents repeats the headings with no body. split_tek17_sections
    keeps the LONGEST body for each section number.
  - Numbered list items ("1. Find the chargeable weight ...") look like headings.
    is_heading_title() rejects them: real headings are short and don't end with
    a full stop, while list items are sentences.
"""
TEK17_HEADING = r"^(?:§|Section)[ \t]*(\d{1,2}[ \t]*-[ \t]*\d{1,2})\.?[ \t]*(?:\n[ \t]*)?([^\n]*\S)[ \t]*$"
LYNGPAKKE_PDF_HEADING = r"^(Rule Z-\d+|\d{1,2}(?:\.\d{1,2})?)\.?\s+(\S.*)$"


# ======================================================================
# STEP 1: read a PDF
# ======================================================================

def read_pdf_pages(path: Path) -> list[str]:
    """
    Return the text of each page, in order.

    pypdf reads the text objects on each page in the order the PDF stores them,
    which is usually reading order for simple documents. A list of pages (not
    one string) lets you inspect and clean page by page.

    WATCH OUT: extract_text() can return None, hence `or ""`. A scanned PDF
    returns empty pages and needs OCR. Tables come out as plain lines, e.g.
    "0000–1999 A", so their structure is lost. Word files keep it.
    """
    reader = PdfReader(path)
    return [page.extract_text() or "" for page in reader.pages]


# ======================================================================
# STEP 2: clean the text
# ======================================================================

def clean_text(text: str, noise: list[str]) -> str:
    """
    Remove noise and tidy whitespace without changing the meaning:
      1. Remove each noise pattern (footers, headers, page numbers).
      2. Join words hyphenated across lines, "require-\\nments" -> "requirements",
         only when both sides are lowercase, so "Z-\\n2" is not turned into "Z2".
      3. Many spaces -> one, and no spaces around line breaks.
      4. Three or more line breaks -> one blank line.

    Line breaks are kept because headings start at the beginning of a line.

    Page numbers are removed only through explicit noise patterns, never by
    deleting every number on its own line, because table cells such as "89"
    can stand alone too. Cleaning is where data quietly gets lost: prefer
    narrow rules you can explain.
    """
    for pattern in noise:
        text = re.sub(pattern, " ", text)
    text = re.sub(r"([a-zæøå])-\n([a-zæøå])", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ======================================================================
# STEP 3: build one section
# ======================================================================

def make_section(source: str, doc: str, reference: str, title: str,
                 text: str, version: str, role: str) -> dict:
    """
    One builder for both parsers, so TEK17 and Lyngpakke sections always have
    exactly the same shape that chunk.py and the index expect.

    .strip() matters: a trailing space in a reference would make the citation
    check fail ("Rule Z-2 " != "Rule Z-2").
    """
    return {
        "source": source,
        "doc": doc,
        "reference": reference.strip(),
        "title": title.strip(),
        "text": text.strip(),
        "version": version,
        "role": role,
    }


# ======================================================================
# Shared: the "split at headings" pattern, and telling headings from list items
# ======================================================================

def is_heading_title(title: str) -> bool:
    """
    A real heading title is short and does not end with a full stop.
    A numbered list item is a sentence: "Find the chargeable weight ... band."

    This is a HEURISTIC, a practical rule that works for these documents.
    Test it on every new source, and mention it in an interview as a
    deliberate parsing decision.
    """
    title = title.strip()
    return (0 < len(title) <= 80
            and not title.endswith(".")
            and not title.startswith("(")                  # "(1) ..." is body text, not a title
            and re.search(r"[a-zæøå]", title) is not None)   # "NOK" or "A" alone is a table cell, not a title


def split_at_headings(text: str, pattern: str) -> list[dict]:
    """
    The core of structure-aware splitting, used by steps 4 and 10.

    1. Find every heading with re.finditer (MULTILINE, so ^ means line start).
    2. Ignore matches whose title isn't a real heading (list items).
    3. Each section's body runs from the end of its heading to the start of
       the next heading. Text before the first heading is dropped.

    Returns [{"label": "Rule Z-2", "heading": "Rule Z-2. Island exceptions",
              "title": "Island exceptions", "text": "..."}].
    """
    matches = [m for m in re.finditer(pattern, text, re.MULTILINE)
               if is_heading_title(m.group(2))]
    sections = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append({
            "label": m.group(1),
            "heading": m.group(0).strip(),
            "title": m.group(2).strip(),
            "text": text[m.end():end].strip(),
        })
    return sections


# ======================================================================
# STEP 4: split TEK17 into sections
# ======================================================================

def split_tek17_sections(text: str) -> list[dict]:
    """
    Cut the cleaned TEK17 text into one piece per numbered section.
    Returns [{"number": "12-2", "title": "...", "text": "..."}].

    TABLE OF CONTENTS: the same headings appear near the start of the PDF with
    little or no body. Keeping the LONGEST body for each number keeps the real
    section and drops the contents entry. Sections with an empty body are dropped.

    INTERVIEW: "I split on the regulation's own section numbers, so each section
    can be cited exactly, and a rule stays with its exceptions."
    """
    best: dict[str, dict] = {}
    for part in split_at_headings(text, TEK17_HEADING):
        if not part["text"]:
            continue
        number = re.sub(r"\s+", "", part["label"])   # "11 -2" in the extracted text -> "11-2"
        if number not in best or len(part["text"]) > len(best[number]["text"]):
            best[number] = {"number": number, "title": part["title"], "text": part["text"]}
    # Sort by chapter, then section, so the output follows the regulation's own order
    return sorted(best.values(), key=lambda s: tuple(int(n) for n in s["number"].split("-")))


# ======================================================================
# STEP 5: keep only your chapters
# ======================================================================

def filter_chapters(sections: list[dict], chapters: list[int]) -> list[dict]:
    """
    The chapter is the number before the dash: "12-2" -> 12.
    Two chapters you can evaluate properly beat the whole regulation you can't.
    """
    return [s for s in sections if int(s["number"].split("-")[0]) in chapters]


# ======================================================================
# STEP 6: the whole TEK17 path
# ======================================================================

def parse_tek17(path: Path, version: str, chapters: list[int]) -> list[dict]:
    """
    Read -> clean each page -> join -> split -> filter -> make sections.

    Pages are cleaned one by one before joining, so a footer at the bottom of a
    page can't merge with the first line of the next.

    role="public": TEK17 is public law. Pricing documents get "pricing", so the
    role filter in search can tell them apart.
    """
    pages = [clean_text(p, TEK17_NOISE) for p in read_pdf_pages(path)]
    sections = filter_chapters(split_tek17_sections("\n".join(pages)), chapters)
    return [
        make_section(source="tek17", doc="tek17", reference=f"TEK17 § {s['number']}",
                     title=s["title"], text=s["text"], version=version, role="public")
        for s in sections
    ]


# ======================================================================
# STEP 7: a Word table as text
# ======================================================================

def table_to_text(table: Table) -> str:
    """
    Header row first, then one line per row, cells joined with " | ":

        Weight band | Base price | Typical parcel
        0–1 kg | 89 NOK | Documents, small accessories

    The header row is what tells the model which column a value belongs to.
    "10 percent | Account manager" means nothing without "Discount on base
    price | Approval" above it. In chunk.py, tables must never be split.
    """
    rows = []
    for row in table.rows:
        rows.append(" | ".join(cell.text.strip() for cell in row.cells))
    return "\n".join(rows)


# ======================================================================
# STEP 8: a Word file in document order
# ======================================================================

def read_docx_blocks(path: Path) -> list[tuple[str, str]]:
    """
    Return everything in order as (kind, text):
        [("Title", "Weight and Size Price Table"),
         ("Heading 1", "1. Base prices by weight band"),
         ("Normal", "Base prices are in NOK ..."),
         ("table", "Weight band | Base price | ...")]

    WHY ORDER MATTERS: python-docx has `document.paragraphs` and
    `document.tables` as two SEPARATE lists. Used separately, you can't tell
    which heading a table sits under. Walking the XML body keeps the order.

    A .docx is XML: paragraphs are <w:p>, tables <w:tbl>. Tags include a
    namespace in braces, e.g. "{http://...}p", so we check the ending.
    """
    document = Document(path)
    blocks = []
    for child in document.element.body.iterchildren():
        if child.tag.endswith("}p"):
            paragraph = Paragraph(child, document)
            text = paragraph.text.strip()
            if text:
                blocks.append((paragraph.style.name, text))
        elif child.tag.endswith("}tbl"):
            blocks.append(("table", table_to_text(Table(child, document))))
    return blocks


# ======================================================================
# STEP 9: split Word blocks at headings
# ======================================================================

def split_blocks_by_heading(blocks: list[tuple[str, str]], doc: str) -> list[dict]:
    """
    One section per "Heading 1" or "Heading 2". Following paragraphs and tables
    belong to it until the next heading.

    Uses the document's own styles instead of regex: the author already marked
    what is a heading, which is more reliable than guessing.

    Blocks before the first heading (the title and the "Owner | Version" line)
    are skipped. The version is stored as metadata instead.

    Returns [{"label": "2", "title": "2. Volume discount tiers", "text": "..."}].
    """
    sections, current = [], None
    for kind, text in blocks:
        if kind in ("Heading 1", "Heading 2"):
            if current and current["text"].strip():
                sections.append(current)
            current = {"label": short_label(text), "title": text, "text": ""}
        elif current is not None:
            current["text"] += text + "\n"
    if current and current["text"].strip():
        sections.append(current)
    return sections


def short_label(title: str) -> str:
    """
    The short, stable part of a heading used in the reference:
        "Rule Z-2. Island exceptions"                -> "Rule Z-2"
        "2.1 Chargeable weight"                      -> "2.1"
        "Version 2026.1, effective 1 January 2026"   -> "Version 2026.1"
    Falls back to the whole title.

    Short, consistent references matter because the model must copy them
    exactly in citations, and the citation check compares them as strings.
    """
    m = re.match(r"(Rule Z-\d+|Version \d+\.\d+|\d+(?:\.\d+)?)", title.strip())
    return m.group(1) if m else title.strip()


# ======================================================================
# STEP 10: split a Lyngpakke PDF at headings
# ======================================================================

def split_pdf_by_headings(text: str, doc: str) -> list[dict]:
    """
    The same split-at-headings idea as step 4, for 01_Pricing_Model_Overview
    and 03_Postal_Zone_Rules.

    THE TABLE-CELL PROBLEM: the worked example table has a cell "217.49 NOK"
    on its own line, which looked like section "217.49". Two fixes: section
    numbers have at most two digits, and a title must contain a
    lowercase letter. Found by reading the parsed output, not by guessing.

    THE LIST-ITEM PROBLEM: section "3. Calculation order" contains a numbered
    list ("1. Find the chargeable weight ... band."). Those lines match the
    heading pattern too. is_heading_title() rejects them because they are
    full sentences ending with a full stop. Tested: 03 gives Rule Z-1 to Z-6,
    and 01 keeps the six-step list inside section 3.

    The full heading is kept as the title, e.g. "Rule Z-2. Island exceptions".
    """
    parts = split_at_headings(text, LYNGPAKKE_PDF_HEADING)
    return [{"label": short_label(p["label"]), "title": p["heading"], "text": p["text"]}
            for p in parts if p["text"]]


# ======================================================================
# STEP 11: the whole Lyngpakke path
# ======================================================================

def parse_lyngpakke(folder: Path, version: str) -> list[dict]:
    """
    .docx -> read_docx_blocks + split_blocks_by_heading
    .pdf  -> read_pdf_pages + clean_text per page + split_pdf_by_headings

    Reference = file name + " / " + short label:
        "03_Postal_Zone_Rules / Rule Z-2"
        "01_Pricing_Model_Overview / 2.1"
        "06_Pricing_Change_Log / Version 2026.1"
    """
    sections = []
    for path in sorted(folder.iterdir()):
        doc = path.stem
        if path.suffix.lower() == ".docx":
            parts = split_blocks_by_heading(read_docx_blocks(path), doc)
        elif path.suffix.lower() == ".pdf":
            pages = [clean_text(p, LYNGPAKKE_NOISE) for p in read_pdf_pages(path)]
            parts = split_pdf_by_headings("\n".join(pages), doc)
        else:
            continue
        for part in parts:
            sections.append(make_section(
                source="pricing", doc=doc, reference=f"{doc} / {part['label']}",
                title=part["title"], text=part["text"], version=version, role="pricing"))
    return sections


# ======================================================================
# STEP 12: check the result
# ======================================================================

def check_sections(sections: list[dict]) -> int:
    """
    Cheap, exact code checks before any model is involved. Prints warnings and
    returns how many it found, but doesn't stop the run.

      - Empty text            -> a heading with nothing under it
      - Duplicate references  -> table of contents caught, or a bad label
      - Fewer than 10 words   -> usually a split in the wrong place
      - Noise left behind     -> a footer or page number pattern that missed

    Same principle as answer evaluation in section 6: code checks first,
    expensive checks after.

    INTERVIEW: "Before embedding, I run code checks on parsed sections: empty
    sections, duplicate references and leftover noise. It caught ..."
    (fill in what it actually caught).
    """
    warnings = 0
    for ref, count in Counter(s["reference"] for s in sections).items():
        if count > 1:
            print(f"WARNING duplicate reference: {ref} x{count}")
            warnings += 1
    for s in sections:
        words = len(s["text"].split())
        if words == 0:
            print(f"WARNING empty section: {s['reference']}")
            warnings += 1
        elif words < 10:
            print(f"WARNING short section ({words} words): {s['reference']}")
            warnings += 1
        if re.search(r"fictional company|Page \d+ of \d+", s["text"]):
            print(f"WARNING noise left in: {s['reference']}")
            warnings += 1
    print(f"Checks done: {warnings} warning(s) in {len(sections)} sections")
    return warnings


# ======================================================================
# STEP 13: save
# ======================================================================

def write_jsonl(sections: list[dict], path: Path) -> None:
    """
    One JSON object per line: easy to read, append to and inspect.
    ensure_ascii=False keeps "§", "–" and Norwegian letters readable.
    encoding="utf-8" makes the result independent of the computer's defaults.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for section in sections:
            f.write(json.dumps(section, ensure_ascii=False) + "\n")


# ======================================================================
# STEP 14: run everything
# ======================================================================

def main() -> None:
    """
    Parse both sources, check, save, and show a summary plus 3 random sections.
    Reading real output is the fastest way to spot problems no check thinks of.

    DONE WHEN:
      - Rule Z-2 is one section with all four postal codes
      - The discount tiers table is whole, with its header row
      - No footer text or page numbers remain
      - Every TEK17 section in your two chapters has a reference and title
    """
    tek17 = []
    if TEK17_PDF.exists():
        tek17 = parse_tek17(TEK17_PDF, TEK17_VERSION, TEK17_CHAPTERS)
    else:
        print(f"Skipping TEK17: {TEK17_PDF} not found")
    pricing = parse_lyngpakke(LYNGPAKKE_DIR, LYNGPAKKE_VERSION)

    all_sections = tek17 + pricing
    check_sections(all_sections)
    write_jsonl(tek17, OUT_DIR / "tek17.jsonl")
    write_jsonl(pricing, OUT_DIR / "pricing.jsonl")

    print("\nSections per document:")
    for doc, count in sorted(Counter(s["doc"] for s in all_sections).items()):
        print(f"  {doc}: {count}")
    print(f"\nWrote {len(tek17)} TEK17 and {len(pricing)} pricing sections to {OUT_DIR}/")

    print("\nThree random sections to read:")
    for s in random.sample(all_sections, k=min(3, len(all_sections))):
        print(f"\n--- {s['reference']} | {s['title']}\n{s['text'][:400]}")


if __name__ == "__main__":
    main()