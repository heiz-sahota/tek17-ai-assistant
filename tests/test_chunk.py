"""Unit tests for ingest/chunk.py. Run: pytest -q"""
import ingest.chunk as chunk

SECTION = {
    "source": "pricing", "doc": "02_Weight_and_Size_Price_Table",
    "reference": "02_Weight_and_Size_Price_Table / 1", "title": "1. Base prices by weight band",
    "version": "2026.1", "role": "pricing",
    "text": "Base prices are in NOK.\nWeight band | Base price\n0–1 kg | 89 NOK\nOver 1–5 kg | 129 NOK",
}


def test_every_chunk_has_a_reference_and_header():
    chunks = chunk.chunk_recursive([SECTION], 500)
    assert chunks
    for c in chunks:
        assert c.reference == SECTION["reference"]
        assert c.content.startswith(f"Reference: {SECTION['reference']}")


def test_tables_are_not_split():
    chunks = chunk.chunk_recursive([SECTION], 500)
    with_table = [c for c in chunks if "89 NOK" in c.content]
    assert len(with_table) == 1
    assert "Weight band | Base price" in with_table[0].content   # header kept with its rows


def test_big_table_repeats_header():
    table = "Band | Price\n" + "\n".join(f"{i} kg | {i} NOK" for i in range(200))
    pieces = chunk.split_big_table(table, 50)
    assert len(pieces) > 1
    assert all(p.startswith("Band | Price\n") for p in pieces)


def test_ids_are_stable_and_unique():
    first = [c.id for c in chunk.chunk_recursive([SECTION], 500)]
    second = [c.id for c in chunk.chunk_recursive([SECTION], 500)]
    assert first == second
    assert len(set(first)) == len(first)


def test_fixed_chunks_cover_the_whole_document():
    long = dict(SECTION, text="word " * 300)
    chunks = chunk.chunk_fixed([long], 100)
    assert len(chunks) >= 3
    assert all(c.reference == SECTION["reference"] for c in chunks)