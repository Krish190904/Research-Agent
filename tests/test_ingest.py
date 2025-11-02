from ingest.loader import Ingestor


def test_ingest_md_and_txt(tmp_path):
    md = tmp_path / "sample.md"
    txt = tmp_path / "sample.txt"
    md.write_text("---\nauthor: tester\n---\n# Title\n\nThis is a test document.")
    txt.write_text("Plain text content with email test@example.com")

    ing = Ingestor()
    chunks = ing.ingest_folder(tmp_path, recursive=False)
    assert any("sample.md" in c.id for c in chunks)
    assert any("sample.txt" in c.id for c in chunks)
    # PII detection should flag email
    txt_chunks = [c for c in chunks if "sample.txt" in c.id]
    assert txt_chunks and txt_chunks[0].metadata.get("pii_warnings")
