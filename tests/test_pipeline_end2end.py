from pathlib import Path

from pipeline import Pipeline
from reasoner.reasoner import Reasoner
from export import Exporter, Report


def test_pipeline_end_to_end(tmp_path):
    # copy sample files to temp folder
    src = Path("sample_data")
    dst = tmp_path / "data"
    dst.mkdir()
    for p in src.iterdir():
        dst.joinpath(p.name).write_text(p.read_text())

    from index.store import Indexer

    idx = Indexer(
        faiss_path=str(tmp_path / "faiss.index"), sqlite_path=str(tmp_path / "meta.db")
    )
    pl = Pipeline(indexer=idx)
    n = pl.run_folder(dst, recursive=False)
    assert n > 0

    # query via reasoner
    idx = pl.indexer
    r = Reasoner(indexer=idx)
    out = r.answer("sample", top_k=2)
    assert "synthesis" in out and out["synthesis"]

    # export
    rep = Report(
        title="E2E Report", synthesis=out["synthesis"], traces=out.get("traces", [])
    )
    ex = Exporter()
    md = ex.export_markdown(rep, tmp_path / "report.md")
    pdf = ex.export_pdf(rep, tmp_path / "report.pdf")
    assert md.exists() and pdf.exists()
