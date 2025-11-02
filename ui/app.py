"""Streamlit UI for interacting with the Deep Researcher Agent."""

from __future__ import annotations

import streamlit as st
from pathlib import Path
from index.store import Indexer
from reasoner.reasoner import Reasoner
from export import Exporter, Report
from pipeline import Pipeline


st.set_page_config(page_title="Deep Researcher Agent")

st.title("Deep Researcher Agent — Local Research Tool")

uploaded = st.file_uploader(
    "Upload documents (PDF/MD/TXT/HTML)", accept_multiple_files=True
)

if st.button("Ingest uploaded"):
    pdir = Path("sample_data")
    pdir.mkdir(parents=True, exist_ok=True)
    for f in uploaded:
        p = pdir / f.name
        with open(p, "wb") as out:
            out.write(f.getbuffer())
    pl = Pipeline()
    n = pl.run_folder(pdir, recursive=False)
    st.success(f"Ingested and indexed {n} chunks")

if st.button("Build index"):
    # Rebuild (empty) index and show stats
    idx = Indexer()
    with st.spinner("Rebuilding index..."):
        idx.rebuild()
    st.success("Index created (empty). You can ingest documents to add vectors.")

st.sidebar.title("Index")
idx = Indexer()
idx.load()
stats = idx.stats()
st.sidebar.write(f"Vectors in index: {stats.ntotal}")

q = st.text_input("Query")
topk = st.number_input("Top-k", min_value=1, max_value=50, value=5)
mmr_enabled = st.checkbox("Enable MMR reranking", value=True)
mmr_lambda = st.slider(
    "MMR lambda (relevance vs diversity)", min_value=0.0, max_value=1.0, value=0.7
)
candidate_mult = st.number_input(
    "Candidate multiplier", min_value=1, max_value=20, value=5
)

if st.button("Ask") and q:
    idx = Indexer()
    idx.load()
    reasoner = Reasoner(indexer=idx)
    out = reasoner.answer(
        q,
        top_k=topk,
        mmr_enabled=mmr_enabled,
        mmr_lambda=mmr_lambda,
        candidate_multiplier=candidate_mult,
    )
    st.subheader("Synthesis")
    st.text(out["synthesis"])
    st.subheader("Traces")
    for t in out["traces"]:
        st.markdown(f"**Subquery:** {t['subquery']}")
        for h in t.get("hits", []):
            src = h.get("meta", {}).get("source", "unknown")
            st.markdown(f"- {src} — {h.get('score'):.3f}\n  - {h.get('text')[:300]}...")

if st.button("Export PDF demo"):
    exp = Exporter()
    rep = Report(title="Demo Report", synthesis="This is a demo synthesis.", traces=[])
    exp.export_pdf(rep, "output/report.pdf")
    st.success("Exported output/report.pdf")

if st.button("Build index from sample_data"):
    pdir = Path("sample_data")
    pl = Pipeline()
    with st.spinner("Ingesting and indexing..."):
        n = pl.run_folder(pdir, recursive=False)
    st.success(f"Indexed {n} chunks")
    idx = Indexer()
    idx.load()
    st.sidebar.write(f"Vectors in index: {idx.stats().ntotal}")
