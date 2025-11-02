from embed.encoder import Embedder
from index.store import Indexer


def test_embed_and_index(tmp_path):
    texts = ["This is a test.", "Another short document."]
    emb = Embedder()
    vecs = emb.encode(texts)
    assert vecs.shape[0] == 2

    idx = Indexer(
        faiss_path=str(tmp_path / "faiss.index"), sqlite_path=str(tmp_path / "meta.db")
    )
    idx.rebuild()
    docs = []
    for i, t in enumerate(texts):
        docs.append({"id": f"doc{i}", "text": t, "metadata": {"chunk_index": 0}})
    idx.add(vecs, docs)
    stats = idx.stats()
    assert stats.ntotal == 2

    # Query using first text
    qv = vecs[0]
    hits = idx.search(qv, top_k=2)
    assert len(hits) >= 1
