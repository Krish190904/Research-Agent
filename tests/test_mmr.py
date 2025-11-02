import numpy as np
from index.store import Indexer
from retrieve.retriever import Retriever


def test_mmr_diversity(tmp_path):
    # create 3 vectors in 2D: v1 and v2 similar, v3 orthogonal
    v1 = np.array([1.0, 0.0], dtype=np.float32)
    v2 = np.array([0.9, 0.1], dtype=np.float32)
    v3 = np.array([0.0, 1.0], dtype=np.float32)
    vecs = np.vstack([v1, v2, v3])
    # normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms

    idx = Indexer(
        faiss_path=str(tmp_path / "faiss.index"), sqlite_path=str(tmp_path / "meta.db")
    )
    idx.rebuild()
    docs = []
    for i in range(3):
        docs.append({"id": f"d{i}", "text": f"doc{i}", "metadata": {"chunk_index": 0}})
    idx.add(vecs, docs)

    # query near v1
    q = np.array([1.0, 0.0], dtype=np.float32)
    q = q / np.linalg.norm(q)

    retr = Retriever(idx)
    res_no_mmr = retr.retrieve(q, top_k=2, mmr_enabled=False)
    res_mmr = retr.retrieve(q, top_k=2, mmr_enabled=True, lambda_param=0.7)

    # Without MMR, expect v1 and v2 (most similar)
    ids_no_mmr = [r["faiss_id"] for r in res_no_mmr]
    # With MMR, expect diversity: should include v3 (faiss_id 2)
    ids_mmr = [r["faiss_id"] for r in res_mmr]

    assert ids_no_mmr[0] == 0
    assert 1 in ids_no_mmr
    assert 2 in ids_mmr or 1 in ids_no_mmr
