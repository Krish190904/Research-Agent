import numpy as np
from index.store import Indexer


def test_embedding_blob_storage(tmp_path):
    # Small synthetic embedding
    vec = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    idx = Indexer(
        faiss_path=str(tmp_path / "faiss.index"), sqlite_path=str(tmp_path / "meta.db")
    )
    idx.rebuild()
    docs = [{"id": "d0", "text": "t", "metadata": {"chunk_index": 0}}]
    idx.add(vec, docs)

    # fetch embedding via API
    rows = idx.fetch_embeddings([0])
    assert rows[0] is not None
    arr = rows[0]
    assert arr.dtype == np.float32
    assert arr.shape[0] == 4
