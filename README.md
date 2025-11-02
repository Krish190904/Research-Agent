# Deep Researcher Agent

This project provides a local-first research agent: document ingestion, local embeddings, FAISS indexing, MMR reranking, multi-step reasoning traces, and export to Markdown/PDF.

Quick highlights
- Local embeddings (sentence-transformers) with TF-IDF fallback when heavy deps are missing
- FAISS vector store + SQLite metadata and binary embedding storage
- MMR reranking with tunable lambda and candidate multiplier
- Streamlit UI and CLI for ingestion, indexing, querying, and export

Requirements
- Python 3.10+
- See `requirements.txt` for full list. `sentence-transformers` and `torch` are optional but recommended for high-quality embeddings.

Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run tests

```bash
pytest -q
```

CLI examples

Ingest folder and index (ingest -> embed -> index):

```bash
python cli.py ingest --folder ./sample_data --recursive
```

Query with MMR

```bash
python cli.py query --q "What topics are covered?" --topk 5 --mmr --mmr-lambda 0.7 --candidate-multiplier 5
```

Streamlit UI

```bash
streamlit run ui/app.py
```

MMR tuning guidance
- `mmr_lambda` in [0,1]: higher = prefer relevance, lower = prefer diversity
- `candidate_multiplier`: number of candidates considered = top_k * multiplier (higher gives more diversity options but costs time)

Storage format
- Embeddings are stored in SQLite using binary `LargeBinary` columns as raw float32 bytes. This is efficient for local setups and enables MMR reranking without re-embedding.

Docker

```bash
docker-compose up --build
```

Notes
- For production, consider migrating the SQLite schema (Alembic) if you change storage formats.
- For large corpora, use on-disk vector stores, sharding, or a dedicated vector DB.
# Deep Researcher Agent

A production-quality, fully local document research and reasoning system. This agent ingests documents (PDF, Markdown, TXT, HTML), builds local embeddings and vector indices, performs multi-step reasoning with explainable traces, and exports research reports to Markdown and PDF.

## Features

- **Local-First Architecture**: All embeddings, indexing, and retrieval run locally. No cloud APIs required.
- **Multi-Format Ingestion**: PDF, TXT, Markdown, HTML with metadata extraction
- **Dense Vector Search**: FAISS-based local indexing with sentence-transformers embeddings
- **Multi-Step Reasoning**: Query decomposition, sub-question retrieval, evidence synthesis
- **Explainable AI**: Full reasoning traces showing sub-tasks and evidence sources
- **Interactive Refinement**: Streamlit UI for follow-up queries and context preservation
- **Structured Export**: Markdown and PDF reports with citations
- **Production Ready**: Unit tests, Docker support, performance metrics

## Hardware Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 2GB for models + your corpus size

**Recommended:**
- CPU: 8+ cores
- RAM: 16GB+
- GPU: Optional (speeds up embedding generation)

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Krish23101996/Agent.git
cd Agent

# Install dependencies
pip install -r requirements.txt

# Download models (happens automatically on first run, ~500MB)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"
```

### Basic Usage

```bash
# 1. Ingest documents
python cli.py ingest --folder ./sample_data --recursive

# 2. Build index
python cli.py index --rebuild

# 3. Query
python cli.py query --q "What are the main topics covered in these documents?" --topk 10

# 4. Export results
python cli.py export --session last --format pdf
```

### Interactive UI

```bash
streamlit run ui/app.py
```

Then open http://localhost:8501 in your browser.

## Docker Usage

```bash
# Build and run
docker-compose up --build

# Access UI at http://localhost:8501
```

## Configuration

Edit `config.yml` to customize:

- Embedding model (default: `all-mpnet-base-v2`)
- Chunk size and overlap
- Retrieval parameters (top_k, similarity threshold)
- LLM settings for synthesis

## Architecture

```
Documents → Chunking → Embedding → FAISS Index
                                        ↓
User Query → Decomposition → Retrieval → Synthesis → Report
                                ↑            ↓
                            Sub-queries   Citations
```

## Project Structure

- `ingest/` - Document loaders and text extraction
- `embed/` - Local embedding generation
- `index/` - FAISS vector index management
- `retrieve/` - Dense retrieval with metadata filtering
- `reasoner/` - Multi-step reasoning pipeline
- `export/` - Markdown and PDF exporters
- `ui/` - Streamlit interactive interface
- `tests/` - Unit and integration tests

## Performance

On sample corpus (10 documents, ~100 pages):
- Indexing: ~50 docs/sec
- Query latency: ~500ms
- Embedding: ~100 chunks/sec (CPU)

## Advanced Features

### Custom LLM Integration

The agent uses a modular LLM interface. By default, it uses extractive summarization. To integrate your own local LLM:

```python
# In config.yml
llm:
  type: "local"  # or "ollama", "llamacpp", "transformers"
  model: "mistral-7b-instruct"
  max_tokens: 512
```

See `reasoner/synthesizer.py` for the LLM adapter interface.

### Scaling to Large Corpora

For >100K documents:
- Enable FAISS IVF indexing in config
- Use on-disk index mode
- Implement document sharding

## Development

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=. tests/

# Type checking
mypy .
```

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Acknowledgments

Built with:
- [sentence-transformers](https://www.sbert.net/) for local embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for vector indexing
- [Streamlit](https://streamlit.io/) for the interactive UI
- [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF processing

---

**Author**: Krish23101996  
**Created**: 2025-11-01

## Deploy locally (quickstart)

If you just want to run the app locally without downloading large ML models, use the lightweight image and the Streamlit UI:

1. Build the lightweight image (uses `requirements-lite.txt`):

```bash
docker build -f Dockerfile.lite -t agent:lite .
docker run -d --rm -p 8502:8501 --name agent_lite agent:lite \
  streamlit run ui/app.py --server.port 8501
```

Open http://localhost:8502 in your browser.

2. For full functionality (sentence-transformers), either run locally after installing `requirements.txt` or build the full Dockerfile (may download ~500MB models on first run):

```bash
docker build -f Dockerfile -t agent:full .
docker run -p 8501:8501 agent:full
```

3. CI and Migrations

- CI: A GitHub Actions workflow is provided at `.github/workflows/ci.yml` to run tests and build the lightweight image.
- Alembic scaffolding is included under `alembic/`. Use `alembic upgrade head` to apply the initial migration to the configured SQLite DB.
