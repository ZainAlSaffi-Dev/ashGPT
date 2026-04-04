# ashGPT — Property Law Exam Assistant

A multimodal LangGraph agent for INFS4205 that assists law students with Australian Property Law analysis. The system separates legal rule extraction from chronological fact-extraction to produce grounded, structured study aids.

## Design Hypothesis

> *In common law property analysis, does a multi-node agent that separates chronological fact-extraction (rendered via Mermaid.js) from ratio decidendi rule-extraction produce more accurate causal analyses than a single-prompt summarisation baseline?*

## Architecture

The agent is a LangGraph state machine with conditional routing based on query intent:

```
User Query → Router → Retrieval → [Ratio Extractor | Chronology Generator | Both] → Synthesis
```

| Node | Purpose |
|------|---------|
| **Query Router** | Classifies intent as `ratio`, `chronology`, `summary`, or `general`; extracts week filters |
| **Retrieval Node** | Queries ChromaDB with MMR for text chunks and VLM-described lecture slides |
| **Ratio Extractor** | Isolates the *ratio decidendi* and produces a full IRAC analysis |
| **Chronology Generator** | Extracts timelines and outputs valid Mermaid.js flowcharts |
| **Synthesis Node** | Compiles upstream outputs into a grounded final answer with source citations |

### Routing Paths

| Intent | Path | When |
|--------|------|------|
| `ratio` | Router → Retrieval → Ratio Extractor → Synthesis | Student asks for a legal rule or IRAC |
| `chronology` | Router → Retrieval → Chronology Generator → Synthesis | Student asks for a timeline or flowchart |
| `summary` | Router → Retrieval → Ratio Extractor → Chronology → Synthesis | Student wants a full case summary |
| `general` | Router → Retrieval → Synthesis | Greetings, simple questions |

## Knowledge Base

The knowledge base is **strictly pre-indexed** (no runtime uploads). It supports two modalities:

- **Text**: PDF case law readings, tutorial sheets, and supplementary notes — chunked (1500 chars, 300 overlap) and embedded via ZeroEntropy `zembed-1`.
- **Image**: Lecture slide PNGs, described by Gemini VLM during indexing and embedded as searchable text with `image_path` metadata.

Retrieval uses **Maximal Marginal Relevance (MMR)** to balance relevance with source diversity, and supports metadata filtering by week and document type.

## Evaluation Results

Evaluated across 10 test queries spanning 6 weeks of content, using LLM-as-a-judge (groundedness scoring, 1-5 scale).

### Gemini 3.1 Pro (Final Configuration)

| Configuration | Avg Groundedness | Avg Latency | Source Diversity |
|---------------|-----------------|-------------|-----------------|
| **Full Agent** | **4.8 / 5.0** | 76.2s | 6.2 |
| Plain LLM Baseline | 2.0 / 5.0 | 24.6s | 0.0 |
| Ablation (no Ratio Extractor) | 5.0 / 5.0 | 23.2s | 5.8 |

### Key Findings

- The full agent achieves **140% higher groundedness** than the plain LLM baseline (4.8 vs 2.0).
- The ablation (skipping the Ratio Extractor) scores marginally higher on strict groundedness (5.0 vs 4.8), but lacks the structured IRAC analysis and explicit *ratio decidendi* extraction that make the full agent pedagogically valuable.
- Model upgrade from Gemini 2.5 Pro to 3.1 Pro improved agent groundedness from 4.5 to 4.8 and widened the baseline gap from 1.8 to 2.8.

## Setup

### Prerequisites

- Python 3.11+
- Conda (for environment management)
- API keys: `GOOGLE_API_KEY` (Gemini), `ZEMBED_API_KEY` (ZeroEntropy embeddings)

### Installation

```bash
# Create and activate conda environment
conda create -n genai python=3.11 pip -y
conda activate genai
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your API keys
```

### Data Preparation

Place your course materials in the `data/` directory:

```
data/
├── week_1/
│   ├── lecture/        # PNG slide images
│   ├── Readings (...).pdf
│   └── Tutorial (...).pdf
├── week_2/
│   └── ...
└── Supplementary Notes.pdf   # Optional root-level PDFs
```

### Build the Index

```bash
python -m src.indexing.build_index
```

### Run the Application

```bash
# Streamlit frontend
streamlit run app.py

# Or inspect outputs in terminal
python -m tests.inspect_outputs --query "What is adverse possession?" --week week_3
```

### Run Evaluation

```bash
python -m src.eval.run_evals --output-dir eval_results
```

### Run Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only (no API keys needed)
pytest tests/ -v -m "not integration"

# Fast integration tests only
pytest tests/ -v -m "integration and not slow"
```

## Project Structure

```
ashGPT/
├── app.py                         # Streamlit frontend
├── src/
│   ├── config.py                  # Centralised constants and model config
│   ├── embeddings.py              # ZeroEntropy embedding wrapper
│   ├── indexing/
│   │   └── build_index.py         # Multimodal ingestion pipeline
│   ├── agent/
│   │   ├── state.py               # LangGraph TypedDict state definition
│   │   ├── tools.py               # ChromaDB retrieval with MMR support
│   │   ├── nodes.py               # 5 cognitive nodes (router, retrieval, ratio, chronology, synthesis)
│   │   └── graph.py               # LangGraph workflow compilation
│   └── eval/
│       └── run_evals.py           # Evaluation, ablation, and plot generation
├── tests/
│   ├── conftest.py                # Shared fixtures, auto-skip for missing API keys
│   ├── test_phase1_indexing.py    # Config, embeddings, PDF extraction, ChromaDB integrity
│   ├── test_phase2_retrieval.py   # State, tools, metadata filtering
│   ├── test_phase3_nodes.py       # Node helpers, routing, reasoning output
│   ├── test_phase4_graph.py       # Graph compilation, conditional routing
│   ├── test_phase5_eval.py        # Eval framework, plotting, baselines
│   └── inspect_outputs.py        # Interactive output inspection
├── data/                          # Course materials (gitignored)
├── chroma_db/                     # Persisted vector store (gitignored)
├── eval_results/                  # Evaluation outputs (gitignored)
├── requirements.txt
├── pyproject.toml                 # Pytest configuration
└── .env.example                   # API key template
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph, LangChain |
| Vector Store | ChromaDB (local, persistent) |
| Embeddings | ZeroEntropy `zembed-1` (2560 dimensions) |
| LLM / VLM | Google Gemini 3.1 Pro |
| Retrieval | MMR with metadata filtering |
| Frontend | Streamlit + streamlit-mermaid |
| Testing | pytest (60+ tests across 5 phases) |
