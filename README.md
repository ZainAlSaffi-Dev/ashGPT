# ashGPT — Property Law Exam Assistant

A multimodal LangGraph agent for INFS4205 that assists law students with Property Law analysis.

## Design Hypothesis

> *In common law property analysis, does a multi-node agent that separates chronological fact-extraction (rendered via Mermaid.js) from ratio decidendi rule-extraction produce more accurate causal analyses than a single-prompt summarisation baseline?*

## Architecture

The agent is built as a LangGraph state machine with distinct cognitive nodes:

1. **Query Router** — classifies the user's intent (case summary, rule extraction, or chronological flowchart).
2. **Retrieval Node** — queries a pre-indexed ChromaDB knowledge base across text and image modalities.
3. **Ratio Extractor** — isolates the *ratio decidendi* from retrieved case law.
4. **Chronology Generator** — extracts timelines and outputs valid Mermaid.js syntax.
5. **Synthesis Node** — compiles outputs into a final IRAC analysis.

## Knowledge Base

The knowledge base is **strictly pre-indexed** (no runtime uploads). It supports two modalities:

- **Text**: PDF case law readings and tutorial sheets, chunked and embedded via `text-embedding-3-large`.
- **Image**: JPEG lecture slides, described by a VLM (Gemini) and embedded as searchable text with `image_path` metadata.

## Setup

```bash
# Create and activate conda environment
conda activate genai
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your GOOGLE_API_KEY and OPENAI_API_KEY

# Populate the data directory
# Place materials in: data/week_X/lecture/, data/week_X/readings/, data/week_X/tutorial/

# Build the index
python -m src.indexing.build_index
```

## Project Structure

```
ashGPT/
├── src/
│   ├── config.py              # Centralised constants
│   ├── indexing/
│   │   └── build_index.py     # Multimodal ingestion pipeline
│   ├── agent/                 # LangGraph state, tools, nodes, graph
│   └── eval/                  # Evaluation and ablation scripts
├── data/                      # Weekly lecture materials (gitignored)
├── chroma_db/                 # Persisted vector store (gitignored)
├── requirements.txt
└── .env.example
```

## Tech Stack

- **Orchestration**: LangGraph, LangChain
- **Vector Store**: ChromaDB (local)
- **Embeddings**: OpenAI `text-embedding-3-large`
- **LLM / VLM**: Google Gemini
- **Frontend**: Streamlit
