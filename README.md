# ashGPT — Property Law Exam Assistant

A multimodal LangGraph agent for INFS4205 that assists law students with Australian Property Law analysis. The system separates legal rule extraction from chronological fact-extraction to produce grounded, structured study aids.

## Design Hypothesis

> *In common law property analysis, does a multi-node agent that separates chronological fact-extraction (rendered via Mermaid.js) from ratio decidendi rule-extraction produce more accurate causal analyses than a single-prompt summarisation baseline?*

## Architecture

The agent is a **LangGraph** state machine with **conditional routing** after retrieval. The router classifies intent; specialised nodes run only when needed; synthesis always grounds the final answer in retrieved sources.

```
User Query → Router → Retrieval → ┬→ Ratio Extractor ──┬→ Synthesis → END
                                  │                    │
                                  ├→ Chronology ───────┘
                                  │
                                  └→ Synthesis (general path only)
```

| Intent | Path |
|--------|------|
| `ratio` | Router → Retrieval → Ratio Extractor → Synthesis |
| `chronology` | Router → Retrieval → Chronology → Synthesis |
| `summary` | Router → Retrieval → Ratio Extractor → Chronology → Synthesis |
| `general` | Router → Retrieval → Synthesis |

| Node | Role |
|------|------|
| **Router** | Classifies `ratio` / `chronology` / `summary` / `general`; may suggest a `week` metadata filter |
| **Retrieval** | ChromaDB over text chunks and VLM-described slides; **MMR** + optional week/type filters |
| **Ratio Extractor** | Isolates *ratio decidendi* and structured **IRAC** from retrieved material |
| **Chronology** | Builds a timeline and emits **Mermaid.js** flowchart syntax |
| **Synthesis** | Produces the student-facing answer with citations; treats upstream IRAC/chronology as *derived* and re-checks facts against primary sources |

### Multi-provider LLM layer

Reasoning is **not** tied to a single vendor. `src/llm.py` dispatches by model name prefix (`gemini-` → Google GenAI, `gpt-` → OpenAI Responses API, `claude-` → Anthropic). Per-node assignments live in `src/config.py` (typical layout—edit there to experiment):

| Role | Model (as configured) |
|------|------------------------|
| Slide description (indexing VLM) | `gemini-2.5-pro` |
| Router | `gemini-3.1-flash-lite-preview` |
| Ratio extractor | `gpt-5.3-chat-latest` |
| Chronology | `gemini-3-flash-preview` |
| Synthesis | `gpt-5.4-mini` |

### Evaluation & ablation (for quantitative comparison)

`src/eval/run_evals.py` compares three configurations on the same queries:

| Configuration | What it measures |
|---------------|------------------|
| **Full agent** | Complete LangGraph pipeline |
| **Baseline** | Plain LLM (`BASELINE_MODEL`), no retrieval, no graph |
| **Ablation** | Retrieval + synthesis only; **no** router-driven ratio or chronology nodes (`intent` fixed to `general`) |

**Metrics** include LLM-as-a-judge **groundedness** (1–5) and **answer relevancy**, **context precision@K**, heuristic **Mermaid validity** and **IRAC compliance**, **latency**, **per-node latency** (agent), and **token usage**. Judging uses a **two-stage** flow (draft judge and critique judge on different providers/models where configured) to reduce self-bias.

After a run, see `eval_results/eval_summary.json` and the generated plots. Numbers change with models, prompts, and data—re-run the suite for your report.

## Knowledge Base

The knowledge base is **strictly pre-indexed** (no runtime uploads). Two modalities:

- **Text**: PDF readings, tutorials, and notes — chunked (1500 characters, 300 overlap) and embedded with ZeroEntropy **`zembed-1`**.
- **Image**: Lecture slides (**JPEG / PNG**) — described at index time by the configured **VLM**; descriptions are embedded as text with metadata (e.g. source, week).

Retrieval uses **maximal marginal relevance (MMR)** with tunable λ and fetch-*k* in `src/config.py`.

## Setup

### Prerequisites

- Python 3.11+
- Conda (recommended)
- API keys as required by your `src/config.py` model choices (at minimum **Google** for Gemini, **ZeroEntropy** for embeddings; **OpenAI** if any `gpt-` model is used)

### Installation

```bash
conda create -n genai python=3.11 pip -y
conda activate genai
pip install -r requirements.txt

cp .env.example .env
# Edit .env: GOOGLE_API_KEY, ZEMBED_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY as needed
```

### Data layout

```
data/
├── week_1/
│   ├── lecture/          # Slide images (.jpg / .jpeg / .png)
│   ├── readings/         # PDFs (optional subfolder name may vary)
│   └── tutorial/         # PDFs
├── week_2/
│   └── ...
└── (optional root-level PDFs, e.g. supplementary notes)
```

Adjust paths to match your course folder naming; the indexer expects a `week_*` structure with lecture images and PDFs discoverable under each week.

### Build the index

```bash
python -m src.indexing.build_index
```

### Run the app

```bash
streamlit run app.py
```

The UI keeps **chat history in the session** for display, but each question is processed as a **single-turn** invocation of `run_query`—prior turns are not passed into the graph unless you extend the code.

### Evaluation

```bash
python -m src.eval.run_evals --output-dir eval_results
```

### Tests

```bash
pytest tests/ -v
pytest tests/ -v -m "not integration"    # no API keys
```

## Project Structure

```
ashGPT/
├── app.py                         # Streamlit frontend
├── src/
│   ├── config.py                  # Paths, chunking, retrieval, per-node models
│   ├── llm.py                     # Unified multi-provider LLM dispatch + token tracking
│   ├── embeddings.py              # ZeroEntropy embedding wrapper
│   ├── indexing/
│   │   └── build_index.py         # Multimodal ingestion (PDF + slide images)
│   ├── agent/
│   │   ├── state.py               # LangGraph state (TypedDict)
│   │   ├── tools.py               # Chroma retrieval (MMR, filters)
│   │   ├── nodes.py               # Router, retrieval, ratio, chronology, synthesis
│   │   └── graph.py               # Conditional graph + run_query()
│   └── eval/
│       └── run_evals.py           # Baseline, ablation, judges, plots, failure notes
├── tests/
├── data/                          # Course materials (gitignored content typical)
├── chroma_db/                     # Local vector store
├── eval_results/                  # Eval JSON + PNG plots (after run_evals)
├── requirements.txt
├── pyproject.toml
└── .env.example
```

## Tech Stack

| Layer | Technology |
|--------|------------|
| Orchestration | LangGraph, LangChain |
| Vector store | ChromaDB (local, persistent) |
| Embeddings | ZeroEntropy `zembed-1` |
| LLMs / VLM | Google Gemini, OpenAI, Anthropic (per `config.py`) |
| Retrieval | MMR, metadata filters (week, doc type) |
| Frontend | Streamlit, streamlit-mermaid |
| Evaluation | LLM judges, structural checks, matplotlib plots |
