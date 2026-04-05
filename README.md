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

**Query families (explicit coursework coverage):** each row in `EVAL_CASES` inside `run_evals.py` is tagged with exactly one of:

| Family | What we test |
|--------|----------------|
| `factual_retrieval` | Direct doctrine / tests / definitions from the indexed KB |
| `cross_modal_retrieval` | Questions phrased to pull **lecture slide (VLM) descriptions** as well as PDF text |
| `analytical_synthesis` | Compare, relate, summarise, or sequence ideas across multiple retrieved sources |
| `conversational_followup` | **Two-turn** scripts: agent and ablation run with `chat_history`; baseline sees **only the final turn** |

Outputs include `query_family`, `case_id`, and `case_rationale` per row in `eval_results.json`; `eval_summary.json` has `by_query_family` aggregates; `failure_analysis.md` opens with a **per-family** checklist and scores; `groundedness_by_query_family.png` plots the agent by family.

**Retrieval metrics** (LLM-as-judge, binary relevance per chunk in ranked order): **precision@K**, **MRR** (reciprocal rank of the first relevant chunk), **Hit@K** (any relevant in the top-K), **NDCG@K**. Summaries and `retrieval_ranking_metrics.png` compare the full agent vs ablation. Optional **`--retrieval-pool-eval`** re-retrieves a larger pool (`EVAL_RETRIEVAL_POOL_K_*` in `config.py`), judges every chunk once, and adds **recall-vs-pool** = (relevant in production top-K) ÷ (relevant anywhere in pool) — a bounded proxy for recall, **not** full-corpus recall (document the limitation in your report).

**Automated metrics** (besides judges): heuristic **Mermaid** structural checks, **IRAC** keyword coverage, **latency**, **per-node latency** (agent), **token usage** (via `src/llm.py`), and on cross-modal cases **retrieval diagnostics** (text vs slide chunk counts).

After a run, see `eval_results/eval_summary.json` and the generated plots. Numbers change with models, prompts, and data—re-run the suite for your report.

### LLM-as-a-judge framework

All judge calls go through `llm_call()` in `src/llm.py` at **temperature 0**. Each judge is instructed to return **only a JSON object**; the harness strips optional Markdown code fences around the payload and parses the result (failed parses are logged and fall back to safe defaults).

Judge models are set in `src/config.py`:

| Variable | Model (current default) | Provider |
|----------|------------------------|----------|
| `JUDGE_DRAFT_MODEL` | `gemini-3-flash-preview` | Google |
| `JUDGE_CRITIQUE_MODEL` | `gpt-5.4` | OpenAI |

**Groundedness (1–5)** — **two-stage**, cross-provider:

1. **Draft (Stage 1)** — `JUDGE_DRAFT_MODEL` receives the question, the **source material** shown to the system under test, and the answer. It scores how well **factual** claims (cases, dates, statutes, holdings) are supported by those sources; **paraphrase and plain-English explanation** are allowed. It returns `{"score": int, "reasoning": str}`.
2. **Critique (Stage 2)** — `JUDGE_CRITIQUE_MODEL` receives the same question, sources, and answer, plus the draft score and reasoning. It acts as a **senior reviewer**: agree with the draft or **override** the score if the draft was too lenient or too harsh, using the same rubric. It returns `{"score": int, "reasoning": str, "agreed_with_draft": bool}`.

The **final groundedness score** recorded in results is the critique model’s `score` (with fallback to the draft score if parsing fails). Stored fields include `draft_score`, `draft_reasoning`, and `agreed_with_draft` for analysis.

**Rationale:** Using **Gemini for the first pass and OpenAI for the second** reduces **same-model self-evaluation bias** (e.g. a system built mainly on OpenAI being scored only by OpenAI).

**Answer relevancy (1–5)** — **single-stage**: one call using **`JUDGE_DRAFT_MODEL`** (question + answer only). Returns `{"score": int, "reasoning": str}`.

**Context precision @K** — **single-stage per chunk**: for each retrieved text/slide chunk, one call using **`JUDGE_DRAFT_MODEL`** asks whether the chunk **helps** answer the question (`{"relevant": true|false}`). Precision is the fraction of chunks marked relevant.

**Baseline groundedness** in the suite is judged against the **same retrieved context bundle as the full agent** for that query (the baseline model does not see that context at generation time); this measures alignment with the course KB slice, not “what the baseline had access to.”

**Non-LLM structural scores** — **Mermaid validity** and **IRAC compliance** use rule-based checks over the dedicated node outputs or, when empty, over the **final answer** text (see `run_evals.py`).

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

#### Conversational memory (multi-turn)

The UI stores messages in `st.session_state`. On each send, **prior** turns are passed to `run_query(..., chat_history=...)` as `AgentState.chat_history`; the **current** message is always `query` alone.

- **Router** sees the transcript so follow-ups like “explain that ratio” or “draw the timeline for that case” resolve correctly.
- **Retrieval** embeds a **packed query**: the new student message plus a **short excerpt** of the last tutor reply (see `CHAT_HISTORY_MAX_ASSISTANT_TAIL_CHARS` in `config.py`), so vector search stays on-topic without dumping the whole prior answer into the embedding.
- **Ratio, chronology, and synthesis** receive the formatted transcript so answers stay coherent across turns; **grounding rules** still require facts to come from retrieved sources.

Limits: `CHAT_HISTORY_MAX_MESSAGES` and `CHAT_HISTORY_MAX_CHARS_PER_MESSAGE` cap cost and context size. **Clear conversation** in the sidebar wipes session memory. Memory is **per browser tab**, not persisted to disk.

The quantitative suite in `src/eval/run_evals.py` remains **single-turn** (no `chat_history`) so baseline and ablation comparisons stay fixed—report multi-turn behaviour as a **product / qualitative** capability unless you add dedicated multi-turn eval scenarios.

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
│   │   ├── chat_memory.py         # Trim/format history, retrieval query packing
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
| Evaluation | Two-stage groundedness judge (Gemini draft → OpenAI critique), single-stage relevancy & per-chunk precision (Gemini), structural checks, matplotlib plots |
