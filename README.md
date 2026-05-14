# ashGPT — Property Law Exam Assistant

A multimodal LangGraph agent for INFS4205 that assists law students with Australian Property Law analysis. The system separates legal rule extraction from chronological fact-extraction to produce grounded, structured study aids.

## Design Hypothesis

> *In common law property analysis, does a multi-node agent that separates chronological fact-extraction (rendered via Mermaid.js) from ratio decidendi rule-extraction produce more accurate causal analyses than a single-prompt summarisation baseline?*

## Architecture

The agent is a **LangGraph** state machine with **conditional routing** after retrieval. The router classifies intent; specialised nodes run only when needed; synthesis grounds the final answer in retrieved sources; a verification node fact-checks every cited case against the retrieved chunks before the answer leaves the graph.

```
User Query → Router → Retrieval → ┬→ Ratio Extractor ──┬→ Synthesis → Verification → END
                                  │                    │
                                  ├→ Chronology ───────┘
                                  │
                                  └→ Synthesis (general path only)
```

| Intent | Path |
|--------|------|
| `ratio` | Router → Retrieval → Ratio Extractor → Synthesis → Verification |
| `chronology` | Router → Retrieval → Chronology → Synthesis → Verification |
| `summary` | Router → Retrieval → Ratio Extractor → Chronology → Synthesis → Verification |
| `general` | Router → Retrieval → Synthesis → Verification |

| Node | Role |
|------|------|
| **Router** | Classifies `ratio` / `chronology` / `summary` / `general`; may suggest a `week` metadata filter |
| **Retrieval** | ChromaDB over text chunks and VLM-described slides; **MMR** with optional week/type filters, followed by a **cross-encoder reranker** (`ms-marco-MiniLM-L-6-v2`) that re-scores the MMR pool and keeps the top-*k* |
| **Ratio Extractor** | Isolates *ratio decidendi* and structured **IRAC** from retrieved material |
| **Chronology** | Builds a timeline and emits **Mermaid.js** flowchart syntax |
| **Synthesis** | Produces the student-facing answer with citations; treats upstream IRAC/chronology as *derived* and re-checks facts against primary sources |
| **Verification** | Extracts every cited case (italicised or plain `X v Y`) from the final answer; flags any citation that does not appear in the retrieved chunk text and asks the synthesis model to remove or hedge the unsupported claim while preserving every supported claim and any `\`\`\`mermaid` block verbatim. Gated by `USE_VERIFICATION` in `src/config.py` |

### Multi-provider LLM layer

Reasoning is **not** tied to a single vendor. `src/llm.py` dispatches by model name prefix (`gemini-` → Google GenAI, `gpt-` → OpenAI Responses API). Per-node assignments live in `src/config.py` (typical layout—edit there to experiment):

| Role | Model (as configured) |
|------|------------------------|
| Slide description (indexing VLM) | `gemini-2.5-pro` |
| Router | `gemini-3.1-flash-lite` |
| Ratio extractor | `gpt-5.3-chat-latest` |
| Chronology | `gemini-3-flash-preview` |
| Synthesis | `gpt-5.4-mini` |

### Evaluation & ablation (for quantitative comparison)

`src/eval/run_evals.py` compares **four configurations** on the same **22 cases** (six factual, five cross-modal, seven analytical, four conversational):

| Configuration | What it measures |
|---------------|------------------|
| **Full agent** | Complete LangGraph pipeline (router → retrieval+rerank → reasoning nodes → synthesis → verification) |
| **Baseline** | Plain LLM (`BASELINE_MODEL`), no retrieval, no graph |
| **Mega-prompt ablation** | Retrieval + a single consolidated LLM call replacing all router-driven reasoning nodes |
| **No-reranker ablation** | Full agent re-run with `USE_RERANKER=False`; isolates the contribution of the cross-encoder reranker |

**Query families (explicit coursework coverage):** each row in `EVAL_CASES` inside `run_evals.py` is tagged with exactly one of:

| Family | What we test |
|--------|----------------|
| `factual_retrieval` | Direct doctrine / tests / definitions from the indexed KB |
| `cross_modal_retrieval` | Questions phrased to pull **lecture slide (VLM) descriptions** as well as PDF text |
| `analytical_synthesis` | Compare, relate, summarise, or sequence ideas across multiple retrieved sources |
| `conversational_followup` | **Two-turn** scripts: agent and ablation run with `chat_history`; baseline sees **only the final turn** |

Outputs include `query_family`, `case_id`, and `case_rationale` per row in `eval_results.json`; `eval_summary.json` has `by_query_family` aggregates; `failure_analysis.md` opens with a **per-family** checklist and scores; `groundedness_by_query_family.png` plots the agent by family.

**Retrieval metrics** (LLM-as-judge, binary relevance per chunk in ranked order): **precision@K**, **MRR** (reciprocal rank of the first relevant chunk), **Hit@K** (any relevant in the top-K), **NDCG@K**. Summaries and `retrieval_ranking_metrics.png` compare the full agent vs ablations. `precision_with_vs_without_reranker.png` and `groundedness_four_way.png` chart the explicit reranker contribution and the four-way groundedness comparison. Optional **`--retrieval-pool-eval`** re-retrieves a larger pool (`EVAL_RETRIEVAL_POOL_K_*` in `config.py`), judges every chunk once, and adds **recall-vs-pool** = (relevant in production top-K) ÷ (relevant anywhere in pool) — a bounded proxy for recall, **not** full-corpus recall (document the limitation in your report).

**Automated metrics** (besides judges): heuristic **Mermaid** structural checks, **IRAC** keyword coverage (returns *not applicable* and is excluded from the average when intent is `chronology` or `general`, so the metric is not penalised for queries that legitimately do not produce IRAC structure), **latency**, **per-node latency** (agent, including verification), **token usage** (via `src/llm.py`), and on cross-modal cases **retrieval diagnostics** (text vs slide chunk counts).

**Verification telemetry.** During the eval, after every case the harness emits a one-line `[verification] case N/M fired=X rewrites=Y unsupported_total=Z` counter, and per-case `verification_report` is persisted to `eval_results.json` with the flagged citations and a `rewrites_applied` flag.

**Human spot-check log.** `eval_results/human_spot_check.md` is generated automatically: five cases are sampled deterministically (`seed=4205`) with blank human-rating fields. After filling, a small parser computes MAE and substantive-disagreement count against the LLM judge — a qualitative validation of the LLM-as-a-judge pipeline.

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

Retrieval is a two-stage dense pipeline:

1. **MMR** (`λ=0.5`, `fetch_k=20`) over ChromaDB dense embeddings produces a diverse candidate pool — over-fetches `RERANKER_FETCH_K_TEXT=16` text and `RERANKER_FETCH_K_SLIDES=8` slide candidates when the reranker is enabled.
2. **Cross-encoder reranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2`, loaded lazily and cached as a module-level singleton) re-scores every (query, chunk) pair and truncates to the production *k* (`k_text=8`, `k_slides=4`). The reranker is gated by `USE_RERANKER` in `src/config.py` and can be flipped per-run via the `use_reranker` state field — this is the hook the no-reranker ablation uses.

No BM25 / sparse retrieval is used; the pipeline is dense throughout.

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
# Edit .env: GOOGLE_API_KEY, ZEMBED_API_KEY, OPENAI_API_KEY as needed
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
│   ├── config.py                  # Paths, chunking, retrieval, per-node models,
│   │                              # USE_RERANKER, RERANKER_FETCH_K_*, USE_VERIFICATION
│   ├── llm.py                     # Unified multi-provider LLM dispatch + token tracking
│   ├── embeddings.py              # ZeroEntropy embedding wrapper
│   ├── indexing/
│   │   └── build_index.py         # Multimodal ingestion (PDF + slide images)
│   ├── agent/
│   │   ├── state.py               # LangGraph state (TypedDict); verification_report,
│   │   │                          # use_reranker override field
│   │   ├── chat_memory.py         # Trim/format history, retrieval query packing
│   │   ├── tools.py               # Chroma retrieval (MMR, filters) + reranker hook
│   │   ├── reranker.py            # Cross-encoder reranker (lazy, singleton)
│   │   ├── verification.py        # Case-citation extraction + unsupported-claim check
│   │   ├── nodes.py               # Router, retrieval, ratio, chronology, synthesis,
│   │   │                          # verification, mega-prompt ablation
│   │   └── graph.py               # Conditional graph + run_query()
│   └── eval/
│       └── run_evals.py           # Baseline, mega-prompt, no-reranker ablations,
│                                  # judges, plots, failure notes, spot-check log
├── tests/
├── data/                          # Course materials (gitignored content typical)
├── chroma_db/                     # Local vector store
├── eval_results/                  # Eval JSON + PNG plots, human_spot_check.md
├── requirements.txt
├── pyproject.toml
└── .env.example
```

## Generalisation

The hypothesis is stated in the language of common law analysis, but the underlying claim — *that separating structurally distinct outputs into specialised nodes improves grounding versus a single mega-prompt* — is not specific to property law. The same cognitive separation should apply to **medical case reasoning** (where a differential-diagnosis ranking, a clinical timeline, and a guideline-grounded recommendation are structurally different outputs that compete for attention inside one prompt), **financial regulatory analysis** (where a control-mapping table, a chronology of disclosure events, and an applied-rule analysis under a regulation each demand a different output shape), and **scientific paper synthesis** (where a methods comparison, an evidence chronology, and a hypothesis-driven discussion all benefit from being generated under different system prompts and then composed).

The pattern is simpler than the legal frame suggests: any domain in which structured outputs — chronology, taxonomy, rule application, mapping tables, sequence diagrams — compete for attention inside a single prompt is a candidate for cognitive separation. The IRAC + Mermaid split implemented here is one specific instance of a more general design: route by the *kind of structure* the user wants, generate each structure in a node optimised for it, and then synthesise. The retrieval modalities, the structural validators, and the per-node model assignments would change between domains; the multi-node + verification + ablation evaluation pattern would not.

## Tech Stack

| Layer | Technology |
|--------|------------|
| Orchestration | LangGraph, LangChain |
| Vector store | ChromaDB (local, persistent) |
| Embeddings | ZeroEntropy `zembed-1` |
| LLMs / VLM | Google Gemini, OpenAI (per `config.py`) |
| Retrieval | MMR + cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`), metadata filters (week, doc type) |
| Verification | Regex-based case-citation extraction + LLM rewrite of unsupported claims |
| Frontend | Streamlit, streamlit-mermaid |
| Evaluation | Two-stage groundedness judge (Gemini draft → OpenAI critique), single-stage relevancy & per-chunk precision (Gemini), structural checks, four-config ablation (agent / baseline / mega-prompt / no-reranker), matplotlib plots, human spot-check log |
