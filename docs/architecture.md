# Architecture

## System Overview

ashGPT is a multimodal LangGraph agent that processes property law queries through a conditional state machine. The architecture separates retrieval, legal rule extraction, and chronological analysis into distinct cognitive nodes, enabling ablation studies and independent evaluation of each component.

## Data Flow

```mermaid
graph LR
    subgraph ingestion ["Phase 1: Offline Indexing"]
        PDFs["PDF Readings\n& Tutorials"] -->|"PyMuPDF text extraction"| Chunker["Text Chunker\n1500 chars, 300 overlap"]
        Slides["PNG Lecture\nSlides"] -->|"Gemini VLM description"| SlideText["Slide Descriptions"]
        Chunker -->|"zembed-1 embedding"| ChromaDB["ChromaDB\nCollection"]
        SlideText -->|"zembed-1 embedding"| ChromaDB
    end
```

## Agent Pipeline

```mermaid
graph TD
    UserQuery["User Query"] --> Router

    subgraph pipeline ["LangGraph State Machine"]
        Router["Router Node\nClassify intent + extract week filter"]
        Router --> Retrieval["Retrieval Node\nMMR search: 8 text chunks + 4 slides"]

        Retrieval --> Decision{"intent?"}

        Decision -->|"ratio"| RatioExtractor["Ratio Extractor\nIRAC analysis + ratio decidendi"]
        Decision -->|"chronology"| Chronology["Chronology Generator\nMermaid.js flowchart"]
        Decision -->|"summary"| RatioExtractor
        Decision -->|"general"| Synthesis

        RatioExtractor -->|"if summary"| Chronology
        RatioExtractor -->|"if ratio"| Synthesis["Synthesis Node\nGrounded final answer with citations"]
        Chronology --> Synthesis
    end

    Synthesis --> FinalAnswer["Final Answer\n+ IRAC + Mermaid diagram"]
```

## State Schema

The `AgentState` TypedDict carries all data between nodes:

```mermaid
graph LR
    subgraph inputState ["Input"]
        Query["query: str"]
        Week["week_filter: str | None"]
    end

    subgraph routerState ["Router Output"]
        Intent["intent: ratio | chronology | summary | general"]
    end

    subgraph retrievalState ["Retrieval Output"]
        Texts["retrieved_texts: list"]
        SlidesDocs["retrieved_slides: list"]
    end

    subgraph ratioState ["Ratio Extractor Output"]
        Ratio["ratio_decidendi: str"]
        IRAC["irac_analysis: str"]
    end

    subgraph chronoState ["Chronology Output"]
        Mermaid["mermaid_diagram: str"]
        ChronoSummary["chronology_summary: str"]
    end

    subgraph synthState ["Synthesis Output"]
        Answer["final_answer: str"]
    end

    subgraph diagState ["Diagnostics"]
        Trace["node_trace: list"]
    end

    inputState --> routerState --> retrievalState --> ratioState --> chronoState --> synthState
    synthState --> diagState
```

## Retrieval Strategy

```mermaid
graph TD
    QueryEmbed["Query embedded via\nzembed-1 (query mode)"] --> Candidates["Fetch top-20 candidates\nfrom ChromaDB"]
    Candidates --> MMR["MMR Selection\nlambda=0.5"]
    MMR --> TextChunks["8 text chunks\nreadings, tutorials, supplementary"]
    MMR --> SlideChunks["4 slide descriptions\nwith image_path metadata"]
    TextChunks --> FilterMeta{"Week filter?"}
    SlideChunks --> FilterMeta
    FilterMeta -->|"yes"| Filtered["Results restricted\nto specified week"]
    FilterMeta -->|"no"| AllWeeks["Results from\nall weeks"]
```

## Evaluation Framework

```mermaid
graph TD
    subgraph configs ["Three Configurations"]
        Agent["Full Agent\nAll 5 nodes"]
        Baseline["Plain LLM Baseline\nNo retrieval, no graph"]
        Ablation["Ablation\nRetrieval + Synthesis only"]
    end

    TestQueries["10 Test Queries\n6 weeks, 4 intent types"] --> Agent
    TestQueries --> Baseline
    TestQueries --> Ablation

    Agent --> Judge["LLM-as-a-Judge\nGroundedness 1-5"]
    Baseline --> Judge
    Ablation --> Judge

    Judge --> Metrics["Metrics Export\nJSON + PNG plots"]
    Agent -->|"per-node timing"| Metrics
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Single ChromaDB collection | Simpler retrieval with metadata filtering vs separate collections per week |
| MMR over similarity search | Balances relevance with source diversity; configurable for ablation |
| Separate VLM_MODEL and REASONING_MODEL | Allows independent model swapping for ablation studies |
| PRIMARY vs DERIVED evidence split in synthesis | Prevents the synthesis node from treating IRAC inferences as ground-truth facts |
| Deterministic document IDs | Safe re-indexing without duplicates |
| node_trace in state | Tracks which nodes fired per query for ablation comparison |
