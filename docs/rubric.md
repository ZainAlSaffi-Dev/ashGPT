# INFS4205 A3 Marking Rubric (Target: HD / 4.0 per category)

**1. Problem Framing & Innovation (4 Marks)**
* **4.0 Requirement:** A clear, compelling design hypothesis is articulated. The innovation is meaningful, well-motivated, and clearly distinct from the teaching demo. 
* *Cursor Directive:* Ensure the LangGraph architecture explicitly tests the separation of rule-extraction (text) vs chronological flow (Mermaid.js/image).

**2. Knowledge Base & Retrieval Design (4 Marks)**
* **4.0 Requirement:** Knowledge base is genuinely personalised. At least two modalities are integrated meaningfully. Retrieval/indexing choices are well justified and compared.
* *Cursor Directive:* Ensure ChromaDB ingestion handles Text (case law) and Images (PDF lecture slides via VLM).

**3. Agent Framework & Tool Orchestration (4 Marks)**
* **4.0 Requirement:** Agent workflow is well designed and clearly useful. Tool usage, routing, memory, or state handling are thoughtful and task-appropriate. 
* *Cursor Directive:* Do not use simple linear RAG. Ensure the graph contains a Router, an IRAC Ratio Extractor Node, and a Chronology Node.

**4. Quantitative Evaluation & Ablation (4 Marks)**
* **4.0 Requirement:** Rigorous evaluation against baselines. Must compare plain LLM/VLM vs final agent system, and at least one ablation on the final design.
* *Cursor Directive:* Build testing scripts in `src/eval/` that measure Groundedness (accuracy of IRAC) and Token/Latency efficiency.

**5. Report, Code & Reproducibility (4 Marks)**
* **4.0 Requirement:** Code and documentation are reproducible. High structural quality.
* *Cursor Directive:* Maintain a clean `README.md`, strict `requirements.txt`, and modular Python architecture.