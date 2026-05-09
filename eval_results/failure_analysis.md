# Failure Analysis Report

Auto-generated from evaluation results. This report identifies queries where the system underperformed and categorises the root causes.

## 0. Query family coverage

Every eval case is assigned **exactly one** of four families. Use this table for success/failure discussion per family.

| Family | Cases | Case IDs |
|--------|-------|----------|
| **factual_retrieval** | 6 | fact_moran_test, fact_fee_simple, fact_bailment_ratio, fact_perry_possessory, fact_fee_tail_vs_simple, fact_pla_chattels |
| **cross_modal_retrieval** | 5 | xmodal_chattels_slides, xmodal_moukataff_slide, xmodal_ap_slides, xmodal_torrens_indefeasibility, xmodal_bailment_categories |
| **analytical_synthesis** | 7 | anal_ap_elements, anal_lease_licence, anal_chronology_ap, anal_torrens_framework, anal_taxonomy, anal_legal_vs_equitable_interests, anal_torrens_title_passing |
| **conversational_followup** | 4 | conv_ap_followup, conv_torrens_followup, conv_chattels_found, conv_lease_three_turn |

- **factual_retrieval:** Direct retrieval of stored knowledge: doctrine, tests, and definitions that should appear verbatim or paraphrased from a small number of chunks.
- **cross_modal_retrieval:** Requires the lecture-slide channel: questions are phrased to target VLM-indexed slide descriptions alongside PDF readings (multimodal KB).
- **analytical_synthesis:** Multi-evidence answers: compare, relate, summarise across concepts, or reconstruct sequences using several retrieved sources.
- **conversational_followup:** Multi-turn: first user message establishes topic; second message is a follow-up resolved using chat_history (baseline sees only the last turn).

### Family: `factual_retrieval`

- **fact_moran_test** (single turn): groundedness **4/5**, relevancy **5/5**
  - *Rationale:* Named case test — should retrieve specific ratio/test from readings.
- **fact_fee_simple** (single turn): groundedness **5/5**, relevancy **5/5**
  - *Rationale:* Core estate definition — single-concept factual answer.
- **fact_bailment_ratio** (single turn): groundedness **5/5**, relevancy **5/5**
  - *Rationale:* Ratio-style question tied to course materials on bailment.
- **fact_perry_possessory** (single turn): groundedness **5/5**, relevancy **5/5**
  - *Rationale:* Named case holding from indexed readings.
- **fact_fee_tail_vs_simple** (single turn): groundedness **5/5**, relevancy **5/5**
  - *Rationale:* Comparative doctrine question — should retrieve estate-type readings.
- **fact_pla_chattels** (single turn): groundedness **4/5**, relevancy **5/5**
  - *Rationale:* Statute lookup — Property Law Act 1974 (Qld) on chattels.

### Family: `cross_modal_retrieval`

- **xmodal_chattels_slides** (single turn): groundedness **5/5**, relevancy **5/5**
  - *Rationale:* Forces reliance on VLM-described lecture slides for chattels possession.
  - *Retrieval:* text chunks=8, slide chunks=4, both_modalities=True
- **xmodal_moukataff_slide** (single turn): groundedness **5/5**, relevancy **5/5**
  - *Rationale:* Slide-anchored case (Moukataff) should appear in slide descriptions.
  - *Retrieval:* text chunks=8, slide chunks=4, both_modalities=True
- **xmodal_ap_slides** (single turn): groundedness **5/5**, relevancy **5/5**
  - *Rationale:* Asks for emphasis from slide-indexed adverse possession teaching.
  - *Retrieval:* text chunks=8, slide chunks=4, both_modalities=True
- **xmodal_torrens_indefeasibility** (single turn): groundedness **5/5**, relevancy **5/5**
  - *Rationale:* Targets lecture slides on Torrens indefeasibility — VLM-described content.
  - *Retrieval:* text chunks=8, slide chunks=4, both_modalities=True
- **xmodal_bailment_categories** (single turn): groundedness **5/5**, relevancy **5/5**
  - *Rationale:* Slide-anchored taxonomy: bailment categories from lecture deck.
  - *Retrieval:* text chunks=8, slide chunks=4, both_modalities=True

### Family: `analytical_synthesis`

- **anal_ap_elements** (single turn): groundedness **5/5**, relevancy **5/5**
  - *Rationale:* Must relate factual possession and animus across sources.
- **anal_lease_licence** (single turn): groundedness **5/5**, relevancy **5/5**
  - *Rationale:* Compare two institutions from materials (distinction test).
- **anal_chronology_ap** (single turn): groundedness **5/5**, relevancy **5/5**
  - *Rationale:* Chronological reconstruction across multiple factual beats.
- **anal_torrens_framework** (single turn): groundedness **5/5**, relevancy **5/5**
  - *Rationale:* Framework summary spanning readings/slides on Torrens.
- **anal_taxonomy** (single turn): groundedness **5/5**, relevancy **5/5**
  - *Rationale:* Taxonomy spans multiple categories (real vs personal property).
- **anal_legal_vs_equitable_interests** (single turn): groundedness **5/5**, relevancy **5/5**
  - *Rationale:* Comparison across multiple chunks of equitable vs legal interests.
- **anal_torrens_title_passing** (single turn): groundedness **4/5**, relevancy **5/5**
  - *Rationale:* Sequenced reasoning: passage of legal title in a Torrens transfer.

### Family: `conversational_followup`

- **conv_ap_followup** (2 turn(s)): groundedness **5/5**, relevancy **5/5**
  - *Rationale:* Second turn references 'that ratio' — needs session memory.
- **conv_torrens_followup** (2 turn(s)): groundedness **5/5**, relevancy **5/5**
  - *Rationale:* Contrast question after Torrens explanation — coreference to prior answer.
- **conv_chattels_found** (2 turn(s)): groundedness **5/5**, relevancy **5/5**
  - *Rationale:* Hypothetical extension after test — requires prior topic + retrieval.
- **conv_lease_three_turn** (3 turn(s)): groundedness **5/5**, relevancy **5/5**
  - *Rationale:* Three-turn case where the third turn references the second.

## 1. Low Groundedness (Agent < 4/5)

No queries scored below 4. All agent answers were well-grounded.

## 2. Ablation Outperformed Agent

Cases where the single mega-prompt outperformed the full agent on groundedness, suggesting the multi-node pipeline introduced unverifiable inferences on those queries.

The agent matched or outperformed the ablation on all queries.

## 3. Low Context Precision (Retrieval < 70%)

Queries where fewer than 70% of retrieved chunks were judged relevant, indicating the retrieval missed the target topic.

### Q: "What is the legal test from Buckinghamshire County Council v Moran?"
- **Precision@K:** 17% (2/12 relevant)
- **Verdicts:** [True, False, False, False, False, False, False, False, False, True, False, False]

### Q: "What is a fee simple estate?"
- **Precision@K:** 67% (8/12 relevant)
- **Verdicts:** [True, True, True, True, True, True, True, True, False, False, False, False]

### Q: "What is the ratio decidendi regarding bailment at will?"
- **Precision@K:** 67% (8/12 relevant)
- **Verdicts:** [True, False, True, True, True, False, True, True, True, True, False, False]

### Q: "What is the ratio in Perry v Clissold regarding possessory title?"
- **Precision@K:** 67% (8/12 relevant)
- **Verdicts:** [True, True, True, True, False, True, True, True, True, False, False, False]

### Q: "According to the indexed lecture slide materials in the knowledge base, what two elements are required to establish possession of chattels?"
- **Precision@K:** 58% (7/12 relevant)
- **Verdicts:** [True, True, True, False, False, True, True, False, False, False, True, True]

### Q: "What does the indexed lecture content describe about Moukataff v BOAC and baggage or chattel possession?"
- **Precision@K:** 67% (8/12 relevant)
- **Verdicts:** [True, True, True, True, True, True, False, True, True, False, False, False]

### Q: "What legal principle governs the distinction between a lease and a licence, and how does each concept apply in outline?"
- **Precision@K:** 42% (5/12 relevant)
- **Verdicts:** [True, False, False, True, False, True, True, True, False, False, False, False]

### Q: "Summarise the legal framework for land registration under the Torrens system as presented in the indexed course materials."
- **Precision@K:** 67% (8/12 relevant)
- **Verdicts:** [True, True, True, True, True, True, True, True, False, False, False, False]

### Q: "In one paragraph, how does that differ from a deeds registry? Stay within our indexed course sources only."
- **Precision@K:** 67% (8/12 relevant)
- **Verdicts:** [True, True, True, True, True, False, True, True, False, False, True, False]

### Q: "What is the difference between a fee tail and a fee simple in our materials?"
- **Precision@K:** 58% (7/12 relevant)
- **Verdicts:** [True, True, True, True, True, False, True, True, False, False, False, False]

### Q: "What does the Property Law Act 1974 (Qld) say about chattels in the readings?"
- **Precision@K:** 33% (4/12 relevant)
- **Verdicts:** [True, False, True, False, False, False, False, False, True, False, False, True]

### Q: "Looking at the indexed lecture slides on the Torrens system, how do those materials present the doctrine of indefeasibility of title?"
- **Precision@K:** 67% (8/12 relevant)
- **Verdicts:** [True, True, True, True, True, True, True, True, False, False, False, False]

### Q: "From the lecture slides indexed in the knowledge base, what categories of bailment do those materials describe?"
- **Precision@K:** 58% (7/12 relevant)
- **Verdicts:** [True, True, True, False, True, True, False, False, True, False, True, False]

### Q: "Sequence the steps by which legal title passes in a Torrens-system land transfer, citing the indexed course materials at each step."
- **Precision@K:** 50% (6/12 relevant)
- **Verdicts:** [True, False, True, True, True, True, False, True, False, False, False, False]

### Q: "Which of those points you just made would change if the licence were irrevocable? Stay within our indexed sources only."
- **Precision@K:** 50% (6/12 relevant)
- **Verdicts:** [False, False, True, False, True, True, True, True, False, False, False, True]

## 4. Low Answer Relevancy (Agent < 4/5)

All agent answers scored 4+ on relevancy.

## 5. Baseline Outperformed Agent

The agent matched or outperformed the baseline on all queries.

## 5b. Reranker Ablation (Agent vs no-reranker)

Compares context precision between the full agent (cross-encoder reranker on) and the same agent run with `USE_RERANKER=False`. This isolates the contribution of the reranker.

- Reranker improved precision on **11/22** queries
- Reranker hurt precision on **3/22** queries
- Tie on **8/22**

## 6. Summary

| Category | Count | Percentage |
|----------|-------|------------|
| Total queries | 22 | 100% |
| Low groundedness (<4) | 0 | 0% |
| Ablation outperformed agent | 0 | 0% |
| Low context precision (<70%) | 15 | 68% |
| Low answer relevancy (<4) | 0 | 0% |
| Baseline outperformed agent | 0 | 0% |
