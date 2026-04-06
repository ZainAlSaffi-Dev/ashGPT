# Failure Analysis Report

Auto-generated from evaluation results. This report identifies queries where the system underperformed and categorises the root causes.

## 0. Query family coverage

Every eval case is assigned **exactly one** of four families. Use this table for success/failure discussion per family.

| Family | Cases | Case IDs |
|--------|-------|----------|
| **factual_retrieval** | 4 | fact_moran_test, fact_fee_simple, fact_bailment_ratio, fact_perry_possessory |
| **cross_modal_retrieval** | 3 | xmodal_chattels_slides, xmodal_moukataff_slide, xmodal_ap_slides |
| **analytical_synthesis** | 5 | anal_ap_elements, anal_lease_licence, anal_chronology_ap, anal_torrens_framework, anal_taxonomy |
| **conversational_followup** | 3 | conv_ap_followup, conv_torrens_followup, conv_chattels_found |

- **factual_retrieval:** Direct retrieval of stored knowledge: doctrine, tests, and definitions that should appear verbatim or paraphrased from a small number of chunks.
- **cross_modal_retrieval:** Requires the lecture-slide channel: questions are phrased to target VLM-indexed slide descriptions alongside PDF readings (multimodal KB).
- **analytical_synthesis:** Multi-evidence answers: compare, relate, summarise across concepts, or reconstruct sequences using several retrieved sources.
- **conversational_followup:** Multi-turn: first user message establishes topic; second message is a follow-up resolved using chat_history (baseline sees only the last turn).

### Family: `factual_retrieval`

- **fact_moran_test** (single turn): groundedness **5/5**, relevancy **5/5**
  - *Rationale:* Named case test — should retrieve specific ratio/test from readings.
- **fact_fee_simple** (single turn): groundedness **5/5**, relevancy **5/5**
  - *Rationale:* Core estate definition — single-concept factual answer.
- **fact_bailment_ratio** (single turn): groundedness **5/5**, relevancy **5/5**
  - *Rationale:* Ratio-style question tied to course materials on bailment.
- **fact_perry_possessory** (single turn): groundedness **5/5**, relevancy **5/5**
  - *Rationale:* Named case holding from indexed readings.

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

### Family: `conversational_followup`

- **conv_ap_followup** (2 turn(s)): groundedness **5/5**, relevancy **5/5**
  - *Rationale:* Second turn references 'that ratio' — needs session memory.
- **conv_torrens_followup** (2 turn(s)): groundedness **5/5**, relevancy **5/5**
  - *Rationale:* Contrast question after Torrens explanation — coreference to prior answer.
- **conv_chattels_found** (2 turn(s)): groundedness **4/5**, relevancy **5/5**
  - *Rationale:* Hypothetical extension after test — requires prior topic + retrieval.

## 1. Low Groundedness (Agent < 4/5)

No queries scored below 4. All agent answers were well-grounded.

## 2. Ablation Outperformed Agent

Cases where the single mega-prompt outperformed the full agent on groundedness, suggesting the multi-node pipeline introduced unverifiable inferences on those queries.

### Q: "Does anything in our indexed materials suggest the answer changes if the chattel was found on the ground rather than handed over? Explain briefly."
- **Agent:** 4/5 | **Ablation:** 5/5
- **Agent judge reasoning:** The draft is a bit too generous. Most of the answer is well grounded: it correctly states the general possession test as physical control plus intention to exclusively control, and it accurately explains the special finder/occupier distinction for items found on land versus in/attached to land, including the occupier’s need to manifest an intention to control and the relevance of Bridges v Hawkesworth. However, the final answer goes beyond the indexed materials when it says, 'If it was merely handed over to someone, those land-based finder rules are not the focus.' The sources provided discuss objects found on/in land and, in Parker, the finder handing the item to an airline employee after finding it; they do not support a separate contrast between 'found on the ground' and 'handed over' as though handing over changes the legal framework. That is an unsupported inference, though not a major fabrication. So the answer is strong but not flawless.

## 3. Low Context Precision (Retrieval < 70%)

Queries where fewer than 70% of retrieved chunks were judged relevant, indicating the retrieval missed the target topic.

### Q: "What is the legal test from Buckinghamshire County Council v Moran?"
- **Precision@K:** 17% (2/12 relevant)
- **Verdicts:** [True, False, False, False, False, False, False, False, False, True, False, False]

### Q: "What is a fee simple estate?"
- **Precision@K:** 67% (8/12 relevant)
- **Verdicts:** [True, True, True, True, True, True, True, True, False, False, False, False]

### Q: "What is the ratio decidendi regarding bailment at will?"
- **Precision@K:** 50% (6/12 relevant)
- **Verdicts:** [True, True, True, True, True, False, False, False, True, False, False, False]

### Q: "What is the ratio in Perry v Clissold regarding possessory title?"
- **Precision@K:** 58% (7/12 relevant)
- **Verdicts:** [True, True, True, True, True, False, True, False, True, False, False, False]

### Q: "According to the indexed lecture slide materials in the knowledge base, what two elements are required to establish possession of chattels?"
- **Precision@K:** 50% (6/12 relevant)
- **Verdicts:** [False, True, True, True, False, True, False, False, False, True, True, False]

### Q: "What does the indexed lecture content describe about Moukataff v BOAC and baggage or chattel possession?"
- **Precision@K:** 42% (5/12 relevant)
- **Verdicts:** [True, True, True, True, False, False, False, False, True, False, False, False]

### Q: "Based on the VLM-described lecture slides on adverse possession in the knowledge base, what sequence or steps do those materials emphasise?"
- **Precision@K:** 67% (8/12 relevant)
- **Verdicts:** [True, True, True, True, True, True, True, True, False, False, False, False]

### Q: "What legal principle governs the distinction between a lease and a licence, and how does each concept apply in outline?"
- **Precision@K:** 42% (5/12 relevant)
- **Verdicts:** [True, True, False, True, False, True, True, False, False, False, False, False]

### Q: "Show the chronological sequence of how adverse possession is established under the materials we indexed."
- **Precision@K:** 58% (7/12 relevant)
- **Verdicts:** [True, True, True, True, True, True, True, False, False, False, False, False]

### Q: "Summarise the legal framework for land registration under the Torrens system as presented in the indexed course materials."
- **Precision@K:** 67% (8/12 relevant)
- **Verdicts:** [True, True, True, True, True, True, True, True, False, False, False, False]

### Q: "In one paragraph, how does that differ from a deeds registry? Stay within our indexed course sources only."
- **Precision@K:** 58% (7/12 relevant)
- **Verdicts:** [True, True, True, True, True, True, False, True, False, False, False, False]

## 4. Low Answer Relevancy (Agent < 4/5)

All agent answers scored 4+ on relevancy.

## 5. Baseline Outperformed Agent

The agent matched or outperformed the baseline on all queries.

## 6. Summary

| Category | Count | Percentage |
|----------|-------|------------|
| Total queries | 15 | 100% |
| Low groundedness (<4) | 0 | 0% |
| Ablation outperformed agent | 1 | 7% |
| Low context precision (<70%) | 11 | 73% |
| Low answer relevancy (<4) | 0 | 0% |
| Baseline outperformed agent | 0 | 0% |
