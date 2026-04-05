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
- **anal_lease_licence** (single turn): groundedness **4/5**, relevancy **5/5**
  - *Rationale:* Compare two institutions from materials (distinction test).
- **anal_chronology_ap** (single turn): groundedness **4/5**, relevancy **5/5**
  - *Rationale:* Chronological reconstruction across multiple factual beats.
- **anal_torrens_framework** (single turn): groundedness **5/5**, relevancy **5/5**
  - *Rationale:* Framework summary spanning readings/slides on Torrens.
- **anal_taxonomy** (single turn): groundedness **5/5**, relevancy **5/5**
  - *Rationale:* Taxonomy spans multiple categories (real vs personal property).

### Family: `conversational_followup`

- **conv_ap_followup** (2 turn(s)): groundedness **4/5**, relevancy **5/5**
  - *Rationale:* Second turn references 'that ratio' — needs session memory.
- **conv_torrens_followup** (2 turn(s)): groundedness **5/5**, relevancy **5/5**
  - *Rationale:* Contrast question after Torrens explanation — coreference to prior answer.
- **conv_chattels_found** (2 turn(s)): groundedness **5/5**, relevancy **5/5**
  - *Rationale:* Hypothetical extension after test — requires prior topic + retrieval.

## 1. Low Groundedness (Agent < 4/5)

No queries scored below 4. All agent answers were well-grounded.

## 2. Ablation Outperformed Agent

Cases where removing the Ratio Extractor improved groundedness, suggesting the node introduced unverifiable inferences.

### Q: "What legal principle governs the distinction between a lease and a licence, and how does each concept apply in outline?"
- **Agent:** 4/5 | **Ablation:** 5/5
- **Agent judge reasoning:** The draft is a bit too generous. The answer is mostly accurate and well supported: it correctly states that a lease is an interest in land of certain duration with reversion, and that a licence is a personal permission that prevents trespass while subsisting. However, saying the governing principle is simply 'the intention of the parties' overstates Source 2. That source reports an argument that the distinction depends on intention rather than exclusive possession; it does not clearly establish this as the definitive governing principle across the materials. The answer also cites 'Derived Analysis' and imports a slightly stronger synthesis than the sources explicitly warrant. Still, there are no clear invented facts, and the outline of each concept is sound. So this is strong but not perfect.

### Q: "Give one short exam tip tailored to that ratio for an open-book exam."
- **Agent:** 4/5 | **Ablation:** 5/5
- **Agent judge reasoning:** The draft is a bit too generous. The answer is short, tailored to an open-book exam, and broadly consistent with the material: *Perry v Clissold* does support the idea that peaceful possession without the true owner’s consent can found a possessory title good against all but the true owner, later maturing if limitation expires. So the tip is useful and mostly grounded.

But there are two issues preventing a 5. First, the answer frames the ratio as the “two possession elements immediately: factual possession and intention to possess.” Those concepts are supported generally in the materials on possession, but the specific ratio from *Perry v Clissold* in the supplied text is more directly the possessory-title point: a person in possession as apparent owner, exercising ordinary rights of ownership peaceably, has title against all but the rightful owner, which can become absolute after the limitation period. So the tip slightly shifts from the asked-for ratio to broader possession analysis. Second, the citation is sloppy: it says “Lecture 3 Slide 3,” but the relevant slide about possessory title is the first lecture slide shown, even though its footer says slide number 3. The junior judge’s claim that this “correctly corresponds to the slide number in the footer of the first lecture slide” is a stretch and overlooks the ambiguity.

Overall, strong and not materially fabricated, but not perfectly precise enough for full marks.

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
- **Verdicts:** [True, True, True, False, True, True, True, True, False, False, False, False]

## 4. Low Answer Relevancy (Agent < 4/5)

All agent answers scored 4+ on relevancy.

## 5. Baseline Outperformed Agent

The agent matched or outperformed the baseline on all queries.

## 6. Summary

| Category | Count | Percentage |
|----------|-------|------------|
| Total queries | 15 | 100% |
| Low groundedness (<4) | 0 | 0% |
| Ablation outperformed agent | 2 | 13% |
| Low context precision (<70%) | 11 | 73% |
| Low answer relevancy (<4) | 0 | 0% |
| Baseline outperformed agent | 0 | 0% |
