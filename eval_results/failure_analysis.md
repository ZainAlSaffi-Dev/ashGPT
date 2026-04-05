# Failure Analysis Report

Auto-generated from evaluation results. This report identifies queries where the system underperformed and categorises the root causes.

## 1. Low Groundedness (Agent < 4/5)

No queries scored below 4. All agent answers were well-grounded.

## 2. Ablation Outperformed Agent

Cases where removing the Ratio Extractor improved groundedness, suggesting the node introduced unverifiable inferences.

The agent matched or outperformed the ablation on all queries.

## 3. Low Context Precision (Retrieval < 70%)

Queries where fewer than 70% of retrieved chunks were judged relevant, indicating the retrieval missed the target topic.

### Q: "What is the ratio decidendi for adverse possession?"
- **Precision@K:** 67% (8/12 relevant)
- **Verdicts:** [True, True, True, True, False, True, True, True, True, False, False, False]

### Q: "What is the ratio in Perry v Clissold regarding possessory title?"
- **Precision@K:** 58% (7/12 relevant)
- **Verdicts:** [True, True, True, True, True, False, True, False, True, False, False, False]

### Q: "What legal principle governs the distinction between a lease and a licence?"
- **Precision@K:** 8% (1/12 relevant)
- **Verdicts:** [False, True, False, False, False, False, False, False, False, False, False, False]

### Q: "What is the legal test from Buckinghamshire County Council v Moran?"
- **Precision@K:** 17% (2/12 relevant)
- **Verdicts:** [True, False, False, False, False, False, False, False, False, True, False, False]

### Q: "What is the ratio decidendi regarding bailment at will?"
- **Precision@K:** 50% (6/12 relevant)
- **Verdicts:** [True, True, True, True, True, False, False, False, True, False, False, False]

### Q: "Show me the timeline of events in Perry v Clissold"
- **Precision@K:** 8% (1/12 relevant)
- **Verdicts:** [True, False, False, False, False, False, False, False, False, False, False, False]

### Q: "Map out the chain of events in Whittlesea City Council v Abbatangelo"
- **Precision@K:** 17% (2/12 relevant)
- **Verdicts:** [True, True, False, False, False, False, False, False, False, False, False, False]

### Q: "Show the chronological sequence of how adverse possession is established"
- **Precision@K:** 67% (8/12 relevant)
- **Verdicts:** [True, True, True, True, True, True, True, True, False, False, False, False]

### Q: "Map out the timeline of events in a compulsory acquisition of land"
- **Precision@K:** 33% (4/12 relevant)
- **Verdicts:** [True, True, True, False, True, False, False, False, False, False, False, False]

### Q: "Explain the relationship between factual possession and animus possidendi"
- **Precision@K:** 67% (8/12 relevant)
- **Verdicts:** [True, True, True, True, True, False, True, False, True, False, True, False]

### Q: "Summarise the legal framework for adverse possession under the Torrens system"
- **Precision@K:** 50% (6/12 relevant)
- **Verdicts:** [True, True, True, False, True, True, False, False, True, False, False, False]

### Q: "What is a fee simple estate?"
- **Precision@K:** 67% (8/12 relevant)
- **Verdicts:** [True, True, True, True, True, True, True, True, False, False, False, False]

### Q: "How does the concept of possession differ between land and chattels?"
- **Precision@K:** 67% (8/12 relevant)
- **Verdicts:** [True, True, False, True, True, True, True, True, True, False, False, False]

### Q: "Explain the Torrens system of land registration"
- **Precision@K:** 67% (8/12 relevant)
- **Verdicts:** [True, True, True, True, True, True, True, True, False, False, False, False]

## 4. Low Answer Relevancy (Agent < 4/5)

All agent answers scored 4+ on relevancy.

## 5. Baseline Outperformed Agent

The agent matched or outperformed the baseline on all queries.

## 6. Summary

| Category | Count | Percentage |
|----------|-------|------------|
| Total queries | 20 | 100% |
| Low groundedness (<4) | 0 | 0% |
| Ablation outperformed agent | 0 | 0% |
| Low context precision (<70%) | 14 | 70% |
| Low answer relevancy (<4) | 0 | 0% |
| Baseline outperformed agent | 0 | 0% |
