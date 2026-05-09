# Human Spot-Check Log

_This log exists to support **qualitative validation of the LLM-as-a-judge**_
_pipeline. Five evaluation cases have been sampled deterministically_
_(seed=4205). Fill in **Human Score (1–5)** and **Comments** by hand,_
_then re-run the harness or compute disagreement (mean absolute error_
_between Human Score and `judge_groundedness.score`) yourself._

Disagreement convention: |human − judge| ≥ 2 indicates a substantive
rating gap that should be discussed in the report.

---

## Spot-check 1 — `fact_perry_possessory` (factual_retrieval)

**Question:** What is the ratio in Perry v Clissold regarding possessory title?

**Agent answer (excerpt):**

```
## Ratio in *Perry v Clissold* on possessory title

The ratio is:

> **A person in peaceful possession of land, in the assumed character of owner and exercising the ordinary rights of ownership, has a perfectly good possessory title against all the world except the rightful owner.** (Source: Readings Week 3; Supplementary Notes)

### In plain English
This means that **possession itself gives a legal title of its own**. So even if someone does **not** have documentary title, they can still have a title that is enforceable against everyone except someone with a better right to possession. (Source: Lecture 3 Slide 1; Supplementary Notes)

### Why that matters in *Perry v Clissold*
Clissold had taken possession of the land, enclosed it, received rents, and paid rates and taxes. The Privy Council held that he was entitled to compensation when the land was compulsorily acquired, because his **possessory title** was enough. (Source: Readings Week 3)

### Short exam-style formulation
**Possession confers a possessory title, and that title is good against all but the true owner.** (Source: Supplementary Notes; Lecture 3 Slide 1)

If you want, I can also turn this into a 2–3 sentence exam an

…[truncated for readability]
```

**Judge groundedness:** 5/5  
**Judge answer relevancy:** 5/5

**Human groundedness (1–5):** 5  
**Human answer relevancy (1–5):** 5

**Comments (free text):**

> 

---

## Spot-check 2 — `xmodal_moukataff_slide` (cross_modal_retrieval)

**Question:** What does the indexed lecture content describe about Moukataff v BOAC and baggage or chattel possession?

**Agent answer (excerpt):**

```
## Moukataff v BOAC: baggage, chattels, and possession

The lecture content uses **Moukataff v BOAC [1967] 1 Lloyd’s Rep 396** to show that **you can have possession of a container and its contents even if you do not know the exact contents**, so long as you knowingly take charge of the general class of items involved. (Source: Lecture 3 Slide 6; Supplementary Notes 77, 37)

### What the case was about
- A bank sent **£20,000 in four packages** to Kuwait by **registered mail**. (Source: Supplementary Notes 77; Reading Week 3)
- The packages were given to **BOAC** for carriage by air. (Source: Supplementary Notes 37; Reading Week 3)
- An employee/loader stole the money while the parcels were being handled by BOAC. (Source: Supplementary Notes 77; Reading Week 3)

### What the lecture says the case stands for
The indexed lecture content says the key point is **“intention to control”**:
- You do **not** need knowledge of the precise contents of what you physically control. (Source: Lecture 3 Slide 6)
- Browne J said it was “almost absurd” to describe BOAC as **involuntary bailees**, because:
  - BOAC knew the mail bags contained mail; and
  - the **red-labelled registered mail bags**

…[truncated for readability]
```

**Judge groundedness:** 5/5  
**Judge answer relevancy:** 5/5

**Human groundedness (1–5):** 5  
**Human answer relevancy (1–5):** 5

**Comments (free text):**

> 

---

## Spot-check 3 — `fact_fee_simple` (factual_retrieval)

**Question:** What is a fee simple estate?

**Agent answer (excerpt):**

```
## Fee simple estate

A **fee simple estate** is the **most extensive freehold estate** known to the common law. It is the closest thing to “full ownership” of land for practical purposes, although under the doctrine of tenure land is still not treated as absolutely owned in the modern sense. (Source: Supplementary Notes - Greg 16.02.2026.pdf)

### Key features
- It is **of uncertain duration** and can potentially last **indefinitely**.  
- It is **inheritable**: it can pass to heirs.  
- It is **alienable**: the holder can transfer it, and can also create lesser interests out of it, such as **life estates** and **leases**.  
- It is **unrestricted**: unlike fee tail, it is not limited to particular heirs. (Source: Supplementary Notes - Greg 16.02.2026.pdf)

### In simple terms
If someone owns land in fee simple, they have the strongest form of estate in land. They can usually use, sell, or pass it on, and the estate can continue for a very long time unless it is brought to an end by some event. (Source: Supplementary Notes - Greg 16.02.2026.pdf)

### Types of fee simple
The notes also say there are **three types**:
1. **Fee simple absolute** — perpetual and not determinable by any

…[truncated for readability]
```

**Judge groundedness:** 5/5  
**Judge answer relevancy:** 5/5

**Human groundedness (1–5):** 5
**Human answer relevancy (1–5):** 5

**Comments (free text):**

> 

---

## Spot-check 4 — `conv_ap_followup` (conversational_followup)

**Question:** Give one short exam tip tailored to that ratio for an open-book exam.

**Agent answer (excerpt):**

```
One short open-book exam tip: **use the facts in a checklist** — first spot **factual possession + intention to possess**, then confirm it was **open, peaceful, and without consent**, and only after that move to the **limitation period**. (Source: Supplementary Notes Source 1; Source 5)
```

**Judge groundedness:** 5/5  
**Judge answer relevancy:** 5/5

**Human groundedness (1–5):** 5  
**Human answer relevancy (1–5):** 5

**Comments (free text):**

> 

---

## Spot-check 5 — `conv_chattels_found` (conversational_followup)

**Question:** Does anything in our indexed materials suggest the answer changes if the chattel was found on the ground rather than handed over? Explain briefly.

**Agent answer (excerpt):**

```
Yes — the materials suggest **the answer can change** if the chattel was **found on the ground** rather than **handed over**.

## Brief explanation

For ordinary possession, the lecture says you need **physical control** plus **an intention to exclusively control** (Source: Lecture 3 Slide 4). The readings also say a person does **not** possess a chattel if they are **unaware of its existence**; possession only arises once they **know of it and assent to possessing it** (Source: Readings Week 3).

But the readings also show a **special rule for chattels found on premises**. In that setting, the occupier’s rights may depend on whether they had, **before the item was found**, **manifested an intention to exercise control over the building and the things which may be upon it or in it** (Source: Readings Week 3). The materials say this is especially strong in a **private residence**, where that intention may be **presumed** and the householder will “almost invariably” possess the lost article, even without knowing it was there (Source: Readings Week 3).

## So the practical difference is

- **Handed over / in someone’s custody:** the normal knowledge-and-assent idea matters more.  
- *

…[truncated for readability]
```

**Judge groundedness:** 5/5  
**Judge answer relevancy:** 5/5

**Human groundedness (1–5):** 5 
**Human answer relevancy (1–5):** 5

**Comments (free text):**

> 

---


## Disagreement Summary (auto-computed)

_Computed from human ratings filled in above (n=5). Mean absolute error_
_(MAE) between human score and judge score; lower is better. |Δ|≥2 counts_
_substantive disagreements per the convention at the top of this file._

| Metric | n | MAE | Exact agreement | Substantive disagreement (\|Δ\|≥2) |
|--------|---|-----|-----------------|-----------------------------------|
| Groundedness | 5 | 0.00 | 5/5 | 0/5 |
| Answer relevancy | 5 | 0.00 | 5/5 | 0/5 |

### Per-case disagreement

| Case | Family | Human/Judge G | \|Δ\| G | Human/Judge R | \|Δ\| R |
|------|--------|---------------|---------|----------------|---------|
| `fact_perry_possessory` | factual_retrieval | 5/5 | 0 | 5/5 | 0 |
| `xmodal_moukataff_slide` | cross_modal_retrieval | 5/5 | 0 | 5/5 | 0 |
| `fact_fee_simple` | factual_retrieval | 5/5 | 0 | 5/5 | 0 |
| `conv_ap_followup` | conversational_followup | 5/5 | 0 | 5/5 | 0 |
| `conv_chattels_found` | conversational_followup | 5/5 | 0 | 5/5 | 0 |
