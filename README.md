# Health Claim Verification with NLP and RAG
> A three-phase pipeline for automated health claim verification on the PubHealth dataset — combining supervised fine-tuning (PubMedBERT), gradient-based explainability (Integrated Gradients), and a zero-shot Retrieval-Augmented Generation (RAG) system backed by live PubMed evidence.

---

## Repository Structure

```
.
├── PUBHEALTH/                  # PubHealth dataset (train / val / test splits)
├── results/                    # Saved outputs from the RAG evaluation run
├── pubhealth_eda.ipynb         # Exploratory data analysis (label distribution, length stats, missing data)
├── asho-uab.ipynb              # Phase 1 — PubMedBERT fine-tuning and evaluation
├── explainability.ipynb        # Phase 2 — Integrated Gradients + attention attribution
├── rag_pipeline.ipynb          # Phase 3 — Zero-shot RAG pipeline (PubMed + Llama 3.1)
├── ASHO UAB FollowUp.pdf       # Project follow-up presentation
└── README.md
```

---

## The Three Phases

### Phase 1 — Supervised Classification Baseline
**Notebook:** `asho-uab.ipynb`

Fine-tunes **PubMedBERT** (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`) on the PubHealth dataset for 4-class health claim classification: `true`, `false`, `mixture`, `unproven`.

Key design decisions:
- **Smart truncation**: claim is always preserved in full; remaining 512-token budget is allocated to the prefix of `main_text`.
- **Class-weighted loss**: inverse-frequency weights applied to cross-entropy to address severe class imbalance (`unproven` = 3% of training data).
- **Macro F1** as the primary metric — treats all four classes equally regardless of frequency.
- Trained on 50% of the training set (4,902 examples) due to Kaggle T4 compute limits.

| Metric | Value |
|---|---|
| Macro F1 (100-sample test subset) | 0.579 |
| Accuracy | 66.7% |

---

### Phase 2 — Explainability Layer
**Notebook:** `explainability.ipynb`

Adds human-readable attribution to the fine-tuned model's predictions using two complementary methods.

**Raw Attention** — CLS-row attention from layer 11, averaged over 12 heads. Fast, zero-overhead proxy for token importance. Rendered as a per-head heatmap and inline token highlights.

**Integrated Gradients (IG)** — Captum `LayerIntegratedGradients` applied to the embedding layer, integrated from an all-`[PAD]` baseline over 30 steps. More faithful attribution. Token scores are summed across the embedding dimension, merged to word level, and extended to **span-level** (top-3 sentences by IG score) for ROUGE evaluation.

**Visualisation** — Three-column HTML card rendered in Jupyter:
- *Left*: input text with tokens coloured yellow→orange→red by IG score + top-10 token table; below it, the **top-3 sentences ranked by span-level IG score** (the candidate explanation used in quantitative validation)
- *Middle*: same text highlighted by raw attention scores
- *Right*: gold `explanation` field for direct side-by-side comparison

**Quantitative validation** against the gold `explanation` field over the full validation split (1,214 samples):

| Metric | Mean | Std |
|---|---|---|
| ROUGE-1 | 0.222 | 0.094 |
| ROUGE-2 | 0.047 | 0.056 |
| ROUGE-L | 0.134 | 0.060 |
| BERTScore-F1 | 0.849 | 0.020 |

---

### Phase 3 — Zero-Shot RAG Pipeline
**Notebook:** `rag_pipeline.ipynb`

A fully zero-shot verification pipeline — no fine-tuning, no labelled data required — built for claims that postdate the training set.

**Four-stage pipeline:**

1. **Query distillation** — Llama 3.1-8b-instant (Groq) extracts 3–5 biomedical keywords from the raw claim. Raises PubMed retrieval hit rate from 15% → 75%.
2. **PubMed retrieval** — NCBI Entrez API fetches top-5 abstracts for the distilled keywords.
3. **Cross-encoder reranking** — `ms-marco-MiniLM-L-6-v2` scores each (claim, abstract) pair and keeps the top 3 most relevant.
4. **LLM verification** — Llama 3.1-8b-instant reasons over the claim + top-3 abstracts and outputs a label + 2–4 sentence explanation.

Evaluated on a 100-sample random test subset (random_state = 42):

| Metric | Value |
|---|---|
| Macro F1 | 0.213 |
| BERTScore-F1 (explanation) | 0.846 ± 0.023 |
| PubMed retrieval coverage | 76% |

> **Key finding:** The RAG and IG explanation approaches achieve near-identical BERTScore-F1 (~0.847) despite operating through entirely different mechanisms — one from internal gradients, one from external evidence.

---

## Dataset

The [PubHealth dataset](https://github.com/neemakot/Health-Fact-Checking) contains real-world health claims annotated by professional fact-checkers with four labels. After cleaning:

| Split | Examples |
|---|---|
| Train | 9,804 |
| Validation | 1,214 |
| Test | 1,233 |

The `PUBHEALTH/` folder contains the raw splits. Run `pubhealth_eda.ipynb` first for a full breakdown of label distribution, text length statistics, and missing value analysis.

---

## Results Summary

| Phase | Method | Primary Metric | Score |
|---|---|---|---|
| 1 | PubMedBERT fine-tuned | Macro F1 | 0.579 |
| 3 | RAG zero-shot | Macro F1 | 0.213 |
| 2 | Integrated Gradients (explanation) | BERTScore-F1 | 0.849 |
| 3 | LLM rationale (explanation) | BERTScore-F1 | 0.846 |

---

## Roadmap

- [x] Exploratory data analysis
- [x] PubMedBERT fine-tuning with smart truncation and class-weighted loss
- [x] Integrated Gradients + raw attention attribution
- [x] Span-level IG evaluation against gold explanations
- [x] RAG pipeline with LLM query distillation, PubMed retrieval, cross-encoder reranking
- [ ] Full-dataset fine-tuning (currently 50% sample due to compute limits)
- [ ] Long-context model (Longformer) for Phase 1 to avoid truncation
- [ ] Web retrieval for non-biomedical claims (political, legal) to improve RAG coverage
- [ ] Multimodal extension (images + text) for modern fake news formats
