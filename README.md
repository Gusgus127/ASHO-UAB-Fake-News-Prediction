# ASHO-UAB-Fake-News-Prediction

## Project Architecture: Health Claim Verification Pipeline

---

### 1. Baseline Model
The foundation of the system focuses on specialized transformer architectures to handle medical linguistics.

* **Model:** Fine-tune a domain-specific transformer (e.g., **BioBERT** or **PubMedBERT**) on the **PubHealth** dataset.
* **Preprocessing:** Truncation is required for long-form medical text.
* **Input:** Concatenated `claim` + `main_text`.
* **Output:** 4-class classification:
    1.  True
    2.  False
    3.  Mixture
    4.  Unproven
* **Metric:** **Macro F1-score** (selected to specifically account for class imbalance within the dataset).

---

### 2. Explainability
To move beyond "black-box" AI, this stage focuses on making the model’s reasoning transparent to medical professionals.

* **Methods:**
  1. **Raw Attention** — Last transformer layer (layer 11), CLS-row attention averaged over 12 heads. Fast proxy; visualised as a per-head heatmap and as inline token highlights.
  2. **Integrated Gradients** (Captum `LayerIntegratedGradients`) — Gradients integrated from an all-PAD baseline to the actual input through the embedding layer. More faithful attribution; used as the primary highlight signal.
* **Visualization:** Three-column HTML card rendered in Jupyter:
  * *Left* — Input text (claim + main\_text) with tokens coloured yellow → orange → red by IG score, plus a top-10 ranked token table.
  * *Middle* — Same text highlighted by raw attention scores.
  * *Right* — Gold `explanation` field for direct side-by-side human comparison.
* **Validation:** Top-20 IG tokens reconstructed as a candidate string and compared against the gold `explanation` field using **ROUGE-1 / ROUGE-2 / ROUGE-L** and **BERTScore F1** over the full validation split (1,221 samples). Results broken down per label class (true / false / mixture / unproven).
* **Notebook:** `explainability.ipynb`
* **Goal:** Bridge the gap between automated predictions and human-readable justifications.



---

### 3. Retrieval Augmentation & Zero-Shot
This stage implements a dynamic verification pipeline to handle emerging health trends.

* **Pipeline:** Implement **Retrieval-Augmented Generation (RAG)**.
    * **Retrieve:** Use the claim as a query to fetch real-time medical evidence from **PubMed** or the **Web** via API.
    * **Verify:** Feed the original `claim` + the `retrieved evidence` into a Large Language Model (e.g., **Llama 3** or **GPT-4o-mini**).
* **Goal:** Enable the system to verify new, unseen health claims (**Zero-Shot**) without requiring constant retraining on new datasets.