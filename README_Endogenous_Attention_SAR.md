# AWARE: Attention-Weighted and Reduced Entropy
### A Clean and Efficient Framework for Semantic-Aware Uncertainty in LLMs

## 📌 Summary
This repository re-implements and extends the core methodology of *"Shifting Attention to Relevance (SAR)"* by Duan et al., introducing:

- ✅ **A modular, memory-efficient SAR-based pipeline**
- ✅ **Endogenous attention-weighted token importance** (no external SBERT)
- ✅ **Dimensionally-collapsed logit entropy** for semantically-refined uncertainty
- ✅ **Support for multiple uncertainty metrics** (SAR, PE, SE, LN-PE, and more)
- ✅ **Minimal dependencies with HuggingFace-native integration**

> This implementation serves as a reproducible and extensible foundation for research on uncertainty estimation, semantic relevance, and token-level confidence in LLMs — now refined into **AWARE**, a novel measure based on Attention-Weighted and Reduced Entropy.

---

## 🔬 Key Contributions
- **Main Pipeline** (`main_pipeline.py`): Unified, batch-based processing for datasets like SciQ, TriviaQA, and CoQA
- **Plug-in Uncertainty Modules**: Including SAR, token-SAR, sentence-SAR, PE, SE, semantic entropy, and more
- **Endogenous Token Importance via Attention Weights**: Replacing SBERT with internal model attention for interpretable and efficient weighting
- **Collapsed Logit Space for Semantic Refinement**: Reducing output vocabulary to task-relevant dimensions for more accurate entropy calculation
- **Reproducible Experiments**: Scripts replicate Duan et al.’s Table 1–4 and Figure 5–6

---

## 📁 Repository Structure

```bash
├── main_pipeline.py
├── config/
│   └── config.py
├── loaders/
│   ├── sciq_loader.py
│   ├── coqa_loader.py
│   └── triviaqa_loader.py
├── models/
│   └── generator.py
├── analysis/
│   ├── likelihoods.py
│   ├── uncertainty.py
│   ├── similarity.py
│   └── correctness.py
├── utils/
│   └── logger.py
├── scripts/
│   └── run_all_uncertainty.sh
└── results/  # output CSVs + pickles
```

---

## 🚀 Getting Started

To reproduce results:

```bash
bash scripts/run_sar_pipeline.sh
```

This will:
- Run generation + uncertainty estimation on SciQ and TriviaQA
- Compute uncertainty metrics across methods
- Generate correctness labels

Outputs will be saved to `results/` and `output_dir/`.

---

## 📊 Replicated Experiments

| Table/Figure | Status | Location |
|--------------|--------|----------|
| Table 1: SciQ Accuracy            | ✅ | `uncertainty.py`
| Table 2: TriviaQA Accuracy       | ✅ | `uncertainty.py`
| Table 3/4: AUC / Correlation     | ✅ | `uncertainty.py`
| Figure 5: Uncertainty Distributions | ⚠️ (optional script) | TBD
| Figure 6: Token Entropy & Importance | ⚠️ (WIP visualization) | TBD

---

## 👥 Authorship & Contributions

**Core Pipeline & Code:** Kwanhee Lee

**Original Methodology Reference:**  
Duan et al., *"Shifting Attention to Relevance: A Framework for Evaluating Semantic Uncertainty in LLMs"* (2023)

> Future contributions (e.g., new datasets, entropy variants, visualization tools) are welcome and may be reflected in updated arXiv or journal versions.

---

## 📝 arXiv Posting Plan

- v1: Solo author submission with AWARE methodology + SAR replication
- v2+: Open to collaborators (metrics, interpretability, applications)
- Authorship will reflect meaningful contributions with prior consent

---

## 📮 Contact

Interested in collaborating, reusing, or building on this work?

- Email: gigantul@korea.ac.kr
- GitHub: https://github.com/gigantul/endogenous_attention_sar

---

## 📚 Citation

BibTeX and full citation will be included after the first arXiv upload. Stay tuned!
