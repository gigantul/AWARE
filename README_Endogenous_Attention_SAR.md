# Endogenous Attention-Weighted SAR: A Clean and Efficient Framework for Semantic-Aware Uncertainty in LLMs

## 📌 Summary
This repository re-implements and extends the core methodology of "Shifting Attention to Relevance (SAR)" by Duan et al., offering a:

- ✅ **Modular, memory-efficient SAR pipeline**
- ✅ **Endogenous attention-weighted token importance** (instead of SBERT)
- ✅ **Native support for multiple uncertainty measures** (SAR, PE, SE, LN-PE, etc.)
- ✅ Clean integration with HuggingFace and minimal external dependencies

> This version serves as a reproducible and extensible base for research in uncertainty estimation, semantic relevance, and token-level confidence in LLM generations.

---

## 🔬 Key Contributions
- **Main Pipeline** (`main_pipeline.py`): Unified, batch-based processing for any dataset (SciQ, TriviaQA, CoQA)
- **Plug-in Uncertainty Modules**: Supports SAR, token-SAR, sentence-SAR, PE, SE, semantic entropy
- **Replaces SBERT with Endogenous Attention Weights**: Token importance derived from model internals (attention heads), increasing interpretability and efficiency
- **Reproducible Figures & Tables**: Scripts replicate Duan et al.'s Table 1–4, Figure 5–6

---

## 📁 Structure

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
│   └── run_sar_pipeline.sh
└── results/  # output CSVs + pickles
```

---

## 🚀 How to Run

```bash
bash scripts/run_sar_pipeline.sh
```
This will:
- Run SAR-style generation and analysis for `sciq` and `triviaqa`
- Compute uncertainty metrics
- Generate correctness labels

Output files (CSV + pickle) are saved under `results/` and `output_dir/`.

---

## 📊 Replicated Tables and Figures

| Table/Figure | Covered? | Location |
|--------------|----------|----------|
| Table 1 (SciQ Accuracy)   | ✅ | `uncertainty.py` output
| Table 2 (TriviaQA)        | ✅ | `uncertainty.py`
| Table 3/4 (AUC/Correlation) | ✅ | `uncertainty.py`
| Figure 5 (Distribution)   | ⚠️ (script optional) | TBD
| Figure 6 (Token entropy/importance) | ⚠️ (visual script WIP) | TBD

---

## 👥 Authorship & Contributions

**Original Code & Pipeline:** [Your Name Here]  
**Paper Reference:** Duan et al., "Shifting Attention to Relevance: A Framework for Evaluating Semantic Uncertainty in LLMs" (2023)

If this code is extended in future work (e.g., additional datasets, interpretability tools, or visualizations), **new contributors may be added to later arXiv versions or journal submissions**.

---

## 📝 arXiv Posting Plan
- Post v1 as sole author
- Invite collaborators to contribute to v2 (e.g., new metrics, datasets, visualization)
- Update authorship with consent upon arXiv resubmission

---

## 📮 Contact
Interested in collaborating or reusing this? Feel free to reach out.

- Email: your.email@example.com
- GitHub: [your-handle]

---

## 📚 Citation
BibTeX coming after arXiv upload. Stay tuned.
