# Transparent Retrieval Agent for Alzheimer's Disease MRI

Code accompanying an anonymous submission to the AIM (Agentic Intelligence for Medical Imaging and Multimodal Clinical Data) workshop at NeurIPS 2026.

This repository reproduces the retrieval, relevance-feedback, and evaluation results reported in the paper: a classical content-based image retrieval (CBIR) system for Alzheimer's disease (AD) diagnosis, reframed as a transparent, interactive retrieval agent with a mathematically specified Observe-Update-Act loop.

## Requirements

```
pandas
numpy
scikit-learn
scipy
matplotlib
joblib
```

Install with:

```bash
pip install -r requirements.txt
```

## Data

This project uses the ADNI (Alzheimer's Disease Neuroimaging Initiative) dataset. ADNI data is **not included** in this repository and cannot be redistributed under the ADNI Data Use Agreement.

To obtain access:

1. Request access at [adni.loni.usc.edu/data-samples/access-data](https://adni.loni.usc.edu/data-samples/access-data/) (typically ~1 week for approval).
2. Download T1-weighted MRI and associated clinical variables (Age, MMSE, CDR-Global) for the ADNI1 and ADNI2 cohorts.
3. Extract the 75-dimensional fused feature vector (25 DCT + 21 DWT + 26 LBP image-texture features, plus 3 clinical variables) as described in Sec. 2.1 of the paper, and format as `img_0`...`img_71`, `AGE`, `MMSCORE`, `CDGLOBAL`, `RID`, `Target` columns.

## Repository structure

| Script | Produces |
|---|---|
| `svd_svm.py` | Grid search over SVD dimensionality and SVM hyperparameters (C, gamma); fusion classification accuracy (Sec. 3) |
| `relevance_feedback.py`, `relevance_feedback_adni2.py` | Retrieval precision before/after feedback, Table 1 |
| `trajectory_adni1.py`, `trajectory_adni2.py` | Feedback-round convergence, Figure 1(a) |
| `noise_adni1.py`, `noise_adni2.py` | Noise-robustness sweep, Figure 1(b) |
| `exhaustive_adni1.py`, `exhaustive_adni2.py` | Exhaustive re-evaluation of all misclassified AD queries (n=127, n=187), Sec. 4 |
| `stats_delta_analysis.py` | Mann-Whitney U test and bootstrap CI on per-query feedback gains, Sec. 4 |
| `make_figure1.py` | Renders Figure 1 from the trajectory/noise results above |
| `best_model.joblib` | Saved grid-search hyperparameters (no patient data; parameter values only) |

## Reproducing the paper's results

```bash
# 1. Hyperparameter selection + classification accuracy
python3 svd_svm.py

# 2. Retrieval precision (Table 1)
python3 relevance_feedback.py
python3 relevance_feedback_adni2.py

# 3. Feedback-round convergence (Figure 1a)
python3 trajectory_adni1.py
python3 trajectory_adni2.py

# 4. Noise robustness (Figure 1b)
python3 noise_adni1.py
python3 noise_adni2.py

# 5. Exhaustive misclassified-query analysis (Sec. 4)
python3 exhaustive_adni1.py
python3 exhaustive_adni2.py
python3 stats_delta_analysis.py

# 6. Regenerate Figure 1
python3 make_figure1.py
```

## Results summary

| Cohort | P@10 before to after feedback | Classification accuracy |
|---|---|---|
| ADNI1 (n=699) | 0.828 to 0.948 | 93.24% |
| ADNI2 (n=840) | 0.832 to 0.884 | 95.94% |

See the paper for full results, including noise-robustness, convergence, and failure-mode analysis.

## License

MIT
