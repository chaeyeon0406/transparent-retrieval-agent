#!/usr/bin/env python3
"""
Classification & Retrieval Pipeline
Replication of Agarwal & Mostafa (2011), Sec. 2-3 of the paper

Phases:
  1. Feature fusion        : image features (DCT+DWT+LBP) + text features (Age, MMSE, CDR)
  2. Dimensionality reduction : TruncatedSVD (k selected via GridSearch)
  3. Classification        : SVM with RBF kernel, 10-fold StratifiedGroupKFold
                             GroupKFold prevents subject leakage (same patient across folds)
  4. Retrieval             : R = p x Euclidean_distance  (paper Eq. 1)
                             p = out-of-fold SVM class probability (no data leakage)
                             Same-subject scans excluded from the database (RID-level)
  5. Evaluation            : Average Precision at cutoffs 10/20/30/40/50
                             25 queries (AD=9, MCI=8, CN=8), one scan per subject

Results saved to: features/retrieval_results.csv
                  features/best_model.joblib
Paper baseline  : Precision@10 = 0.816 (with classification)
"""

import warnings
import joblib

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv('features/full_features.csv')
df = df.dropna().reset_index(drop=True)
print(f"Dataset: {len(df)} rows, {df['RID'].nunique()} unique subjects")
print(f"Class distribution:\n{df['Target'].value_counts()}\n")

img_cols  = [c for c in df.columns if c.startswith('img_')]
text_cols = ['AGE', 'MMSCORE', 'CDGLOBAL']

X      = df[img_cols + text_cols].values
y      = df['Target'].values
groups = df['RID'].values  # used to prevent subject leakage in CV


# ---------------------------------------------------------------------------
# Step 1: GridSearch - SVD dimensionality + SVM hyperparameters
# StratifiedGroupKFold ensures:
#   - class ratios are preserved per fold (Stratified)
#   - scans from the same subject stay in the same fold (Group)
# Note: class_weight not used - not mentioned in paper
# ---------------------------------------------------------------------------
print("=== GridSearch (StratifiedGroupKFold, ~30 min) ===")

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svd',    TruncatedSVD(random_state=42)),
    ('svm',    SVC(kernel='rbf', probability=True, random_state=42))
])

param_grid = {
    'svd__n_components': [10, 20, 30, 40, 50],
    'svm__C':            [0.1, 1, 10, 100],
    'svm__gamma':        ['scale', 0.001, 0.01, 0.1]
}

cv   = StratifiedGroupKFold(n_splits=10)
grid = GridSearchCV(pipe, param_grid, cv=cv, scoring='accuracy',
                    n_jobs=8, verbose=1)
grid.fit(X, y, groups=groups)

print(f"\nBest params : {grid.best_params_}")
print(f"Best CV accuracy: {grid.best_score_ * 100:.2f}%  (paper: 86.7%)\n")

# Save best hyperparameters for downstream scripts
joblib.dump({'params': grid.best_params_},
            'features/best_model.joblib')
print("Saved: features/best_model.joblib")


# ---------------------------------------------------------------------------
# Step 2: Out-of-fold predictions (no data leakage)
# OOF probabilities are used for retrieval ranking so that classification
# confidence is not inflated by training-set membership.
# The full pipeline is also fitted on all data to obtain the shared SVD
# feature space used for Euclidean distance computation.
# ---------------------------------------------------------------------------
params = grid.best_params_

pipe_best = Pipeline([
    ('scaler', StandardScaler()),
    ('svd',    TruncatedSVD(n_components=params['svd__n_components'],
                            random_state=42)),
    ('svm',    SVC(kernel='rbf',
                   C=params['svm__C'],
                   gamma=params['svm__gamma'],
                   probability=True, random_state=42))
])

print("=== Computing out-of-fold probabilities ===")
oof_probs = cross_val_predict(pipe_best, X, y, cv=cv, groups=groups,
                               method='predict_proba', n_jobs=8)

pipe_best.fit(X, y)
svm_classes = pipe_best.named_steps['svm'].classes_
oof_preds   = svm_classes[np.argmax(oof_probs, axis=1)]
X_svd       = pipe_best[:-1].transform(X)
c2i         = {c: i for i, c in enumerate(svm_classes)}

oof_acc = np.mean(oof_preds == y)
print(f"OOF accuracy: {oof_acc * 100:.2f}%  (paper: 86.7%)\n")


# ---------------------------------------------------------------------------
# Step 3: Retrieval ranking (paper Eq. 1)
# R = p x sqrt( Sum (x_ij - y_ij)^2 )
# Lower R = more similar; search restricted to same predicted class.
# Scans from the same subject (RID) are excluded from the database to
# prevent subject-level leakage in the retrieval evaluation.
# ---------------------------------------------------------------------------
def retrieval_ranks(q_idx):
    q_feat = X_svd[q_idx]
    q_pred = oof_preds[q_idx]
    q_rid  = groups[q_idx]
    q_p    = float(oof_probs[q_idx, c2i[q_pred]])

    # Exclude same subject and restrict to same predicted class
    mask = (groups != q_rid) & (oof_preds == q_pred)
    pos  = np.where(mask)[0]

    dists = np.linalg.norm(X_svd[pos] - q_feat, axis=1)
    R     = q_p * dists
    order = np.argsort(R)

    return [(pos[i], float(R[i]), bool(y[pos[i]] == y[q_idx]))
            for i in order]


def precision_at_k(ranks, k):
    return sum(1 for _, _, rel in ranks[:k] if rel) / k


# ---------------------------------------------------------------------------
# Step 4: Evaluate on 25 queries (paper setup: AD=9, MCI=8, CN=8)
# One scan per subject; subjects sampled without replacement.
# Uses the same query selection as relevance_feedback.py for consistency.
# ---------------------------------------------------------------------------
print("=== Retrieval Evaluation (25 queries) ===")

rng = np.random.default_rng(42)
query_idxs = []
for cls, n in [('AD', 9), ('MCI', 8), ('CN', 8)]:
    cls_mask    = (y == cls)
    unique_rids = np.unique(groups[cls_mask])
    chosen_rids = rng.choice(unique_rids, size=n, replace=False)
    for rid in chosen_rids:
        cands = np.where(cls_mask & (groups == rid))[0]
        query_idxs.append(int(cands[0]))

cutoffs = [10, 20, 30, 40, 50]
results = {k: [] for k in cutoffs}

for q in query_idxs:
    ranks = retrieval_ranks(q)
    if len(ranks) < max(cutoffs):
        continue
    for k in cutoffs:
        results[k].append(precision_at_k(ranks, k))

# Paper baseline (Table 1: with classification)
paper = {10: 0.816, 20: 0.783, 30: 0.776, 40: 0.750, 50: 0.730}

print(f"\n{'Cutoff':<8} {'Ours':>8} {'Paper':>8}")
print("-" * 26)
for k in cutoffs:
    our = np.mean(results[k]) if results[k] else 0
    print(f"at {k:<5}  {our:.3f}    {paper[k]:.3f}")

# Save results
out_df = pd.DataFrame({
    'cutoff':          cutoffs,
    'our_precision':   [np.mean(results[k]) if results[k] else 0 for k in cutoffs],
    'paper_precision': [paper[k] for k in cutoffs]
})
out_df.to_csv('features/retrieval_results.csv', index=False)
print("\nSaved: features/retrieval_results.csv")
