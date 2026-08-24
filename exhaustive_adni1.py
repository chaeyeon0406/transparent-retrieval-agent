#!/usr/bin/env python3
"""
exhaustive_adni1.py
Exhaustive re-evaluation of every misclassified AD query (Sec. 4,
Discussion of the paper).

Unlike relevance_feedback_adni1.py, which evaluates a fixed 25-query
sample, this script evaluates every AD scan misclassified as MCI by the
classifier -- i.e. the full population of misclassified AD queries, not
a sample. This removes the small-sample variance of the standard 25-query
protocol and was used to test whether the cross-cohort disparity observed
in that protocol reflects a genuine cohort difference or a sampling
artifact.

Each query is evaluated as an independent session: the inter-session
class profile is reset before every query, so no relevance judgments
leak across queries (see the isolation fix noted inline below).

Feedback is applied only to the top-FEEDBACK_WINDOW retrieved results
(the paper's UI presents results as an 8-image matrix; we use a window
of 20 as a practical compromise, matching Sec. 2.3 of the paper, W=20).
"""

import warnings, joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

FEAT_CSV         = 'features/full_features.csv'
MODEL_PATH       = 'features/best_model.joblib'
OUT_RF           = 'features/exhaustive_misclassified_results_adni1.csv'

IMG_COLS         = [f'img_{i}' for i in range(72)]
TEXT_COLS        = ['AGE', 'MMSCORE', 'CDGLOBAL']
FEAT_COLS        = IMG_COLS + TEXT_COLS

QUERY_DIST       = {'AD': 9, 'MCI': 8, 'CN': 8}
KS               = [10, 20, 30, 40, 50]
PT               = 0.5    # Eq.3 relevance threshold
FEEDBACK_WINDOW  = 20     # number of top results shown to the simulated user (Sec. 2.3, W=20)
SEED             = 42

PAPER_REF = {
    10: (0.816, 0.895), 20: (0.783, 0.889), 30: (0.776, 0.883),
    40: (0.750, 0.880), 50: (0.730, 0.875),
}

# -- 1. Load -------------------------------------------------------------
print("=" * 65)
print("STEP 1: Load features")
df = pd.read_csv(FEAT_CSV).dropna().reset_index(drop=True)
print(f"  {len(df)} rows x {len(df.columns)} cols")

X      = df[FEAT_COLS].values.astype(float)
y      = df['Target'].values
groups = df['RID'].values
print(f"  Labels: { {c: int((y==c).sum()) for c in ['AD','MCI','CN']} }")

# -- 2. Params -------------------------------------------------------------
print("\nSTEP 2: Load params")
saved  = joblib.load(MODEL_PATH)
params = saved['params']
print(f"  {params}")

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svd',    TruncatedSVD(n_components=params['svd__n_components'], random_state=SEED)),
    ('svm',    SVC(kernel='rbf', C=params['svm__C'], gamma=params['svm__gamma'],
                   probability=True, random_state=SEED)),
])

# -- 3. Out-of-fold predictions -------------------------------------------
print("\nSTEP 3: OOF probabilities (StratifiedGroupKFold n=10)")
cv = StratifiedGroupKFold(n_splits=10)
oof_probs = cross_val_predict(
    pipe, X, y, cv=cv, groups=groups, method='predict_proba', n_jobs=8
)
pipe.fit(X, y)
svm_classes = pipe.named_steps['svm'].classes_
oof_preds   = svm_classes[np.argmax(oof_probs, axis=1)]
X_svd       = pipe[:-1].transform(X)
c2i         = {c: i for i, c in enumerate(svm_classes)}
print(f"  OOF Accuracy: {np.mean(oof_preds == y)*100:.2f}%")
mci_pool_size = int((oof_preds == 'MCI').sum())
ad_in_mci_pool = int(((oof_preds == 'MCI') & (y == 'AD')).sum())
contam_rate = ad_in_mci_pool / mci_pool_size if mci_pool_size > 0 else 0.0
print(f'  MCI-predicted pool size: {mci_pool_size}, true-AD contamination: {ad_in_mci_pool} ({contam_rate*100:.2f}%)')

# -- 4. Query selection (EXHAUSTIVE): all true-AD scans misclassified as MCI --
print("\nSTEP 4: Select ALL misclassified-AD queries (exhaustive post-hoc)")
query_idxs = [int(i) for i in np.where((y == 'AD') & (oof_preds == 'MCI'))[0]]
ql = [y[i] for i in query_idxs]
print(f"  Misclassified AD-as-MCI queries: {len(query_idxs)}")
print(f"  Feedback window: top-{FEEDBACK_WINDOW} results (PT={PT})")

# -- 5. Helpers -------------------------------------------------------------

def prec_at_k(labels, q_label, k):
    n = min(k, len(labels))
    return float(np.mean(np.array(labels[:n]) == q_label))


def retrieve_eq1(q_idx):
    """
    Eq.1: R = p x dist
    Same-subject scans are excluded; search is restricted to the
    predicted class.
    """
    q_feat = X_svd[q_idx]
    q_pred = oof_preds[q_idx]
    q_rid  = groups[q_idx]
    q_p    = float(oof_probs[q_idx, c2i[q_pred]])

    mask = (groups != q_rid) & (oof_preds == q_pred)
    pos  = np.where(mask)[0]
    if len(pos) == 0:
        return []

    dists = np.linalg.norm(X_svd[pos] - q_feat, axis=1)
    order = np.argsort(q_p * dists)

    return [
        {'idx': int(pos[i]), 'dist': float(dists[i]),
         'label': y[pos[i]], 'R': float(q_p * dists[i])}
        for i in order
    ]


def simulate_eq3(results, q_label, window=FEEDBACK_WINDOW, pt=PT):
    """
    Simulated user feedback (paper Eq.3 + Sec. 3 user model).

    Feedback is applied only to the top `window` results (Sec. 2.3):
    applying it to the full result set would trivially reach P@10=1.0.

    Rule 1: label != q_label                      -> non-relevant
    Rule 2: label == q_label, pi_norm >= pt        -> relevant
             pi_norm = exp(-(d-mu)^2/(2*sig^2))
    mu, sig = mean/std of distances within the window
    """
    subset = results[:window]
    if not subset:
        return [], []

    dists = np.array([r['dist'] for r in subset])
    mu    = float(np.mean(dists))
    sig   = float(np.std(dists)) + 1e-9

    rel_idxs, nonrel_idxs = [], []
    for r in subset:
        if r['label'] != q_label:
            nonrel_idxs.append(r['idx'])
        else:
            pi_norm = float(np.exp(-(r['dist'] - mu)**2 / (2 * sig**2)))
            if pi_norm >= pt:
                rel_idxs.append(r['idx'])
            else:
                nonrel_idxs.append(r['idx'])
    return rel_idxs, nonrel_idxs


def rerank_eq2(results, q_p, rel_idxs, nonrel_idxs):
    """
    Eq.2: R_k = p x D_k x min{d(k,rel)} / max{d(k,nonrel)}
    If either rel_idxs or nonrel_idxs is empty, the original order is kept.
    """
    if not rel_idxs or not nonrel_idxs or not results:
        return results

    rel_feats    = X_svd[np.array(rel_idxs)]
    nonrel_feats = X_svd[np.array(nonrel_idxs)]

    reranked = []
    for r in results:
        x       = X_svd[r['idx']]
        min_dik = float(np.linalg.norm(x - rel_feats,    axis=1).min())
        max_djk = float(np.linalg.norm(x - nonrel_feats, axis=1).max())
        max_djk = max(max_djk, 1e-9)
        R_new   = q_p * r['dist'] * (min_dik / max_djk)
        reranked.append({**r, 'R': R_new})

    reranked.sort(key=lambda x: x['R'])
    return reranked


# -- 6. Main loop -------------------------------------------------------------
print("\nSTEP 5: Relevance Feedback evaluation")
print("-" * 65)

records = []

for qi, q_idx in enumerate(query_idxs):
    q_label = y[q_idx]
    q_pred  = oof_preds[q_idx]
    q_p     = float(oof_probs[q_idx, c2i[q_pred]])

    # (A) Initial retrieval: Eq.1
    results_A = retrieve_eq1(q_idx)

    if len(results_A) < max(KS):
        print(f"[{qi+1:>2}] {q_label} skip: retrieved {len(results_A)} < {max(KS)}")
        continue

    labels_A = [r['label'] for r in results_A]
    p_before = {k: prec_at_k(labels_A, q_label, k) for k in KS}

    # (B) Inter-session re-ranking
    # Each query is isolated as an independent session: the class profile
    # is reset before every query so no relevance judgments leak across
    # queries (inter == before by construction; this is intentional).
    class_profile = {c: [] for c in svm_classes}
    prof = class_profile.get(q_label, [])

    if len(prof) >= 1:
        nonrel_inter = [r['idx'] for r in results_A if r['label'] != q_label]
        if not nonrel_inter:
            nonrel_inter = [results_A[-1]['idx']]
        results_B = rerank_eq2(results_A, q_p, prof, nonrel_inter)
    else:
        results_B = results_A

    labels_B = [r['label'] for r in results_B]
    p_inter  = {k: prec_at_k(labels_B, q_label, k) for k in KS}

    # (C) Eq.3 user simulation -- applied only to the top FEEDBACK_WINDOW results
    rel_idxs, nonrel_idxs = simulate_eq3(results_B, q_label,
                                          window=FEEDBACK_WINDOW, pt=PT)

    # (D) Intra-session re-ranking: Eq.2 (re-ranks the full results_B set)
    results_C = rerank_eq2(results_B, q_p, rel_idxs, nonrel_idxs)

    labels_C = [r['label'] for r in results_C]
    p_intra  = {k: prec_at_k(labels_C, q_label, k) for k in KS}

    # Update class profile (only relevant items within the window accumulate)
    class_profile[q_label].extend(rel_idxs)

    row = {
        'query_no':     qi + 1,
        'query_idx':    q_idx,
        'query_label':  q_label,
        'query_pred':   q_pred,
        'correct_pred': int(q_pred == q_label),
        'n_retrieved':  len(results_A),
        'n_rel':        len(rel_idxs),
        'n_nonrel':     len(nonrel_idxs),
        'profile_size': len(class_profile[q_label]),
    }
    for k in KS:
        row[f'p{k}_before'] = round(p_before[k], 4)
        row[f'p{k}_inter']  = round(p_inter[k],  4)
        row[f'p{k}_intra']  = round(p_intra[k],  4)
    records.append(row)

    corr = 'O' if q_pred == q_label else 'X'
    print(
        f"[{qi+1:>2}] {q_label:>3}(pred:{q_pred}{corr})  "
        f"P@10 {p_before[10]:.3f}->{p_inter[10]:.3f}->{p_intra[10]:.3f}  "
        f"[rel={len(rel_idxs):>2} nonrel={len(nonrel_idxs):>2} "
        f"profile={len(class_profile[q_label]):>3}]"
    )

# -- 7. Results -------------------------------------------------------------
res_df = pd.DataFrame(records)
res_df.to_csv(OUT_RF, index=False)

print()
print("=" * 65)
print(f"AVERAGE PRECISION (exhaustive misclassified-AD queries, feedback_window={FEEDBACK_WINDOW})")
print(f"{'K':>5} | {'Before':>8} | {'Inter':>8} | {'Intra':>8} | Paper(w/o) | Paper(+RF)")
print("-" * 65)
for k in KS:
    b        = float(res_df[f'p{k}_before'].mean())
    i        = float(res_df[f'p{k}_inter'].mean())
    a        = float(res_df[f'p{k}_intra'].mean())
    pno, prf = PAPER_REF[k]
    print(f"{k:>5} | {b:>8.4f} | {i:>8.4f} | {a:>8.4f} | {pno:>10.4f} | {prf:>10.4f}")

print()
print("Per-class P@10 (intra):")
for cls in ['AD', 'MCI', 'CN']:
    sub = res_df[res_df['query_label'] == cls]
    if len(sub) == 0:
        continue
    print(f"  {cls} (n={len(sub)}):  "
          f"before={sub['p10_before'].mean():.4f}  "
          f"inter={sub['p10_inter'].mean():.4f}  "
          f"intra={sub['p10_intra'].mean():.4f}")

print(f"\nQuery SVM accuracy: "
      f"{res_df['correct_pred'].sum()}/{len(res_df)} "
      f"({res_df['correct_pred'].mean()*100:.1f}%)")
print(f"Avg rel/query:    {res_df['n_rel'].mean():.1f}")
print(f"Avg nonrel/query: {res_df['n_nonrel'].mean():.1f}")

print(f"\nSaved: {OUT_RF}")
print("Done.")
