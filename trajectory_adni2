#!/usr/bin/env python3
"""
trajectory_adni2.py
Feedback-round convergence analysis on the ADNI2 cohort, replicating the
same protocol as trajectory_adni1.py (Sec. 3, "Sample-efficient
convergence", Figure 1a of the paper).

ADNI2 diagnostic labels are mapped to the three-class scheme used
throughout the paper: SMC -> CN; EMCI, LMCI, MCI -> MCI; AD, CN unchanged.

Restricted to intra-session feedback only: starting from the initial
Eq.1 ranking, relevance judgments (Eq.3) and re-ranking (Eq.2) are applied
repeatedly for up to 4 rounds within a single query, accumulating relevant/
non-relevant judgments round over round. This isolates within-query
convergence speed, separately from the inter-session (cross-query) update
evaluated in relevance_feedback_adni2.py.

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

FEAT_CSV         = 'adni2_validation/features/full_features.csv'
MODEL_PATH       = 'features/best_model.joblib'
OUT_RF           = 'adni2_validation/features/trajectory_results_adni2.csv'

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
label_map = {'CN': 'CN', 'SMC': 'CN', 'EMCI': 'MCI', 'LMCI': 'MCI', 'MCI': 'MCI', 'AD': 'AD'}
df['Target_mapped'] = df['Target'].map(label_map)
print(f"  {len(df)} rows x {len(df.columns)} cols")

X      = df[FEAT_COLS].values.astype(float)
y      = df['Target_mapped'].values
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

# -- 4. Query selection -------------------------------------------------------------
print("\nSTEP 4: Select 25 queries")
rng = np.random.default_rng(SEED)
query_idxs = []
for cls, n in QUERY_DIST.items():
    cls_mask    = (y == cls)
    unique_rids = np.unique(groups[cls_mask])
    chosen_rids = rng.choice(unique_rids, size=n, replace=False)
    for rid in chosen_rids:
        cands = np.where(cls_mask & (groups == rid))[0]
        query_idxs.append(int(cands[0]))
ql = [y[i] for i in query_idxs]
print(f"  AD={ql.count('AD')}, MCI={ql.count('MCI')}, CN={ql.count('CN')}")
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


# -- 6. Main loop: multi-round intra-session feedback -------------------------
print("\nSTEP 5: Relevance Feedback Trajectory (0-4 rounds, intra-session only)")
print("-" * 65)

MAX_ROUNDS = 4
records = []

for qi, q_idx in enumerate(query_idxs):
    q_label = y[q_idx]
    q_pred  = oof_preds[q_idx]
    q_p     = float(oof_probs[q_idx, c2i[q_pred]])

    results_A = retrieve_eq1(q_idx)

    if len(results_A) < max(KS):
        print(f"[{qi+1:>2}] {q_label} skip: retrieved {len(results_A)} < {max(KS)}")
        continue

    labels_A = [r['label'] for r in results_A]
    p_by_round = {0: {k: prec_at_k(labels_A, q_label, k) for k in KS}}

    rel_cum, nonrel_cum = [], []
    current = results_A
    for rnd in range(1, MAX_ROUNDS + 1):
        rel_r, nonrel_r = simulate_eq3(current, q_label, window=FEEDBACK_WINDOW, pt=PT)
        rel_cum    = list(set(rel_cum + rel_r))
        nonrel_cum = list(set(nonrel_cum + nonrel_r))
        current = rerank_eq2(results_A, q_p, rel_cum, nonrel_cum)
        labels_r = [r['label'] for r in current]
        p_by_round[rnd] = {k: prec_at_k(labels_r, q_label, k) for k in KS}

    row = {
        'query_no':     qi + 1,
        'query_idx':    q_idx,
        'query_label':  q_label,
        'query_pred':   q_pred,
        'correct_pred': int(q_pred == q_label),
        'n_retrieved':  len(results_A),
    }
    for rnd in range(0, MAX_ROUNDS + 1):
        for k in KS:
            row[f'p{k}_r{rnd}'] = round(p_by_round[rnd][k], 4)
    records.append(row)

    corr = 'O' if q_pred == q_label else 'X'
    traj = " -> ".join(f"{p_by_round[r][10]:.2f}" for r in range(0, MAX_ROUNDS + 1))
    print(f"[{qi+1:>2}] {q_label:>3}(pred:{q_pred}{corr})  P@10 trajectory: {traj}")

# -- 7. Results -------------------------------------------------------------
res_df = pd.DataFrame(records)
res_df.to_csv(OUT_RF, index=False)

print()
print("=" * 65)
print(f"P@10 TRAJECTORY BY FEEDBACK ROUND ({len(res_df)} queries, feedback_window={FEEDBACK_WINDOW})")
print(f"{'Round':>6} | {'P@10':>8} | {'P@20':>8} | {'P@30':>8} | {'P@40':>8} | {'P@50':>8}")
print("-" * 65)
for rnd in range(0, MAX_ROUNDS + 1):
    vals = [float(res_df[f'p{k}_r{rnd}'].mean()) for k in KS]
    print(f"{rnd:>6} | " + " | ".join(f"{v:>8.4f}" for v in vals))

print(f"\nQuery SVM accuracy: {res_df['correct_pred'].sum()}/{len(res_df)} "
      f"({res_df['correct_pred'].mean()*100:.1f}%)")

print(f"\nSaved: {OUT_RF}")
print("Done.")
