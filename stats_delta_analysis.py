"""
Statistical significance analysis for the exhaustive misclassified-AD-query
re-evaluation (Sec. 4, Discussion of the NeurIPS AIM workshop paper).

Compares per-query feedback-associated P@10 gains (delta = intra - before)
between ADNI1 (n=127) and ADNI2 (n=187) using a Mann-Whitney U test on the
correctly-specified paired change score, plus a bootstrap CI on the
between-cohort difference in mean gain.

Inputs (produced by exhaustive_adni1.py / exhaustive_adni2.py):
  features/exhaustive_misclassified_results_adni1.csv
  features/exhaustive_misclassified_results_adni2.csv
"""
import pandas as pd
import numpy as np
from scipy import stats

FEAT1 = 'features/exhaustive_misclassified_results_adni1.csv'
FEAT2 = 'features/exhaustive_misclassified_results_adni2.csv'

df1 = pd.read_csv(FEAT1)
df2 = pd.read_csv(FEAT2)

# per-query change score: delta = P@10 after one feedback round - P@10 before
delta1 = (df1.p10_intra - df1.p10_before).values
delta2 = (df2.p10_intra - df2.p10_before).values

print(f"ADNI1 n={len(delta1)}, mean delta={delta1.mean():.4f}, median delta={np.median(delta1):.4f}")
print(f"ADNI2 n={len(delta2)}, mean delta={delta2.mean():.4f}, median delta={np.median(delta2):.4f}")

# primary test: do per-query gains differ between cohorts? (independent groups)
u_stat, p_val = stats.mannwhitneyu(delta1, delta2, alternative='two-sided')
print(f"\nMann-Whitney U test on delta (ADNI1 vs ADNI2 gains): U={u_stat:.1f}, p={p_val:.4f}")

# effect size: rank-biserial correlation
n1, n2 = len(delta1), len(delta2)
r_rb = 1 - (2 * u_stat) / (n1 * n2)
print(f"Rank-biserial correlation (effect size): {r_rb:.4f}")

# bootstrap 95% CI on the difference in mean delta between cohorts
# (resampled independently within each cohort; pairing already preserved
#  in the construction of each delta value)
rng = np.random.default_rng(42)
n_boot = 10000
diffs = np.empty(n_boot)
for b in range(n_boot):
    s1 = rng.choice(delta1, size=n1, replace=True)
    s2 = rng.choice(delta2, size=n2, replace=True)
    diffs[b] = s1.mean() - s2.mean()

obs_diff = delta1.mean() - delta2.mean()
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"\nMean delta difference (ADNI1 - ADNI2): {obs_diff:.4f}")
print(f"95% bootstrap CI for the difference: [{lo:.4f}, {hi:.4f}]")
print(f"CI excludes 0: {lo > 0 or hi < 0}")
