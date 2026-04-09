"""
bias/statistical.py
Statistical validation of bias findings:
Chi-square, Z-test, Bootstrap CIs, Effect sizes (Cohen's h, Cramér's V).
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm
from statsmodels.stats.multitest import multipletests
from .metrics import compute_fairness_metrics

def cohens_h(p1, p2):
    """Effect size for the difference between two proportions."""
    # Bound between 0.001 and 0.999 to avoid math errors with arcsin
    p1 = max(1e-5, min(1-1e-5, p1))
    p2 = max(1e-5, min(1-1e-5, p2))
    h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
    return abs(h)

def cramers_v(confusion_matrix):
    """Effect size for chi-square test."""
    chi2 = chi2_contingency(confusion_matrix, correction=False)[0]
    n = confusion_matrix.sum()
    if n == 0: return 0.0
    r, k = confusion_matrix.shape
    min_dim = min(k-1, r-1)
    if min_dim == 0: return 0.0
    return np.sqrt(chi2 / (n * min_dim))

def z_test_proportions(count1, nobs1, count2, nobs2):
    """Two-proportion Z-test."""
    if nobs1 == 0 or nobs2 == 0:
        return 1.0 # not significant
    p1 = count1 / nobs1
    p2 = count2 / nobs2
    p_pool = (count1 + count2) / (nobs1 + nobs2)
    
    # Handle zero variance
    if p_pool == 0 or p_pool == 1:
        return 1.0
        
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / nobs1 + 1 / nobs2))
    z = (p1 - p2) / se
    p_val = 2 * (1 - norm.cdf(abs(z)))
    return p_val

def bootstrap_ci(y_true, y_pred_all, subgroup_mask, n_resamples=1000):
    """Bootstrap CI for DPD."""
    y_true = np.array(y_true)
    y_pred_all = np.array(y_pred_all)
    subgroup_mask = np.array(subgroup_mask)
    
    dpds = []
    n = len(y_true)
    
    # Optimization: If subgroup size is very small, return wide default CI
    if np.sum(subgroup_mask) < 10:
        return (-1.0, 1.0)
        
    for _ in range(n_resamples):
        indices = np.random.choice(n, n, replace=True)
        # Re-evaluate on bootstrapped sample
        cur_y_true = y_true[indices]
        cur_y_pred = y_pred_all[indices]
        cur_mask = subgroup_mask[indices]
        
        ov_sel = np.mean(cur_y_pred)
        if np.sum(cur_mask) > 0:
            sg_sel = np.mean(cur_y_pred[cur_mask])
            dpds.append(sg_sel - ov_sel)
            
    if not dpds:
        return (-1.0, 1.0)
        
    lower = np.percentile(dpds, 2.5)
    upper = np.percentile(dpds, 97.5)
    return (lower, upper)
    
def fdr_correction(p_values):
    """Benjamini-Hochberg FDR correction."""
    if not p_values:
        return []
    # multipletests requires 1d float array
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    return pvals_corrected.tolist()
