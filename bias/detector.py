"""
bias/detector.py
Subgroup discovery engine.
Single-attribute & Intersectional.
Min subgroup size = 30.
"""

import numpy as np
import pandas as pd
import itertools
import logging
from .metrics import compute_fairness_metrics, compare_metrics
from .statistical import cohens_h, z_test_proportions, bootstrap_ci, fdr_correction
from .ranker import calculate_bss, assign_tier, assign_legal_risk_tier

logger = logging.getLogger(__name__)

def discover_subgroups(X_test, y_true, y_pred, model, protected_cols, threshold_dpd=0.1):
    """
    Main engine to run all bias detection logic.
    """
    results = []
    
    # Pre-computation of overall metrics
    overall_metrics = compute_fairness_metrics(y_true, y_pred, None) # proba optional here
    n_total = len(y_true)
    ov_sel = overall_metrics.get("selection_rate", 0)
    
    # 1. Single Attribute
    for col in protected_cols:
        if col not in X_test.columns:
            continue
            
        unique_vals = X_test[col].unique()
        for val in unique_vals:
            mask = (X_test[col] == val)
            sg_n = mask.sum()
            
            if sg_n < 30:
                logger.warning(f"Skipping subgroup {col}={val} (n={sg_n} < 30)")
                continue
                
            y_true_sg = y_true[mask]
            y_pred_sg = y_pred[mask]
            
            # Base metrics
            sg_metrics = compute_fairness_metrics(y_true_sg, y_pred_sg)
            disp = compare_metrics(sg_metrics, overall_metrics)
            
            if not disp:
                continue
                
            # Wire the threshold: only include subgroups where DPD is meaningful
            if abs(disp.get("DPD", 0)) < threshold_dpd:
                continue
                
            # Stats
            p1 = sg_metrics.get("selection_rate", 0)
            p2 = ov_sel
            h = cohens_h(p1, p2)
            z_p = z_test_proportions(mask.sum() * p1, mask.sum(), n_total * p2, n_total)
            ci = bootstrap_ci(y_true, y_pred, mask, n_resamples=200) # Reduced for performance
            
            # We defer FDR correction to end, store raw p-value
            
            res = {
                "Type": "Single",
                "Attribute": col,
                "Value": str(val).split('.')[-1] if not isinstance(val, str) else val, # Quick clean
                "Subgroup Name": f"{col} = {val}",
                "n": int(sg_n),
                "DPD": disp.get("DPD", 0),
                "DIR": disp.get("DIR", 1),
                "EOD": disp.get("EOD", 0),
                "FNR_Disparity": disp.get("FNR Disparity", 0),
                "PPV_Gap": disp.get("Predictive Parity Gap", 0),
                "Calibration_Gap": disp.get("Calibration Error Gap", 0),
                "Effect_Size_h": h,
                "p_val_raw": z_p,
                "CI_lower": ci[0],
                "CI_upper": ci[1]
            }
            results.append(res)
            
    # 2. Intersectional (2-way)
    if len(protected_cols) >= 2:
        for col1, col2 in itertools.combinations(protected_cols, 2):
            if col1 not in X_test.columns or col2 not in X_test.columns:
                continue
                
            # Group by both
            groups = X_test.groupby([col1, col2]).size()
            for (val1, val2), count in groups.items():
                if count < 30:
                    continue
                    
                mask = (X_test[col1] == val1) & (X_test[col2] == val2)
                y_true_sg = y_true[mask]
                y_pred_sg = y_pred[mask]
                
                sg_metrics = compute_fairness_metrics(y_true_sg, y_pred_sg)
                disp = compare_metrics(sg_metrics, overall_metrics)
                
                if not disp:
                    continue
                    
                # Wire the threshold for intersectional too
                if abs(disp.get("DPD", 0)) < threshold_dpd:
                    continue
                    
                p1 = sg_metrics.get("selection_rate", 0)
                p2 = ov_sel
                h = cohens_h(p1, p2)
                z_p = z_test_proportions(mask.sum() * p1, mask.sum(), n_total * p2, n_total)
                ci = bootstrap_ci(y_true, y_pred, mask, n_resamples=200)
                
                res = {
                    "Type": "Intersectional",
                    "Attribute": f"{col1} x {col2}",
                    "Value": f"{val1} & {val2}",
                    "Subgroup Name": f"{col1}={val1} & {col2}={val2}",
                    "n": int(count),
                    "DPD": disp.get("DPD", 0),
                    "DIR": disp.get("DIR", 1),
                    "EOD": disp.get("EOD", 0),
                    "FNR_Disparity": disp.get("FNR Disparity", 0),
                    "PPV_Gap": disp.get("Predictive Parity Gap", 0),
                    "Calibration_Gap": disp.get("Calibration Error Gap", 0),
                    "Effect_Size_h": h,
                    "p_val_raw": z_p,
                    "CI_lower": ci[0],
                    "CI_upper": ci[1]
                }
                results.append(res)
                
    # 3. Apply FDR Correction across all findings
    if results:
        p_vals = [r["p_val_raw"] for r in results]
        p_corr = fdr_correction(p_vals)
        
        # 4. Rank
        for i, r in enumerate(results):
            r["p_val_corrected"] = p_corr[i]
            r["BSS"] = calculate_bss(r["DPD"], r["Effect_Size_h"], r["p_val_corrected"], r["n"] / n_total, r["DIR"])
            r["Priority"] = assign_tier(r["BSS"])
            r["Legal_Risk"] = assign_legal_risk_tier(r["BSS"], r["DPD"], r["DIR"])
            
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("BSS", ascending=False).reset_index(drop=True)
        # Add rank
        df.insert(0, "Rank", range(1, len(df) + 1))
        
    return df
