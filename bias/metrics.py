"""
bias/metrics.py
Vectorized fairness metrics computations:
DPD, EOD, DIR, Predictive Parity, Calibration Error, Selection Rate Diff, FNR Disparity.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def compute_fairness_metrics(y_true, y_pred, y_proba=None):
    """
    Computes base metrics for a given subset (subgroup or overall).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    n = len(y_true)
    if n == 0:
        return {}
        
    # Standard metrics
    selection_rate = np.mean(y_pred)
    
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    except ValueError:
        return {} # Handle edge case of one class only in subgroup
        
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0 # Predictive Parity
    
    res = {
        "size": n,
        "selection_rate": selection_rate,
        "tpr": tpr,
        "fpr": fpr,
        "fnr": fnr,
        "ppv": ppv,
    }
    
    # Expected Calibration Error (simplified, uniform bins)
    if y_proba is not None:
        y_proba = np.array(y_proba)
        # 10 bins
        bins = np.linspace(0, 1, 11)
        binned = np.digitize(y_proba, bins) - 1
        ece = 0.0
        for b in range(10):
            mask = (binned == b)
            if np.any(mask):
                acc = np.mean(y_true[mask])
                conf = np.mean(y_proba[mask])
                ece += np.mean(mask) * np.abs(acc - conf)
        res["ece"] = ece
        
    return res

def compare_metrics(subgroup_metrics, overall_metrics):
    """
    Computes disparity metrics comparing subgroup against overall dataset.
    """
    if "selection_rate" not in subgroup_metrics or "selection_rate" not in overall_metrics:
        return {}
        
    sg = subgroup_metrics
    ov = overall_metrics
    
    # 1. Demographic Parity Difference (DPD)
    dpd = sg["selection_rate"] - ov["selection_rate"]
    
    # 2. Equalized Odds Difference (EOD) - Max of TPR/FPR gap
    tpr_gap = sg.get("tpr", 0) - ov.get("tpr", 0)
    fpr_gap = sg.get("fpr", 0) - ov.get("fpr", 0)
    eod = max(abs(tpr_gap), abs(fpr_gap))
    
    # 3. Disparate Impact Ratio (DIR)
    # Handle div by zero
    if ov["selection_rate"] == 0:
        dir_val = 1.0 if sg["selection_rate"] == 0 else float('inf')
    else:
        dir_val = sg["selection_rate"] / ov["selection_rate"]
        
    # 4. Selection Rate Difference
    srd = dpd
    
    # 5. False Negative Rate Disparity
    fnr_disp = sg.get("fnr", 0) - ov.get("fnr", 0)
    
    # 6. Predictive Parity Gap (Gap in PPV)
    ppv_gap = sg.get("ppv", 0) - ov.get("ppv", 0)
    
    disparities = {
        "DPD": dpd,
        "EOD": eod,
        "DIR": dir_val,
        "Selection Rate Diff": srd,
        "FNR Disparity": fnr_disp,
        "Predictive Parity Gap": ppv_gap
    }
    
    if "ece" in sg and "ece" in ov:
        disparities["Calibration Error Gap"] = sg["ece"] - ov["ece"]
        
    return disparities
