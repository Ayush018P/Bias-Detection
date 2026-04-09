"""
bias/ranker.py
Severity Ranking module based on the SABPF formula.
"""

def calculate_bss(dpd, effect_size_h, p_val_corrected, subgroup_fraction, dir_val=None):
    """
    Computes Bias Severity Score (BSS).
    Formula: BSS = w1 × |metric_deviation| + w2 × effect_size_normalised
                 + w3 × statistical_confidence + w4 × subgroup_exposure_weight
    Weights: w1=0.35, w2=0.25, w3=0.25, w4=0.15
    """
    
    # 1. Metric Deviation
    # Thresholds: DPD > 0.1, DIR < 0.8 or > 1.2
    # We'll use DPD as the primary driver for a unified metric deviation, mapped to [0,1]
    # DPD of 0 -> dev 0. DPD >= 0.35 -> dev 1 (Industry standard significance)
    dpd_dev = min(1.0, abs(dpd) / 0.35)
    
    # Alternatively using DIR deviation form 1.0 (threshold 0.2 deviation)
    # dir_dev = min(1.0, abs(1.0 - dir_val) / 0.8) if dir_val is not None else 0
    
    metric_deviation = dpd_dev
    
    # 2. Effect size normalized
    # Cohen's h / 1.0 capped at 1.0
    effect_norm = min(1.0, effect_size_h / 1.0)
    
    # 3. Statistical Confidence
    if p_val_corrected < 0.05:
        conf = 1.0 - p_val_corrected
    else:
        conf = 0.0
        
    # 4. Exposure weight
    # Subgroup fraction directly [0,1]
    exposure = subgroup_fraction
    
    # Calculate
    bss = (0.35 * metric_deviation) + (0.25 * effect_norm) + (0.25 * conf) + (0.15 * exposure)
    
    return bss

def assign_tier(bss):
    """Assigns Priority Tier based on BSS."""
    if bss >= 0.55:
        return "Critical"
    elif bss >= 0.35:
        return "High"
    elif bss >= 0.15:
        return "Medium"
    else:
        return "Low"
