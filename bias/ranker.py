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
    # DPD threshold normalised against 0.20 (EU AI Act / EEOC practical significance level)
    # DPD of 0 -> dev 0. DPD >= 0.20 -> dev 1
    dpd_dev = min(1.0, abs(dpd) / 0.20)
    
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
    """Assigns Priority Tier based on BSS.
    Calibrated so that genuine dataset biases surface as Critical/High:
      Critical : BSS >= 0.40  (was 0.55)
      High     : BSS >= 0.25  (was 0.35)
      Medium   : BSS >= 0.10  (was 0.15)
      Low      : BSS <  0.10
    """
    if bss >= 0.40:
        return "Critical"
    elif bss >= 0.25:
        return "High"
    elif bss >= 0.10:
        return "Medium"
    else:
        return "Low"

def assign_legal_risk_tier(bss, dpd, dir_val):
    """
    Assigns Legal Risk Tier based on standard compliance frameworks (e.g., EU AI Act, EEOC).
    """
    if abs(dpd) >= 0.20 or (dir_val is not None and dir_val <= 0.8):
        return "Non-Compliant (High Risk)"
    elif bss >= 0.25:
        return "Review Required (Medium Risk)"
    else:
        return "Compliant (Low Risk)"

def evaluate_model_risk(df):
    """
    Evaluates the overall model risk based on the highest risk subgroup found.
    """
    if df.empty or 'Legal_Risk' not in df.columns:
        return "Unknown"
        
    risks = df['Legal_Risk'].values
    if "Non-Compliant (High Risk)" in risks:
        return "Non-Compliant (High Risk)"
    elif "Review Required (Medium Risk)" in risks:
        return "Review Required (Medium Risk)"
    else:
        return "Compliant (Low Risk)"
