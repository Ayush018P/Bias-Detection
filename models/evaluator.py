"""
models/evaluator.py
Calculates model performance metrics: Accuracy, AUC, F1, Precision, Recall, Brier Score.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, brier_score_loss, roc_curve
)

def evaluate_model(model, X_test, y_test, model_name: str):
    """
    Computes standard performance metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0)
    }
    
    if y_proba is not None:
        metrics["AUC-ROC"] = roc_auc_score(y_test, y_proba)
        metrics["Brier Score"] = brier_score_loss(y_test, y_proba)
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        # Downsample ROC curve data slightly to avoid sending massive arrays to streamlit
        idx = np.linspace(0, len(fpr) - 1, min(len(fpr), 100)).astype(int)
        metrics["roc_curve"] = {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist()}
    else:
        metrics["AUC-ROC"] = np.nan
        metrics["Brier Score"] = np.nan
        metrics["roc_curve"] = None
        
    return metrics
