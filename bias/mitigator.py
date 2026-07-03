import pandas as pd
import numpy as np
from fairlearn.preprocessing import CorrelationRemover
from fairlearn.postprocessing import ThresholdOptimizer

def mitigate_data(X: pd.DataFrame, protected_cols: list) -> pd.DataFrame:
    """
    Removes linear correlations between the features and the protected attributes
    using orthogonal projection (equivalent to Fairlearn's CorrelationRemover).
    """
    if not protected_cols:
        return X.copy()
        
    # Safety check: ignore any protected columns that might have been removed (e.g. target labels)
    valid_protected = [c for c in protected_cols if c in X.columns]
    if not valid_protected:
        return X.copy()
        
    # Extract sensitive features and remaining features
    X_cr_df = X.copy()
    S = X[valid_protected].values.astype(float)
    S_centered = S - S.mean(axis=0)
    
    for col in X.columns:
        if col not in protected_cols:
            v = X[col].values.astype(float)
            v_mean = v.mean()
            v_centered = v - v_mean
            
            # Regress feature on sensitive features to find the correlated component
            # Using lstsq for numerical stability
            beta, _, _, _ = np.linalg.lstsq(S_centered, v_centered, rcond=None)
            
            # Subtract the projection (the correlated part) and add the mean back
            v_residual = v_centered - S_centered.dot(beta)
            X_cr_df[col] = v_residual + v_mean
            
    return X_cr_df

def mitigate_model(model, X_preprocessed: pd.DataFrame, y_true: pd.Series, sensitive_features: pd.Series, constraint: str = "demographic_parity"):
    """
    Applies ThresholdOptimizer to adjust the model's decision boundaries for different groups.
    Constraint can be 'demographic_parity' or 'equalized_odds'.
    """
    # ThresholdOptimizer requires a pre-fit model
    optimizer = ThresholdOptimizer(
        estimator=model,
        constraints=constraint,
        predict_method='predict',
        prefit=True
    )
    
    # Fit the optimizer to learn the thresholds using the PREPROCESSED data for the model,
    # and the RAW sensitive features for the fairness constraints.
    optimizer.fit(X_preprocessed, y_true, sensitive_features=sensitive_features)
    
    return optimizer
