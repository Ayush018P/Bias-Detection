"""
data/preprocessor.py
Preprocesses real-world tabular data.
Handles missing values (median/mode) and OrdinalEncoding of categoricals.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

def preprocess_data(X: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=42):
    """
    Cleans, encodes, and splits data.
    Returns: X_train, X_test, y_train, y_test, preprocessed_X
    """
    X_processed = X.copy()
    
    # 1. Identify column types
    numeric_cols = X_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_processed.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # 2. Impute missing values
    if numeric_cols:
        num_imputer = SimpleImputer(strategy='median')
        X_processed[numeric_cols] = num_imputer.fit_transform(X_processed[numeric_cols])
    
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_processed[categorical_cols] = cat_imputer.fit_transform(X_processed[categorical_cols])
    
    # 3. Encode categoricals using OrdinalEncoder (preserves subgroup interpretability vs OneHot)
    if categorical_cols:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_processed[categorical_cols] = encoder.fit_transform(X_processed[categorical_cols])
        
        # Ensure they remain integers if possible
    # 4. Force entirely new writeable memory allocations to avoid scikit-learn read-only view crashes
    X_processed = pd.DataFrame({col: np.array(X_processed[col]) for col in X_processed.columns}, index=X_processed.index)
    
    # Strictly enforce y mapping to exactly {0, 1} for Scikit-Learn ROC metrics
    y_vals = np.array(y)
    if len(np.unique(y_vals)) == 2:
        y_proc_vals = (y_vals == np.max(y_vals)).astype(int)
    else:
        y_proc_vals = y_vals.astype(int)
        
    y_processed = pd.Series(y_proc_vals, index=y.index, name=getattr(y, 'name', 'target'))
    
    # Train test split (stratified)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, test_size=test_size, random_state=random_state, stratify=y_processed
        )
    except ValueError:
        # Fallback if a class has fewer than 2 members and cannot be stratified
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, test_size=test_size, random_state=random_state, stratify=None
        )
    
    return X_train, X_test, y_train, y_test, X_processed
