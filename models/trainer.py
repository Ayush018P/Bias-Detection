"""
models/trainer.py
Multi-model training pipeline.
Trains LogReg, RF, XGBoost, DecisionTree, and SVM.
Logs metrics and saves models via joblib.
"""

import os
import joblib
import pathlib
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pandas as pd

from .evaluator import evaluate_model

logger = logging.getLogger(__name__)

MODELS_DIR = pathlib.Path(__file__).parent / "saved"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def get_models():
    """Returns the dictionary of initialized models based on SABPF specs."""
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=42)),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, min_samples_split=10, min_samples_leaf=4, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=400, max_depth=7, learning_rate=0.08, subsample=0.9, colsample_bytree=0.8, random_state=42, eval_metric='logloss'),
        # SVM needs scaling, so we wrap it in a pipeline
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=1.5, probability=True, random_state=42))
    }

def train_and_evaluate(dataset_name: str, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, selected_models=None):
    """
    Trains models, evaluates them, and saves artifacts.
    """
    all_models = get_models()
    if selected_models:
        models_to_train = {k: v for k, v in all_models.items() if k in selected_models}
    else:
        models_to_train = all_models
        
    results = {}
    trained_models = {}
    
    for name, model in models_to_train.items():
        logger.info(f"Training {name} on {dataset_name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test, name)
        results[name] = metrics
        trained_models[name] = model
        
        # Save
        model_path = MODELS_DIR / f"{dataset_name.replace(' ', '_').lower()}_{name.replace(' ', '_').lower()}.joblib"
        joblib.dump(model, model_path)
        
    return trained_models, results

def load_trained_models(dataset_name: str, X_test: pd.DataFrame, y_test: pd.Series, selected_models=None):
    """
    Attempts to load already trained models from disk for a given dataset to save time.
    """
    all_models = get_models()
    if selected_models:
        models_to_load = {k: v for k, v in all_models.items() if k in selected_models}
    else:
        models_to_load = all_models
        
    loaded_models = {}
    results = {}
    
    for name in models_to_load.keys():
        model_path = MODELS_DIR / f"{dataset_name.replace(' ', '_').lower()}_{name.replace(' ', '_').lower()}.joblib"
        if model_path.exists():
            try:
                model = joblib.load(model_path)
                loaded_models[name] = model
                metrics = evaluate_model(model, X_test, y_test, name)
                results[name] = metrics
                logger.info(f"Loaded existing {name} for {dataset_name}.")
            except Exception as e:
                logger.warning(f"Could not load {name}: {e}")
                
    return loaded_models, results
