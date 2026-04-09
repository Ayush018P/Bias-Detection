"""
data/loaders.py
Dataset downloaders for SABPF.
All loaders return a standardised dict:
  {X, y, protected_cols, feature_names, dataset_name, task_type}
Data is cached to data/raw/ to avoid re-downloading.

"""

import os
import pickle
import logging
import pathlib
import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

RAW_DIR = pathlib.Path(__file__).parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(name: str) -> pathlib.Path:
    return RAW_DIR / f"{name}.pkl"


def _load_cache(name: str):
    p = _cache_path(name)
    if p.exists():
        logger.info(f"Loading {name} from cache.")
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def _save_cache(name: str, data: dict):
    with open(_cache_path(name), "wb") as f:
        pickle.dump(data, f)
    logger.info(f"Cached {name}.")


# ─────────────────────────────────────────────
# 1. Adult Income (UCI)
# ─────────────────────────────────────────────
def load_adult() -> dict:
    """UCI Adult Income dataset. Target: income >50K."""
    name = "adult"
    cached = _load_cache(name)
    if cached:
        return cached

    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=2)
        X = dataset.data.features.copy()
        y_raw = dataset.data.targets.copy()
    except Exception as e:
        logger.warning(f"ucimlrepo failed: {e}. Falling back to URL download.")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        cols = [
            "age", "workclass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship", "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
        ]
        df = pd.read_csv(url, names=cols, na_values=" ?", skipinitialspace=True)
        X = df.drop("income", axis=1)
        y_raw = df[["income"]]

    # Normalise target
    if hasattr(y_raw, "iloc"):
        y_series = y_raw.iloc[:, 0].astype(str).str.strip().str.rstrip(".")
    else:
        y_series = pd.Series(y_raw).astype(str).str.strip().str.rstrip(".")
    y = (y_series.str.contains(">50K")).astype(int)

    # Drop rows where y is NaN
    mask = y.notna()
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    result = {
        "X": X,
        "y": y,
        "protected_cols": ["sex", "race"],
        "feature_names": list(X.columns),
        "dataset_name": "Adult Income",
        "task_type": "binary_classification",
    }
    _save_cache(name, result)
    return result


# ─────────────────────────────────────────────
# 2. COMPAS Recidivism (ProPublica)
# ─────────────────────────────────────────────
def load_compas() -> dict:
    """COMPAS Recidivism dataset from ProPublica. Target: two_year_recid."""
    name = "compas"
    cached = _load_cache(name)
    if cached:
        return cached

    url = (
        "https://raw.githubusercontent.com/propublica/compas-analysis/"
        "master/compas-scores-two-years.csv"
    )
    try:
        df = pd.read_csv(url)
    except Exception as e:
        raise RuntimeError(f"Failed to download COMPAS dataset: {e}")

    # Filter as per ProPublica methodology
    df = df[
        (df["days_b_screening_arrest"] <= 30)
        & (df["days_b_screening_arrest"] >= -30)
        & (df["is_recid"] != -1)
        & (df["c_charge_degree"] != "O")
        & (df["score_text"] != "N/A")
    ].copy()

    keep_cols = [
        "sex", "race", "age", "age_cat", "juv_fel_count", "juv_misd_count",
        "juv_other_count", "priors_count", "c_charge_degree", "two_year_recid"
    ]
    df = df[keep_cols].dropna()

    X = df.drop("two_year_recid", axis=1).reset_index(drop=True)
    y = df["two_year_recid"].astype(int).reset_index(drop=True)

    result = {
        "X": X,
        "y": y,
        "protected_cols": ["race", "sex"],
        "feature_names": list(X.columns),
        "dataset_name": "COMPAS Recidivism",
        "task_type": "binary_classification",
    }
    _save_cache(name, result)
    return result


# ─────────────────────────────────────────────
# 3. German Credit (UCI)
# ─────────────────────────────────────────────
def load_german_credit() -> dict:
    """UCI German Credit dataset. Target: credit risk (1=good, 2=bad → binary)."""
    name = "german_credit"
    cached = _load_cache(name)
    if cached:
        return cached

    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=144)
        X = dataset.data.features.copy()
        y_raw = dataset.data.targets.copy()
        y = (y_raw.iloc[:, 0] == 2).astype(int)  # 2 = bad credit
    except Exception as e:
        logger.warning(f"ucimlrepo failed: {e}. Falling back to URL download.")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
        col_names = [f"A{i}" for i in range(1, 21)] + ["target"]
        df = pd.read_csv(url, sep=" ", header=None, names=col_names)
        X = df.drop("target", axis=1)
        y = (df["target"] == 2).astype(int)

    # Create derived protected columns
    X = X.copy()
    # Age: binarise to young (<25) vs older
    age_col = None
    for c in X.columns:
        if "age" in str(c).lower():
            age_col = c
            break
    if age_col:
        X["age_group"] = (X[age_col] < 25).astype(int).map({0: "25+", 1: "Under25"})

    # Sex: look for personal_status column (A_11 contains sex info in UCI encoding)
    sex_col = None
    for c in X.columns:
        if "personal" in str(c).lower() or c in ["A9", "personal_status_and_sex"]:
            sex_col = c
            break

    protected = []
    if age_col:
        protected.append("age_group")
    if sex_col:
        protected.append(sex_col)
    if not protected:
        protected = [X.columns[0]]

    result = {
        "X": X.reset_index(drop=True),
        "y": y.reset_index(drop=True),
        "protected_cols": protected,
        "feature_names": list(X.columns),
        "dataset_name": "German Credit",
        "task_type": "binary_classification",
    }
    _save_cache(name, result)
    return result


# ─────────────────────────────────────────────
# 4. Bank Marketing (UCI)
# ─────────────────────────────────────────────
def load_bank_marketing() -> dict:
    """UCI Bank Marketing dataset. Target: subscribed to term deposit (y)."""
    name = "bank_marketing"
    cached = _load_cache(name)
    if cached:
        return cached

    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=222)
        X = dataset.data.features.copy()
        y_raw = dataset.data.targets.copy()
        y = (y_raw.iloc[:, 0].astype(str).str.strip() == "yes").astype(int)
    except Exception as e:
        logger.warning(f"ucimlrepo failed: {e}. Falling back to URL download.")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
        import io, zipfile
        r = requests.get(url, timeout=60)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        with z.open("bank-additional/bank-additional-full.csv") as f:
            df = pd.read_csv(f, sep=";")
        X = df.drop("y", axis=1)
        y = (df["y"] == "yes").astype(int)

    # Age group
    if "age" in X.columns:
        X = X.copy()
        X["age_group"] = pd.cut(
            X["age"], bins=[0, 30, 50, 100],
            labels=["Young", "Middle", "Senior"]
        ).astype(str)

    protected = []
    if "age_group" in X.columns:
        protected.append("age_group")
    if "marital" in X.columns:
        protected.append("marital")
    if not protected:
        protected = [X.columns[0]]

    result = {
        "X": X.reset_index(drop=True),
        "y": y.reset_index(drop=True),
        "protected_cols": protected,
        "feature_names": list(X.columns),
        "dataset_name": "Bank Marketing",
        "task_type": "binary_classification",
    }
    _save_cache(name, result)
    return result


# ─────────────────────────────────────────────
# 5. Law School Admissions (OpenML)
# ─────────────────────────────────────────────
def load_law_school() -> dict:
    """Law School Admissions dataset. Target: bar passage (pass_bar)."""
    name = "law_school"
    cached = _load_cache(name)
    if cached:
        return cached

    try:
        import openml
        dataset = openml.datasets.get_dataset(43)
        X_df, y_df, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        X = X_df.copy()
        y = pd.Series(y_df).astype(int)
    except Exception:
        # Fallback: direct CSV from known source
        url = "http://www.seaphe.org/databases/LSAC/LSAC.csv"
        try:
            df = pd.read_csv(url)
        except Exception:
            # Second fallback: use a curated version
            url2 = "https://raw.githubusercontent.com/fairlearn/fairlearn/main/docs/user_guide/datasets/lawschool_passage.csv"
            df = pd.read_csv(url2)

        target_col = None
        for c in df.columns:
            if "pass" in c.lower() or "bar" in c.lower():
                target_col = c
                break
        if target_col is None:
            target_col = df.columns[-1]

        X = df.drop(target_col, axis=1)
        y = (df[target_col] >= 1).astype(int)

    protected = []
    for c in X.columns:
        if c.lower() in ["race", "race1", "race2", "gender", "sex"]:
            protected.append(c)
    if not protected:
        for c in X.columns:
            if X[c].dtype == object and X[c].nunique() < 15:
                protected.append(c)
    if not protected:
        protected = [X.columns[0]]
    protected = protected[:2]  # max 2

    result = {
        "X": X.reset_index(drop=True),
        "y": y.reset_index(drop=True),
        "protected_cols": protected,
        "feature_names": list(X.columns),
        "dataset_name": "Law School",
        "task_type": "binary_classification",
    }
    _save_cache(name, result)
    return result


# ─────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────
DATASET_LOADERS = {
    "Adult Income": load_adult,
    "COMPAS Recidivism": load_compas,
    "German Credit": load_german_credit,
    "Bank Marketing": load_bank_marketing,
    "Law School": load_law_school,
}


def load_dataset(name: str) -> dict:
    """Load dataset by name from the registry."""
    if name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(DATASET_LOADERS.keys())}")
    return DATASET_LOADERS[name]()
