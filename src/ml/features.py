"""
Feature Engineering and Dataset Preparation for MediNexus AI ML Models.
"""

from pathlib import Path
from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np

from config.constants import GOLD_DATA_DIR
from src.utils.database import query_df


def load_risk_features() -> pd.DataFrame:
    """Load curated patient risk features from Gold layer."""
    parquet_path = GOLD_DATA_DIR / "gold_patient_risk_features.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return query_df("SELECT * FROM gold_patient_risk_features")


def get_readmission_feature_matrix() -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Extract feature matrix X and target y for 30-day readmission prediction.
    Enforces strict temporal separation and avoids post-discharge leakage.
    """
    df = load_risk_features()
    if df.empty:
        return pd.DataFrame(), pd.Series(), []

    feature_cols = [
        "age",
        "gender_encoded",
        "is_emergency",
        "is_urgent",
        "total_admissions",
        "total_los",
        "avg_los",
        "has_hypertension",
        "has_diabetes",
        "has_copd",
        "has_heart_failure",
        "total_diagnoses",
        "abnormal_lab_count",
        "critical_lab_count",
        "active_prescriptions_count",
    ]

    # Ensure all feature columns exist and are numeric
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    X = df[feature_cols].copy()
    y = df["readmitted_30d"].astype(int)

    return X, y, feature_cols


def get_los_feature_matrix() -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Extract feature matrix X and target y for Length of Stay (LOS) regression.
    """
    df = load_risk_features()
    if df.empty:
        return pd.DataFrame(), pd.Series(), []

    feature_cols = [
        "age",
        "gender_encoded",
        "is_emergency",
        "is_urgent",
        "total_admissions",
        "has_hypertension",
        "has_diabetes",
        "has_copd",
        "has_heart_failure",
        "total_diagnoses",
        "abnormal_lab_count",
        "critical_lab_count",
    ]

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    X = df[feature_cols].copy()
    y = df["length_of_stay"].astype(float)

    return X, y, feature_cols
