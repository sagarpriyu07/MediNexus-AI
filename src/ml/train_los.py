"""
Model 2: Length of Stay (LOS) Regression Model Training.
Predicts expected hospital stay in days; evaluates MAE, RMSE, and R2.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.ml.features import get_los_feature_matrix
from src.ml.model_registry import register_model
from src.utils.logging_utils import get_logger

logger = get_logger("train_los")


def train_los_model() -> Dict[str, Any]:
    """
    Train and evaluate Length of Stay (LOS) regression model.
    """
    logger.info("Extracting Length of Stay features from Gold layer...")
    X, y, feature_cols = get_los_feature_matrix()

    if X.empty or len(X) < 50:
        logger.warning("Insufficient data in Gold features matrix to train LOS model.")
        return {"status": "FAILED", "error": "Insufficient gold data"}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    reg = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.08,
        random_state=42,
    )
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    y_pred = np.clip(y_pred, 1.0, 45.0)

    mae = round(float(mean_absolute_error(y_test, y_pred)), 3)
    rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 3)
    r2 = round(float(r2_score(y_test, y_pred)), 3)

    importances = reg.feature_importances_
    feat_imp = sorted(
        [{"feature": f, "importance": round(float(imp), 4)} for f, imp in zip(feature_cols, importances)],
        key=lambda x: x["importance"],
        reverse=True,
    )

    metrics = {
        "mae_days": mae,
        "rmse_days": rmse,
        "r2_score": r2,
        "feature_importance": feat_imp[:6],
        "test_samples": len(y_test),
        "mean_actual_los": round(float(y.mean()), 2),
    }

    model_artifact = {
        "model": reg,
        "features": feature_cols,
        "metrics": metrics,
        "version": "1.2.0",
        "trained_at": datetime.now().isoformat(),
        "type": "los_regressor",
    }

    path = register_model(
        model_name="length_of_stay_model",
        version="1.2.0",
        dataset="gold_patient_risk_features",
        features=feature_cols,
        metrics=metrics,
        model_obj=model_artifact,
        filename="los_model.joblib",
    )

    logger.info(f"LOS Model trained successfully. MAE: {mae} days, RMSE: {rmse} days, R2: {r2}")
    return {
        "status": "SUCCESS",
        "model_name": "length_of_stay_model",
        "path": path,
        "metrics": metrics,
    }


if __name__ == "__main__":
    train_los_model()
