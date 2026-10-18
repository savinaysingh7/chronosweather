"""
ChronosGuard: Model Training and Probability Calibration
Trains a baseline Gradient Boosting classifier with isotonic/sigmoid probability calibration.
Probability calibration is critical for accurate risk quantification and CBPE performance estimation.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Any
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

FEATURE_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "direct_radiation",
    "temp_demand_index",
    "dew_point_2m",
    "cloud_cover"
]
TARGET_COLUMN = "is_peak_strain"

def train_baseline_model(
    train_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = TARGET_COLUMN,
    random_state: int = 42
) -> Tuple[Any, Dict[str, float]]:
    """
    Trains a HistGradientBoostingClassifier wrapped in CalibratedClassifierCV.
    HistGradientBoosting natively handles complex non-linear tabular interactions.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS
        
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    
    base_clf = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.08,
        max_depth=5,
        random_state=random_state
    )
    
    # Wrap in probability calibrator (isotonic or sigmoid)
    calibrated_model = CalibratedClassifierCV(
        estimator=base_clf,
        method="isotonic",
        cv=3
    )
    
    calibrated_model.fit(X_train, y_train)
    
    train_probs = calibrated_model.predict_proba(X_train)[:, 1]
    train_auc = float(roc_auc_score(y_train, train_probs))
    train_brier = float(brier_score_loss(y_train, train_probs))
    
    metrics = {
        "train_samples": len(train_df),
        "train_auc": round(train_auc, 4),
        "train_brier": round(train_brier, 4),
        "train_default_rate": round(float(y_train.mean()), 4)
    }
    
    return calibrated_model, metrics

def save_model(model: Any, filepath: str = "models/baseline_model.joblib") -> str:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    return filepath

def load_model(filepath: str = "models/baseline_model.joblib") -> Any:
    return joblib.load(filepath)
