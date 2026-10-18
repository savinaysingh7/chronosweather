"""
ChronosGuard: Adversarial Domain Classifier for Multivariate Drift
Trains a discriminator to distinguish between Reference Data (y=0) and Current Batch (y=1).
- If AUC ≈ 0.50: Distributions are indistinguishable (No Drift).
- If AUC > 0.65: Moderate multivariate shift.
- If AUC > 0.75: Severe multivariate drift.
Also extracts feature importance to identify the primary drivers of multivariate drift.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def evaluate_adversarial_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: List[str],
    n_splits: int = 3,
    subsample_per_domain: int = 2000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains a discriminator to predict domain origin (Ref vs Current).
    Returns mean cross-validated ROC-AUC and feature importance rankings.
    """
    ref_sub = reference_df[feature_cols].copy()
    curr_sub = current_df[feature_cols].copy()
    
    # Subsample if necessary for fast real-time monitoring
    if len(ref_sub) > subsample_per_domain:
        ref_sub = ref_sub.sample(n=subsample_per_domain, random_state=random_state)
    if len(curr_sub) > subsample_per_domain:
        curr_sub = curr_sub.sample(n=subsample_per_domain, random_state=random_state)
        
    ref_sub["_domain_target"] = 0
    curr_sub["_domain_target"] = 1
    
    combined = pd.concat([ref_sub, curr_sub], ignore_index=True)
    X = combined[feature_cols].values
    y = combined["_domain_target"].values
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    auc_scores = []
    feature_importances = np.zeros(len(feature_cols))
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        clf = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=random_state,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        preds = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)
        auc_scores.append(auc)
        feature_importances += clf.feature_importances_
        
    mean_auc = float(np.mean(auc_scores))
    feature_importances /= n_splits
    
    # Ranking feature contributions to drift
    drift_drivers = pd.DataFrame({
        "feature": feature_cols,
        "importance": feature_importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)
    
    if mean_auc >= 0.75:
        status = "DRIFT_DETECTED"
    elif mean_auc >= 0.65:
        status = "WARNING"
    else:
        status = "STABLE"
        
    return {
        "adversarial_auc": round(mean_auc, 4),
        "auc_std": round(float(np.std(auc_scores)), 4),
        "status": status,
        "feature_importances": drift_drivers.to_dict(orient="records")
    }
