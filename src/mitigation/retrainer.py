"""
ChronosGuard: Adaptive Retraining Pipeline
Implements sliding-window and recency-weighted model retraining on drifted data streams.
Evaluates model recovery metrics (Delta AUC, Delta F1) and operational cost reduction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from src.model.trainer import train_baseline_model, FEATURE_COLUMNS, TARGET_COLUMN
from src.model.evaluator import evaluate_predictions
from src.mitigation.cost_model import calculate_operational_cost, DEFAULT_COST_FP, DEFAULT_COST_FN

def execute_retraining_pipeline(
    reference_df: pd.DataFrame,
    drifted_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = TARGET_COLUMN,
    window_strategy: str = "sliding_blend",
    drift_sample_weight: float = 2.0,
    cost_fp: float = DEFAULT_COST_FP,
    cost_fn: float = DEFAULT_COST_FN,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Simulates automated retraining triggered by drift/cost alert.
    Blends historical reference data with recent drifted batch and re-calibrates model.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS
        
    # Split drifted data into training portion (70%) and evaluation test portion (30%)
    np.random.seed(random_state)
    n_drift = len(drifted_df)
    perm = np.random.permutation(n_drift)
    split_idx = int(0.7 * n_drift)
    
    drift_train = drifted_df.iloc[perm[:split_idx]].copy()
    drift_test = drifted_df.iloc[perm[split_idx:]].copy()
    
    # Formulate retraining dataset
    if window_strategy == "sliding_blend":
        # Blend reference + recent drift
        retrain_pool = pd.concat([reference_df, drift_train], ignore_index=True)
    elif window_strategy == "drift_only":
        retrain_pool = drift_train
    else:
        retrain_pool = pd.concat([reference_df.sample(n=len(drift_train), random_state=random_state), drift_train], ignore_index=True)
        
    # Train new model
    new_model, train_meta = train_baseline_model(
        train_df=retrain_pool,
        feature_cols=feature_cols,
        target_col=target_col,
        random_state=random_state
    )
    
    # Evaluate new model on the unseen portion of drifted data
    X_test = drift_test[feature_cols].values
    y_test = drift_test[target_col].values
    
    new_probs = new_model.predict_proba(X_test)[:, 1]
    post_eval = evaluate_predictions(y_test, new_probs)
    
    post_cost = calculate_operational_cost(post_eval["fp"], post_eval["fn"], cost_fp, cost_fn)
    
    return {
        "retrained_model": new_model,
        "window_strategy": window_strategy,
        "retrain_sample_size": len(retrain_pool),
        "post_accuracy": post_eval["accuracy"],
        "post_f1": post_eval["f1"],
        "post_roc_auc": post_eval["roc_auc"],
        "post_brier": post_eval["brier_score"],
        "post_ece": post_eval["ece"],
        "post_operational_cost": round(post_cost, 2),
        "post_fp": post_eval["fp"],
        "post_fn": post_eval["fn"]
    }
