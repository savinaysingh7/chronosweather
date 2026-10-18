"""
ChronosGuard: SHAP Drift Explainability & Root-Cause Attribution Engine
Quantifies *why* a model's predictions drifted and attributes model decay
to specific feature distributional shifts using Shapley additive explanations (SHAP).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import shap


def _extract_tree_estimators(model: Any) -> List[Any]:
    """Extracts underlying tree estimators from a model or CalibratedClassifierCV."""
    if hasattr(model, "calibrated_classifiers_"):
        estimators = []
        for cc in model.calibrated_classifiers_:
            if hasattr(cc, "estimator"):
                estimators.append(cc.estimator)
            elif hasattr(cc, "base_estimator"):
                estimators.append(cc.base_estimator)
        if estimators:
            return estimators
    elif hasattr(model, "estimators_"):
        return [model]
    elif hasattr(model, "predict_proba"):
        return [model]
    return [model]


def compute_shap_drift_attribution(
    model: Any,
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: List[str],
    max_samples: int = 300,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Computes SHAP explanations comparing reference and current (drifted) data.
    
    Returns:
        Dict containing:
        - feature_attribution_df: Comparative DataFrame of mean |SHAP|, shift, and direction
        - top_drift_driver: Name of the feature driving the largest prediction impact shift
        - reference_shap_values: Raw SHAP values array for reference sample
        - current_shap_values: Raw SHAP values array for current sample
        - mean_ref_shap: Mean absolute SHAP per feature in reference
        - mean_curr_shap: Mean absolute SHAP per feature in current
        - sample_ref_df: Sampled reference dataframe
        - sample_curr_df: Sampled current dataframe
    """
    rng = np.random.default_rng(random_state)
    
    n_ref = min(len(reference_df), max_samples)
    ref_idx = rng.choice(len(reference_df), size=n_ref, replace=False)
    ref_sample = reference_df.iloc[ref_idx][feature_cols].copy().reset_index(drop=True)
    
    n_curr = min(len(current_df), max_samples)
    curr_idx = rng.choice(len(current_df), size=n_curr, replace=False)
    curr_sample = current_df.iloc[curr_idx][feature_cols].copy().reset_index(drop=True)
    
    estimators = _extract_tree_estimators(model)
    
    # Compute SHAP values across estimators (averaging over calibration folds if present)
    ref_shap_accum = np.zeros(ref_sample.shape)
    curr_shap_accum = np.zeros(curr_sample.shape)
    
    computed_count = 0
    for est in estimators:
        try:
            tree_explainer = shap.TreeExplainer(est)
            r_vals = tree_explainer.shap_values(ref_sample.values)
            c_vals = tree_explainer.shap_values(curr_sample.values)
            
            # Handle binary classification shapes (some SHAP versions return 2 arrays or 3D)
            if isinstance(r_vals, list) and len(r_vals) == 2:
                r_vals = r_vals[1]
                c_vals = c_vals[1]
            elif isinstance(r_vals, np.ndarray) and r_vals.ndim == 3:
                r_vals = r_vals[:, :, 1]
                c_vals = c_vals[:, :, 1]
                
            ref_shap_accum += r_vals
            curr_shap_accum += c_vals
            computed_count += 1
        except Exception:
            continue
            
    if computed_count == 0:
        # Fallback to Kernel/Exact Explainer if TreeExplainer was incompatible
        explainer = shap.Explainer(lambda x: model.predict_proba(x)[:, 1], ref_sample.values[:50])
        r_expl = explainer(ref_sample.values)
        c_expl = explainer(curr_sample.values)
        ref_shap_accum = r_expl.values
        curr_shap_accum = c_expl.values
        computed_count = 1
        
    ref_shap = ref_shap_accum / computed_count
    curr_shap = curr_shap_accum / computed_count
    
    # Calculate feature-level attributions
    mean_ref_abs = np.mean(np.abs(ref_shap), axis=0)
    mean_curr_abs = np.mean(np.abs(curr_shap), axis=0)
    mean_ref_raw = np.mean(ref_shap, axis=0)
    mean_curr_raw = np.mean(curr_shap, axis=0)
    
    shift_abs = mean_curr_abs - mean_ref_abs
    directional_push = mean_curr_raw - mean_ref_raw
    
    records = []
    for i, col in enumerate(feature_cols):
        records.append({
            "feature": col,
            "ref_mean_abs_shap": float(mean_ref_abs[i]),
            "curr_mean_abs_shap": float(mean_curr_abs[i]),
            "attribution_shift": float(shift_abs[i]),
            "directional_push": float(directional_push[i]),
            "risk_direction": "Increased Risk" if directional_push[i] > 0 else "Decreased Risk",
            "impact_magnitude": abs(float(shift_abs[i]))
        })
        
    attr_df = pd.DataFrame(records).sort_values(by="impact_magnitude", ascending=False).reset_index(drop=True)
    top_driver = attr_df.iloc[0]["feature"] if len(attr_df) > 0 else "None"
    
    return {
        "feature_attribution_df": attr_df,
        "top_drift_driver": top_driver,
        "reference_shap_values": ref_shap,
        "current_shap_values": curr_shap,
        "sample_ref_df": ref_sample,
        "sample_curr_df": curr_sample,
        "feature_names": feature_cols
    }


def explain_single_sample(
    model: Any,
    sample_series: pd.Series,
    feature_cols: List[str]
) -> Dict[str, Any]:
    """
    Generates a localized SHAP explanation for a single live incoming tick / sample.
    """
    estimators = _extract_tree_estimators(model)
    x = sample_series[feature_cols].values.reshape(1, -1)
    
    shap_accum = np.zeros(len(feature_cols))
    count = 0
    for est in estimators:
        try:
            tree_explainer = shap.TreeExplainer(est)
            vals = tree_explainer.shap_values(x)
            if isinstance(vals, list) and len(vals) == 2:
                vals = vals[1]
            elif isinstance(vals, np.ndarray) and vals.ndim == 3:
                vals = vals[:, :, 1]
            shap_accum += vals.ravel()
            count += 1
        except Exception:
            continue
            
    if count == 0:
        shap_vals = np.zeros(len(feature_cols))
    else:
        shap_vals = shap_accum / count
        
    records = []
    for i, col in enumerate(feature_cols):
        records.append({
            "feature": col,
            "feature_value": float(sample_series[col]),
            "shap_value": float(shap_vals[i]),
            "effect": "Pushes Risk UP" if shap_vals[i] > 0 else "Pushes Risk DOWN"
        })
        
    res_df = pd.DataFrame(records).sort_values(by="shap_value", key=abs, ascending=False).reset_index(drop=True)
    
    return {
        "explanation_df": res_df,
        "top_risk_factor": res_df.iloc[0]["feature"] if len(res_df) > 0 else "N/A",
        "raw_shap": shap_vals
    }
