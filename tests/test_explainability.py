"""
ChronosGuard: Unit Tests for SHAP Explainability Engine
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

from src.drift.explainability import compute_shap_drift_attribution, explain_single_sample


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    feature_cols = ["feat_a", "feat_b", "feat_c"]
    
    # Reference data
    X_ref = rng.normal(0, 1, size=(200, 3))
    y_ref = (X_ref[:, 0] + X_ref[:, 1] > 0).astype(int)
    ref_df = pd.DataFrame(X_ref, columns=feature_cols)
    ref_df["target"] = y_ref
    
    # Shifted data: feat_a shifted heavily by +2.0
    X_curr = rng.normal(0, 1, size=(100, 3))
    X_curr[:, 0] += 2.0
    y_curr = (X_curr[:, 0] + X_curr[:, 1] > 0).astype(int)
    curr_df = pd.DataFrame(X_curr, columns=feature_cols)
    curr_df["target"] = y_curr
    
    # Train calibrated model
    base_clf = HistGradientBoostingClassifier(random_state=42)
    model = CalibratedClassifierCV(estimator=base_clf, method="isotonic", cv=2)
    model.fit(X_ref, y_ref)
    
    return model, ref_df, curr_df, feature_cols


def test_shap_drift_attribution(sample_data):
    model, ref_df, curr_df, feature_cols = sample_data
    
    attribution_results = compute_shap_drift_attribution(
        model=model,
        reference_df=ref_df,
        current_df=curr_df,
        feature_cols=feature_cols,
        max_samples=100
    )
    
    assert "feature_attribution_df" in attribution_results
    assert "top_drift_driver" in attribution_results
    
    attr_df = attribution_results["feature_attribution_df"]
    assert len(attr_df) == len(feature_cols)
    assert "attribution_shift" in attr_df.columns
    assert "directional_push" in attr_df.columns
    
    # Shifted feature 'feat_a' should be identified as top driver
    assert attribution_results["top_drift_driver"] == "feat_a"


def test_single_sample_explanation(sample_data):
    model, _, curr_df, feature_cols = sample_data
    sample = curr_df.iloc[0]
    
    exp = explain_single_sample(model, sample, feature_cols)
    assert "explanation_df" in exp
    assert "top_risk_factor" in exp
    assert len(exp["explanation_df"]) == len(feature_cols)
