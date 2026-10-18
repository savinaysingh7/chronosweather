"""
ChronosGuard: Unit Tests for Drift Detection & Performance Estimation Suite
"""

import pytest
import numpy as np
import pandas as pd

from src.drift.univariate import calculate_psi, calculate_ks_test, calculate_wasserstein
from src.drift.multivariate import calculate_mmd, evaluate_multivariate_drift
from src.drift.cbpe import estimate_metrics_cbpe, estimate_confusion_matrix, estimate_expected_auc
from src.mitigation.cost_model import calculate_operational_cost, evaluate_retraining_decision

def test_psi_identical_distributions():
    np.random.seed(42)
    ref = np.random.normal(loc=100, scale=15, size=2000)
    curr = np.random.normal(loc=100, scale=15, size=2000)
    
    psi, _ = calculate_psi(ref, curr)
    assert psi < 0.05, f"Expected PSI for identical distributions to be < 0.05, got {psi}"

def test_psi_shifted_distribution():
    np.random.seed(42)
    ref = np.random.normal(loc=100, scale=15, size=2000)
    curr = np.random.normal(loc=130, scale=20, size=2000)  # Significant shift
    
    psi, _ = calculate_psi(ref, curr)
    assert psi >= 0.20, f"Expected PSI for shifted distributions to be >= 0.20, got {psi}"

def test_ks_test():
    np.random.seed(42)
    ref = np.random.normal(loc=0, scale=1, size=1500)
    curr_same = np.random.normal(loc=0, scale=1, size=1500)
    curr_shift = np.random.normal(loc=0.8, scale=1, size=1500)
    
    stat_same, p_same = calculate_ks_test(ref, curr_same)
    assert p_same > 0.01, f"Expected p-value > 0.01 for same distributions, got {p_same}"
    
    stat_shift, p_shift = calculate_ks_test(ref, curr_shift)
    assert p_shift < 1e-4, f"Expected p-value < 1e-4 for shifted distributions, got {p_shift}"
    assert stat_shift > 0.15

def test_mmd_detection():
    np.random.seed(42)
    X = np.random.normal(0, 1, size=(500, 4))
    Y_same = np.random.normal(0, 1, size=(500, 4))
    Y_shift = np.random.normal(1.2, 1.5, size=(500, 4))
    
    mmd_same = calculate_mmd(X, Y_same)
    mmd_shift = calculate_mmd(X, Y_shift)
    
    assert mmd_shift > mmd_same
    assert mmd_shift > 0.10

def test_cbpe_estimation():
    probs = np.array([0.9, 0.85, 0.8, 0.2, 0.1, 0.05])
    cm = estimate_confusion_matrix(probs, threshold=0.5)
    
    assert cm["expected_tp"] == pytest.approx(0.9 + 0.85 + 0.8)
    assert cm["expected_fp"] == pytest.approx(0.1 + 0.15 + 0.2)
    assert cm["expected_tn"] == pytest.approx(0.8 + 0.9 + 0.95)
    assert cm["expected_fn"] == pytest.approx(0.2 + 0.1 + 0.05)
    
    metrics = estimate_metrics_cbpe(probs)
    assert 0.0 <= metrics["estimated_accuracy"] <= 1.0
    assert 0.0 <= metrics["estimated_f1"] <= 1.0
    assert 0.0 <= metrics["estimated_auc"] <= 1.0

def test_cost_decision_model():
    # Baseline: 10 FP, 5 FN
    # Scenario A: Nominal (no decay)
    res_a = evaluate_retraining_decision(
        current_fp=10, current_fn=5,
        baseline_fp=10, baseline_fn=5,
        retrain_expense=1500
    )
    assert not res_a["should_retrain"]
    assert res_a["recommendation"] == "NO_ACTION_REQUIRED"
    
    # Scenario B: Catastrophic decay (100 FN defaults!)
    res_b = evaluate_retraining_decision(
        current_fp=25, current_fn=100,
        baseline_fp=10, baseline_fn=5,
        retrain_expense=1500
    )
    assert res_b["should_retrain"]
    assert res_b["recommendation"] == "TRIGGER_RETRAINING"
    assert res_b["net_expected_roi"] > 0
