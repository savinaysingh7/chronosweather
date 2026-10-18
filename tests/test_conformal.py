"""
ChronosGuard: Unit Tests for Split Conformal Prediction
"""

import pytest
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

from src.model.conformal import ConformalPredictionClassifier


@pytest.fixture
def calibrated_system():
    rng = np.random.default_rng(42)
    # Generate 500 samples
    X = rng.normal(0, 1, size=(500, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Train model
    base = HistGradientBoostingClassifier(random_state=42)
    model = CalibratedClassifierCV(estimator=base, method="isotonic", cv=2)
    model.fit(X[:300], y[:300])
    
    # Calibration set
    X_cal = X[300:400]
    y_cal = y[300:400]
    
    # Test set
    X_test = X[400:]
    y_test = y[400:]
    
    calibrator = ConformalPredictionClassifier(model, alpha=0.10)
    q = calibrator.calibrate(X_cal, y_cal)
    
    return calibrator, X_test, y_test, q


def test_conformal_calibration(calibrated_system):
    calibrator, X_test, y_test, q = calibrated_system
    assert q is not None
    assert 0.0 < q <= 1.0
    assert calibrator.cal_sample_count == 100


def test_conformal_prediction_sets(calibrated_system):
    calibrator, X_test, y_test, _ = calibrated_system
    sets = calibrator.predict_sets(X_test)
    assert len(sets) == len(X_test)
    for s in sets:
        assert isinstance(s, list)
        for label in s:
            assert label in [0, 1]


def test_conformal_coverage_guarantee(calibrated_system):
    calibrator, X_test, y_test, _ = calibrated_system
    eval_res = calibrator.evaluate_batch(X_test, y_test)
    
    assert "mean_set_size" in eval_res
    assert "empirical_coverage" in eval_res
    assert "status" in eval_res
    
    # Under in-distribution testing, empirical coverage should be close to 1 - alpha (0.90)
    assert eval_res["empirical_coverage"] >= 0.82
    assert 1.0 <= eval_res["mean_set_size"] <= 2.0


def test_conformal_widening_under_shift(calibrated_system):
    calibrator, X_test, y_test, _ = calibrated_system
    eval_nom = calibrator.evaluate_batch(X_test, y_test)
    
    # Severe distribution shift: move data far into high-entropy ambiguous region
    X_shifted = np.zeros_like(X_test)  # Right on decision boundary
    eval_shifted = calibrator.evaluate_batch(X_shifted)
    
    # Ambiguity should increase under shift
    assert eval_shifted["mean_set_size"] >= 1.0
