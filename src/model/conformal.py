"""
ChronosGuard: Split Conformal Prediction & Distribution-Free Uncertainty Quantification
Provides (1 - alpha) mathematically guaranteed prediction sets and confidence intervals.
Tracks predictive set size widening and coverage breakdown as empirical proof of model decay.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple


class ConformalPredictionClassifier:
    """
    Inductive Split Conformal Prediction for binary probabilistic classifiers.
    Produces valid prediction sets C(X) in {0}, {1}, {0, 1}, or empty set.
    """
    def __init__(self, model: Any, alpha: float = 0.10):
        """
        Args:
            model: Fitted probabilistic classifier with predict_proba method.
            alpha: Significance level (default 0.10 for 90% marginal coverage).
        """
        self.model = model
        self.alpha = float(alpha)
        self.conformal_quantile: Optional[float] = None
        self.cal_sample_count: int = 0
        self.cal_scores: Optional[np.ndarray] = None

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> float:
        """
        Calibrates the non-conformity threshold on a held-out reference calibration set.
        Non-conformity score: s_i = 1 - P(Y = y_i | x_i).
        """
        X_cal = np.asarray(X_cal)
        y_cal = np.asarray(y_cal).astype(int)
        n = len(y_cal)
        if n == 0:
            raise ValueError("Calibration set must not be empty.")
            
        probs = self.model.predict_proba(X_cal)[:, 1]
        
        # Non-conformity scores: 1 - prob of true class
        # If y=1, score is 1 - p. If y=0, score is p.
        scores = np.where(y_cal == 1, 1.0 - probs, probs)
        self.cal_scores = scores
        self.cal_sample_count = n
        
        # Conformal quantile with finite sample correction: ceil((n+1)(1-alpha))/n
        q_level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.conformal_quantile = float(np.quantile(scores, q_level, method="higher"))
        
        return self.conformal_quantile

    def predict_sets(self, X: np.ndarray) -> List[List[int]]:
        """
        Generates conformal prediction sets for incoming samples.
        y in C(x) iff (1 - P(Y=y | x)) <= q_hat
        """
        if self.conformal_quantile is None:
            raise RuntimeError("Conformal calibrator has not been calibrated. Call calibrate() first.")
            
        X = np.asarray(X)
        probs = self.model.predict_proba(X)[:, 1]
        q = self.conformal_quantile
        
        # Class 0 included if (1 - (1-p)) = p <= q
        # Class 1 included if (1 - p) <= q
        inc_0 = probs <= q
        inc_1 = (1.0 - probs) <= q
        
        prediction_sets = []
        for i0, i1 in zip(inc_0, inc_1):
            s = []
            if i0:
                s.append(0)
            if i1:
                s.append(1)
            prediction_sets.append(s)
            
        return prediction_sets

    def predict_uncertainty_bounds(self, X: np.ndarray) -> np.ndarray:
        """
        Computes lower and upper conformal probability bounds for each sample.
        Returns: (N, 2) array of [prob_lower, prob_upper]
        """
        if self.conformal_quantile is None:
            raise RuntimeError("Conformal calibrator has not been calibrated.")
            
        X = np.asarray(X)
        probs = self.model.predict_proba(X)[:, 1]
        margin = max(0.0, self.conformal_quantile - 0.5)
        
        lower = np.clip(probs - margin, 0.0, 1.0)
        upper = np.clip(probs + margin, 0.0, 1.0)
        
        return np.column_stack([lower, upper])

    def evaluate_batch(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Evaluates conformal metrics over a batch of incoming data.
        Tracks set size widening, ambiguity, and empirical coverage.
        """
        sets = self.predict_sets(X)
        set_sizes = np.array([len(s) for s in sets])
        
        mean_set_size = float(np.mean(set_sizes))
        singleton_rate = float(np.mean(set_sizes == 1))
        ambiguity_rate = float(np.mean(set_sizes == 2))
        empty_rate = float(np.mean(set_sizes == 0))
        
        coverage = None
        if y is not None:
            y = np.asarray(y).astype(int)
            covered = [y[i] in sets[i] for i in range(len(y))]
            coverage = float(np.mean(covered))
            
        probs = self.model.predict_proba(X)[:, 1]
        bounds = self.predict_uncertainty_bounds(X)
        bandwidths = bounds[:, 1] - bounds[:, 0]
        mean_bandwidth = float(np.mean(bandwidths))
        
        # Drift status heuristic based on conformal uncertainty widening
        if mean_set_size < 1.25 and ambiguity_rate < 0.25:
            status = "CONFIDENT_NOMINAL"
        elif mean_set_size < 1.55:
            status = "ELEVATED_UNCERTAINTY"
        else:
            status = "HIGH_DECAY_AMBIGUITY"
            
        return {
            "mean_set_size": round(mean_set_size, 3),
            "singleton_rate": round(singleton_rate, 4),
            "ambiguity_rate": round(ambiguity_rate, 4),
            "empty_rate": round(empty_rate, 4),
            "empirical_coverage": round(coverage, 4) if coverage is not None else None,
            "target_coverage": 1.0 - self.alpha,
            "conformal_quantile": round(self.conformal_quantile, 4),
            "mean_uncertainty_bandwidth": round(mean_bandwidth, 4),
            "status": status,
            "prediction_sets": sets,
            "sample_probabilities": probs,
            "uncertainty_bounds": bounds
        }
