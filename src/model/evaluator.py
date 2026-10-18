"""
ChronosGuard: Ground-Truth Model Evaluation Engine
Calculates standard classification metrics, Brier score, and Expected Calibration Error (ECE).
Used to evaluate realized model decay when delayed ground-truth labels become available.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix
)

def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Computes Expected Calibration Error (ECE):
    ECE = sum_{b=1}^B (n_b / N) * |accuracy(b) - confidence(b)|
    """
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)
    
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)
    
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i+1]
        mask = (y_probs >= low) & (y_probs < high) if i < n_bins - 1 else (y_probs >= low) & (y_probs <= high)
        n_in_bin = np.sum(mask)
        
        if n_in_bin > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_probs[mask])
            ece += (n_in_bin / total_samples) * np.abs(bin_acc - bin_conf)
            
    return float(ece)

def evaluate_predictions(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Calculates comprehensive classification and calibration metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.clip(np.asarray(y_probs, dtype=float), 0.0, 1.0)
    y_pred = (y_probs >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = average_precision_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.0
    brier = brier_score_loss(y_true, y_probs)
    ece = compute_expected_calibration_error(y_true, y_probs)
    
    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "brier_score": round(float(brier), 4),
        "ece": round(float(ece), 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "total": int(len(y_true)),
        "default_rate": round(float(np.mean(y_true)), 4),
        "strain_rate": round(float(np.mean(y_true)), 4)
    }
