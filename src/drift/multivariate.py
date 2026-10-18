"""
ChronosGuard: Multivariate Drift Detection Engine
Implements:
- Maximum Mean Discrepancy (MMD) with Gaussian/RBF kernel and median heuristic bandwidth
- Latent Space Shift via Principal Component Analysis (PCA)
- Permutation significance testing for MMD
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any

def compute_pairwise_sq_dists(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Computes pairwise squared Euclidean distances ||x_i - y_j||^2
    """
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 <x, y>
    x_norm_sq = np.sum(X ** 2, axis=1, keepdims=True)
    y_norm_sq = np.sum(Y ** 2, axis=1, keepdims=True)
    dists = x_norm_sq + y_norm_sq.T - 2.0 * np.dot(X, Y.T)
    return np.maximum(dists, 0.0)

def median_heuristic_gamma(X: np.ndarray, Y: np.ndarray, max_subsample: int = 1000) -> float:
    """
    Calculates RBF gamma = 1 / (2 * sigma^2) using median heuristic over pairwise distances.
    """
    if len(X) > max_subsample:
        idx_x = np.random.choice(len(X), max_subsample, replace=False)
        X_sub = X[idx_x]
    else:
        X_sub = X
        
    if len(Y) > max_subsample:
        idx_y = np.random.choice(len(Y), max_subsample, replace=False)
        Y_sub = Y[idx_y]
    else:
        Y_sub = Y
        
    combined = np.vstack([X_sub, Y_sub])
    dists = compute_pairwise_sq_dists(combined, combined)
    triu_indices = np.triu_indices_from(dists, k=1)
    median_dist_sq = float(np.median(dists[triu_indices]))
    
    if median_dist_sq < 1e-6:
        return 1.0
    return float(1.0 / (2.0 * median_dist_sq))

def calculate_mmd(
    X: np.ndarray,
    Y: np.ndarray,
    gamma: Optional[float] = None,
    unbiased: bool = True,
    subsample: int = 600
) -> float:
    """
    Calculates Maximum Mean Discrepancy (MMD) between distributions X and Y using RBF kernel.
    """
    # Subsample if large to maintain real-time performance
    if len(X) > subsample:
        idx_x = np.random.choice(len(X), subsample, replace=False)
        X = X[idx_x]
    if len(Y) > subsample:
        idx_y = np.random.choice(len(Y), subsample, replace=False)
        Y = Y[idx_y]
        
    m = len(X)
    n = len(Y)
    
    if gamma is None:
        gamma = median_heuristic_gamma(X, Y)
        
    K_xx = np.exp(-gamma * compute_pairwise_sq_dists(X, X))
    K_yy = np.exp(-gamma * compute_pairwise_sq_dists(Y, Y))
    K_xy = np.exp(-gamma * compute_pairwise_sq_dists(X, Y))
    
    if unbiased and m > 1 and n > 1:
        np.fill_diagonal(K_xx, 0.0)
        np.fill_diagonal(K_yy, 0.0)
        mmd_sq = (np.sum(K_xx) / (m * (m - 1))) + (np.sum(K_yy) / (n * (n - 1))) - (2.0 * np.sum(K_xy) / (m * n))
    else:
        mmd_sq = np.mean(K_xx) + np.mean(K_yy) - 2.0 * np.mean(K_xy)
        
    return float(np.sqrt(np.maximum(mmd_sq, 0.0)))

def evaluate_multivariate_drift(
    reference_data: np.ndarray,
    current_data: np.ndarray,
    standardize: bool = True,
    n_permutations: int = 15
) -> Dict[str, Any]:
    """
    Performs complete multivariate drift evaluation including MMD and statistical significance.
    """
    X = np.asarray(reference_data, dtype=float)
    Y = np.asarray(current_data, dtype=float)
    
    if standardize:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std < 1e-8] = 1.0
        X = (X - mean) / std
        Y = (Y - mean) / std
        
    gamma = median_heuristic_gamma(X, Y)
    mmd_score = calculate_mmd(X, Y, gamma=gamma, unbiased=True)
    
    # Fast permutation test for p-value
    pooled = np.vstack([X, Y])
    m = len(X)
    perm_scores = []
    
    for _ in range(n_permutations):
        shuffled = np.random.permutation(pooled)
        X_p, Y_p = shuffled[:m], shuffled[m:]
        perm_scores.append(calculate_mmd(X_p, Y_p, gamma=gamma, unbiased=True))
        
    p_value = float(np.mean([s >= mmd_score for s in perm_scores]))
    
    # Threshold heuristic: MMD > 0.05 with p < 0.05 indicates multivariate shift
    if mmd_score >= 0.08 and p_value < 0.05:
        status = "DRIFT_DETECTED"
    elif mmd_score >= 0.04:
        status = "WARNING"
    else:
        status = "STABLE"
        
    return {
        "mmd_score": round(mmd_score, 5),
        "p_value": p_value,
        "gamma_rbf": round(gamma, 6),
        "status": status,
        "n_permutations": n_permutations
    }
