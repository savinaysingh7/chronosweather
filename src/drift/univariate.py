"""
ChronosGuard: Univariate Drift Detection Engine
Implements:
- Population Stability Index (PSI) with adaptive binning and epsilon smoothing
- Kolmogorov-Smirnov (KS) Two-Sample Test
- Wasserstein Distance (Earth Mover's Distance)
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any

def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4
) -> Tuple[float, pd.DataFrame]:
    """
    Computes Population Stability Index (PSI) between reference and current distribution.
    Uses reference quantiles as bin edges for robust continuous feature binning.
    
    Thresholds:
    - PSI < 0.1: No significant change (Stable)
    - 0.1 <= PSI < 0.2: Moderate shift (Warning)
    - PSI >= 0.2: Significant distribution shift (Drift Detected)
    """
    ref_clean = reference[~np.isnan(reference)]
    curr_clean = current[~np.isnan(current)]
    
    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return 0.0, pd.DataFrame()
        
    # Generate quantile bin edges from reference distribution
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref_clean, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    # Ensure unique edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) <= 2:
        # Fallback to uniform if quantiles collapse
        bin_edges = np.linspace(np.min(ref_clean) - 1e-5, np.max(ref_clean) + 1e-5, num_bins + 1)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        
    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    curr_counts, _ = np.histogram(curr_clean, bins=bin_edges)
    
    # Calculate proportions with smoothing
    ref_prop = (ref_counts / len(ref_clean)) + epsilon
    curr_prop = (curr_counts / len(curr_clean)) + epsilon
    
    # Re-normalize after smoothing
    ref_prop /= np.sum(ref_prop)
    curr_prop /= np.sum(curr_prop)
    
    # Component-wise PSI
    psi_components = (curr_prop - ref_prop) * np.log(curr_prop / ref_prop)
    total_psi = float(np.sum(psi_components))
    
    details_df = pd.DataFrame({
        "bin_lower": bin_edges[:-1],
        "bin_upper": bin_edges[1:],
        "ref_count": ref_counts,
        "curr_count": curr_counts,
        "ref_prop": ref_prop,
        "curr_prop": curr_prop,
        "psi_contribution": psi_components
    })
    
    return total_psi, details_df

def calculate_ks_test(reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
    """
    Two-sample Kolmogorov-Smirnov test for continuous distributions.
    Returns: (statistic, p_value)
    """
    ref_clean = reference[~np.isnan(reference)]
    curr_clean = current[~np.isnan(current)]
    
    res = stats.ks_2samp(ref_clean, curr_clean)
    return float(res.statistic), float(res.pvalue)

def calculate_wasserstein(reference: np.ndarray, current: np.ndarray) -> float:
    """
    Calculates 1D Wasserstein (Earth Mover's) Distance.
    Normalized by reference standard deviation for cross-feature comparability.
    """
    ref_clean = reference[~np.isnan(reference)]
    curr_clean = current[~np.isnan(current)]
    
    std = np.std(ref_clean)
    if std < 1e-8:
        std = 1.0
        
    raw_dist = stats.wasserstein_distance(ref_clean, curr_clean)
    return float(raw_dist / std)

def evaluate_univariate_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: List[str],
    psi_threshold_warn: float = 0.1,
    psi_threshold_drift: float = 0.2,
    ks_alpha: float = 0.01
) -> pd.DataFrame:
    """
    Runs full univariate drift suite across all specified feature columns.
    Returns summary DataFrame sorted by severity.
    """
    results = []
    
    for feat in feature_cols:
        ref_vals = reference_df[feat].to_numpy(dtype=float)
        curr_vals = current_df[feat].to_numpy(dtype=float)
        
        psi_val, _ = calculate_psi(ref_vals, curr_vals)
        ks_stat, ks_p = calculate_ks_test(ref_vals, curr_vals)
        wass_dist = calculate_wasserstein(ref_vals, curr_vals)
        
        if psi_val >= psi_threshold_drift or (ks_p < ks_alpha and ks_stat > 0.15):
            status = "DRIFT_DETECTED"
            severity = 2
        elif psi_val >= psi_threshold_warn or (ks_p < ks_alpha and ks_stat > 0.08):
            status = "WARNING"
            severity = 1
        else:
            status = "STABLE"
            severity = 0
            
        results.append({
            "feature": feat,
            "psi": round(psi_val, 4),
            "ks_statistic": round(ks_stat, 4),
            "ks_p_value": ks_p,
            "wasserstein_dist_norm": round(wass_dist, 4),
            "status": status,
            "severity_score": severity
        })
        
    summary_df = pd.DataFrame(results)
    summary_df = summary_df.sort_values(by=["severity_score", "psi"], ascending=[False, False]).reset_index(drop=True)
    return summary_df
