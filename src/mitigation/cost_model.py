"""
ChronosGuard Weather: Cost-Aware Energy Grid Reliability & Retraining Decision Model
Evaluates financial and operational loss due to weather model decay versus compute/telemetry retraining expense.

Cost Matrix for Energy Grid Reliability:
- False Negative (FN): Failure to predict Peak Grid Strain (Unhedged blackout / emergency deficit = $8,500/MWh)
- False Positive (FP): False alarm on Grid Strain (Unnecessary peaker plant firing / idle spinning reserve = $350/MWh)
- Compute & Calibration Expense (C_retrain): Retraining pipeline and telemetry re-indexing ($1,200)
"""

from typing import Dict, Any, Optional

DEFAULT_COST_FN = 8500.0   # Emergency grid blackout & reserve penalty
DEFAULT_COST_FP = 350.0    # Peaker plant spinning reserve startup loss
DEFAULT_RETRAIN_EXPENSE = 1200.0  # Compute & telemetry calibration expense

def calculate_operational_cost(
    fp: float,
    fn: float,
    cost_fp: float = DEFAULT_COST_FP,
    cost_fn: float = DEFAULT_COST_FN
) -> float:
    """
    Computes direct financial loss resulting from misclassification errors.
    """
    return float(fp * cost_fp + fn * cost_fn)

def evaluate_retraining_decision(
    current_fp: float,
    current_fn: float,
    baseline_fp: float,
    baseline_fn: float,
    cost_fp: float = DEFAULT_COST_FP,
    cost_fn: float = DEFAULT_COST_FN,
    retrain_expense: float = DEFAULT_RETRAIN_EXPENSE,
    projected_volume_multiplier: float = 1.0
) -> Dict[str, Any]:
    """
    Compares Cost of Model Decay against Cost of Retraining.
    Determines whether automated retraining has positive net expected ROI.
    """
    current_loss = calculate_operational_cost(current_fp, current_fn, cost_fp, cost_fn)
    baseline_loss = calculate_operational_cost(baseline_fp, baseline_fn, cost_fp, cost_fn)
    
    # Financial loss attributable purely to model degradation
    decay_loss_per_batch = max(0.0, current_loss - baseline_loss)
    projected_decay_loss = decay_loss_per_batch * projected_volume_multiplier
    
    # Net savings if retrained back to baseline performance
    net_roi = projected_decay_loss - retrain_expense
    roi_ratio = (projected_decay_loss / retrain_expense) if retrain_expense > 0 else 0.0
    
    should_retrain = (net_roi > 0.0) and (decay_loss_per_batch > 0.0)
    
    if should_retrain:
        recommendation = "TRIGGER_RETRAINING"
        reason = (
            f"Decay loss (${projected_decay_loss:,.2f}) exceeds retraining expense (${retrain_expense:,.2f}). "
            f"Estimated Net ROI: +${net_roi:,.2f} ({roi_ratio:.1f}x payback)."
        )
    elif decay_loss_per_batch > 0:
        recommendation = "DEFER_RETRAINING"
        reason = (
            f"Model decay loss (${projected_decay_loss:,.2f}) is below retraining threshold (${retrain_expense:,.2f}). "
            f"Retraining would yield negative short-term ROI (-${abs(net_roi):,.2f})."
        )
    else:
        recommendation = "NO_ACTION_REQUIRED"
        reason = "Model is performing within nominal baseline bounds. Zero economic decay observed."
        
    return {
        "current_loss": round(current_loss, 2),
        "baseline_loss": round(baseline_loss, 2),
        "decay_loss": round(decay_loss_per_batch, 2),
        "projected_decay_loss": round(projected_decay_loss, 2),
        "retrain_expense": round(retrain_expense, 2),
        "net_expected_roi": round(net_roi, 2),
        "roi_ratio": round(roi_ratio, 2),
        "should_retrain": should_retrain,
        "recommendation": recommendation,
        "reason": reason
    }

def evaluate_retraining_from_costs(
    current_total_cost: float,
    baseline_cost_per_sample: float,
    current_batch_size: int,
    retrain_expense: float = DEFAULT_RETRAIN_EXPENSE,
    projected_volume_multiplier: float = 1.0
) -> Dict[str, Any]:
    """
    Compares current batch operational cost against per-sample baseline projected
    onto the current batch size. This normalization correctly handles:
    - Covariate drift: class prevalence shifts change absolute error counts
    - Concept drift: P(Y|X) shifts change error patterns
    - Batch size differences: different stream window sizes
    
    Decay Cost = max(0, current_total_cost - baseline_cost_per_sample * batch_size)
    """
    expected_baseline_cost = baseline_cost_per_sample * current_batch_size
    decay_loss = max(0.0, current_total_cost - expected_baseline_cost)
    projected_decay = decay_loss * projected_volume_multiplier
    
    net_roi = projected_decay - retrain_expense
    roi_ratio = (projected_decay / retrain_expense) if retrain_expense > 0 else 0.0
    
    should_retrain = (net_roi > 0.0) and (decay_loss > 0.0)
    
    if should_retrain:
        recommendation = "TRIGGER_RETRAINING"
        reason = (
            f"Decay loss (${projected_decay:,.2f}) exceeds retraining expense (${retrain_expense:,.2f}). "
            f"Estimated Net ROI: +${net_roi:,.2f} ({roi_ratio:.1f}x payback)."
        )
    elif decay_loss > 0:
        recommendation = "DEFER_RETRAINING"
        reason = (
            f"Model decay loss (${projected_decay:,.2f}) is below retraining threshold (${retrain_expense:,.2f}). "
            f"Retraining would yield negative short-term ROI (-${abs(net_roi):,.2f})."
        )
    else:
        recommendation = "NO_ACTION_REQUIRED"
        reason = "Model is performing within nominal baseline bounds. No excess cost observed."
        
    return {
        "current_loss": round(current_total_cost, 2),
        "baseline_loss": round(expected_baseline_cost, 2),
        "decay_loss": round(decay_loss, 2),
        "projected_decay_loss": round(projected_decay, 2),
        "retrain_expense": round(retrain_expense, 2),
        "net_expected_roi": round(net_roi, 2),
        "roi_ratio": round(roi_ratio, 2),
        "should_retrain": should_retrain,
        "recommendation": recommendation,
        "reason": reason
    }

