"""
ChronosGuard: Unified End-to-End MLOps Pipeline Orchestrator
Executes:
1. Data ingestion across chronological periods T0 to T4
2. Baseline Model Training & Probability Calibration on T0 (Train Split)
3. Proper Out-of-Sample Baseline Evaluation on T0 (Validation Split)
4. Batch-by-Batch Monitoring:
   - Univariate Drift (PSI, KS-Test, Wasserstein)
   - Multivariate Drift (MMD RBF Kernel)
   - Adversarial Domain Classifier AUC
   - Label-Free Performance Estimation (CBPE)
   - Cost-Aware Retraining Decision (Loss of Decay vs Cost of Compute)
   - Ground-Truth Verification (Actual vs Estimated Decay)
   - Mitigation / Adaptive Retraining on Trigger
5. Exports structured telemetry for Dashboard and Research Report
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from data.generator import generate_all_periods
from src.model.trainer import train_baseline_model, save_model, FEATURE_COLUMNS, TARGET_COLUMN
from src.model.evaluator import evaluate_predictions
from src.drift.univariate import evaluate_univariate_drift
from src.drift.multivariate import evaluate_multivariate_drift
from src.drift.adversarial import evaluate_adversarial_drift
from src.drift.cbpe import estimate_metrics_cbpe
from src.mitigation.cost_model import (
    evaluate_retraining_decision,
    evaluate_retraining_from_costs,
    calculate_operational_cost
)
from src.mitigation.retrainer import execute_retraining_pipeline

def run_chronos_guard_pipeline(
    data_dir: str = "data/raw",
    output_dir: str = "results",
    seed: int = 42
) -> Dict[str, Any]:
    """
    Executes the full pipeline across T0 (Reference) and T1..T4 (Analysis streams).
    Uses a proper train/validation split of T0 so that the baseline cost comparison
    is computed on held-out data, preventing in-distribution batches (T1) from
    spuriously triggering retraining due to train-set optimism.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("CHRONOSGUARD: DATA DRIFT & MODEL DECAY DETECTION PIPELINE")
    print("=" * 70)
    
    # ----------------------------------------------
    # Step 1: Load or Generate Datasets
    # ----------------------------------------------
    print("\n[Step 1] Ingesting Temporal Stream Data...")
    batches = generate_all_periods(save_dir=data_dir, seed=seed)
    
    ref_df = batches["T0_Reference"]
    print(f"Reference Dataset (T0): {len(ref_df)} rows, Default Rate: {ref_df[TARGET_COLUMN].mean():.2%}")
    
    # ----------------------------------------------
    # Step 2: Train/Validation Split of T0
    # ----------------------------------------------
    print("\n[Step 2] Splitting T0 into Train (80%) / Validation (20%)...")
    np.random.seed(seed)
    perm = np.random.permutation(len(ref_df))
    split_idx = int(0.8 * len(ref_df))
    
    train_df = ref_df.iloc[perm[:split_idx]].copy().reset_index(drop=True)
    val_df   = ref_df.iloc[perm[split_idx:]].copy().reset_index(drop=True)
    
    print(f"Train Split: {len(train_df)} rows | Validation Split: {len(val_df)} rows")
    
    # ----------------------------------------------
    # Step 3: Train & Calibrate Baseline Classifier
    # ----------------------------------------------
    print("\n[Step 3] Training Baseline Classifier with Isotonic Calibration...")
    baseline_model, train_meta = train_baseline_model(
        train_df, FEATURE_COLUMNS, TARGET_COLUMN, random_state=seed
    )
    save_model(baseline_model, os.path.join(output_dir, "baseline_model.joblib"))
    print(f"Baseline Model Trained: ROC-AUC = {train_meta['train_auc']:.4f}, "
          f"Brier Loss = {train_meta['train_brier']:.4f}")
    
    # ----------------------------------------------
    # Step 4: Out-of-Sample Baseline Evaluation on Validation Set
    # ----------------------------------------------
    print("\n[Step 4] Evaluating Baseline on Held-Out Validation Set...")
    val_X = val_df[FEATURE_COLUMNS].values
    val_y = val_df[TARGET_COLUMN].values
    val_probs = baseline_model.predict_proba(val_X)[:, 1]
    
    val_eval = evaluate_predictions(val_y, val_probs)
    val_cbpe = estimate_metrics_cbpe(val_probs)
    val_op_cost = calculate_operational_cost(val_eval["fp"], val_eval["fn"])
    
    # Compute per-sample baseline cost for cross-batch normalization
    baseline_cost_per_sample = val_op_cost / len(val_df)
    
    print(f"Validation Baseline: AUC = {val_eval['roc_auc']:.4f} | F1 = {val_eval['f1']:.4f} | "
          f"Brier = {val_eval['brier_score']:.4f}")
    print(f"Baseline Cost/Sample = ${baseline_cost_per_sample:,.2f} | "
          f"Operational Cost = ${val_op_cost:,.2f}")
    
    # Full T0 reference features for drift detection comparisons
    ref_X = ref_df[FEATURE_COLUMNS].values
    ref_y = ref_df[TARGET_COLUMN].values
    
    # ----------------------------------------------
    # Step 5: Stream Processing across Analysis Batches
    # ----------------------------------------------
    analysis_batches = [
        "T1_Normal", "T2_Gradual_Drift", "T3_Sudden_Shock", "T4_Concept_Drift"
    ]
    batch_results = []
    univariate_all = []
    
    for batch_name in analysis_batches:
        print("\n" + "-" * 60)
        print(f"ANALYZING BATCH: {batch_name}")
        print("-" * 60)
        
        curr_df = batches[batch_name]
        curr_X = curr_df[FEATURE_COLUMNS].values
        curr_y = curr_df[TARGET_COLUMN].values
        
        # 5a. Model Inference
        curr_probs = baseline_model.predict_proba(curr_X)[:, 1]
        
        # 5b. Univariate Drift
        uni_summary = evaluate_univariate_drift(ref_df, curr_df, FEATURE_COLUMNS)
        uni_summary["batch_name"] = batch_name
        univariate_all.append(uni_summary)
        
        n_drifted_features = int((uni_summary["status"] == "DRIFT_DETECTED").sum())
        max_psi = float(uni_summary["psi"].max())
        top_drift_feature = uni_summary.iloc[0]["feature"]
        
        # 5c. Multivariate Drift (MMD)
        multi_eval = evaluate_multivariate_drift(ref_X, curr_X)
        
        # 5d. Adversarial Domain Discriminator
        adv_eval = evaluate_adversarial_drift(
            ref_df, curr_df, FEATURE_COLUMNS, random_state=seed
        )
        
        # 5e. Label-Free Performance Estimation (CBPE)
        cbpe_metrics = estimate_metrics_cbpe(curr_probs)
        
        # 5f. Ground-Truth Realized Decay
        ground_truth = evaluate_predictions(curr_y, curr_probs)
        
        # 5g. Cost of Decay vs Cost of Retraining
        # Per-sample cost normalization: compare current batch total cost against
        # baseline validation cost-per-sample projected onto current batch size.
        current_cost = calculate_operational_cost(
            ground_truth["fp"], ground_truth["fn"]
        )
        
        cost_decision = evaluate_retraining_from_costs(
            current_total_cost=current_cost,
            baseline_cost_per_sample=baseline_cost_per_sample,
            current_batch_size=len(curr_df)
        )
        
        # 5h. Adaptive Retraining Simulation
        retrain_result = None
        if cost_decision["should_retrain"]:
            print(f"  [+] Triggering automated retraining pipeline...")
            retrain_result = execute_retraining_pipeline(
                reference_df=ref_df,
                drifted_df=curr_df,
                feature_cols=FEATURE_COLUMNS,
                target_col=TARGET_COLUMN,
                random_state=seed
            )
            # Remove non-serializable model from JSON report
            retrain_result.pop("retrained_model", None)
            
        # -- Print Summary --
        print(f"  Drift:  Max PSI = {max_psi:.4f} ({top_drift_feature}) | "
              f"MMD = {multi_eval['mmd_score']:.4f} | Adv-AUC = {adv_eval['adversarial_auc']:.4f}")
        print(f"  Perf:   CBPE AUC = {cbpe_metrics['estimated_auc']:.4f} | "
              f"Realized AUC = {ground_truth['roc_auc']:.4f}")
        print(f"  Cost:   Current Loss = ${cost_decision['current_loss']:,.2f} | "
              f"Decay = +${cost_decision['decay_loss']:,.2f}")
        print(f"  Action: {cost_decision['recommendation']}")
        print(f"          {cost_decision['reason']}")
        
        batch_record = {
            "batch_name": batch_name,
            "sample_size": len(curr_df),
            "default_rate": round(float(curr_y.mean()), 4),
            "univariate": {
                "max_psi": max_psi,
                "top_drift_feature": top_drift_feature,
                "drifted_features_count": n_drifted_features,
                "features_summary": uni_summary.to_dict(orient="records")
            },
            "multivariate": multi_eval,
            "adversarial": adv_eval,
            "cbpe_label_free": cbpe_metrics,
            "ground_truth": ground_truth,
            "cost_decision": cost_decision,
            "retraining_mitigation": retrain_result
        }
        batch_results.append(batch_record)
        
    # ----------------------------------------------
    # Step 6: Export Telemetry
    # ----------------------------------------------
    combined_uni_df = pd.concat(univariate_all, ignore_index=True)
    combined_uni_df.to_csv(
        os.path.join(output_dir, "univariate_drift_matrix.csv"), index=False
    )
    
    full_report = {
        "framework": "ChronosGuard v1.0",
        "reference_baseline": {
            "total_samples": len(ref_df),
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "training_metrics": train_meta,
            "validation_eval": val_eval,
            "validation_cbpe": val_cbpe,
            "validation_operational_cost": val_op_cost,
            "baseline_cost_per_sample": round(baseline_cost_per_sample, 2),
            "baseline_fpr": round(val_eval["fp"] / max(1, val_eval["fp"] + val_eval["tn"]), 4),
            "baseline_fnr": round(val_eval["fn"] / max(1, val_eval["fn"] + val_eval["tp"]), 4)
        },
        "stream_evaluations": batch_results
    }
    
    report_path = os.path.join(output_dir, "pipeline_metrics.json")
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2)
        
    print("\n" + "=" * 70)
    print(f"Pipeline complete! Metrics exported to {report_path}")
    print("=" * 70)
    
    return full_report

if __name__ == "__main__":
    run_chronos_guard_pipeline()
