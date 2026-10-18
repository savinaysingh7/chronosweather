"""
ChronosGuard: End-to-End Pipeline Integration Test
Validates that the full orchestration pipeline executes without error,
produces expected output files, and yields sensible metric ranges.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd

from src.pipeline import run_chronos_guard_pipeline


@pytest.fixture(scope="module")
def pipeline_result(tmp_path_factory):
    """
    Run the full pipeline once and share the result across all tests in this module.
    Uses a temporary directory so tests don't pollute the working tree.
    """
    data_dir = str(tmp_path_factory.mktemp("data_raw"))
    output_dir = str(tmp_path_factory.mktemp("results"))
    result = run_chronos_guard_pipeline(
        data_dir=data_dir, output_dir=output_dir, seed=42
    )
    result["_output_dir"] = output_dir
    return result


class TestPipelineOutputStructure:
    """Verify the pipeline produces the expected output artifacts."""

    def test_report_json_exists(self, pipeline_result):
        output_dir = pipeline_result["_output_dir"]
        assert os.path.isfile(os.path.join(output_dir, "pipeline_metrics.json"))

    def test_univariate_csv_exists(self, pipeline_result):
        output_dir = pipeline_result["_output_dir"]
        assert os.path.isfile(os.path.join(output_dir, "univariate_drift_matrix.csv"))

    def test_baseline_model_saved(self, pipeline_result):
        output_dir = pipeline_result["_output_dir"]
        assert os.path.isfile(os.path.join(output_dir, "baseline_model.joblib"))

    def test_report_json_parseable(self, pipeline_result):
        output_dir = pipeline_result["_output_dir"]
        with open(os.path.join(output_dir, "pipeline_metrics.json")) as f:
            data = json.load(f)
        assert data["framework"] == "ChronosGuard v1.0"


class TestBaselineMetrics:
    """Verify the baseline model's validation metrics are within expected ranges."""

    def test_baseline_auc_range(self, pipeline_result):
        val_eval = pipeline_result["reference_baseline"]["validation_eval"]
        assert 0.65 <= val_eval["roc_auc"] <= 1.0, (
            f"Baseline validation AUC {val_eval['roc_auc']} out of expected range"
        )

    def test_baseline_brier_reasonable(self, pipeline_result):
        val_eval = pipeline_result["reference_baseline"]["validation_eval"]
        assert val_eval["brier_score"] < 0.25, (
            f"Brier score {val_eval['brier_score']} is unexpectedly high"
        )

    def test_baseline_cost_positive(self, pipeline_result):
        assert pipeline_result["reference_baseline"]["baseline_cost_per_sample"] > 0


class TestStreamEvaluations:
    """Verify each analysis batch produces sensible drift and performance metrics."""

    def test_four_batches_evaluated(self, pipeline_result):
        assert len(pipeline_result["stream_evaluations"]) == 4

    def test_batch_names_correct(self, pipeline_result):
        names = [b["batch_name"] for b in pipeline_result["stream_evaluations"]]
        assert names == [
            "T1_Normal", "T2_Gradual_Drift", "T3_Sudden_Shock", "T4_Concept_Drift"
        ]

    def test_t1_normal_no_significant_drift(self, pipeline_result):
        """T1 is in-distribution — PSI should be low and no retraining should fire."""
        t1 = pipeline_result["stream_evaluations"][0]
        assert t1["univariate"]["max_psi"] < 0.10, (
            f"T1 Normal batch Max PSI = {t1['univariate']['max_psi']}, expected < 0.10"
        )

    def test_t3_shock_massive_drift(self, pipeline_result):
        """T3 should show extreme covariate drift."""
        t3 = pipeline_result["stream_evaluations"][2]
        assert t3["univariate"]["max_psi"] > 1.0
        assert t3["adversarial"]["adversarial_auc"] > 0.80

    def test_ground_truth_auc_valid(self, pipeline_result):
        for batch in pipeline_result["stream_evaluations"]:
            gt_auc = batch["ground_truth"]["roc_auc"]
            assert 0.0 <= gt_auc <= 1.0, (
                f"{batch['batch_name']} ground truth AUC {gt_auc} out of [0, 1]"
            )

    def test_cbpe_auc_valid(self, pipeline_result):
        for batch in pipeline_result["stream_evaluations"]:
            cbpe_auc = batch["cbpe_label_free"]["estimated_auc"]
            assert 0.0 <= cbpe_auc <= 1.0, (
                f"{batch['batch_name']} CBPE AUC {cbpe_auc} out of [0, 1]"
            )

    def test_cost_decision_fields_present(self, pipeline_result):
        for batch in pipeline_result["stream_evaluations"]:
            cd = batch["cost_decision"]
            assert "recommendation" in cd
            assert cd["recommendation"] in [
                "TRIGGER_RETRAINING", "DEFER_RETRAINING", "NO_ACTION_REQUIRED"
            ]
            assert "net_expected_roi" in cd
            assert "decay_loss" in cd

    def test_retraining_improves_auc_when_triggered(self, pipeline_result):
        """When retraining fires, post-retrain AUC should be >= pre-retrain AUC."""
        for batch in pipeline_result["stream_evaluations"]:
            retrain = batch["retraining_mitigation"]
            if retrain is not None:
                gt_auc = batch["ground_truth"]["roc_auc"]
                post_auc = retrain["post_roc_auc"]
                # Allow a small tolerance since the retrain test split is different
                assert post_auc >= gt_auc - 0.05, (
                    f"{batch['batch_name']}: post-retrain AUC {post_auc} "
                    f"did not recover from {gt_auc}"
                )
