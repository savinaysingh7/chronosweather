"""
Unit tests for LLM & Generative AI Drift Detection Module
"""

import pytest
import numpy as np
from src.drift.llm_drift import (
    compute_lexical_diversity,
    compute_embedding_mmd,
    compute_semantic_centroid_shift,
    LlmDriftAnalyzer,
    get_llm_analyzer,
    BENCHMARK_PROMPTS
)

def test_lexical_diversity():
    texts = [
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox repeats repeats repeats"
    ]
    metrics = compute_lexical_diversity(texts)
    assert metrics["vocab_size"] > 0
    assert metrics["total_tokens"] > 0
    assert 0.0 < metrics["ttr"] <= 1.0
    assert metrics["entropy"] > 0.0

def test_embedding_mmd():
    rng = np.random.RandomState(42)
    ref = rng.normal(0, 1, size=(20, 10))
    # In-distribution sample
    in_dist = rng.normal(0, 1, size=(20, 10))
    # Drifted sample
    drifted = rng.normal(5, 1, size=(20, 10))
    
    mmd_in = compute_embedding_mmd(ref, in_dist)
    mmd_drift = compute_embedding_mmd(ref, drifted)
    
    assert mmd_in >= 0.0
    assert mmd_drift > mmd_in

def test_semantic_centroid_shift():
    ref = np.array([[1.0, 0.0], [0.9, 0.1]])
    curr_same = np.array([[1.0, 0.0], [0.95, 0.05]])
    curr_diff = np.array([[0.0, 1.0], [-0.1, 0.9]])
    
    shift_same = compute_semantic_centroid_shift(ref, curr_same)
    shift_diff = compute_semantic_centroid_shift(ref, curr_diff)
    
    assert shift_same < 0.1
    assert shift_diff > 0.5

def test_llm_drift_analyzer_cohorts():
    analyzer = get_llm_analyzer()
    
    # Analyze in-distribution diurnal
    res_l1 = analyzer.analyze_cohort("L1_Diurnal")
    assert res_l1["success"] is True
    assert res_l1["executive_summary"]["mmd_embedding_drift"] < 0.35
    
    # Analyze model collapse
    res_l3 = analyzer.analyze_cohort("L3_Model_Collapse")
    assert res_l3["success"] is True
    # Model collapse should exhibit lower entropy or higher collapse score
    assert res_l3["executive_summary"]["model_collapse_score"] > 0.1
    
    # Check scatter points and sample ledger
    assert len(res_l3["scatter_points"]) > 0
    assert len(res_l3["sample_ledger"]) > 0

def test_llm_drift_simulation():
    analyzer = get_llm_analyzer()
    res = analyzer.simulate_custom_drift(synthetic_ratio=0.9, prompt_shift_factor=0.8, geo_bias=0.5)
    assert res["success"] is True
    assert res["metrics"]["simulated_mmd"] > 0.2
    assert res["metrics"]["model_collapse_index"] > 0.5
    assert res["metrics"]["hallucination_risk_pct"] > 50.0
