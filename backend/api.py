"""
ChronosGuard Backend API
Comprehensive REST API for real-time live streaming, multi-tier drift detection,
conformal prediction uncertainty, SHAP explainability, and cost-aware model decay mitigation.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, Any

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import ChronosGuard core modules
from data.generator import generate_all_periods, generate_credit_batch
from src.model.trainer import train_baseline_model, save_model, load_model, FEATURE_COLUMNS, TARGET_COLUMN
from src.model.evaluator import evaluate_predictions
from src.model.conformal import ConformalPredictionClassifier
from src.drift.univariate import evaluate_univariate_drift
from src.drift.multivariate import evaluate_multivariate_drift
from src.drift.adversarial import evaluate_adversarial_drift
from src.drift.cbpe import estimate_metrics_cbpe
from src.drift.explainability import compute_shap_drift_attribution, explain_single_sample
from src.mitigation.cost_model import evaluate_retraining_decision, evaluate_retraining_from_costs, calculate_operational_cost
from src.mitigation.retrainer import execute_retraining_pipeline
from src.live.controller import start_daemon, stop_daemon, get_daemon_status
from src.live.storage import LiveStreamStorage

app = Flask(__name__)
CORS(app)  # Enable CORS for modern web frontend

# Global state to store initialized system
system_state = {
    'batches': None,
    'baseline_model': None,
    'conformal_calibrator': None,
    'ref_eval': None,
    'ref_cbpe': None,
    'baseline_fpr': None,
    'baseline_fnr': None,
    'ref_df': None,
    'initialized': False
}

def auto_init_system(seed=42):
    """Automatically initializes the baseline system if not already initialized"""
    if system_state['initialized']:
        return
    print("Auto-initializing ChronosGuard Engine...")
    batches = generate_all_periods(save_dir="data/raw", seed=seed)
    ref_df = batches["T0_Reference"]
    
    np.random.seed(seed)
    perm = np.random.permutation(len(ref_df))
    split_idx = int(0.8 * len(ref_df))
    train_df = ref_df.iloc[perm[:split_idx]].copy().reset_index(drop=True)
    val_df = ref_df.iloc[perm[split_idx:]].copy().reset_index(drop=True)
    
    model, train_meta = train_baseline_model(train_df, FEATURE_COLUMNS, TARGET_COLUMN, random_state=seed)
    
    val_X = val_df[FEATURE_COLUMNS].values
    val_y = val_df[TARGET_COLUMN].values
    val_probs = model.predict_proba(val_X)[:, 1]
    val_eval = evaluate_predictions(val_y, val_probs)
    val_cbpe = estimate_metrics_cbpe(val_probs)
    
    # Calibrate split conformal predictor on validation split (90% target coverage)
    calibrator = ConformalPredictionClassifier(model, alpha=0.10)
    calibrator.calibrate(val_X, val_y)
    
    n_val_neg = val_eval["tn"] + val_eval["fp"]
    n_val_pos = val_eval["tp"] + val_eval["fn"]
    baseline_fpr = val_eval["fp"] / n_val_neg if n_val_neg > 0 else 0.0
    baseline_fnr = val_eval["fn"] / n_val_pos if n_val_pos > 0 else 0.0
    
    system_state['batches'] = batches
    system_state['baseline_model'] = model
    system_state['conformal_calibrator'] = calibrator
    system_state['ref_eval'] = val_eval
    system_state['ref_cbpe'] = val_cbpe
    system_state['baseline_fpr'] = baseline_fpr
    system_state['baseline_fnr'] = baseline_fnr
    system_state['ref_df'] = ref_df
    
    # Precompute deterministic trajectory across batches once
    batch_order = ["T0_Reference", "T1_Normal", "T2_Gradual_Drift", "T3_Sudden_Shock", "T4_Concept_Drift"]
    traj_records = []
    for b_name in batch_order:
        b_df = batches[b_name]
        b_X = b_df[FEATURE_COLUMNS].values
        b_y = b_df[TARGET_COLUMN].values
        b_p = model.predict_proba(b_X)[:, 1]
        b_gt = evaluate_predictions(b_y, b_p)
        b_cbpe = estimate_metrics_cbpe(b_p)
        b_uni = evaluate_univariate_drift(ref_df, b_df, FEATURE_COLUMNS)
        b_conf = calibrator.evaluate_batch(b_X, b_y)
        traj_records.append({
            "period": b_name,
            "realized_auc": b_gt["roc_auc"],
            "cbpe_auc": b_cbpe["estimated_auc"],
            "max_psi": float(b_uni["psi"].max()),
            "default_rate": b_gt["default_rate"],
            "conformal_set_size": b_conf["mean_set_size"],
            "empirical_coverage": b_conf["empirical_coverage"]
        })
    system_state['trajectory_records'] = traj_records
    system_state['initialized'] = True
    print(f"ChronosGuard Engine ready. Baseline AUC={val_eval['roc_auc']:.4f}, Conformal q={calibrator.conformal_quantile:.4f}")
    
    # Pre-seed live stream tables with realistic ticks and drift snapshots if fresh
    try:
        storage = LiveStreamStorage()
        seed_initial_live_data(storage)
    except Exception as e:
        print(f"Live storage pre-seed note: {e}")

def seed_initial_live_data(storage: LiveStreamStorage):
    """Pre-seeds realistic live stream telemetry for NYC and Phoenix so dashboard is instantly rich."""
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    
    # 1. NYC Metro Coastal Grid
    if len(storage.get_recent_ticks('weather_grid_nyc', limit=5)) < 5:
        nyc_ref = system_state['batches']['T1_Normal']
        for i in range(25):
            row = nyc_ref.iloc[i % len(nyc_ref)]
            feat = {col: float(row[col]) for col in FEATURE_COLUMNS}
            prob = float(system_state['baseline_model'].predict_proba(row[FEATURE_COLUMNS].values.reshape(1, -1))[0, 1])
            label = int(prob >= 0.5)
            ts = (now - timedelta(seconds=(25 - i) * 3)).isoformat()
            storage.record_tick('weather_grid_nyc', feat, label, prob, timestamp=ts)
    
    if len(storage.get_recent_snapshots('weather_grid_nyc', limit=3)) < 3:
        for s_idx, (m_psi, m_auc, c_size) in enumerate([(0.045, 0.885, 1.05), (0.052, 0.880, 1.08), (0.058, 0.875, 1.10)]):
            storage.record_drift_snapshot(
                provider='weather_grid_nyc',
                window_size=30,
                max_psi=m_psi,
                top_drift_feature='relative_humidity_2m',
                mmd_score=0.012,
                adversarial_auc=0.518,
                cbpe_estimated_auc=m_auc,
                realized_auc=m_auc + 0.01,
                status='NOMINAL_STABLE',
                recommendation='NO_ACTION_REQUIRED',
                action_detail='Grid telemetry in-distribution',
                conformal_set_size=c_size,
                uncertainty_bandwidth=0.22
            )

    # 2. Phoenix Desert Solar Grid
    if len(storage.get_recent_ticks('weather_grid_phoenix', limit=5)) < 5:
        phx_ref = system_state['batches']['T2_Gradual_Drift']
        for i in range(25):
            row = phx_ref.iloc[i % len(phx_ref)]
            feat = {col: float(row[col]) for col in FEATURE_COLUMNS}
            prob = float(system_state['baseline_model'].predict_proba(row[FEATURE_COLUMNS].values.reshape(1, -1))[0, 1])
            label = int(prob >= 0.5)
            ts = (now - timedelta(seconds=(25 - i) * 3)).isoformat()
            storage.record_tick('weather_grid_phoenix', feat, label, prob, timestamp=ts)
        
    if len(storage.get_recent_snapshots('weather_grid_phoenix', limit=3)) < 3:
        for s_idx, (m_psi, m_auc, c_size) in enumerate([(0.112, 0.842, 1.18), (0.135, 0.825, 1.25), (0.158, 0.810, 1.32)]):
            storage.record_drift_snapshot(
                provider='weather_grid_phoenix',
                window_size=30,
                max_psi=m_psi,
                top_drift_feature='direct_radiation',
                mmd_score=0.065,
                adversarial_auc=0.642,
                cbpe_estimated_auc=m_auc,
                realized_auc=m_auc - 0.02,
                status='WARNING_ELEVATED',
                recommendation='MONITOR_THERMAL_LOAD',
                action_detail='Solar radiation climbing outside nominal envelope',
                conformal_set_size=c_size,
                uncertainty_bandwidth=0.34
            )

@app.route('/', methods=['GET'])
def index():
    """Root landing endpoint with system status and frontend link"""
    return jsonify({
        'service': 'ChronosWeather Reliability Vanguard REST API',
        'status': 'healthy',
        'version': '2.0.0',
        'domain': 'Weather Telemetry & Energy Grid Blackout Prevention',
        'frontend_url': 'http://localhost:3001',
        'endpoints': {
            'health': '/health',
            'cohorts': '/api/cohorts',
            'analyze': '/api/analyze/<batch_key>',
            'shap_explain': '/api/shap/explain/<batch_key>',
            'shap_sample': '/api/shap/sample/<batch_key>/<sample_idx>',
            'conformal_cohorts': '/api/conformal/cohorts',
            'live_status': '/api/live/status',
            'live_dual': '/api/live/dual',
            'retrain': '/api/retrain',
            'simulate': '/api/simulate'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'ChronosWeather Reliability Vanguard REST API',
        'version': '2.0.0',
        'initialized': system_state['initialized']
    })

@app.route('/api/initialize', methods=['POST'])
def initialize_system():
    """Explicitly initializes the system"""
    try:
        data = request.get_json() or {}
        seed = data.get('seed', 42)
        system_state['initialized'] = False
        auto_init_system(seed=seed)
        return jsonify({
            'success': True,
            'message': 'ChronosGuard system initialized successfully',
            'baseline_auc': system_state['ref_eval']['roc_auc'],
            'baseline_f1': system_state['ref_eval']['f1']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/cohorts', methods=['GET'])
def list_cohorts():
    """Lists available historical batches and their metadata"""
    auto_init_system()
    batch_options = {
        "T0_Reference": {
            "name": "T0: Reference Baseline (Spring 2023)",
            "scenario": "Nominal seasonal baseline atmospheric conditions & standard grid baseload",
            "period": "Jan - Mar 2023"
        },
        "T1_Normal": {
            "name": "T1: In-Distribution Stream (Summer 2023)",
            "scenario": "Standard day/night diurnal variations, mild cooling load within nominal envelope",
            "period": "Apr - Jun 2023"
        },
        "T2_Gradual_Drift": {
            "name": "T2: Gradual Covariate Shift (Summer Heatwave)",
            "scenario": "Persistent creeping summer heatwave, elevated temperatures & high solar irradiance",
            "period": "Jul - Sep 2023"
        },
        "T3_Sudden_Shock": {
            "name": "T3: Sudden Macro Shock (Atmospheric Squall)",
            "scenario": "Extreme heat dome followed by rapid barometric pressure drop & storm squalls",
            "period": "Oct - Dec 2023"
        },
        "T4_Concept_Drift": {
            "name": "T4: Structural Concept Drift (Grid Anomaly)",
            "scenario": "P(Grid Strain | Weather) structural shift: transmission line thermal de-rating & solar inverter clipping",
            "period": "Jan - Mar 2024"
        }
    }
    return jsonify({'success': True, 'cohorts': batch_options})

_cohort_analysis_cache = {}
_shap_analysis_cache = {}

@app.route('/api/analyze/<batch_key>', methods=['GET'])
def analyze_batch(batch_key):
    """Deep analysis of a specific cohort: Univariate, Multivariate, Adversarial, CBPE, Conformal"""
    try:
        if batch_key in _cohort_analysis_cache:
            return jsonify(_cohort_analysis_cache[batch_key])

        auto_init_system()
        if batch_key not in system_state['batches']:
            return jsonify({'success': False, 'error': f'Batch {batch_key} not found'}), 404
            
        current_df = system_state['batches'][batch_key]
        curr_X = current_df[FEATURE_COLUMNS].values
        curr_y = current_df[TARGET_COLUMN].values
        curr_probs = system_state['baseline_model'].predict_proba(curr_X)[:, 1]
        
        # Univariate Drift
        uni_summary = evaluate_univariate_drift(system_state['ref_df'], current_df, FEATURE_COLUMNS)
        max_psi = float(uni_summary["psi"].max())
        n_drifted = int((uni_summary["status"] == "DRIFT_DETECTED").sum())
        
        # Performance & CBPE
        cbpe_curr = estimate_metrics_cbpe(curr_probs)
        gt_curr = evaluate_predictions(curr_y, curr_probs)
        
        # Conformal Prediction
        conformal_eval = system_state['conformal_calibrator'].evaluate_batch(curr_X, curr_y)
        conformal_eval.pop("prediction_sets", None)
        conformal_eval.pop("sample_probabilities", None)
        conformal_eval.pop("uncertainty_bounds", None)
        
        # Cost Decision: Blackout Prevention ($8,500/MWh) vs Peaker Plant Reserve ($350/MWh)
        cost_decision = evaluate_retraining_decision(
            current_fp=gt_curr["fp"],
            current_fn=gt_curr["fn"],
            baseline_fp=system_state['baseline_fpr'] * (gt_curr["tn"] + gt_curr["fp"]),
            baseline_fn=system_state['baseline_fnr'] * (gt_curr["tp"] + gt_curr["fn"]),
            cost_fp=350.0,
            cost_fn=8500.0,
            retrain_expense=1200.0
        )
        
        # Multivariate & Adversarial
        multi_res = evaluate_multivariate_drift(system_state['ref_df'][FEATURE_COLUMNS].values, curr_X)
        adv_res = evaluate_adversarial_drift(system_state['ref_df'], current_df, FEATURE_COLUMNS)
        
        # Trajectory (slice from precomputed records)
        traj_records = []
        for r in system_state.get('trajectory_records', []):
            traj_records.append(r)
            if r["period"] == batch_key:
                break
                
        payload = {
            'success': True,
            'batch_key': batch_key,
            'sample_count': len(current_df),
            'executive_summary': {
                'system_health': 'STABLE' if max_psi < 0.1 and adv_res["adversarial_auc"] < 0.75 else ('WARNING' if max_psi < 0.25 else 'CRITICAL DRIFT'),
                'max_psi': max_psi,
                'drifted_features_count': n_drifted,
                'total_features': len(FEATURE_COLUMNS),
                'realized_auc': gt_curr["roc_auc"],
                'cbpe_estimated_auc': cbpe_curr["estimated_auc"],
                'realized_f1': gt_curr["f1"],
                'cbpe_estimated_f1': cbpe_curr["estimated_f1"],
                'conformal_set_size': conformal_eval["mean_set_size"],
                'empirical_coverage': conformal_eval["empirical_coverage"]
            },
            'cost_analysis': cost_decision,
            'univariate_drift': uni_summary.to_dict('records'),
            'multivariate_drift': multi_res,
            'adversarial_drift': adv_res,
            'conformal_evaluation': conformal_eval,
            'trajectory': traj_records,
            'confusion_matrices': {
                'realized': {'tp': gt_curr["tp"], 'fp': gt_curr["fp"], 'tn': gt_curr["tn"], 'fn': gt_curr["fn"]},
                'cbpe_expected': {'tp': cbpe_curr["expected_tp"], 'fp': cbpe_curr["expected_fp"], 'tn': cbpe_curr["expected_tn"], 'fn': cbpe_curr["expected_fn"]}
            }
        }
        _cohort_analysis_cache[batch_key] = payload
        return jsonify(payload)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/shap/explain/<batch_key>', methods=['GET'])
def get_shap_explanation(batch_key):
    """Computes global SHAP attribution shift comparing reference vs specified batch"""
    try:
        if batch_key in _shap_analysis_cache:
            return jsonify(_shap_analysis_cache[batch_key])

        auto_init_system()
        if batch_key not in system_state['batches']:
            return jsonify({'success': False, 'error': f'Batch {batch_key} not found'}), 404
            
        curr_df = system_state['batches'][batch_key]
        shap_res = compute_shap_drift_attribution(
            model=system_state['baseline_model'],
            reference_df=system_state['ref_df'],
            current_df=curr_df,
            feature_cols=FEATURE_COLUMNS,
            max_samples=150
        )
        
        attr_df = shap_res["feature_attribution_df"]
        payload = {
            'success': True,
            'batch_key': batch_key,
            'top_drift_driver': shap_res["top_drift_driver"],
            'feature_attributions': attr_df.to_dict('records')
        }
        _shap_analysis_cache[batch_key] = payload
        return jsonify(payload)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/shap/sample/<batch_key>/<int:sample_idx>', methods=['GET'])
def get_shap_sample_explanation(batch_key, sample_idx):
    """Computes sample-level SHAP explanation waterfall for a specific borrower"""
    try:
        auto_init_system()
        if batch_key not in system_state['batches']:
            return jsonify({'success': False, 'error': f'Batch {batch_key} not found'}), 404
            
        curr_df = system_state['batches'][batch_key]
        idx = max(0, min(sample_idx, len(curr_df) - 1))
        sample_row = curr_df.iloc[idx]
        
        sample_exp = explain_single_sample(
            model=system_state['baseline_model'],
            sample_series=sample_row,
            feature_cols=FEATURE_COLUMNS
        )
        
        pred_prob = float(system_state['baseline_model'].predict_proba(sample_row[FEATURE_COLUMNS].values.reshape(1, -1))[0, 1])
        
        return jsonify({
            'success': True,
            'batch_key': batch_key,
            'sample_index': idx,
            'predicted_probability': pred_prob,
            'top_risk_factor': sample_exp["top_risk_factor"],
            'explanation': sample_exp["explanation_df"].to_dict('records')
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/conformal/cohorts', methods=['GET'])
def get_conformal_cohorts():
    """Returns conformal uncertainty sets and coverage metrics across all timeline cohorts"""
    try:
        auto_init_system()
        batch_order = ["T0_Reference", "T1_Normal", "T2_Gradual_Drift", "T3_Sudden_Shock", "T4_Concept_Drift"]
        results = []
        for b_name in batch_order:
            b_df = system_state['batches'][b_name]
            b_X = b_df[FEATURE_COLUMNS].values
            b_y = b_df[TARGET_COLUMN].values
            eval_res = system_state['conformal_calibrator'].evaluate_batch(b_X, b_y)
            results.append({
                "batch": b_name,
                "mean_set_size": eval_res["mean_set_size"],
                "singleton_rate": eval_res["singleton_rate"],
                "ambiguity_rate": eval_res["ambiguity_rate"],
                "empty_rate": eval_res["empty_rate"],
                "empirical_coverage": eval_res["empirical_coverage"],
                "target_coverage": eval_res["target_coverage"],
                "bandwidth": eval_res["mean_uncertainty_bandwidth"],
                "status": eval_res["status"]
            })
        return jsonify({'success': True, 'cohorts': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ==========================================
# REAL-TIME LIVE STREAMING ENDPOINTS
# ==========================================

@app.route('/api/live/status', methods=['GET'])
def live_daemon_status():
    """Returns real-time status of the autonomous background streaming daemon"""
    try:
        status_info = get_daemon_status()
        is_running = (status_info.get('status') == 'RUNNING')
        return jsonify({'success': True, 'data': status_info, 'running': is_running})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/live/start', methods=['POST'])
def live_daemon_start():
    """Starts the autonomous background daemon on a real live stream"""
    try:
        data = request.get_json() or {}
        provider = data.get('provider', 'weather_grid_nyc')
        interval = float(data.get('interval', 2.5))
        res = start_daemon(provider_key=provider, interval=interval)
        return jsonify({'success': True, 'result': res})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/live/stop', methods=['POST'])
def live_daemon_stop():
    """Stops the autonomous background streaming daemon"""
    try:
        res = stop_daemon()
        return jsonify({'success': True, 'result': res})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/live/ticks', methods=['GET'])
def live_recent_ticks():
    """Returns the most recent live stream ticks recorded in persistent storage"""
    try:
        provider = request.args.get('provider', 'weather_grid_nyc')
        limit = int(request.args.get('limit', 40))
        storage = LiveStreamStorage()
        ticks = storage.get_recent_ticks(provider=provider, limit=limit)
        return jsonify({'success': True, 'count': len(ticks), 'ticks': ticks})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/live/snapshots', methods=['GET'])
def live_drift_snapshots():
    """Returns rolling window drift snapshots evaluated on the live stream"""
    try:
        provider = request.args.get('provider', 'weather_grid_nyc')
        limit = int(request.args.get('limit', 30))
        storage = LiveStreamStorage()
        snapshots = storage.get_recent_snapshots(provider=provider, limit=limit)
        return jsonify({'success': True, 'count': len(snapshots), 'snapshots': snapshots})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/live/dual', methods=['GET'])
def live_dual_telemetry():
    """Returns synchronized recent ticks and snapshots for NYC Coastal and Phoenix Desert Grids"""
    try:
        limit = int(request.args.get('limit', 20))
        storage = LiveStreamStorage()
        nyc_ticks = storage.get_recent_ticks('weather_grid_nyc', limit=limit)
        phx_ticks = storage.get_recent_ticks('weather_grid_phoenix', limit=limit)
        nyc_snaps = storage.get_recent_snapshots('weather_grid_nyc', limit=limit)
        phx_snaps = storage.get_recent_snapshots('weather_grid_phoenix', limit=limit)
        
        return jsonify({
            'success': True,
            'nyc': {'ticks': nyc_ticks, 'snapshots': nyc_snaps, 'name': 'NYC Metro Coastal Grid'},
            'phoenix': {'ticks': phx_ticks, 'snapshots': phx_snaps, 'name': 'Phoenix Desert Solar Grid'},
            'weather': {'ticks': nyc_ticks, 'snapshots': nyc_snaps, 'name': 'NYC Metro Coastal Grid'},
            'crypto': {'ticks': phx_ticks, 'snapshots': phx_snaps, 'name': 'Phoenix Desert Solar Grid'}
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/live/simulate_tick', methods=['POST'])
def live_simulate_tick():
    """Injects simulated live ticks for rapid demo/testing"""
    try:
        auto_init_system()
        data = request.get_json() or {}
        provider = data.get('provider', 'weather_grid_nyc')
        count = int(data.get('count', 5))
        storage = LiveStreamStorage()
        batch_source = system_state['batches']['T2_Gradual_Drift'] if 'phoenix' in provider else system_state['batches']['T1_Normal']
        
        for _ in range(count):
            sample_idx = np.random.randint(0, len(batch_source))
            row = batch_source.iloc[sample_idx]
            feat = {col: float(row[col]) for col in FEATURE_COLUMNS}
            prob = float(system_state['baseline_model'].predict_proba(row[FEATURE_COLUMNS].values.reshape(1, -1))[0, 1])
            label = int(prob >= 0.5)
            storage.record_tick(provider, feat, label, prob)
            
        return jsonify({'success': True, 'injected': count, 'provider': provider})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/retrain', methods=['POST'])
def trigger_retraining():
    """Trigger automated retraining pipeline"""
    try:
        auto_init_system()
        data = request.get_json() or {}
        batch_key = data.get('batch_key', 'T4_Concept_Drift')
        cost_fp = data.get('cost_fp', 350.0)
        cost_fn = data.get('cost_fn', 8500.0)
        
        current_df = system_state['batches'][batch_key]
        retrain_res = execute_retraining_pipeline(
            reference_df=system_state['ref_df'],
            drifted_df=current_df,
            feature_cols=FEATURE_COLUMNS,
            target_col=TARGET_COLUMN,
            cost_fp=cost_fp,
            cost_fn=cost_fn
        )
        _cohort_analysis_cache.clear()
        _shap_analysis_cache.clear()
        retrain_res.pop("retrained_model", None)
        return jsonify({'success': True, 'results': retrain_res})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/simulate', methods=['POST'])
def simulate_drift():
    """Generate synthetic stream with custom drift parameters"""
    try:
        auto_init_system()
        data = request.get_json() or {}
        n_samples = data.get('n_samples', 2500)
        covariate_drift = data.get('covariate_drift', 0.5)
        shock_factor = data.get('shock_factor', 0.0)
        concept_drift = data.get('concept_drift', 0.0)
        
        sim_df = generate_credit_batch(
            n_samples=n_samples,
            batch_name="Custom_Weather_Stream",
            period_id=99,
            covariate_drift_factor=covariate_drift,
            shock_factor=shock_factor,
            concept_drift_factor=concept_drift
        )
        
        sim_X = sim_df[FEATURE_COLUMNS].values
        sim_y = sim_df[TARGET_COLUMN].values
        sim_p = system_state['baseline_model'].predict_proba(sim_X)[:, 1]
        
        sim_uni = evaluate_univariate_drift(system_state['ref_df'], sim_df, FEATURE_COLUMNS)
        sim_cbpe = estimate_metrics_cbpe(sim_p)
        sim_gt = evaluate_predictions(sim_y, sim_p)
        sim_conf = system_state['conformal_calibrator'].evaluate_batch(sim_X, sim_y)
        
        return jsonify({
            'success': True,
            'simulated_batch': {
                'default_rate': float(sim_df['is_peak_strain'].mean()),
                'strain_rate': float(sim_df['is_peak_strain'].mean()),
                'sample_size': len(sim_df),
                'max_psi': float(sim_uni['psi'].max()),
                'realized_auc': sim_gt["roc_auc"],
                'cbpe_estimated_auc': sim_cbpe["estimated_auc"],
                'conformal_set_size': sim_conf["mean_set_size"],
                'empirical_coverage': sim_conf["empirical_coverage"]
            },
            'univariate_drift': sim_uni[["feature", "psi", "ks_statistic", "status"]].to_dict('records')
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    import threading
    port = int(os.environ.get("PORT", 5001))
    print(f"Starting ChronosWeather Backend API on port {port}...", flush=True)
    # Start auto-initialization in a thread so server listens immediately
    threading.Thread(target=auto_init_system, daemon=True).start()
    app.run(host='127.0.0.1', port=port, debug=False)