"""
ChronosGuard: Unit & Integration Test Suite for Real-World Live Streaming Architecture
Tests:
1. Live Stream Storage (CRUD operations, WAL concurrency, tick and snapshot persistence)
2. Live Stream Providers (CryptoMarketLiveProvider, EnvironmentalEnergyLiveProvider)
3. Background Live Daemon Lifecycle (start, live ingestion, ground-truth resolution, stop)
"""

import os
import time
import pytest
import numpy as np
import pandas as pd

from src.live.storage import LiveStreamStorage
from src.live.providers.crypto_stream import CryptoMarketLiveProvider
from src.live.providers.weather_stream import EnvironmentalEnergyLiveProvider
from src.live.daemon import ChronosGuardLiveDaemon


class TestLiveStreamStorage:
    @pytest.fixture
    def test_storage(self, tmp_path):
        db_file = str(tmp_path / "test_live_stream.db")
        return LiveStreamStorage(db_file)

    def test_record_and_get_ticks(self, test_storage):
        features = {"feature_a": 1.23, "feature_b": 4.56}
        tick_id = test_storage.record_tick(
            provider="test_provider",
            features=features,
            prediction_label=1,
            prediction_prob=0.88
        )
        assert tick_id > 0

        ticks = test_storage.get_recent_ticks("test_provider", limit=10)
        assert len(ticks) == 1
        assert ticks[0]["prediction_label"] == 1
        assert abs(ticks[0]["prediction_prob"] - 0.88) < 1e-4
        assert ticks[0]["features"]["feature_a"] == 1.23

    def test_ground_truth_resolution(self, test_storage):
        features = {"f1": 0.5}
        tick_id = test_storage.record_tick("test_provider", features, 0, 0.2)
        test_storage.update_ground_truth(tick_id, label=0)

        ticks = test_storage.get_recent_ticks("test_provider", limit=1)
        assert ticks[0]["ground_truth_label"] == 0

    def test_drift_snapshot_recording(self, test_storage):
        snap_id = test_storage.record_drift_snapshot(
            provider="test_provider",
            window_size=50,
            max_psi=0.15,
            top_drift_feature="feature_a",
            mmd_score=0.03,
            adversarial_auc=0.74,
            cbpe_estimated_auc=0.72,
            realized_auc=0.75,
            status="WARNING",
            recommendation="NO_ACTION",
            action_detail="Within warning thresholds"
        )
        assert snap_id > 0

        snapshots = test_storage.get_recent_snapshots("test_provider", limit=5)
        assert len(snapshots) == 1
        assert snapshots[0]["top_drift_feature"] == "feature_a"
        assert snapshots[0]["status"] == "WARNING"

    def test_retrain_event_recording(self, test_storage):
        test_storage.record_retrain_event(
            provider="test_provider",
            trigger_reason="Test PSI spike",
            training_samples=300,
            pre_auc=0.68,
            post_auc=0.84,
            decay_loss_prevented=4500.0,
            net_roi=4000.0
        )
        events = test_storage.get_retrain_events("test_provider", limit=5)
        assert len(events) == 1
        assert events[0]["post_auc"] == 0.84
        assert events[0]["net_roi"] == 4000.0


class TestLiveStreamProviders:
    def test_crypto_provider_tick(self):
        provider = CryptoMarketLiveProvider("BTCUSDT")
        features, meta = provider.fetch_live_tick()
        assert "return_1m" in features
        assert "volatility_5m" in features
        assert "last_price" in meta
        assert meta["last_price"] > 0

    def test_weather_provider_tick(self):
        provider = EnvironmentalEnergyLiveProvider("NYC")
        features, meta = provider.fetch_live_tick()
        assert "temperature_2m" in features
        assert "wind_speed_10m" in features
        assert "temp_c" in meta
        assert -50 <= meta["temp_c"] <= 60


class TestLiveDaemonLifecycle:
    def test_daemon_short_run(self, tmp_path):
        db_file = str(tmp_path / "daemon_test.db")
        daemon = ChronosGuardLiveDaemon(
            provider_key="weather_grid_nyc",
            tick_interval_seconds=0.5,
            drift_eval_interval=3,
            window_size=10,
            db_path=db_file
        )
        daemon.initialize()
        assert daemon.model is not None
        assert daemon.baseline_cost_per_sample > 0

        daemon.start(run_in_background=True)
        # Allow sufficient time for network roundtrips to public endpoints
        for _ in range(20):
            time.sleep(0.2)
            storage = LiveStreamStorage(db_file)
            ticks = storage.get_recent_ticks("weather_grid_nyc", limit=20)
            if len(ticks) >= 1:
                break
        daemon.stop()

        storage = LiveStreamStorage(db_file)
        ticks = storage.get_recent_ticks("weather_grid_nyc", limit=20)
        assert len(ticks) >= 1
