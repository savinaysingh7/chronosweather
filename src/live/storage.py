"""
ChronosGuard: Real-Time Live Stream Storage & Telemetry Database
Thread-safe SQLite storage with WAL (Write-Ahead Logging) for concurrent access
between background daemon producers and dashboard / consumer readers.
"""

import sqlite3
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

DEFAULT_DB_PATH = os.path.join("data", "live_stream.db")

class LiveStreamStorage:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for high-concurrency background writing & dashboard reading
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Live Ticks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS live_ticks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    prediction_label INTEGER,
                    prediction_prob REAL,
                    ground_truth_label INTEGER DEFAULT NULL,
                    resolved_at TEXT DEFAULT NULL,
                    is_drift_flagged INTEGER DEFAULT 0
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ticks_provider_time ON live_ticks(provider, timestamp);")

            # 2. Rolling Drift Snapshots
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS drift_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    window_size INTEGER NOT NULL,
                    max_psi REAL,
                    top_drift_feature TEXT,
                    mmd_score REAL,
                    adversarial_auc REAL,
                    cbpe_estimated_auc REAL,
                    realized_auc REAL DEFAULT NULL,
                    status TEXT,
                    recommendation TEXT,
                    action_detail TEXT,
                    conformal_set_size REAL DEFAULT NULL,
                    uncertainty_bandwidth REAL DEFAULT NULL
                );
            """)

            # Safe column additions if table already existed
            for col in ["conformal_set_size", "uncertainty_bandwidth"]:
                try:
                    cursor.execute(f"ALTER TABLE drift_snapshots ADD COLUMN {col} REAL DEFAULT NULL;")
                except sqlite3.OperationalError:
                    pass

            # 3. Retraining Log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS retrain_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    trigger_reason TEXT NOT NULL,
                    training_samples INTEGER NOT NULL,
                    pre_auc REAL,
                    post_auc REAL,
                    decay_loss_prevented REAL,
                    net_roi REAL
                );
            """)

            # 4. Daemon Status & Heartbeat
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daemon_state (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                );
            """)
            conn.commit()

    def record_tick(
        self,
        provider: str,
        features: Dict[str, float],
        prediction_label: int,
        prediction_prob: float,
        timestamp: Optional[str] = None
    ) -> int:
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO live_ticks (provider, timestamp, features_json, prediction_label, prediction_prob)
                VALUES (?, ?, ?, ?, ?)
            """, (provider, ts, json.dumps(features), prediction_label, float(prediction_prob)))
            conn.commit()
            return cursor.lastrowid

    def update_ground_truth(self, tick_id: int, label: int):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE live_ticks 
                SET ground_truth_label = ?, resolved_at = ?
                WHERE id = ?
            """, (label, datetime.now(timezone.utc).isoformat(), tick_id))
            conn.commit()

    def get_recent_ticks(self, provider: str, limit: int = 200) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, provider, timestamp, features_json, prediction_label, prediction_prob, ground_truth_label
                FROM live_ticks
                WHERE provider = ?
                ORDER BY id DESC
                LIMIT ?
            """, (provider, limit))
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item["features"] = json.loads(item.pop("features_json"))
                results.append(item)
            return results[::-1]  # Return in chronological order

    def get_unresolved_ticks(self, provider: str, older_than_seconds: float = 30.0) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, provider, timestamp, features_json, prediction_prob
                FROM live_ticks
                WHERE provider = ? AND ground_truth_label IS NULL
                ORDER BY id ASC
                LIMIT 100
            """, (provider,))
            rows = cursor.fetchall()
            results = []
            now = datetime.now(timezone.utc)
            for r in rows:
                item = dict(r)
                try:
                    tick_time = datetime.fromisoformat(item["timestamp"])
                    age = (now - tick_time).total_seconds()
                    if age >= older_than_seconds:
                        item["features"] = json.loads(item.pop("features_json"))
                        item["age_seconds"] = age
                        results.append(item)
                except Exception:
                    continue
            return results

    def record_drift_snapshot(
        self,
        provider: str,
        window_size: int,
        max_psi: float,
        top_drift_feature: str,
        mmd_score: float,
        adversarial_auc: float,
        cbpe_estimated_auc: float,
        realized_auc: Optional[float],
        status: str,
        recommendation: str,
        action_detail: str,
        conformal_set_size: Optional[float] = None,
        uncertainty_bandwidth: Optional[float] = None
    ) -> int:
        ts = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO drift_snapshots (
                    provider, timestamp, window_size, max_psi, top_drift_feature,
                    mmd_score, adversarial_auc, cbpe_estimated_auc, realized_auc,
                    status, recommendation, action_detail, conformal_set_size, uncertainty_bandwidth
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                provider, ts, window_size, float(max_psi), top_drift_feature,
                float(mmd_score), float(adversarial_auc), float(cbpe_estimated_auc),
                float(realized_auc) if realized_auc is not None else None,
                status, recommendation, action_detail,
                float(conformal_set_size) if conformal_set_size is not None else None,
                float(uncertainty_bandwidth) if uncertainty_bandwidth is not None else None
            ))
            conn.commit()
            return cursor.lastrowid

    def get_recent_snapshots(self, provider: str, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM drift_snapshots
                WHERE provider = ?
                ORDER BY id DESC
                LIMIT ?
            """, (provider, limit))
            return [dict(r) for r in cursor.fetchall()][::-1]

    def record_retrain_event(
        self,
        provider: str,
        trigger_reason: str,
        training_samples: int,
        pre_auc: float,
        post_auc: float,
        decay_loss_prevented: float,
        net_roi: float
    ):
        ts = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO retrain_events (
                    provider, timestamp, trigger_reason, training_samples,
                    pre_auc, post_auc, decay_loss_prevented, net_roi
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                provider, ts, trigger_reason, training_samples,
                float(pre_auc), float(post_auc), float(decay_loss_prevented), float(net_roi)
            ))
            conn.commit()

    def get_retrain_events(self, provider: str, limit: int = 20) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM retrain_events
                WHERE provider = ?
                ORDER BY id DESC
                LIMIT ?
            """, (provider, limit))
            return [dict(r) for r in cursor.fetchall()]

    def set_daemon_state(self, key: str, value: Any):
        ts = datetime.now(timezone.utc).isoformat()
        val_str = json.dumps(value) if not isinstance(value, str) else value
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO daemon_state (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """, (key, val_str, ts))
            conn.commit()

    def get_daemon_state(self, key: str) -> Optional[Any]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM daemon_state WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                try:
                    return json.loads(row["value"])
                except Exception:
                    return row["value"]
            return None

    def get_total_tick_count(self, provider: str) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as cnt FROM live_ticks WHERE provider = ?", (provider,))
            row = cursor.fetchone()
            return row["cnt"] if row else 0

    def purge_provider(self, provider: str):
        """Reset records for a specific provider to start a fresh live session"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM live_ticks WHERE provider = ?", (provider,))
            cursor.execute("DELETE FROM drift_snapshots WHERE provider = ?", (provider,))
            cursor.execute("DELETE FROM retrain_events WHERE provider = ?", (provider,))
            conn.commit()
