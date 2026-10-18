"""
ChronosGuard: Live Daemon Management & Controller
Provides CLI and programmatic commands to control and inspect the autonomous background engine,
supporting single live stream mode and synchronized dual-stream mode.
"""

import sys
import time
import json
from typing import Dict, Any, Optional, Union

from src.live.daemon import ChronosGuardLiveDaemon, DualChronosGuardDaemon, PROVIDER_REGISTRY
from src.live.storage import LiveStreamStorage

_ACTIVE_DAEMON: Optional[Union[ChronosGuardLiveDaemon, DualChronosGuardDaemon]] = None


def get_daemon_instance(provider_key: str = "weather_grid_nyc") -> Union[ChronosGuardLiveDaemon, DualChronosGuardDaemon]:
    global _ACTIVE_DAEMON
    if _ACTIVE_DAEMON is None or getattr(_ACTIVE_DAEMON, "provider_key", None) != provider_key:
        if provider_key == "dual_weather_grids":
            _ACTIVE_DAEMON = DualChronosGuardDaemon(tick_interval_seconds=2.5)
        else:
            _ACTIVE_DAEMON = ChronosGuardLiveDaemon(provider_key=provider_key, tick_interval_seconds=2.5)
    return _ACTIVE_DAEMON


def start_daemon(provider_key: str = "weather_grid_nyc", interval: float = 3.0) -> Dict[str, Any]:
    global _ACTIVE_DAEMON
    if _ACTIVE_DAEMON is not None:
        try:
            _ACTIVE_DAEMON.stop()
        except Exception:
            pass
        _ACTIVE_DAEMON = None

    if provider_key == "dual_weather_grids":
        daemon = DualChronosGuardDaemon(tick_interval_seconds=interval)
    else:
        daemon = ChronosGuardLiveDaemon(provider_key=provider_key, tick_interval_seconds=interval)
        
    _ACTIVE_DAEMON = daemon
    daemon.initialize()
    daemon.start(run_in_background=True)
    return {
        "status": "RUNNING",
        "provider": provider_key,
        "interval_seconds": interval
    }


def stop_daemon() -> Dict[str, Any]:
    global _ACTIVE_DAEMON
    storage = LiveStreamStorage()
    storage.set_daemon_state("status", "STOPPED")
    if _ACTIVE_DAEMON:
        try:
            _ACTIVE_DAEMON.stop()
        except Exception:
            pass
        _ACTIVE_DAEMON = None
    return {"status": "STOPPED"}


def get_daemon_status() -> Dict[str, Any]:
    storage = LiveStreamStorage()
    status = storage.get_daemon_state("status") or "STOPPED"
    provider = storage.get_daemon_state("provider") or "weather_grid_nyc"
    heartbeat = storage.get_daemon_state("heartbeat") or {}
    baseline = storage.get_daemon_state("baseline_meta") or {}
    
    nyc_ticks = storage.get_total_tick_count("weather_grid_nyc")
    phoenix_ticks = storage.get_total_tick_count("weather_grid_phoenix")
    if provider == "dual_weather_grids":
        total_ticks = nyc_ticks + phoenix_ticks
    else:
        total_ticks = storage.get_total_tick_count(provider)
    
    return {
        "status": status,
        "provider": provider,
        "heartbeat": heartbeat,
        "baseline": baseline,
        "total_ticks_recorded": total_ticks,
        "nyc_ticks": nyc_ticks,
        "phoenix_ticks": phoenix_ticks
    }


if __name__ == "__main__":
    action = sys.argv[1].lower() if len(sys.argv) > 1 else "status"
    provider = sys.argv[2] if len(sys.argv) > 2 else "crypto_btc"

    if action == "start":
        print(f"Starting ChronosGuard Live Daemon for '{provider}'...")
        if provider == "dual_crypto_weather":
            daemon = DualChronosGuardDaemon(tick_interval_seconds=2.5)
        else:
            daemon = ChronosGuardLiveDaemon(provider_key=provider, tick_interval_seconds=2.5)
        daemon.initialize()
        print("Initialization complete! Starting continuous live streaming...")
        daemon.start(run_in_background=False)

    elif action == "stop":
        print("Stopping daemon...")
        stop_daemon()
        print("Done.")

    elif action == "status":
        st = get_daemon_status()
        print(json.dumps(st, indent=2))
