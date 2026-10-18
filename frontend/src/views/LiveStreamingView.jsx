import React, { useState, useEffect } from 'react';
import {
  Radio,
  Play,
  Square,
  Activity,
  Layers,
  Target,
  RefreshCw,
  Coins,
  CloudSun,
  Flame,
  ArrowRight
} from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend
} from 'recharts';
import {
  fetchLiveStatus,
  startDaemon,
  stopDaemon,
  fetchLiveTicks,
  fetchLiveSnapshots,
  fetchLiveDual,
  simulateLiveTick
} from '../api';

export default function LiveStreamingView() {
  const [mode, setMode] = useState('single'); // 'single' or 'dual'
  const [provider, setProvider] = useState('weather_grid_nyc');
  const [statusInfo, setStatusInfo] = useState(null);
  const [ticks, setTicks] = useState([]);
  const [snapshots, setSnapshots] = useState([]);
  const [dualData, setDualData] = useState({ 
    nyc: { ticks: [], snapshots: [] }, 
    phoenix: { ticks: [], snapshots: [] },
    crypto: { ticks: [], snapshots: [] }, 
    weather: { ticks: [], snapshots: [] } 
  });
  const [loading, setLoading] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Poll live data
  const loadLiveData = async () => {
    try {
      const statusRes = await fetchLiveStatus();
      if (statusRes.success) setStatusInfo(statusRes.data);

      if (mode === 'dual') {
        const dualRes = await fetchLiveDual(20);
        if (dualRes.success) setDualData(dualRes);
      } else {
        const [ticksRes, snapsRes] = await Promise.all([
          fetchLiveTicks(provider, 25),
          fetchLiveSnapshots(provider, 30),
        ]);
        if (ticksRes.success) setTicks(ticksRes.ticks);
        if (snapsRes.success) setSnapshots(snapsRes.snapshots);
      }
    } catch (err) {
      console.error('Live polling error:', err);
    }
  };

  useEffect(() => {
    loadLiveData();
    if (!autoRefresh) return;
    const interval = setInterval(loadLiveData, 2500);
    return () => clearInterval(interval);
  }, [provider, mode, autoRefresh]);

  const handleStart = async (pKey) => {
    setLoading(true);
    try {
      await startDaemon(pKey || provider, 2.5);
      await loadLiveData();
    } finally {
      setLoading(false);
    }
  };

  const handleStop = async () => {
    setLoading(true);
    try {
      await stopDaemon();
      await loadLiveData();
    } finally {
      setLoading(false);
    }
  };

  const handleSimulate = async () => {
    setLoading(true);
    try {
      if (mode === 'dual') {
        await Promise.all([
          simulateLiveTick('weather_grid_nyc', 5),
          simulateLiveTick('weather_grid_phoenix', 5)
        ]);
      } else {
        await simulateLiveTick(provider, 5);
      }
      await loadLiveData();
    } catch (err) {
      console.error('Simulate error:', err);
    } finally {
      setLoading(false);
    }
  };

  const isRunning = statusInfo?.status === 'RUNNING';
  const heartbeat = statusInfo?.heartbeat || {};
  const latestSnapshot = snapshots.length > 0 ? snapshots[snapshots.length - 1] : null;

  return (
    <div className="p-8 space-y-6 max-w-7xl mx-auto">
      {/* View Mode & Controls Bar */}
      <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 flex flex-wrap items-center justify-between gap-4">
        {/* Mode Selector */}
        <div className="flex items-center gap-2 bg-slate-950 p-1 rounded-xl border border-slate-800">
          <button
            onClick={() => setMode('single')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
              mode === 'single' ? 'bg-cyan-500 text-white shadow-sm' : 'text-slate-400 hover:text-white'
            }`}
          >
            Single Stream Focus
          </button>
          <button
            onClick={() => {
              setMode('dual');
              setProvider('dual_weather_grids');
            }}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition flex items-center gap-1.5 ${
              mode === 'dual' ? 'bg-cyan-500 text-white shadow-sm' : 'text-slate-400 hover:text-white'
            }`}
          >
            <span>Side-by-Side Dual Grid Stream</span>
            <span className="text-[10px] px-1 py-0.2 bg-white/20 rounded font-mono">LIVE</span>
          </button>
        </div>

        {/* Provider Dropdown (if single) */}
        {mode === 'single' && (
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400 font-medium">Weather Grid Station:</span>
            <select
              value={provider}
              onChange={(e) => setProvider(e.target.value)}
              className="bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-white font-medium focus:outline-none focus:border-cyan-500 cursor-pointer"
            >
              <option value="weather_grid_nyc">⚡ Open-Meteo NYC (Coastal Metro Grid)</option>
              <option value="weather_grid_phoenix">☀️ Open-Meteo Phoenix (Desert Solar Grid)</option>
              <option value="weather_grid_london">🌧️ Open-Meteo London (Oceanic Mild Grid)</option>
              <option value="weather_grid_tokyo">🗾 Open-Meteo Tokyo (Subtropical Urban Grid)</option>
            </select>
          </div>
        )}

        {/* Actions */}
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="rounded bg-slate-950 border-slate-800 text-cyan-500 focus:ring-0"
            />
            Auto-refresh (2.5s)
          </label>

          <button
            onClick={handleSimulate}
            disabled={loading}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-cyan-500/15 border border-cyan-500/30 text-cyan-300 hover:bg-cyan-500/25 text-xs font-bold transition disabled:opacity-50"
            title="Inject 5 real-time ticks to test streaming charts"
          >
            <Activity className="w-3.5 h-3.5" />
            Inject Telemetry
          </button>

          {isRunning ? (
            <button
              onClick={handleStop}
              disabled={loading}
              className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-rose-500/15 border border-rose-500/30 text-rose-300 hover:bg-rose-500/25 text-xs font-bold transition disabled:opacity-50"
            >
              <Square className="w-3.5 h-3.5 fill-current" />
              Stop Background Daemon
            </button>
          ) : (
            <button
              onClick={() => handleStart(mode === 'dual' ? 'dual_weather_grids' : provider)}
              disabled={loading}
              className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-emerald-500/15 border border-emerald-500/30 text-emerald-300 hover:bg-emerald-500/25 text-xs font-bold transition disabled:opacity-50"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              Launch Daemon
            </button>
          )}
        </div>
      </div>

      {/* DUAL STREAM MODE VIEW */}
      {mode === 'dual' ? (
        <div className="space-y-6">
          <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800 text-xs text-slate-300 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
              <span>
                <b>Dual Weather Grid Active:</b> Comparing <b>NYC Coastal Grid (Humidity & Cold Influx)</b> against{' '}
                <b>Phoenix Desert Solar Grid (Extreme Heat & Solar Influx)</b> simultaneously in SQLite WAL mode.
              </span>
            </div>
            <span className="text-[11px] font-mono text-cyan-400">Zero API Key Dependencies &middot; Open-Meteo Free</span>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Column 1: Phoenix Desert Grid */}
            {(() => {
              const phxData = dualData.phoenix?.ticks?.length ? dualData.phoenix : (dualData.crypto || { ticks: [], snapshots: [] });
              return (
                <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4">
                  <div className="flex justify-between items-start">
                    <div className="flex items-center gap-2.5">
                      <div className="w-8 h-8 rounded-lg bg-amber-500/20 text-amber-400 flex items-center justify-center font-bold">
                        ☀️
                      </div>
                      <div>
                        <h4 className="text-sm font-bold text-white">Phoenix Desert Solar Grid</h4>
                        <p className="text-[11px] text-slate-400">Extreme Solar Radiation &middot; Inverter Peak Stress</p>
                      </div>
                    </div>
                    <span className="text-xs font-mono font-bold text-amber-400 bg-amber-950/60 border border-amber-800/60 px-2 py-0.5 rounded">
                      {phxData.ticks.length > 0
                        ? phxData.ticks[phxData.ticks.length - 1]?.features?.direct_radiation != null
                          ? `${phxData.ticks[phxData.ticks.length - 1].features.direct_radiation.toFixed(0)} W/m²`
                          : (phxData.ticks[phxData.ticks.length - 1]?.features?.temperature_2m != null
                              ? `${phxData.ticks[phxData.ticks.length - 1].features.temperature_2m.toFixed(1)} °C`
                              : 'Active')
                        : 'Connecting...'}
                    </span>
                  </div>

                  {/* Phoenix Ticks Table */}
                  <div className="overflow-x-auto max-h-60 rounded-xl border border-slate-800/80">
                    <table className="w-full text-left text-xs">
                      <thead className="bg-slate-950 text-slate-400 font-semibold border-b border-slate-800">
                        <tr>
                          <th className="py-2 px-3">Time</th>
                          <th className="py-2 px-3">Strain Prob</th>
                          <th className="py-2 px-3">State</th>
                          <th className="py-2 px-3">Radiation</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/50 font-mono">
                        {phxData.ticks.slice(-8).reverse().map((t, idx) => (
                          <tr key={idx} className="hover:bg-slate-800/30">
                            <td className="py-1.5 px-3 text-slate-400">{t.timestamp ? t.timestamp.slice(11, 19) : '--:--:--'}</td>
                            <td className="py-1.5 px-3 text-cyan-400">{(t.prediction_prob * 100).toFixed(1)}%</td>
                            <td className="py-1.5 px-3">
                              <span
                                className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                                  t.prediction_label === 1
                                    ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                                    : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                                }`}
                              >
                                {t.prediction_label === 1 ? 'PEAK STRAIN' : 'BASELOAD'}
                              </span>
                            </td>
                            <td className="py-1.5 px-3 text-slate-300">
                              {t.features?.direct_radiation != null 
                                ? `${t.features.direct_radiation.toFixed(0)} W/m²` 
                                : (t.features?.temperature_2m != null ? `${t.features.temperature_2m.toFixed(1)} °C` : 'N/A')}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              );
            })()}

            {/* Column 2: Weather Grid Stream */}
            {(() => {
              const nycData = dualData.nyc?.ticks?.length ? dualData.nyc : (dualData.weather || { ticks: [], snapshots: [] });
              return (
                <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4">
                  <div className="flex justify-between items-start">
                    <div className="flex items-center gap-2.5">
                      <div className="w-8 h-8 rounded-lg bg-blue-500/20 text-blue-400 flex items-center justify-center font-bold">
                        ⚡
                      </div>
                      <div>
                        <h4 className="text-sm font-bold text-white">NYC Metro Coastal Grid</h4>
                        <p className="text-[11px] text-slate-400">Diurnal Covariate Shift &middot; Humidity & Barometric Load</p>
                      </div>
                    </div>
                    <span className="text-xs font-mono font-bold text-blue-400 bg-blue-950/60 border border-blue-800/60 px-2 py-0.5 rounded">
                      {nycData.ticks.length > 0
                        ? nycData.ticks[nycData.ticks.length - 1]?.features?.surface_pressure != null
                          ? `${nycData.ticks[nycData.ticks.length - 1].features.surface_pressure.toFixed(1)} hPa`
                          : 'Active'
                        : 'Connecting...'}
                    </span>
                  </div>

                  {/* Weather Ticks Table */}
                  <div className="overflow-x-auto max-h-60 rounded-xl border border-slate-800/80">
                    <table className="w-full text-left text-xs">
                      <thead className="bg-slate-950 text-slate-400 font-semibold border-b border-slate-800">
                        <tr>
                          <th className="py-2 px-3">Time</th>
                          <th className="py-2 px-3">Strain Prob</th>
                          <th className="py-2 px-3">State</th>
                          <th className="py-2 px-3">Pressure</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/50 font-mono">
                        {nycData.ticks.slice(-8).reverse().map((t, idx) => (
                          <tr key={idx} className="hover:bg-slate-800/30">
                            <td className="py-1.5 px-3 text-slate-400">{t.timestamp ? t.timestamp.slice(11, 19) : '--:--:--'}</td>
                            <td className="py-1.5 px-3 text-cyan-400">{(t.prediction_prob * 100).toFixed(1)}%</td>
                            <td className="py-1.5 px-3">
                              <span
                                className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                                  t.prediction_label === 1
                                    ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                                    : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                                }`}
                              >
                                {t.prediction_label === 1 ? 'PEAK LOAD' : 'BASE LOAD'}
                              </span>
                            </td>
                            <td className="py-1.5 px-3 text-slate-300">
                              {t.features?.surface_pressure != null ? `${t.features.surface_pressure.toFixed(1)} hPa` : 'N/A'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              );
            })()}
          </div>

          {/* Synchronized Dual Stream Rolling Drift Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Phoenix Drift Telemetry */}
            {(() => {
              const phxSnaps = dualData.phoenix?.snapshots?.length ? dualData.phoenix.snapshots : (dualData.crypto?.snapshots || []);
              return (
                <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
                  <div className="flex justify-between items-center">
                    <div>
                      <h4 className="text-sm font-bold text-white tracking-tight">☀️ Phoenix Solar Grid Drift & Decay Telemetry</h4>
                      <p className="text-xs text-slate-400">Rolling window PSI, Adversarial AUC, and CBPE Estimated AUC</p>
                    </div>
                    <span className="text-[10px] font-mono text-amber-400 bg-amber-950/60 border border-amber-800/60 px-2 py-0.5 rounded">
                      {phxSnaps.length} Snapshots
                    </span>
                  </div>
                  <div className="h-56 w-full">
                    {phxSnaps.length > 0 ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={phxSnaps.map((s) => ({ ...s, timeShort: s.timestamp ? s.timestamp.slice(11, 19) : '' }))}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
                          <XAxis dataKey="timeShort" stroke="#94a3b8" fontSize={10} tickLine={false} />
                          <YAxis stroke="#94a3b8" fontSize={10} tickLine={false} domain={[0, 1.2]} />
                          <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '0.75rem', fontSize: '11px' }} />
                          <Legend verticalAlign="top" height={30} iconType="circle" />
                          <Line type="monotone" dataKey="max_psi" name="Max PSI" stroke="#ef4444" strokeWidth={2} dot={{ r: 2 }} />
                          <Line type="monotone" dataKey="adversarial_auc" name="Adversarial AUC" stroke="#f59e0b" strokeWidth={2} dot={{ r: 2 }} />
                          <Line type="monotone" dataKey="cbpe_estimated_auc" name="CBPE Est. AUC" stroke="#38bdf8" strokeWidth={2} strokeDasharray="3 3" dot={{ r: 2 }} />
                        </LineChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="h-full flex items-center justify-center text-xs text-slate-500">
                        Awaiting 30 rolling ticks to compute live desert grid drift statistics...
                      </div>
                    )}
                  </div>
                </div>
              );
            })()}

            {/* Weather Drift Telemetry */}
            {(() => {
              const nycSnaps = dualData.nyc?.snapshots?.length ? dualData.nyc.snapshots : (dualData.weather?.snapshots || []);
              return (
                <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
                  <div className="flex justify-between items-center">
                    <div>
                      <h4 className="text-sm font-bold text-white tracking-tight">⚡ NYC Coastal Grid Drift & Decay Telemetry</h4>
                      <p className="text-xs text-slate-400">Rolling window PSI, Adversarial AUC, and CBPE Estimated AUC</p>
                    </div>
                    <span className="text-[10px] font-mono text-blue-400 bg-blue-950/60 border border-blue-800/60 px-2 py-0.5 rounded">
                      {nycSnaps.length} Snapshots
                    </span>
                  </div>
                  <div className="h-56 w-full">
                    {nycSnaps.length > 0 ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={nycSnaps.map((s) => ({ ...s, timeShort: s.timestamp ? s.timestamp.slice(11, 19) : '' }))}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
                          <XAxis dataKey="timeShort" stroke="#94a3b8" fontSize={10} tickLine={false} />
                          <YAxis stroke="#94a3b8" fontSize={10} tickLine={false} domain={[0, 1.2]} />
                          <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '0.75rem', fontSize: '11px' }} />
                          <Legend verticalAlign="top" height={30} iconType="circle" />
                          <Line type="monotone" dataKey="max_psi" name="Max PSI" stroke="#ef4444" strokeWidth={2} dot={{ r: 2 }} />
                          <Line type="monotone" dataKey="adversarial_auc" name="Adversarial AUC" stroke="#38bdf8" strokeWidth={2} dot={{ r: 2 }} />
                          <Line type="monotone" dataKey="cbpe_estimated_auc" name="CBPE Est. AUC" stroke="#10b981" strokeWidth={2} strokeDasharray="3 3" dot={{ r: 2 }} />
                        </LineChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="h-full flex items-center justify-center text-xs text-slate-500">
                        Awaiting 30 rolling ticks to compute live coastal grid drift statistics...
                      </div>
                    )}
                  </div>
                </div>
              );
            })()}
          </div>
        </div>
      ) : (
        /* SINGLE STREAM FOCUS VIEW */
        <div className="space-y-6">
          {/* Top Live KPIs */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-1">
                Latest Sensor Reading
              </span>
              <div className="text-2xl font-bold text-white font-mono">
                {heartbeat?.latest_price_or_temp != null
                  ? `${heartbeat.latest_price_or_temp.toFixed(1)} °C`
                  : 'Awaiting Reading...'}
              </div>
              <span className="text-[11px] text-slate-400 mt-1 block">
                Total Ingested: <b>{statusInfo?.total_ticks_recorded || 0}</b>
              </span>
            </div>

            <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-1">
                Peak Strain Probability
              </span>
              <div className="text-2xl font-bold text-cyan-400 font-mono">
                {heartbeat?.latest_prob ? `${(heartbeat.latest_prob * 100).toFixed(1)}%` : '0.0%'}
              </div>
              <span className="text-[11px] text-slate-400 mt-1 block">Zero-Downtime Calibrated</span>
            </div>

            <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-1">
                Rolling Drift Threat Status
              </span>
              <div className="text-2xl font-bold font-mono">
                <span
                  className={
                    latestSnapshot?.status === 'DRIFT_DETECTED'
                      ? 'text-rose-400'
                      : latestSnapshot?.status === 'WARNING'
                      ? 'text-amber-400'
                      : 'text-emerald-400'
                  }
                >
                  {latestSnapshot?.status || 'INITIALIZING'}
                </span>
              </div>
              <span className="text-[11px] text-slate-400 mt-1 block">
                Max PSI: <b>{latestSnapshot?.max_psi?.toFixed(3) || '0.000'}</b>
              </span>
            </div>

            <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-1">
                CBPE Estimated AUC
              </span>
              <div className="text-2xl font-bold text-purple-400 font-mono">
                {latestSnapshot?.cbpe_estimated_auc?.toFixed(3) || '0.750'}
              </div>
              <span className="text-[11px] text-slate-400 mt-1 block">Label-Free Projected Decay</span>
            </div>
          </div>

          {/* Rolling Drift Telemetry Chart */}
          <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800">
            <div className="flex justify-between items-center mb-4">
              <div>
                <h3 className="text-sm font-bold text-white tracking-tight">
                  Rolling Window Drift & Decay Telemetry (Live Ingestion)
                </h3>
                <p className="text-xs text-slate-400">
                  Evaluated every 15 live ticks across PSI, Adversarial AUC, and CBPE Estimated AUC
                </p>
              </div>
              <span className="text-xs font-mono text-emerald-400 bg-emerald-950/60 border border-emerald-800/60 px-2.5 py-0.5 rounded">
                Live Pulse Active
              </span>
            </div>

            <div className="h-64 w-full">
              {snapshots.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={snapshots.map((s) => ({
                      ...s,
                      timeShort: s.timestamp ? s.timestamp.slice(11, 19) : '',
                    }))}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
                    <XAxis dataKey="timeShort" stroke="#94a3b8" fontSize={11} tickLine={false} />
                    <YAxis stroke="#94a3b8" fontSize={11} tickLine={false} domain={[0, 1.2]} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#0f172a',
                        borderColor: '#334155',
                        borderRadius: '0.75rem',
                        fontSize: '12px'
                      }}
                    />
                    <Legend verticalAlign="top" height={36} iconType="circle" />
                    <Line
                      type="monotone"
                      dataKey="max_psi"
                      name="Max Feature PSI"
                      stroke="#ef4444"
                      strokeWidth={2}
                      dot={{ r: 3 }}
                    />
                    <Line
                      type="monotone"
                      dataKey="adversarial_auc"
                      name="Adversarial Domain AUC"
                      stroke="#f59e0b"
                      strokeWidth={2}
                      dot={{ r: 3 }}
                    />
                    <Line
                      type="monotone"
                      dataKey="cbpe_estimated_auc"
                      name="CBPE Est. AUC"
                      stroke="#38bdf8"
                      strokeWidth={2}
                      strokeDasharray="4 4"
                      dot={{ r: 3 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-full flex items-center justify-center text-xs text-slate-400">
                  Accumulating live ticks in background (rolling window evaluated every 15 ticks)...
                </div>
              )}
            </div>
          </div>

          {/* Recent Live Ingested Ticks Table */}
          <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800">
            <h3 className="text-sm font-bold text-white tracking-tight mb-3">
              ⚡ Recent Real-World Ingested Records (Persistent SQLite WAL)
            </h3>
            <div className="overflow-x-auto rounded-xl border border-slate-800/80">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-950 text-slate-400 font-semibold border-b border-slate-800">
                  <tr>
                    <th className="py-2.5 px-4">Tick ID</th>
                    <th className="py-2.5 px-4">Timestamp</th>
                    <th className="py-2.5 px-4">Predicted Prob</th>
                    <th className="py-2.5 px-4">Model Output</th>
                    <th className="py-2.5 px-4">Ground Truth</th>
                    <th className="py-2.5 px-4">Key Features</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/50 font-mono text-slate-300">
                  {ticks.slice(-10).reverse().map((t, i) => (
                    <tr key={i} className="hover:bg-slate-800/30">
                      <td className="py-2 px-4 text-slate-400">#{t.id}</td>
                      <td className="py-2 px-4">{t.timestamp.slice(11, 19)}</td>
                      <td className="py-2 px-4 text-cyan-400 font-semibold">
                        {(t.prediction_prob * 100).toFixed(1)}%
                      </td>
                      <td className="py-2 px-4">
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                            t.prediction_label === 1
                              ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                              : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                          }`}
                        >
                          {t.prediction_label === 1 ? '🔴 STRAIN' : '🟢 BASELOAD'}
                        </span>
                      </td>
                      <td className="py-2 px-4">
                        <span className="text-slate-400 text-[11px]">
                          {t.ground_truth_label === null ? '⏳ PENDING' : (t.ground_truth_label === 1 ? '⚠️ STRAIN' : 'NOMINAL')}
                        </span>
                      </td>
                      <td className="py-2 px-4 text-slate-400 text-[11px] truncate max-w-xs">
                        {JSON.stringify(t.features)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
