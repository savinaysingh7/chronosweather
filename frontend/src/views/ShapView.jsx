import React, { useState, useEffect } from 'react';
import { Search, Flame, ArrowUpRight, ArrowDownRight, UserCheck, ShieldAlert, RefreshCw } from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Cell
} from 'recharts';
import { fetchShapAttribution, fetchShapSample } from '../api';

export default function ShapView({ selectedCohort = 'T3_Sudden_Shock' }) {
  const [cohort, setCohort] = useState(selectedCohort);
  const [shapData, setShapData] = useState(null);
  const [sampleIdx, setSampleIdx] = useState(0);
  const [sampleData, setSampleData] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        const [attrRes, sampleRes] = await Promise.all([
          fetchShapAttribution(cohort),
          fetchShapSample(cohort, sampleIdx)
        ]);
        if (attrRes && attrRes.success) setShapData(attrRes);
        if (sampleRes && sampleRes.success) setSampleData(sampleRes);
      } catch (err) {
        console.error('SHAP fetch error:', err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [cohort, sampleIdx]);

  const topDriver = shapData?.feature_attributions?.[0];
  const isRiskUp = topDriver ? topDriver.risk_direction === 'Increased Risk' : true;

  return (
    <div className="p-8 space-y-6 max-w-7xl mx-auto">
      {/* Header & Cohort Picker */}
      <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h3 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
            <Search className="w-5 h-5 text-purple-400" />
            SHAP Root-Cause Feature Attribution Studio
          </h3>
          <p className="text-xs text-slate-400">
            Shapley additive explanations decomposing prediction shifts into feature contributions
          </p>
        </div>

        <div className="flex items-center gap-3">
          {loading && (
            <div className="flex items-center gap-1.5 text-xs text-cyan-400 font-mono animate-pulse">
              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
              <span>Analyzing...</span>
            </div>
          )}
          <span className="text-xs text-slate-400">Cohort:</span>
          <select
            value={cohort}
            onChange={(e) => setCohort(e.target.value)}
            className="bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-white font-medium focus:outline-none focus:border-cyan-500 cursor-pointer"
          >
            <option value="T0_Reference">T0: Reference Baseline (Spring 2023)</option>
            <option value="T1_Normal">T1: Diurnal Variations (Normal)</option>
            <option value="T2_Gradual_Drift">T2: Summer Heatwave Creep</option>
            <option value="T3_Sudden_Shock">T3: Atmospheric Squall & Heat Dome</option>
            <option value="T4_Concept_Drift">T4: Grid Thermal De-Rating & Inverter Shock</option>
          </select>
        </div>
      </div>

      {/* Top Driver Callout Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-1">
            Top Meteorological Driver
          </span>
          <div className="text-2xl font-bold text-white font-mono flex items-center gap-2 truncate">
            <Flame className="w-5 h-5 text-rose-400 shrink-0" />
            <span>{shapData?.top_drift_driver || (loading ? 'Computing...' : 'None')}</span>
          </div>
          <span className="text-[11px] text-slate-400 mt-1 block">
            Largest attribution shift &Delta; |SHAP|
          </span>
        </div>

        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-1">
            Primary Impact Direction
          </span>
          <div className={`text-2xl font-bold font-mono flex items-center gap-1.5 ${isRiskUp ? 'text-rose-400' : 'text-emerald-400'}`}>
            {isRiskUp ? <ArrowUpRight className="w-6 h-6 shrink-0" /> : <ArrowDownRight className="w-6 h-6 shrink-0" />}
            <span>{topDriver ? topDriver.risk_direction : 'Analyzing...'}</span>
          </div>
          <span className="text-[11px] text-slate-400 mt-1 block">
            {isRiskUp ? 'Pushes grid telemetry toward peak strain / blackout risk' : 'Pushes predictions toward baseload grid stability'}
          </span>
        </div>

        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-1">
            Observation Under Inspection
          </span>
          <div className="text-2xl font-bold text-cyan-400 font-mono flex items-center gap-2">
            <UserCheck className="w-5 h-5 shrink-0" />
            <span>Observation #{sampleIdx}</span>
          </div>
          <span className="text-[11px] text-slate-400 mt-1 block">
            Strain Probability: <b>{sampleData?.predicted_probability != null ? `${(sampleData.predicted_probability * 100).toFixed(1)}%` : '0%'}</b>
          </span>
        </div>
      </div>

      {/* Global Attribution Shift Chart & Scorecard */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Bar Chart */}
        <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
          <div className="flex justify-between items-center">
            <div>
              <h4 className="text-sm font-bold text-white tracking-tight">
                Feature Attribution Shift (&Delta; |SHAP|) vs Baseline
              </h4>
              <p className="text-xs text-slate-400">
                Red: Pushes risk UP &middot; Green: Pushes risk DOWN
              </p>
            </div>
            <span className="text-[10px] font-mono text-purple-400 bg-purple-950/60 border border-purple-800/60 px-2 py-0.5 rounded">
              TreeExplainer
            </span>
          </div>

          <div className="h-72 w-full pt-2">
            {shapData?.feature_attributions && shapData.feature_attributions.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={shapData.feature_attributions}
                  layout="vertical"
                  margin={{ top: 5, right: 30, left: 50, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
                  <XAxis
                    type="number"
                    stroke="#94a3b8"
                    fontSize={11}
                    tickLine={false}
                    domain={['auto', 'auto']}
                    tickFormatter={(v) => v.toFixed(3)}
                  />
                  <YAxis dataKey="feature" type="category" stroke="#94a3b8" fontSize={11} tickLine={false} width={120} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0f172a',
                      borderColor: '#334155',
                      borderRadius: '0.75rem',
                      fontSize: '12px'
                    }}
                    formatter={(val) => [typeof val === 'number' ? (val > 0 ? `+${val.toFixed(4)}` : val.toFixed(4)) : val, 'Attribution Shift (Δ|SHAP|)']}
                  />
                  <Bar dataKey="attribution_shift" radius={3}>
                    {shapData.feature_attributions.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={entry.risk_direction === 'Increased Risk' ? '#ef4444' : '#10b981'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-xs text-slate-400">
                {loading ? 'Computing Shapley feature values...' : 'No attribution data available'}
              </div>
            )}
          </div>
        </div>

        {/* Feature Attribution Table */}
        <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
          <h4 className="text-sm font-bold text-white tracking-tight">
            Directional Attribution Scorecard
          </h4>
          <p className="text-xs text-slate-400">
            Attribution shift (&Delta; |SHAP|) and directional push on grid blackout/strain risk
          </p>

          <div className="overflow-x-auto rounded-xl border border-slate-800/80 max-h-96">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950 text-slate-400 font-semibold border-b border-slate-800">
                <tr>
                  <th className="py-2.5 px-3">Feature</th>
                  <th className="py-2.5 px-3">&Delta; |SHAP|</th>
                  <th className="py-2.5 px-3">Impact Direction</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/50 font-mono text-slate-300">
                {shapData?.feature_attributions?.map((f, i) => (
                  <tr key={i} className="hover:bg-slate-800/30">
                    <td className="py-2 px-3 font-semibold text-white">{f.feature}</td>
                    <td className="py-2 px-3 text-cyan-400">{f.attribution_shift > 0 ? `+${f.attribution_shift.toFixed(4)}` : f.attribution_shift.toFixed(4)}</td>
                    <td className="py-2 px-3">
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                          f.risk_direction === 'Increased Risk'
                            ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                            : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                        }`}
                      >
                        {f.risk_direction}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Individual Weather Observation Waterfall Inspector */}
      <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h4 className="text-sm font-bold text-white tracking-tight">
              Sensor Observation Drill-Down Waterfall (Local Explanation)
            </h4>
            <p className="text-xs text-slate-400">
              Inspect how specific weather parameters push an individual observation towards peak grid strain or baseload
            </p>
          </div>

          <div className="flex items-center gap-3">
            <span className="text-xs text-slate-400 font-medium">Observation Index:</span>
            <input
              type="range"
              min={0}
              max={50}
              value={sampleIdx}
              onChange={(e) => setSampleIdx(Number(e.target.value))}
              className="accent-cyan-500 cursor-pointer"
            />
            <span className="text-xs font-mono text-white bg-slate-950 px-2 py-1 rounded border border-slate-800">
              #{sampleIdx}
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
          {sampleData?.explanation && sampleData.explanation.length > 0 ? (
            sampleData.explanation.map((row, i) => (
              <div
                key={i}
                className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800/80 flex items-center justify-between"
              >
                <div>
                  <span className="text-xs font-bold text-white block">{row.feature}</span>
                  <span className="text-[11px] text-slate-400 font-mono">
                    Value: <b>{typeof row.feature_value === 'number' ? (row.feature_value >= 1000 ? row.feature_value.toLocaleString() : row.feature_value) : row.feature_value}</b>
                  </span>
                </div>
                <div className="text-right">
                  <span
                    className={`text-xs font-mono font-bold block ${
                      row.shap_value > 0 ? 'text-rose-400' : 'text-emerald-400'
                    }`}
                  >
                    {row.shap_value > 0 ? `+${row.shap_value.toFixed(4)}` : row.shap_value.toFixed(4)}
                  </span>
                  <span className={`text-[10px] font-semibold ${row.shap_value > 0 ? 'text-rose-400/80' : 'text-emerald-400/80'}`}>
                    {row.effect}
                  </span>
                </div>
              </div>
            ))
          ) : (
            <div className="col-span-2 py-6 text-center text-xs text-slate-400">
              {loading ? 'Calculating local sensor observation Shapley waterfall...' : 'No sensor observation data available.'}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
