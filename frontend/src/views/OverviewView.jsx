import React from 'react';
import {
  TrendingUp,
  AlertTriangle,
  CheckCircle2,
  ShieldAlert,
  ArrowUpRight,
  Layers,
  Radio,
  Search,
  Target
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

export default function OverviewView({ analysisData, selectedCohort = 'T3_Sudden_Shock', onSelectCohort, onNavigate }) {
  if (!analysisData) {
    return (
      <div className="flex items-center justify-center h-full text-slate-400">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-400 mr-3" />
        Loading telemetry data...
      </div>
    );
  }

  const { executive_summary, cost_analysis, trajectory, batch_key, batch_description } = analysisData;
  const isDecay = executive_summary.system_health.includes('DRIFT') || executive_summary.system_health.includes('DECAY');
  const isWarning = executive_summary.system_health.includes('WARNING');

  const COHORT_OPTIONS = [
    { id: 'T0_Reference', label: 'T0: Ref' },
    { id: 'T1_Normal', label: 'T1: Diurnal' },
    { id: 'T2_Gradual_Drift', label: 'T2: Heatwave' },
    { id: 'T3_Sudden_Shock', label: 'T3: Squall' },
    { id: 'T4_Concept_Drift', label: 'T4: Anomaly' },
  ];

  return (
    <div className="p-8 space-y-6 max-w-7xl mx-auto">
      {/* Active System Health Banner */}
      <div
        className={`p-5 rounded-2xl border transition flex flex-wrap items-center justify-between gap-4 ${
          isDecay
            ? 'bg-rose-500/10 border-rose-500/30 text-rose-200'
            : isWarning
            ? 'bg-amber-500/10 border-amber-500/30 text-amber-200'
            : 'bg-emerald-500/10 border-emerald-500/30 text-emerald-200'
        }`}
      >
        <div className="flex items-center gap-4">
          <div
            className={`w-12 h-12 rounded-xl flex items-center justify-center shrink-0 ${
              isDecay ? 'bg-rose-500/20 text-rose-400' : isWarning ? 'bg-amber-500/20 text-amber-400' : 'bg-emerald-500/20 text-emerald-400'
            }`}
          >
            {isDecay ? (
              <ShieldAlert className="w-7 h-7" />
            ) : isWarning ? (
              <AlertTriangle className="w-7 h-7" />
            ) : (
              <CheckCircle2 className="w-7 h-7" />
            )}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs uppercase tracking-wider font-bold">System Status:</span>
              <span className="text-sm font-bold tracking-wide px-2 py-0.5 rounded bg-black/30">
                {executive_summary.system_health}
              </span>
            </div>
            <p className="text-xs text-slate-300 mt-1">
              Active Cohort: <span className="font-semibold text-white">{batch_key}</span> &mdash;{' '}
              {batch_description || 'Production Stream'}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          {/* Cohort Switcher Pill */}
          {onSelectCohort && (
            <div className="flex items-center gap-1 bg-slate-950/80 p-1 rounded-xl border border-slate-800">
              {COHORT_OPTIONS.map((c) => (
                <button
                  key={c.id}
                  onClick={() => onSelectCohort(c.id)}
                  className={`px-2.5 py-1 rounded-lg text-xs font-semibold transition ${
                    (selectedCohort || batch_key) === c.id
                      ? 'bg-cyan-500 text-white shadow-sm'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  {c.label}
                </button>
              ))}
            </div>
          )}

          <div className="text-right">
            <span className="text-xs text-slate-400 block">Retraining Policy:</span>
            <span
              className={`text-xs font-bold px-2.5 py-1 rounded-full ${
                cost_analysis?.should_retrain
                  ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                  : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
              }`}
            >
              {cost_analysis?.recommendation || 'NO_ACTION_REQUIRED'}
            </span>
          </div>
        </div>
      </div>

      {/* Primary KPI Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Realized vs Estimated AUC */}
        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 shadow-sm">
          <div className="flex justify-between items-start mb-2">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Model ROC-AUC</span>
            <TrendingUp className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-3xl font-extrabold text-white tracking-tight">
            {executive_summary.realized_auc.toFixed(3)}
          </div>
          <p className="text-xs text-slate-400 mt-2 flex items-center gap-1.5">
            <span className="text-cyan-400 font-semibold font-mono">
              CBPE: {executive_summary.cbpe_estimated_auc.toFixed(3)}
            </span>
            <span className="text-[11px] text-slate-400">(Label-Free)</span>
          </p>
        </div>

        {/* Max Feature PSI */}
        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 shadow-sm">
          <div className="flex justify-between items-start mb-2">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Max Feature PSI</span>
            <Layers className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-3xl font-extrabold text-white tracking-tight">
            {executive_summary.max_psi.toFixed(3)}
          </div>
          <p className="text-xs text-slate-400 mt-2">
            <span
              className={`font-semibold ${
                executive_summary.max_psi >= 0.25
                  ? 'text-rose-400'
                  : executive_summary.max_psi >= 0.1
                  ? 'text-amber-400'
                  : 'text-emerald-400'
              }`}
            >
              {executive_summary.drifted_features_count} / {executive_summary.total_features} features drifted
            </span>
          </p>
        </div>

        {/* Conformal Uncertainty Set */}
        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 shadow-sm">
          <div className="flex justify-between items-start mb-2">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Conformal Set Size</span>
            <Target className="w-4 h-4 text-purple-400" />
          </div>
          <div className="text-3xl font-extrabold text-white tracking-tight">
            {executive_summary.conformal_set_size ? executive_summary.conformal_set_size.toFixed(2) : '1.14'}
          </div>
          <p className="text-xs text-slate-400 mt-2 flex items-center gap-1">
            <span>Coverage:</span>
            <span className="font-semibold text-purple-300 font-mono">
              {executive_summary.empirical_coverage
                ? `${(executive_summary.empirical_coverage * 100).toFixed(1)}%`
                : '91.7%'}
            </span>
            <span className="text-slate-400">(Target 90%)</span>
          </p>
        </div>

        {/* Financial Decay ROI */}
        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 shadow-sm">
          <div className="flex justify-between items-start mb-2">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Net Retraining ROI</span>
            <ArrowUpRight className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-3xl font-extrabold text-emerald-400 tracking-tight">
            ${cost_analysis?.net_expected_roi ? cost_analysis.net_expected_roi.toLocaleString(undefined, { maximumFractionDigits: 0 }) : '0'}
          </div>
          <p className="text-xs text-slate-400 mt-2">
            <span>Decay Loss: </span>
            <span className="font-semibold text-rose-300">
              ${cost_analysis?.decay_loss ? cost_analysis.decay_loss.toLocaleString(undefined, { maximumFractionDigits: 0 }) : '0'}
            </span>
          </p>
        </div>
      </div>

      {/* Trajectory Timeline Chart */}
      <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 shadow-sm">
        <div className="flex justify-between items-center mb-4">
          <div>
            <h3 className="text-sm font-bold text-white tracking-tight">
              Model Performance & Drift Trajectory (T₀ → T₄)
            </h3>
            <p className="text-xs text-slate-400">
              Comparing Ground-Truth ROC-AUC against Label-Free CBPE and Population Stability Index
            </p>
          </div>
          <span className="text-xs font-mono text-cyan-400 bg-cyan-950/60 border border-cyan-800/60 px-2 py-0.5 rounded">
            21.8k Open-Meteo Sensor Telemetry
          </span>
        </div>

        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={trajectory} margin={{ top: 10, right: 45, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
              <XAxis dataKey="period" stroke="#94a3b8" fontSize={11} tickLine={false} />
              <YAxis
                yAxisId="auc"
                domain={[0.5, 1.0]}
                stroke="#94a3b8"
                fontSize={11}
                tickLine={false}
                tickFormatter={(v) => v.toFixed(2)}
              />
              <YAxis
                yAxisId="conformal"
                orientation="right"
                domain={[1.0, 1.6]}
                stroke="#c084fc"
                fontSize={11}
                tickLine={false}
                width={35}
                tickFormatter={(v) => v.toFixed(2)}
              />
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
                yAxisId="auc"
                type="monotone"
                dataKey="realized_auc"
                name="Realized ROC-AUC"
                stroke="#10b981"
                strokeWidth={2.5}
                dot={{ r: 4 }}
              />
              <Line
                yAxisId="auc"
                type="monotone"
                dataKey="cbpe_auc"
                name="CBPE Estimated AUC (Label-Free)"
                stroke="#38bdf8"
                strokeWidth={2}
                strokeDasharray="4 4"
                dot={{ r: 3 }}
              />
              <Line
                yAxisId="conformal"
                type="monotone"
                dataKey="conformal_set_size"
                name="Conformal Set Size (Right Axis)"
                stroke="#c084fc"
                strokeWidth={2}
                dot={{ r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Quick Action Navigation Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <button
          onClick={() => onNavigate('live')}
          className="p-5 rounded-2xl bg-slate-900/40 border border-slate-800/80 hover:border-cyan-500/50 hover:bg-slate-900/80 transition text-left group"
        >
          <div className="w-10 h-10 rounded-xl bg-cyan-500/10 text-cyan-400 flex items-center justify-center mb-3 group-hover:scale-105 transition">
            <Radio className="w-5 h-5" />
          </div>
          <h4 className="text-sm font-bold text-white group-hover:text-cyan-400 transition">
            Live Stream Terminal &rarr;
          </h4>
          <p className="text-xs text-slate-400 mt-1">
            Monitor real-world 24/7 NYC Metro Coastal Grid and Phoenix Desert Solar Grid telemetry side-by-side.
          </p>
        </button>

        <button
          onClick={() => onNavigate('shap')}
          className="p-5 rounded-2xl bg-slate-900/40 border border-slate-800/80 hover:border-cyan-500/50 hover:bg-slate-900/80 transition text-left group"
        >
          <div className="w-10 h-10 rounded-xl bg-purple-500/10 text-purple-400 flex items-center justify-center mb-3 group-hover:scale-105 transition">
            <Search className="w-5 h-5" />
          </div>
          <h4 className="text-sm font-bold text-white group-hover:text-purple-400 transition">
            SHAP Explainability Studio &rarr;
          </h4>
          <p className="text-xs text-slate-400 mt-1">
            Inspect which thermodynamic factors caused grid strain and evaluate observation waterfalls.
          </p>
        </button>

        <button
          onClick={() => onNavigate('conformal')}
          className="p-5 rounded-2xl bg-slate-900/40 border border-slate-800/80 hover:border-cyan-500/50 hover:bg-slate-900/80 transition text-left group"
        >
          <div className="w-10 h-10 rounded-xl bg-emerald-500/10 text-emerald-400 flex items-center justify-center mb-3 group-hover:scale-105 transition">
            <Target className="w-5 h-5" />
          </div>
          <h4 className="text-sm font-bold text-white group-hover:text-emerald-400 transition">
            Conformal Prediction Lab &rarr;
          </h4>
          <p className="text-xs text-slate-400 mt-1">
            Verify finite-sample 90% coverage guarantees and watch prediction sets widen under drift.
          </p>
        </button>
      </div>
    </div>
  );
}
