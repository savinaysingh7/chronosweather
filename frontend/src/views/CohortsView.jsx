import React, { useState, useEffect } from 'react';
import { BarChart3, Layers, ShieldCheck, AlertTriangle, ArrowRight } from 'lucide-react';
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
import { analyzeCohort } from '../api';

export default function CohortsView({ initialCohort = 'T3_Sudden_Shock' }) {
  const [selectedCohort, setSelectedCohort] = useState(initialCohort);
  const [cohortData, setCohortData] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        const res = await analyzeCohort(selectedCohort);
        if (res.success) setCohortData(res);
      } catch (err) {
        console.error('Cohort analysis error:', err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [selectedCohort]);

  const uniSummary = cohortData?.univariate_drift || [];
  const multiDrift = cohortData?.multivariate_drift || {};
  const advDrift = cohortData?.adversarial_drift || {};
  const confMatrices = cohortData?.confusion_matrices || {};

  return (
    <div className="p-8 space-y-6 max-w-7xl mx-auto">
      {/* Cohort Selector Header */}
      <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h3 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-cyan-400" />
            Historical Cohort Analysis & Multi-Tier Drift Suite
          </h3>
          <p className="text-xs text-slate-400">
            Univariate statistical tests (PSI / KS) paired with multivariate kernel MMD & adversarial classifiers
          </p>
        </div>

        <div className="flex items-center gap-2 bg-slate-950 p-1 rounded-xl border border-slate-800">
          {[
            { id: 'T0_Reference', label: 'T0: Reference' },
            { id: 'T1_Normal', label: 'T1: Diurnal' },
            { id: 'T2_Gradual_Drift', label: 'T2: Heatwave' },
            { id: 'T3_Sudden_Shock', label: 'T3: Squall' },
            { id: 'T4_Concept_Drift', label: 'T4: Grid Anomaly' },
          ].map((c) => (
            <button
              key={c.id}
              onClick={() => setSelectedCohort(c.id)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
                selectedCohort === c.id
                  ? 'bg-cyan-500 text-white shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              {c.label}
            </button>
          ))}
        </div>
      </div>

      {/* High-Level Diagnostics Bar */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-1">
            Max Feature PSI
          </span>
          <div className="text-2xl font-bold font-mono text-white">
            {cohortData?.executive_summary?.max_psi?.toFixed(4) || '0.000'}
          </div>
          <span className="text-[11px] text-slate-400 mt-1 block">
            Threshold: &gt;0.25 critical drift
          </span>
        </div>

        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-1">
            MMD RBF Kernel Distance
          </span>
          <div className="text-2xl font-bold font-mono text-cyan-400">
            {multiDrift.mmd_score ? multiDrift.mmd_score.toFixed(4) : '0.000'}
          </div>
          <span className="text-[11px] text-slate-400 mt-1 block">
            Permutation p-value: <b>{multiDrift.p_value ? multiDrift.p_value.toFixed(3) : '1.000'}</b>
          </span>
        </div>

        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-1">
            Adversarial Domain AUC
          </span>
          <div className="text-2xl font-bold font-mono text-amber-400">
            {advDrift.adversarial_auc ? advDrift.adversarial_auc.toFixed(4) : '0.500'}
          </div>
          <span className="text-[11px] text-slate-400 mt-1 block">
            Joint distribution separability
          </span>
        </div>
      </div>

      {/* Univariate PSI Chart & Scorecard */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* PSI Bar Chart */}
        <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
          <h4 className="text-sm font-bold text-white tracking-tight">
            Population Stability Index (PSI) by Feature
          </h4>
          <p className="text-xs text-slate-400">
            Green (&lt;0.10) Stable &middot; Amber (0.10-0.25) Warning &middot; Red (&ge;0.25) Critical Drift
          </p>

          <div className="h-80 w-full pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={uniSummary} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
                <XAxis type="number" stroke="#94a3b8" fontSize={11} tickLine={false} domain={[0, 'auto']} />
                <YAxis dataKey="feature" type="category" stroke="#94a3b8" fontSize={11} tickLine={false} width={110} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0f172a',
                    borderColor: '#334155',
                    borderRadius: '0.75rem',
                    fontSize: '12px'
                  }}
                />
                <Bar dataKey="psi" radius={[0, 4, 4, 0]}>
                  {uniSummary.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.psi >= 0.25 ? '#ef4444' : entry.psi >= 0.1 ? '#f59e0b' : '#10b981'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Feature Scorecard Table */}
        <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
          <h4 className="text-sm font-bold text-white tracking-tight">
            Feature Drift Statistical Scorecard
          </h4>
          <p className="text-xs text-slate-400">
            Detailed two-sample KS statistics and normalized Wasserstein distances
          </p>

          <div className="overflow-x-auto rounded-xl border border-slate-800/80 max-h-96">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950 text-slate-400 font-semibold border-b border-slate-800">
                <tr>
                  <th className="py-2.5 px-3">Feature</th>
                  <th className="py-2.5 px-3">PSI</th>
                  <th className="py-2.5 px-3">KS Stat</th>
                  <th className="py-2.5 px-3">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/50 font-mono text-slate-300">
                {uniSummary.map((f, i) => (
                  <tr key={i} className="hover:bg-slate-800/30">
                    <td className="py-2 px-3 font-semibold text-white">{f.feature}</td>
                    <td className="py-2 px-3 text-cyan-400">{f.psi.toFixed(4)}</td>
                    <td className="py-2 px-3">{f.ks_statistic.toFixed(4)}</td>
                    <td className="py-2 px-3">
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                          f.status === 'DRIFT_DETECTED'
                            ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                            : f.status === 'WARNING'
                            ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                            : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                        }`}
                      >
                        {f.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Confusion Matrices: CBPE vs Realized */}
      <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4">
        <h4 className="text-sm font-bold text-white tracking-tight">
          Confusion Matrix Comparison: Label-Free CBPE vs Realized Ground Truth
        </h4>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* CBPE Expected */}
          <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 space-y-2">
            <span className="text-xs font-bold text-cyan-400 uppercase tracking-wider block">
              CBPE Expected Matrix (Zero Labels Needed)
            </span>
            <div className="grid grid-cols-2 gap-2 text-center font-mono">
              <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                <span className="text-[10px] text-slate-400 block">Expected TN</span>
                <span className="text-lg font-bold text-emerald-400">
                  {confMatrices?.cbpe_expected?.tn?.toFixed(0) || '0'}
                </span>
              </div>
              <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                <span className="text-[10px] text-slate-400 block">Expected FP</span>
                <span className="text-lg font-bold text-amber-400">
                  {confMatrices?.cbpe_expected?.fp?.toFixed(0) || '0'}
                </span>
              </div>
              <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                <span className="text-[10px] text-slate-400 block">Expected FN</span>
                <span className="text-lg font-bold text-rose-400">
                  {confMatrices?.cbpe_expected?.fn?.toFixed(0) || '0'}
                </span>
              </div>
              <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                <span className="text-[10px] text-slate-400 block">Expected TP</span>
                <span className="text-lg font-bold text-cyan-400">
                  {confMatrices?.cbpe_expected?.tp?.toFixed(0) || '0'}
                </span>
              </div>
            </div>
          </div>

          {/* Realized Ground Truth */}
          <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 space-y-2">
            <span className="text-xs font-bold text-emerald-400 uppercase tracking-wider block">
              Realized Ground-Truth Matrix (Delayed Labels)
            </span>
            <div className="grid grid-cols-2 gap-2 text-center font-mono">
              <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                <span className="text-[10px] text-slate-400 block">Realized TN</span>
                <span className="text-lg font-bold text-emerald-400">
                  {confMatrices?.realized?.tn || '0'}
                </span>
              </div>
              <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                <span className="text-[10px] text-slate-400 block">Realized FP</span>
                <span className="text-lg font-bold text-amber-400">
                  {confMatrices?.realized?.fp || '0'}
                </span>
              </div>
              <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                <span className="text-[10px] text-slate-400 block">Realized FN</span>
                <span className="text-lg font-bold text-rose-400">
                  {confMatrices?.realized?.fn || '0'}
                </span>
              </div>
              <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                <span className="text-[10px] text-slate-400 block">Realized TP</span>
                <span className="text-lg font-bold text-cyan-400">
                  {confMatrices?.realized?.tp || '0'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
