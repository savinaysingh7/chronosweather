import React, { useState, useEffect } from 'react';
import {
  Sparkles,
  Bot,
  BrainCircuit,
  Sliders,
  AlertTriangle,
  CheckCircle2,
  TrendingDown,
  Layers,
  FileText,
  DollarSign,
  Compass,
  ArrowRight,
  ShieldAlert,
  Zap,
  BookOpen
} from 'lucide-react';
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  Tooltip,
  CartesianGrid,
  Legend,
  LineChart,
  Line,
  ReferenceLine
} from 'recharts';
import { fetchLlmCohorts, analyzeLlmCohort, simulateLlmDrift } from '../api';

const INVERSE_LAW_DATA = [
  { synthetic_pct: 0, concept_drift: 0.95, model_collapse: 0.05, perf_delta: 0.10 },
  { synthetic_pct: 20, concept_drift: 0.88, model_collapse: 0.12, perf_delta: 0.18 },
  { synthetic_pct: 40, concept_drift: 0.72, model_collapse: 0.28, perf_delta: 0.35 },
  { synthetic_pct: 60, concept_drift: 0.51, model_collapse: 0.54, perf_delta: 0.62 },
  { synthetic_pct: 80, concept_drift: 0.30, model_collapse: 0.82, perf_delta: 0.86 },
  { synthetic_pct: 100, concept_drift: 0.12, model_collapse: 0.98, perf_delta: 0.96 }
];

export default function LlmDriftView() {
  const [selectedCohort, setSelectedCohort] = useState('L2_Domain_Creep');
  const [cohortsMeta, setCohortsMeta] = useState({});
  const [analysisData, setAnalysisData] = useState(null);
  const [loading, setLoading] = useState(false);

  // Simulation Sliders State
  const [syntheticRatio, setSyntheticRatio] = useState(0.40);
  const [promptShift, setPromptShift] = useState(0.55);
  const [geoBias, setGeoBias] = useState(0.30);
  const [simResults, setSimResults] = useState(null);

  // Load Cohorts Metadata once
  useEffect(() => {
    async function loadCohorts() {
      try {
        const res = await fetchLlmCohorts();
        if (res.success) {
          setCohortsMeta(res.cohorts);
        }
      } catch (err) {
        console.error('Failed to load LLM cohorts:', err);
      }
    }
    loadCohorts();
  }, []);

  // Fetch telemetry when selected cohort changes
  useEffect(() => {
    async function runAnalysis() {
      setLoading(true);
      try {
        const res = await analyzeLlmCohort(selectedCohort);
        if (res.success) {
          setAnalysisData(res);
        }
      } catch (err) {
        console.error('Failed to analyze LLM cohort:', err);
      } finally {
        setLoading(false);
      }
    }
    runAnalysis();
  }, [selectedCohort]);

  // Run real-time simulation whenever sliders adjust
  useEffect(() => {
    let active = true;
    async function runSimulation() {
      try {
        const res = await simulateLlmDrift(syntheticRatio, promptShift, geoBias);
        if (active && res.success) {
          setSimResults(res.metrics);
        }
      } catch (err) {
        console.error('Simulation error:', err);
      }
    }
    runSimulation();
    return () => {
      active = false;
    };
  }, [syntheticRatio, promptShift, geoBias]);

  const summary = analysisData?.executive_summary || {};
  const metadata = analysisData?.metadata || {};
  const scatterPoints = analysisData?.scatter_points || [];
  const refPoints = scatterPoints.filter((p) => p.type.startsWith('Reference'));
  const currPoints = scatterPoints.filter((p) => !p.type.startsWith('Reference'));
  const sampleLedger = analysisData?.sample_ledger || [];

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto">
      {/* Academic Citation Header Banner */}
      <div className="p-6 rounded-2xl bg-gradient-to-r from-slate-900 via-indigo-950/40 to-slate-900 border border-indigo-500/30 shadow-xl">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="space-y-2">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-bold uppercase tracking-wider bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 flex items-center gap-1.5">
                <Sparkles className="w-3 h-3 text-indigo-400" />
                IEEE 2025 Benchmark Standard
              </span>
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-mono text-cyan-300 bg-cyan-950/60 border border-cyan-800/60">
                DOI: 10.1109/ICCIAA65327.2025.11013692
              </span>
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-medium text-emerald-300 bg-emerald-950/50 border border-emerald-800/50">
                100% Free Local Execution
              </span>
            </div>
            <h2 className="text-xl font-extrabold text-white tracking-tight flex items-center gap-2">
              <BrainCircuit className="w-6 h-6 text-indigo-400" />
              LLM & Generative AI Semantic Drift Studio
            </h2>
            <p className="text-xs text-slate-300 max-w-3xl leading-relaxed">
              Monitoring high-dimensional embedding divergence, recursive synthetic model collapse{' '}
              <span className="font-mono text-indigo-300 font-semibold">(D(t) ∝ 1/M(t))</span>, Shannon vocabulary entropy, and hallucination escalation without proprietary API costs.
            </p>
          </div>

          <div className="text-right shrink-0">
            <div className="text-xs font-semibold text-slate-400">Target Framework</div>
            <div className="text-sm font-bold text-white font-mono">Kernel MMD + RBF Latent PCA</div>
            <div className="text-[11px] text-indigo-400">Siciliani (2024) & Hajmohammed et al. (2025)</div>
          </div>
        </div>
      </div>

      {/* Cohort Selector Navigation Pills */}
      <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 flex flex-wrap items-center justify-between gap-4">
        <div>
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Select Evaluation Cohort:
          </span>
          <div className="text-sm font-bold text-white">
            {metadata.title || selectedCohort}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2 bg-slate-950 p-1.5 rounded-xl border border-slate-800">
          {[
            { id: 'L0_Reference', label: 'L0: Human Reference', badge: 'Baseline' },
            { id: 'L1_Diurnal_Style', label: 'L1: Diurnal Style', badge: 'Syntax' },
            { id: 'L2_Domain_Creep', label: 'L2: EU AI Act & GEO', badge: 'Reg/Web' },
            { id: 'L3_Model_Collapse', label: 'L3: Model Collapse', badge: 'Synthetic' },
            { id: 'L4_Semantic_Inversion', label: 'L4: Semantic Inversion', badge: 'Critical' },
          ].map((c) => (
            <button
              key={c.id}
              onClick={() => setSelectedCohort(c.id)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition flex items-center gap-1.5 ${
                selectedCohort === c.id
                  ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
              }`}
            >
              <span>{c.label}</span>
              <span
                className={`text-[9px] px-1 py-0.2 rounded font-mono ${
                  selectedCohort === c.id ? 'bg-indigo-700 text-indigo-100' : 'bg-slate-800 text-slate-400'
                }`}
              >
                {c.badge}
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* Cohort Description Card */}
      <div className="px-5 py-3.5 rounded-xl bg-slate-900/40 border border-slate-800/80 flex items-start gap-3">
        <BookOpen className="w-4 h-4 text-indigo-400 mt-0.5 shrink-0" />
        <div className="text-xs text-slate-300">
          <strong className="text-white">Active Scenario: </strong>
          {metadata.scenario || 'Analyzing semantic shift across query token distributions.'}
        </div>
      </div>

      {/* 4 Primary KPI Executive Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Card 1: Embedding MMD */}
        <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">
              Embedding MMD Drift
            </span>
            <Compass className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-2xl font-black text-white font-mono">
            {summary.mmd_embedding_drift !== undefined ? summary.mmd_embedding_drift.toFixed(3) : '---'}
          </div>
          <div className="flex items-center justify-between text-xs pt-1 border-t border-slate-800/80">
            <span className="text-slate-400">Centroid Shift:</span>
            <span className="font-mono text-cyan-400 font-semibold">
              {summary.centroid_cosine_shift !== undefined ? summary.centroid_cosine_shift.toFixed(3) : '---'}
            </span>
          </div>
          <div className="text-[11px] text-slate-400">
            Threshold: <span className="text-slate-300 font-mono">≥ 0.18</span> Caution, <span className="text-slate-300 font-mono">≥ 0.35</span> Critical
          </div>
        </div>

        {/* Card 2: Lexical Entropy & TTR */}
        <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">
              Shannon Token Entropy
            </span>
            <TrendingDown className="w-4 h-4 text-purple-400" />
          </div>
          <div className="text-2xl font-black text-white font-mono">
            {summary.vocabulary_entropy !== undefined ? summary.vocabulary_entropy.toFixed(2) : '---'}{' '}
            <span className="text-xs text-slate-400 font-normal">bits</span>
          </div>
          <div className="flex items-center justify-between text-xs pt-1 border-t border-slate-800/80">
            <span className="text-slate-400">Type-Token Ratio (TTR):</span>
            <span className="font-mono text-purple-400 font-semibold">
              {summary.type_token_ratio !== undefined ? summary.type_token_ratio.toFixed(3) : '---'}
            </span>
          </div>
          <div className="text-[11px] text-slate-400">
            Baseline Entropy: <span className="font-mono text-slate-300">{summary.baseline_entropy?.toFixed(2) || '5.54'} bits</span>
          </div>
        </div>

        {/* Card 3: Model Collapse Score */}
        <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">
              Model Collapse Index
            </span>
            <ShieldAlert className="w-4 h-4 text-rose-400" />
          </div>
          <div className="text-2xl font-black text-rose-400 font-mono">
            {summary.model_collapse_score !== undefined ? (summary.model_collapse_score * 100).toFixed(1) : '---'}%
          </div>
          <div className="flex items-center justify-between text-xs pt-1 border-t border-slate-800/80">
            <span className="text-slate-400">Synthetic Saturation:</span>
            <span className="font-mono text-amber-400 font-semibold">
              {summary.synthetic_saturation !== undefined ? `${(summary.synthetic_saturation * 100).toFixed(0)}%` : '---'}
            </span>
          </div>
          <div className="text-[11px] text-slate-400">
            Homogenization Law: <span className="text-indigo-300 font-mono">M(t) ∝ S(t)</span>
          </div>
        </div>

        {/* Card 4: Hallucination Risk & Status */}
        <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">
              Hallucination Risk
            </span>
            <Zap className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-black text-amber-400 font-mono">
            {summary.hallucination_risk_pct !== undefined ? `${summary.hallucination_risk_pct}%` : '---'}
          </div>
          <div className="flex items-center justify-between text-xs pt-1 border-t border-slate-800/80">
            <span className="text-slate-400">Health State:</span>
            <span
              className={`font-mono text-xs font-bold px-1.5 py-0.5 rounded ${
                summary.status === 'NOMINAL_STABLE'
                  ? 'bg-emerald-500/20 text-emerald-300'
                  : summary.status === 'ELEVATED_SEMANTIC_DRIFT'
                  ? 'bg-amber-500/20 text-amber-300'
                  : 'bg-rose-500/20 text-rose-300'
              }`}
            >
              {summary.status || 'ANALYZING'}
            </span>
          </div>
          <div className="text-[11px] text-slate-400 truncate">
            Rec: <span className="text-cyan-300 font-mono">{summary.recommendation || 'MONITOR'}</span>
          </div>
        </div>
      </div>

      {/* Two Interactive Charts Side by Side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Chart: 2D PCA Semantic Vector Scatter Plot */}
        <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-md space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <Compass className="w-4 h-4 text-cyan-400" />
                2D PCA Latent Semantic Space
              </h3>
              <p className="text-xs text-slate-400">
                Visualizing prompt embedding divergence against human-curated reference anchor
              </p>
            </div>
            <span className="text-[11px] font-mono text-slate-400 bg-slate-800 px-2 py-0.5 rounded">
              TF-IDF + PCA
            </span>
          </div>

          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.4} />
                <XAxis
                  type="number"
                  dataKey="x"
                  name="PCA 1"
                  stroke="#64748b"
                  fontSize={11}
                  domain={[-1, 1]}
                />
                <YAxis
                  type="number"
                  dataKey="y"
                  name="PCA 2"
                  stroke="#64748b"
                  fontSize={11}
                  domain={[-1, 1]}
                />
                <Tooltip
                  cursor={{ strokeDasharray: '3 3' }}
                  content={({ payload }) => {
                    if (payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="bg-slate-900 border border-slate-700 p-2.5 rounded-lg shadow-xl text-xs space-y-1">
                          <p className="font-bold text-white">{data.label}</p>
                          <p className="text-slate-300">{data.type}</p>
                          <p className="font-mono text-cyan-400">
                            X: {data.x}, Y: {data.y}
                          </p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Legend
                  verticalAlign="top"
                  align="right"
                  wrapperStyle={{ fontSize: '11px', paddingBottom: '10px' }}
                />
                <Scatter
                  name="Reference (L0 Baseline)"
                  data={refPoints}
                  fill="#06b6d4"
                  shape="circle"
                />
                <Scatter
                  name={`Current (${selectedCohort})`}
                  data={currPoints}
                  fill="#f43f5e"
                  shape="triangle"
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <div className="text-[11px] text-slate-400 flex items-center justify-between">
            <span>🔵 Baseline Human Anchors</span>
            <span className="text-rose-400">▲ Current Cohort Outliers</span>
          </div>
        </div>

        {/* Right Chart: Inverse Law Curve (Paper Theorem) */}
        <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-md space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <BrainCircuit className="w-4 h-4 text-indigo-400" />
                Theorem 1: Model Collapse vs. Concept Drift
              </h3>
              <p className="text-xs text-slate-400">
                IEEE 2025 finding: Synthetic text homogenizes data, masking drift while inducing collapse
              </p>
            </div>
            <span className="text-[11px] font-mono text-indigo-300 bg-indigo-950/60 border border-indigo-800/60 px-2 py-0.5 rounded">
              D(t) ∝ 1/M(t)
            </span>
          </div>

          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={INVERSE_LAW_DATA} margin={{ top: 20, right: 20, bottom: 20, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.4} />
                <XAxis
                  dataKey="synthetic_pct"
                  stroke="#64748b"
                  fontSize={11}
                  unit="%"
                  label={{ value: 'Synthetic Data Ratio S(t)', position: 'insideBottom', offset: -10, fill: '#94a3b8', fontSize: 10 }}
                />
                <YAxis
                  stroke="#64748b"
                  fontSize={11}
                  domain={[0, 1]}
                />
                <Tooltip
                  content={({ payload, label }) => {
                    if (payload && payload.length) {
                      return (
                        <div className="bg-slate-900 border border-slate-700 p-2.5 rounded-lg shadow-xl text-xs space-y-1">
                          <p className="font-bold text-white">Synthetic Saturation: {label}%</p>
                          {payload.map((p) => (
                            <p key={p.name} style={{ color: p.color }}>
                              {p.name}: {Number(p.value).toFixed(2)}
                            </p>
                          ))}
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Legend
                  verticalAlign="top"
                  align="right"
                  wrapperStyle={{ fontSize: '11px', paddingBottom: '10px' }}
                />
                <Line
                  type="monotone"
                  dataKey="concept_drift"
                  name="Observable Drift D(t)"
                  stroke="#06b6d4"
                  strokeWidth={2.5}
                  dot={{ r: 4 }}
                />
                <Line
                  type="monotone"
                  dataKey="model_collapse"
                  name="Model Collapse M(t)"
                  stroke="#f43f5e"
                  strokeWidth={2.5}
                  dot={{ r: 4 }}
                />
                <Line
                  type="monotone"
                  dataKey="perf_delta"
                  name="Perf Degradation ΔP(t)"
                  stroke="#fbbf24"
                  strokeDasharray="4 4"
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="text-[11px] text-slate-400">
            *As synthetic feedback reaches 100%, observable drift plummets because prompts look artificially uniform, masking real-world degradation.
          </div>
        </div>
      </div>

      {/* Interactive Simulation Sandbox */}
      <div className="p-6 rounded-2xl bg-slate-900/90 border border-indigo-500/30 shadow-lg space-y-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
          <div>
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Sliders className="w-5 h-5 text-indigo-400" />
              Live GenAI Drift Simulation & Financial ROI Sandbox
            </h3>
            <p className="text-xs text-slate-400">
              Manipulate external data shifts, recursive synthetic feedback, and GEO alteration to observe real-time degradation and retraining breakeven
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400">Decision Engine:</span>
            <span
              className={`px-3 py-1 rounded-full text-xs font-mono font-bold ${
                simResults?.should_retrain
                  ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                  : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
              }`}
            >
              {simResults?.should_retrain ? '⚠️ RETRAIN / RE-INDEX REQUIRED' : '✅ OPERATION NOMINAL'}
            </span>
          </div>
        </div>

        {/* 3 Interactive Sliders */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 p-4 rounded-xl bg-slate-950/80 border border-slate-800">
          {/* Slider 1: Synthetic Data Ratio */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="text-slate-300 font-medium">Recursive Synthetic Ratio S(t)</span>
              <span className="font-mono text-indigo-400 font-bold">{(syntheticRatio * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={syntheticRatio}
              onChange={(e) => setSyntheticRatio(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
            />
            <p className="text-[10px] text-slate-400 leading-tight">
              Simulates proportion of web corpora generated recursively by LLMs.
            </p>
          </div>

          {/* Slider 2: Prompt Domain Shift */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="text-slate-300 font-medium">Prompt Domain Shift Factor</span>
              <span className="font-mono text-cyan-400 font-bold">{(promptShift * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={promptShift}
              onChange={(e) => setPromptShift(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
            />
            <p className="text-[10px] text-slate-400 leading-tight">
              Simulates emergence of novel regulatory frameworks (e.g. EU AI Act Art. 10).
            </p>
          </div>

          {/* Slider 3: GEO Bias */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="text-slate-300 font-medium">GEO Content Manipulation Bias</span>
              <span className="font-mono text-amber-400 font-bold">{(geoBias * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={geoBias}
              onChange={(e) => setGeoBias(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-amber-500"
            />
            <p className="text-[10px] text-slate-400 leading-tight">
              Simulates web publishers optimizing text to manipulate AI search engine summaries.
            </p>
          </div>
        </div>

        {/* Live Simulation Metric Outputs */}
        {simResults && (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 text-center">
              <div className="text-[10px] text-slate-400 uppercase">Simulated Entropy</div>
              <div className="text-lg font-bold font-mono text-purple-400">{simResults.simulated_entropy} b</div>
            </div>
            <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 text-center">
              <div className="text-[10px] text-slate-400 uppercase">Simulated MMD</div>
              <div className="text-lg font-bold font-mono text-cyan-400">{simResults.simulated_mmd}</div>
            </div>
            <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 text-center">
              <div className="text-[10px] text-slate-400 uppercase">Collapse Index</div>
              <div className="text-lg font-bold font-mono text-rose-400">{(simResults.model_collapse_index * 100).toFixed(1)}%</div>
            </div>
            <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 text-center">
              <div className="text-[10px] text-slate-400 uppercase">Hallucination Risk</div>
              <div className="text-lg font-bold font-mono text-amber-400">{simResults.hallucination_risk_pct}%</div>
            </div>
            <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 text-center">
              <div className="text-[10px] text-slate-400 uppercase">Operational Loss</div>
              <div className="text-lg font-bold font-mono text-slate-200">${simResults.operational_loss_usd.toLocaleString()}</div>
            </div>
            <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 text-center">
              <div className="text-[10px] text-slate-400 uppercase">Net Retrain ROI</div>
              <div className={`text-lg font-bold font-mono ${simResults.net_retrain_roi_usd > 0 ? 'text-emerald-400' : 'text-slate-400'}`}>
                ${simResults.net_retrain_roi_usd.toLocaleString()}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Sample Prompt Telemetry Ledger Table */}
      <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-md space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <FileText className="w-4 h-4 text-cyan-400" />
              Live Prompt & Token Telemetry Ledger
            </h3>
            <p className="text-xs text-slate-400">
              Inspecting incoming queries for synthetic fingerprints, semantic similarity, and hallucination flags
            </p>
          </div>
          <span className="text-xs text-slate-400 font-mono">
            {sampleLedger.length} Sample Prompts Displayed
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 uppercase tracking-wider font-semibold">
                <th className="py-2.5 px-3">#</th>
                <th className="py-2.5 px-3">Incoming Prompt Content</th>
                <th className="py-2.5 px-3 text-center">Tokens</th>
                <th className="py-2.5 px-3 text-center">Centroid Cosine</th>
                <th className="py-2.5 px-3 text-center">Synthetic</th>
                <th className="py-2.5 px-3 text-center">Risk Flag</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-300 font-sans">
              {sampleLedger.map((row) => (
                <tr key={row.id} className="hover:bg-slate-800/40 transition">
                  <td className="py-3 px-3 font-mono text-slate-400">{row.id}</td>
                  <td className="py-3 px-3 font-mono text-slate-200 max-w-xl truncate">
                    {row.prompt}
                  </td>
                  <td className="py-3 px-3 text-center font-mono text-slate-400">
                    {row.token_count}
                  </td>
                  <td className="py-3 px-3 text-center font-mono">
                    <span
                      className={`px-2 py-0.5 rounded text-[11px] ${
                        row.similarity_to_ref >= 0.70
                          ? 'text-emerald-400 bg-emerald-950/60'
                          : row.similarity_to_ref >= 0.40
                          ? 'text-amber-400 bg-amber-950/60'
                          : 'text-rose-400 bg-rose-950/60'
                      }`}
                    >
                      {row.similarity_to_ref.toFixed(3)}
                    </span>
                  </td>
                  <td className="py-3 px-3 text-center font-mono">
                    {row.synthetic_flag ? (
                      <span className="text-[10px] px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 border border-rose-500/40">
                        SYNTHETIC
                      </span>
                    ) : (
                      <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                        ORGANIC
                      </span>
                    )}
                  </td>
                  <td className="py-3 px-3 text-center font-mono">
                    {row.hallucination_flag ? (
                      <span className="text-[10px] px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/40 flex items-center justify-center gap-1">
                        <AlertTriangle className="w-3 h-3 text-amber-400" /> HIGH
                      </span>
                    ) : (
                      <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 flex items-center justify-center gap-1">
                        <CheckCircle2 className="w-3 h-3 text-emerald-400" /> NOMINAL
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Architectural Concept Cards from the IEEE Paper */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
          <div className="flex items-center gap-2 text-indigo-400 font-bold text-xs uppercase tracking-wider">
            <Zap className="w-4 h-4" />
            1. Generative Engine Opt (GEO)
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            Content providers systematically structure text to dominate AI summaries. This deliberate web alteration causes covariate prompt drift independent of real-world user intent changes.
          </p>
        </div>

        <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
          <div className="flex items-center gap-2 text-cyan-400 font-bold text-xs uppercase tracking-wider">
            <ShieldAlert className="w-4 h-4" />
            2. EU AI Act Article 10 Mandates
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            Legal mandates requiring bias mitigation and high-quality training sets create artificial distribution bounds. Compliance filtering shifts token distributions relative to raw web scraped data.
          </p>
        </div>

        <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
          <div className="flex items-center gap-2 text-rose-400 font-bold text-xs uppercase tracking-wider">
            <TrendingDown className="w-4 h-4" />
            3. The Model Collapse Paradox
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            When models train recursively on synthetic web text, vocabulary collapses to common modes. Traditional drift tests can paradoxically report <em>low drift</em> because output variance shrinks.
          </p>
        </div>
      </div>
    </div>
  );
}
