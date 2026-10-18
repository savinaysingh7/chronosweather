import React, { useState } from 'react';
import {
  GraduationCap,
  SunMedium,
  Zap,
  Radio,
  HelpCircle,
  Wrench,
  Sliders,
  ShieldCheck,
  AlertTriangle,
  CloudLightning
} from 'lucide-react';

const METAPHORS = [
  {
    icon: SunMedium,
    title: '1. The Solar Farm in a Summer Heatwave',
    concept: 'Covariate Shift: P(Weather) changes',
    badge: 'Covariate Shift',
    badgeColor: 'bg-blue-500/20 text-blue-300 border-blue-500/40',
    story:
      'Imagine you train a machine learning model using mild spring weather (65°F / 18°C). It predicts electricity baseload with 99.8% precision. But when July arrives, a persistent 100°F (38°C) heat dome blankets the city. Millions of residential air conditioners turn on at maximum blast simultaneously.',
    takeaway:
      'The physical atmospheric environment shifted outside the training envelope. This is Covariate Shift: input features (temperature, humidity, solar radiation) moved, while physical laws stayed the same.'
  },
  {
    icon: Zap,
    title: '2. The Overheated Transmission Lines',
    concept: 'Concept Drift: P(Blackout | Weather) flips',
    badge: 'Concept Drift',
    badgeColor: 'bg-rose-500/20 text-rose-300 border-rose-500/40',
    story:
      'The thermometers read 32°C—the exact same temperature as last year. But due to high transmission line sag and solar inverter thermal de-rating, transformers now trip offline at much lower stress levels than historically. The weather readings look familiar, but the grid’s response has structurally transformed.',
    takeaway:
      'This is Concept Drift: the relationship between inputs (weather) and output (grid strain) changed. A model predicting baseload will catastrophically underestimate blackout risk.'
  },
  {
    icon: Radio,
    title: '3. The Grid Dispatcher Without Tomorrow’s Meter Bills',
    concept: 'Label-Free Monitoring (CBPE)',
    badge: 'Label-Free CBPE',
    badgeColor: 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40',
    story:
      'When an extreme atmospheric squall hits the city, utility dispatchers cannot wait 30 days for delayed billing meters and post-incident investigation reports to learn if their AI is failing. They need to know the model’s reliability *right now* in real time.',
    takeaway:
      'Confidence-Based Performance Estimation (CBPE) mathematically predicts current model AUC and F1 from incoming prediction probabilities alone—with ZERO delayed labels.'
  },
  {
    icon: HelpCircle,
    title: '4. The "Spin Up the Reserves" Uncertainty Button',
    concept: 'Split Conformal Prediction Sets',
    badge: 'Conformal Sets',
    badgeColor: 'bg-purple-500/20 text-purple-300 border-purple-500/40',
    story:
      'Standard AI always makes a single, overconfident binary guess: "Grid Normal" or "Peak Strain." Split Conformal Prediction gives the model an honest uncertainty set with a guaranteed 90% finite-sample mathematical proof.',
    takeaway:
      'In normal spring diurnal cycles, the prediction set is a confident singleton {BASELOAD}. When an unprecedented storm hits, it expands to {BASELOAD, PEAK_STRAIN}. That widening proves to operators that backup peaker plants must be readied.'
  },
  {
    icon: Wrench,
    title: '5. Peaker Plants vs Total City Blackout',
    concept: 'Cost-Aware Retraining ROI',
    badge: 'Cost Optimization',
    badgeColor: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40',
    story:
      'Firing an emergency natural gas peaker plant costs $350/MWh. A total unhedged citywide blackout costs $8,500/MWh in equipment damage and economic shutdown. But retraining an ML pipeline costs $1,200 in compute and safety recertification.',
    takeaway:
      'ChronosWeather computes the net financial ROI: it only triggers autonomous retraining when the operational losses from model decay exceed the compute expense of retraining.'
  }
];

const GLOSSARY = [
  { term: 'Covariate Shift (Data Drift)', math: 'P(X_train) ≠ P(X_test)', meaning: 'Atmospheric conditions changed (heatwave, humidity spike), but grid physical laws stayed the same.' },
  { term: 'Concept Drift', math: 'P(Y|X_train) ≠ P(Y|X_test)', meaning: 'The physical relationship flipped (transmission lines de-rate and trip under lower temperatures).' },
  { term: 'Model Decay', math: 'Metric(T_t) < Metric(T_0)', meaning: 'The model becomes progressively inaccurate because seasonal climate dynamics have shifted.' },
  { term: 'Population Stability Index (PSI)', math: 'Σ (Actual% - Expected%) × ln(A%/E%)', meaning: 'A statistical smoke alarm that alerts when current atmospheric sensor distributions diverge from training baselines.' },
  { term: 'Maximum Mean Discrepancy (MMD)', math: '||μ_P - μ_Q||_H', meaning: 'A multi-dimensional kernel sensor that detects joint temperature-humidity-radiation correlations that 1D tests miss.' },
  { term: 'CBPE Estimation', math: 'E[ConfMatrix | P(Y|X)]', meaning: 'Estimating current ROC-AUC and F1 score in real time without waiting for post-outage utility audit labels.' },
  { term: 'Split Conformal Prediction', math: 'P(Y ∈ C(X)) ≥ 1 - α', meaning: 'A mathematical guarantee providing a prediction set {BASELOAD, PEAK_STRAIN} guaranteed to cover the truth 90% of the time.' },
  { term: 'SHAP Attribution', math: 'φ_i(v)', meaning: 'A root-cause detective revealing whether temperature, humidity, or solar radiation is the primary driver of grid stress.' }
];

export default function Eli5View() {
  const [volatility, setVolatility] = useState(0);

  // Dynamic calculations for simulator
  const simulatedAcc = Math.max(48.0, 99.8 - volatility * 0.48).toFixed(1);
  const simulatedBleed = Math.round(15000 + Math.pow(volatility, 1.9) * 110);
  const conformalWidth = (1.00 + volatility * 0.007).toFixed(2);
  const isRetrainTriggered = volatility >= 45;

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto">
      {/* Title */}
      <div>
        <h3 className="text-xl font-extrabold text-white tracking-tight flex items-center gap-2">
          <GraduationCap className="w-6 h-6 text-cyan-400" />
          The Beginner's Guide: Why ML Models Silently Break in the Real World
        </h3>
        <p className="text-xs text-slate-400 mt-1 max-w-3xl leading-relaxed">
          If you are new to machine learning, statistics, or MLOps, welcome! Here is the fundamental truth:{' '}
          <span className="text-white font-semibold">deploying a model is only 10% of the work</span>. The other 90% is making sure it doesn’t silently self-destruct as the world shifts around it.
        </p>
      </div>

      {/* Interactive Shift-the-World Sandbox */}
      <div className="p-6 rounded-2xl bg-gradient-to-r from-slate-900 via-slate-900 to-cyan-950/40 border border-slate-800 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <span className="text-xs font-bold text-cyan-400 uppercase tracking-wider flex items-center gap-1.5 mb-1">
              <Sliders className="w-4 h-4" />
              Interactive Story Simulator
            </span>
            <h4 className="text-base font-bold text-white tracking-tight">
              "Shift the Climate" Atmospheric Heatwave & Storm Stress Simulator
            </h4>
            <p className="text-xs text-slate-400 mt-0.5">
              Drag the slider to introduce extreme heatwave temperatures and watch an unmonitored grid model decay
            </p>
          </div>

          <div className="text-right">
            <span className="text-xs text-slate-400 block font-medium">Atmospheric Heatwave Index:</span>
            <span className="text-2xl font-black text-cyan-400 font-mono">{volatility}%</span>
          </div>
        </div>

        {/* Slider */}
        <input
          type="range"
          min={0}
          max={100}
          step={5}
          value={volatility}
          onChange={(e) => setVolatility(Number(e.target.value))}
          className="w-full accent-cyan-500 cursor-pointer h-2 bg-slate-950 rounded-lg"
        />

        {/* Dynamic Simulator KPIs */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 pt-2 font-mono">
          <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800">
            <span className="text-[11px] text-slate-400 block">Model True Accuracy</span>
            <span className="text-2xl font-bold text-white">{simulatedAcc}%</span>
            <span className="text-[10px] text-slate-400 block mt-0.5">
              {volatility === 0 ? 'Peak Spring Baseline' : `Dropped by ${(99.8 - simulatedAcc).toFixed(1)}%`}
            </span>
          </div>

          <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800">
            <span className="text-[11px] text-slate-400 block">Blackout & Reserve Deficit</span>
            <span className="text-2xl font-bold text-rose-400">${simulatedBleed.toLocaleString()}</span>
            <span className="text-[10px] text-slate-400 block mt-0.5">Unhedged grid shortfall</span>
          </div>

          <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800">
            <span className="text-[11px] text-slate-400 block">Conformal Uncertainty</span>
            <span className="text-2xl font-bold text-purple-400">{conformalWidth} sets</span>
            <span className="text-[10px] text-slate-400 block mt-0.5">Widened from singleton {`{BASELOAD}`}</span>
          </div>

          <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800">
            <span className="text-[11px] text-slate-400 block">ChronosWeather Radar</span>
            <span
              className={`text-sm font-bold block mt-1.5 ${
                isRetrainTriggered ? 'text-rose-400 animate-pulse' : volatility >= 25 ? 'text-amber-400' : 'text-emerald-400'
              }`}
            >
              {isRetrainTriggered ? '🔴 RETRAIN TRIGGERED' : volatility >= 25 ? '🟡 WARNING ELEVATED' : '🟢 GRID STABLE'}
            </span>
          </div>
        </div>

        {/* Dynamic Explanation Alert */}
        <div
          className={`p-4 rounded-xl border text-xs flex items-center gap-3 ${
            isRetrainTriggered
              ? 'bg-rose-500/10 border-rose-500/30 text-rose-200'
              : volatility >= 25
              ? 'bg-amber-500/10 border-amber-500/30 text-amber-200'
              : 'bg-emerald-500/10 border-emerald-500/30 text-emerald-200'
          }`}
        >
          {isRetrainTriggered ? (
            <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0" />
          ) : (
            <ShieldCheck className="w-5 h-5 text-emerald-400 shrink-0" />
          )}
          <span>
            {isRetrainTriggered ? (
              <b>Autonomous Retraining Activated:</b>
            ) : volatility >= 25 ? (
              <b>Elevated Weather Uncertainty:</b>
            ) : (
              <b>Grid In-Distribution:</b>
            )}{' '}
            {isRetrainTriggered
              ? 'Blackout deficit and peaker penalties have exceeded the compute retraining expense. ChronosWeather autonomously hot-swapped the model on a sliding window without human downtime.'
              : volatility >= 25
              ? 'Atmospheric temperatures are climbing outside training baselines. Conformal set sizes are widening to {BASELOAD, STRAIN}, warning operators of impending blackout risk.'
              : 'Weather matches historical spring baselines. High confidence, zero unnecessary peaker plant firings or retraining churn.'}
          </span>
        </div>
      </div>

      {/* The 5 Story Cards */}
      <div className="space-y-4">
        <h4 className="text-sm font-bold text-white tracking-tight uppercase">
          5 Everyday Stories That Explain Enterprise MLOps
        </h4>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {METAPHORS.map((m, i) => {
            const Icon = m.icon;
            return (
              <div
                key={i}
                className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3 flex flex-col justify-between"
              >
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <div className="w-10 h-10 rounded-xl bg-slate-800 flex items-center justify-center text-cyan-400">
                      <Icon className="w-5 h-5" />
                    </div>
                    <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${m.badgeColor}`}>
                      {m.badge}
                    </span>
                  </div>

                  <h5 className="text-sm font-bold text-white tracking-tight">{m.title}</h5>
                  <p className="text-xs text-slate-300 mt-2 leading-relaxed">{m.story}</p>
                </div>

                <div className="pt-3 border-t border-slate-800/80">
                  <p className="text-xs text-cyan-300 font-medium">
                    <b>Why it matters:</b> {m.takeaway}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Jargon Buster Table */}
      <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
        <h4 className="text-sm font-bold text-white tracking-tight uppercase">
          📚 Jargon Buster Glossary: Academic Terms Translated to Plain English
        </h4>

        <div className="overflow-x-auto rounded-xl border border-slate-800/80">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950 text-slate-400 font-semibold border-b border-slate-800">
              <tr>
                <th className="py-2.5 px-4">Academic Term</th>
                <th className="py-2.5 px-4">Mathematical Notation</th>
                <th className="py-2.5 px-4">Plain English Meaning</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/50 font-mono text-slate-300">
              {GLOSSARY.map((g, i) => (
                <tr key={i} className="hover:bg-slate-800/30">
                  <td className="py-2 px-4 font-bold text-white">{g.term}</td>
                  <td className="py-2 px-4 text-cyan-400">{g.math}</td>
                  <td className="py-2 px-4 font-sans text-slate-300">{g.meaning}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
