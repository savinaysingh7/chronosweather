import React from 'react';
import {
  Shield,
  LayoutDashboard,
  Radio,
  Search,
  Target,
  BarChart3,
  CircleDollarSign,
  GraduationCap,
  Activity,
  CheckCircle2,
  AlertCircle
} from 'lucide-react';

const NAV_ITEMS = [
  { id: 'overview', label: 'Command Center', icon: LayoutDashboard, badge: null },
  { id: 'live', label: 'Live Stream Ops', icon: Radio, badge: '24/7' },
  { id: 'shap', label: 'SHAP Explainability', icon: Search, badge: 'New' },
  { id: 'conformal', label: 'Conformal Prediction', icon: Target, badge: '90%' },
  { id: 'cohorts', label: 'Cohort Drift Suite', icon: BarChart3, badge: 'T0-T4' },
  { id: 'roi', label: 'Cost & Retrain ROI', icon: CircleDollarSign, badge: 'ROI' },
  { id: 'eli5', label: "Beginner's Guide", icon: GraduationCap, badge: 'ELI5' },
];

export default function Sidebar({ activeTab, onTabChange, daemonRunning, backendOnline }) {
  return (
    <aside className="w-68 bg-slate-900/80 backdrop-blur-md border-r border-slate-800 flex flex-col justify-between p-4 shrink-0 select-none">
      <div>
        {/* Brand Header */}
        <div className="flex items-center gap-3 px-2 py-3 mb-6">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 to-emerald-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <Shield className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="font-bold text-base tracking-tight text-white leading-tight">
              ChronosWeather
            </h1>
            <p className="text-xs text-slate-400 font-medium">Grid Reliability MLOps</p>
          </div>
        </div>

        {/* Live Daemon Status Pill */}
        <div className="mb-6 px-3 py-2.5 rounded-lg bg-slate-950/70 border border-slate-800/80 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span
              className={`w-2 h-2 rounded-full ${
                daemonRunning ? 'bg-emerald-400 shadow-sm shadow-emerald-400/80 animate-pulse' : 'bg-slate-500'
              }`}
            />
            <span className="text-xs font-semibold text-slate-300">
              {daemonRunning ? 'Daemon Active' : 'Daemon Idle'}
            </span>
          </div>
          <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400 bg-slate-800/80 px-1.5 py-0.5 rounded">
            WAL
          </span>
        </div>

        {/* Navigation Section */}
        <div className="space-y-1">
          <p className="px-2 text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-2">
            Monitoring Views
          </p>
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onTabChange(item.id)}
                className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-sm font-medium transition-all ${
                  isActive
                    ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/30 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                }`}
              >
                <div className="flex items-center gap-3">
                  <Icon className={`w-4 h-4 ${isActive ? 'text-cyan-400' : 'text-slate-400'}`} />
                  <span>{item.label}</span>
                </div>
                {item.badge && (
                  <span
                    className={`text-[10px] px-1.5 py-0.5 rounded font-mono font-bold ${
                      isActive
                        ? 'bg-cyan-500/20 text-cyan-300'
                        : 'bg-slate-800 text-slate-400'
                    }`}
                  >
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* Backend Status Footer */}
      <div className="pt-4 border-t border-slate-800/80 text-xs">
        <div className="flex items-center justify-between text-slate-400 px-2">
          <span className="flex items-center gap-1.5">
            {backendOnline ? (
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
            ) : (
              <AlertCircle className="w-3.5 h-3.5 text-rose-400" />
            )}
            Engine 5001
          </span>
          <span className="font-mono text-[11px] text-slate-500">v2.0 (Weather Grid)</span>
        </div>
      </div>
    </aside>
  );
}
