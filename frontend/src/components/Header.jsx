import React from 'react';
import { Play, Square, RefreshCw, Radio } from 'lucide-react';

export default function Header({
  activeTabTitle,
  subtitle,
  daemonRunning,
  onStartDaemon,
  onStopDaemon,
  onRefresh,
  loading
}) {
  return (
    <header className="h-16 border-b border-slate-800 bg-slate-900/40 backdrop-blur-md px-6 flex items-center justify-between shrink-0">
      <div>
        <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
          {activeTabTitle}
        </h2>
        {subtitle && <p className="text-xs text-slate-400 font-normal">{subtitle}</p>}
      </div>

      <div className="flex items-center gap-3">
        {/* Quick Daemon Controls */}
        {daemonRunning ? (
          <button
            onClick={onStopDaemon}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-rose-500/15 border border-rose-500/30 text-rose-300 hover:bg-rose-500/25 text-xs font-semibold transition"
          >
            <Square className="w-3.5 h-3.5 fill-current" />
            Stop Daemon
          </button>
        ) : (
          <button
            onClick={onStartDaemon}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-500/15 border border-emerald-500/30 text-emerald-300 hover:bg-emerald-500/25 text-xs font-semibold transition"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            Start Live Daemon
          </button>
        )}

        {/* Refresh button */}
        <button
          onClick={onRefresh}
          disabled={loading}
          className="p-1.5 rounded-lg bg-slate-800/80 border border-slate-700/60 text-slate-300 hover:text-white hover:bg-slate-700 transition disabled:opacity-50"
          title="Refresh Data"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin text-cyan-400' : ''}`} />
        </button>
      </div>
    </header>
  );
}
