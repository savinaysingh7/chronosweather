import React, { useState, useEffect, useCallback } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import OverviewView from './views/OverviewView';
import LiveStreamingView from './views/LiveStreamingView';
import ShapView from './views/ShapView';
import ConformalView from './views/ConformalView';
import CohortsView from './views/CohortsView';
import CostMitigationView from './views/CostMitigationView';
import Eli5View from './views/Eli5View';
import ErrorBoundary from './components/ErrorBoundary';
import { checkHealth, fetchLiveStatus, analyzeCohort, startDaemon, stopDaemon } from './api';

const TAB_CONFIG = {
  overview: {
    title: 'Executive Command Center',
    subtitle: 'Real-time model decay trajectory, drift alert matrix, and production system health.'
  },
  live: {
    title: 'Live Streaming Telemetry Ops',
    subtitle: 'Continuous 24/7 high-frequency drift surveillance with SQLite WAL persistence.'
  },
  shap: {
    title: 'Root-Cause Explainability Studio',
    subtitle: 'SHAP feature attribution shifts (Δ|SHAP|) and individual weather observation waterfall explanations.'
  },
  conformal: {
    title: 'Split Conformal Prediction Lab',
    subtitle: 'Finite-sample 90% coverage guarantees and dynamic prediction set sizing.'
  },
  cohorts: {
    title: 'Cohort Drift Detection Suite',
    subtitle: 'Univariate PSI/KS, multivariate MMD, adversarial AUC, and CBPE verification.'
  },
  roi: {
    title: 'Cost-Aware Mitigation & Retraining ROI',
    subtitle: 'Asymmetric loss modeling (Cost of False Negatives vs False Positives) and automated zero-downtime model updates.'
  },
  eli5: {
    title: "Beginner's Intuition & Simulation Lab",
    subtitle: 'No-jargon interactive guide, real-world analogies, and live synthetic drift sandbox.'
  }
};

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedCohort, setSelectedCohort] = useState('T3_Sudden_Shock');
  const [analysisData, setAnalysisData] = useState(null);
  const [daemonRunning, setDaemonRunning] = useState(false);
  const [backendOnline, setBackendOnline] = useState(false);
  const [loading, setLoading] = useState(false);

  // Poll daemon and health
  const pollStatus = useCallback(async () => {
    try {
      const health = await checkHealth();
      setBackendOnline(health?.status === 'healthy' || health?.status === 'ok');

      const live = await fetchLiveStatus();
      if (live && live.success) {
        setDaemonRunning(live.running);
      }
    } catch (err) {
      setBackendOnline(false);
    }
  }, []);

  // Fetch telemetry for chosen cohort
  const loadTelemetry = useCallback(async (cohort = selectedCohort) => {
    setLoading(true);
    try {
      await pollStatus();
      const res = await analyzeCohort(cohort);
      if (res && res.success) {
        setAnalysisData(res);
      }
    } catch (err) {
      console.error('Failed to load telemetry:', err);
    } finally {
      setLoading(false);
    }
  }, [pollStatus, selectedCohort]);

  useEffect(() => {
    loadTelemetry(selectedCohort);
    const interval = setInterval(pollStatus, 5000);
    return () => clearInterval(interval);
  }, [selectedCohort]);

  const handleSelectCohort = async (cohortId) => {
    setSelectedCohort(cohortId);
    await loadTelemetry(cohortId);
  };

  const handleStartDaemon = async () => {
    try {
      await startDaemon('weather_grid_nyc', 2.5);
      await pollStatus();
    } catch (err) {
      console.error('Failed to start daemon:', err);
    }
  };

  const handleStopDaemon = async () => {
    try {
      await stopDaemon();
      await pollStatus();
    } catch (err) {
      console.error('Failed to stop daemon:', err);
    }
  };

  const currentMeta = TAB_CONFIG[activeTab] || TAB_CONFIG.overview;

  const renderActiveView = () => {
    switch (activeTab) {
      case 'overview':
        return (
          <OverviewView
            analysisData={analysisData}
            selectedCohort={selectedCohort}
            onSelectCohort={handleSelectCohort}
            onNavigate={(tab) => setActiveTab(tab)}
          />
        );
      case 'live':
        return <LiveStreamingView />;
      case 'shap':
        return <ShapView selectedCohort={selectedCohort} />;
      case 'conformal':
        return <ConformalView />;
      case 'cohorts':
        return <CohortsView initialCohort={selectedCohort} />;
      case 'roi':
        return <CostMitigationView analysisData={analysisData} />;
      case 'eli5':
        return <Eli5View />;
      default:
        return (
          <OverviewView
            analysisData={analysisData}
            selectedCohort={selectedCohort}
            onSelectCohort={handleSelectCohort}
            onNavigate={(tab) => setActiveTab(tab)}
          />
        );
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-950 text-slate-100 font-sans">
      {/* Modern Left Sidebar Navigation */}
      <Sidebar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        daemonRunning={daemonRunning}
        backendOnline={backendOnline}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <Header
          activeTabTitle={currentMeta.title}
          subtitle={currentMeta.subtitle}
          daemonRunning={daemonRunning}
          onStartDaemon={handleStartDaemon}
          onStopDaemon={handleStopDaemon}
          onRefresh={loadTelemetry}
          loading={loading}
        />

        {/* View Content Canvas with Smooth Scroll */}
        <main className="flex-1 overflow-y-auto bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950">
          <ErrorBoundary key={activeTab}>
            {renderActiveView()}
          </ErrorBoundary>
        </main>
      </div>
    </div>
  );
}
