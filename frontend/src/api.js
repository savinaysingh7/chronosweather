/**
 * ChronosGuard API Client
 * Connects to the local backend with resilient fallbacks.
 */

const API_BASE = '/api';

export async function checkHealth() {
  try {
    const res = await fetch('/health');
    return await res.json();
  } catch (err) {
    return { status: 'offline', error: err.message };
  }
}

export async function fetchCohorts() {
  const res = await fetch(`${API_BASE}/cohorts`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

export async function analyzeCohort(batchKey) {
  const res = await fetch(`${API_BASE}/analyze/${batchKey}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

export async function fetchShapAttribution(batchKey) {
  const res = await fetch(`${API_BASE}/shap/explain/${batchKey}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

export async function fetchShapSample(batchKey, sampleIdx = 0) {
  const res = await fetch(`${API_BASE}/shap/sample/${batchKey}/${sampleIdx}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

export async function fetchConformalCohorts() {
  const res = await fetch(`${API_BASE}/conformal/cohorts`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

export async function fetchLiveStatus() {
  const res = await fetch(`${API_BASE}/live/status`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

export async function startDaemon(provider = 'weather_grid_nyc', interval = 2.5) {
  const res = await fetch(`${API_BASE}/live/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ provider, interval }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

export async function stopDaemon() {
  const res = await fetch(`${API_BASE}/live/stop`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

export async function fetchLiveTicks(provider = 'weather_grid_nyc', limit = 30) {
  const res = await fetch(`${API_BASE}/live/ticks?provider=${provider}&limit=${limit}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

export async function fetchLiveSnapshots(provider = 'weather_grid_nyc', limit = 25) {
  const res = await fetch(`${API_BASE}/live/snapshots?provider=${provider}&limit=${limit}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

export async function fetchLiveDual(limit = 20) {
  const res = await fetch(`${API_BASE}/live/dual?limit=${limit}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

export async function triggerRetraining(batchKey, costFp = 350, costFn = 8500) {
  const res = await fetch(`${API_BASE}/retrain`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ batch_key: batchKey, cost_fp: costFp, cost_fn: costFn }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

export async function simulateDrift(covariateDrift, shockFactor, conceptDrift) {
  const res = await fetch(`${API_BASE}/simulate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      covariate_drift: covariateDrift,
      shock_factor: shockFactor,
      concept_drift: conceptDrift,
      n_samples: 2500,
    }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

export async function simulateLiveTick(provider = 'weather_grid_nyc', count = 5) {
  const res = await fetch(`${API_BASE}/live/simulate_tick`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ provider, count }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

