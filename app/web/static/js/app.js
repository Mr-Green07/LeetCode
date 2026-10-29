/* MedDiag dashboard - vanilla JS, no external dependencies. */
'use strict';

const API = '/api/v1';

/* ------------------------------------------------------------------ utils */
const $ = (id) => document.getElementById(id);
const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
};
const escapeHtml = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => (
  { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
));

function toast(message, kind) {
  const node = el('div', 'toast' + (kind ? ' is-' + kind : ''), message);
  $('toasts').appendChild(node);
  setTimeout(() => node.remove(), 5200);
}

function setStatus(id, message, kind) {
  const node = $(id);
  node.textContent = message || '';
  node.className = 'status-line' + (kind ? ' is-' + kind : '');
}

async function api(path, options) {
  const response = await fetch(API + path, options);
  let payload = null;
  try { payload = await response.json(); } catch (error) { payload = null; }
  if (!response.ok) {
    const detail = (payload && payload.error && payload.error.message)
      || (payload && payload.detail)
      || `${response.status} ${response.statusText}`;
    const error = new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
    error.status = response.status;
    throw error;
  }
  return payload;
}

function parseSamples(text) {
  if (!text) return [];
  return text.split(/[\s,;]+/).filter(Boolean).map(Number).filter((v) => Number.isFinite(v));
}

function formatBytes(bytes) {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = bytes; let index = 0;
  while (value >= 1024 && index < units.length - 1) { value /= 1024; index += 1; }
  return `${value.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function formatDate(value) {
  if (!value) return '-';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString();
}

/* ------------------------------------------------------------ tab wiring */
document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((t) => t.classList.remove('is-active'));
    document.querySelectorAll('.panel').forEach((p) => p.classList.remove('is-active'));
    tab.classList.add('is-active');
    $(`panel-${tab.dataset.panel}`).classList.add('is-active');
    if (tab.dataset.panel === 'history') loadHistory();
    if (tab.dataset.panel === 'reports') loadReports();
    if (tab.dataset.panel === 'about') loadModels();
  });
});

/* --------------------------------------------------------------- health */
async function loadHealth() {
  try {
    const health = await api('/health');
    const pill = $('health-pill');
    pill.className = 'pill pill-ok';
    pill.textContent = `● service online · v${health.version} · ${health.device}`;
    renderModelChips(health.models || {});
  } catch (error) {
    const pill = $('health-pill');
    pill.className = 'pill pill-danger';
    pill.textContent = '● service unreachable';
  }
}

function renderModelChips(models) {
  const container = $('model-chips');
  container.innerHTML = '';
  Object.keys(models).forEach((key) => {
    const model = models[key];
    const chip = el('span', 'chip' + (model.source === 'heuristic_baseline' ? ' is-baseline' : ''));
    chip.innerHTML = `${escapeHtml(key.toUpperCase())} · <b>${escapeHtml(model.source === 'trained' ? 'trained' : 'baseline')}</b>`;
    container.appendChild(chip);
  });
}

/* --------------------------------------------------------------- models */
async function loadModels() {
  const container = $('about-models');
  try {
    const payload = await api('/models');
    container.innerHTML = '';
    Object.values(payload.models || {}).forEach((model) => {
      const card = el('div', 'model-card');
      card.innerHTML = `
        <h3>${escapeHtml((model.modality || '').toUpperCase())}</h3>
        <span class="pill ${model.source === 'trained' ? 'pill-ok' : 'pill-warn'}">${escapeHtml(model.source)}</span>
        <dl>
          <dt>Architecture</dt><dd>${escapeHtml(model.architecture || '-')}</dd>
          <dt>Classes</dt><dd>${escapeHtml((model.classes || []).join(', '))}</dd>
          <dt>Parameters</dt><dd>${Number(model.parameters || 0).toLocaleString()}</dd>
          <dt>Accuracy</dt><dd>${model.accuracy != null ? (model.accuracy * 100).toFixed(1) + '%' : 'n/a'}</dd>
          <dt>Version</dt><dd>${escapeHtml(model.version || '-')}</dd>
        </dl>`;
      container.appendChild(card);
    });
  } catch (error) {
    container.innerHTML = `<p class="muted">Could not load model metadata: ${escapeHtml(error.message)}</p>`;
  }
}

/* ----------------------------------------------------------- result view */
function renderDiagnosis(result, containerId, emptyId) {
  const container = $(containerId);
  container.innerHTML = '';

  const prediction = result.prediction || {};
  const probabilities = prediction.class_probabilities || {};
  const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

  const card = el('div', 'diagnosis');
  const left = el('div');
  left.innerHTML = `
    <div class="diagnosis-label">${escapeHtml(prediction.label || '-')}</div>
    <div class="diagnosis-desc">${escapeHtml(prediction.description || '')}</div>`;
  const right = el('div', 'diagnosis-meta');
  right.innerHTML = `
    <span class="pill sev-${escapeHtml(prediction.severity || 'normal')}">${escapeHtml((prediction.severity || 'normal').toUpperCase())}</span>
    <div class="confidence">
      <div class="confidence-value">${((prediction.confidence || 0) * 100).toFixed(1)}%</div>
      <div class="confidence-label">Confidence</div>
      <div class="bar"><span style="width:${Math.max(2, Math.min(100, (prediction.confidence || 0) * 100))}%"></span></div>
    </div>`;
  card.appendChild(left);
  card.appendChild(right);
  container.appendChild(card);

  if (sorted.length) {
    container.appendChild(el('div', 'section-title', 'Class probability distribution'));
    const list = el('div', 'prob-list');
    sorted.forEach(([label, value], index) => {
      const row = el('div', 'prob-row' + (index === 0 ? ' is-top' : ''));
      row.innerHTML = `
        <span>${escapeHtml(label)}</span>
        <span class="track"><span class="fill" style="width:${Math.max(1, value * 100)}%"></span></span>
        <span class="val">${(value * 100).toFixed(1)}%</span>`;
      list.appendChild(row);
    });
    container.appendChild(list);
  }

  const findings = result.findings || [];
  if (findings.length) {
    container.appendChild(el('div', 'section-title', 'Clinical findings'));
    const wrap = el('div', 'table-wrap');
    const table = el('table', 'findings');
    table.innerHTML = `<thead><tr><th>Finding</th><th>Value</th><th>Reference</th><th>Status</th></tr></thead>`;
    const tbody = el('tbody');
    findings.forEach((finding) => {
      const status = finding.status || 'normal';
      const row = el('tr');
      row.innerHTML = `
        <td title="${escapeHtml(finding.explanation || '')}">${escapeHtml(finding.name)}</td>
        <td>${escapeHtml(finding.value)} ${escapeHtml(finding.unit || '')}</td>
        <td class="muted">${escapeHtml(finding.reference_range || '-')}</td>
        <td class="status"><span class="pill sev-${escapeHtml(status)}">${escapeHtml(status)}</span></td>`;
      tbody.appendChild(row);
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
    container.appendChild(wrap);
  }

  const explainability = result.explainability || {};
  if (explainability.summary || (explainability.top_features || []).length) {
    container.appendChild(el('div', 'section-title', 'Explainable AI'));
    if (explainability.summary) {
      const summary = el('p', 'hint', explainability.summary);
      container.appendChild(summary);
    }
    if (explainability.attribution_method) {
      container.appendChild(el('p', 'hint', `Attribution method: ${explainability.attribution_method} · ${(explainability.methods || []).join(', ')}`));
    }
    const topFeatures = explainability.top_features || [];
    if (topFeatures.length) {
      const list = el('div', 'feature-list');
      topFeatures.forEach((feature) => {
        const row = el('div', 'feature-row');
        const magnitude = Math.min(1, Math.abs(feature.contribution || 0));
        row.innerHTML = `
          <span>${escapeHtml(feature.name)}</span>
          <span class="track"><span class="fill" style="width:${Math.max(2, magnitude * 100)}%"></span></span>
          <span class="val">${(feature.contribution || 0).toFixed(2)}</span>`;
        list.appendChild(row);
      });
      container.appendChild(list);
    }

    const visuals = el('div', 'visuals');
    const entries = [
      ['saliency_png_base64', 'Attribution map'],
      ['overlay_png_base64', 'Grad-CAM localisation'],
      ['waveform_png_base64', 'Conditioned recording'],
    ];
    entries.forEach(([key, caption]) => {
      const data = explainability[key];
      if (!data) return;
      const figure = el('figure');
      figure.appendChild(el('figcaption', null, caption));
      const img = el('img');
      img.src = 'data:image/png;base64,' + data;
      img.alt = caption;
      figure.appendChild(img);
      visuals.appendChild(figure);
    });
    if (visuals.children.length) container.appendChild(visuals);
  }

  const recommendations = result.recommendations || [];
  if (recommendations.length) {
    container.appendChild(el('div', 'section-title', 'Clinical recommendations'));
    const list = el('ul', 'advice');
    recommendations.forEach((item) => list.appendChild(el('li', null, item)));
    container.appendChild(list);
  }

  const quality = result.quality || {};
  container.appendChild(el('div', 'section-title', 'Input quality'));
  const qualityRow = el('p', 'hint',
    `Signal quality index ${quality.signal_quality_index ?? '-'} · ${quality.usable ? 'usable' : 'not usable'}` +
    ` · SNR ${quality.snr_db ?? '-'} dB · model ${escapeHtml(result.model_source || '-')} v${escapeHtml(result.model_version || '-')}`);
  container.appendChild(qualityRow);

  const actions = el('div', 'btn-row');
  const pdfBtn = el('button', 'btn btn-primary btn-sm', 'Generate PDF report');
  pdfBtn.addEventListener('click', () => generateReport(result.analysis_id, 'pdf'));
  const htmlBtn = el('button', 'btn btn-sm', 'Generate HTML report');
  htmlBtn.addEventListener('click', () => generateReport(result.analysis_id, 'html'));
  actions.appendChild(pdfBtn);
  actions.appendChild(htmlBtn);
  const idLabel = el('span', 'hint', `Analysis ${result.analysis_id}`);
  actions.appendChild(idLabel);
  container.appendChild(actions);

  $(emptyId).hidden = true;
  container.hidden = false;
}

// async function generateReport(analysisId, format) {
//   try {
//     const report = await api(`/reports/${analysisId}?format=${format}`, { method: 'POST' });
//     // `download_url` is already a complete path ("/api/v1/reports/..."), so it
//     // must NOT be prefixed with API again - that would build
//     // "/api/v1/api/v1/reports/..." and 404.
//     window.open(report.download_url, '_blank');
//     toast(`Report ${report.report_id} generated (${format.toUpperCase()})`, 'ok');
//     loadReports();
//   } catch (error) {
//     toast(`Report generation failed: ${error.message}`, 'error');
//   }
// }

/* ------------------------------------------------------------ ECG / EEG */
function patientFrom(prefix) {
  const ageRaw = $(`${prefix}-age`).value;
  return {
    patient_id: $(`${prefix}-patient`).value || `PT-${prefix.toUpperCase()}`,
    age: ageRaw ? Number(ageRaw) : null,
    sex: $(`${prefix}-sex`).value || 'unknown',
    clinical_notes: $(`${prefix}-notes`).value || null,
  };
}

async function analyzeSignal(modality) {
  const prefix = modality;
  const samples = parseSamples($(`${prefix}-samples`).value);
  if (samples.length < 32) {
    setStatus(`${prefix}-status`, 'Provide at least 32 samples (or load the demo sample).', 'error');
    return;
  }
  const rate = Number($(`${prefix}-rate`).value) || 500;
  setStatus(`${prefix}-status`, 'Analysing…');
  $(`${prefix}-analyze`).disabled = true;

  try {
    const payload = {
      patient: patientFrom(prefix),
      samples: samples,
      sampling_rate: rate,
    };
    const result = await api(`/${modality}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    renderDiagnosis(result, `${prefix}-output`, `${prefix}-empty`);
    setStatus(`${prefix}-status`, `Completed · ${result.prediction.label} (${(result.prediction.confidence * 100).toFixed(1)}%)`, 'ok');
    toast(`${modality.toUpperCase()} analysis complete`, 'ok');
  } catch (error) {
    setStatus(`${prefix}-status`, error.message, 'error');
    toast(`${modality.toUpperCase()} analysis failed: ${error.message}`, 'error');
  } finally {
    $(`${prefix}-analyze`).disabled = false;
  }
}

async function loadDemoSignal(modality) {
  const labels = modality === 'ecg'
    ? ['NORM', 'AFIB', 'MI', 'STDE', 'PVC']
    : ['NORMAL', 'EPILEPSY', 'PARKINSON', 'DEPRESSION'];
  const label = labels[Math.floor(Math.random() * labels.length)];
  try {
    setStatus(`${modality}-status`, `Loading demo sample (${label})…`);
    const sample = await api(`/demo/${modality}/${label}`);
    $(`${modality}-samples`).value = sample.samples.join(', ');
    $(`${modality}-rate`).value = sample.sampling_rate;
    setStatus(`${modality}-status`, `Demo sample '${label}' loaded - press Analyse.`, 'ok');
  } catch (error) {
    setStatus(`${modality}-status`, error.message, 'error');
  }
}

async function uploadSignal(modality, file) {
  if (!file) return;
  const form = new FormData();
  form.append('file', file);
  form.append('patient_id', $(`${modality}-patient`).value || 'PT-UPLOAD');
  form.append('sampling_rate', String($(`${modality}-rate`).value || 500));
  if ($(`${modality}-age`).value) form.append('age', $(`${modality}-age`).value);
  form.append('sex', $(`${modality}-sex`).value || 'unknown');
  if ($(`${modality}-notes`).value) form.append('clinical_notes', $(`${modality}-notes`).value);

  setStatus(`${modality}-status`, `Uploading ${file.name}…`);
  $(`${modality}-analyze`).disabled = true;
  try {
    const result = await api(`/${modality}/analyze/upload`, { method: 'POST', body: form });
    renderDiagnosis(result, `${modality}-output`, `${modality}-empty`);
    setStatus(`${modality}-status`, `Completed · ${result.prediction.label}`, 'ok');
    toast(`${modality.toUpperCase()} file analysed`, 'ok');
  } catch (error) {
    setStatus(`${modality}-status`, error.message, 'error');
    toast(`${modality.toUpperCase()} upload failed: ${error.message}`, 'error');
  } finally {
    $(`${modality}-analyze`).disabled = false;
  }
}

/* ----------------------------------------------------------------- X-ray */
let xrayFile = null;

async function analyzeXray() {
  if (!xrayFile) {
    setStatus('xray-status', 'Choose a radiograph first (or load the demo sample).', 'error');
    return;
  }
  const form = new FormData();
  form.append('file', xrayFile);
  form.append('patient_id', $('xray-patient').value || 'PT-XRAY');
  form.append('view', $('xray-view').value || 'UNKNOWN');
  if ($('xray-age').value) form.append('age', $('xray-age').value);
  form.append('sex', $('xray-sex').value || 'unknown');
  if ($('xray-notes').value) form.append('clinical_notes', $('xray-notes').value);

  setStatus('xray-status', 'Analysing radiograph…');
  $('xray-analyze').disabled = true;
  try {
    const result = await api('/xray/analyze', { method: 'POST', body: form });
    renderDiagnosis(result, 'xray-output', 'xray-empty');
    setStatus('xray-status', `Completed · ${result.prediction.label}`, 'ok');
    toast('Radiograph analysis complete', 'ok');
  } catch (error) {
    setStatus('xray-status', error.message, 'error');
    toast('Radiograph analysis failed: ' + error.message, 'error');
  } finally {
    $('xray-analyze').disabled = false;
  }
}

async function loadDemoXray() {
  const labels = ['NORMAL', 'PNEUMONIA', 'CONSOLIDATION', 'EFFUSION', 'TUBERCULOSIS', 'COVID19'];
  const label = labels[Math.floor(Math.random() * labels.length)];
  try {
    setStatus('xray-status', `Loading demo radiograph (${label})…`);
    const sample = await api(`/demo/xray/${label}`);
    const binary = atob(sample.image_base64);
    const buffer = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) buffer[i] = binary.charCodeAt(i);
    xrayFile = new File([buffer], `${label.toLowerCase()}.png`, { type: 'image/png' });
    const preview = $('xray-preview');
    preview.src = URL.createObjectURL(xrayFile);
    preview.hidden = false;
    setStatus('xray-status', `Demo radiograph '${label}' loaded - press Analyse.`, 'ok');
  } catch (error) {
    setStatus('xray-status', error.message, 'error');
  }
}

/* --------------------------------------------------------------- history */
async function loadHistory() {
  const body = $('history-body');
  try {
    const [history, summary] = await Promise.all([api('/history?limit=100'), api('/history/summary')]);
    const stats = $('history-stats');
    stats.innerHTML = '';
    const cards = [
      ['Total cases', summary.total],
      ['ECG', summary.modality_distribution.ecg || 0],
      ['EEG', summary.modality_distribution.eeg || 0],
      ['X-ray', summary.modality_distribution.xray || 0],
    ];
    cards.forEach(([key, value]) => {
      const stat = el('div', 'stat');
      stat.innerHTML = `<div class="k">${escapeHtml(key)}</div><div class="v">${escapeHtml(String(value))}</div>`;
      stats.appendChild(stat);
    });

    if (!history.items.length) {
      body.innerHTML = '<tr><td colspan="8" class="muted">No cases recorded yet.</td></tr>';
      return;
    }
    body.innerHTML = '';
    history.items.forEach((item) => {
      const row = el('tr');
      row.innerHTML = `
        <td><code>${escapeHtml(item.analysis_id)}</code></td>
        <td>${escapeHtml(item.modality.toUpperCase())}</td>
        <td>${escapeHtml(item.patient_id)}</td>
        <td>${escapeHtml(item.predicted_label)}</td>
        <td>${(item.confidence * 100).toFixed(1)}%</td>
        <td><span class="pill sev-${escapeHtml(item.severity)}">${escapeHtml(item.severity)}</span></td>
        <td>${escapeHtml(formatDate(item.created_at))}</td>
        <td><button class="btn btn-sm" data-report="${escapeHtml(item.analysis_id)}">Report</button></td>`;
      body.appendChild(row);
    });
    body.querySelectorAll('button[data-report]').forEach((button) => {
      button.addEventListener('click', () => generateReport(button.dataset.report, 'pdf'));
    });
  } catch (error) {
    body.innerHTML = `<tr><td colspan="8" class="muted">Could not load history: ${escapeHtml(error.message)}</td></tr>`;
  }
}

// /* --------------------------------------------------------------- reports */
async function loadReports() {
  const body = $('reports-body');
  try {
    const payload = await api('/reports/list?limit=100');
    if (!payload.reports.length) {
      body.innerHTML = '<tr><td colspan="6" class="muted">No reports generated yet.</td></tr>';
      return;
    }
    body.innerHTML = '';
    payload.reports.forEach((report) => {
      const row = el('tr');
      row.innerHTML = `
        <td><code>${escapeHtml(report.report_id)}</code></td>
        <td><code>${escapeHtml(report.analysis_id)}</code></td>
        <td>${escapeHtml(report.format.toUpperCase())}</td>
        <td>${escapeHtml(formatBytes(report.size_bytes))}</td>
        <td>${escapeHtml(formatDate(report.created_at))}</td>
        <td><a class="btn btn-sm" href="${escapeHtml(report.download_url)}" target="_blank" rel="noopener">Open</a></td>`;
      body.appendChild(row);
    });
  } catch (error) {
    body.innerHTML = `<tr><td colspan="6" class="muted">Could not load reports: ${escapeHtml(error.message)}</td></tr>`;
  }
}

/* ------------------------------------------------------------- bindings */
function bindSignal(modality) {
  $(`${modality}-analyze`).addEventListener('click', () => analyzeSignal(modality));
  $(`${modality}-demo`).addEventListener('click', () => loadDemoSignal(modality));
  $(`${modality}-clear`).addEventListener('click', () => {
    $(`${modality}-samples`).value = '';
    setStatus(`${modality}-status`, '');
    $(`${modality}-output`).hidden = true;
    $(`${modality}-empty`).hidden = false;
  });
  $(`${modality}-file`).addEventListener('change', (event) => uploadSignal(modality, event.target.files[0]));
}

function bindXray() {
  const drop = $('xray-drop');
  const input = $('xray-file');

  const acceptFile = (file) => {
    if (!file) return;
    xrayFile = file;
    const preview = $('xray-preview');
    if (file.type.startsWith('image/')) {
      preview.src = URL.createObjectURL(file);
      preview.hidden = false;
    } else {
      preview.hidden = true;
    }
    setStatus('xray-status', `${file.name} ready - press Analyse.`);
  };

  drop.addEventListener('click', () => input.click());
  input.addEventListener('change', (event) => acceptFile(event.target.files[0]));
  ['dragenter', 'dragover'].forEach((name) => drop.addEventListener(name, (event) => {
    event.preventDefault(); drop.classList.add('is-over');
  }));
  ['dragleave', 'drop'].forEach((name) => drop.addEventListener(name, (event) => {
    event.preventDefault(); drop.classList.remove('is-over');
  }));
  drop.addEventListener('drop', (event) => acceptFile(event.dataTransfer.files[0]));

  $('xray-analyze').addEventListener('click', analyzeXray);
  $('xray-demo').addEventListener('click', loadDemoXray);
  $('xray-clear').addEventListener('click', () => {
    xrayFile = null;
    input.value = '';
    $('xray-preview').hidden = true;
    setStatus('xray-status', '');
    $('xray-output').hidden = true;
    $('xray-empty').hidden = false;
  });
}

/* ----------------------------------------------------------------- init */
bindSignal('ecg');
bindSignal('eeg');
bindXray();
$('history-refresh').addEventListener('click', loadHistory);
$('reports-refresh').addEventListener('click', loadReports);
loadHealth();
loadModels();
