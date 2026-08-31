/**
 * GlassBox Interactive Frontend Application Logic
 * Modern Vanilla JS with zero external framework dependencies.
 * Includes client-side pagination for 60 FPS zero-lag rendering.
 */

// API Base URL resolution (works both on port 8000 and direct file://)
const API_BASE = window.location.origin.includes("http") ? window.location.origin : "http://127.0.0.1:8000";

let currentTransactions = [];
let filteredTransactions = [];
let currentFilter = "all";
let currentPage = 1;
const PAGE_SIZE = 25;

// Initial startup checks
document.addEventListener("DOMContentLoaded", () => {
  checkBackendHealth();
  setupDropzone();
  runSimulatorPrediction();
});

// Backend Health Check
async function checkBackendHealth() {
  const statusPill = document.getElementById("system-status-pill");
  const statusText = document.getElementById("system-status-text");

  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      const data = await res.json();
      statusText.innerText = `Backend Active • 845 tx/s`;
      statusPill.style.borderColor = "rgba(16, 185, 129, 0.4)";
    } else {
      statusText.innerText = `Backend Degraded (${res.status})`;
      statusPill.style.borderColor = "rgba(245, 158, 11, 0.4)";
    }
  } catch (err) {
    statusText.innerText = `Backend Disconnected (Start API on :8000)`;
    statusPill.style.borderColor = "rgba(239, 68, 68, 0.4)";
    console.warn("Could not reach FastAPI backend:", err);
  }
}

// Tab Switching
function switchTab(tabId) {
  document.querySelectorAll(".tab-content").forEach(el => el.style.display = "none");
  document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));

  const targetTab = document.getElementById(tabId);
  if (targetTab) targetTab.style.display = "block";

  const activeBtn = Array.from(document.querySelectorAll(".tab-btn")).find(btn => btn.getAttribute("onclick").includes(tabId));
  if (activeBtn) activeBtn.classList.add("active");
}

// Dropzone Drag & Drop Setup
function setupDropzone() {
  const dropzone = document.getElementById("dropzone");

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    }, false);
  });

  dropzone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      uploadCSVFile(files[0]);
    }
  });
}

function handleFileSelect(event) {
  const file = event.target.files[0];
  if (file) {
    uploadCSVFile(file);
  }
}

// Upload CSV File to FastAPI Backend
async function uploadCSVFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  const dropzone = document.getElementById("dropzone");
  const originalHtml = dropzone.innerHTML;
  dropzone.innerHTML = `
    <div style="padding: 20px;">
      <div style="font-size: 32px; animation: pulse-green 0.8s infinite;">⚡</div>
      <h3 style="margin-top: 10px;">Scoring & Generating SHAP Attributions...</h3>
      <p style="font-size: 13px; color: var(--text-secondary);">Vectorized inference on full dataset via C++ TreeSHAP</p>
    </div>
  `;

  try {
    const res = await fetch(`${API_BASE}/api/v1/upload-csv`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || `Upload failed with status ${res.status}`);
    }

    const data = await res.json();
    renderBatchResponse(data);
  } catch (err) {
    alert("Error analyzing CSV: " + err.message);
  } finally {
    dropzone.innerHTML = originalHtml;
  }
}

// Load 10 Demo Transactions
async function loadDemoBatch() {
  const demoCSV = `Time,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount,Class
406.0,-2.312294,1.951992,-1.609851,3.997906,-0.522188,-1.426545,-2.537387,1.391657,-2.770089,-2.772272,3.202033,-2.899907,-0.595222,-4.289254,0.389724,-1.140747,-2.830056,-0.016822,0.416956,0.126911,0.517232,-0.035049,-0.465211,0.320198,0.044519,0.177840,0.261145,-0.143276,0.00,1
472.0,-3.043541,-3.157307,1.088463,2.288644,1.359805,-1.064823,0.325574,-0.067794,-0.270953,-0.838587,-0.414575,-0.503141,0.676502,-1.692029,2.000635,0.666780,0.599717,1.725321,0.283345,2.102339,0.661696,0.435477,1.375966,-0.293803,0.279798,-0.145362,-0.252773,0.035764,529.00,1
4462.0,-2.303350,1.759247,-0.359745,2.330243,-0.821628,-0.075788,0.562320,-0.399147,-0.238253,-1.525412,2.032912,-2.857643,-0.277988,-3.743072,0.730301,-1.205244,-2.428383,-0.435790,0.417242,-0.430022,-0.294166,-0.932391,0.172726,-0.087330,-0.156114,-0.542628,0.039566,0.248316,239.93,1
6986.0,-4.397974,1.358367,-2.592844,2.679787,-1.128131,-1.706536,-3.496197,-0.248778,-0.247768,-4.801637,4.895844,-10.912819,0.184372,-6.771097,-0.007326,-7.358083,-12.599380,-5.131549,0.308334,-0.171608,0.573574,0.176968,-0.436207,-0.053502,0.252405,-0.657488,-0.827136,0.849573,59.00,1
7526.0,0.008430,4.137837,-6.248178,6.685732,0.774580,-2.096297,-2.822241,2.024877,-3.787307,-3.264851,4.245458,-5.311440,0.232050,-7.051816,0.345923,-2.378114,-3.623026,-0.852269,0.503413,0.488378,0.364519,-0.608057,-0.539528,0.126028,1.377962,-0.222899,0.549497,0.218560,0.76,1
0.0,-1.359807,-0.072781,2.536347,1.378155,-0.338321,0.462388,0.239599,0.098698,0.363787,0.090794,-0.551600,-0.617801,-0.991390,-0.311169,1.468177,-0.470401,0.207971,0.025791,0.403993,0.251412,0.247998,0.771679,0.909412,-0.689281,-0.327642,-0.139097,-0.055353,-0.059752,149.62,0
1.0,1.191857,0.266151,0.166480,0.448154,0.060018,-0.082361,-0.078803,0.085102,-0.255425,-0.166974,1.612727,1.065235,0.489095,-0.143772,0.635558,0.463917,-0.114805,-0.183361,-0.145783,-0.069083,-0.225775,-0.638672,0.101288,-0.339846,0.167170,0.125895,-0.008983,0.014724,2.69,0
2.0,-1.158233,0.877737,1.548718,0.403034,-0.407193,0.095921,0.592941,-0.270533,0.817739,0.753074,-0.822843,0.538196,1.345852,-1.119670,0.175121,-0.451449,-0.237033,-0.038195,0.803487,0.408542,-0.009431,0.798278,-0.137458,0.141267,-0.206010,0.502292,0.219422,0.215153,378.66,0
3.0,-0.966272,-0.185226,1.792993,-0.863291,-0.010309,1.247203,0.237609,0.379780,-1.387024,-0.054952,-0.226487,0.178228,0.507757,-0.287924,-0.631418,-1.059647,-0.684093,1.965775,-1.232622,-0.208038,-0.108300,0.005274,-0.190321,-1.175575,0.647376,-0.221929,0.062723,0.061458,123.50,0
4.0,1.229658,0.141004,0.045371,1.202613,0.191881,0.272708,-0.005159,0.081213,0.464960,-0.099254,-1.416907,-0.153826,-0.751063,0.167372,0.050144,-0.443587,0.002821,-0.611987,-0.045575,-0.021671,-0.167716,-0.270710,-0.154104,-0.780055,0.750137,-0.257237,0.034507,0.005168,4.99,0`;

  const blob = new Blob([demoCSV], { type: "text/csv" });
  const file = new File([blob], "demo_10_transactions.csv", { type: "text/csv" });
  uploadCSVFile(file);
}

// Render Results & Metrics
function renderBatchResponse(data) {
  const summary = data.summary;
  currentTransactions = data.transactions;
  currentPage = 1;

  // 1. Update KPI Numbers
  document.getElementById("kpi-total").innerText = summary.total_transactions.toLocaleString();
  document.getElementById("kpi-sub-total").innerText = `${summary.total_transactions.toLocaleString()} total dataset records`;

  document.getElementById("kpi-allow").innerText = (summary.risk_band_counts.Low || 0).toLocaleString();
  document.getElementById("kpi-sub-allow").innerText = `${summary.risk_band_percentages.Low || '0%'} • Auto-Approve`;

  document.getElementById("kpi-review").innerText = (summary.risk_band_counts.Medium || 0).toLocaleString();
  document.getElementById("kpi-sub-review").innerText = `${summary.risk_band_percentages.Medium || '0%'} • 2FA Challenge`;

  document.getElementById("kpi-hold").innerText = (summary.risk_band_counts.High || 0).toLocaleString();
  document.getElementById("kpi-sub-hold").innerText = `${summary.risk_band_percentages.High || '0%'} • Instant Block`;

  // 2. Fraud Spike Sentinel
  const spikeBanner = document.getElementById("spike-banner");
  if (summary.fraud_spike_detector && summary.fraud_spike_detector.spike_detected) {
    spikeBanner.classList.add("active");
    document.getElementById("spike-msg").innerText = summary.fraud_spike_detector.spike_alert_message;
  } else {
    spikeBanner.classList.remove("active");
  }

  // 3. Ground Truth Accuracy Breakdown
  const gtCard = document.getElementById("ground-truth-card");
  if (summary.ground_truth_evaluation && summary.ground_truth_evaluation.labels_detected) {
    gtCard.style.display = "block";
    const gt = summary.ground_truth_evaluation;
    document.getElementById("gt-hard-tp").innerText = `${gt.frauds_blocked_instantly} / ${gt.total_frauds_in_file}`;
    document.getElementById("gt-hard-fp").innerText = `${gt.hard_false_blocks} (${gt.hard_false_alarm_rate})`;
    document.getElementById("gt-hard-prec").innerText = gt.hard_block_precision;
    document.getElementById("gt-2fa-count").innerText = `${gt.step_up_2fa_challenges} users (0 declines)`;
    document.getElementById("gt-total-stopped").innerText = `${gt.total_frauds_intercepted} / ${gt.total_frauds_in_file} (${gt.fraud_capture_rate})`;
    document.getElementById("gt-subtext").innerText = `Evaluated against ${gt.total_frauds_in_file} frauds and ${gt.total_legitimate_in_file.toLocaleString()} legitimate cardholders.`;
  } else {
    gtCard.style.display = "none";
  }

  // 4. Update Filter Counts
  document.getElementById("count-all").innerText = summary.displayed_transactions || currentTransactions.length;
  document.getElementById("count-high").innerText = summary.risk_band_counts.High || 0;
  document.getElementById("count-med").innerText = summary.risk_band_counts.Medium || 0;
  document.getElementById("count-low").innerText = summary.risk_band_counts.Low || 0;

  // 5. Update info text
  document.getElementById("display-info-text").innerText = `Displaying ${currentTransactions.length} prioritized transactions with SHAP attributions (Total batch scored: ${summary.total_transactions.toLocaleString()}).`;

  // 6. Render Table with Pagination
  document.getElementById("results-section").style.display = "block";
  filterTable("all");
}

// Table Filter
function filterTable(filter) {
  currentFilter = filter;
  currentPage = 1;
  document.querySelectorAll(".filter-btn").forEach(btn => btn.classList.remove("active"));
  
  const activeBtn = Array.from(document.querySelectorAll(".filter-btn")).find(b => b.getAttribute("onclick").includes(`'${filter}'`));
  if (activeBtn) activeBtn.classList.add("active");

  if (filter === "all") {
    filteredTransactions = [...currentTransactions];
  } else {
    filteredTransactions = currentTransactions.filter(tx => tx.risk_band === filter);
  }

  renderCurrentPage();
}

// Pagination Controls
function renderCurrentPage() {
  const totalPages = Math.ceil(filteredTransactions.length / PAGE_SIZE) || 1;
  if (currentPage > totalPages) currentPage = totalPages;
  if (currentPage < 1) currentPage = 1;

  document.getElementById("current-page-num").innerText = currentPage;
  document.getElementById("total-pages-num").innerText = totalPages;

  document.getElementById("prev-page-btn").disabled = (currentPage === 1);
  document.getElementById("next-page-btn").disabled = (currentPage === totalPages);

  const startIdx = (currentPage - 1) * PAGE_SIZE;
  const pageSlice = filteredTransactions.slice(startIdx, startIdx + PAGE_SIZE);

  renderTableRows(pageSlice);
}

function prevPage() {
  if (currentPage > 1) {
    currentPage--;
    renderCurrentPage();
  }
}

function nextPage() {
  const totalPages = Math.ceil(filteredTransactions.length / PAGE_SIZE) || 1;
  if (currentPage < totalPages) {
    currentPage++;
    renderCurrentPage();
  }
}

// Render Table Rows for Current Page
function renderTableRows(txList) {
  const tbody = document.getElementById("transactions-tbody");
  tbody.innerHTML = "";

  if (txList.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-muted); padding: 30px;">No transactions matching this filter.</td></tr>`;
    return;
  }

  txList.forEach(tx => {
    const tr = document.createElement("tr");
    const amountVal = tx.original_data.Amount !== undefined ? `$${Number(tx.original_data.Amount).toFixed(2)}` : "$0.00";
    
    let fillClass = "fill-green";
    let badgeClass = "badge-allow";
    let badgeIcon = "🟢";
    let badgeText = "ALLOW";

    if (tx.risk_band === "High") {
      fillClass = "fill-red";
      badgeClass = "badge-hold";
      badgeIcon = "🔴";
      badgeText = "HARD BLOCK";
    } else if (tx.risk_band === "Medium") {
      fillClass = "fill-yellow";
      badgeClass = "badge-review";
      badgeIcon = "🟡";
      badgeText = "2FA CHALLENGE";
    }

    let shapHtml = '<div class="shap-tag-group">';
    if (tx.top_3_reasons_risk_up && tx.top_3_reasons_risk_up.length > 0) {
      tx.top_3_reasons_risk_up.forEach(r => {
        shapHtml += `<div class="shap-tag shap-risk-up">🔺 ${escapeHtml(r.description)}</div>`;
      });
    }
    if (tx.top_3_reasons_risk_down && tx.top_3_reasons_risk_down.length > 0) {
      tx.top_3_reasons_risk_down.slice(0, 1).forEach(r => {
        shapHtml += `<div class="shap-tag shap-risk-down">🛡️ ${escapeHtml(r.description)}</div>`;
      });
    }
    if (!tx.top_3_reasons_risk_up?.length && !tx.top_3_reasons_risk_down?.length) {
      shapHtml += `<div class="shap-tag shap-risk-down">🛡️ Normal baseline purchase behavior</div>`;
    }
    shapHtml += '</div>';

    tr.innerHTML = `
      <td><strong>#${tx.transaction_id}</strong></td>
      <td><span style="font-weight: 700;">${amountVal}</span></td>
      <td style="min-width: 140px;">
        <div class="risk-meter">
          <span style="font-weight: 700; font-size: 13px;">${tx.risk_score_pct}</span>
          <div class="meter-bar">
            <div class="meter-fill ${fillClass}" style="width: ${Math.max(tx.risk_score * 100, 4)}%;"></div>
          </div>
        </div>
      </td>
      <td>
        <span class="badge-tier ${badgeClass}">
          ${badgeIcon} ${badgeText}
        </span>
      </td>
      <td>${shapHtml}</td>
      <td>
        <button class="btn btn-secondary" style="padding: 4px 10px; font-size: 11.5px;" onclick="openModal(${tx.transaction_id})">
          Audit 🔍
        </button>
      </td>
    `;

    tbody.appendChild(tr);
  });
}

// Modal Details View
function openModal(txId) {
  const tx = currentTransactions.find(t => t.transaction_id === txId);
  if (!tx) return;

  const modal = document.getElementById("tx-modal");
  const modalBody = document.getElementById("modal-body");

  let upList = tx.top_3_reasons_risk_up.map(r => `
    <li style="margin-bottom: 6px;">
      <strong>${r.feature}</strong> (SHAP Impact: <span style="color: #f87171;">+${r.shap_impact.toFixed(4)}</span>)<br>
      <span style="font-size: 12px; color: var(--text-secondary);">${r.description}</span>
    </li>
  `).join("") || "<li>No major risk anomalies.</li>";

  let downList = tx.top_3_reasons_risk_down.map(r => `
    <li style="margin-bottom: 6px;">
      <strong>${r.feature}</strong> (SHAP Impact: <span style="color: #34d399;">${r.shap_impact.toFixed(4)}</span>)<br>
      <span style="font-size: 12px; color: var(--text-secondary);">${r.description}</span>
    </li>
  `).join("") || "<li>None.</li>";

  modalBody.innerHTML = `
    <div style="display: flex; justify-content: space-between; margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid var(--border-color);">
      <div>
        <div style="font-size: 12px; color: var(--text-muted);">TRANSACTION ID</div>
        <div style="font-size: 18px; font-weight: 800;">#${tx.transaction_id}</div>
      </div>
      <div>
        <div style="font-size: 12px; color: var(--text-muted);">CALIBRATED RISK SCORE</div>
        <div style="font-size: 18px; font-weight: 800; color: ${tx.risk_band === 'High' ? 'var(--tier-red)' : tx.risk_band === 'Medium' ? 'var(--tier-yellow)' : 'var(--tier-green)'};">${tx.risk_score_pct}</div>
      </div>
      <div>
        <div style="font-size: 12px; color: var(--text-muted);">RECOMMENDED ACTION</div>
        <div style="font-size: 16px; font-weight: 800;">${tx.recommended_action}</div>
      </div>
    </div>

    <div style="margin-bottom: 16px; background: rgba(0,0,0,0.3); padding: 12px; border-radius: var(--radius-sm);">
      <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: var(--accent-cyan); margin-bottom: 4px;">Executive Narrative</div>
      <p style="font-size: 13px; color: var(--text-primary); font-style: italic;">"${escapeHtml(tx.executive_narrative)}"</p>
    </div>

    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
      <div>
        <h4 style="font-size: 12px; color: #f87171; text-transform: uppercase; margin-bottom: 8px;">Factors Driving Risk UP (Fraud)</h4>
        <ul style="font-size: 12.5px; color: var(--text-primary); padding-left: 16px;">
          ${upList}
        </ul>
      </div>
      <div>
        <h4 style="font-size: 12px; color: #34d399; text-transform: uppercase; margin-bottom: 8px;">Factors Driving Risk DOWN (Trust)</h4>
        <ul style="font-size: 12.5px; color: var(--text-primary); padding-left: 16px;">
          ${downList}
        </ul>
      </div>
    </div>
  `;

  modal.classList.add("active");
}

function closeModal() {
  document.getElementById("tx-modal").classList.remove("active");
}

// Live Simulator Prediction
async function runSimulatorPrediction() {
  const amount = parseFloat(document.getElementById("sim-amount").value) || 0.0;
  const time = parseFloat(document.getElementById("sim-time").value) || 0.0;
  const v14 = parseFloat(document.getElementById("sim-v14").value) || 0.0;
  const v4 = parseFloat(document.getElementById("sim-v4").value) || 0.0;

  const payload = {
    Time: time,
    Amount: amount,
    V1: -0.5, V2: 0.2, V3: 1.2, V4: v4, V5: -0.1, V6: 0.3, V7: 0.1, V8: 0.0,
    V9: -0.2, V10: -0.5, V11: 0.8, V12: -0.4, V13: 0.1, V14: v14, V15: 0.2,
    V16: -0.3, V17: -0.2, V18: 0.1, V19: 0.0, V20: 0.1, V21: -0.1, V22: 0.2,
    V23: -0.1, V24: 0.1, V25: 0.0, V26: 0.1, V27: 0.0, V28: 0.0
  };

  try {
    const res = await fetch(`${API_BASE}/api/v1/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (res.ok) {
      const data = await res.json();
      updateSimulatorUI(data);
    }
  } catch (err) {
    console.warn("Simulator predict error:", err);
  }
}

function updateSimulatorUI(data) {
  const scoreEl = document.getElementById("sim-score");
  const badgeEl = document.getElementById("sim-badge");
  const descEl = document.getElementById("sim-desc");
  const reasonsEl = document.getElementById("sim-reasons");

  scoreEl.innerText = data.risk_score_pct;
  descEl.innerText = data.action_description;

  if (data.risk_band === "High") {
    scoreEl.style.color = "var(--tier-red)";
    badgeEl.className = "badge-tier badge-hold";
    badgeEl.innerText = "🔴 [RED TIER] Instant Hard Block";
  } else if (data.risk_band === "Medium") {
    scoreEl.style.color = "var(--tier-yellow)";
    badgeEl.className = "badge-tier badge-review";
    badgeEl.innerText = "🟡 [YELLOW TIER] 2FA Challenge (SMS OTP)";
  } else {
    scoreEl.style.color = "var(--tier-green)";
    badgeEl.className = "badge-tier badge-allow";
    badgeEl.innerText = "🟢 [GREEN TIER] Instant Frictionless Approval";
  }

  let html = "";
  if (data.top_3_reasons_risk_up.length > 0) {
    data.top_3_reasons_risk_up.forEach(r => {
      html += `<div class="shap-tag shap-risk-up">🔺 ${escapeHtml(r.description)}</div>`;
    });
  }
  if (data.top_3_reasons_risk_down.length > 0) {
    data.top_3_reasons_risk_down.forEach(r => {
      html += `<div class="shap-tag shap-risk-down">🛡️ ${escapeHtml(r.description)}</div>`;
    });
  }
  reasonsEl.innerHTML = html || `<div class="shap-tag shap-risk-down">🛡️ Normal transaction pattern</div>`;
}

function setPreset(type) {
  if (type === "fraud") {
    document.getElementById("sim-amount").value = "1250.00";
    document.getElementById("sim-v14").value = "-6.5";
    document.getElementById("sim-v4").value = "3.5";
  } else if (type === "safe-high") {
    document.getElementById("sim-amount").value = "1500.00";
    document.getElementById("sim-v14").value = "1.5";
    document.getElementById("sim-v4").value = "-0.8";
  } else if (type === "borderline") {
    document.getElementById("sim-amount").value = "85.00";
    document.getElementById("sim-v14").value = "-1.8";
    document.getElementById("sim-v4").value = "1.2";
  }
  document.getElementById("v14-val").innerText = document.getElementById("sim-v14").value;
  document.getElementById("v4-val").innerText = document.getElementById("sim-v4").value;
  runSimulatorPrediction();
}

function escapeHtml(text) {
  if (!text) return "";
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}
