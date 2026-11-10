"""Standalone UI for `LMFT-Scholar` — Foundational Two-Hemisphere Socratic & Curriculum Studio (`scholar_ui.py`).

Plain-English Purpose:
Provides a clean dark-minimalist single-page interface for `LMFT-Scholar` (`Impish_LLAMA_4B` in native 16-bit
paired with the 40-node Right-Brain Fuzzy Tree, 3 MiniLM bridges, and 3-Branch Archives):
  1. Foundational Curriculum Shelf & Step-by-Step Study Loop across Cultural, Philosophical, Experiential, and Psychological dilemmas.
  2. Reading & Study Passage Canvas for analyzing custom texts, parables, and questions.
  3. Socratic Dialogue Stream with per-turn Right-Brain Telemetry, Savoring (3-Branch Archive harvesting), Coaching (REM Sleep queueing), and one-click Brain Graduation to `LMFT-Chess` or `LMFT-Bloodmoon`.
"""

from __future__ import annotations

SCHOLAR_SINGLE_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LMFT-Scholar · Two-Hemisphere Socratic &amp; Curriculum Studio</title>
<style>
  :root {
    --bg-0: #0a0d14;
    --bg-1: #111726;
    --bg-2: #182033;
    --bg-3: #222d47;
    --border: #2b3958;
    --text-0: #f3f6fc;
    --text-1: #b8c2d9;
    --text-2: #7e8ba8;
    --accent: #f59e0b;
    --accent-soft: rgba(245, 158, 11, 0.16);
    --right-accent: #38bdf8;
    --sleep-accent: #a855f7;
    --ok: #10b981;
    --warn: #ef4444;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg-0);
    color: var(--text-0);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  header {
    height: 54px;
    background: var(--bg-1);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 16px;
    flex-shrink: 0;
    gap: 10px;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 700;
    font-size: 15px;
    letter-spacing: 0.3px;
    white-space: nowrap;
  }
  .brand-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 10px var(--accent);
  }
  .status-strip {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: var(--text-1);
    flex-wrap: wrap;
  }
  .pill {
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 4px 10px;
    font-size: 11px;
    white-space: nowrap;
  }
  .btn {
    background: var(--bg-2);
    color: var(--text-0);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 11px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
    white-space: nowrap;
  }
  .btn:hover { background: var(--bg-3); border-color: var(--text-2); }
  .btn-primary {
    background: var(--accent);
    border-color: var(--accent);
    color: #090d16;
  }
  .btn-primary:hover { filter: brightness(1.08); }
  .btn-loop-active {
    background: var(--ok);
    border-color: var(--ok);
    color: #052e16;
  }
  .curriculum-banner {
    background: linear-gradient(90deg, rgba(245,158,11,0.10), rgba(56,189,248,0.10), rgba(168,85,247,0.10));
    border-bottom: 1px solid var(--border);
    padding: 10px 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
    flex-shrink: 0;
  }
  .curriculum-controls {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  input[type="text"], select, textarea {
    background: var(--bg-1);
    color: var(--text-0);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
  }
  .workspace {
    flex: 1;
    display: grid;
    grid-template-columns: 280px 1fr 1.2fr;
    min-height: 0;
  }
  .sidebar {
    background: var(--bg-1);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    padding: 12px;
    gap: 10px;
    overflow-y: auto;
  }
  .thread-item {
    padding: 8px 10px;
    border-radius: 6px;
    border: 1px solid transparent;
    background: var(--bg-2);
    cursor: pointer;
    font-size: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .thread-item.active { border-color: var(--accent); background: var(--bg-3); }
  .telemetry-box {
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px;
    font-size: 11px;
    line-height: 1.5;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .canvas-pane {
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    padding: 14px;
    gap: 10px;
    min-height: 0;
  }
  #study-canvas-textarea {
    flex: 1;
    width: 100%;
    resize: none;
    font-size: 13px;
    line-height: 1.6;
    padding: 12px;
    background: var(--bg-1);
  }
  .stream-pane {
    display: flex;
    flex-direction: column;
    padding: 14px;
    gap: 10px;
    min-height: 0;
  }
  #chat-messages {
    flex: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding-right: 4px;
  }
  .msg-card {
    background: var(--bg-1);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px;
    font-size: 13px;
    line-height: 1.5;
  }
  .msg-role {
    font-size: 11px;
    font-weight: 700;
    color: var(--text-1);
    margin-bottom: 4px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
  }
  .msg-text { white-space: pre-wrap; }
  .msg-actions {
    margin-top: 8px;
    display: flex;
    gap: 6px;
    align-items: center;
    flex-wrap: wrap;
  }
  .btn-mini {
    background: var(--bg-2);
    color: var(--text-1);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 3px 8px;
    font-size: 11px;
    cursor: pointer;
  }
  .btn-mini:hover { background: var(--bg-3); color: var(--text-0); }
  .chat-bar {
    display: flex;
    gap: 8px;
    align-items: center;
  }
  #chat-input { flex: 1; }
  .invariant-note {
    font-size: 11px;
    color: var(--text-2);
  }
  .tree-inspector-pane {
    flex: 1;
    display: none;
    flex-direction: column;
    padding: 16px;
    gap: 14px;
    overflow-y: auto;
    min-height: 0;
  }
  .tree-summary-strip {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 10px;
  }
  .tree-stat-card {
    background: var(--bg-1);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px;
  }
  .tree-stat-label {
    font-size: 11px;
    color: var(--text-2);
    margin-bottom: 4px;
  }
  .tree-stat-val {
    font-size: 16px;
    font-weight: 700;
    color: var(--text-0);
  }
  .dept-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
  }
  .dept-card {
    background: var(--bg-1);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .bough-block {
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 9px 10px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .leaf-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    font-size: 11px;
    padding: 3px 6px;
    border-radius: 4px;
    background: rgba(10, 13, 20, 0.45);
  }
  .leaf-row.grown-tier-2 {
    border-left: 3px solid var(--accent);
    background: rgba(245, 158, 11, 0.10);
  }
  .leaf-row.grown-tier-3 {
    margin-left: 14px;
    border-left: 3px solid var(--sleep-accent);
    background: rgba(168, 85, 247, 0.12);
  }
  .bar-track {
    width: 72px;
    height: 6px;
    background: var(--bg-0);
    border-radius: 999px;
    overflow: hidden;
    flex-shrink: 0;
  }
  .bar-fill {
    height: 100%;
    background: var(--right-accent);
    border-radius: 999px;
  }
</style>
</head>
<body>
<header>
  <div class="brand">
    <span class="brand-dot"></span>
    <span>LMFT-Scholar · Foundational Two-Hemisphere Socratic Studio</span>
  </div>
  <div class="status-strip">
    <span class="pill" id="model-status-pill">Impish_LLAMA_4B</span>
    <span class="pill" id="rules-count-pill">Rules: 2</span>
    <span class="pill" id="leaves-count-pill">Leaves: 24+</span>
    <span class="pill" id="sleep-queue-pill">Sleep Queue: 0</span>
    <select id="lineage-select" onchange="switchLineage(this.value)" title="Active Brain Lineage">
      <option value="scholar" selected>Lineage: Scholar Studio (Trainable)</option>
      <option value="virgin">Lineage: Cycle 0 Virgin (Read-Only)</option>
    </select>
    <button class="btn" id="toggle-tree-inspector-btn" onclick="toggleTreeInspectorView()">🌳 Visual Tree &amp; Canopy Inspector</button>
    <button class="btn" onclick="runRemSleepNow()">REM Sleep Now</button>
    <button class="btn" onclick="graduateBrain('chess')" title="Export learned domain-general rules, archives &amp; Will edges to LMFT-Chess">Graduate ➔ Chess</button>
    <button class="btn" onclick="graduateBrain('bloodmoon')" title="Export learned domain-general rules, archives &amp; Will edges to LMFT-Bloodmoon">Graduate ➔ Bloodmoon</button>
  </div>
</header>

<section class="curriculum-banner" id="curriculum-evolution-bar">
  <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
    <strong style="font-size:12px;">12-Book Classic Literature Seminar:</strong>
    <select id="lesson-select" style="min-width:300px;"></select>
    <span class="pill" id="lesson-step-pill">Steps: 0</span>
  </div>
  <div class="curriculum-controls">
    <label style="font-size:11px;display:flex;align-items:center;gap:5px;cursor:pointer;" title="Automatically enter REM Sleep after each book essay to sprout Tier-2 and Tier-3 leaves">
      <input type="checkbox" id="sleep-after-essay-toggle" checked />
      Sleep &amp; Grow Canopy After Essay
    </label>
    <span class="pill" title="Left Brain stops naturally when Right-Brain equilibrium settles">Natural Equilibrium Flow</span>
    <button class="btn btn-primary" id="step-curriculum-btn" onclick="stepCurriculumLesson()">Step Literature Essay</button>
    <button class="btn" id="loop-curriculum-btn" onclick="toggleCurriculumLoop()">Run Seminar Loop</button>
  </div>
</section>

<div class="workspace" id="scholar-studio-view">
  <aside class="sidebar">
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <strong style="font-size:12px;">Study Sessions</strong>
      <button class="btn" onclick="createNewThread()">+ New</button>
    </div>
    <div id="thread-list-container" style="display:flex;flex-direction:column;gap:6px;"></div>
    <label style="font-size:11px;display:flex;align-items:center;gap:6px;margin-top:4px;">
      <input type="checkbox" id="auto-sleep-toggle" checked onchange="toggleAutoSleep(this.checked)" />
      Auto-REM Sleep on Queue Threshold
    </label>
    <hr style="border:none;border-top:1px solid var(--border);margin:4px 0;" />
    <div class="telemetry-box" id="right-brain-telemetry-box">
      <strong>40-Node Right-Brain Telemetry</strong>
      <div class="invariant-note" id="telemetry-summary-text">Send a Socratic prompt or step a Curriculum Lesson to inspect live Pass-1 Fact vs. Pass-2 Theory coordinates.</div>
    </div>
  </aside>

  <section class="canvas-pane">
    <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;">
      <div>
        <strong style="font-size:13px;">Reading &amp; Study Passage Canvas</strong>
        <div class="invariant-note">Paste a parable, philosophical dilemma, or essay to contemplate with the 40-node tree</div>
      </div>
      <div style="display:flex;gap:6px;">
        <button class="btn btn-primary" id="study-passage-btn" onclick="studyCanvasPassage()">Contemplate Passage ➔</button>
        <button class="btn" onclick="saveStudyCanvas()">Save Canvas</button>
      </div>
    </div>
    <textarea id="study-canvas-textarea" oninput="markCanvasDirty()" placeholder="Paste a passage, dilemma, or thought experiment here to study together, or select a lesson from the Foundational Curriculum bar above..."></textarea>
    <div class="invariant-note" id="last-scholar-receipt">Ready for Socratic contemplation or Curriculum study.</div>
  </section>

  <section class="stream-pane">
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <strong style="font-size:13px;">Socratic Dialogue &amp; Coaching Stream</strong>
      <span class="invariant-note" id="active-checkpoint-label">Checkpoint: models/left_brain_scholar</span>
    </div>
    <div id="chat-messages"></div>
    <div class="chat-bar">
      <input type="text" id="chat-input" placeholder="Chat naturally, ask a Socratic question, or pose a dilemma..." onkeydown="if(event.key==='Enter')sendChatMessage()" />
      <label style="font-size:11px;display:flex;align-items:center;gap:5px;cursor:pointer;white-space:nowrap;color:var(--text-1);" title="When checked, grounds your chat message in the Reading Canvas passage on the left">
        <input type="checkbox" id="anchor-canvas-toggle" />
        Use Canvas
      </label>
      <button class="btn btn-primary" onclick="sendChatMessage()">Send</button>
    </div>
  </section>
</div>

<div class="tree-inspector-pane" id="tree-inspector-view">
  <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;">
    <div>
      <strong style="font-size:15px;">🌳 Visual Tree &amp; Canopy Development Inspector</strong>
      <div class="invariant-note">Live view of the 40 Founding Seed Nodes, sprouted Tier-2 &amp; Tier-3 Learned Leaves, Callosum anchor weights, Fuzzy Concept Rules, and 9 Will Synapses.</div>
    </div>
    <button class="btn" onclick="toggleTreeInspectorView()">← Return to Socratic Studio</button>
  </div>
  <div class="tree-summary-strip" id="tree-summary-strip"></div>
  <div class="dept-grid" id="tree-dept-grid"></div>
  <div style="display:grid;grid-template-columns:1.4fr 1fr;gap:12px;">
    <div class="dept-card">
      <strong style="font-size:13px;">Consolidated Domain-General Fuzzy Rules (Right-Brain Canopy)</strong>
      <div id="tree-rules-list" style="display:flex;flex-direction:column;gap:6px;font-size:11px;"></div>
    </div>
    <div class="dept-card">
      <strong style="font-size:13px;">9 Plastic Will &amp; Contemplation Synapses (will_edges.json)</strong>
      <div id="tree-will-list" style="display:flex;flex-direction:column;gap:6px;font-size:11px;"></div>
    </div>
  </div>
</div>

<script>
let lastRenderedChatSig = "";
let curriculumLoopTimer = null;
let scholarBusy = false;
let canvasDirty = false;
let lastServerCanvasText = null;
let populatedLessons = false;
let treeInspectorOpen = false;

function toggleTreeInspectorView() {
  treeInspectorOpen = !treeInspectorOpen;
  const studioEl = document.getElementById("scholar-studio-view");
  const treeEl = document.getElementById("tree-inspector-view");
  const btnEl = document.getElementById("toggle-tree-inspector-btn");
  if (treeInspectorOpen) {
    studioEl.style.display = "none";
    treeEl.style.display = "flex";
    if (btnEl) btnEl.classList.add("btn-primary");
  } else {
    studioEl.style.display = "grid";
    treeEl.style.display = "none";
    if (btnEl) btnEl.classList.remove("btn-primary");
  }
}

function markCanvasDirty() {
  canvasDirty = true;
}

function escapeHtml(str) {
  return String(str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function apiPost(path, payload) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {})
  });
  const data = await res.json();
  if (!res.ok) {
    const errMsg = data.error || "Request failed";
    const receiptEl = document.getElementById("last-scholar-receipt");
    if (receiptEl) receiptEl.textContent = "Notice: " + errMsg;
    throw new Error(errMsg);
  }
  renderState(data);
  return data;
}

function renderState(s) {
  if (!s) return;
  document.getElementById("model-status-pill").textContent = s.left_brain_status || "Impish_LLAMA_4B";
  document.getElementById("rules-count-pill").textContent = "Rules: " + (s.rules_count || 0);
  document.getElementById("leaves-count-pill").textContent = "Leaves: " + (24 + Number(s.learned_leaves_count || 0));
  document.getElementById("sleep-queue-pill").textContent = "Sleep Queue: " + (s.sleep_queue_count || 0);
  document.getElementById("lesson-step-pill").textContent = "Steps: " + (s.curriculum_step_counter || 0);
  document.getElementById("active-checkpoint-label").textContent = s.active_checkpoint_label || "models/left_brain_scholar";

  const linSel = document.getElementById("lineage-select");
  if (linSel && s.active_lineage) linSel.value = s.active_lineage;

  if (!populatedLessons && Array.isArray(s.curriculum_lessons) && s.curriculum_lessons.length > 0) {
    const sel = document.getElementById("lesson-select");
    sel.innerHTML = `<option value="">Next in Sequence (Auto-Rotate)</option>` + s.curriculum_lessons.map(l =>
      `<option value="${escapeHtml(l.lesson_id)}">[${escapeHtml(l.department)}] ${escapeHtml(l.title)}</option>`
    ).join("");
    populatedLessons = true;
  }

  const canvasEl = document.getElementById("study-canvas-textarea");
  if (typeof s.study_canvas_text === "string") {
    const serverChanged = s.study_canvas_text !== lastServerCanvasText;
    lastServerCanvasText = s.study_canvas_text;
    if (serverChanged || (!canvasDirty && document.activeElement !== canvasEl)) {
      canvasEl.value = s.study_canvas_text;
      if (serverChanged) canvasDirty = false;
    }
  }

  const receiptEl = document.getElementById("last-scholar-receipt");
  if (s.last_graduation_receipt) {
    receiptEl.textContent = s.last_graduation_receipt;
  } else if (s.last_rem_sleep_receipt) {
    receiptEl.textContent = s.last_rem_sleep_receipt.split("\\n")[0];
  }

  const threadBox = document.getElementById("thread-list-container");
  if (threadBox && Array.isArray(s.threads)) {
    threadBox.innerHTML = s.threads.map(t => `
      <div class="thread-item ${t.thread_id === s.active_thread_id ? 'active' : ''}" onclick="switchThread('${escapeHtml(t.thread_id)}')">
        <span>${escapeHtml(t.title)}</span>
        <small style="color:var(--text-2)">${t.message_count || 0}</small>
      </div>
    `).join("");
  }

  const msgs = s.chat_messages || [];
  const lastCompanion = [...msgs].reverse().find(m => m.sender === "companion" && m.telemetry);
  if (lastCompanion && lastCompanion.telemetry) {
    const tel = lastCompanion.telemetry;
    const fc = tel.fact_coordinates || {};
    const topLeaves = Object.entries(fc)
      .sort((a, b) => Number(b[1]) - Number(a[1]))
      .slice(0, 5)
      .map(([k, v]) => `${k.split("/").slice(-1)[0]}=${Number(v).toFixed(2)}`)
      .join(", ");
    document.getElementById("telemetry-summary-text").innerHTML =
      `<div><strong>Gear:</strong> ${escapeHtml(tel.gear_used || "")}</div>` +
      `<div><strong>Outcome:</strong> ${escapeHtml(tel.outcome || "")} (${tel.settled ? "Settled" : "Queued for Sleep"})</div>` +
      `<div><strong>Active Concept:</strong> ${escapeHtml(tel.active_concept || "none")}</div>` +
      `<div><strong>Truth (Fact / Theory):</strong> ${Number(tel.fact || 0).toFixed(2)} / ${Number(tel.theory || 0).toFixed(2)}</div>` +
      `<div><strong>Will (Intention / Exec):</strong> ${Number(tel.intention || 0).toFixed(2)} / ${Number(tel.execution || 0).toFixed(2)}</div>` +
      `<div><strong>Clarity / Conf:</strong> ${Number(tel.clarity || 0).toFixed(2)} / ${Number(tel.confidence || 0).toFixed(2)}</div>` +
      `<div><strong>Valence / Arousal:</strong> ${Number(tel.valence || 0).toFixed(2)} / ${Number(tel.arousal || 0).toFixed(2)}</div>` +
      `<div><strong>Top Fact Leaves:</strong> ${escapeHtml(topLeaves)}</div>`;
  }

  const sig = JSON.stringify(msgs.map(m => [m.id, m.text, m.coached, m.savored, m.auto_rem_sleep_receipt]));
  if (sig !== lastRenderedChatSig) {
    lastRenderedChatSig = sig;
    const box = document.getElementById("chat-messages");
    box.innerHTML = msgs.map(m => {
      const isComp = m.sender === "companion" && typeof m.id === "number";
      const badge = (m.telemetry && m.telemetry.active_concept)
        ? `<span class="pill">${escapeHtml(m.telemetry.active_concept)} · ${escapeHtml(m.telemetry.gear_used || "")}</span>`
        : "";
      const actions = isComp ? `
        <div class="msg-actions">
          <button class="btn-mini" onclick="savorTurn(${m.id})">${m.savored ? "★ Savored in Archives" : "☆ Savor Vibe (Archive)"}</button>
          <button class="btn-mini" onclick="coachTurn(${m.id})">${m.coached ? "⚑ Coached (In Sleep Queue)" : "⚑ Coach / Challenge"}</button>
        </div>
      ` : "";
      return `
        <div class="msg-card">
          <div class="msg-role">
            <span>${escapeHtml(m.role_label || m.sender)}</span>
            ${badge}
          </div>
          <div class="msg-text">${escapeHtml(m.text)}</div>
          ${actions}
          ${m.auto_rem_sleep_receipt ? `<div style="margin-top:8px;padding:6px 8px;border-radius:6px;background:rgba(168,85,247,0.12);border:1px solid rgba(168,85,247,0.3);font-size:11px;">${escapeHtml(m.auto_rem_sleep_receipt)}</div>` : ""}
        </div>
      `;
    }).join("");
    box.scrollTop = box.scrollHeight;
  }

  if (s.tree_inspector) {
    renderTreeInspector(s.tree_inspector);
  }
}

function renderTreeInspector(ti) {
  const stripEl = document.getElementById("tree-summary-strip");
  if (stripEl) {
    const tc = ti.tier_counts || {};
    const arc = ti.archive_counts || {};
    stripEl.innerHTML = `
      <div class="tree-stat-card">
        <div class="tree-stat-label">Total Tree Nodes</div>
        <div class="tree-stat-val">${ti.total_tree_nodes || 40} <span style="font-size:11px;color:var(--text-2);font-weight:400;">(40 Seed + ${ti.learned_leaves_count || 0} Grown)</span></div>
      </div>
      <div class="tree-stat-card">
        <div class="tree-stat-label">Canopy Leaf Tiers</div>
        <div class="tree-stat-val">T1: ${tc.tier_1 || 24} · T2: ${tc.tier_2 || 0} · T3: ${tc.tier_3 || 0}</div>
      </div>
      <div class="tree-stat-card">
        <div class="tree-stat-label">Consolidated Fuzzy Rules</div>
        <div class="tree-stat-val">${ti.rules_count || 0}</div>
      </div>
      <div class="tree-stat-card">
        <div class="tree-stat-label">3-Branch Archives</div>
        <div class="tree-stat-val" style="font-size:13px;">Exec: ${arc.executive || 0} · Leg: ${arc.legislative || 0} · Jud: ${arc.judicial || 0}</div>
      </div>
      <div class="tree-stat-card">
        <div class="tree-stat-label">Weight-Space Plasticity</div>
        <div class="tree-stat-val" style="font-size:12px;color:${ti.has_live_callosum_projector ? 'var(--ok)' : 'var(--text-1)'};">${ti.has_live_callosum_projector ? 'Callosum + LoRA Live' : 'Right-Brain Canopy + Will'}</div>
      </div>
    `;
  }

  const deptGrid = document.getElementById("tree-dept-grid");
  if (deptGrid && Array.isArray(ti.departments)) {
    deptGrid.innerHTML = ti.departments.map(dept => {
      const boughsHtml = (dept.boughs || []).map(b => {
        const foundingHtml = (b.founding_leaves || []).map(fl => {
          const pct = Math.max(0, Math.min(100, Math.round(Number(fl.activation || 0) * 100)));
          return `
            <div class="leaf-row">
              <span><strong>[T1]</strong> ${escapeHtml(fl.short_name)}</span>
              <div style="display:flex;align-items:center;gap:6px;">
                <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
                <span style="width:28px;text-align:right;">${Number(fl.activation || 0).toFixed(2)}</span>
              </div>
            </div>
          `;
        }).join("");

        const grownHtml = (b.learned_leaves || []).map(gl => {
          const tier = Number(gl.tier || 2);
          const pct = Math.max(0, Math.min(100, Math.round(Number(gl.activation || 0) * 100)));
          const topAnchors = Object.entries(gl.anchor_weights || {})
            .map(([k, v]) => `${k.split("/").slice(-1)[0]}:${Number(v).toFixed(2)}`)
            .join(", ");
          return `
            <div class="leaf-row ${tier >= 3 ? 'grown-tier-3' : 'grown-tier-2'}" title="${escapeHtml(gl.description || '')}">
              <div>
                <div><strong>[T${tier} Grown]</strong> ${escapeHtml(gl.short_name)}</div>
                <div style="font-size:10px;color:var(--text-2);">${escapeHtml(gl.description || '')}${topAnchors ? ' · Callosum Anchors: ' + escapeHtml(topAnchors) : ''}</div>
              </div>
              <div style="display:flex;align-items:center;gap:6px;">
                <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${tier >= 3 ? 'var(--sleep-accent)' : 'var(--accent)'};"></div></div>
                <span style="width:28px;text-align:right;">${Number(gl.activation || 0).toFixed(2)}</span>
              </div>
            </div>
          `;
        }).join("");

        return `
          <div class="bough-block">
            <div style="display:flex;justify-content:space-between;align-items:center;font-size:12px;">
              <strong>Bough: ${escapeHtml(b.title)} <span style="font-weight:400;color:var(--text-2);">(${escapeHtml(b.bough_id)})</span></strong>
              <span class="pill">Act: ${Number(b.activation || 0).toFixed(2)} · Grown: ${(b.learned_leaves || []).length}</span>
            </div>
            ${foundingHtml}
            ${grownHtml}
          </div>
        `;
      }).join("");

      return `
        <div class="dept-card">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <strong style="font-size:14px;">Department: ${escapeHtml(dept.title)}</strong>
            <span class="pill">Dept Activation: ${Number(dept.activation || 0).toFixed(2)}</span>
          </div>
          ${boughsHtml}
        </div>
      `;
    }).join("");
  }

  const rulesEl = document.getElementById("tree-rules-list");
  if (rulesEl && Array.isArray(ti.rules)) {
    rulesEl.innerHTML = ti.rules.map(r => {
      const clauses = Object.entries(r.top_clauses || {})
        .map(([k, v]) => `${k.split("/").slice(-1)[0]}≥${Number(v).toFixed(2)}`)
        .join(", ");
      return `
        <div class="leaf-row" style="padding:6px 8px;">
          <div>
            <strong>${escapeHtml(r.concept_name)}</strong> ➔ <span style="color:var(--right-accent)">${escapeHtml(r.target_leaf)}</span>
            <div style="color:var(--text-2);font-size:10px;">Clauses: ${escapeHtml(clauses)} · Source: ${escapeHtml(r.source)}</div>
          </div>
          <span class="pill">Conf: ${Number(r.confidence || 0).toFixed(2)}</span>
        </div>
      `;
    }).join("");
  }

  const willEl = document.getElementById("tree-will-list");
  if (willEl && Array.isArray(ti.will_synapses)) {
    willEl.innerHTML = ti.will_synapses.map(w => {
      const pct = Math.max(0, Math.min(100, Math.round((Number(w.weight || 0) / 1.5) * 100)));
      return `
        <div class="leaf-row" style="padding:5px 8px;">
          <span>${escapeHtml(w.edge_id)}</span>
          <div style="display:flex;align-items:center;gap:6px;">
            <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:var(--ok);"></div></div>
            <span style="width:34px;text-align:right;">${Number(w.weight || 0).toFixed(3)}</span>
          </div>
        </div>
      `;
    }).join("");
  }
}

async function sendChatMessage() {
  const inp = document.getElementById("chat-input");
  const txt = inp.value.trim();
  if (!txt || scholarBusy) return;
  const anchorEl = document.getElementById("anchor-canvas-toggle");
  const anchorToCanvas = anchorEl ? !!anchorEl.checked : false;
  scholarBusy = true;
  try {
    inp.value = "";
    await apiPost("/api/chat", { text: txt, anchor_to_canvas: anchorToCanvas });
  } finally {
    scholarBusy = false;
  }
}

async function studyCanvasPassage() {
  if (scholarBusy) return;
  const txt = document.getElementById("study-canvas-textarea").value.trim();
  if (!txt) {
    alert("Paste a passage or dilemma into the canvas first.");
    return;
  }
  scholarBusy = true;
  try {
    canvasDirty = false;
    await apiPost("/api/canvas/study", { passage_text: txt });
  } finally {
    scholarBusy = false;
  }
}

async function saveStudyCanvas() {
  const txt = document.getElementById("study-canvas-textarea").value;
  canvasDirty = false;
  await apiPost("/api/canvas/update", { study_text: txt });
}

async function stepCurriculumLesson() {
  if (scholarBusy) return;
  scholarBusy = true;
  try {
    const lessonId = document.getElementById("lesson-select").value;
    const sleepEl = document.getElementById("sleep-after-essay-toggle");
    const sleepAfterEssay = sleepEl ? !!sleepEl.checked : true;
    await apiPost("/api/curriculum/step", { lesson_id: lessonId, sleep_after_essay: sleepAfterEssay });
  } finally {
    scholarBusy = false;
  }
}

function toggleCurriculumLoop() {
  const btn = document.getElementById("loop-curriculum-btn");
  if (curriculumLoopTimer) {
    clearInterval(curriculumLoopTimer);
    curriculumLoopTimer = null;
    btn.classList.remove("btn-loop-active");
    btn.textContent = "Run Seminar Loop";
    return;
  }
  btn.classList.add("btn-loop-active");
  btn.textContent = "Stop Seminar Loop";
  stepCurriculumLesson();
  curriculumLoopTimer = setInterval(() => {
    if (!scholarBusy) stepCurriculumLesson();
  }, 2400);
}

async function savorTurn(id) { await apiPost("/api/turn/savor", { scene_index: id }); }
async function coachTurn(id) {
  const note = prompt("Enter an optional Socratic coaching note (or leave blank to queue for REM Sleep):", "");
  if (note === null) return;
  await apiPost("/api/turn/coach", { scene_index: id, user_note: note });
}
async function createNewThread() { await apiPost("/api/threads/create", {}); }
async function switchThread(tid) { await apiPost("/api/threads/switch", { thread_id: tid }); }
async function switchLineage(lineage) { await apiPost("/api/lineage/switch", { lineage }); }
async function runRemSleepNow() { await apiPost("/api/sleep/run", {}); }
async function toggleAutoSleep(enabled) { await apiPost("/api/sleep/auto", { enabled: !!enabled }); }
async function graduateBrain(targetApp) {
  await apiPost("/api/graduate", { target_app: targetApp });
}

async function pollState() {
  try {
    const res = await fetch("/api/state");
    if (res.ok) renderState(await res.json());
  } catch (_e) {}
}
pollState();
setInterval(pollState, 3000);
</script>
</body>
</html>
"""
