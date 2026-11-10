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
    <button class="btn" onclick="runRemSleepNow()">REM Sleep Now</button>
    <button class="btn" onclick="graduateBrain('chess')" title="Export learned domain-general rules, archives &amp; Will edges to LMFT-Chess">Graduate ➔ Chess</button>
    <button class="btn" onclick="graduateBrain('bloodmoon')" title="Export learned domain-general rules, archives &amp; Will edges to LMFT-Bloodmoon">Graduate ➔ Bloodmoon</button>
  </div>
</header>

<section class="curriculum-banner" id="curriculum-evolution-bar">
  <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
    <strong style="font-size:12px;">Foundational 4-Department Curriculum:</strong>
    <select id="lesson-select" style="min-width:290px;"></select>
    <span class="pill" id="lesson-step-pill">Steps: 0</span>
  </div>
  <div class="curriculum-controls">
    <span class="pill" title="Left Brain stops naturally when Right-Brain equilibrium settles">Natural Equilibrium Flow</span>
    <button class="btn btn-primary" id="step-curriculum-btn" onclick="stepCurriculumLesson()">Step Curriculum Lesson</button>
    <button class="btn" id="loop-curriculum-btn" onclick="toggleCurriculumLoop()">Run Curriculum Loop</button>
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
      <input type="text" id="chat-input" placeholder="Ask a Socratic question, pose a dilemma, or test a concept..." onkeydown="if(event.key==='Enter')sendChatMessage()" />
      <button class="btn btn-primary" onclick="sendChatMessage()">Send</button>
    </div>
  </section>
</div>

<script>
let lastRenderedChatSig = "";
let curriculumLoopTimer = null;
let scholarBusy = false;
let canvasDirty = false;
let lastServerCanvasText = null;
let populatedLessons = false;

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
      `<div><strong>Modality:</strong> ${escapeHtml(tel.speech_mode || "Public Speech")}</div>` +
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
      const modeTag = (m.telemetry && m.telemetry.speech_mode) ? ` · ${escapeHtml(m.telemetry.speech_mode)}` : "";
      const badge = (m.telemetry && m.telemetry.active_concept)
        ? `<span class="pill">${escapeHtml(m.telemetry.active_concept)}${modeTag}</span>`
        : "";
      const splicedClauses = (m.spliced_clauses && m.spliced_clauses.length)
        ? m.spliced_clauses
        : ((m.telemetry && m.telemetry.spliced_clauses) || []);
      const traceBlock = (isComp && splicedClauses.length)
        ? `<details style="margin-top:6px;font-size:11px;color:var(--text-2);"><summary style="cursor:pointer;">Inspect Internal Monologue &amp; &lt;|subliminal_gap|&gt; Splice (${splicedClauses.length} sentence${splicedClauses.length === 1 ? '' : 's'})</summary><div style="margin-top:4px;padding:6px 8px;background:var(--bg-2);border-radius:6px;white-space:pre-wrap;">${splicedClauses.map(c => escapeHtml(c)).join("\n")}</div></details>`
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
          ${traceBlock}
          ${actions}
          ${m.auto_rem_sleep_receipt ? `<div style="margin-top:8px;padding:6px 8px;border-radius:6px;background:rgba(168,85,247,0.12);border:1px solid rgba(168,85,247,0.3);font-size:11px;">${escapeHtml(m.auto_rem_sleep_receipt)}</div>` : ""}
        </div>
      `;
    }).join("");
    box.scrollTop = box.scrollHeight;
  }
}

async function sendChatMessage() {
  const inp = document.getElementById("chat-input");
  const txt = inp.value.trim();
  if (!txt || scholarBusy) return;
  scholarBusy = true;
  try {
    inp.value = "";
    await apiPost("/api/chat", { text: txt });
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
    await apiPost("/api/curriculum/step", { lesson_id: lessonId });
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
    btn.textContent = "Run Curriculum Loop";
    return;
  }
  btn.classList.add("btn-loop-active");
  btn.textContent = "Stop Curriculum Loop";
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
