"""Standalone UI for `LMFT-Bloodmoon.exe` — Clockwise Content-Evolution Studio (`bloodmoon_ui.py`).

Plain-English Purpose:
Provides a dedicated dark-minimalist single-page interface for the `Impish_LLAMA_4B`
standalone application centered on the 3-Phase Clockwise Evolution Loop:
  1. Phase 1 — Female Script-Writer (The Producer · She/Her)
  2. Phase 2 — Male Content-Lover (The Audience & Self-Immunizer · He/Him)
  3. Phase 3 — REM Sleep (Best Sleep of Its Life)
"""

from __future__ import annotations

BLOODMOON_SINGLE_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LMFT-Bloodmoon · Clockwise Content-Evolution Studio</title>
<style>
  :root {
    --bg-0: #090a0f;
    --bg-1: #11131c;
    --bg-2: #181b28;
    --bg-3: #222638;
    --border: #2a2f45;
    --text-0: #f3f4f8;
    --text-1: #b5bad0;
    --text-2: #7e85a3;
    --accent: #e11d48;
    --accent-soft: rgba(225, 29, 72, 0.16);
    --male-accent: #38bdf8;
    --sleep-accent: #a855f7;
    --ok: #10b981;
    --warn: #f59e0b;
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
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 700;
    font-size: 15px;
    letter-spacing: 0.3px;
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
    gap: 10px;
    font-size: 12px;
    color: var(--text-1);
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
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
  }
  .btn:hover { background: var(--bg-3); border-color: var(--text-2); }
  .btn-primary {
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
  }
  .btn-primary:hover { filter: brightness(1.1); }
  .btn-loop-active {
    background: var(--ok);
    border-color: var(--ok);
    color: #052e16;
  }
  .clockwise-banner {
    background: linear-gradient(90deg, rgba(225,29,72,0.10), rgba(56,189,248,0.10), rgba(168,85,247,0.10));
    border-bottom: 1px solid var(--border);
    padding: 10px 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    flex-wrap: wrap;
    flex-shrink: 0;
  }
  .dial-steps {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .dial-node {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    border-radius: 8px;
    background: var(--bg-1);
    border: 1px solid var(--border);
    font-size: 12px;
    color: var(--text-2);
  }
  .dial-node.active-phase-1 {
    border-color: var(--accent);
    color: var(--text-0);
    box-shadow: 0 0 12px rgba(225, 29, 72, 0.28);
  }
  .dial-node.active-phase-2 {
    border-color: var(--male-accent);
    color: var(--text-0);
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.28);
  }
  .dial-node.active-phase-3 {
    border-color: var(--sleep-accent);
    color: var(--text-0);
    box-shadow: 0 0 12px rgba(168, 85, 247, 0.28);
  }
  .dial-arrow { color: var(--text-2); font-size: 13px; }
  .dial-controls {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  input[type="text"], input[type="password"], select, textarea {
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
    grid-template-columns: 230px 1fr 1.15fr;
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
  .canvas-pane {
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    padding: 14px;
    gap: 10px;
    min-height: 0;
  }
  #script-canvas-textarea {
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
  }
  .msg-text { white-space: pre-wrap; }
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
    <span>LMFT-Bloodmoon · Clockwise Content-Evolution Studio</span>
  </div>
  <div class="status-strip">
    <span class="pill" id="model-status-pill">Impish_LLAMA_4B</span>
    <span class="pill" id="cycle-count-pill">Cycles: 0</span>
    <span class="pill" id="rules-count-pill">Rules: 2</span>
    <span class="pill" id="sleep-queue-pill">Sleep Queue: 0</span>
    <div id="patreon-unlock-bar" style="display:flex;gap:6px;align-items:center;">
      <input type="password" id="patreon-passphrase-input" placeholder="Patreon Unlock Password (one-time)" style="width:205px;" onkeydown="if(event.key==='Enter')unlockPatreonBundle()" />
      <button class="btn btn-primary" onclick="unlockPatreonBundle()">Unlock Studio</button>
    </div>
    <button class="btn" onclick="runRemSleepNow()">REM Sleep Now</button>
  </div>
</header>

<section class="clockwise-banner" id="clockwise-evolution-dial">
  <div class="dial-steps">
    <div class="dial-node active-phase-1" id="dial-phase-1">
      <strong>1. Female Producer (She/Her)</strong>
      <span>Writes Script for Him</span>
    </div>
    <span class="dial-arrow">➔</span>
    <div class="dial-node" id="dial-phase-2">
      <strong>2. Male Audience &amp; Immunizer (He/Him)</strong>
      <span>Absorbs &amp; Forges Rules</span>
    </div>
    <span class="dial-arrow">➔</span>
    <div class="dial-node" id="dial-phase-3">
      <strong>3. REM Sleep</strong>
      <span>Consolidates &amp; Upgrades</span>
    </div>
    <span class="dial-arrow">↻</span>
  </div>
  <div class="dial-controls">
    <input type="text" id="clockwise-seed-input" placeholder="Optional induction theme or seed..." style="width:230px;" />
    <select id="clockwise-sentences-select" title="Sentences per script">
      <option value="1">1 Sentence</option>
      <option value="2">2 Sentences</option>
      <option value="3" selected>3 Sentences (1 Stanza)</option>
      <option value="4">4 Sentences</option>
      <option value="5">5 Sentences</option>
      <option value="6">2 Stanzas (6 Sentences)</option>
      <option value="9">3 Stanzas (9 Sentences)</option>
    </select>
    <button class="btn btn-primary" id="step-clockwise-btn" onclick="stepClockwisePhase()">Step Next Clockwise Phase</button>
    <button class="btn" id="full-cycle-btn" onclick="runFullClockwiseCycle()">Complete Full Cycle (1➔2➔3)</button>
    <button class="btn" id="loop-clockwise-btn" onclick="toggleClockwiseLoop()">Run Clockwise Loop</button>
  </div>
</section>

<div class="workspace" id="antigravity-studio-view">
  <aside class="sidebar">
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <strong style="font-size:12px;">Script Sessions</strong>
      <button class="btn" onclick="createNewThread()">+ New</button>
    </div>
    <div id="thread-list-container" style="display:flex;flex-direction:column;gap:6px;"></div>
    <hr style="border:none;border-top:1px solid var(--border);margin:6px 0;" />
    <div class="invariant-note">
      <strong>One-Way Polarity Invariant:</strong><br/>
      Female Script-Writer (She/Her) strictly produces scripts for the Male Audience (He/Him). Reverse polarity is disabled in code.
    </div>
    <label style="font-size:11px;display:flex;align-items:center;gap:6px;margin-top:6px;">
      <input type="checkbox" id="auto-sleep-toggle" checked onchange="toggleAutoSleep(this.checked)" />
      Auto-REM Sleep on Queue Threshold
    </label>
  </aside>

  <section class="canvas-pane">
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <div>
        <strong style="font-size:13px;">Female Producer Hypnosis Script Canvas</strong>
        <div class="invariant-note">Active script crafted for Phase 2 Male Audience absorption</div>
      </div>
      <div style="display:flex;gap:6px;">
        <button class="btn btn-primary" id="seed-custom-script-btn" onclick="seedWithMyScript()">Seed Cycle with My Script ➔ Phase 2</button>
        <button class="btn" onclick="saveScriptCanvas()">Save Canvas</button>
      </div>
    </div>
    <textarea id="script-canvas-textarea" oninput="markCanvasDirty()" placeholder="Paste your own baseline hypnosis script here to initiate the cycle at your standard, or let Phase 1 (Female Script-Writer) generate one..."></textarea>
    <div class="invariant-note" id="last-sparring-summary">Ready for Phase 1 (Female Script-Writer).</div>
  </section>

  <section class="stream-pane">
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <strong style="font-size:13px;">Clockwise Evolution &amp; Sanctuary Stream</strong>
      <span class="invariant-note" id="active-phase-label">Next: Phase 1 — Female Producer</span>
    </div>
    <div id="chat-messages"></div>
    <div class="chat-bar">
      <input type="text" id="chat-input" placeholder="Direct Sanctuary prompt or custom script note..." onkeydown="if(event.key==='Enter')sendChatMessage()" />
      <select id="chat-sentences-select" title="How many sentences the AI generates in its reply">
        <option value="1">Reply: 1 Sentence</option>
        <option value="2">Reply: 2 Sentences</option>
        <option value="3" selected>Reply: 3 Sentences (1 Stanza)</option>
        <option value="4">Reply: 4 Sentences</option>
        <option value="6">Reply: 2 Stanzas (6 Sentences)</option>
      </select>
      <button class="btn btn-primary" onclick="sendChatMessage()">Send</button>
    </div>
  </section>
</div>

<script>
let lastRenderedChatSig = "";
let clockwiseLoopTimer = null;
let clockwiseBusy = false;
let canvasDirty = false;
let lastServerCanvasText = null;

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
    const summaryEl = document.getElementById("last-sparring-summary");
    if (summaryEl) summaryEl.textContent = "Notice: " + errMsg;
    throw new Error(errMsg);
  }
  renderState(data);
  return data;
}

function renderState(s) {
  if (!s) return;
  document.getElementById("model-status-pill").textContent = s.left_brain_status || "Impish_LLAMA_4B";
  document.getElementById("cycle-count-pill").textContent = "Cycles: " + (s.clockwise_cycle_count || 0);
  document.getElementById("rules-count-pill").textContent = "Rules: " + (s.rules_count || 0);
  document.getElementById("sleep-queue-pill").textContent = "Sleep Queue: " + (s.sleep_queue_count || 0);

  const unlocked = Boolean(s.patreon_unlocked);
  const unlockBar = document.getElementById("patreon-unlock-bar");
  if (unlockBar) unlockBar.style.display = unlocked ? "none" : "flex";
  for (const btnId of ["step-clockwise-btn", "full-cycle-btn", "loop-clockwise-btn", "seed-custom-script-btn"]) {
    const btnEl = document.getElementById(btnId);
    if (btnEl) btnEl.disabled = !unlocked;
  }

  const phase = Number(s.clockwise_phase || 1);
  for (let i = 1; i <= 3; i++) {
    const el = document.getElementById("dial-phase-" + i);
    if (!el) continue;
    el.className = "dial-node" + (i === phase ? (" active-phase-" + i) : "");
  }
  const phaseNames = {
    1: "Next: Phase 1 — Female Script-Writer (Producer)",
    2: "Next: Phase 2 — Male Content-Lover (Audience & Immunizer)",
    3: "Next: Phase 3 — REM Sleep (Best Sleep of Its Life)"
  };
  document.getElementById("active-phase-label").textContent = unlocked
    ? (phaseNames[phase] || phaseNames[1])
    : "LOCKED — Enter Patreon Unlock Password once in the top bar";

  const canvasEl = document.getElementById("script-canvas-textarea");
  if (typeof s.script_canvas_text === "string") {
    const serverChanged = s.script_canvas_text !== lastServerCanvasText;
    lastServerCanvasText = s.script_canvas_text;
    if (serverChanged || (!canvasDirty && document.activeElement !== canvasEl)) {
      canvasEl.value = s.script_canvas_text;
      if (serverChanged) canvasDirty = false;
    }
  }

  if (s.last_sparring_result) {
    const r = s.last_sparring_result;
    const pot = r.induction_potency || {};
    const res = r.lucid_resistance || {};
    const bpInfo = s.master_cadence_blueprint
      ? ` | Master Wave: ${s.master_cadence_blueprint.sentence_count}-Sent Blueprint Active`
      : "";
    document.getElementById("last-sparring-summary").textContent =
      `Last Verdict: ${r.verdict || "PENDING"} | Potency Entrainment=${Number(pot.entrainment || 0).toFixed(2)}, Pull=${Number(pot.subservience_pull || 0).toFixed(2)} | Male Needlessness=${Number(res.needlessness || 0).toFixed(2)}${bpInfo}`;
  } else if (s.master_cadence_blueprint) {
    document.getElementById("last-sparring-summary").textContent =
      `Master Cadence Wave Active: ${s.master_cadence_blueprint.sentence_count}-sentence <|subliminal_gap|> chord wave extracted and ready.`;
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
  const sig = JSON.stringify(msgs.map(m => [m.id, m.text, m.coached, m.savored, m.auto_rem_sleep_receipt]));
  if (sig !== lastRenderedChatSig) {
    lastRenderedChatSig = sig;
    const box = document.getElementById("chat-messages");
    box.innerHTML = msgs.map(m => `
      <div class="msg-card">
        <div class="msg-role">
          <span>${escapeHtml(m.role_label || m.sender)}</span>
          <span>#${escapeHtml(m.id)}</span>
        </div>
        <div class="msg-text">${escapeHtml(m.text)}</div>
        ${m.auto_rem_sleep_receipt ? `<div style="margin-top:8px;padding:6px 8px;border-radius:6px;background:rgba(167,139,250,0.12);border:1px solid rgba(167,139,250,0.3);font-size:11px;color:var(--phase3);">${escapeHtml(m.auto_rem_sleep_receipt)}</div>` : ""}
      </div>
    `).join("");
    box.scrollTop = box.scrollHeight;
  }
}

async function seedWithMyScript() {
  if (clockwiseBusy) return;
  const txt = document.getElementById("script-canvas-textarea").value.trim();
  if (!txt) {
    alert("Paste your baseline hypnosis script into the canvas first.");
    return;
  }
  clockwiseBusy = true;
  try {
    canvasDirty = false;
    await apiPost("/api/clockwise/seed_script", { script_text: txt });
  } finally {
    clockwiseBusy = false;
  }
}

async function stepClockwisePhase() {
  if (clockwiseBusy) return;
  clockwiseBusy = true;
  try {
    const seed = document.getElementById("clockwise-seed-input").value;
    const sents = Number(document.getElementById("clockwise-sentences-select").value || 3);
    const customTxt = canvasDirty ? document.getElementById("script-canvas-textarea").value.trim() : "";
    canvasDirty = false;
    await apiPost("/api/clockwise/step", { seed_theme: seed, num_sentences: sents, custom_script: customTxt });
  } finally {
    clockwiseBusy = false;
  }
}

async function runFullClockwiseCycle() {
  if (clockwiseBusy) return;
  clockwiseBusy = true;
  try {
    const seed = document.getElementById("clockwise-seed-input").value;
    const sents = Number(document.getElementById("clockwise-sentences-select").value || 3);
    const customTxt = canvasDirty ? document.getElementById("script-canvas-textarea").value.trim() : "";
    canvasDirty = false;
    await apiPost("/api/clockwise/cycle", { seed_theme: seed, num_sentences: sents, custom_script: customTxt });
  } finally {
    clockwiseBusy = false;
  }
}

function toggleClockwiseLoop() {
  const btn = document.getElementById("loop-clockwise-btn");
  if (clockwiseLoopTimer) {
    clearInterval(clockwiseLoopTimer);
    clockwiseLoopTimer = null;
    btn.classList.remove("btn-loop-active");
    btn.textContent = "Run Clockwise Loop";
    return;
  }
  btn.classList.add("btn-loop-active");
  btn.textContent = "Stop Clockwise Loop";
  stepClockwisePhase();
  clockwiseLoopTimer = setInterval(() => {
    if (!clockwiseBusy) stepClockwisePhase();
  }, 2200);
}

async function sendChatMessage() {
  const inp = document.getElementById("chat-input");
  const txt = inp.value.trim();
  if (!txt) return;
  inp.value = "";
  const sents = Number(document.getElementById("chat-sentences-select").value || 3);
  await apiPost("/api/chat", { text: txt, num_sentences: sents });
}

async function saveScriptCanvas() {
  const txt = document.getElementById("script-canvas-textarea").value;
  canvasDirty = false;
  await apiPost("/api/canvas/update", { script_text: txt, append: false });
}

async function createNewThread() { await apiPost("/api/threads/create", {}); }
async function switchThread(tid) { await apiPost("/api/threads/switch", { thread_id: tid }); }
async function runRemSleepNow() { await apiPost("/api/sleep/run", {}); }
async function toggleAutoSleep(enabled) { await apiPost("/api/sleep/auto", { enabled: !!enabled }); }
async function unlockPatreonBundle() {
  const inp = document.getElementById("patreon-passphrase-input");
  const pw = inp.value;
  try {
    await apiPost("/api/patreon/unlock", { passphrase: pw });
    inp.value = "";
  } catch (err) {
    alert(err.message || "Invalid Patreon unlock password.");
  }
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
