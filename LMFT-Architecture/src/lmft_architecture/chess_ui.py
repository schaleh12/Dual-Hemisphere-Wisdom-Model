"""Standalone UI for `LMFT-Chess.exe` — Chess Wisdom & Self-Kibitzer App (`chess_ui.py`).

Plain-English Purpose:
Provides the dedicated dark-minimalist single-page interface for the `Qwen3-4B` Chess
Wisdom & Self-Kibitzer application, focused on simultaneous integration in an objective
harmony between chess thinking (`Theory`) and chess playing (`Board Fact`) across one
continuous `Pretext -> Subtext -> Context` stream.
"""

from __future__ import annotations

CHESS_SINGLE_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LMFT-Chess · Wisdom &amp; Self-Kibitzer Arena</title>
<style>
  :root {
    --bg-0: #0a0c10;
    --bg-1: #12161f;
    --bg-2: #191f2c;
    --bg-3: #222a3a;
    --border: #2b3548;
    --text-0: #f1f5f9;
    --text-1: #cbd5e1;
    --text-2: #94a3b8;
    --accent: #10b981;
    --accent-2: #38bdf8;
    --warn: #f59e0b;
    --danger: #ef4444;
    --sq-light: #dce3ec;
    --sq-dark: #4b6484;
    --sq-hl: rgba(250, 204, 21, 0.45);
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
    height: 52px;
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
    white-space: nowrap;
  }
  .btn:hover { background: var(--bg-3); }
  .btn-primary {
    background: var(--accent);
    border-color: var(--accent);
    color: #042f2e;
  }
  .btn-active-loop {
    background: var(--accent-2);
    border-color: var(--accent-2);
    color: #082f49;
  }
  #manifesto-banner {
    background: var(--bg-1);
    border-bottom: 1px solid var(--border);
    padding: 6px 16px;
    font-size: 11px;
    color: var(--text-2);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-shrink: 0;
  }
  .arena-layout {
    flex: 1;
    display: grid;
    grid-template-columns: 210px 470px 1fr;
    min-height: 0;
  }
  .sidebar {
    background: var(--bg-1);
    border-right: 1px solid var(--border);
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    overflow-y: auto;
  }
  .thread-item {
    padding: 7px 10px;
    border-radius: 6px;
    background: var(--bg-2);
    border: 1px solid transparent;
    cursor: pointer;
    font-size: 12px;
    display: flex;
    justify-content: space-between;
  }
  .thread-item.active { border-color: var(--accent); background: var(--bg-3); }
  .left-pane {
    background: var(--bg-1);
    border-right: 1px solid var(--border);
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    overflow-y: auto;
    overflow-x: hidden;
  }
  .board-frame {
    width: 400px;
    height: 400px;
    margin: 0 auto;
    display: grid;
    grid-template-columns: repeat(8, 1fr);
    grid-template-rows: repeat(8, 1fr);
    border: 2px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    user-select: none;
  }
  .sq {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
  }
  .sq.light { background: var(--sq-light); color: #1e293b; }
  .sq.dark { background: var(--sq-dark); color: #f8fafc; }
  .sq.selected, .sq.last-move { box-shadow: inset 0 0 0 999px var(--sq-hl); }
  .sq.legal-dest::after {
    content: "";
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: rgba(16, 185, 129, 0.72);
    position: absolute;
  }
  .coord-rank {
    position: absolute;
    top: 2px;
    left: 4px;
    font-size: 9px;
    font-weight: 700;
    opacity: 0.78;
  }
  .coord-file {
    position: absolute;
    bottom: 2px;
    right: 4px;
    font-size: 9px;
    font-weight: 700;
    opacity: 0.78;
  }
  #board-reflex-strip {
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 7px 10px;
    font-size: 11px;
    color: var(--text-1);
  }
  .right-pane {
    display: flex;
    flex-direction: column;
    padding: 12px;
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
  .msg-header {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    font-weight: 700;
    color: var(--text-2);
    margin-bottom: 4px;
  }
  .msg-text { white-space: pre-wrap; }
  .subtext-strip {
    margin-top: 6px;
    padding-top: 6px;
    border-top: 1px dashed var(--border);
    font-size: 11px;
    color: var(--text-2);
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 6px;
  }
  input[type="text"], select {
    background: var(--bg-1);
    color: var(--text-0);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
  }
</style>
</head>
<body>
<header>
  <div class="brand">
    <span class="brand-dot"></span>
    <span>LMFT-Chess · Simultaneous Integration &amp; Self-Kibitzer Arena (Qwen3-4B)</span>
  </div>
  <div class="status-strip">
    <select id="lineage-select" onchange="switchLineage(this.value)">
      <option value="schooled" selected>Schooled (Cycle 2 · Chess Wisdom)</option>
      <option value="virgin">Virgin Baseline (Cycle 0 · Read-Only)</option>
    </select>
    <span class="pill" id="model-status-pill">Qwen3-4B</span>
    <span class="pill" id="rules-count-pill">Rules: 2</span>
    <span class="pill" id="sleep-queue-pill">Sleep Queue: 0</span>
    <button class="btn" id="step-ai-move-btn" onclick="playAiMove()">Step AI Move</button>
    <button class="btn" id="auto-self-play-btn" onclick="toggleAutoSelfPlay()">Auto Self-Play &amp; Kibitz</button>
    <button class="btn btn-primary" onclick="runRemSleepNow()">End-of-Game REM Sleep</button>
  </div>
</header>

<div id="manifesto-banner">
  <span>Objective Harmony: Board Fact (python-chess) ⟷ Subliminal Chord (&lt;|subliminal_gap|&gt;) ⟷ Perpetual Self-Kibitzing Context Stream</span>
  <span id="posture-summary-span">White to move</span>
</div>

<div class="arena-layout" id="chess-arena-view">
  <aside class="sidebar">
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <strong style="font-size:12px;">Study &amp; Game Threads</strong>
      <button class="btn" onclick="createNewThread()">+ New</button>
    </div>
    <div id="thread-list-container" style="display:flex;flex-direction:column;gap:6px;"></div>
    <hr style="border:none;border-top:1px solid var(--border);margin:6px 0;" />
    <label style="font-size:11px;display:flex;align-items:center;gap:6px;">
      <input type="checkbox" id="auto-sleep-toggle" checked onchange="toggleAutoSleep(this.checked)" />
      Auto-REM Sleep on Surprise Threshold
    </label>
    <div style="font-size:11px;color:var(--text-2);margin-top:6px;">
      <strong>6-Piece / 6-Color Harmony</strong>
      <div id="color-harmony-bars" style="margin-top:6px;display:flex;flex-direction:column;gap:5px;"></div>
    </div>
  </aside>

  <section class="left-pane">
    <div style="display:flex;gap:6px;align-items:center;justify-content:space-between;">
      <select id="preset-select" onchange="loadPresetFen(this.value)" style="flex:1;">
        <option value="start">Standard Starting Position</option>
        <option value="exposed_king_attack">Exposed King vs. Reckless Attack (Psychosis Test)</option>
        <option value="naked_king_siege">Stripped King Under Siege (Paranoia Test)</option>
        <option value="knight_outpost_chaos">Sharp Tactical Tension (Green Knight / Ice)</option>
        <option value="rook_seventh_bastion">7th-Rank Rook Bastion Endgame (Yellow Rook / Ground)</option>
      </select>
      <button class="btn" onclick="undoChessMove()">Undo</button>
      <button class="btn" onclick="loadPresetFen('start')">Reset</button>
    </div>

    <div class="board-frame" id="chessboard-grid"></div>

    <div id="board-reflex-strip">Ready — Click a piece or click [Step AI Move] / [Auto Self-Play &amp; Kibitz].</div>

    <div style="display:flex;gap:6px;">
      <input type="text" id="manual-move-input" placeholder="Enter SAN or UCI (e.g. e4, Nf3, e2e4)..." style="flex:1;" onkeydown="if(event.key==='Enter')submitManualMove()" />
      <button class="btn" onclick="submitManualMove()">Play Move</button>
    </div>

    <div style="max-height:110px;overflow-y:auto;border:1px solid var(--border);border-radius:6px;padding:6px;font-size:11px;">
      <table style="width:100%;border-collapse:collapse;">
        <tbody id="pgn-move-table-body"></tbody>
      </table>
    </div>
  </section>

  <section class="right-pane">
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <strong style="font-size:13px;">Perpetual Self-Kibitz &amp; Teaching Stream (Pretext ➔ Subtext ➔ Context)</strong>
      <label style="font-size:11px;display:flex;align-items:center;gap:5px;">
        <input type="checkbox" id="anchor-board-toggle" checked />
        Anchor Chat to Board Fact
      </label>
    </div>
    <div id="chat-messages"></div>
    <div style="display:flex;gap:8px;align-items:center;">
      <input type="text" id="chat-input" placeholder="Ask about the position, test a plan, or kibitz with Qwen3-4B..." style="flex:1;" onkeydown="if(event.key==='Enter')sendChatMessage()" />
      <select id="chat-sentences-select">
        <option value="1" selected>1 Sent</option>
        <option value="2">2 Sent</option>
        <option value="3">3 Sent</option>
      </select>
      <button class="btn btn-primary" onclick="sendChatMessage()">Send</button>
    </div>
  </section>
</div>

<script>
const STAUNTON_SVG_PIECES = {
  P: "♙", N: "♘", B: "♗", R: "♖", Q: "♕", K: "♔",
  p: "♟", n: "♞", b: "♝", r: "♜", q: "♛", k: "♚"
};
const LEAF_TO_PIECE = {
  "philosophical/being/agency": "Queen (Red/Volition)",
  "philosophical/being/existence": "King (Cyan/Wellspring)",
  "philosophical/structure/order": "Bishop (Magenta/Order)",
  "philosophical/structure/chaos": "Knight (Green/Chaos)",
  "philosophical/persistence/continuity": "Pawn (Blue/Flow)",
  "philosophical/persistence/discreteness": "Rook (Yellow/Ground)"
};
let lastRenderedChatSig = "";
let lastRenderedBoardSig = "";
let selectedSquare = null;
let currentLegalMovesUci = [];
let autoSelfPlayTimer = null;
let autoSelfPlayBusy = false;

function escapeHtml(str) {
  return String(str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function shortBranch(leafId) {
  return LEAF_TO_PIECE[leafId] || String(leafId || "").split("/").pop();
}

async function apiPost(path, payload) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {})
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Request failed");
  renderState(data);
  return data;
}

function parseFenBoard(fen) {
  const rows = String(fen || "").split(" ")[0].split("/");
  const grid = [];
  for (let r = 0; r < 8; r++) {
    const row = [];
    for (const ch of (rows[r] || "")) {
      if (ch >= "1" && ch <= "8") {
        for (let k = 0; k < Number(ch); k++) row.push("");
      } else {
        row.push(ch);
      }
    }
    grid.push(row);
  }
  return grid;
}

function renderBoard(boardState) {
  if (!boardState) return;
  currentLegalMovesUci = boardState.legal_moves_uci || [];
  const sig = JSON.stringify([boardState.fen, selectedSquare, boardState.last_move_uci]);
  if (sig === lastRenderedBoardSig) return;
  lastRenderedBoardSig = sig;

  const grid = parseFenBoard(boardState.fen);
  const files = ["a", "b", "c", "d", "e", "f", "g", "h"];
  const lastUci = boardState.last_move_uci || "";
  const legalDests = new Set(
    selectedSquare
      ? currentLegalMovesUci.filter(m => m.startsWith(selectedSquare)).map(m => m.slice(2, 4))
      : []
  );

  let html = "";
  for (let r = 0; r < 8; r++) {
    const rankNum = 8 - r;
    for (let c = 0; c < 8; c++) {
      const sqName = files[c] + rankNum;
      const isLight = (r + c) % 2 === 0;
      const piece = (grid[r] && grid[r][c]) || "";
      const isSel = selectedSquare === sqName;
      const isLast = lastUci && (lastUci.slice(0, 2) === sqName || lastUci.slice(2, 4) === sqName);
      const isDest = legalDests.has(sqName);
      const cls = [
        "sq",
        isLight ? "light" : "dark",
        isSel ? "selected" : "",
        isLast ? "last-move" : "",
        isDest ? "legal-dest" : ""
      ].filter(Boolean).join(" ");
      const rankLabel = c === 0 ? `<span class="coord-rank">${rankNum}</span>` : "";
      const fileLabel = r === 7 ? `<span class="coord-file">${files[c]}</span>` : "";
      const glyph = piece ? `<span style="font-size:32px;line-height:1;">${STAUNTON_SVG_PIECES[piece] || piece}</span>` : "";
      html += `<div class="${cls}" onclick="onSquareClick('${sqName}')">${rankLabel}${glyph}${fileLabel}</div>`;
    }
  }
  document.getElementById("chessboard-grid").innerHTML = html;
}

async function onSquareClick(sqName) {
  if (selectedSquare) {
    const cand = selectedSquare + sqName;
    const match = currentLegalMovesUci.find(m => m === cand || m === cand + "q");
    if (match) {
      selectedSquare = null;
      await apiPost("/api/chess/move", { move: match });
      return;
    }
  }
  const hasMoves = currentLegalMovesUci.some(m => m.startsWith(sqName));
  selectedSquare = hasMoves ? sqName : null;
  lastRenderedBoardSig = "";
  pollState();
}

function renderColorBars(posture) {
  if (!posture || !posture.philosophical_coordinates) return;
  const coords = posture.philosophical_coordinates;
  const box = document.getElementById("color-harmony-bars");
  box.innerHTML = Object.keys(LEAF_TO_PIECE).map(k => {
    const val = Number(coords[k] || 0);
    const pct = Math.round(val * 100);
    return `
      <div>
        <div style="display:flex;justify-content:space-between;white-space:nowrap;">
          <span>${escapeHtml(shortBranch(k))}</span>
          <span>${val.toFixed(2)}</span>
        </div>
        <div style="height:4px;background:var(--bg-3);border-radius:3px;overflow:hidden;">
          <div style="height:100%;width:${pct}%;background:var(--accent);"></div>
        </div>
      </div>
    `;
  }).join("");
}

function renderState(s) {
  if (!s) return;
  document.getElementById("model-status-pill").textContent = s.left_brain_status || "Qwen3-4B";
  document.getElementById("rules-count-pill").textContent = "Rules: " + (s.rules_count || 0);
  document.getElementById("sleep-queue-pill").textContent = "Sleep Queue: " + (s.sleep_queue_count || 0);

  if (s.board) {
    renderBoard(s.board);
    if (s.board.posture) {
      document.getElementById("posture-summary-span").textContent = s.board.posture.board_summary || "";
      renderColorBars(s.board.posture);
    }
    const ref = s.board.last_board_reflex;
    if (ref) {
      const gearBadge = ref.gear_used === "GEAR_3_UNRESOLVED_QUEUED_FOR_SLEEP"
        ? "GEAR 3 · QUEUED FOR SLEEP"
        : ref.gear_used;
      document.getElementById("board-reflex-strip").textContent =
        `Last Move: ${ref.move_san} (${gearBadge}, Clarity=${ref.clarity}, Confidence=${ref.confidence}, ${ref.elapsed_ms} ms)`;
    }
    const sanList = s.board.move_history_san || [];
    let rowsHtml = "";
    for (let i = 0; i < sanList.length; i += 2) {
      const mNum = (i / 2) + 1;
      rowsHtml += `<tr><td style="color:var(--text-2);width:36px;">${mNum}.</td><td>${escapeHtml(sanList[i])}</td><td>${escapeHtml(sanList[i+1] || "")}</td></tr>`;
    }
    document.getElementById("pgn-move-table-body").innerHTML = rowsHtml;
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
  const sig = JSON.stringify(msgs.map(m => [m.id, m.text, m.coached, m.savored]));
  if (sig !== lastRenderedChatSig) {
    lastRenderedChatSig = sig;
    const box = document.getElementById("chat-messages");
    box.innerHTML = msgs.map(m => {
      const t = m.telemetry || {};
      const gearLabel = t.gear_used === "GEAR_3_UNRESOLVED_QUEUED_FOR_SLEEP"
        ? "GEAR 3 · QUEUED FOR SLEEP"
        : (t.gear_used || "");
      const sub = t.pretext_subtext_context;
      const subHtml = sub
        ? `<div class="subtext-strip"><span>Pretext turns: ${sub.pretext_turns || 0}</span><span>Subtext Dominant: ${escapeHtml(sub.dominant_piece || "")} (${escapeHtml(sub.dominant_color || "")})</span><span>${escapeHtml(gearLabel)}</span></div>`
        : "";
      const actionsHtml = (m.sender === "companion" && typeof m.id === "number")
        ? `<div style="margin-top:6px;display:flex;gap:6px;">
             <button class="btn" style="padding:3px 8px;font-size:10px;" onclick="coachTurn(${m.id})">${m.coached ? "Queued for Sleep" : "Coach / Queue for Sleep"}</button>
             <button class="btn" style="padding:3px 8px;font-size:10px;" onclick="savorTurn(${m.id})">${m.savored ? "Savored" : "Savor Vibe"}</button>
           </div>`
        : "";
      return `
        <div class="msg-card">
          <div class="msg-header">
            <span>${escapeHtml(m.role_label || m.sender)}</span>
            <span>#${escapeHtml(m.id)}</span>
          </div>
          <div class="msg-text">${escapeHtml(m.text)}</div>
          ${subHtml}
          ${actionsHtml}
        </div>
      `;
    }).join("");
    box.scrollTop = box.scrollHeight;
  }
}

async function submitManualMove() {
  const inp = document.getElementById("manual-move-input");
  const mv = inp.value.trim();
  if (!mv) return;
  inp.value = "";
  await apiPost("/api/chess/move", { move: mv });
}

async function playAiMove() {
  if (autoSelfPlayBusy) return;
  autoSelfPlayBusy = true;
  try {
    await apiPost("/api/chess/ai_move", {});
  } finally {
    autoSelfPlayBusy = false;
  }
}

function toggleAutoSelfPlay() {
  const btn = document.getElementById("auto-self-play-btn");
  if (autoSelfPlayTimer) {
    clearInterval(autoSelfPlayTimer);
    autoSelfPlayTimer = null;
    btn.classList.remove("btn-active-loop");
    btn.textContent = "Auto Self-Play & Kibitz";
    return;
  }
  btn.classList.add("btn-active-loop");
  btn.textContent = "Stop Self-Play Loop";
  apiPost("/api/microwave/step", {});
  autoSelfPlayTimer = setInterval(async () => {
    if (autoSelfPlayBusy) return;
    autoSelfPlayBusy = true;
    try {
      await apiPost("/api/microwave/step", {});
    } finally {
      autoSelfPlayBusy = false;
    }
  }, 2200);
}

async function undoChessMove() { await apiPost("/api/chess/undo", {}); }
async function loadPresetFen(fen) { selectedSquare = null; await apiPost("/api/chess/fen", { fen }); }
async function sendChatMessage() {
  const inp = document.getElementById("chat-input");
  const txt = inp.value.trim();
  if (!txt) return;
  inp.value = "";
  const anchor = document.getElementById("anchor-board-toggle").checked;
  const sents = Number(document.getElementById("chat-sentences-select").value || 1);
  await apiPost("/api/chat", { text: txt, anchor_to_board: anchor, num_sentences: sents });
}
async function coachTurn(sceneIndex) { await apiPost("/api/turn/coach", { scene_index: sceneIndex }); }
async function savorTurn(sceneIndex) { await apiPost("/api/turn/savor", { scene_index: sceneIndex }); }
async function createNewThread() { await apiPost("/api/threads/create", {}); }
async function switchThread(tid) { await apiPost("/api/threads/switch", { thread_id: tid }); }
async function switchLineage(lin) { await apiPost("/api/lineage", { lineage: lin }); }
async function runRemSleepNow() { await apiPost("/api/sleep/run", {}); }
async function toggleAutoSleep(enabled) { await apiPost("/api/sleep/auto", { enabled: !!enabled }); }

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
