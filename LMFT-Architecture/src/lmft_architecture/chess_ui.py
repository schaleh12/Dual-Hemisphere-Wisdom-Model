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
  .chess-piece-svg {
    width: 82%;
    height: 82%;
    pointer-events: none;
    filter: drop-shadow(0 2px 3px rgba(0, 0, 0, 0.35));
  }
  .board-frame.theme-green {
    --sq-light: #ebecd0;
    --sq-dark: #739552;
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
    <span class="pill" id="elo-status-pill" title="Objective FIDE Elo vs. Strength-Calibrated Sparring Engine">Elo: 1000.0 (0W-0L-0D)</span>
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
      <select id="board-theme-select" onchange="setBoardTheme(this.value)">
        <option value="icy" selected>Icy Sea</option>
        <option value="green">Classic Green</option>
      </select>
      <button class="btn" onclick="flipChessBoard()">Flip</button>
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
  P: `<svg class="chess-piece-svg" viewBox="0 0 45 45"><path d="M 22.5,9 C 19.5,9 17,11.5 17,14.5 C 17,16.8 18.4,18.7 20.4,19.5 C 17.5,21 15.5,24 15.5,28.5 L 29.5,28.5 C 29.5,24 27.5,21 24.6,19.5 C 26.6,18.7 28,16.8 28,14.5 C 28,11.5 25.5,9 22.5,9 z M 12.5,33 L 32.5,33 L 31,29.5 L 14,29.5 L 12.5,33 z M 11,36.5 L 34,36.5 L 34,34 L 11,34 L 11,36.5 z" fill="#ffffff" stroke="#1b2430" stroke-width="1.8"/></svg>`,
  N: `<svg class="chess-piece-svg" viewBox="0 0 45 45"><path d="M 22,10 C 32.5,11 38.5,18 38,39 L 15,39 C 15,30 25,32.5 23,18" fill="#ffffff" stroke="#1b2430" stroke-width="1.8"/><path d="M 24,18 C 24.38,20.91 18.45,25.37 16,27 C 13,29 13.18,31.34 11,31 C 9.958,30.06 12.41,27.96 11,28 C 10,28 11.19,29.23 10,30 C 9,30 5.997,31 6,26 C 6,24 12,14 12,14 C 12,14 13.89,12.1 14,10.5 C 13.27,9.506 13.5,8.5 13.5,7.5 C 14.5,6.5 16.5,10 16.5,10 L 18.5,10 C 18.5,10 19.28,8.008 21,7 C 22,7 22,10 22,10" fill="#ffffff" stroke="#1b2430" stroke-width="1.8"/></svg>`,
  B: `<svg class="chess-piece-svg" viewBox="0 0 45 45"><g fill="#ffffff" stroke="#1b2430" stroke-width="1.8"><path d="M 9,36 C 12.39,35.03 19.11,36.43 22.5,34 C 25.89,36.43 32.61,35.03 36,36 C 36,36 37.65,36.54 39,38 C 38.32,38.97 37.35,38.99 36,38.5 C 32.61,37.53 25.89,38.96 22.5,37.5 C 19.11,38.96 12.39,37.53 9,38.5 C 7.646,38.99 6.677,38.97 6,38 C 7.354,36.06 9,36 9,36 z"/><path d="M 15,32 C 17.5,34.5 27.5,34.5 30,32 C 30.5,30.5 30,30 30,30 C 30,27.5 27.5,26 27.5,26 C 33,24.5 33.5,14.5 22.5,10.5 C 11.5,14.5 12,24.5 17.5,26 C 17.5,26 15,27.5 15,30 C 15,30 14.5,30.5 15,32 z"/><circle cx="22.5" cy="8" r="2.2"/></g></svg>`,
  R: `<svg class="chess-piece-svg" viewBox="0 0 45 45"><g fill="#ffffff" stroke="#1b2430" stroke-width="1.8"><path d="M 9,39 L 36,39 L 36,36 L 9,36 L 9,39 z M 12,36 L 12,32 L 33,32 L 33,36 L 12,36 z M 11,14 L 11,9 L 15,9 L 15,11 L 20,11 L 20,9 L 25,9 L 25,11 L 30,11 L 30,9 L 34,9 L 34,14"/><path d="M 34,14 L 31,17 L 14,17 L 11,14"/><path d="M 31,17 L 31,29.5 L 14,29.5 L 14,17"/><path d="M 14,29.5 L 12,32 L 33,32 L 31,29.5"/></g></svg>`,
  Q: `<svg class="chess-piece-svg" viewBox="0 0 45 45"><g fill="#ffffff" stroke="#1b2430" stroke-width="1.8"><path d="M 9,26 C 17.5,24.5 30,24.5 36,26 L 38.5,13.5 L 31,25 L 30.7,10.9 L 25.5,24.5 L 22.5,10 L 19.5,24.5 L 14.3,10.9 L 14,25 L 6.5,13.5 L 9,26 z"/><path d="M 9,26 C 9,28 10.5,28 11.5,30 C 12.5,31.5 12.5,31 12,33.5 C 10.5,34.5 11,36 11,36 C 9.5,37.5 11,38.5 11,38.5 C 17.5,39.5 27.5,39.5 34,38.5 C 34,38.5 35.5,37.5 34,36 C 34,36 34.5,34.5 33,33.5 C 32.5,31 32.5,31.5 33.5,30 C 34.5,28 36,28 36,26 C 27.5,24.5 17.5,24.5 9,26 z"/><circle cx="6" cy="12" r="1.8"/><circle cx="14" cy="9" r="1.8"/><circle cx="22.5" cy="8" r="1.8"/><circle cx="31" cy="9" r="1.8"/><circle cx="39" cy="12" r="1.8"/></g></svg>`,
  K: `<svg class="chess-piece-svg" viewBox="0 0 45 45"><g fill="#ffffff" stroke="#1b2430" stroke-width="1.8"><path d="M 22.5,11.63 L 22.5,6 M 20,8 L 25,8"/><path d="M 22.5,25 C 22.5,25 27,17.5 25.5,14.5 C 25.5,14.5 24.5,12 22.5,12 C 20.5,12 19.5,14.5 19.5,14.5 C 18,17.5 22.5,25 22.5,25"/><path d="M 11.5,37 C 17,40.5 27,40.5 32.5,37 L 32.5,30 C 32.5,30 41.5,25.5 38.5,19.5 C 34.5,13 25,16 22.5,23.5 L 22.5,27 L 22.5,23.5 C 19,16 9.5,13 6.5,19.5 C 3.5,25.5 11.5,29.5 11.5,29.5 L 11.5,37 z"/></g></svg>`,
  p: `<svg class="chess-piece-svg" viewBox="0 0 45 45"><path d="M 22.5,9 C 19.5,9 17,11.5 17,14.5 C 17,16.8 18.4,18.7 20.4,19.5 C 17.5,21 15.5,24 15.5,28.5 L 29.5,28.5 C 29.5,24 27.5,21 24.6,19.5 C 26.6,18.7 28,16.8 28,14.5 C 28,11.5 25.5,9 22.5,9 z M 12.5,33 L 32.5,33 L 31,29.5 L 14,29.5 L 12.5,33 z M 11,36.5 L 34,36.5 L 34,34 L 11,34 L 11,36.5 z" fill="#1e242b" stroke="#e2e8f0" stroke-width="1.5"/></svg>`,
  n: `<svg class="chess-piece-svg" viewBox="0 0 45 45"><path d="M 22,10 C 32.5,11 38.5,18 38,39 L 15,39 C 15,30 25,32.5 23,18" fill="#1e242b" stroke="#e2e8f0" stroke-width="1.5"/><path d="M 24,18 C 24.38,20.91 18.45,25.37 16,27 C 13,29 13.18,31.34 11,31 C 9.958,30.06 12.41,27.96 11,28 C 10,28 11.19,29.23 10,30 C 9,30 5.997,31 6,26 C 6,24 12,14 12,14 C 12,14 13.89,12.1 14,10.5 C 13.27,9.506 13.5,8.5 13.5,7.5 C 14.5,6.5 16.5,10 16.5,10 L 18.5,10 C 18.5,10 19.28,8.008 21,7 C 22,7 22,10 22,10" fill="#1e242b" stroke="#e2e8f0" stroke-width="1.5"/></svg>`,
  b: `<svg class="chess-piece-svg" viewBox="0 0 45 45"><g fill="#1e242b" stroke="#e2e8f0" stroke-width="1.5"><path d="M 9,36 C 12.39,35.03 19.11,36.43 22.5,34 C 25.89,36.43 32.61,35.03 36,36 C 36,36 37.65,36.54 39,38 C 38.32,38.97 37.35,38.99 36,38.5 C 32.61,37.53 25.89,38.96 22.5,37.5 C 19.11,38.96 12.39,37.53 9,38.5 C 7.646,38.99 6.677,38.97 6,38 C 7.354,36.06 9,36 9,36 z"/><path d="M 15,32 C 17.5,34.5 27.5,34.5 30,32 C 30.5,30.5 30,30 30,30 C 30,27.5 27.5,26 27.5,26 C 33,24.5 33.5,14.5 22.5,10.5 C 11.5,14.5 12,24.5 17.5,26 C 17.5,26 15,27.5 15,30 C 15,30 14.5,30.5 15,32 z"/><circle cx="22.5" cy="8" r="2.2"/></g></svg>`,
  r: `<svg class="chess-piece-svg" viewBox="0 0 45 45"><g fill="#1e242b" stroke="#e2e8f0" stroke-width="1.5"><path d="M 9,39 L 36,39 L 36,36 L 9,36 L 9,39 z M 12,36 L 12,32 L 33,32 L 33,36 L 12,36 z M 11,14 L 11,9 L 15,9 L 15,11 L 20,11 L 20,9 L 25,9 L 25,11 L 30,11 L 30,9 L 34,9 L 34,14"/><path d="M 34,14 L 31,17 L 14,17 L 11,14"/><path d="M 31,17 L 31,29.5 L 14,29.5 L 14,17"/><path d="M 14,29.5 L 12,32 L 33,32 L 31,29.5"/></g></svg>`,
  q: `<svg class="chess-piece-svg" viewBox="0 0 45 45"><g fill="#1e242b" stroke="#e2e8f0" stroke-width="1.5"><path d="M 9,26 C 17.5,24.5 30,24.5 36,26 L 38.5,13.5 L 31,25 L 30.7,10.9 L 25.5,24.5 L 22.5,10 L 19.5,24.5 L 14.3,10.9 L 14,25 L 6.5,13.5 L 9,26 z"/><path d="M 9,26 C 9,28 10.5,28 11.5,30 C 12.5,31.5 12.5,31 12,33.5 C 10.5,34.5 11,36 11,36 C 9.5,37.5 11,38.5 11,38.5 C 17.5,39.5 27.5,39.5 34,38.5 C 34,38.5 35.5,37.5 34,36 C 34,36 34.5,34.5 33,33.5 C 32.5,31 32.5,31.5 33.5,30 C 34.5,28 36,28 36,26 C 27.5,24.5 17.5,24.5 9,26 z"/><circle cx="6" cy="12" r="1.8"/><circle cx="14" cy="9" r="1.8"/><circle cx="22.5" cy="8" r="1.8"/><circle cx="31" cy="9" r="1.8"/><circle cx="39" cy="12" r="1.8"/></g></svg>`,
  k: `<svg class="chess-piece-svg" viewBox="0 0 45 45"><g fill="#1e242b" stroke="#e2e8f0" stroke-width="1.5"><path d="M 22.5,11.63 L 22.5,6 M 20,8 L 25,8"/><path d="M 22.5,25 C 22.5,25 27,17.5 25.5,14.5 C 25.5,14.5 24.5,12 22.5,12 C 20.5,12 19.5,14.5 19.5,14.5 C 18,17.5 22.5,25 22.5,25"/><path d="M 11.5,37 C 17,40.5 27,40.5 32.5,37 L 32.5,30 C 32.5,30 41.5,25.5 38.5,19.5 C 34.5,13 25,16 22.5,23.5 L 22.5,27 L 22.5,23.5 C 19,16 9.5,13 6.5,19.5 C 3.5,25.5 11.5,29.5 11.5,29.5 L 11.5,37 z"/></g></svg>`
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
let boardFlipped = false;

function flipChessBoard() {
  boardFlipped = !boardFlipped;
  lastRenderedBoardSig = "";
  pollState();
}

function setBoardTheme(theme) {
  const el = document.getElementById("chessboard-grid");
  if (!el) return;
  if (theme === "green") el.classList.add("theme-green");
  else el.classList.remove("theme-green");
}

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
  if (!res.ok) {
    const errMsg = data.error || "Request failed";
    const strip = document.getElementById("board-reflex-strip");
    if (strip) strip.textContent = "Notice: " + errMsg;
    throw new Error(errMsg);
  }
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
  const sig = JSON.stringify([boardState.fen, selectedSquare, boardState.last_move_uci, boardFlipped]);
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
  for (let dr = 0; dr < 8; dr++) {
    const r = boardFlipped ? (7 - dr) : dr;
    const rankNum = 8 - r;
    for (let dc = 0; dc < 8; dc++) {
      const c = boardFlipped ? (7 - dc) : dc;
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
      const rankLabel = dc === 0 ? `<span class="coord-rank">${rankNum}</span>` : "";
      const fileLabel = dr === 7 ? `<span class="coord-file">${files[c]}</span>` : "";
      const glyph = piece ? (STAUNTON_SVG_PIECES[piece] || "") : "";
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
  if (s.elo) {
    const eloEl = document.getElementById("elo-status-pill");
    if (eloEl) {
      eloEl.textContent = `Elo: ${Number(s.elo.lmft_elo || 1000).toFixed(1)} (${s.elo.wins || 0}W-${s.elo.losses || 0}L-${s.elo.draws || 0}D · ${s.elo.lmft_color || "White"})`;
    }
  }
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

async function stepSelfPlayOnce() {
  if (autoSelfPlayBusy) return;
  autoSelfPlayBusy = true;
  try {
    await apiPost("/api/microwave/step", {});
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
  stepSelfPlayOnce();
  autoSelfPlayTimer = setInterval(stepSelfPlayOnce, 2200);
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
