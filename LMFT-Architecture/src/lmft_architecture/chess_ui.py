"""Single-Page HTML/CSS/JS Frontend for `LMFT-Chess.exe` (`chess_ui.py`).

Plain-English Purpose:
Provides a clean, Chess.com-inspired 3-column desktop arena for `LMFT-Chess.exe`:
1. **Left Column (Board + Eval Bar + Player Cards)**:
   - Live vertical centipawn Evaluation Bar (`+0.2` / `-1.5`),
   - Top Opponent Card (`LMFT_Grandmaster · Qwen3-4B`) with AI Elo, Rules count, and GPU status,
   - 8x8 Staunton SVG Chessboard with Classic Chess.com Green (`#eeeed2` / `#769656`) and Icy Sea themes,
   - Bottom Player Card (`You · Challenger`) with your assigned live Elo, move accuracy %, and turn badge,
   - Automatic AI reply when `Play vs. AI & Chat` mode is active.
2. **Center Column (2-Button Control Hub + Move List + Chess.com Game Review Tally)**:
   - Two unmistakable primary mode buttons:
     * `Play vs. AI & Chat` (you move a piece, the AI automatically replies and coaches you)
     * `Self-Train (AFK Oven)` (hands-free self-play sparring + automatic REM Sleep rule evolution)
   - Scrollable PGN move list, live Game Review badge tally (`!!`, `!`, `✓`, `?`, `??`), REM Sleep trigger, and 6-Piece / 6-Color Fuzzy Tree bars.
3. **Right Column (Live Kibitz, Custom Coach Cards & Chat Stream)** + **`? New User Guide` Modal**.
"""

from __future__ import annotations

CHESS_SINGLE_PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>LMFT-Chess — Dual-Hemisphere Chess Wisdom &amp; Self-Kibitzer Arena</title>
<style>
  :root {
    --bg-0: #161512;
    --bg-1: #21201d;
    --bg-2: #2b2926;
    --bg-3: #363430;
    --border: #3d3b37;
    --text-0: #f8fafc;
    --text-1: #d6d3cd;
    --text-2: #9e9b95;
    --chess-green: #81b64c;
    --chess-green-hover: #95bb4a;
    --chess-green-dark: #537a32;
    --accent: #38bdf8;
    --warn: #f59e0b;
    --danger: #ef4444;
    --brilliant: #1baca6;
    --great: #5c8bb0;
    --sq-light: #eeeed2;
    --sq-dark: #769656;
    --sq-hl: rgba(255, 255, 51, 0.48);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    background: var(--bg-0);
    color: var(--text-0);
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 16px;
    background: var(--bg-1);
    border-bottom: 1px solid var(--border);
    gap: 12px;
    flex-shrink: 0;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 800;
    font-size: 15px;
    white-space: nowrap;
  }
  .brand-pawn {
    width: 24px;
    height: 24px;
    border-radius: 6px;
    background: var(--chess-green);
    color: #fff;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 15px;
    font-weight: 900;
    box-shadow: 0 2px 0 var(--chess-green-dark);
  }
  .header-controls {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: nowrap;
  }
  .pill {
    padding: 4px 9px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 700;
    background: var(--bg-2);
    border: 1px solid var(--border);
    color: var(--text-1);
    white-space: nowrap;
  }
  .btn {
    padding: 6px 12px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--bg-2);
    color: var(--text-0);
    font-size: 12px;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.14s ease;
    white-space: nowrap;
  }
  .btn:hover { background: var(--bg-3); }
  .btn-chess-green {
    background: var(--chess-green);
    border-color: var(--chess-green-dark);
    color: #ffffff;
    box-shadow: 0 2px 0 var(--chess-green-dark);
  }
  .btn-chess-green:hover { background: var(--chess-green-hover); }
  .btn-mode-big {
    padding: 10px 12px;
    font-size: 13px;
    font-weight: 800;
    border-radius: 8px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
    flex: 1;
    text-align: center;
  }
  .btn-mode-big small {
    font-size: 10px;
    font-weight: 600;
    opacity: 0.88;
  }
  .btn-mode-active {
    background: var(--chess-green) !important;
    border-color: var(--chess-green-dark) !important;
    color: #fff !important;
    box-shadow: 0 3px 0 var(--chess-green-dark);
  }
  .btn-oven-active {
    background: #d97706 !important;
    border-color: #92400e !important;
    color: #fff !important;
    box-shadow: 0 3px 0 #92400e;
  }
  .btn-tutorial {
    background: rgba(56, 189, 248, 0.16);
    border-color: #0284c7;
    color: #7dd3fc;
  }
  .btn-tutorial:hover { background: rgba(56, 189, 248, 0.26); }

  .arena-grid {
    flex: 1;
    display: grid;
    grid-template-columns: 468px 315px 1fr;
    min-height: 0;
    overflow: hidden;
  }

  /* Left Column: Player Banners + Eval Bar + 8x8 Chessboard */
  .board-column {
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 10px 14px;
    gap: 8px;
    background: var(--bg-0);
    border-right: 1px solid var(--border);
    min-height: 0;
  }
  .player-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--bg-1);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 7px 10px;
    gap: 8px;
  }
  .player-id {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    font-weight: 800;
  }
  .avatar-box {
    width: 28px;
    height: 28px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 15px;
    background: var(--bg-3);
    border: 1px solid var(--border);
  }
  .board-and-eval-row {
    display: flex;
    align-items: stretch;
    gap: 8px;
    justify-content: center;
  }
  .eval-bar-outer {
    width: 24px;
    height: 404px;
    border-radius: 5px;
    background: #262421;
    border: 1px solid var(--border);
    position: relative;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    flex-shrink: 0;
  }
  .eval-bar-white {
    width: 100%;
    height: 50%;
    background: #f1f5f9;
    transition: height 0.25s ease;
  }
  .eval-bar-score {
    position: absolute;
    left: 0;
    right: 0;
    bottom: 4px;
    text-align: center;
    font-size: 9px;
    font-weight: 800;
    color: #1e293b;
    mix-blend-mode: difference;
    color: #fff;
    pointer-events: none;
  }
  .board-frame {
    width: 404px;
    height: 404px;
    display: grid;
    grid-template-columns: repeat(8, 1fr);
    grid-template-rows: repeat(8, 1fr);
    border: 3px solid #312e2b;
    border-radius: 6px;
    overflow: hidden;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
    user-select: none;
    flex-shrink: 0;
  }
  .board-frame.theme-icy {
    --sq-light: #dce6f2;
    --sq-dark: #4b7399;
  }
  .sq {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
  }
  . chess-piece-svg, .chess-piece-svg {
    width: 96%;
    height: 96%;
    pointer-events: none;
  }
  .sq.light { background: var(--sq-light); color: #312e2b; }
  .sq.dark { background: var(--sq-dark); color: #f8fafc; }
  .sq.selected, .sq.last-move { box-shadow: inset 0 0 0 999px var(--sq-hl); }
  .sq.legal-dest::after {
    content: "";
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: rgba(0, 0, 0, 0.24);
    position: absolute;
  }
  .coord-rank {
    position: absolute;
    top: 2px;
    left: 4px;
    font-size: 9px;
    font-weight: 800;
    opacity: 0.82;
  }
  .coord-file {
    position: absolute;
    bottom: 2px;
    right: 4px;
    font-size: 9px;
    font-weight: 800;
    opacity: 0.82;
  }
  #board-reflex-strip {
    background: var(--bg-1);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 11px;
    color: var(--text-1);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  /* Center Column: 2-Button Control Hub + Move Table + Game Review Box */
  .center-column {
    display: flex;
    flex-direction: column;
    background: var(--bg-1);
    border-right: 1px solid var(--border);
    padding: 10px;
    gap: 8px;
    min-height: 0;
    overflow-y: auto;
  }
  .review-tally-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 4px;
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 4px;
    text-align: center;
    font-size: 10px;
    font-weight: 700;
  }
  .tally-cell span {
    display: block;
    font-size: 13px;
    font-weight: 900;
  }
  .thread-item {
    padding: 6px 8px;
    border-radius: 5px;
    background: var(--bg-2);
    border: 1px solid transparent;
    cursor: pointer;
    font-size: 11px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .thread-item.active {
    border-color: var(--chess-green);
    background: rgba(129, 182, 76, 0.14);
  }

  /* Right Column: Live Kibitz, Custom Coach Cards & Chat Stream */
  .right-column {
    display: flex;
    flex-direction: column;
    padding: 10px 12px;
    gap: 8px;
    background: var(--bg-0);
    min-height: 0;
  }
  #chat-messages {
    flex: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 9px;
    padding-right: 4px;
  }
  .msg-card {
    background: var(--bg-1);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px;
    font-size: 13px;
    line-height: 1.48;
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
  .coach-banner {
    margin-top: 7px;
    padding: 7px 10px;
    border-radius: 6px;
    font-size: 11.5px;
    line-height: 1.4;
    border-left: 4px solid var(--chess-green);
    background: rgba(129, 182, 76, 0.12);
    color: #e2e8f0;
  }
  .coach-banner.haz {
    border-left-color: var(--danger);
    background: rgba(239, 68, 68, 0.14);
  }
  .coach-banner.brill {
    border-left-color: var(--brilliant);
    background: rgba(27, 172, 166, 0.16);
  }
  input[type="text"], select {
    background: var(--bg-2);
    color: var(--text-0);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 9px;
    font-size: 12px;
  }

  /* New User Tutorial Modal */
  .modal-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.76);
    display: none;
    align-items: center;
    justify-content: center;
    z-index: 9999;
    padding: 20px;
  }
  .modal-overlay.open { display: flex; }
  .modal-card {
    background: var(--bg-1);
    border: 2px solid var(--chess-green);
    border-radius: 12px;
    max-width: 640px;
    width: 100%;
    padding: 22px 24px;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.7);
  }
  .tutorial-step {
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 11px 14px;
    margin-top: 10px;
    font-size: 13px;
    line-height: 1.5;
  }
  .tutorial-step strong { color: #a3e635; }
</style>
</head>
<body>
<header>
  <div class="brand">
    <span class="brand-pawn">♟</span>
    <span>LMFT-Chess · Dual-Hemisphere Chess.com Arena (Qwen3-4B)</span>
  </div>
  <div class="header-controls">
    <button class="btn btn-tutorial" id="new-user-tutorial-btn" onclick="openTutorialModal()">? New User Guide (How It Works)</button>
    <select id="lineage-select" onchange="switchLineage(this.value)">
      <option value="schooled" selected>Schooled (Cycle 2 · Chess Wisdom)</option>
      <option value="virgin">Virgin Baseline (Cycle 0 · Read-Only)</option>
    </select>
    <span class="pill" id="model-status-pill">Qwen3-4B</span>
    <span class="pill" id="rules-count-pill">Rules: 2</span>
    <span class="pill" id="sleep-queue-pill">Sleep Queue: 0</span>
  </div>
</header>

<div class="arena-grid" id="chess-arena-view">
  <!-- LEFT COLUMN: Player Banners + Vertical Eval Bar + 8x8 Board -->
  <section class="board-column">
    <!-- Opponent (AI) Top Banner -->
    <div class="player-banner" style="padding:5px 10px;">
      <div class="player-id" style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
        <div class="avatar-box" id="top-player-avatar">✨</div>
        <div style="overflow:hidden;text-overflow:ellipsis;">
          <span id="top-player-name">Celestriad 1.0</span> <small id="top-player-meta" style="color:var(--text-2);font-weight:600;">(⚫ Playing Black)</small>
          <span id="posture-summary-span" style="display:none;">White to move</span>
        </div>
      </div>
      <span class="pill" id="elo-status-pill" title="Celestriad 1.0 FIDE Elo vs. Strength-Calibrated Stockfish 19">Celestriad Elo: 1000.0 (0W-0L-0D)</span>
    </div>

    <!-- Eval Bar + 8x8 Chessboard -->
    <div class="board-and-eval-row">
      <div class="eval-bar-outer" id="eval-bar-container" title="Objective Position Evaluation (White vs. Black)">
        <div class="eval-bar-white" id="eval-bar-white-fill" style="height:50%;"></div>
        <div class="eval-bar-score" id="eval-bar-label">0.0</div>
      </div>
      <div class="board-frame" id="chessboard-grid"></div>
    </div>

    <!-- Human Player / Stockfish Sparring Partner Bottom Banner -->
    <div class="player-banner" style="padding:5px 10px;">
      <div class="player-id" style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
        <div class="avatar-box" id="bottom-player-avatar">👤</div>
        <div style="overflow:hidden;text-overflow:ellipsis;">
          <span id="bottom-player-name">You (⚪ Playing White)</span> <small style="color:var(--chess-green);font-weight:700;" id="turn-indicator-pill">Your Turn</small>
          <span style="display:none;" id="bottom-player-subtitle">Auto-Coached on Every Move (!! / ! / ? / ??)</span>
        </div>
      </div>
      <span class="pill" id="player-elo-pill" style="border-color:var(--chess-green);color:#d9f99d;" title="Assigned Player Elo or Stockfish Target Elo">Your Elo: 1000.0</span>
    </div>

    <div id="board-reflex-strip">Ready — Click any piece on the board to play against LMFT, or click [Self-Train (AFK Oven)].</div>

    <div style="display:flex;gap:6px;">
      <input type="text" id="manual-move-input" placeholder="Or type a move (e.g. e4, Nf3, O-O)..." style="flex:1;" onkeydown="if(event.key==='Enter')submitManualMove()" />
      <button class="btn" onclick="submitManualMove()">Play Move</button>
    </div>
  </section>

  <!-- CENTER COLUMN: The 2 Primary Buttons + Move Table + Chess.com Game Review Hub -->
  <section class="center-column">
    <div style="font-size:11px;font-weight:800;color:var(--text-2);text-transform:uppercase;letter-spacing:0.5px;">Choose Mode (2 Simple Buttons)</div>
    <div style="display:flex;gap:6px;">
      <button class="btn btn-mode-big btn-mode-active" id="mode-play-vs-ai-btn" onclick="activatePlayVsAiMode()">
        <span>🎮 1. Play &amp; Chat</span>
        <small>AI Replies &amp; Coaches You</small>
      </button>
      <button class="btn btn-mode-big" id="auto-self-play-btn" onclick="toggleAutoSelfPlay()">
        <span>🔥 2. Self-Train</span>
        <small>AFK Spar &amp; REM Sleep</small>
      </button>
    </div>

    <!-- Quick Board Controls -->
    <div style="display:flex;gap:4px;flex-wrap:wrap;">
      <button class="btn" style="flex:1;padding:5px 6px;font-size:11px;" onclick="loadPresetFen('start')">+ New Game</button>
      <button class="btn" style="flex:1;padding:5px 6px;font-size:11px;" onclick="undoChessMove()">↩ Takeback</button>
      <button class="btn" style="flex:1;padding:5px 6px;font-size:11px;" onclick="flipChessBoard()">⇅ Flip</button>
      <button class="btn" style="flex:1;padding:5px 6px;font-size:11px;" id="step-ai-move-btn" onclick="playAiMove()">⚡ AI Move</button>
    </div>

    <div style="display:flex;gap:4px;">
      <select id="preset-select" onchange="loadPresetFen(this.value)" style="flex:1;font-size:11px;">
        <option value="start">Board: Standard Starting Position</option>
        <option value="exposed_king_attack">Preset: Exposed King vs. Reckless Attack</option>
        <option value="naked_king_siege">Preset: Stripped King Under Siege</option>
        <option value="knight_outpost_chaos">Preset: Sharp Tactical Tension</option>
        <option value="rook_seventh_bastion">Preset: 7th-Rank Rook Bastion Endgame</option>
      </select>
      <select id="board-theme-select" onchange="setBoardTheme(this.value)" style="font-size:11px;">
        <option value="green" selected>Chess.com Green</option>
        <option value="icy">Icy Sea</option>
      </select>
    </div>

    <!-- PGN Move List Table -->
    <div id="pgn-move-table-scroll" style="flex:1;min-height:220px;overflow-y:auto;background:var(--bg-0);border:1px solid var(--border);border-radius:6px;padding:6px 8px;font-size:12.5px;">
      <table style="width:100%;border-collapse:collapse;">
        <tbody id="pgn-move-table-body"></tbody>
      </table>
    </div>

    <!-- Chess.com Game Review Tally & REM Sleep Button -->
    <div class="review-tally-grid" id="game-review-tally-box">
      <div class="tally-cell" style="color:#2dd4bf;"><span id="tally-brill">0</span>!! Brill</div>
      <div class="tally-cell" style="color:#60a5fa;"><span id="tally-great">0</span>! Great</div>
      <div class="tally-cell" style="color:#a3e635;"><span id="tally-solid">0</span>✓ Solid</div>
      <div class="tally-cell" style="color:#fbbf24;"><span id="tally-mistake">0</span>? Inacc</div>
      <div class="tally-cell" style="color:#f87171;"><span id="tally-blunder">0</span>?? Blund</div>
    </div>

    <button class="btn btn-chess-green" style="width:100%;padding:9px;font-size:12.5px;" onclick="runRemSleepNow()">
      🌙 Run REM Sleep &amp; Consolidate Rules
    </button>
    <label style="font-size:11px;display:flex;align-items:center;gap:6px;color:var(--text-1);">
      <input type="checkbox" id="auto-sleep-toggle" checked onchange="toggleAutoSleep(this.checked)" />
      Auto-REM Sleep on Blunder/Brilliancy Queue
    </label>

    <!-- 6-Piece / 6-Color Fuzzy Tree Harmony -->
    <div style="font-size:11px;color:var(--text-2);border-top:1px solid var(--border);padding-top:6px;">
      <strong>6-Piece / 6-Color Fuzzy Tree Posture</strong>
      <div id="color-harmony-bars" style="margin-top:5px;display:flex;flex-direction:column;gap:4px;"></div>
    </div>

    <!-- Session Threads -->
    <div style="font-size:11px;color:var(--text-2);border-top:1px solid var(--border);padding-top:6px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
        <strong>Game &amp; Study Threads</strong>
        <button class="btn" style="padding:2px 7px;font-size:10px;" onclick="createNewThread()">+ New</button>
      </div>
      <div id="thread-list-container" style="display:flex;flex-direction:column;gap:4px;max-height:72px;overflow-y:auto;"></div>
    </div>
  </section>

  <!-- RIGHT COLUMN: Live Kibitz, Custom Coach Cards & Chat Stream -->
  <section class="right-column">
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <strong style="font-size:13px;">💬 Live Kibitz, Game Review &amp; Custom Coach Stream</strong>
      <label style="font-size:11px;display:flex;align-items:center;gap:5px;color:var(--text-1);">
        <input type="checkbox" id="anchor-board-toggle" checked />
        Anchor Chat to Board Fact
      </label>
    </div>
    <div id="chat-messages"></div>
    <div style="display:flex;gap:6px;align-items:center;">
      <input type="text" id="chat-input" placeholder="Talk to Celestriad 1.0: ask why a move was a blunder, ask for a plan, or chat..." style="flex:1;" onkeydown="if(event.key==='Enter')sendChatMessage()" />
      <select id="chat-sentences-select" title="How many sentences Celestriad 1.0 speaks when replying">
        <option value="1" selected>Reply: 1 Sentence</option>
        <option value="2">Reply: 2 Sentences</option>
        <option value="3">Reply: 3 Sentences</option>
      </select>
      <button class="btn btn-chess-green" onclick="sendChatMessage()">Send</button>
    </div>
  </section>
</div>

<!-- NEW USER TUTORIAL MODAL -->
<div class="modal-overlay" id="tutorial-modal-overlay" onclick="if(event.target===this)closeTutorialModal()">
  <div class="modal-card">
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <h2 style="margin:0;font-size:18px;">♟ Welcome to LMFT-Chess — 30-Second Quick Start</h2>
      <button class="btn" onclick="closeTutorialModal()">✕</button>
    </div>
    <p style="font-size:13px;color:var(--text-1);margin:8px 0 4px 0;">
      Unlike standard chess bots that brute-force millions of moves in silence, <strong>LMFT-Chess</strong> plays on 1-step intuition using a <strong>40-Node Fuzzy Tree</strong> wired directly into <strong>Qwen3-4B</strong> at the comma (<code>&lt;|subliminal_gap|&gt;</code>)—and learns permanent rules during <strong>REM Sleep</strong>.
    </p>
    <div class="tutorial-step">
      <strong>Button #1: 🎮 1. Play &amp; Chat (Default Mode)</strong><br/>
      Just click any White piece on the board to make your move. The AI will:
      <ul style="margin:5px 0 0 18px;padding:0;">
        <li>Grade your move Chess.com-style (<code>!! Brilliant</code>, <code>! Great</code>, <code>✓ Solid</code>, <code>? Mistake</code>, or <code>?? Blunder</code>),</li>
        <li>Update your personal <strong>Your Elo</strong> rating and live <strong>Accuracy %</strong> at the bottom of the board,</li>
        <li>Automatically play its own reply move and explain its thinking in the chat window!</li>
      </ul>
    </div>
    <div class="tutorial-step">
      <strong>Button #2: 🔥 2. Self-Train (AFK Oven Mode)</strong><br/>
      Click <strong>🔥 2. Self-Train</strong> whenever you want the AI to train itself hands-free (even overnight while you sleep). It spars against a strength-calibrated Stockfish referee, flags its own blunders (<code>??</code>) and brilliancies (<code>!!</code>), and runs <strong>🌙 REM Sleep</strong> to forge permanent <code>hazard_</code> and <code>coached_</code> fuzzy rules that automatically sync to the GitHub release seed.
    </div>
    <div class="tutorial-step">
      <strong>💬 Talk or Coach Anytime</strong><br/>
      Use the chat bar on the right to talk to the AI about the board, or click <strong>Coach / Queue for Sleep</strong> under any move to tell the AI to study that exact position during its next sleep cycle.
    </div>
    <div style="margin-top:14px;display:flex;justify-content:flex-end;gap:8px;">
      <button class="btn btn-chess-green" style="padding:8px 18px;font-size:13px;" onclick="closeTutorialModal()">Got It — Let's Play!</button>
    </div>
  </div>
</div>

<script>
const STAUNTON_SVG_PIECES = {
  P: `<svg class="chess-piece-svg" viewBox="0 0 45 45"><image width="45" height="45" href="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAJYAAACWCAMAAAAL34HQAAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAA8UExURUxpcUFBQUNDQ0REREREREVFRUVFRUVFRf////j4+O/v7+Li4tPT08TExK+vr5SUlICAgGtra1VVVUZGRh1ZfCsAAAAIdFJOUwAgPF19qsrhYMPqTAAAA6lJREFUeNrtmw1y4yAMRmub2BbgP7j/XTd4mH671YRkUix1trzpAd5IQgjF/Wg0Go1Go9FoNBqNRqPRaDQaP5auv9P9KKXejDEzmv6HhMlM8R8m86GPiZxp0E7fFBNhXxd3Z1n3nEvVKhti4lhonilj1yMmeuUEhmU+pcAa4p1B1WqjbEXAHopefbyzzrACdlPLYzfBiuGS16RR9zdYEcenI3nTOYT7Qyty/lBJY7puLKwYftUIV59TSI9wZxo7+coKtqy1xBiNsFbqWCUrsj5V1ySfQ1/UIu83ZFGwwVPRivyZxUG6tI65rOW8D9LFNZ5Ni0r4s7hu0lrbS1qjqNb0ktauokUlLKIlXFtUwqG2ZE/iMy2cRNm+Vc4h+pZsl1/KwUKXF74Ty8HCnSg9QRS7AyYI6XmrmMJz3uoVptPgSlYLplPpWf5xBj1meVlGpJFXu8fLR5guZi+WwATeifJphBekYBV71R0E0ud8BjsIpec+6t66T46IFCrtAXdLjBQt9FJR+jEvt4g4bgvYCcqHarPzgytxQcDkQ1XYjHj5gJnI9oC8pyJgkgcw+DlRGiH8KhmwIYeKafG+uuxoqyKT6V0o/RGDtXtJLU9P8JlVbHDuSnM8Sh7Rknu77lTm79txFNwxu2dWyKERnOP3shWChdISCRdY/CcIEbiJPnuA/8oSAYIlHK7gGRGIXtadyYwxHp4RYhxNhgVLqlc4plVYIAmuIpjXzpcPCosb5oVVjQ4mN1bHO8SgqJUqnoh7BdXi6vBOdF+LK+rm0BPz0s7ilHPIvVLn0vw4YyXmhZWbWrACGxxwLY56lbXOzAvhGrSOYeCzcmYJSsuR8XwAPfTadLaBJuafFLkXBlOjcQqDhRb3WhQWgn1kv6Ezr1Xcq5/41xnca4eXXKy2GVrqXtiNFDYjVmO1a5gVxyFeMuexG2EFrbLX2MksApdnVkjkFhODQKiCg1VZDCvBW3dprWMRCK2yWN6hTv3Vn5m+aIXfW7aARF5jtduXrWCWAzZclsG1IFUyc+isF3T2YgLLrNdsb6aIWNE7bJij6xbWBqt3SHk09SfkgyD1DjYgjRX3bP59K5SXqR2sHVbvktpE9Vc9rN4FnyjVe9XD6n2wAxD4gPl1sI+rlUNbwYpc1SyOZw6pAtjHVVrc1tGquezt2LP+W62rqzg7OGh9t0X0Ap9NatQ8tMJMdcD9U+XbV6pE+A1aYatEVa2K/P9aU1VM+7fqRqPRaDQal/AHIBXUlyHpytIAAAAASUVORK5CYII="/></svg>`,
  R: `<svg class="chess-piece-svg" viewBox="0 0 45 45"><image width="45" height="45" href="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAJYAAACWCAMAAAAL34HQAAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAA/UExURUxpcUFBQUNDQ0REREREREVFRUVFRUVFRf////j4+O/v7+Li4tPT08XFxbS0tKCgoI+Pj3p6emNjY1NTU0ZGRi8lq24AAAAIdFJOUwAiRmyXweDwOT8tswAABJNJREFUeNrtnOFu2kgURs+dMWn2h0m6u+//hLvbFtRCwL7fimIxJMEqtmcorXyQIkV2yPGZYRgZCWZmZmZmZmZmZmZmZmZmSBg5MMMAENJdaEUzW3LOSlL7s7Uelrxntbutlh0AaMSRv+kQYHT8wxGrAHSAIVRcTwjhlOZTmy6rcZccA8yCxQjWWYSnU0B3L1DLYqxJfG4AIPwJm/bVM8VH+K9TqJ5JrNtWeWtZfKbDhQUC6bL0ygon/RbARTfwNXy+UixwDfHhGUDNbvN1s902sEwa8tfVJeiwJTSbzbevm93R5/khZtSyGmhfvm13DtAmDzvN9YRIR7uTff/ysnegtoxagLabvdI4Es61jMtaAdSCAUa72wFk1BJYQ6rir2sZ9NbyJI0DylsLI+HwlLSMPq0ncIxE9loEklp7+ssadOHs+vTkLSd/rHwtjL5/pPOjrXHCCtQyEnIIANY5wiXpkA6miyiohUPor9WnBaCcq/y6xt6s5HUjMNB7LRkWwOo3KQ3W5NQSr2mBj72XL+M5nfiqlsg55fV2EAX0a0GH3tYSBV+J4Ndp+cjdZjWuFi/Qz/t3GRtYq2IcGnPQgPxzawJWbm7ZVCsoUAubHqtALcJkrTurZRSqpTWEDFrrvFp4Bq0ATl4tQTTGETq5ACJ7LRaMIK0skfy1fAULmxQrwiq3Fi3YA8OxyJHKoCW71heohntZoLOK8KXNf8emARa2U9/0MUAgQSIYRxYRaEq8W8WPgO+bFMIOYGacIwkdcAJHYgzApza3VvLCG/8uFPgxLiRZDCSr7FqEaslYVo1TSAuqWHOORPfgiNE9jHPWbQOltNL9N9cBlwAzjPdYB6S7bSW00gTb+AUHerAPaVqV20QZyHmLvHVxEQmM0lqAcwl5j5hIlKwFDBErXOuKTZNcP0mrBtGP+yWtZTmtK/fjci5gN9Ea5KUbaTkDvAprJSQGeUlA+VoaKi6we9DC71ML3fiVuLxOyycuXCHb+tDvJcCKazlXoBtpJTT0LAHcRy1040HU8EVXlNdyhlJeC9DwPaxKz60AGu4vCEW1ahCDEdQltYyxtbDCWj5cC8pricFeXl5LYjgqqwUIxk2ukrXGa9ndaKEbTfkliBEIluW0DPAxtQRYUS0xkPJaTKnF71Cr6G5LdHhhLTGO+9TSXWnpJlr1lFp1KS0DfLQWVkxryiBiv36tavyMNzMDw7qDKY8QSBInZMW0ALdg38HsqkgSknBZsUVu8YSM0XzZF9AKIda9STind7jWrXterRCfOKEDHH8g8R4zjONYHyAVaz2flsVn0udzLoZhwUIMBukTzxxatlgCNG0jxmNVrABWe2XRsoca2O+dqYTFAljvlEPrYQn+0pKD+CHAapdBq3qGdivOMRlXIUOcY48RPjeTtewv8I3SyaMRAPZHgH81XivF2jYYeRDVY8o1XutDTbslJ4+R9cvEHUSooSUrLdRhopYBDVlpAJuu5SIr8ixaIjP6bbUoo8XkWgWYXqsuU6sepVWeWev316r4IdHIS8ijFedBvLLWav5uipmZmZmZmZlfhf8Bdgj15hIjYjgAAAAASUVORK5CYII="/></svg>`,
  N: `<svg class="chess-piece-svg" viewBox="0 0 45 45"><image width="45" height="45" href="data:image/png;base64,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"/></svg>`,
  B: `<svg class="chess-piece-svg" viewBox="0 0 45 45"><image width="45" height="45" href="data:image/png;base64,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"/></svg>`,
  Q: `<svg class="chess-piece-svg" viewBox="0 0 45 45"><image width="45" height="45" href="data:image/png;base64,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"/></svg>`,
  K: `<svg class="chess-piece-svg" viewBox="0 0 45 45"><image width="45" height="45" href="data:image/png;base64,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"/></svg>`,
  p: `<svg class="chess-piece-svg" viewBox="0 0 45 45"><image width="45" height="45" href="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAJYAAACWCAMAAAAL34HQAAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAwUExURUxpcSUlJSwsKyUlJSQkJCUlJSUlJSUlJSUlJVZTUiYmJkVCQnd0dDo5OU1KSWZjYph8/hkAAAAJdFJOUwDg+3ofVjfEo/HPvWQAAANwSURBVHja7ZsLjqswDEUHAwHnu//dvjYg3QqG0GmDjfRyVnB0Y4zj0p9Go9FoNBqNRqPRaDQajUajcVsGM45muJXSOHd2ZerNTWLqsxPo+h99erI7SFvMdIuH45QSswur5DSoFlW2COy9T7ziKIuOigeYBdIiBZx9MqpaOf+AN5Ci12gfMKJ6JdgHRqUxEKx2xKdXp1H305EVCn/WOcKAutoSs5dRCQtWe1x4ti+NsCKO8C5xzUtYiY+1nLW2F9aipWPxMe4ZVydrZay1CWEB4HJ1DeINHmEdaDnxVj/n7pDOtXrp9uDOtUi6o3a5PfC51iSvlZrWH2qLS0TUlvCTWA4LT6Js30rvaI3iXf7kDNHlZd+JoWyFd6Igczkut57hrDGcFq0wbwl3LhsLR4jp9C5xOZ2wMMw73hNhNWvdE92RFe6J8scIL0jByhrFHURgOEW3Qpq7EbOsjHghArI4Qp2wbEi8I5LaTtBMS1SeAbwCdoLiURXmeecQmPTKlFJhM+IQmHhUCGsfl3RgA6LyngtasoGNdt3WlBc2aGBGTCsmTnkVD47bvREbmCOf4DKSg/OAV+F7WnITYOAyr2/HTnCOj2dWCKsXvI2FMys8iYPkYAocQERAbkQ1ZS1nLaBB8Jb4vlYv+kPwSmct/ao19SubsOR6RXQbsEDSIj+T8Co3K9Hadwyv3RtHb5xHo4KW1r0HpUW88SrsauTbfdwWF+meYWR43eQUKZ/hStx2LtWCB3ETl1Es+MQgaseFsBL/7kWoLvn9FvnEB16q30FAC16orlnnCJ335cG011gFks9aR14KqzdD+A29dBcjI27FCEvdC1bRPzi5JIp6jadrJI3Vbo+NDYOCVy/RRSdYpeINFl7TIPOZKfsMl3i9y9IoEBUlj7BKgQmtBEdain1vdb4SJHP9Z6YZfoMYEdh48WemyOptM4LXFd2KC1EV1cJFXgZWiOoPhGu2N93nVvCaLhmvIPWpV19/u0yw+gwcY809W0Ktf4arHdeAkeEbyFqqXVnpayuOiKva0g9Wn0NVH0aDEfk7sI8T+vj1/VMca95UCUf4DVWXvYTvJr8/xa5me2DPNXAoriqzQ0JY3xaXqVnxDLRrHlrkuQ4VG+qMrd/dtLgSVFcr1AHLuBpa9fgPtLqq9O1v1Y1Go9FoNC7hH8tnsgZelOfoAAAAAElFTkSuQmCC"/></svg>`,
  r: `<svg class="chess-piece-svg" viewBox="0 0 45 45"><image width="45" height="45" href="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAJYAAACWBAMAAADOL2zRAAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAwUExURUxpcSUlJSUlJSYmJiUlJSYmJiUlJYiGhXh1dW1ramBeXVZTUk1LSkVCQjk3NyYmJt1oLkIAAAAHdFJOUwApUXmew+Gm1HRSAAAEE0lEQVR42u3aT28bVRQF8PMmCZK74ZkUljBxYU1w2KcFwTYBUdYgAduw84IPwARQ5WUd/qhLYsQXYFrKFkWGig0SycReAm1mXtMorapkbp89447TdJ7vs0fNZn4baybR0bl3ni1ZMkqlUqlUKk1C1JaWlmoS03v1Q0p8+hamtUZDh5jQC2++c/k9aNQXUh+0Fy9/8PYCbDjvknYPgKCjzS8b3/rXOxFJAMukfeKCbT4Z7ACAQw8aja997c9B1koy7htPbYDTzt9pAojSvxGcRWgKEoBLXQVU/rgElhkiPZjvd0n2L+7rCftuko4U9LDR+D4iokVWLwH8/8VfaREBJVwMJJcxnJd3FSBZWYSHV+ACCu4ga354WwIOCHUgVFCcLC0GXIDSXi5Ge/VmJAAF8Holc4JwDahgXyLVBM7h1sIglN1LpFkOtH0kCJqL/bQmv5czeBWAhISWXSbR0qLXgHLSrOy2Sy4gbHqJNOBcWiDrdY3SaHYvJFkK9ZqLjFO7AJNZnPb5N8OsDkAYUjLIHvJ98HopkTdGFk1g717m/b9KFiDBtBI3fJNOENABd0aYSbsZGQrpJax2D/kMe4miekm7Xi7GIfDM0d++QRAEXVpl9oqNvQT6euysmTFZEj0wrdFPhhE1OuTuHrdw0XiIBI7ANUexn+N6oEW0yu51BLGIjKhWqxKJKgAh8RvYXifaTlrshjQU7gVbHd1ql2gffA5RvBOEdFoY9u+6sPASmXwMK8uUikItCDUaugdLK0Tx3s6W3lkneGwQeABby7SVnYNR+bUc5JBQSFFXgcPhfKpQqJARE/QaFeIxNW0WutP0ahJGUQ9DFVaWaSlxTmXGIp3j45s4wXGHrws96+eYW2zaGTWFlJw+K5o8S+UMTUX0gpqwVwUqN6tpmSUNT1YwszIqNwu2vXLjCcI6S+VWtc/KY58FYtzj9srPV8XMqCaasUnIVbHLEsYzJy2zCHlUMVnE7GWPv6/Cegko5CHrLAPLrAqU4VA0rbIkTIRVFoy9YNfLuHtR3IxnmUU58xXdS9llNQkGlbN6bwsQ483NzTJRllkEE35WRlRrtQv1pb56vVarStibpdud3TCipwjDvS5dBN8cma2C7fwajYpiXfFEycNLYHotnWbHb7euepn1VtvvhDTwPrMVadubXp6NG6Sxmgk94D+e2c96TMn72v67N853vC/wa/SfN2rdW19vtbyr3gk/0CHnaMXpojfa7bZ/kr6zmT6OiHHIXqF/PU8/MJN2y/Ou0F2Ms0I/bvjjbX5FB+PXFfssv0RjFybo2OfpkhzzOeEgBk8P7pgsAT4Jszm67fP8SqtF9jJkFe152vZ5bpAqsleZNZVZPGl+vris6hnNaPbcZ3wflb8HKZVKlh4BcIrQAbkOku4AAAAASUVORK5CYII="/></svg>`,
  n: `<svg class="chess-piece-svg" viewBox="0 0 45 45"><image width="45" height="45" href="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAJYAAACWCAMAAAAL34HQAAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAwUExURUxpcSwsLCUlJSUlJSUlJSUlJSUlJSUlJSUlJVZTUiYmJkVCQnd0dDo5OE5LSmZjYqaWABUAAAAJdFJOUwD94G2UEkjAKYmXrTcAAAS8SURBVHja7ZztbqswDIbnfBDyhrj3f7enkkUDEm28JmHoiOfvpu7pa2NCgvZzc3Nzc3Nzc3Pz4yc7XU5qDgBgL2Y1EXA9r5mAC3oFAPnBF/OaAeRlWdK1vByA5Um8lleQsJZ4LS8CeHnyuJTXDOAhWlfysgAW0bqSlwOoaImXs5P/WytvgLRqiZdAwdn5T2sYX1riVTBu+rOGp2XVEhJhA/2B2WxeYS1xA6eci5yx/nyrVMLawZxeauROFJsMgLwI8QBOYnaq2ERiVbL6aGbsSdfg3qouFuYTskLpq/gRfomRPWVJyrWoSmApS2CDO8zsxqhKjADAzKMbK22s9IHRNPZOSGorIa2B2RPuhDFeyCusYW3/Jmu88kCvGQBvS8gEAEntNQ192Fk2a5kLeBkgb8JirMQKLF5j5sS0NnwUMlZY5UUAwrAalrD0WjGtXm5EDdNhWBSj3msaXEN6WUlY2vbyg2tIOWeinPRTNeXuZfQEpI1WEh01PKaMdn26jxWqcYXODU8S1pfwy2vqfZdu10oAzIAlTfya1D8uB31YiWtxhe4P97EKV7Q6xmXKJluV9yuwJHSbXQ5lZmk6+7MWAfCdbjukthKtShVtl6tQP0lTTUviCn2OK1hpxRqt3KOKFmVRqguLFVq2w+M9PbRWH7Vi6lfFACBqrfRaptNs0FiBpLVqWhnA3DwblFYMIKu0EgDbPBuUE4sAkE6LANc6G/RWEhertELrbNBbSVyxrpUBtMwGKaHSyjgAH7R4pzW3zQa9lZ8AZFZoJQBT29af3urH/0bLNj1UlMZ6u6mVILu2ohWVWu77FWk8suL9KGeCWMmU03S8TIim89X3n86brQj3uiWwViu0nK9+LAUnkaKplL1aw3atvLx9Os70BILzJd+k08qA+T6tx/GHZ0IhTJtdCqqsL9rXEBYAH2rR8SFrAJAqWu1pzcdX+2oVgnN22qX7OSzea4G+0zr+9tLlsP6g5iCuhtWuNeHAKyXZyD6yAlfCai5i+VOZ9x9OR/cNH8SqGlZ7y4vXXoyP981mI1bKsNoXXNLelHlTQpqPTq+Jozas9uXp7CBQTvymhBZ1K057IFotYoQd4bADdY9q7QubgrcBBeMPml2sNGF13ePy1hkcHYZ7AwBJb1W0fLcXhZ2dDi5B0hx0KudDO3orPtIKg62i1krf8eOtpITqjm/HK634WMsPsgpqqzNby/3mfFM549uxAIg1Vme21gwArCrhma0lK1dNWGe2lgWQoyasM1vLE0Aqq1NbywFIX2sRQGP6XRvWia0lYfH3WgDs+Z1VSG9bax70jmdqqqEZVcO4gZ9IDsecNB4IyDGy7CYRVNATMZTxMOg6JHwJjRkP3poPiezBG8j6zlKOUCA5SXk8Wd7y/OHmJfohL9JPBCFzLCZ6noYZgunXX06UKkZVN5YvZ3taZXFqQzJz3RYziEsfGJ3mhByX7TtFiBUeUX5vH1in1wTDKysx+ZbVL3Yp4yzHZTUjvdySAfI9Oit2hQHYDjWk2BfqUEUAOfYlA6ZHa8W+JAC+fenHsX9zzVfreNGamrUo9qZdyw3SsrfWrXVr/VIr96aH1gD+Wy1rxmDv/wV0c3Nzc3Nz05N/qiE0gNQ56owAAAAASUVORK5CYII="/></svg>`,
  b: `<svg class="chess-piece-svg" viewBox="0 0 45 45"><image width="45" height="45" href="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAJYAAACWCAMAAAAL34HQAAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAqUExURUxpcSUlJSwsLCUlJSUlJSUlJSUlJSUlJVZTUiYmJkVCQnd0dDw6OmdkZI9zXhoAAAAIdFJOUwDe+xarNIJVd1ItuwAABB5JREFUeNrtnN1y4yoQhA8DCIaf93/d40SKezGryJYYRG3R17n4qrsZRoqS/6ampqampqbaSuvhkOyi+CFl3EhQhp9Sw4AtXMiMEeZqVfLeh7QapkfxKsSY/ZfSIFyWH8ox+k2BmNncjqVKKh++uezNVG5N0EMhDWCXYaaSarXr5nYRc1jbDq7EzO72wudYUD3EzMv9WLnGYnN740GFbrG5362Caq0W3dstzcy+Ljy7+6dpKrEIVPfeiJVZZogrkV5GvNJjrDVpoAgxIsCFCEegohzz0yzSg+yAKcaYn2Yt42ym0Q9iFrZ4v1JVZllzM1X2fzNLsbmPKv9QwSzkawagKrdSTXwL1/JChTUL0P253EYVy9vQYrNg+HUPFbY/9adZIXDnMWYJkwEZgkGvF3hnLq02qgyqLUMUL/iNy/U8hAFUdYbq26yNi2y/Q5hiEWGZodvM2riU7lV3yjCrztAwVlasOh2KVVEVGdKfCz6J1x5WxId2H1ntT4bgcj0iTLE0a8Ny1WNHj3rhrsu1WQn3Ic5hUa/+EZbV0siwV4wWEVZYBjEjwz4xKkRYVWupqyV/GjEna7Mwtepq4T2vlTSLYmUWGl9NLennR7QGZlWNr6ZW/fpZ3qzjxkMYtz3Nqhvfxy687N7FcvgpqrFE7MKYRIa/HMRUY0k8cSOdDLOOD6L023rYALNesai4evqlaKvCH88HCJYKZIjCfzQfgOUEMzyFJZSirjM8vqihDZ4kZml+D4t8LYwRoWr5Fx1OU9ALlMtg/7uCtbRvfCiw9p8R0x4Wte88dvjfsdTvWEroIO4PymOs1PwoWjxKv+ojLO6NtekIS4tg+R0sqCuWA1alG7Es3ivXesEKg2CFQjtUklj+pILQSdSNsEhgyl/AwpRvK/rCyqepcCcKXNVDYWGxyaepsNgIrIHXsZzA0nwhQyzN7SfEBSo8Ygh0/jKWEXlfc45KqPEoVzhFJVQtlCudxcKMHyTFUGU4QopBMEP8uiedoBLLECnyaaqEDCV+4XOKCq9OB7Ar1GYNYFfoYBbsCieoxMzCYaSPqDCzlPQnEB9TyX/2qRDju1A9vty1BK5jKFAp3eGbkY+p2Pb4PIPeggIVTqE8FxjSU0QpUdrUlco64oeI35dyVpRoMYpPSpnFijCZfYOo0D5aczKnGFoLFHzOOe4q57y2ruA0LcG04VXfLPFj5ewD/YDppjOUOeR4SR5/VdmOysfrygFcTb66Szk2UaYvroafc0J5kz9Q9nlVAYbV6/pK+sTxp/Xkw6J61awcS6ArcDHDLoGnw/MiZiXwLH1VgZl1gzXZN1aDFdowkwDW0uQv/ZqXywyIla5jEXMQwFIXsXhILC2ERRPrXYWJ1R0rtRY1wJLQP4ulhDT/edPU1NTU1NRUe/0Pg7vnGpbEyCwAAAAASUVORK5CYII="/></svg>`,
  q: `<svg class="chess-piece-svg" viewBox="0 0 45 45"><image width="45" height="45" href="data:image/png;base64,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"/></svg>`,
  k: `<svg class="chess-piece-svg" viewBox="0 0 45 45"><image width="45" height="45" href="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAJYAAACWCAMAAAAL34HQAAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAzUExURUxpcSIiIiUlJSUlJSUlJSUlJSUlJSUlJSUlJYWDgnd0dGNgYFZTUk5MS0NAQDIxMSYmJtBHgnAAAAAJdFJOUwARMVJ8nr7f8e5A/hcAAASpSURBVHja7ZzdcqQgEEZF/lyhgfd/2l0tx68cZEDFjNni5C6l40l32+k2VrpGo9FoNH4hPKLvHsAQ3hFN6xdq2T8r46O0xhf2UVrGLjStplWvt8daEv3+i90q0gKiaT1VK+7t0DL43te0xhX7YgRf0zL2E02raTWtpvU4rdE+UcutkAXkXnxJCzgL/IMmCAvcY7T8c7TYCw0tVDpb6L4FQ2mhuOSPXb3nQs4IwXvWrfAQAr1r6bczl1NxZg0Ylzps0Upw1k3IgIpHcbHlRKGGsEWJvqtBL4cAoiuopbQAhRB41/VChX20uBwzjs/23jvnvd9eAaW12RTFNr5+AmcO8pIYk2HGO2vGCfMPa8lt5Mhu8QF454gQSHJ+ERMXQjWECbco2Q3rFYJ9w61KZHdYzlNnAybDBG2dYjMffRtO+9AsNvBTCVRzpBJSMAtuZ3SAE4gDys9aWUgliQ0gmgnY8QKbrLzBZloddyZecrJCqG7zOtZb+b1W8NKsK4dNncHcaIX6kgcLy95qhQ7GD6XQweo2aE5jV4oOIeAefEC4MELB6kYcwlVWWUjhndCBcDEE634tV3wziqk5IFi34qZwDawwhx7BulvLlWZxzmE6WITtvgCi6Wi6nsUeOUw05sSMd+ZomrOoC0sLOYwvAzzlmxJwaa3AimYHj2BFVimx/MEuURGYIzIV76AVrQ5y2g/Zum3Rx/FzkLzv5t0s4UXFNa/nik9cSfWoQfkhN/59H+T7PwMVPwYbUlrRdNSrVLxmq5cUvNw1LQstQDvBxvIaael+px1e0ArQinKYfVwDLbl3h6e05CUtfVILc8meli/SGn5OC0k8fyeiv5zTkkgicNAq6FupkpdntLC1+JQWL+3yibu+P60lcSggV9DlUQMGWoDQuI5r8ZAI1jxwFQ6ndrSJcMlzWv2Ago9zqAr3HjealJc4o8V0osej4suGZmMTaQziuBZTIdHiMdcUzYE02qQXL9PKWSFY6sBCln36U641W1EyWPi8fLicyXiVa8n0TIP1tTRcFkReQ1+glbdCsMrD5TNepVoibZWvrHxLBh5eWa28FTp8HqZzXprltDAru5QV+uCBNAb65KVYgVaftVLdIUTei3PxaWgWgvMhZzWwM4+abQIKIKE1k7A6XlhAZbzKtHzS6uyfMXSmTUx472l3z/clVqI7Qf/Ri8hmIfpoJbtT9AO8ahBbPcMLVmgN572eZQWvZ1nB61lW8KptxboqXu4pVoDD67qVr2OF6eRRVvCiSla6jhXGnEdZYcy5aoUtoBrqQnm5aJSpBtNxGonIrfjl6wX9A1aY3KuCJY3I4XWpPH4isXxVK3vYHAZrXFWY+BwSEJJoXjuFMsTvihljxjTGmDnlG08tqiZweAlZqBzAWLeKVUskU4vSeAlDPtTsEQpv4l3E+npec1mZsQ5YWi/Cl/fLgFmwnzEvtgHDg7ZL6DVWGZO85BovUSVYNELoGpOaR7iuVZatCtWoriEEX19LVPgN7WxdfAiqRmnVf1VRV5gbbH2tcLXiUVp1a75p/ZyWukmLN61SmtbPa7nK+Cpa9flvtaS+B97+t1Sj0Wg0Go3a/AV5W8+wGCGcewAAAABJRU5ErkJggg=="/></svg>`
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
let lastRenderedMoveSig = "__init__";
let selectedSquare = null;
let currentLegalMovesUci = [];
let autoSelfPlayTimer = null;
let autoSelfPlayBusy = false;
let playVsAiAutoReply = true;
let boardFlipped = false;

function openTutorialModal() {
  const el = document.getElementById("tutorial-modal-overlay");
  if (el) el.classList.add("open");
}

function closeTutorialModal() {
  const el = document.getElementById("tutorial-modal-overlay");
  if (el) el.classList.remove("open");
  try { localStorage.setItem("lmft_chess_tutorial_seen", "1"); } catch (_e) {}
}

function activatePlayVsAiMode() {
  if (autoSelfPlayTimer) {
    toggleAutoSelfPlay();
  }
  playVsAiAutoReply = true;
  const playBtn = document.getElementById("mode-play-vs-ai-btn");
  if (playBtn) playBtn.classList.add("btn-mode-active");
  const strip = document.getElementById("board-reflex-strip");
  if (strip) strip.textContent = "Play vs. Celestriad 1.0 Active — Click any piece on the board and Celestriad 1.0 will reply automatically.";
  pollState();
}

function flipChessBoard() {
  boardFlipped = !boardFlipped;
  lastRenderedBoardSig = "";
  pollState();
}

function setBoardTheme(theme) {
  const el = document.getElementById("chessboard-grid");
  if (!el) return;
  if (theme === "icy") el.classList.add("theme-icy");
  else el.classList.remove("theme-icy");
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

function renderEvalBar(evalWhiteCp) {
  const cp = Number(evalWhiteCp || 0);
  const pawns = cp / 100.0;
  const pct = Math.min(96, Math.max(4, 50 + (pawns * 8.5)));
  const fill = document.getElementById("eval-bar-white-fill");
  const label = document.getElementById("eval-bar-label");
  if (fill) fill.style.height = `${pct}%`;
  if (label) {
    const sign = pawns > 0 ? "+" : "";
    label.textContent = `${sign}${pawns.toFixed(1)}`;
  }
}

function renderBoard(boardState) {
  if (!boardState) return;
  currentLegalMovesUci = boardState.legal_moves_uci || [];
  renderEvalBar(boardState.eval_white_cp);
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

async function playHumanMoveAndAutoReply(moveStr) {
  const turnPill = document.getElementById("turn-indicator-pill");
  const snap = await apiPost("/api/chess/move", { move: moveStr });
  if (playVsAiAutoReply && !autoSelfPlayTimer && snap && snap.board && !snap.board.is_game_over) {
    if (turnPill) turnPill.textContent = "LMFT Thinking & Kibitzing...";
    setTimeout(async () => {
      try {
        await playAiMove();
      } finally {
        if (turnPill) turnPill.textContent = "Your Turn — Click a Piece to Play";
      }
    }, 180);
  }
}

async function onSquareClick(sqName) {
  if (selectedSquare) {
    const cand = selectedSquare + sqName;
    const match = currentLegalMovesUci.find(m => m === cand || m === cand + "q");
    if (match) {
      selectedSquare = null;
      await playHumanMoveAndAutoReply(match);
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
          <div style="height:100%;width:${pct}%;background:var(--chess-green);"></div>
        </div>
      </div>
    `;
  }).join("");
}

function renderState(s) {
  if (!s) return;
  document.getElementById("model-status-pill").textContent = s.left_brain_status || "Qwen3-4B";
  const isSelfTrain = !!autoSelfPlayTimer;
  const lmftColor = (s.elo && s.elo.lmft_color) || "White";
  const stockfishColor = lmftColor === "White" ? "Black" : "White";
  const bottomBoardColor = boardFlipped ? "Black" : "White";
  const topBoardColor = boardFlipped ? "White" : "Black";

  const topAvatar = document.getElementById("top-player-avatar");
  const topName = document.getElementById("top-player-name");
  const topMeta = document.getElementById("top-player-meta");
  const botAvatar = document.getElementById("bottom-player-avatar");
  const botName = document.getElementById("bottom-player-name");
  const botSubtitle = document.getElementById("bottom-player-subtitle");
  const eloEl = document.getElementById("elo-status-pill");
  const pEloEl = document.getElementById("player-elo-pill");

  const eloData = s.elo || {};
  const hpEloText = `Celestriad Elo: ${Number(eloData.lmft_elo || 1000).toFixed(1)} (${eloData.wins || 0}W-${eloData.losses || 0}L-${eloData.draws || 0}D · ${lmftColor})`;
  const oppTarget = Number(eloData.opponent_target_elo || eloData.lmft_elo || 1000).toFixed(1);
  const sfEloText = `Stockfish 19 Target Elo: ${oppTarget} (${stockfishColor} · Game #${(eloData.games_completed || 0) + 1})`;
  const pBase = Number(eloData.player_elo || 1000).toFixed(1);
  const pAcc = eloData.player_game_accuracy_pct;
  const pEst = Number(eloData.player_estimated_game_elo || eloData.player_elo || 1000).toFixed(0);
  const humanEloText = pAcc != null
    ? `Your Elo: ${pBase} · Acc: ${Number(pAcc).toFixed(0)}% (Est. ${pEst})`
    : `Your Elo: ${pBase} (${eloData.player_wins || 0}W-${eloData.player_losses || 0}L-${eloData.player_draws || 0}D)`;

  if (isSelfTrain) {
    const celestriadOnBottom = (lmftColor === bottomBoardColor);
    const topBadge = topBoardColor === "White" ? "⚪ Playing White" : "⚫ Playing Black";
    const botBadge = bottomBoardColor === "White" ? "⚪ Playing White" : "⚫ Playing Black";
    if (celestriadOnBottom) {
      if (topAvatar) topAvatar.textContent = "🐟";
      if (topName) topName.textContent = `Stockfish 19 (${topBadge})`;
      if (topMeta) topMeta.textContent = "(Strength-Calibrated Stockfish 19 Sparring Partner)";
      if (eloEl) eloEl.textContent = sfEloText;

      if (botAvatar) botAvatar.textContent = "✨";
      if (botName) botName.textContent = `Celestriad 1.0 (${botBadge})`;
      if (botSubtitle) botSubtitle.textContent = "40-Node Fuzzy Tree + Qwen3-4B (Swaps White/Black Each Game)";
      if (pEloEl) pEloEl.textContent = hpEloText;
    } else {
      if (topAvatar) topAvatar.textContent = "✨";
      if (topName) topName.textContent = `Celestriad 1.0 (${topBadge})`;
      if (topMeta) topMeta.textContent = "(40-Node Fuzzy Tree + Qwen3-4B)";
      if (eloEl) eloEl.textContent = hpEloText;

      if (botAvatar) botAvatar.textContent = "🐟";
      if (botName) botName.textContent = `Stockfish 19 (${botBadge})`;
      if (botSubtitle) botSubtitle.textContent = "Strength-Calibrated Stockfish 19 Sparring Partner (Swaps White/Black Each Game)";
      if (pEloEl) pEloEl.textContent = sfEloText;
    }
  } else {
    const humanOnBottom = !boardFlipped;
    if (humanOnBottom) {
      if (topAvatar) topAvatar.textContent = "✨";
      if (topName) topName.textContent = "Celestriad 1.0";
      if (topMeta) topMeta.textContent = "(40-Node Tree + Qwen3-4B · ⚫ Playing Black)";
      if (eloEl) eloEl.textContent = hpEloText;

      if (botAvatar) botAvatar.textContent = "👤";
      if (botName) botName.textContent = "You (⚪ Playing White)";
      if (botSubtitle) botSubtitle.textContent = "Auto-Coached on Every Move (!! / ! / ? / ??)";
      if (pEloEl) pEloEl.textContent = humanEloText;
    } else {
      if (topAvatar) topAvatar.textContent = "👤";
      if (topName) topName.textContent = "You (⚪ Playing White)";
      if (topMeta) topMeta.textContent = "(Auto-Coached on Every Move)";
      if (eloEl) eloEl.textContent = humanEloText;

      if (botAvatar) botAvatar.textContent = "✨";
      if (botName) botName.textContent = "Celestriad 1.0 (⚫ Playing Black)";
      if (botSubtitle) botSubtitle.textContent = "40-Node Fuzzy Tree + Qwen3-4B";
      if (pEloEl) pEloEl.textContent = hpEloText;
    }
  }
  document.getElementById("rules-count-pill").textContent = "Rules: " + (s.rules_count || 0);
  document.getElementById("sleep-queue-pill").textContent = "Sleep Queue: " + (s.sleep_queue_count || 0);

  if (s.board) {
    renderBoard(s.board);
    const turnPill = document.getElementById("turn-indicator-pill");
    if (turnPill && turnPill.textContent !== "LMFT Thinking & Kibitzing...") {
      if (s.board.is_game_over) {
        turnPill.textContent = "Game Over — Click New Game";
      } else if (isSelfTrain) {
        const whiteToMove = s.board.turn === "w";
        const isCelestriadTurn = (whiteToMove && lmftColor === "White") || (!whiteToMove && lmftColor === "Black");
        turnPill.textContent = isCelestriadTurn
          ? `🔥 Celestriad 1.0 (${lmftColor}) to Move`
          : `🔥 Stockfish 19 (${stockfishColor}) to Move`;
      } else {
        turnPill.textContent = s.board.turn === "w" ? "White to Move — Click a Piece" : "Black to Move — Click a Piece";
      }
    }
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
    const moveSig = sanList.join(" ");
    if (moveSig !== lastRenderedMoveSig) {
      lastRenderedMoveSig = moveSig;
      let rowsHtml = "";
      for (let i = 0; i < sanList.length; i += 2) {
        const mNum = (i / 2) + 1;
        const wLatest = (i === sanList.length - 1);
        const bLatest = (i + 1 === sanList.length - 1);
        const wStyle = wLatest
          ? "padding:3px 6px;font-weight:800;background:rgba(129,182,76,0.25);border-radius:4px;color:#fff;"
          : "padding:3px 6px;font-weight:700;";
        const bStyle = bLatest
          ? "padding:3px 6px;font-weight:800;background:rgba(129,182,76,0.25);border-radius:4px;color:#fff;"
          : "padding:3px 6px;font-weight:700;";
        rowsHtml += `<tr><td style="color:var(--text-2);width:36px;padding:3px 4px;">${mNum}.</td><td style="${wStyle}">${escapeHtml(sanList[i])}</td><td style="${bStyle}">${escapeHtml(sanList[i+1] || "")}</td></tr>`;
      }
      document.getElementById("pgn-move-table-body").innerHTML = rowsHtml;
      const pgnScrollBox = document.getElementById("pgn-move-table-scroll");
      if (pgnScrollBox) {
        pgnScrollBox.scrollTop = pgnScrollBox.scrollHeight;
      }
    }
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
  let cBrill = 0, cGreat = 0, cSolid = 0, cMistake = 0, cBlund = 0;
  for (const m of msgs) {
    const t = m.telemetry || {};
    const rev = t.player_coaching || t.game_review;
    if (rev && rev.classification) {
      const cls = String(rev.classification);
      if (cls.startsWith("BRILLIANCY")) cBrill++;
      else if (cls.startsWith("GREAT")) cGreat++;
      else if (cls.startsWith("MISTAKE")) cMistake++;
      else if (cls.startsWith("BLUNDER")) cBlund++;
      else cSolid++;
    }
  }
  const elBrill = document.getElementById("tally-brill");
  const elGreat = document.getElementById("tally-great");
  const elSolid = document.getElementById("tally-solid");
  const elMistake = document.getElementById("tally-mistake");
  const elBlund = document.getElementById("tally-blunder");
  if (elBrill) elBrill.textContent = String(cBrill);
  if (elGreat) elGreat.textContent = String(cGreat);
  if (elSolid) elSolid.textContent = String(cSolid);
  if (elMistake) elMistake.textContent = String(cMistake);
  if (elBlund) elBlund.textContent = String(cBlund);

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
      const pc = t.player_coaching;
      const gr = t.game_review;
      const bannerCls = (pc && pc.is_hazard) || (gr && gr.is_hazard)
        ? "coach-banner haz"
        : ((pc && pc.glyph) || (gr && gr.glyph) ? "coach-banner brill" : "coach-banner");
      const coachStripHtml = pc
        ? `<div class="${bannerCls}"><strong>Custom Coach (${escapeHtml(pc.classification || "SOLID")} · ${Number(pc.move_accuracy_pct || 0).toFixed(0)}% Acc · Est. ${Number(pc.estimated_performance_elo || 1000).toFixed(0)} Elo):</strong> ${escapeHtml(pc.coaching_advice || "")}</div>`
        : (gr ? `<div class="${bannerCls}">${escapeHtml(gr.annotation_summary || "")}</div>` : "");
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
          ${coachStripHtml}
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
  await playHumanMoveAndAutoReply(mv);
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
  const playBtn = document.getElementById("mode-play-vs-ai-btn");
  if (autoSelfPlayTimer) {
    clearInterval(autoSelfPlayTimer);
    autoSelfPlayTimer = null;
    btn.classList.remove("btn-oven-active");
    btn.innerHTML = `<span>🔥 2. Self-Train</span><small>AFK Spar &amp; REM Sleep</small>`;
    if (playBtn) playBtn.classList.add("btn-mode-active");
    pollState();
    return;
  }
  if (playBtn) playBtn.classList.remove("btn-mode-active");
  btn.classList.add("btn-oven-active");
  btn.innerHTML = `<span>⏹ Stop Self-Train</span><small>Oven Running...</small>`;
  pollState();
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
try {
  if (!localStorage.getItem("lmft_chess_tutorial_seen")) {
    setTimeout(openTutorialModal, 400);
  }
} catch (_e) {}
pollState();
setInterval(pollState, 3000);
</script>
</body>
</html>
"""
