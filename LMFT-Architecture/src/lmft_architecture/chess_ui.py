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
    width: 84%;
    height: 84%;
    pointer-events: none;
    filter: drop-shadow(0 1px 2px rgba(0, 0, 0, 0.32));
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
  P: `<svg class="chess-piece-svg" viewBox="0 0 45 45"><g stroke="#2b2926" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M 12,35.5 C 13.5,29.5 16.8,26.2 18.8,23.2 C 16.8,22.4 16.2,21.2 18.2,20.2 C 16.5,18.8 15.8,16.5 16.2,14.2 C 16.8,11.2 19.4,9.2 22.5,9.2 C 25.6,9.2 28.2,11.2 28.8,14.2 C 29.2,16.5 28.5,18.8 26.8,20.2 C 28.8,21.2 28.2,22.4 26.2,23.2 C 28.2,26.2 31.5,29.5 33,35.5 Z" fill="#f9f9f9"/><path d="M 24.5,9.6 C 27.2,10.8 28.5,14.2 26.2,19.8 C 28.2,21.2 26.5,22.5 25.5,23.2 C 27.8,26.5 30.2,29.8 31.5,35.2 L 33,35.2 C 31.5,29.5 28.2,26.2 26.2,23.2 C 28.2,22.4 28.8,21.2 26.8,20.2 C 28.5,18.8 29.2,16.5 28.8,14.2 C 28.3,11.5 26.5,9.9 24.5,9.6 Z" fill="#dcdcdc" stroke="none"/><rect x="10.5" y="35.2" width="24" height="4.2" rx="2.1" fill="#ececec"/></g></svg>`,
  R: `<svg class="chess-piece-svg" viewBox="0 0 45 45"><g stroke="#2b2926" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M 11.5,10.2 L 16.2,10.2 L 16.2,13.8 L 19.8,13.8 L 19.8,10.2 L 25.2,10.2 L 25.2,13.8 L 28.8,13.8 L 28.8,10.2 L 33.5,10.2 L 33.5,18.2 C 33.5,19.8 31.2,20.8 30.2,21.8 L 31.8,35.2 L 13.2,35.2 L 14.8,21.8 C 13.8,20.8 11.5,19.8 11.5,18.2 Z" fill="#f9f9f9"/><path d="M 29.2,10.8 L 33,10.8 L 33,18 C 33,19.5 30.8,20.5 29.8,21.8 L 31.2,35 L 28.2,35 L 27,21.5 C 28,20.2 29.2,19.2 29.2,17.8 Z" fill="#dcdcdc" stroke="none"/><rect x="9.5" y="35.2" width="26" height="4.2" rx="2.1" fill="#ececec"/></g></svg>`,
  N: `<svg class="chess-piece-svg" viewBox="0 0 45 45"><g stroke="#2b2926" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M 13.5,35.2 C 14.2,28.8 19.2,25.8 21.8,22.2 C 19.2,23.2 15.2,25.8 11.5,25.2 C 9.2,24.8 8.4,22.8 9.5,20.2 C 10.8,17.2 14.8,13.2 18.2,10.8 L 17.8,6.8 L 22.4,9.6 C 29.8,10.5 34.8,17.8 33.2,35.2 Z" fill="#f9f9f9"/><path d="M 23.5,10.2 C 29.5,12.2 33.8,18.8 32.5,35 L 29.2,35 C 30.2,21.2 27.2,14.2 23.5,10.2 Z" fill="#dcdcdc" stroke="none"/><rect x="9.8" y="35.2" width="25.4" height="4.2" rx="2.1" fill="#ececec"/></g></svg>`,
  B: `<svg class="chess-piece-svg" viewBox="0 0 45 45"><g stroke="#2b2926" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M 13.2,35.2 C 12.2,27.5 13.2,19.2 16.8,14.5 L 20.2,19.8 C 21.2,21.2 22.8,20.2 21.8,18.5 L 18.5,12.8 C 20.2,11.2 22.2,10.2 23.8,10.2 C 29.5,12.8 32.5,22.2 31.8,35.2 Z" fill="#f9f9f9"/><circle cx="24.2" cy="7.8" r="2.1" fill="#f9f9f9"/><path d="M 24.5,10.8 C 29.2,14.2 31.5,22.8 31,35 L 27.8,35 C 28.5,23.5 26.8,15.2 24.5,10.8 Z" fill="#dcdcdc" stroke="none"/><rect x="10" y="35.2" width="25" height="4.2" rx="2.1" fill="#ececec"/></g></svg>`,
  Q: `<svg class="chess-piece-svg" viewBox="0 0 45 45"><g stroke="#2b2926" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M 12.2,35.2 L 8.2,17.2 C 7.2,16.5 6.8,15.2 7.5,14.1 C 8.2,13 9.8,13 10.6,14.1 C 11.1,14.8 11,15.8 10.4,16.4 L 15.8,24.2 L 15.2,14.2 C 14.4,13.5 14.2,12.2 15,11.2 C 15.8,10.2 17.4,10.2 18.2,11.2 C 18.8,12 18.6,13.2 17.8,13.9 L 21.8,23.5 L 21.5,12.8 C 20.8,12 20.8,10.6 21.8,9.8 C 22.6,9 24.2,9.2 24.8,10.2 C 25.4,11 25.2,12.2 24.2,12.9 L 27.8,23.5 L 29.2,13.9 C 28.4,13.2 28.2,12 28.8,11.2 C 29.6,10.2 31.2,10.2 32,11.2 C 32.8,12.2 32.6,13.5 31.8,14.2 L 29.8,24.2 L 34.6,16.4 C 34,15.8 33.9,14.8 34.4,14.1 C 35.2,13 36.8,13 37.5,14.1 C 38.2,15.2 37.8,16.5 36.8,17.2 L 32.8,35.2 Z" fill="#f9f9f9"/><rect x="9.5" y="35.2" width="26" height="4.2" rx="2.1" fill="#ececec"/></g></svg>`,
  K: `<svg class="chess-piece-svg" viewBox="0 0 45 45"><g stroke="#2b2926" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M 20.8,6.2 L 24.2,6.2 L 24.2,8.5 L 26.8,8.5 L 26.8,11.5 L 24.2,11.5 L 24.2,14.5 L 20.8,14.5 L 20.8,11.5 L 18.2,11.5 L 18.2,8.5 L 20.8,8.5 Z" fill="#f9f9f9"/><path d="M 12.2,35.2 C 10.5,27.8 7.5,20.2 13.2,16.2 C 17.2,13.5 21,15.8 22.5,19.2 C 24,15.8 27.8,13.5 31.8,16.2 C 37.5,20.2 34.5,27.8 32.8,35.2 Z" fill="#f9f9f9"/><path d="M 14.2,20.8 C 12.5,23.2 14.5,27.5 18.8,27.2 C 19.8,23.8 17.5,19.5 14.2,20.8 Z" fill="#64748b"/><path d="M 30.8,20.8 C 32.5,23.2 30.5,27.5 26.2,27.2 C 25.2,23.8 27.5,19.5 30.8,20.8 Z" fill="#64748b"/><rect x="9.5" y="35.2" width="26" height="4.2" rx="2.1" fill="#ececec"/></g></svg>`,
  p: `<svg class="chess-piece-svg" viewBox="0 0 45 45"><g stroke="#22201e" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M 12,35.5 C 13.5,29.5 16.8,26.2 18.8,23.2 C 16.8,22.4 16.2,21.2 18.2,20.2 C 16.5,18.8 15.8,16.5 16.2,14.2 C 16.8,11.2 19.4,9.2 22.5,9.2 C 25.6,9.2 28.2,11.2 28.8,14.2 C 29.2,16.5 28.5,18.8 26.8,20.2 C 28.8,21.2 28.2,22.4 26.2,23.2 C 28.2,26.2 31.5,29.5 33,35.5 Z" fill="#575451"/><path d="M 24.5,9.6 C 27.2,10.8 28.5,14.2 26.2,19.8 C 28.2,21.2 26.5,22.5 25.5,23.2 C 27.8,26.5 30.2,29.8 31.5,35.2 L 33,35.2 C 31.5,29.5 28.2,26.2 26.2,23.2 C 28.2,22.4 28.8,21.2 26.8,20.2 C 28.5,18.8 29.2,16.5 28.8,14.2 C 28.3,11.5 26.5,9.9 24.5,9.6 Z" fill="#423f3d" stroke="none"/><rect x="10.5" y="35.2" width="24" height="4.2" rx="2.1" fill="#4a4744"/></g></svg>`,
  r: `<svg class="chess-piece-svg" viewBox="0 0 45 45"><g stroke="#22201e" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M 11.5,10.2 L 16.2,10.2 L 16.2,13.8 L 19.8,13.8 L 19.8,10.2 L 25.2,10.2 L 25.2,13.8 L 28.8,13.8 L 28.8,10.2 L 33.5,10.2 L 33.5,18.2 C 33.5,19.8 31.2,20.8 30.2,21.8 L 31.8,35.2 L 13.2,35.2 L 14.8,21.8 C 13.8,20.8 11.5,19.8 11.5,18.2 Z" fill="#575451"/><path d="M 29.2,10.8 L 33,10.8 L 33,18 C 33,19.5 30.8,20.5 29.8,21.8 L 31.2,35 L 28.2,35 L 27,21.5 C 28,20.2 29.2,19.2 29.2,17.8 Z" fill="#423f3d" stroke="none"/><rect x="9.5" y="35.2" width="26" height="4.2" rx="2.1" fill="#4a4744"/></g></svg>`,
  n: `<svg class="chess-piece-svg" viewBox="0 0 45 45"><g stroke="#22201e" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M 13.5,35.2 C 14.2,28.8 19.2,25.8 21.8,22.2 C 19.2,23.2 15.2,25.8 11.5,25.2 C 9.2,24.8 8.4,22.8 9.5,20.2 C 10.8,17.2 14.8,13.2 18.2,10.8 L 17.8,6.8 L 22.4,9.6 C 29.8,10.5 34.8,17.8 33.2,35.2 Z" fill="#575451"/><path d="M 23.5,10.2 C 29.5,12.2 33.8,18.8 32.5,35 L 29.2,35 C 30.2,21.2 27.2,14.2 23.5,10.2 Z" fill="#423f3d" stroke="none"/><rect x="9.8" y="35.2" width="25.4" height="4.2" rx="2.1" fill="#4a4744"/></g></svg>`,
  b: `<svg class="chess-piece-svg" viewBox="0 0 45 45"><g stroke="#22201e" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M 13.2,35.2 C 12.2,27.5 13.2,19.2 16.8,14.5 L 20.2,19.8 C 21.2,21.2 22.8,20.2 21.8,18.5 L 18.5,12.8 C 20.2,11.2 22.2,10.2 23.8,10.2 C 29.5,12.8 32.5,22.2 31.8,35.2 Z" fill="#575451"/><circle cx="24.2" cy="7.8" r="2.1" fill="#575451"/><path d="M 24.5,10.8 C 29.2,14.2 31.5,22.8 31,35 L 27.8,35 C 28.5,23.5 26.8,15.2 24.5,10.8 Z" fill="#423f3d" stroke="none"/><rect x="10" y="35.2" width="25" height="4.2" rx="2.1" fill="#4a4744"/></g></svg>`,
  q: `<svg class="chess-piece-svg" viewBox="0 0 45 45"><g stroke="#22201e" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M 12.2,35.2 L 8.2,17.2 C 7.2,16.5 6.8,15.2 7.5,14.1 C 8.2,13 9.8,13 10.6,14.1 C 11.1,14.8 11,15.8 10.4,16.4 L 15.8,24.2 L 15.2,14.2 C 14.4,13.5 14.2,12.2 15,11.2 C 15.8,10.2 17.4,10.2 18.2,11.2 C 18.8,12 18.6,13.2 17.8,13.9 L 21.8,23.5 L 21.5,12.8 C 20.8,12 20.8,10.6 21.8,9.8 C 22.6,9 24.2,9.2 24.8,10.2 C 25.4,11 25.2,12.2 24.2,12.9 L 27.8,23.5 L 29.2,13.9 C 28.4,13.2 28.2,12 28.8,11.2 C 29.6,10.2 31.2,10.2 32,11.2 C 32.8,12.2 32.6,13.5 31.8,14.2 L 29.8,24.2 L 34.6,16.4 C 34,15.8 33.9,14.8 34.4,14.1 C 35.2,13 36.8,13 37.5,14.1 C 38.2,15.2 37.8,16.5 36.8,17.2 L 32.8,35.2 Z" fill="#575451"/><rect x="9.5" y="35.2" width="26" height="4.2" rx="2.1" fill="#4a4744"/></g></svg>`,
  k: `<svg class="chess-piece-svg" viewBox="0 0 45 45"><g stroke="#22201e" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M 20.8,6.2 L 24.2,6.2 L 24.2,8.5 L 26.8,8.5 L 26.8,11.5 L 24.2,11.5 L 24.2,14.5 L 20.8,14.5 L 20.8,11.5 L 18.2,11.5 L 18.2,8.5 L 20.8,8.5 Z" fill="#575451"/><path d="M 12.2,35.2 C 10.5,27.8 7.5,20.2 13.2,16.2 C 17.2,13.5 21,15.8 22.5,19.2 C 24,15.8 27.8,13.5 31.8,16.2 C 37.5,20.2 34.5,27.8 32.8,35.2 Z" fill="#575451"/><path d="M 14.2,20.8 C 12.5,23.2 14.5,27.5 18.8,27.2 C 19.8,23.8 17.5,19.5 14.2,20.8 Z" fill="#3a3735"/><path d="M 30.8,20.8 C 32.5,23.2 30.5,27.5 26.2,27.2 C 25.2,23.8 27.5,19.5 30.8,20.8 Z" fill="#3a3735"/><rect x="9.5" y="35.2" width="26" height="4.2" rx="2.1" fill="#4a4744"/></g></svg>`
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
