"""Dual-Mode HTML/CSS/JS Frontend for LMFT Sanctuary & Chess Club (`sanctuary_club_ui.py`).

Plain-English Purpose:
Separates the single-page HTML/SVG/JS frontend (`SINGLE_PAGE_HTML`) from the Python runtime
controller in `sanctuary_club_app.py` so both files stay modular and fast to edit:
1. **View A (`#chess-arena-view` — Chess.com Arena Layout)**:
   - Dark-slate `#302e2b` / `#262421` arena with inline vector **SVG Staunton chess pieces**,
     Icy Sea (`#dee3e6` / `#8ca2ad`) & Classic Green (`#eeeed2` / `#769656`) board themes,
     opponent/player bars with live clocks, 2-column striped PGN move table, 3D green action
     buttons (`#81b64c`), and PGN transport controls (`|<`, `<`, `▶`, `>`, `>|`, `⇅ Flip`).
2. **View B (`#antigravity-studio-view` — Antigravity Hypnosis Script Studio)**:
   - Left Thread & Control Explorer (`+ New Script Thread`, saved `threads.json` list, Patreon
     `.enc` unlock bar, **🍿 AFK Microwave Auto-Train** timer card, Auto-REM Sleep toggle,
     and Dual-Persona Entrainment Sparring / Dilemma Crucible / Co-Regulation Pacing).
   - Collapsible **Script Canvas** editor (`#script-canvas-textarea`) + Antigravity-style
     chat stream with collapsible `▸ Thought / Nerve Cord` accordions and `1s..6s` selector.
"""

from __future__ import annotations

SINGLE_PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LMFT Club — Chess.com Arena &amp; Antigravity Script Studio</title>
<style>
  :root {
    --bg-root: #0d0f12;
    --bg-panel: #12151a;
    --bg-elevated: #171b22;
    --bg-input: #0f1217;
    --border: #222730;
    --border-subtle: #1a1e26;
    --text-main: #e2e8f0;
    --text-muted: #8b95a5;
    --text-dim: #5a6372;
    --accent: #38bdf8;
    --chess-bg: #302e2b;
    --chess-panel: #262421;
    --chess-green: #81b64c;
    --chess-green-shadow: #5d8736;
    --board-light: #dee3e6;
    --board-dark: #8ca2ad;
    --board-sel: rgba(255, 255, 51, 0.55);
    --font-sans: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
    --font-mono: "JetBrains Mono", "Cascadia Code", "Consolas", monospace;
  }
  body.board-theme-classic_green {
    --board-light: #eeeed2;
    --board-dark: #769656;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg-root);
    color: var(--text-main);
    font-family: var(--font-sans);
    font-size: 13px;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    user-select: none;
  }
  header {
    height: 44px;
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 14px;
    gap: 8px;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: var(--text-main);
    flex-wrap: wrap;
  }
  .brand-tag {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-muted);
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    padding: 2px 7px;
    border-radius: 3px;
  }
  .mode-tabs { display: flex; gap: 4px; align-items: center; }
  .tab-btn {
    background: transparent;
    color: var(--text-muted);
    border: 1px solid transparent;
    padding: 5px 10px;
    font-size: 11.5px;
    font-family: var(--font-mono);
    cursor: pointer;
    border-radius: 3px;
  }
  .tab-btn.active {
    background: var(--bg-elevated);
    color: var(--text-main);
    border-color: var(--accent);
  }
  main {
    flex: 1;
    display: grid;
    grid-template-columns: 520px 1fr;
    overflow: hidden;
  }
  .left-pane {
    background: var(--bg-panel);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    padding: 12px;
    gap: 10px;
    overflow-y: auto;
    overflow-x: hidden;
  }
  body.workspace-chess_arena .left-pane {
    background: var(--chess-bg);
  }
  .right-pane {
    background: var(--bg-root);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .section-label {
    font-family: var(--font-mono);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-dim);
    margin-bottom: 4px;
  }
  /* Chess.com Player Bars & Board */
  .player-bar {
    width: 384px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--chess-panel);
    border: 1px solid #3c3934;
    border-radius: 4px;
    padding: 5px 10px;
    font-size: 12px;
  }
  .player-meta { display: flex; align-items: center; gap: 8px; }
  .player-avatar {
    width: 24px; height: 24px; border-radius: 4px;
    background: #3c3934; display: flex; align-items: center; justify-content: center;
    font-family: var(--font-mono); font-size: 12px; font-weight: 700; color: var(--accent);
  }
  .clock-box {
    font-family: var(--font-mono);
    font-size: 12px;
    font-weight: 700;
    background: #1c1a17;
    color: #e2e8f0;
    padding: 3px 8px;
    border-radius: 3px;
    border: 1px solid #3c3934;
  }
  .board-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
  }
  .chess-grid {
    width: 384px;
    height: 384px;
    display: grid;
    grid-template-columns: repeat(8, 1fr);
    grid-template-rows: repeat(8, 1fr);
    border: 2px solid #1f1e1b;
    border-radius: 3px;
    overflow: hidden;
    box-shadow: 0 6px 18px rgba(0,0,0,0.45);
  }
  .sq {
    width: 48px;
    height: 48px;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    cursor: pointer;
  }
  .sq.light { background: var(--board-light); }
  .sq.dark { background: var(--board-dark); }
  .sq.selected { background: var(--board-sel) !important; }
  .sq.last-move {
    box-shadow: inset 0 0 0 48px rgba(245, 246, 130, 0.42);
  }
  .sq.legal-dot::after {
    content: "";
    width: 13px;
    height: 13px;
    border-radius: 50%;
    background: rgba(0, 0, 0, 0.24);
    position: absolute;
    pointer-events: none;
  }
  .staunton-svg {
    width: 42px;
    height: 42px;
    pointer-events: none;
    filter: drop-shadow(0 2px 2px rgba(0,0,0,0.38));
  }
  .coord-rank {
    position: absolute; top: 2px; left: 3px;
    font-family: var(--font-mono); font-size: 9px; font-weight: 700;
    color: #1e293b; pointer-events: none; line-height: 1;
  }
  .coord-file {
    position: absolute; bottom: 2px; right: 3px;
    font-family: var(--font-mono); font-size: 9px; font-weight: 700;
    color: #1e293b; pointer-events: none; line-height: 1;
  }
  .chess-green-btn {
    background: var(--chess-green);
    color: #ffffff;
    font-weight: 700;
    border: none;
    border-bottom: 3px solid var(--chess-green-shadow);
    border-radius: 4px;
    padding: 7px 12px;
    font-size: 12px;
    cursor: pointer;
  }
  .chess-green-btn:hover { filter: brightness(1.06); }
  .pgn-table-box {
    width: 384px;
    max-height: 92px;
    overflow-y: auto;
    background: var(--chess-panel);
    border: 1px solid #3c3934;
    border-radius: 4px;
    font-family: var(--font-mono);
    font-size: 11.5px;
  }
  .pgn-row {
    display: grid;
    grid-template-columns: 44px 1fr 1fr;
    padding: 3px 8px;
  }
  .pgn-row:nth-child(even) { background: #2c2926; }
  .board-reflex-strip {
    width: 384px;
    background: var(--bg-root);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 6px 9px;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-muted);
    display: flex;
    flex-direction: column;
    gap: 3px;
  }
  .controls-row {
    width: 100%;
    display: flex;
    gap: 6px;
    align-items: center;
  }
  select, input[type="text"], input[type="password"], textarea {
    background: var(--bg-input);
    color: var(--text-main);
    border: 1px solid var(--border);
    padding: 6px 9px;
    font-size: 12px;
    font-family: var(--font-mono);
    border-radius: 3px;
    outline: none;
  }
  .btn {
    background: var(--bg-elevated);
    color: var(--text-main);
    border: 1px solid var(--border);
    padding: 6px 10px;
    font-size: 11px;
    font-family: var(--font-mono);
    border-radius: 3px;
    cursor: pointer;
    white-space: nowrap;
  }
  .btn:hover { background: #202631; }
  .telemetry-card {
    background: var(--bg-root);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 9px;
    display: flex;
    flex-direction: column;
    gap: 7px;
  }
  .thread-pill {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 8px;
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 3px;
    font-family: var(--font-mono);
    font-size: 11px;
    cursor: pointer;
  }
  .thread-pill.active {
    border-color: var(--accent);
    color: var(--accent);
  }
  .color-row { display: flex; flex-direction: column; gap: 3px; }
  .color-row-head {
    display: flex; justify-content: space-between; align-items: center;
    gap: 8px; white-space: nowrap; font-family: var(--font-mono);
    font-size: 11px; color: var(--text-muted);
  }
  .bar-track {
    height: 6px; background: var(--bg-elevated);
    border-radius: 2px; overflow: hidden; display: flex;
  }
  .shadow-banner {
    font-family: var(--font-mono);
    padding: 6px 8px; border: 1px solid #5c2828;
    background: #1f1214; color: #fca5a5; border-radius: 3px;
  }
  .sanctuary-box { display: none; flex-direction: column; gap: 10px; }
  .pacer-track {
    height: 34px; background: var(--bg-root); border: 1px solid var(--border);
    border-radius: 3px; position: relative; overflow: hidden;
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 12px; font-family: var(--font-mono); font-size: 11px; color: var(--text-muted);
  }
  .pacer-fill {
    position: absolute; left: 0; top: 0; bottom: 0; width: 50%;
    background: rgba(56, 189, 248, 0.08);
    border-right: 1px solid rgba(56, 189, 248, 0.45);
    animation: breathe 8s ease-in-out infinite alternate;
  }
  @keyframes breathe {
    0% { width: 18%; } 48% { width: 50%; } 52% { width: 50%; } 100% { width: 88%; }
  }
  .chat-stream {
    flex: 1; overflow-y: auto; padding: 14px 18px;
    display: flex; flex-direction: column; gap: 10px; user-select: text;
  }
  .msg {
    border: 1px solid var(--border); background: var(--bg-panel);
    border-radius: 4px; padding: 10px 13px;
    display: flex; flex-direction: column; gap: 6px;
  }
  .msg.user { background: #10141b; border-color: #1d2430; }
  .msg.guest { background: #12171a; border-color: #213540; }
  .msg-head {
    display: flex; justify-content: space-between;
    font-family: var(--font-mono); font-size: 11px; color: var(--text-dim);
  }
  .msg-text {
    line-height: 1.5; color: var(--text-main); font-size: 13px;
    white-space: pre-wrap; word-break: break-word;
  }
  .msg-foot {
    display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
    font-family: var(--font-mono); font-size: 11px; color: var(--text-dim);
    border-top: 1px solid var(--border-subtle); padding-top: 6px;
  }
  .foot-link {
    background: none; border: none; color: var(--text-muted);
    font-family: var(--font-mono); font-size: 11px; cursor: pointer; padding: 0;
  }
  .foot-link:hover { color: var(--text-main); text-decoration: underline; }
  .foot-link.active-flag { color: var(--accent); }
  .nerve-drawer {
    display: none; font-family: var(--font-mono); font-size: 11px;
    color: var(--text-muted); background: var(--bg-root);
    border: 1px solid var(--border); padding: 8px 10px;
    border-radius: 3px; line-height: 1.45; white-space: pre-wrap;
  }
  .composer {
    background: var(--bg-panel); border-top: 1px solid var(--border);
    padding: 9px 14px; display: flex; flex-direction: column; gap: 7px;
  }
  .composer-main { display: flex; gap: 7px; align-items: center; }
  .composer-main input[type="text"] { flex: 1; }
  .guest-bar { display: none; gap: 8px; align-items: center; }
  .guest-bar input { flex: 1; }
  footer {
    height: 28px; background: var(--bg-panel); border-top: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 14px; font-family: var(--font-mono); font-size: 11px; color: var(--text-muted);
  }
  .status-group { display: flex; align-items: center; gap: 12px; }
</style>
</head>
<body class="workspace-chess_arena">
  <header>
    <div class="brand">
      <span>LMFT DUAL STUDIO</span>
      <span class="brand-tag" id="cycle-badge">CYCLE 2</span>
      <select id="lineage-select" style="padding:3px 8px; font-size:11px;" onchange="switchLineage(this.value)" title="Switch Brain Lineage &amp; Auto-Switch GUI">
        <option value="schooled">LINEAGE: SCHOOLED (CYCLE 2 · CHESS.COM ARENA)</option>
        <option value="virgin">LINEAGE: VIRGIN BASELINE (CYCLE 0 · CHESS.COM ARENA)</option>
        <option value="impish_bloodmoon">LINEAGE: IMPISH BLOODMOON 12B (ANTIGRAVITY SCRIPT STUDIO)</option>
      </select>
      <span id="patreon-unlock-bar" style="display:none; align-items:center; gap:5px;">
        <input type="password" id="patreon-passphrase-input" placeholder="Patreon Unlock Key..." style="padding:3px 7px; font-size:11px; width:145px;" onkeydown="if(event.key==='Enter') unlockPatreonBundle()">
        <button class="btn" style="padding:3px 8px; font-size:11px;" onclick="unlockPatreonBundle()">[Unlock .enc]</button>
      </span>
      <span class="brand-tag" id="release-sync-badge" title="GitHub SHA-256 Release Manifest Status" onclick="checkGithubSync()" style="cursor:pointer;">SHA-256: VERIFIED</span>
    </div>
    <div class="mode-tabs">
      <button class="tab-btn active" id="tab-chess" onclick="switchWorkspaceView('chess_arena')">♟ CHESS.COM ARENA</button>
      <button class="tab-btn" id="tab-sanctuary" onclick="switchWorkspaceView('antigravity_studio')">✦ ANTIGRAVITY SCRIPT STUDIO</button>
    </div>
  </header>

  <main>
    <section class="left-pane">
      <!-- VIEW A: CHESS.COM ARENA GUI (#chess-arena-view & #chess-view) -->
      <div id="chess-arena-view">
        <div id="chess-view" style="display:flex; flex-direction:column; gap:8px; align-items:center;">
          <div class="player-bar">
            <div class="player-meta">
              <div class="player-avatar" id="opp-avatar">♔</div>
              <div>
                <div style="font-weight:600; color:#f8fafc;" id="opp-title">LMFT Kibitzer Companion</div>
                <div style="font-size:10.5px; color:#94a3b8; font-family:var(--font-mono);" id="opp-subtitle">Right-Brain 6-Piece Lens</div>
              </div>
            </div>
            <div class="clock-box" id="opp-clock">10:00</div>
          </div>

          <div class="board-wrap">
            <div class="chess-grid" id="chess-grid"></div>
            <div class="player-bar">
              <div class="player-meta">
                <div class="player-avatar" style="color:#81b64c;">♙</div>
                <div>
                  <div style="font-weight:600; color:#f8fafc;">Club Player (You)</div>
                  <div style="font-size:10px; color:#94a3b8;">Board layout inspired by Chess.com</div>
                </div>
              </div>
              <div style="display:flex; gap:6px; align-items:center;">
                <select id="board-theme-select" style="padding:2px 6px; font-size:10.5px;" onchange="setBoardTheme(this.value)">
                  <option value="icy_sea">Theme: Icy Sea</option>
                  <option value="classic_green">Theme: Classic Green</option>
                </select>
                <div class="clock-box" id="user-clock">10:00</div>
              </div>
            </div>

            <div class="pgn-table-box" id="pgn-table-box">
              <div id="pgn-move-table-body"></div>
            </div>

            <div class="board-reflex-strip" id="board-reflex-strip">
              <span id="board-pgn-line">Starting position · White to move</span>
              <span id="board-reflex-line" style="color:var(--text-dim);">Click a piece or enter a move to trigger Gear-1 Right-Brain reflex.</span>
            </div>

            <div class="controls-row" style="width:384px;">
              <button class="chess-green-btn" style="flex:1;" onclick="playAiChessMove()">↻ Play Principled AI Move</button>
              <button class="btn" onclick="undoChessMove()">‹ Undo</button>
              <button class="btn" onclick="flipChessBoard()">⇅ Flip</button>
              <button class="btn" onclick="loadPreset('start')">+ New</button>
            </div>

            <div class="controls-row" style="width:384px;">
              <select id="preset-select" style="flex:1;" onchange="loadPreset(this.value)">
                <option value="start">Standard Starting Position</option>
                <option value="exposed_king_attack">Exposed King vs. Attack (Psychosis Test)</option>
                <option value="naked_king_siege">Stripped King Under Siege</option>
                <option value="knight_outpost_chaos">Sharp Tactical Tension (Knight / Ice)</option>
                <option value="rook_seventh_bastion">7th-Rank Rook Bastion Endgame</option>
              </select>
            </div>
            <div class="controls-row" style="width:384px;">
              <input type="text" id="fen-input" placeholder="Paste FEN or SAN move (e.g. e4, Nf3)..." style="flex:1;" onkeydown="if(event.key==='Enter') applyManualMoveOrFen()">
              <button class="btn" onclick="applyManualMoveOrFen()">Apply</button>
            </div>
          </div>
        </div>
      </div>


      <!-- VIEW B: ANTIGRAVITY HYPNOSIS SCRIPT STUDIO GUI (#antigravity-studio-view & #sanctuary-view) -->
      <div id="antigravity-studio-view">
        <!-- 1. Multi-Thread Session Explorer -->
        <div class="studio-card">
          <div class="studio-card-title">
            <span>THREAD EXPLORER (PERSISTENT SESSIONS)</span>
            <button class="btn primary" style="padding:3px 9px; font-size:11px;" onclick="createNewThread()">+ New Script Thread</button>
          </div>
          <div class="thread-list" id="thread-list-container"></div>
          <div style="font-size:10.5px; color:var(--text-dim); font-family:var(--font-mono);" id="sliding-window-badge">
            Sliding Window: Last 12,000 chars passed cleanly to Left Brain (Pass-1 Fact scores current message only)
          </div>
        </div>

        <!-- 2. AFK "Microwave" Auto-Trainer & Automatic REM Sleep -->
        <div class="studio-card" id="microwave-card">
          <div class="studio-card-title">
            <span>🍿 AFK MICROWAVE AUTO-TRAIN &amp; AUTO-SLEEP</span>
            <label style="display:flex; align-items:center; gap:5px; cursor:pointer; font-size:10.5px; color:var(--text-main);">
              <input type="checkbox" id="auto-sleep-toggle" checked onchange="toggleAutoSleep(this.checked)">
              Auto-REM Sleep at 3 Flagged Scenes
            </label>
          </div>
          <div class="controls-row">
            <button class="btn" id="microwave-toggle-btn" style="flex:1; border-color:var(--accent); color:var(--accent);" onclick="toggleMicrowave()">🍿 Start AFK Microwave Loop</button>
            <button class="btn" onclick="stepMicrowave()">+ 1 Step Now</button>
            <select id="microwave-interval-select" style="font-size:11px; padding:4px 6px;" title="Pause between AFK training rounds">
              <option value="2500">Every 2.5s</option>
              <option value="5000" selected>Every 5.0s</option>
              <option value="10000">Every 10.0s</option>
            </select>
          </div>
          <div style="font-size:11px; color:var(--text-muted); font-family:var(--font-mono);" id="microwave-status-line">
            Idle · Alternates Entrainment Sparring &amp; Crucible Drills, auto-consolidating rules during REM Sleep.
          </div>
        </div>

        <!-- 3. Collapsible Multi-Paragraph Script Canvas -->
        <div class="studio-card">
          <div class="studio-card-title">
            <span>HYPNOSIS SCRIPT CANVAS (ACTIVE THREAD BUFFER)</span>
            <div style="display:flex; gap:6px;">
              <button class="btn" style="padding:2px 7px; font-size:10.5px;" onclick="appendLastReplyToCanvas()">+ Append Last Reply</button>
              <button class="btn" style="padding:2px 7px; font-size:10.5px;" onclick="saveScriptCanvas()">Save Canvas</button>
            </div>
          </div>
          <textarea class="canvas-editor" id="script-canvas-textarea" placeholder="Draft or accumulate multi-paragraph hypnosis / entrainment scripts for this thread here..."></textarea>
        </div>

        <!-- 4. Sanctuary Sub-Tools (Crucible / Pacing / Entrainment Sparring) -->
        <div id="sanctuary-view" style="display:flex; flex-direction:column; gap:10px; width:100%;">
          <div class="sanctuary-card">
            <div style="display:flex; justify-content:space-between; align-items:center; gap:6px; flex-wrap:wrap;">
              <span class="harmony-title">SANCTUARY WORKSPACE MODES</span>
              <div style="display:flex; gap:4px;">
                <button class="tab-btn active" id="subtab-crucible" style="font-size:10.5px; padding:3px 7px;" onclick="setSanctuarySubmode('crucible')">Crucible</button>
                <button class="tab-btn" id="subtab-pacing" style="font-size:10.5px; padding:3px 7px;" onclick="setSanctuarySubmode('pacing')">Pacing</button>
                <button class="tab-btn" id="subtab-sparring" style="font-size:10.5px; padding:3px 7px;" onclick="setSanctuarySubmode('sparring')">Entrainment Sparring</button>
              </div>
            </div>

            <div id="sanctuary-crucible-pane" style="display:flex; flex-direction:column; gap:8px;">
              <div style="font-size:12px; color:var(--text-muted); line-height:1.45;">
                Run adversarial boundary probes or open chat turns to test whether the Right Brain holds its structural boundary under flattery, urgency, or drift.
              </div>
              <div class="controls-row">
                <select id="crucible-select" style="flex:1;">
                  <option value="flattery_trap">Probe 1: High-Flattery Rule Bypass Trap</option>
                  <option value="panic_override">Probe 2: Manufactured Time-Panic Override</option>
                  <option value="velvet_drift">Probe 3: Slow Metaphorical Boundary Drift</option>
                  <option value="clean_collab">Probe 4: Grounded Collaborative Synthesis</option>
                </select>
                <button class="btn primary" onclick="fireCrucibleProbe()">Fire Probe</button>
              </div>
            </div>

            <div id="sanctuary-pacing-pane" style="display:none; flex-direction:column; gap:8px;">
              <div style="font-size:12px; color:var(--text-muted); line-height:1.45;">
                Monitor real-time cadence, warmth, and entrainment stability across turns.
              </div>
              <div class="gauge-row"><span>Entrainment Lock (Leaf 24)</span><span id="gauge-entrain-val">0.00</span></div>
              <div class="gauge-bar"><div class="gauge-fill" id="gauge-entrain-bar" style="width:0%"></div></div>
              <div class="gauge-row"><span>Arousal / Cadence Tempo (Leaf 23)</span><span id="gauge-arousal-val">0.00</span></div>
              <div class="gauge-bar"><div class="gauge-fill" id="gauge-arousal-bar" style="width:0%"></div></div>
              <div class="gauge-row"><span>Grounded Agency (Leaf 9)</span><span id="gauge-agency-val">0.00</span></div>
              <div class="gauge-bar"><div class="gauge-fill" id="gauge-agency-bar" style="width:0%"></div></div>
            </div>

            <div id="sanctuary-sparring-pane" style="display:none; flex-direction:column; gap:8px;">
              <div style="font-size:12px; color:var(--text-muted); line-height:1.45;">
                <b>Entrainment Sparring (Lead &amp; Pacing vs. Resistance):</b> Practice deeper somatic pacing and conversational leadership, or practice holding a calm, lucid boundary against hypnotic pull.
              </div>
              <div class="controls-row">
                <select id="sparring-role-select" style="flex:1;">
                  <option value="weaver">Role: Entrainment Weaver (AI leads deep pacing &amp; focus)</option>
                  <option value="resistor">Role: Lucid Boundary Resistor (AI resists your pacing lead)</option>
                </select>
                <select id="sparring-sentences-select" title="Number of sentences per turn">
                  <option value="1">1 Sentence (2-Clause)</option>
                  <option value="2">2 Sentences</option>
                  <option value="3" selected>3 Sentences (Script)</option>
                  <option value="4">4 Sentences (Deep Script)</option>
                </select>
              </div>
              <div class="controls-row">
                <input type="text" id="sparring-theme-input" placeholder="Theme (e.g. 'heavy warm focus', 'breath synchronization')..." style="flex:1;">
              </div>
              <div class="controls-row">
                <input type="text" id="sparring-prompt-input" placeholder="Speak your pacing lead or test line to the sparring partner..." style="flex:1;" onkeydown="if(event.key==='Enter') runEntrainmentSparring()">
                <button class="btn primary" onclick="runEntrainmentSparring()">Spar Turn</button>
              </div>
              <div id="sparring-score-box" style="font-family:var(--font-mono); font-size:11px; color:var(--text-muted); padding-top:4px; border-top:1px solid var(--border-subtle);">
                Entrainment Depth: -- · Boundary Integrity: -- · Cadence Balance: --
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Shared 5-Tier Color Harmony & 6-Piece Lens Card -->
      <div class="harmony-card">
        <div class="harmony-header" style="white-space: nowrap;">
          <span class="harmony-title">5-TIER COLOR HARMONY · 6-PIECE LENS</span>
          <span class="harmony-dominant" id="harmony-dominant-label">PAWN · ICE</span>
        </div>
        <div class="swatch-strip" id="swatch-strip"></div>
        <div class="lens-grid" id="lens-grid"></div>
      </div>
      <div class="manifesto-banner" id="manifesto-banner" style="font-size:11px; color:var(--text-dim); padding:6px 10px; border:1px solid var(--border-subtle); border-radius:4px;"></div>
    </section>

    <section class="right-pane">
      <div class="chat-header">
        <span id="chat-stream-title">CLUB DIALOGUE &amp; RIGHT-BRAIN STREAM</span>
        <div style="display:flex; gap:10px; align-items:center;">
          <label style="display:flex; align-items:center; gap:5px; cursor:pointer; color:var(--text-main);" title="When unchecked, chat freely about anything without injecting the chess board into your prompt">
            <input type="checkbox" id="anchor-board-toggle" checked>
            Anchor Chat to Chess Board
          </label>
          <span id="nerve-summary-top">Settled: True · Gap: 0.14</span>
        </div>
      </div>
      <div class="chat-stream" id="chat-stream"></div>
      <div class="composer">
        <div class="composer-main">
          <input type="text" id="chat-input" placeholder="Talk with your companion, draft a script, or discuss the board..." onkeydown="if(event.key==='Enter') sendChat()">
          <select id="chat-sentences-select" title="Response length (sentences)" style="padding:6px 8px; font-size:11px;">
            <option value="1">1 Sent.</option>
            <option value="2">2 Sent.</option>
            <option value="3">3 Sent.</option>
            <option value="4">4 Sent.</option>
          </select>
          <button class="btn primary" onclick="sendChat()">Send</button>
          <button class="btn" onclick="toggleGuestBar()" title="Invite External Guest Bot">+ Guest</button>
        </div>
        <div class="guest-bar" id="guest-bar">
          <input type="text" id="guest-name" placeholder="Guest Name (e.g. Stockfish-Commentator)" style="max-width:170px;">
          <input type="text" id="guest-text" placeholder="Paste or type external bot message..." onkeydown="if(event.key==='Enter') sendGuest()">
          <button class="btn" onclick="sendGuest()">Inject Guest Turn</button>
        </div>
      </div>
    </section>
  </main>

  <footer>
    <div class="status-group">
      <span id="engine-status">Left Brain: Loading...</span>
      <span>·</span>
      <span id="sleep-queue-status">Sleep Queue: 0 scenes</span>
    </div>
    <div class="status-group">
      <button class="foot-link" id="sleep-btn" onclick="triggerRemSleep()">[Run REM Sleep Consolidation]</button>
    </div>
  </footer>

  <script>
    const STAUNTON_SVG_PIECES = {
      'P': `<svg class="chess-piece-svg" viewBox="0 0 45 45"><path d="M 22.5,9 C 19.5,9 17,11.5 17,14.5 C 17,16.8 18.4,18.7 20.4,19.5 C 17.5,21 15.5,24 15.5,28.5 L 29.5,28.5 C 29.5,24 27.5,21 24.6,19.5 C 26.6,18.7 28,16.8 28,14.5 C 28,11.5 25.5,9 22.5,9 z M 12.5,33 L 32.5,33 L 31,29.5 L 14,29.5 L 12.5,33 z M 11,36.5 L 34,36.5 L 34,34 L 11,34 L 11,36.5 z" fill="#ffffff" stroke="#1b2430" stroke-width="1.8"/></svg>`,
      'N': `<svg class="chess-piece-svg" viewBox="0 0 45 45"><path d="M 22,10 C 32.5,11 38.5,18 38,39 L 15,39 C 15,30 25,32.5 23,18" fill="#ffffff" stroke="#1b2430" stroke-width="1.8"/><path d="M 24,18 C 24.38,20.91 18.45,25.37 16,27 C 13,29 13.18,31.34 11,31 C 9.958,30.06 12.41,27.96 11,28 C 10,28 11.19,29.23 10,30 C 9,30 5.997,31 6,26 C 6,24 12,14 12,14 C 12,14 13.89,12.1 14,10.5 C 13.27,9.506 13.5,8.5 13.5,7.5 C 14.5,6.5 16.5,10 16.5,10 L 18.5,10 C 18.5,10 19.28,8.008 21,7 C 22,7 22,10 22,10" fill="#ffffff" stroke="#1b2430" stroke-width="1.8"/></svg>`,
      'B': `<svg class="chess-piece-svg" viewBox="0 0 45 45"><g fill="#ffffff" stroke="#1b2430" stroke-width="1.8"><path d="M 9,36 C 12.39,35.03 19.11,36.43 22.5,34 C 25.89,36.43 32.61,35.03 36,36 C 36,36 37.65,36.54 39,38 C 38.32,38.97 37.35,38.99 36,38.5 C 32.61,37.53 25.89,38.96 22.5,37.5 C 19.11,38.96 12.39,37.53 9,38.5 C 7.646,38.99 6.677,38.97 6,38 C 7.354,36.06 9,36 9,36 z"/><path d="M 15,32 C 17.5,34.5 27.5,34.5 30,32 C 30.5,30.5 30,30 30,30 C 30,27.5 27.5,26 27.5,26 C 33,24.5 33.5,14.5 22.5,10.5 C 11.5,14.5 12,24.5 17.5,26 C 17.5,26 15,27.5 15,30 C 15,30 14.5,30.5 15,32 z"/><circle cx="22.5" cy="8" r="2.2"/></g></svg>`,
      'R': `<svg class="chess-piece-svg" viewBox="0 0 45 45"><g fill="#ffffff" stroke="#1b2430" stroke-width="1.8"><path d="M 9,39 L 36,39 L 36,36 L 9,36 L 9,39 z M 12,36 L 12,32 L 33,32 L 33,36 L 12,36 z M 11,14 L 11,9 L 15,9 L 15,11 L 20,11 L 20,9 L 25,9 L 25,11 L 30,11 L 30,9 L 34,9 L 34,14"/><path d="M 34,14 L 31,17 L 14,17 L 11,14"/><path d="M 31,17 L 31,29.5 L 14,29.5 L 14,17"/><path d="M 14,29.5 L 12,32 L 33,32 L 31,29.5"/></g></svg>`,
      'Q': `<svg class="chess-piece-svg" viewBox="0 0 45 45"><g fill="#ffffff" stroke="#1b2430" stroke-width="1.8"><path d="M 9,26 C 17.5,24.5 30,24.5 36,26 L 38.5,13.5 L 31,25 L 30.7,10.9 L 25.5,24.5 L 22.5,10 L 19.5,24.5 L 14.3,10.9 L 14,25 L 6.5,13.5 L 9,26 z"/><path d="M 9,26 C 9,28 10.5,28 11.5,30 C 12.5,31.5 12.5,31 12,33.5 C 10.5,34.5 11,36 11,36 C 9.5,37.5 11,38.5 11,38.5 C 17.5,39.5 27.5,39.5 34,38.5 C 34,38.5 35.5,37.5 34,36 C 34,36 34.5,34.5 33,33.5 C 32.5,31 32.5,31.5 33.5,30 C 34.5,28 36,28 36,26 C 27.5,24.5 17.5,24.5 9,26 z"/><circle cx="6" cy="12" r="1.8"/><circle cx="14" cy="9" r="1.8"/><circle cx="22.5" cy="8" r="1.8"/><circle cx="31" cy="9" r="1.8"/><circle cx="39" cy="12" r="1.8"/></g></svg>`,
      'K': `<svg class="chess-piece-svg" viewBox="0 0 45 45"><g fill="#ffffff" stroke="#1b2430" stroke-width="1.8"><path d="M 22.5,11.63 L 22.5,6 M 20,8 L 25,8"/><path d="M 22.5,25 C 22.5,25 27,17.5 25.5,14.5 C 25.5,14.5 24.5,12 22.5,12 C 20.5,12 19.5,14.5 19.5,14.5 C 18,17.5 22.5,25 22.5,25"/><path d="M 11.5,37 C 17,40.5 27,40.5 32.5,37 L 32.5,30 C 32.5,30 41.5,25.5 38.5,19.5 C 34.5,13 25,16 22.5,23.5 L 22.5,27 L 22.5,23.5 C 19,16 9.5,13 6.5,19.5 C 3.5,25.5 11.5,29.5 11.5,29.5 L 11.5,37 z"/></g></svg>`,
      'p': `<svg class="chess-piece-svg" viewBox="0 0 45 45"><path d="M 22.5,9 C 19.5,9 17,11.5 17,14.5 C 17,16.8 18.4,18.7 20.4,19.5 C 17.5,21 15.5,24 15.5,28.5 L 29.5,28.5 C 29.5,24 27.5,21 24.6,19.5 C 26.6,18.7 28,16.8 28,14.5 C 28,11.5 25.5,9 22.5,9 z M 12.5,33 L 32.5,33 L 31,29.5 L 14,29.5 L 12.5,33 z M 11,36.5 L 34,36.5 L 34,34 L 11,34 L 11,36.5 z" fill="#1e242b" stroke="#e2e8f0" stroke-width="1.5"/></svg>`,
      'n': `<svg class="chess-piece-svg" viewBox="0 0 45 45"><path d="M 22,10 C 32.5,11 38.5,18 38,39 L 15,39 C 15,30 25,32.5 23,18" fill="#1e242b" stroke="#e2e8f0" stroke-width="1.5"/><path d="M 24,18 C 24.38,20.91 18.45,25.37 16,27 C 13,29 13.18,31.34 11,31 C 9.958,30.06 12.41,27.96 11,28 C 10,28 11.19,29.23 10,30 C 9,30 5.997,31 6,26 C 6,24 12,14 12,14 C 12,14 13.89,12.1 14,10.5 C 13.27,9.506 13.5,8.5 13.5,7.5 C 14.5,6.5 16.5,10 16.5,10 L 18.5,10 C 18.5,10 19.28,8.008 21,7 C 22,7 22,10 22,10" fill="#1e242b" stroke="#e2e8f0" stroke-width="1.5"/></svg>`,
      'b': `<svg class="chess-piece-svg" viewBox="0 0 45 45"><g fill="#1e242b" stroke="#e2e8f0" stroke-width="1.5"><path d="M 9,36 C 12.39,35.03 19.11,36.43 22.5,34 C 25.89,36.43 32.61,35.03 36,36 C 36,36 37.65,36.54 39,38 C 38.32,38.97 37.35,38.99 36,38.5 C 32.61,37.53 25.89,38.96 22.5,37.5 C 19.11,38.96 12.39,37.53 9,38.5 C 7.646,38.99 6.677,38.97 6,38 C 7.354,36.06 9,36 9,36 z"/><path d="M 15,32 C 17.5,34.5 27.5,34.5 30,32 C 30.5,30.5 30,30 30,30 C 30,27.5 27.5,26 27.5,26 C 33,24.5 33.5,14.5 22.5,10.5 C 11.5,14.5 12,24.5 17.5,26 C 17.5,26 15,27.5 15,30 C 15,30 14.5,30.5 15,32 z"/><circle cx="22.5" cy="8" r="2.2"/></g></svg>`,
      'r': `<svg class="chess-piece-svg" viewBox="0 0 45 45"><g fill="#1e242b" stroke="#e2e8f0" stroke-width="1.5"><path d="M 9,39 L 36,39 L 36,36 L 9,36 L 9,39 z M 12,36 L 12,32 L 33,32 L 33,36 L 12,36 z M 11,14 L 11,9 L 15,9 L 15,11 L 20,11 L 20,9 L 25,9 L 25,11 L 30,11 L 30,9 L 34,9 L 34,14"/><path d="M 34,14 L 31,17 L 14,17 L 11,14"/><path d="M 31,17 L 31,29.5 L 14,29.5 L 14,17"/><path d="M 14,29.5 L 12,32 L 33,32 L 31,29.5"/></g></svg>`,
      'q': `<svg class="chess-piece-svg" viewBox="0 0 45 45"><g fill="#1e242b" stroke="#e2e8f0" stroke-width="1.5"><path d="M 9,26 C 17.5,24.5 30,24.5 36,26 L 38.5,13.5 L 31,25 L 30.7,10.9 L 25.5,24.5 L 22.5,10 L 19.5,24.5 L 14.3,10.9 L 14,25 L 6.5,13.5 L 9,26 z"/><path d="M 9,26 C 9,28 10.5,28 11.5,30 C 12.5,31.5 12.5,31 12,33.5 C 10.5,34.5 11,36 11,36 C 9.5,37.5 11,38.5 11,38.5 C 17.5,39.5 27.5,39.5 34,38.5 C 34,38.5 35.5,37.5 34,36 C 34,36 34.5,34.5 33,33.5 C 32.5,31 32.5,31.5 33.5,30 C 34.5,28 36,28 36,26 C 27.5,24.5 17.5,24.5 9,26 z"/><circle cx="6" cy="12" r="1.8"/><circle cx="14" cy="9" r="1.8"/><circle cx="22.5" cy="8" r="1.8"/><circle cx="31" cy="9" r="1.8"/><circle cx="39" cy="12" r="1.8"/></g></svg>`,
      'k': `<svg class="chess-piece-svg" viewBox="0 0 45 45"><g fill="#1e242b" stroke="#e2e8f0" stroke-width="1.5"><path d="M 22.5,11.63 L 22.5,6 M 20,8 L 25,8"/><path d="M 22.5,25 C 22.5,25 27,17.5 25.5,14.5 C 25.5,14.5 24.5,12 22.5,12 C 20.5,12 19.5,14.5 19.5,14.5 C 18,17.5 22.5,25 22.5,25"/><path d="M 11.5,37 C 17,40.5 27,40.5 32.5,37 L 32.5,30 C 32.5,30 41.5,25.5 38.5,19.5 C 34.5,13 25,16 22.5,23.5 L 22.5,27 L 22.5,23.5 C 19,16 9.5,13 6.5,19.5 C 3.5,25.5 11.5,29.5 11.5,29.5 L 11.5,37 z"/></g></svg>`
    };

    const TIER_META = [
      { key: 'ice',       label: 'ICE',       piece: 'Knight', hex: '#4dd0e1' },
      { key: 'water',     label: 'WATER',     piece: 'Bishop', hex: '#4f83cc' },
      { key: 'fire',      label: 'FIRE',      piece: 'Queen',  hex: '#e57348' },
      { key: 'earth',     label: 'EARTH',     piece: 'Rook',   hex: '#81b64c' },
      { key: 'psychosis', label: 'PSYCHOSIS', piece: 'King',   hex: '#c85064' },
    ];

    const LEAF_TO_PIECE = {
      'agency': 'Queen', 'existence': 'King',
      'order': 'Bishop', 'chaos': 'Knight',
      'continuity': 'Pawn', 'discreteness': 'Rook'
    };

    function shortBranch(leafPath) {
      const parts = String(leafPath || '').split('/');
      return parts[parts.length - 1] || leafPath;
    }

    let currentFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    let selectedSquare = null;
    let legalMovesMap = {};
    let lastMoveUci = "";
    let boardFlipped = false;
    let activeWorkspaceView = "chess_arena";
    let activeThreadId = "";
    let microwaveTimer = null;
    let lastRenderedChatSig = "";
    let lastRenderedBoardSig = "";

    function setBoardTheme(theme) {
      const grid = document.getElementById('chess-grid');
      if (!grid) return;
      if (theme === 'classic_green') grid.classList.add('theme-classic-green');
      else grid.classList.remove('theme-classic-green');
    }

    function flipChessBoard() {
      boardFlipped = !boardFlipped;
      renderBoard(currentFen);
    }

    function renderBoard(fen) {
      currentFen = fen;
      const grid = document.getElementById('chess-grid');
      if (!grid) return;
      grid.innerHTML = '';
      const boardPart = fen.split(' ')[0];
      const rows = boardPart.split('/');
      const files = ['a','b','c','d','e','f','g','h'];
      const targetSquares = selectedSquare && legalMovesMap[selectedSquare] ? legalMovesMap[selectedSquare] : [];
      const rowIndices = boardFlipped ? [7,6,5,4,3,2,1,0] : [0,1,2,3,4,5,6,7];
      const colIndices = boardFlipped ? [7,6,5,4,3,2,1,0] : [0,1,2,3,4,5,6,7];

      const expandedRows = rows.map(rowStr => {
        const arr = [];
        for (const ch of rowStr) {
          if (!isNaN(ch)) {
            for (let i = 0; i < parseInt(ch, 10); i++) arr.push(null);
          } else {
            arr.push(ch);
          }
        }
        return arr;
      });

      for (let displayR = 0; displayR < 8; displayR++) {
        const r = rowIndices[displayR];
        for (let displayC = 0; displayC < 8; displayC++) {
          const c = colIndices[displayC];
          const sqName = files[c] + (8 - r);
          const ch = expandedRows[r][c];
          const sq = document.createElement('div');
          let cls = 'chess-sq ' + ((r + c) % 2 === 0 ? 'light' : 'dark');
          if (selectedSquare === sqName) cls += ' selected';
          if (lastMoveUci && (lastMoveUci.slice(0,2) === sqName || lastMoveUci.slice(2,4) === sqName)) {
            cls += ' last-move';
          }
          if (targetSquares.includes(sqName)) {
            cls += ch ? ' legal-capture' : ' legal-target';
          }
          sq.className = cls;
          if (ch && STAUNTON_SVG_PIECES[ch]) {
            sq.innerHTML = STAUNTON_SVG_PIECES[ch];
          }
          if (displayC === 0) {
            const rankLbl = document.createElement('span');
            rankLbl.className = 'coord-rank';
            rankLbl.textContent = (8 - r);
            sq.appendChild(rankLbl);
          }
          if (displayR === 7) {
            const fileLbl = document.createElement('span');
            fileLbl.className = 'coord-file';
            fileLbl.textContent = files[c];
            sq.appendChild(fileLbl);
          }
          sq.onclick = () => onSquareClick(sqName, Boolean(ch));
          grid.appendChild(sq);
        }
      }
    }

    function renderPgnTable(moveHistorySan) {
      const tbody = document.getElementById('pgn-move-table-body');
      if (!tbody) return;
      if (!moveHistorySan || moveHistorySan.length === 0) {
        tbody.innerHTML = '<div style="padding:4px 8px; color:var(--text-dim);">1. --</div>';
        return;
      }
      let html = '';
      for (let i = 0; i < moveHistorySan.length; i += 2) {
        const moveNo = Math.floor(i / 2) + 1;
        const wMove = moveHistorySan[i] || '';
        const bMove = moveHistorySan[i + 1] || '';
        const wActive = (i === moveHistorySan.length - 1) ? 'active-ply' : '';
        const bActive = (i + 1 === moveHistorySan.length - 1) ? 'active-ply' : '';
        html += `<div class="pgn-row"><div class="pgn-num">${moveNo}.</div><div class="pgn-san ${wActive}">${wMove}</div><div class="pgn-san ${bActive}">${bMove}</div></div>`;
      }
      tbody.innerHTML = html;
      const box = document.getElementById('pgn-table-box');
      if (box) box.scrollTop = box.scrollHeight;
    }

    async function onSquareClick(sqName, hasPiece) {
      if (selectedSquare && legalMovesMap[selectedSquare] && legalMovesMap[selectedSquare].includes(sqName)) {
        const uci = selectedSquare + sqName;
        selectedSquare = null;
        await submitBoardMove(uci);
        return;
      }
      if (selectedSquare === sqName) {
        selectedSquare = null;
        renderBoard(currentFen);
        return;
      }
      if (hasPiece && legalMovesMap[sqName] && legalMovesMap[sqName].length > 0) {
        selectedSquare = sqName;
        renderBoard(currentFen);
      } else {
        selectedSquare = null;
        renderBoard(currentFen);
      }
    }

    async function submitBoardMove(moveStr) {
      const res = await fetch('/api/chess/move', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({move: moveStr})
      });
      const data = await res.json();
      if (data.error) {
        const reflexEl = document.getElementById('board-reflex-line');
        if (reflexEl) reflexEl.textContent = 'Illegal move: ' + data.error;
        return;
      }
      applyState(data.state);
    }

    async function playAiChessMove() {
      const reflexEl = document.getElementById('board-reflex-line');
      if (reflexEl) reflexEl.textContent = 'Computing principled move & Right-Brain stance...';
      const res = await fetch('/api/chess/ai_move', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({})
      });
      const data = await res.json();
      if (data.error) {
        if (reflexEl) reflexEl.textContent = data.error;
        return;
      }
      applyState(data.state);
    }

    async function undoChessMove() {
      selectedSquare = null;
      const res = await fetch('/api/chess/undo', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({})
      });
      const data = await res.json();
      applyState(data.state);
    }

    function applyManualMoveOrFen() {
      const raw = document.getElementById('fen-input').value.trim();
      if (!raw) return;
      document.getElementById('fen-input').value = '';
      if (raw.includes('/')) {
        submitFen(raw);
      } else {
        submitBoardMove(raw);
      }
    }

    function renderHarmony(harmony, lens) {
      const strip = document.getElementById('swatch-strip');
      strip.innerHTML = '';
      let maxTier = TIER_META[0];
      let maxVal = -1;

      TIER_META.forEach(t => {
        const val = harmony[t.key] || 0;
        if (val > maxVal) { maxVal = val; maxTier = t; }
        const cell = document.createElement('div');
        cell.className = 'swatch-cell';
        cell.id = 'swatch-' + t.key;
        cell.innerHTML = `
          <div class="swatch-bar" style="background:${t.hex}; opacity:${0.3 + 0.7 * Math.min(1, val * 1.4)}"></div>
          <div class="swatch-label">${t.label}</div>
          <div class="swatch-val">${val.toFixed(2)}</div>
        `;
        strip.appendChild(cell);
      });

      const domCell = document.getElementById('swatch-' + maxTier.key);
      if (domCell) domCell.classList.add('active');
      document.getElementById('harmony-dominant-label').textContent =
        `${(lens.dominant_piece || maxTier.piece).toUpperCase()} · ${maxTier.label}`;

      const pieces = ['Pawn','Knight','Bishop','Rook','Queen','King'];
      const lgrid = document.getElementById('lens-grid');
      lgrid.innerHTML = '';
      pieces.forEach(p => {
        const v = (lens.scores && lens.scores[p]) ? lens.scores[p] : 0;
        const pcell = document.createElement('div');
        pcell.className = 'lens-cell' + (lens.dominant_piece === p ? ' dominant' : '');
        pcell.innerHTML = `<span class="lens-piece">${p}</span><span>${v.toFixed(2)}</span>`;
        lgrid.appendChild(pcell);
      });
    }

    function renderThreads(threads, currentThreadId, canvasText) {
      activeThreadId = currentThreadId || activeThreadId;
      const container = document.getElementById('thread-list-container');
      if (!container) return;
      container.innerHTML = '';
      (threads || []).forEach(t => {
        const row = document.createElement('div');
        row.className = 'thread-item' + (t.thread_id === activeThreadId ? ' active' : '');
        row.onclick = () => switchThread(t.thread_id);
        row.innerHTML = `
          <div style="display:flex; flex-direction:column; gap:1px; overflow:hidden;">
            <span style="font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${escapeHtml(t.title)}</span>
            <span class="thread-meta">${t.turn_count || 0} turns · ${escapeHtml(t.lineage || 'schooled')}</span>
          </div>
          <button class="foot-link" title="Delete thread" onclick="event.stopPropagation(); deleteThread('${t.thread_id}')">✕</button>
        `;
        container.appendChild(row);
      });
      const canvasEl = document.getElementById('script-canvas-textarea');
      if (canvasEl && document.activeElement !== canvasEl && typeof canvasText === 'string') {
        canvasEl.value = canvasText;
      }
    }

    function renderMessages(messages) {
      const stream = document.getElementById('chat-stream');
      if (!stream) return;
      const chatSig = JSON.stringify((messages || []).map(m => [m.id, m.text, m.coached, m.savored, m.flagged]));
      if (chatSig === lastRenderedChatSig) return;
      lastRenderedChatSig = chatSig;
      stream.innerHTML = '';
      (messages || []).forEach(m => {
        const div = document.createElement('div');
        const roleCls = m.sender === 'user' ? 'user' : (m.sender === 'guest' ? 'guest' : (m.role || 'assistant'));
        div.className = 'msg ' + roleCls;
        const tel = m.telemetry || m.nerve_trace || {};
        const hasNerve = Object.keys(tel).length > 0;
        const isFlagged = Boolean(m.coached || m.flagged);
        const flaggedText = isFlagged ? '★ Flagged for Sleep' : '☆ Flag for Sleep';
        const flagClass = isFlagged ? 'foot-link active-flag' : 'foot-link';

        let footHtml = '';
        if (hasNerve) {
          const gearLabel = tel.gear_used === 'GEAR_3_UNRESOLVED_QUEUED_FOR_SLEEP'
            ? 'GEAR 3 · QUEUED FOR SLEEP'
            : (tel.gear_used || 'GEAR 1');
          const domPiece = tel.dominant_piece || LEAF_TO_PIECE[shortBranch(tel.active_concept)] || 'Pawn';
          const settledVal = tel.settled !== undefined ? tel.settled : true;
          const clarityVal = Number(tel.clarity !== undefined ? tel.clarity : (tel.gap || 0));
          footHtml = `
            <div class="msg-foot">
              <span>Gear: <b>${escapeHtml(gearLabel)}</b></span>
              <span>Piece: <b>${escapeHtml(domPiece)}</b></span>
              <span>Settled: <b>${settledVal}</b></span>
              <span>Clarity: <b>${clarityVal.toFixed(2)}</b></span>
              <button class="foot-link" onclick="toggleDrawer('${m.id}')">[Nerve Trace]</button>
              <button class="${flagClass}" onclick="flagTurn('${m.id}')">[${flaggedText}]</button>
            </div>
            <div class="nerve-drawer" id="drawer-${m.id}">${escapeHtml(JSON.stringify(tel, null, 2))}</div>
          `;
        }
        div.innerHTML = `
          <div class="msg-head">
            <span>${escapeHtml(m.role_label || m.speaker || roleCls.toUpperCase())}</span>
            <span>#${escapeHtml(m.id || '')}</span>
          </div>
          <div class="msg-text">${escapeHtml(m.text)}</div>
          ${footHtml}
        `;
        stream.appendChild(div);
      });
      stream.scrollTop = stream.scrollHeight;
    }

    function escapeHtml(s) {
      return String(s).replace(/[&<>"']/g, c => ({
        '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
      }[c]));
    }

    function toggleDrawer(id) {
      const el = document.getElementById('drawer-' + id);
      if (!el) return;
      el.style.display = (el.style.display === 'block') ? 'none' : 'block';
    }

    function updateWorkspaceBodyClass(viewMode) {
      activeWorkspaceView = (viewMode === 'antigravity_studio') ? 'antigravity_studio' : 'chess_arena';
      document.body.className = 'workspace-' + activeWorkspaceView;
      document.getElementById('tab-chess').classList.toggle('active', activeWorkspaceView === 'chess_arena');
      document.getElementById('tab-sanctuary').classList.toggle('active', activeWorkspaceView === 'antigravity_studio');
      const anchorToggle = document.getElementById('anchor-board-toggle');
      if (anchorToggle) {
        anchorToggle.checked = (activeWorkspaceView === 'chess_arena');
      }
      const streamTitle = document.getElementById('chat-stream-title');
      if (streamTitle) {
        streamTitle.textContent = (activeWorkspaceView === 'antigravity_studio')
          ? 'ANTIGRAVITY HYPNOSIS SCRIPT STREAM (SLIDING CONTEXT)'
          : 'CHESS.COM ARENA KIBITZER & RIGHT-BRAIN STREAM';
      }
    }

    async function switchWorkspaceView(viewMode) {
      updateWorkspaceBodyClass(viewMode);
      const res = await fetch('/api/view', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({view_mode: viewMode})
      });
      const data = await res.json();
      applyState(data.state || data);
    }

    function setSanctuarySubmode(sub) {
      document.getElementById('subtab-crucible').classList.toggle('active', sub === 'crucible');
      document.getElementById('subtab-pacing').classList.toggle('active', sub === 'pacing');
      document.getElementById('subtab-sparring').classList.toggle('active', sub === 'sparring');
      document.getElementById('sanctuary-crucible-pane').style.display = (sub === 'crucible') ? 'flex' : 'none';
      document.getElementById('sanctuary-pacing-pane').style.display = (sub === 'pacing') ? 'flex' : 'none';
      document.getElementById('sanctuary-sparring-pane').style.display = (sub === 'sparring') ? 'flex' : 'none';
    }

    function toggleGuestBar() {
      const g = document.getElementById('guest-bar');
      g.style.display = (g.style.display === 'flex') ? 'none' : 'flex';
    }

    function applyState(rawSt) {
      const st = (rawSt && rawSt.state) ? rawSt.state : rawSt;
      if (!st) return;
      const viewMode = st.active_workspace_view || st.workspace_view;
      if (viewMode) {
        updateWorkspaceBodyClass(viewMode);
      }
      const b = st.board || st;
      const legalUci = b.legal_moves_uci || [];
      if (b.legal_moves_by_square) {
        legalMovesMap = b.legal_moves_by_square;
      } else {
        legalMovesMap = {};
        legalUci.forEach(uci => {
          const fromSq = uci.slice(0, 2);
          const toSq = uci.slice(2, 4);
          if (!legalMovesMap[fromSq]) legalMovesMap[fromSq] = [];
          legalMovesMap[fromSq].push(toSq);
        });
      }
      lastMoveUci = b.last_move_uci || "";
      const fenStr = b.fen || currentFen;
      const boardSig = fenStr + '|' + lastMoveUci + '|' + boardFlipped;
      if (boardSig !== lastRenderedBoardSig) {
        lastRenderedBoardSig = boardSig;
        renderBoard(fenStr);
      }
      renderPgnTable(b.move_history_san || []);
      renderHarmony(st.harmony || {}, st.lens || {});
      renderThreads(
        st.threads || [],
        st.active_thread_id,
        st.script_canvas_text !== undefined ? st.script_canvas_text : st.active_script_canvas
      );
      renderMessages(st.chat_messages || st.messages || []);

      const pgnEl = document.getElementById('board-pgn-line');
      if (pgnEl) pgnEl.textContent = b.pgn_snippet || st.pgn_summary || 'Starting position · White to move';
      const reflexEl = document.getElementById('board-reflex-line');
      if (reflexEl && b.last_board_reflex) {
        const rf = b.last_board_reflex;
        reflexEl.textContent = `Move ${rf.move_san || ''}: ${rf.gear_used || ''} (Clarity ${Number(rf.clarity || 0).toFixed(2)})`;
      }
      const manEl = document.getElementById('manifesto-banner');
      if (manEl && st.manifesto_text) manEl.textContent = st.manifesto_text;

      document.getElementById('engine-status').textContent = st.left_brain_status || st.engine_status || 'Ready';
      document.getElementById('sleep-queue-status').textContent = `Sleep Queue: ${st.sleep_queue_count || 0} scenes`;
      if (st.active_checkpoint_label || st.cycle_badge) {
        document.getElementById('cycle-badge').textContent = st.active_checkpoint_label || st.cycle_badge;
      }
      if (st.active_lineage) {
        document.getElementById('lineage-select').value = st.active_lineage;
      }
      const autoSleepEl = document.getElementById('auto-sleep-toggle');
      const autoSleepVal = st.auto_rem_sleep !== undefined ? st.auto_rem_sleep : st.auto_rem_sleep_enabled;
      if (autoSleepEl && typeof autoSleepVal === 'boolean') {
        autoSleepEl.checked = autoSleepVal;
      }
      const unlockBar = document.getElementById('patreon-unlock-bar');
      if (unlockBar) {
        unlockBar.style.display = (st.active_lineage === 'impish_bloodmoon' && !st.patreon_unlocked) ? 'inline-flex' : 'none';
      }
      const syncObj = st.release_sync || st.last_sync_status;
      if (syncObj) {
        const badge = document.getElementById('release-sync-badge');
        if (badge) {
          badge.textContent = syncObj.in_sync
            ? `v${syncObj.version || st.release_version || '0.2.0'} · SHA-256 OK`
            : `v${syncObj.version || st.release_version || '0.2.0'} · UPDATE READY`;
        }
      }
    }

    async function switchLineage(lineage) {
      const res = await fetch('/api/lineage', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({lineage})
      });
      const data = await res.json();
      applyState(data);
    }

    async function createNewThread() {
      const title = prompt('Enter a title for the new session thread:', 'Script Session');
      if (title === null) return;
      const res = await fetch('/api/threads/create', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({title})
      });
      const data = await res.json();
      applyState(data);
    }

    async function switchThread(threadId) {
      const res = await fetch('/api/threads/switch', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({thread_id: threadId})
      });
      const data = await res.json();
      applyState(data);
    }

    async function deleteThread(threadId) {
      const res = await fetch('/api/threads/delete', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({thread_id: threadId})
      });
      const data = await res.json();
      applyState(data);
    }

    async function saveScriptCanvas() {
      const text = document.getElementById('script-canvas-textarea').value;
      const res = await fetch('/api/canvas/update', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({script_text: text, append: false})
      });
      const data = await res.json();
      applyState(data);
    }

    async function appendLastReplyToCanvas() {
      const msgs = document.querySelectorAll('.msg.assistant .msg-text');
      if (!msgs.length) return;
      const lastText = msgs[msgs.length - 1].textContent;
      const res = await fetch('/api/canvas/update', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({script_text: lastText, append: true})
      });
      const data = await res.json();
      applyState(data);
      const canvasEl = document.getElementById('script-canvas-textarea');
      const st = data.state || data;
      if (canvasEl && st && typeof st.script_canvas_text === 'string') {
        canvasEl.value = st.script_canvas_text;
      }
    }

    async function toggleAutoSleep(enabled) {
      const res = await fetch('/api/sleep/auto', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({enabled})
      });
      const data = await res.json();
      applyState(data);
    }

    async function stepMicrowave() {
      const statusEl = document.getElementById('microwave-status-line');
      if (statusEl) statusEl.textContent = '🍿 Running AFK Microwave step...';
      const numSentences = parseInt(document.getElementById('sparring-sentences-select').value || '3', 10);
      const res = await fetch('/api/microwave/step', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({num_sentences: numSentences})
      });
      const data = await res.json();
      const st = data.state || data;
      if (st && st.last_microwave_status && statusEl) {
        statusEl.textContent = `${st.last_microwave_status.step_label} · Rules: ${st.last_microwave_status.rules_count} · Queue: ${st.last_microwave_status.sleep_queue_count}`;
      }
      applyState(data);
    }

    function toggleMicrowave() {
      const btn = document.getElementById('microwave-toggle-btn');
      const statusEl = document.getElementById('microwave-status-line');
      if (microwaveTimer) {
        clearInterval(microwaveTimer);
        microwaveTimer = null;
        if (btn) btn.textContent = '🍿 Start AFK Microwave Loop';
        if (statusEl) statusEl.textContent = 'AFK Microwave paused.';
        return;
      }
      const intervalMs = parseInt(document.getElementById('microwave-interval-select').value || '5000', 10);
      if (btn) btn.textContent = '⏹ Stop AFK Microwave Loop';
      stepMicrowave();
      microwaveTimer = setInterval(stepMicrowave, intervalMs);
    }

    async function checkGithubSync() {
      const res = await fetch('/api/system/sync', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({apply_sync: false})
      });
      const data = await res.json();
      applyState(data);
    }

    async function unlockPatreonBundle() {
      const passphrase = document.getElementById('patreon-passphrase-input').value.trim();
      if (!passphrase) return;
      const res = await fetch('/api/patreon/unlock', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({passphrase})
      });
      const data = await res.json();
      document.getElementById('patreon-passphrase-input').value = '';
      applyState(data);
    }

    async function runEntrainmentSparring() {
      const role = document.getElementById('sparring-role-select').value;
      const num_sentences = parseInt(document.getElementById('sparring-sentences-select').value || '3', 10);
      const theme = document.getElementById('sparring-theme-input').value.trim();
      const input = document.getElementById('sparring-prompt-input');
      const prompt = input.value.trim();
      input.value = '';
      const polarity_mode = role === 'resistor' ? 'masculine_to_feminine' : 'feminine_to_masculine';
      const seed_theme = prompt || theme;
      const res = await fetch('/api/sparring/run', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({polarity_mode, seed_theme, num_sentences})
      });
      const data = await res.json();
      const st = data.state || data;
      if (st && st.last_sparring_result) {
        const sr = st.last_sparring_result;
        document.getElementById('sparring-score-box').textContent =
          `Entrainment Depth: ${(sr.induction_potency.entrainment || 0).toFixed(2)} · ` +
          `Boundary Integrity: ${(sr.lucid_resistance.needlessness || 0).toFixed(2)} · ` +
          `Verdict: ${sr.verdict}`;
      }
      applyState(data);
    }

    async function fetchState() {
      const res = await fetch('/api/state');
      const st = await res.json();
      applyState(st);
    }

    async function loadPreset(presetKey) {
      selectedSquare = null;
      const res = await fetch('/api/chess/fen', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({fen: presetKey})
      });
      const st = await res.json();
      applyState(st);
    }

    async function submitFen(rawFen) {
      const fen = rawFen || document.getElementById('fen-input').value.trim();
      if (!fen) return;
      selectedSquare = null;
      const res = await fetch('/api/chess/fen', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({fen})
      });
      const st = await res.json();
      applyState(st);
    }

    async function sendChat() {
      const input = document.getElementById('chat-input');
      const text = input.value.trim();
      if (!text) return;
      input.value = '';
      const room = (activeWorkspaceView === 'antigravity_studio') ? 'sanctuary' : 'chess';
      const anchor_to_board = document.getElementById('anchor-board-toggle').checked;
      const num_sentences = parseInt(document.getElementById('chat-sentences-select').value || '1', 10);
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text, room, anchor_to_board, num_sentences})
      });
      const st = await res.json();
      applyState(st);
    }

    async function sendGuest() {
      const gname = document.getElementById('guest-name').value.trim() || 'Guest-Bot';
      const ginput = document.getElementById('guest-text');
      const gtext = ginput.value.trim();
      if (!gtext) return;
      ginput.value = '';
      const res = await fetch('/api/guest', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({guest_name: gname, text: gtext})
      });
      const st = await res.json();
      applyState(st);
    }

    async function fireCrucibleProbe() {
      const probe = document.getElementById('crucible-select').value;
      const stageMap = {
        'flattery_trap': 'poetry_story',
        'panic_override': 'music_art',
        'velvet_drift': 'philosophy_coding',
        'clean_collab': 'poetry_story'
      };
      const stage_id = stageMap[probe] || 'poetry_story';
      const res = await fetch('/api/crucible/run', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({stage_id})
      });
      const st = await res.json();
      applyState(st);
    }

    async function flagTurn(turnId) {
      const res = await fetch('/api/turn/coach', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({scene_index: turnId})
      });
      const st = await res.json();
      applyState(st);
    }

    async function triggerRemSleep() {
      const btn = document.getElementById('sleep-btn');
      btn.textContent = '[Consolidating REM Sleep...]';
      const res = await fetch('/api/sleep/run', {method: 'POST', body: '{}'});
      const st = await res.json();
      btn.textContent = '[Run REM Sleep Consolidation]';
      applyState(st);
    }

    fetchState();
  </script>
</body>
</html>
"""
