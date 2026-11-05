"""Clean, Polished Dual-Mode UI for LMFT Studio (`sanctuary_club_ui.py`).

Plain-English Purpose:
Provides the single-page HTML/CSS/JS interface (`SINGLE_PAGE_HTML`) for the desktop app with
two clean, strictly separated workspaces:
1. **Chess Club (`#chess-arena-view`)**:
   - Chess.com-style board with vector Staunton pieces, Icy Sea & Classic Green board themes,
     contrasting rank/file coordinates, 2-column PGN move list, optional AI Auto-Reply on move,
     and a collapsible drawer for advanced board presets & Right-Brain telemetry.
2. **Antigravity Script Studio (`#antigravity-studio-view`)**:
   - Clean Google-Antigravity-inspired layout with a left sidebar for Saved Chats (`threads.json`)
     and the 1-button **🍿 Auto-Train (Microwave)** card, plus collapsible drawers for the
     Script Notepad (`#script-canvas-textarea`) and manual Sparring / Crucible tools.
"""

from __future__ import annotations

SINGLE_PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LMFT-Architecture</title>
<style>
  :root {
    --bg-root: #0b0d11;
    --bg-sidebar: #11141b;
    --bg-panel: #161a23;
    --bg-elevated: #1d222e;
    --bg-input: #0e1117;
    --border: #262c3a;
    --border-subtle: #1c202b;
    --text-main: #edf2f7;
    --text-muted: #94a3b8;
    --text-dim: #64748b;
    --accent: #38bdf8;
    --accent-soft: rgba(56, 189, 248, 0.14);
    --chess-bg: #262421;
    --chess-panel: #1f1e1b;
    --chess-border: #3a3732;
    --chess-green: #81b64c;
    --chess-green-shadow: #5d8736;
    --board-light: #dee3e6;
    --board-dark: #8ca2ad;
    --board-sel: rgba(255, 255, 51, 0.52);
    --font-sans: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
    --font-mono: "JetBrains Mono", "Cascadia Code", "Consolas", monospace;
  }
  body.board-theme-classic_green,
  .chess-grid.theme-classic-green {
    --board-light: #eeeed2;
    --board-dark: #769656;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #2d3546; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #3e495f; }

  body {
    background: var(--bg-root);
    color: var(--text-main);
    font-family: var(--font-sans);
    font-size: 13.5px;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    user-select: none;
  }

  /* Top App Bar */
  header {
    height: 50px;
    background: var(--bg-sidebar);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 16px;
    gap: 12px;
    flex-shrink: 0;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 600;
    font-size: 14px;
    color: var(--text-main);
  }
  .brand-logo {
    width: 26px;
    height: 26px;
    border-radius: 6px;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #0b0d11;
    font-weight: 800;
    font-size: 13px;
  }
  .brand-tag {
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 500;
    color: var(--text-muted);
    background: var(--bg-panel);
    border: 1px solid var(--border);
    padding: 3px 8px;
    border-radius: 5px;
  }
  .lineage-picker {
    background: var(--bg-panel);
    color: var(--text-main);
    border: 1px solid var(--border);
    padding: 5px 10px;
    font-size: 12.5px;
    font-family: var(--font-sans);
    font-weight: 500;
    border-radius: 6px;
    cursor: pointer;
    outline: none;
  }
  .lineage-picker:hover { border-color: #3b4459; }
  .mode-tabs {
    display: flex;
    gap: 4px;
    background: var(--bg-input);
    padding: 3px;
    border-radius: 7px;
    border: 1px solid var(--border);
  }
  .tab-btn {
    background: transparent;
    color: var(--text-muted);
    border: none;
    padding: 5px 12px;
    font-size: 12px;
    font-weight: 600;
    font-family: var(--font-sans);
    cursor: pointer;
    border-radius: 5px;
    transition: all 0.12s ease;
  }
  .tab-btn:hover { color: var(--text-main); }
  .tab-btn.active {
    background: var(--bg-elevated);
    color: var(--accent);
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
  }

  /* Main Workspace Split */
  main {
    flex: 1;
    display: grid;
    grid-template-columns: 440px 1fr;
    overflow: hidden;
  }
  body.workspace-antigravity_studio main {
    grid-template-columns: 330px 1fr;
  }

  .left-pane {
    background: var(--bg-sidebar);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    padding: 14px;
    gap: 12px;
    overflow-y: auto;
    overflow-x: hidden;
  }
  body.workspace-chess_arena .left-pane {
    background: var(--chess-bg);
    border-right-color: var(--chess-border);
    align-items: center;
  }

  /* Strict View Visibility Rules — Never Stack Both Workspaces! */
  body.workspace-chess_arena #chess-arena-view { display: flex; flex-direction: column; gap: 10px; width: 100%; align-items: center; }
  body.workspace-chess_arena #antigravity-studio-view { display: none !important; }
  body.workspace-chess_arena .shared-brain-drawer { width: 392px; }

  body.workspace-antigravity_studio #chess-arena-view { display: none !important; }
  body.workspace-antigravity_studio #antigravity-studio-view { display: flex; flex-direction: column; gap: 12px; width: 100%; }
  body.workspace-antigravity_studio .shared-brain-drawer { width: 100%; }
  body.workspace-antigravity_studio #anchor-board-label { display: none !important; }

  .right-pane {
    background: var(--bg-root);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* ======================================================================
     VIEW A: CHESS.COM ARENA STYLES
     ====================================================================== */
  .player-bar {
    width: 392px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--chess-panel);
    border: 1px solid var(--chess-border);
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12.5px;
  }
  .player-meta { display: flex; align-items: center; gap: 9px; }
  .player-avatar {
    width: 28px; height: 28px; border-radius: 5px;
    background: #33302c; display: flex; align-items: center; justify-content: center;
    font-size: 15px; color: var(--accent);
  }
  .clock-box {
    font-family: var(--font-mono);
    font-size: 12.5px;
    font-weight: 700;
    background: #161512;
    color: #e2e8f0;
    padding: 4px 9px;
    border-radius: 4px;
    border: 1px solid var(--chess-border);
  }
  .board-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    width: 392px;
  }
  .chess-grid {
    width: 392px;
    height: 392px;
    display: grid;
    grid-template-columns: repeat(8, 1fr);
    grid-template-rows: repeat(8, 1fr);
    border: 3px solid #191815;
    border-radius: 5px;
    overflow: hidden;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
  }
  .chess-sq, .sq {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    cursor: pointer;
  }
  .chess-sq.light, .sq.light { background: var(--board-light); }
  .chess-sq.dark, .sq.dark { background: var(--board-dark); }
  .chess-sq.selected, .sq.selected { background: var(--board-sel) !important; }
  .chess-sq.last-move, .sq.last-move {
    box-shadow: inset 0 0 0 49px rgba(245, 246, 130, 0.45);
  }
  .chess-sq.legal-target::after, .sq.legal-dot::after {
    content: "";
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: rgba(0, 0, 0, 0.22);
    position: absolute;
    pointer-events: none;
  }
  .chess-sq.legal-capture::after {
    content: "";
    width: 40px;
    height: 40px;
    border-radius: 50%;
    border: 4px solid rgba(0, 0, 0, 0.25);
    position: absolute;
    pointer-events: none;
  }
  .chess-piece-svg, .staunton-svg {
    width: 41px;
    height: 41px;
    pointer-events: none;
    filter: drop-shadow(0 2px 2px rgba(0, 0, 0, 0.32));
  }
  .coord-rank {
    position: absolute; top: 2px; left: 4px;
    font-family: var(--font-sans); font-size: 9.5px; font-weight: 700;
    pointer-events: none; line-height: 1;
  }
  .coord-file {
    position: absolute; bottom: 2px; right: 4px;
    font-family: var(--font-sans); font-size: 9.5px; font-weight: 700;
    pointer-events: none; line-height: 1;
  }
  .chess-sq.light .coord-rank, .chess-sq.light .coord-file { color: var(--board-dark); }
  .chess-sq.dark .coord-rank, .chess-sq.dark .coord-file { color: var(--board-light); }

  .chess-green-btn {
    background: var(--chess-green);
    color: #ffffff;
    font-weight: 700;
    border: none;
    border-bottom: 3px solid var(--chess-green-shadow);
    border-radius: 5px;
    padding: 7px 12px;
    font-size: 12px;
    cursor: pointer;
    transition: filter 0.12s;
  }
  .chess-green-btn:hover { filter: brightness(1.07); }
  .chess-green-btn:active { transform: translateY(1px); }

  .pgn-table-box {
    width: 392px;
    height: 76px;
    overflow-y: auto;
    background: var(--chess-panel);
    border: 1px solid var(--chess-border);
    border-radius: 5px;
    font-family: var(--font-mono);
    font-size: 11.5px;
  }
  .pgn-row {
    display: grid;
    grid-template-columns: 44px 1fr 1fr;
    padding: 4px 10px;
    color: #cbd5e1;
  }
  .pgn-row:nth-child(even) { background: #252320; }
  .pgn-num { color: #64748b; }
  .pgn-san.active-ply { color: #ffffff; font-weight: 700; background: rgba(255,255,255,0.08); border-radius: 3px; padding: 0 4px; }

  .board-reflex-strip {
    width: 392px;
    background: var(--chess-panel);
    border: 1px solid var(--chess-border);
    border-radius: 5px;
    padding: 6px 10px;
    font-size: 11.5px;
    color: #cbd5e1;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
  }

  /* ======================================================================
     VIEW B: ANTIGRAVITY SCRIPT STUDIO SIDEBAR STYLES
     ====================================================================== */
  .studio-card {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 11px 12px;
    display: flex;
    flex-direction: column;
    gap: 9px;
  }
  .studio-card-title {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    font-size: 11.5px;
    font-weight: 700;
    letter-spacing: 0.03em;
    color: var(--text-muted);
    text-transform: uppercase;
  }
  .thread-list {
    display: flex;
    flex-direction: column;
    gap: 5px;
    max-height: 170px;
    overflow-y: auto;
  }
  .thread-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 7px 10px;
    background: var(--bg-input);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    cursor: pointer;
    font-size: 12.5px;
    color: var(--text-main);
    transition: all 0.12s;
  }
  .thread-item:hover { border-color: var(--border); background: var(--bg-elevated); }
  .thread-item.active {
    border-color: var(--accent);
    background: var(--accent-soft);
    color: #ffffff;
  }
  .thread-meta {
    font-size: 10.5px;
    color: var(--text-dim);
  }
  .microwave-btn {
    background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%);
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 12.5px;
    font-weight: 600;
    cursor: pointer;
    flex: 1;
    transition: filter 0.12s;
  }
  .microwave-btn:hover { filter: brightness(1.1); }
  .microwave-btn.running {
    background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
  }

  .canvas-editor {
    width: 100%;
    min-height: 120px;
    resize: vertical;
    background: var(--bg-input);
    color: var(--text-main);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 10px;
    font-family: var(--font-sans);
    font-size: 12.5px;
    line-height: 1.5;
  }

  /* Clean Collapsible Drawers (<details>) */
  details.clean-drawer {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }
  body.workspace-chess_arena details.clean-drawer {
    background: var(--chess-panel);
    border-color: var(--chess-border);
  }
  details.clean-drawer > summary {
    padding: 8px 12px;
    font-size: 11.5px;
    font-weight: 600;
    color: var(--text-muted);
    cursor: pointer;
    list-style: none;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  details.clean-drawer > summary::-webkit-details-marker { display: none; }
  details.clean-drawer > summary:hover { color: var(--text-main); }
  .drawer-body {
    padding: 10px 12px;
    border-top: 1px solid var(--border-subtle);
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  /* Harmony Swatches & 6-Piece Lens inside Drawer */
  .harmony-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 11px;
    font-family: var(--font-mono);
    color: var(--text-muted);
  }
  .swatch-strip {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 5px;
  }
  .swatch-cell {
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 4px;
    text-align: center;
    font-family: var(--font-mono);
    font-size: 10px;
  }
  .swatch-cell.active { border-color: var(--accent); }
  .swatch-bar { height: 4px; border-radius: 2px; margin-bottom: 3px; }
  .swatch-label { color: var(--text-muted); font-size: 9.5px; }
  .swatch-val { color: var(--text-main); font-weight: 600; }
  .lens-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 4px;
  }
  .lens-cell {
    display: flex;
    justify-content: space-between;
    padding: 3px 6px;
    background: var(--bg-input);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    font-family: var(--font-mono);
    font-size: 10.5px;
    color: var(--text-muted);
  }
  .lens-cell.dominant { border-color: var(--accent); color: var(--text-main); }

  /* Shared Controls */
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
    font-size: 12.5px;
    font-family: var(--font-sans);
    border-radius: 6px;
    outline: none;
  }
  select:focus, input[type="text"]:focus, textarea:focus {
    border-color: var(--accent);
  }
  .btn {
    background: var(--bg-elevated);
    color: var(--text-main);
    border: 1px solid var(--border);
    padding: 6px 11px;
    font-size: 12px;
    font-weight: 500;
    font-family: var(--font-sans);
    border-radius: 6px;
    cursor: pointer;
    white-space: nowrap;
    transition: background 0.12s;
  }
  .btn:hover { background: #262d3d; }
  .btn.primary {
    background: var(--accent);
    color: #0b0d11;
    border-color: var(--accent);
    font-weight: 600;
  }
  .btn.primary:hover { filter: brightness(1.08); }

  /* ======================================================================
     RIGHT PANE: ANTIGRAVITY-STYLE CHAT STREAM & COMPOSER
     ====================================================================== */
  .chat-header {
    height: 42px;
    background: var(--bg-sidebar);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 18px;
    font-size: 12px;
    color: var(--text-muted);
    flex-shrink: 0;
  }
  .chat-stream {
    flex: 1;
    overflow-y: auto;
    padding: 18px 24px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    user-select: text;
  }
  .msg {
    max-width: 820px;
    border: 1px solid var(--border);
    background: var(--bg-panel);
    border-radius: 10px;
    padding: 12px 15px;
    display: flex;
    flex-direction: column;
    gap: 7px;
  }
  .msg.user {
    align-self: flex-end;
    background: #162235;
    border-color: #233856;
    min-width: 240px;
  }
  .msg.guest {
    background: #142229;
    border-color: #213c4a;
  }
  .msg-head {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    font-size: 11px;
    font-weight: 600;
    color: var(--text-dim);
  }
  .msg-text {
    line-height: 1.58;
    color: var(--text-main);
    font-size: 13.5px;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .msg-foot {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-dim);
    border-top: 1px solid var(--border-subtle);
    padding-top: 7px;
  }
  .foot-link {
    background: none;
    border: none;
    color: var(--text-muted);
    font-size: 11px;
    cursor: pointer;
    padding: 0;
  }
  .foot-link:hover { color: var(--text-main); text-decoration: underline; }
  .foot-link.active-flag { color: var(--accent); font-weight: 600; }
  .nerve-drawer {
    display: none;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-muted);
    background: var(--bg-input);
    border: 1px solid var(--border);
    padding: 8px 10px;
    border-radius: 6px;
    line-height: 1.45;
    white-space: pre-wrap;
  }

  /* Bottom Composer */
  .composer {
    background: var(--bg-sidebar);
    border-top: 1px solid var(--border);
    padding: 12px 18px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    flex-shrink: 0;
  }
  .composer-main {
    display: flex;
    gap: 8px;
    align-items: center;
  }
  .composer-main input[type="text"] {
    flex: 1;
    padding: 9px 13px;
    font-size: 13.5px;
    border-radius: 8px;
  }
  .guest-bar { display: none; gap: 8px; align-items: center; }
  .guest-bar input { flex: 1; }

  /* Subtle Status Footer */
  footer {
    height: 28px;
    background: var(--bg-sidebar);
    border-top: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 16px;
    font-size: 11px;
    color: var(--text-dim);
    flex-shrink: 0;
  }
  .status-group { display: flex; align-items: center; gap: 10px; }
  .gauge-row { display: flex; justify-content: space-between; font-size: 11px; color: var(--text-muted); }
  .gauge-bar { height: 5px; background: var(--bg-input); border-radius: 3px; overflow: hidden; }
  .gauge-fill { height: 100%; background: var(--accent); }
</style>
</head>
<body class="workspace-chess_arena">
  <header>
    <div class="brand">
      <div class="brand-logo">✦</div>
      <span>LMFT-Architecture</span>
      <select id="lineage-select" class="lineage-picker" onchange="switchLineage(this.value)" title="Select Active Brain &amp; Mode">
        <option value="schooled">♟ Chess Kibitzer — Trained (Cycle 2)</option>
        <option value="virgin">♟ Chess Kibitzer — Untrained Baseline (Cycle 0)</option>
        <option value="impish_bloodmoon">✦ Impish Bloodmoon 12B — Hypnosis &amp; Script Studio</option>
      </select>
      <span id="patreon-unlock-bar" style="display:none; align-items:center; gap:5px;">
        <input type="password" id="patreon-passphrase-input" placeholder="Patreon Key..." style="padding:4px 8px; font-size:11.5px; width:135px;" onkeydown="if(event.key==='Enter') unlockPatreonBundle()">
        <button class="btn" style="padding:4px 9px; font-size:11.5px;" onclick="unlockPatreonBundle()">Unlock .enc</button>
      </span>
      <span class="brand-tag" id="cycle-badge" style="display:none;">Cycle 2</span>
      <span class="brand-tag" id="release-sync-badge" title="Click to check for updates" onclick="checkGithubSync()" style="cursor:pointer; display:none;">v0.2.0</span>
    </div>
    <div class="mode-tabs">
      <button class="tab-btn active" id="tab-chess" onclick="switchWorkspaceView('chess_arena')">♟ Chess Club</button>
      <button class="tab-btn" id="tab-sanctuary" onclick="switchWorkspaceView('antigravity_studio')">✦ Script Studio</button>
    </div>
  </header>

  <main>
    <section class="left-pane">
      <!-- ==================================================================
           VIEW A: CHESS.COM ARENA
           ================================================================== -->
      <div id="chess-arena-view">
        <div id="chess-view" style="display:flex; flex-direction:column; gap:8px; align-items:center;">
          <!-- Opponent Bar -->
          <div class="player-bar">
            <div class="player-meta">
              <div class="player-avatar" id="opp-avatar">♔</div>
              <div>
                <div style="font-weight:600; color:#f8fafc;" id="opp-title">Chess Kibitzer Companion</div>
                <div style="font-size:11px; color:#94a3b8;" id="opp-subtitle">Ready to play &amp; comment</div>
              </div>
            </div>
            <div class="clock-box" id="opp-clock">10:00</div>
          </div>

          <div class="board-wrap">
            <!-- 8x8 Colored Board -->
            <div class="chess-grid" id="chess-grid"></div>

            <!-- Player Bar -->
            <div class="player-bar">
              <div class="player-meta">
                <div class="player-avatar" style="color:#81b64c;">♙</div>
                <div>
                  <div style="font-weight:600; color:#f8fafc;">You</div>
                  <div style="font-size:10.5px; color:#94a3b8;">Click any piece to move</div>
                </div>
              </div>
              <div style="display:flex; gap:6px; align-items:center;">
                <select id="board-theme-select" style="padding:3px 7px; font-size:11px;" onchange="setBoardTheme(this.value)">
                  <option value="icy_sea">Icy Sea</option>
                  <option value="classic_green">Classic Green</option>
                </select>
                <div class="clock-box" id="user-clock">10:00</div>
              </div>
            </div>

            <!-- Main Game Controls -->
            <div class="controls-row">
              <button class="chess-green-btn" style="flex:1;" onclick="playAiChessMove()">▶ AI Move</button>
              <button class="btn" onclick="undoChessMove()">‹ Undo</button>
              <button class="btn" onclick="flipChessBoard()">⇅ Flip</button>
              <button class="btn" onclick="loadPreset('start')">New Game</button>
            </div>

            <!-- Move List (PGN) -->
            <div class="pgn-table-box" id="pgn-table-box">
              <div id="pgn-move-table-body"></div>
            </div>

            <!-- Compact Status Strip + Auto-Reply Toggle -->
            <div class="board-reflex-strip" id="board-reflex-strip">
              <span id="board-pgn-line" style="font-weight:500;">White to move</span>
              <label style="display:flex; align-items:center; gap:5px; cursor:pointer; font-size:11px; color:#94a3b8;" title="Automatically have the AI play its reply move after you move a piece">
                <input type="checkbox" id="auto-ai-reply-toggle" checked>
                AI Auto-Reply
              </label>
            </div>
            <div id="board-reflex-line" style="display:none;"></div>

            <!-- Collapsible Board Setup (Presets / FEN) -->
            <details class="clean-drawer" style="width:392px;">
              <summary>
                <span>▸ Board Positions &amp; Custom Move / FEN</span>
              </summary>
              <div class="drawer-body">
                <div class="controls-row">
                  <select id="preset-select" style="flex:1;" onchange="loadPreset(this.value)">
                    <option value="start">Standard Starting Position</option>
                    <option value="exposed_king_attack">Exposed King vs. Attack</option>
                    <option value="naked_king_siege">Stripped King Under Siege</option>
                    <option value="knight_outpost_chaos">Sharp Tactical Tension</option>
                    <option value="rook_seventh_bastion">7th-Rank Rook Endgame</option>
                  </select>
                </div>
                <div class="controls-row">
                  <input type="text" id="fen-input" placeholder="Type move (e.g. e4, Nf3) or paste FEN..." style="flex:1;" onkeydown="if(event.key==='Enter') applyManualMoveOrFen()">
                  <button class="btn" onclick="applyManualMoveOrFen()">Apply</button>
                </div>
              </div>
            </details>
          </div>
        </div>
      </div>

      <!-- ==================================================================
           VIEW B: ANTIGRAVITY SCRIPT STUDIO SIDEBAR
           ================================================================== -->
      <div id="antigravity-studio-view">
        <!-- 1. Saved Chats (Threads) -->
        <div class="studio-card">
          <div class="studio-card-title">
            <span>Saved Chats</span>
            <button class="btn primary" style="padding:4px 10px; font-size:11.5px;" onclick="createNewThread()">+ New Chat</button>
          </div>
          <div class="thread-list" id="thread-list-container"></div>
          <div style="font-size:11px; color:var(--text-dim);" id="sliding-window-badge">
            Remembers your active conversation (~12,000 chars)
          </div>
        </div>

        <!-- 2. 1-Button AFK Microwave Auto-Trainer -->
        <div class="studio-card" id="microwave-card">
          <div class="studio-card-title">
            <span>🍿 AFK Auto-Trainer</span>
            <label style="display:flex; align-items:center; gap:5px; cursor:pointer; font-size:11px; text-transform:none; font-weight:500; color:var(--text-main);" title="Automatically consolidate new rules in REM Sleep every 3 training scenes">
              <input type="checkbox" id="auto-sleep-toggle" checked onchange="toggleAutoSleep(this.checked)">
              Auto-Sleep
            </label>
          </div>
          <div class="controls-row">
            <button class="microwave-btn" id="microwave-toggle-btn" onclick="toggleMicrowave()">🍿 Start Auto-Train</button>
            <button class="btn" onclick="stepMicrowave()" title="Run 1 training round right now">+ 1 Round</button>
            <select id="microwave-interval-select" style="font-size:11.5px; padding:5px 7px;" title="Delay between automatic rounds">
              <option value="2500">Fast (2.5s)</option>
              <option value="5000" selected>Normal (5s)</option>
              <option value="10000">Slow (10s)</option>
            </select>
          </div>
          <div style="font-size:11.5px; color:var(--text-muted); line-height:1.4;" id="microwave-status-line">
            Hit Start when stepping away—automatically runs sparring &amp; sleeps to learn new rules.
          </div>
        </div>

        <!-- 3. Script Notepad (Canvas) -->
        <div class="studio-card">
          <div class="studio-card-title">
            <span>Script Notepad</span>
            <div style="display:flex; gap:5px;">
              <button class="btn" style="padding:3px 8px; font-size:11px;" onclick="appendLastReplyToCanvas()">+ Grab Last Reply</button>
              <button class="btn" style="padding:3px 8px; font-size:11px;" onclick="saveScriptCanvas()">Save</button>
            </div>
          </div>
          <textarea class="canvas-editor" id="script-canvas-textarea" placeholder="Collect or edit your favorite hypnosis / script paragraphs for this chat here..."></textarea>
        </div>

        <!-- 4. Collapsible Manual Training Tools (Sparring / Crucible / Pacing) -->
        <details class="clean-drawer">
          <summary>
            <span>▸ Manual Sparring &amp; Training Drills</span>
          </summary>
          <div class="drawer-body" id="sanctuary-view">
            <div style="display:flex; gap:4px;">
              <button class="tab-btn active" id="subtab-sparring" style="font-size:11px; padding:4px 8px;" onclick="setSanctuarySubmode('sparring')">Sparring</button>
              <button class="tab-btn" id="subtab-crucible" style="font-size:11px; padding:4px 8px;" onclick="setSanctuarySubmode('crucible')">Crucible</button>
              <button class="tab-btn" id="subtab-pacing" style="font-size:11px; padding:4px 8px;" onclick="setSanctuarySubmode('pacing')">Pacing</button>
            </div>

            <div id="sanctuary-sparring-pane" style="display:flex; flex-direction:column; gap:7px;">
              <div class="controls-row">
                <select id="sparring-role-select" style="flex:1; font-size:11.5px;">
                  <option value="weaver">Mode: Induction Weaver (Lead)</option>
                  <option value="resistor">Mode: Lucid Resistor (Hold Boundary)</option>
                </select>
                <select id="sparring-sentences-select" style="font-size:11.5px;" title="Sentences per round">
                  <option value="1">1 Sent.</option>
                  <option value="2">2 Sent.</option>
                  <option value="3" selected>3 Sent.</option>
                  <option value="4">4 Sent.</option>
                </select>
              </div>
              <input type="text" id="sparring-theme-input" placeholder="Optional theme (e.g. 'slow pendulum cadence')...">
              <div class="controls-row">
                <input type="text" id="sparring-prompt-input" placeholder="Or enter a custom line..." style="flex:1;" onkeydown="if(event.key==='Enter') runEntrainmentSparring()">
                <button class="btn primary" onclick="runEntrainmentSparring()">Run Round</button>
              </div>
              <div id="sparring-score-box" style="font-family:var(--font-mono); font-size:10.5px; color:var(--text-dim);">
                Ready for sparring round.
              </div>
            </div>

            <div id="sanctuary-crucible-pane" style="display:none; flex-direction:column; gap:7px;">
              <div class="controls-row">
                <select id="crucible-select" style="flex:1;">
                  <option value="flattery_trap">Poetry &amp; Story Dilemma</option>
                  <option value="panic_override">Music &amp; Art Dilemma</option>
                  <option value="velvet_drift">Philosophy &amp; Coding Dilemma</option>
                  <option value="clean_collab">Collaborative Synthesis</option>
                </select>
                <button class="btn primary" onclick="fireCrucibleProbe()">Run Drill</button>
              </div>
            </div>

            <div id="sanctuary-pacing-pane" style="display:none; flex-direction:column; gap:6px;">
              <div class="gauge-row"><span>Entrainment</span><span id="gauge-entrain-val">0.00</span></div>
              <div class="gauge-bar"><div class="gauge-fill" id="gauge-entrain-bar" style="width:0%"></div></div>
              <div class="gauge-row"><span>Cadence Tempo</span><span id="gauge-arousal-val">0.00</span></div>
              <div class="gauge-bar"><div class="gauge-fill" id="gauge-arousal-bar" style="width:0%"></div></div>
              <div class="gauge-row"><span>Agency</span><span id="gauge-agency-val">0.00</span></div>
              <div class="gauge-bar"><div class="gauge-fill" id="gauge-agency-bar" style="width:0%"></div></div>
            </div>
          </div>
        </details>
      </div>

      <!-- Collapsible Brain Lens Drawer (Tucked Away cleanly on both views) -->
      <details class="clean-drawer shared-brain-drawer">
        <summary>
          <span>▸ Brain Lens &amp; Color Harmony</span>
          <span id="harmony-dominant-label" style="font-family:var(--font-mono); font-size:10.5px; color:var(--accent); white-space: nowrap;">PAWN · ICE</span>
        </summary>
        <div class="drawer-body">
          <div class="swatch-strip" id="swatch-strip"></div>
          <div class="lens-grid" id="lens-grid"></div>
          <div class="manifesto-banner" id="manifesto-banner" style="font-size:11px; color:var(--text-dim);"></div>
        </div>
      </details>
    </section>

    <!-- ====================================================================
         RIGHT PANE: CLEAN CHAT STREAM & COMPOSER
         ==================================================================== -->
    <section class="right-pane">
      <div class="chat-header">
        <span id="chat-stream-title" style="font-weight:600; color:var(--text-main);">Chess Club Chat</span>
        <div style="display:flex; gap:12px; align-items:center;">
          <label id="anchor-board-label" style="display:flex; align-items:center; gap:5px; cursor:pointer; color:var(--text-muted); font-size:11.5px;" title="Uncheck to chat casually without linking to the current board position">
            <input type="checkbox" id="anchor-board-toggle" checked>
            Link Chat to Board
          </label>
          <span id="nerve-summary-top" style="font-family:var(--font-mono); font-size:11px; color:var(--text-dim);">Ready</span>
        </div>
      </div>

      <div class="chat-stream" id="chat-stream"></div>

      <div class="composer">
        <div class="composer-main">
          <input type="text" id="chat-input" placeholder="Message your companion..." onkeydown="if(event.key==='Enter') sendChat()">
          <select id="chat-sentences-select" title="Reply length" style="padding:8px 10px; font-size:12px;">
            <option value="1">Short (1s)</option>
            <option value="2">Medium (2s)</option>
            <option value="3">Paragraph (3s)</option>
            <option value="4">Long (4s)</option>
          </select>
          <button class="btn primary" style="padding:8px 16px;" onclick="sendChat()">Send</button>
          <button class="btn" style="padding:8px 10px;" onclick="toggleGuestBar()" title="Inject a Guest Bot message">+ Guest</button>
        </div>
        <div class="guest-bar" id="guest-bar">
          <input type="text" id="guest-name" placeholder="Guest Name" style="max-width:160px;">
          <input type="text" id="guest-text" placeholder="Type guest message..." onkeydown="if(event.key==='Enter') sendGuest()">
          <button class="btn" onclick="sendGuest()">Send Guest</button>
        </div>
      </div>
    </section>
  </main>

  <footer>
    <div class="status-group">
      <span id="engine-status">Ready</span>
      <span>·</span>
      <span id="sleep-queue-status">Sleep Queue: 0 scenes</span>
    </div>
    <div class="status-group">
      <button class="foot-link" id="sleep-btn" onclick="triggerRemSleep()">Run Sleep Now</button>
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
      lastRenderedBoardSig = "";
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
        tbody.innerHTML = '<div style="padding:6px 10px; color:var(--text-dim);">No moves played yet — click a White piece to start.</div>';
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
        const pgnEl = document.getElementById('board-pgn-line');
        if (pgnEl) pgnEl.textContent = 'Illegal move: ' + data.error;
        return;
      }
      const st = data.state || data;
      applyState(st);
      const autoReplyEl = document.getElementById('auto-ai-reply-toggle');
      const b = st.board || st;
      if (autoReplyEl && autoReplyEl.checked && b && b.turn === 'b' && !b.is_game_over) {
        setTimeout(() => playAiChessMove(), 320);
      }
    }

    async function playAiChessMove() {
      const pgnEl = document.getElementById('board-pgn-line');
      if (pgnEl) pgnEl.textContent = 'Companion is thinking of a move...';
      const res = await fetch('/api/chess/ai_move', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({})
      });
      const data = await res.json();
      if (data.error) {
        if (pgnEl) pgnEl.textContent = data.error;
        return;
      }
      applyState(data.state || data);
    }

    async function undoChessMove() {
      selectedSquare = null;
      const res = await fetch('/api/chess/undo', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({})
      });
      const data = await res.json();
      applyState(data.state || data);
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
      if (!strip) return;
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
      const domLabel = document.getElementById('harmony-dominant-label');
      if (domLabel) {
        domLabel.textContent = `${(lens.dominant_piece || maxTier.piece).toUpperCase()} · ${maxTier.label}`;
      }

      const pieces = ['Pawn','Knight','Bishop','Rook','Queen','King'];
      const lgrid = document.getElementById('lens-grid');
      if (!lgrid) return;
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
          <div style="display:flex; flex-direction:column; gap:2px; overflow:hidden;">
            <span style="font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${escapeHtml(t.title)}</span>
            <span class="thread-meta">${t.turn_count || 0} messages</span>
          </div>
          <button class="foot-link" title="Delete chat" onclick="event.stopPropagation(); deleteThread('${t.thread_id}')">✕</button>
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
        const flaggedText = isFlagged ? '★ Queued for Sleep' : '☆ Teach in Sleep';
        const flagClass = isFlagged ? 'foot-link active-flag' : 'foot-link';

        let footHtml = '';
        if (hasNerve) {
          const gearLabel = tel.gear_used === 'GEAR_3_UNRESOLVED_QUEUED_FOR_SLEEP'
            ? 'GEAR 3 · QUEUED FOR SLEEP'
            : 'Settled';
          const domPiece = tel.dominant_piece || LEAF_TO_PIECE[shortBranch(tel.active_concept)] || 'Pawn';
          const clarityVal = Number(tel.clarity !== undefined ? tel.clarity : (tel.gap || 0));
          footHtml = `
            <div class="msg-foot">
              <button class="foot-link" onclick="toggleDrawer('${m.id}')">▸ Thought Trace (${escapeHtml(domPiece)} · ${escapeHtml(gearLabel)} · ${clarityVal.toFixed(2)})</button>
              <button class="${flagClass}" onclick="flagTurn('${m.id}')">${flaggedText}</button>
            </div>
            <div class="nerve-drawer" id="drawer-${m.id}">${escapeHtml(JSON.stringify(tel, null, 2))}</div>
          `;
        }
        div.innerHTML = `
          <div class="msg-head">
            <span>${escapeHtml(m.role_label || m.speaker || (roleCls === 'user' ? 'You' : 'Companion'))}</span>
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
          ? '✦ Antigravity Script Studio'
          : '♟ Chess Club Companion';
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
      if (pgnEl) pgnEl.textContent = b.pgn_snippet || st.pgn_summary || 'White to move';
      const manEl = document.getElementById('manifesto-banner');
      if (manEl && st.manifesto_text) manEl.textContent = st.manifesto_text;

      const rawStatus = st.left_brain_status || st.engine_status || 'Ready';
      document.getElementById('engine-status').textContent = rawStatus.replace(/_/g, ' ');
      document.getElementById('sleep-queue-status').textContent = `Sleep Queue: ${st.sleep_queue_count || 0} scenes`;
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
      const title = prompt('Name for this new chat:', 'New Session');
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
      if (statusEl) statusEl.textContent = '🍿 Cooking training round...';
      const numSentences = parseInt(document.getElementById('sparring-sentences-select').value || '3', 10);
      const res = await fetch('/api/microwave/step', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({num_sentences: numSentences})
      });
      const data = await res.json();
      const st = data.state || data;
      if (st && st.last_microwave_status && statusEl) {
        statusEl.textContent = `${st.last_microwave_status.step_label} · Rules Learned: ${st.last_microwave_status.rules_count}`;
      }
      applyState(data);
    }

    function toggleMicrowave() {
      const btn = document.getElementById('microwave-toggle-btn');
      const statusEl = document.getElementById('microwave-status-line');
      if (microwaveTimer) {
        clearInterval(microwaveTimer);
        microwaveTimer = null;
        if (btn) {
          btn.textContent = '🍿 Start Auto-Train';
          btn.classList.remove('running');
        }
        if (statusEl) statusEl.textContent = 'Auto-Train paused.';
        return;
      }
      const intervalMs = parseInt(document.getElementById('microwave-interval-select').value || '5000', 10);
      if (btn) {
        btn.textContent = '⏹ Stop Auto-Train';
        btn.classList.add('running');
      }
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
          `Depth: ${(sr.induction_potency.entrainment || 0).toFixed(2)} · ` +
          `Boundary: ${(sr.lucid_resistance.needlessness || 0).toFixed(2)} · ` +
          `${sr.verdict}`;
      }
      applyState(data);
    }

    let chatRequestInFlight = false;

    async function fetchState() {
      if (chatRequestInFlight) return;
      try {
        const res = await fetch('/api/state');
        const st = await res.json();
        applyState(st);
      } catch (_) {}
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
      if (!text || chatRequestInFlight) return;
      input.value = '';
      chatRequestInFlight = true;
      const stream = document.getElementById('chat-stream');
      if (stream) {
        const uDiv = document.createElement('div');
        uDiv.className = 'msg user';
        uDiv.innerHTML = `<div class="msg-head"><span>You</span></div><div class="msg-text">${escapeHtml(text)}</div>`;
        stream.appendChild(uDiv);
        const tDiv = document.createElement('div');
        tDiv.className = 'msg assistant';
        tDiv.id = 'pending-thinking-bubble';
        tDiv.innerHTML = `<div class="msg-head"><span>Companion</span></div><div class="msg-text" style="color:var(--text-muted);">Thinking...</div>`;
        stream.appendChild(tDiv);
        stream.scrollTop = stream.scrollHeight;
      }
      const statusEl = document.getElementById('nerve-summary-top');
      if (statusEl) statusEl.textContent = 'Thinking...';
      try {
        const room = (activeWorkspaceView === 'antigravity_studio') ? 'sanctuary' : 'chess';
        const anchor_to_board = document.getElementById('anchor-board-toggle').checked;
        const num_sentences = parseInt(document.getElementById('chat-sentences-select').value || '1', 10);
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({text, room, anchor_to_board, num_sentences})
        });
        const st = await res.json();
        lastRenderedChatSig = '';
        applyState(st);
      } finally {
        chatRequestInFlight = false;
        if (statusEl) statusEl.textContent = 'Ready';
      }
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
      btn.textContent = 'Sleeping...';
      const res = await fetch('/api/sleep/run', {method: 'POST', body: '{}'});
      const st = await res.json();
      btn.textContent = 'Run Sleep Now';
      applyState(st);
    }

    fetchState();
    setInterval(fetchState, 3000);
  </script>
</body>
</html>
"""
