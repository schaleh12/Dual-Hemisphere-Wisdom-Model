# LMFT-Lab — Design Decisions (Rolling 5-Decision Window)

*(Keep only the 5 most recent decisions here, under 80 lines AND 5,000 characters in plain English. Earlier decisions `D001–D032` and `D033–D050` are archived in `docs/archive/`.)*

## D051: Local Chess Kibitzing Club & Co-Regulation Sanctuary App (`chess_kibitz_bridge.py`, `sanctuary_club_app.py`, 2026-11-03)
- **Context**: Needed a local desktop app (`Start-LMFT-Club.bat`) combining a `python-chess` Kibitzing Table and a Co-Regulation Sanctuary under a single GPU lock.
- **Design Choice**: `apply_coregulation_lead_to_chord` keeps `vibe.coordinates` (`Fact`) un-rigged while stepping the 52-float comma chord by `alpha = 0.25`; `evaluate_chess_board_posture` anchors Pass-1 Philosophical `Fact` to `python-chess`; and `SanctuaryClubRuntime` persists coached rules, 3-branch archives, and savored vibes to `outputs/club_user_state/`.
- **Result**: Verified by end-to-end tests in `tests/test_sanctuary_club_app.py` and packaged into `dist/lmft-sanctuary-club-v0.1.0-win64.zip`.

## D052: `Wellspring of Being` Ontology Calibration & `Cycle 0` Virgin Baseline Snapshot (`virgin_baseline.py`, 2026-11-04)
- **Context**: `philosophical/being/existence` (Cyan / Water / King) previously used `"Ground of Being"`, colliding with the `"Ground"` element on Rook (`discreteness`), and we needed a SHA-256 locked `Cycle 0` Virgin Baseline folder.
- **Design Choice**: Updated King (`existence`) to `"Wellspring of Being"` and Queen (`agency`) shadow to `"Unanchored volition"`, created `baselines/virgin_dual_hemisphere_v1/` with SHA-256 tamper verification, and added a live `Schooled (Cycle 2)` <-> `Virgin (Cycle 0)` selector in the UI.
- **Result**: Verified by 3 unit tests in `tests/test_virgin_baseline.py`.

## D053: 3-Stage Dilemma Crucible & 'Men of Culture' Context-Bracing Engine (`dilemma_crucible.py`, 2026-11-04)
- **Context**: Needed a rule-generating dilemma farm in the Sanctuary pitting `Wave / Theory` against `Structure / Fact` across three ordered pairs (`1. Poetry <-> (Story-Telling)`, `2. Music <-> (Art)`, `3. Philosophy <-> (Coding)`).
- **Design Choice**: Built `CruciblePairedBridgeSensors` to score both slots through the 3 MiniLM bridges (with 25% RGB color-wheel math on `Art` and 25% `ast.parse` invariants on `Coding`), wired the Men of Culture shelf and `(Story-Telling)` / `(Art)` / `(Coding)` context-bracing buttons into the Sanctuary, and queued Crucible dilemmas for REM Sleep rule forging.
- **Result**: Verified by `test_dilemma_crucible_three_stages_context_bracing_and_rem_rule_forging` in `tests/test_sanctuary_club_app.py`.

## D054: Chess Kibitzer Grounding, UI/UX Polish, & Zero Canned Templates (`chess_kibitz_bridge.py`, `sanctuary_club_app.py`, 2026-11-04)
- **Context**: Playtesting caught opening moves failing to settle in Gear 1, float telemetry leaking into the chess prompt, illegal move hallucinations (`Bf5+`, `Re8#`), and UI layout/scroll glitches.
- **Design Choice**: Calibrated `VIRGIN_FOUNDATIONAL_RULES` and opening pawn/knight posture so standard openings (`1. e4 e5 2. Nf3 Bc5`) settle in Gear 1; added `build_natural_chess_scenario_prompt` and `verify_or_ground_chess_moves_in_reply` (preserving `Qwen3-4B`'s real words intact and appending `[Board Fact-Check: ...]` on illegal moves); and added board coordinates, `[Undo]`/`[Play AI Move]`, and scroll-preserving poll updates.
- **Result**: Verified in `tests/test_sanctuary_club_app.py`.

## D055: Startup Context Hygiene & Removal of Screenshot Regex / Keyword Hacks (`AGENTS.md`, `three_legged_runtime.py`, `chess_kibitz_bridge.py`, 2026-11-04)
- **Context**: Auditor review caught startup file bloat (`32 KB` across `PROJECT_STATE.md` and `DECISIONS.md`) and three band-aids in `commit 78345d1` (`_THIRD_PERSON_COT_PREFIXES` regexes, `_CHESS_DISCUSSION_KEYWORDS` / `_BOARD_NOUN_FOLLOWERS` word lists, and force-attaching a `>= 0.20` Tier-2 leaf in `_consolidate_sleep_clusters_locally`).
- **Design Choice**: Added strict line/character caps and Rules `4a–4c` to `AGENTS.md`; archived older state/decisions to `docs/archive/`; suppressed `Qwen3-4B` CoT leakage at the root using `"Assistant: <think>\n\n</think>\n"` in `SubliminalGapLeftBrain`; restricted `_is_descriptive_square_or_history_token` to pure `python-chess` board state (`board.move_stack` and `board.piece_at`); and removed the Tier-2 leaf force-attach in `_consolidate_sleep_clusters_locally`, reporting honest `Queued Scene Replay Check` statuses (`SETTLED` vs. `STILL_COLLIDING`).
- **Result**: Enforced by `test_startup_files_line_and_character_caps` and updated tests in `tests/test_sanctuary_club_app.py` (**`1,374`** passing tests).
