# LMFT-Architecture — Design Decisions (Rolling 5-Decision Window)

*(Keep only the 5 most recent decisions here, under 80 lines AND 5,000 characters in plain English. Earlier decisions `D001–D057` are archived in `LMFT-Architecture/docs/archive/`.)*

## D058: Sliding Context Threads, AFK Microwave Trainer, & GitHub SHA-256 Sync (`three_legged_runtime.py`, `github_auto_sync.py`, 2026-11-05)
- **Context**: Needed multi-turn conversation memory without polluting Pass-1 `Fact`, persistent session threads, hands-free REM Sleep + AFK training, and SHA-256 release verification.
- **Design Choice**: Added `format_sliding_conversation_history` (sliding window of clean dialogue while Pass-1 `Fact` scores only the current turn); added `threads.json` CRUD; added automatic REM Sleep at threshold (`Virgin` read-only); and built `github_auto_sync.py` (`Guardrail #4` protecting `outputs/*`).
- **Result**: Verified in `tests/test_chess_app.py` and `tests/test_bloodmoon_app.py`.

## D059: Two Standalone Applications (`LMFT-Chess.exe` & `LMFT-Bloodmoon.exe`) (`chess_app.py`, `bloodmoon_app.py`, 2026-11-05)
- **Context**: Combining `Qwen3` Chess and `Impish Bloodmoon 12B` Hypnosis into one app caused UI clutter and VRAM swapping across two distinct cognitive loops.
- **Design Choice**: Sunset `sanctuary_club_app.py`, `sanctuary_club_ui.py`, and `LMFT-Architecture.exe`; built **`LMFT-Chess.exe`** (`chess_app.py`, `chess_ui.py`) around `python-chess` in `outputs/chess_kibitzer_state/`; and built **`LMFT-Bloodmoon.exe`** (`bloodmoon_app.py`, `bloodmoon_ui.py`) around `Impish_Bloodmoon_12B` with one-way `feminine_to_masculine` polarity and a 3-Phase Clockwise State Machine.
- **Result**: Verified across `tests/test_bloodmoon_app.py`, `tests/test_chess_app.py`, and `tests/test_virgin_baseline.py`.

## D060: `Intention -> Theory -> Confidence -> Execution` Pipeline & REM Polarity (`chess_kibitz_bridge.py`, `sleep_consolidation.py`, 2026-11-06)
- **Context**: A 1-ply board sensor could not project opponent replies into `psychological/truth/theory`, and REM sleep was attaching low shadow leaves (`<= 0.45`) to `hazard_` rules.
- **Design Choice**: Wired `evaluate_candidate_moves_with_right_brain` to generate impulses on `psychological/will/intention` (`Ply 1`), project top intentions through `Ply 2–3` using the 40-node Fuzzy Tree + `BitboardIntuitionSleeve` to ground `psychological/truth/theory` and `psychological/contemplation/confidence`, and bid on `psychological/will/execution`; enforced strict virtue/shadow polarity in `sleep_consolidation.py`.
- **Result**: Verified `24/24` passing tests across `tests/test_chess_app.py` and `tests/test_sleep_consolidation.py`.

## D061: `Qwen/Qwen3-8B` 4-Bit Left Brain & Two-Tier REM Dream Contemplation (`chess_app.py`, `sleep_consolidation.py`, 2026-11-07)
- **Context**: `Qwen3-4B` (`2560`-d) had limited poetic depth during REM Sleep, and a 2-sentence reflection could not separately articulate felt tension, structural geometry, and future execution for the 3-Branch Archives.
- **Design Choice**: Trained a 4-bit NF4 `Qwen/Qwen3-8B` (`4096`-d) `SubliminalGapProjector` + `LoRALinear` checkpoint (`models/left_brain_qwen3_8b/`); upgraded `_generate_rem_sleep_self_talk` to emit 2-sentence cat-naps (`auto_triggered=True`) and 4-sentence deep-sleep poems; and added `apply_dream_reflection_to_archives` to harvest those sentences into the 3-Branch Archives.
- **Result**: Verified `5/6` holdout chord-swap shifts (`+0.88` margin, `0.0` unplugged shift, `6/6` text changes, `1.41` MiniLM L2) and `37/37` passing tests.

## D062: Persistent `past_kv` Streaming, Multi-Stanza Discourse & Longitudinal REM Dream Vows (`three_legged_runtime.py`, `chess_app.py`, 2026-11-07)
- **Context**: Discarding `past_kv` at every `<|subliminal_gap|>` caused `O(N^2)` prefill overhead across multi-sentence turns, and REM dreams lacked multi-stanza structure or cross-cycle vow tracking.
- **Design Choice**: Added `_PersistentTurnStreamState` so prompt prefill runs once per turn and `past_kv` persists across all `<|subliminal_gap|>` chords and `\n\n` stanza breaks; added `generate_multi_stanza_discourse` with equilibrium/diminishing-returns stopping (`max_theory_delta < 0.04`) and `compute_stanza_residual_chord`; and added bounded `rem_dream_chronicles` (`HONORED` / `BROKEN` / `NEW_VOW` in pure coordinate space).
- **Result**: Verified `40/40` passing tests across `test_three_legged_runtime.py`, `test_chess_app.py`, `test_sleep_consolidation.py`, `test_virgin_baseline.py`, and `test_right_brain_equilibrium.py`.
