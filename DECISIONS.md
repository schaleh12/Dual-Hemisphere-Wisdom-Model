# LMFT-Architecture — Design Decisions (Rolling 5-Decision Window)

*(Keep only the 5 most recent decisions here, under 80 lines AND 5,000 characters in plain English. Earlier decisions `D001–D056` are archived in `LMFT-Architecture/docs/archive/`.)*

## D057: Dual-Persona Adversarial Entrainment Sparring & Multi-Sentence Chaining (`entrainment_sparring.py`, `three_legged_runtime.py`, 2026-11-04)
- **Context**: Needed a self-play sparring mode where the model alternates between generating multi-sentence hypnotic induction scripts (`Induction Weaver`) and absorbing/grounding itself (`Lucid Resistor`).
- **Design Choice**: Added `SubliminalGapLeftBrain.generate_multi_sentence_script` (`1..6` sentences); built `entrainment_sparring.py` with Weaver and Resistor comma chords, un-clamped `SparringResistorFactSensors`, honest `RESISTED_LUCIDLY` vs. `SWEPT_BY_ENTRAINMENT` verdicts, and `outputs/impish_sparring_trajectories.jsonl` export.
- **Result**: Verified in `tests/test_bloodmoon_app.py`.

## D058: Sliding Context Threads, AFK Microwave Trainer, & GitHub SHA-256 Sync (`three_legged_runtime.py`, `github_auto_sync.py`, 2026-11-05)
- **Context**: Needed multi-turn conversation memory without polluting Pass-1 `Fact`, persistent session threads, hands-free REM Sleep + AFK training, and SHA-256 release verification.
- **Design Choice**: Added `format_sliding_conversation_history` (`12,000` char sliding window of clean dialogue while Pass-1 `Fact` scores only the current turn); added `threads.json` CRUD; added automatic REM Sleep at threshold (`Virgin` read-only); and built `github_auto_sync.py` (`Guardrail #4` protecting `outputs/*`).
- **Result**: Verified in `tests/test_chess_app.py` and `tests/test_bloodmoon_app.py`.

## D059: Two Standalone Applications (`LMFT-Chess.exe` & `LMFT-Bloodmoon.exe`) & Sunset of Combined App (`chess_app.py`, `bloodmoon_app.py`, `entrainment_sparring.py`, 2026-11-05)
- **Context**: Combining `Qwen3` Chess and `Impish Bloodmoon 12B` Hypnosis into one app caused UI clutter and VRAM swapping, while the two domains follow fundamentally different cognitive loops.
- **Design Choice**: Sunset and deleted `sanctuary_club_app.py`, `sanctuary_club_ui.py`, and `LMFT-Architecture.exe`; built **Standalone App #1 (`LMFT-Chess.exe`, `chess_app.py`, `chess_ui.py`)** around `python-chess` in `outputs/chess_kibitzer_state/`; and built **Standalone App #2 (`LMFT-Bloodmoon.exe`, `bloodmoon_app.py`, `bloodmoon_ui.py`)** around `Impish_Bloodmoon_12B` with one-way `feminine_to_masculine` polarity and a 3-Phase Clockwise State Machine.
- **Result**: Verified across `tests/test_bloodmoon_app.py`, `tests/test_chess_app.py`, `tests/test_virgin_baseline.py`, and `tests/test_impish_bloodmoon_adapter.py`.

## D060: `Intention -> Theory -> Confidence -> Execution` Pipeline & REM Polarity Hygiene (`chess_kibitz_bridge.py`, `sleep_consolidation.py`, 2026-11-06)
- **Context**: A 1-ply board posture sensor could not project opponent replies into `psychological/truth/theory`, and REM sleep was attaching low shadow leaves (`<= 0.45`) to `hazard_` rules.
- **Design Choice**: Wired `evaluate_candidate_moves_with_right_brain` to generate candidate impulses on `psychological/will/intention` (`Ply 1`), project top intentions through the opponent's best replies (`Ply 2`) and our best follow-up (`Ply 3`) using only the 40-node Fuzzy Tree + `BitboardIntuitionSleeve` to ground `psychological/truth/theory` and `psychological/contemplation/confidence`, and bid on `psychological/will/execution`; enforced strict virtue/shadow polarity in `sleep_consolidation.py`.
- **Result**: Verified `24/24` passing tests across `tests/test_chess_app.py` and `tests/test_sleep_consolidation.py` with zero external engine calls during play.

## D061: `Qwen/Qwen3-8B` 4-Bit Left Brain & Two-Tier REM Dream Contemplation (`chess_app.py`, `sleep_consolidation.py`, 2026-11-07)
- **Context**: `Qwen3-4B` (`2560`-d) had limited poetic depth during REM Sleep, and a single 2-sentence reflection could not separately articulate felt tension, structural geometry, and future execution for the 3-Branch Archives.
- **Design Choice**: Trained a 4-bit NF4 `Qwen/Qwen3-8B` (`hidden_size = 4096`) `SubliminalGapProjector` + `LoRALinear` checkpoint (`models/left_brain_qwen3_8b/`); upgraded `_generate_rem_sleep_self_talk` to emit a 2-sentence reflection on mid-match cat-naps (`auto_triggered=True` while active) and a 4-sentence `<|subliminal_gap|>` poem on end-of-game/manual deep sleep; and added `apply_dream_reflection_to_archives` to harvest those sentences into the Executive, Legislative, and Judicial archives.
- **Result**: Verified `5/6` positive holdout chord-swap shifts (`+0.88` net swap margin, `5/6` zero-calibrated flips, `4/6` no-gap flips, `0.0` unplugged shift, `6/6` greedy text changes, `1.41` MiniLM L2) and `37/37` passing verification tests.
