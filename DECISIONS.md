# LMFT-Architecture — Design Decisions (Rolling 5-Decision Window)

*(Keep only the 5 most recent decisions here, under 80 lines AND 5,000 characters in plain English. Earlier decisions `D001–D055` are archived in `LMFT-Architecture/docs/archive/`.)*

## D056: Impish Bloodmoon 12B Adapter, Isolated Virgin Clone, & Patreon Bundle Encryption (`impish_bloodmoon_adapter.py`, 2026-11-04)
- **Context**: Needed an isolated `Cycle 0` Virgin Baseline clone paired with `SicariusSicariiStuff/Impish_Bloodmoon_12B` (`hidden_size = 5120`) in 4-bit NF4 on a 16 GB RTX 4080, strict Left-Brain lineage separation, and Patreon `.enc` bundle encryption.
- **Design Choice**: Added 4-bit `BitsAndBytesConfig` support to `SubliminalGapLeftBrain`; built `clone_virgin_baseline_for_impish_bloodmoon` (`outputs/impish_bloodmoon_state/`), `train_impish_bloodmoon_callosum_and_lora` (`models/left_brain_impish_bloodmoon/`), and PBKDF2-HMAC-SHA256 `.enc` encryption.
- **Result**: Trained 6 epochs on RTX 4080 (`6/6` positive holdout chord-swap shifts, `4/6` no-gap calibrated flips, `0.0` unplugged shift, `6/6` greedy text changes) and verified in `tests/test_impish_bloodmoon_adapter.py`.

## D057: Dual-Persona Adversarial Entrainment Sparring & Multi-Sentence Chaining (`entrainment_sparring.py`, `three_legged_runtime.py`, 2026-11-04)
- **Context**: Needed a self-play sparring mode where the model alternates between generating multi-sentence hypnotic induction scripts (`Induction Weaver`) and absorbing/grounding itself (`Lucid Resistor`).
- **Design Choice**: Added `SubliminalGapLeftBrain.generate_multi_sentence_script` (`1..6` sentences); built `entrainment_sparring.py` with Weaver and Resistor comma chords, un-clamped `SparringResistorFactSensors`, honest `RESISTED_LUCIDLY` vs. `SWEPT_BY_ENTRAINMENT` verdicts, and `outputs/impish_sparring_trajectories.jsonl` export.
- **Result**: Verified in `tests/test_bloodmoon_app.py`.

## D058: Sliding Context Threads, AFK Microwave Trainer, & GitHub SHA-256 Sync (`three_legged_runtime.py`, `github_auto_sync.py`, 2026-11-05)
- **Context**: Needed multi-turn conversation memory without polluting Pass-1 `Fact`, persistent session threads, hands-free REM Sleep + AFK training, and SHA-256 release verification.
- **Design Choice**: Added `format_sliding_conversation_history` (`12,000` char sliding window of clean dialogue while Pass-1 `Fact` scores only the current turn); added `threads.json` CRUD; added automatic REM Sleep at threshold (`Virgin` read-only); and built `github_auto_sync.py` (`Guardrail #4` protecting `outputs/*`).
- **Result**: Verified in `tests/test_chess_app.py` and `tests/test_bloodmoon_app.py`.

## D059: Two Standalone Applications (`LMFT-Chess.exe` & `LMFT-Bloodmoon.exe`) & Sunset of Combined App (`chess_app.py`, `bloodmoon_app.py`, `entrainment_sparring.py`, 2026-11-05)
- **Context**: Combining `Qwen3-4B` Chess and `Impish Bloodmoon 12B` Hypnosis into one app caused UI clutter and VRAM swapping, while the two domains follow fundamentally different cognitive loops.
- **Design Choice**: Sunset and deleted `sanctuary_club_app.py`, `sanctuary_club_ui.py`, and `LMFT-Architecture.exe`; built **Standalone App #1 (`LMFT-Chess.exe`, `chess_app.py`, `chess_ui.py`)** around `Qwen3-4B` + `python-chess` in `outputs/chess_kibitzer_state/`; and built **Standalone App #2 (`LMFT-Bloodmoon.exe`, `bloodmoon_app.py`, `bloodmoon_ui.py`)** around `Impish_Bloodmoon_12B` with one-way `feminine_to_masculine` polarity and a 3-Phase Clockwise State Machine.
- **Result**: Verified across `tests/test_bloodmoon_app.py`, `tests/test_chess_app.py`, `tests/test_virgin_baseline.py`, and `tests/test_impish_bloodmoon_adapter.py`.

## D060: Bitboard `Fact`/`Theory` Sensor, `BitboardIntuitionSleeve` (`Confidence`), & `Execution` Action Bids (`chess_kibitz_bridge.py`, `chess_app.py`, 2026-11-06)
- **Context**: Hand-tuned `tactical_score` bonuses and check-spam incentives distorted move selection during overnight self-play instead of letting the 40-node Fuzzy Tree govern moves.
- **Design Choice**: Stripped `375` lines of competing heuristics and legacy Crucible/Guest/Canvas bloatware; wired 64-square `python-chess` bitboards directly into `psychological/truth/fact` and `psychological/truth/theory`; built single-pass `BitboardIntuitionSleeve` (`bitboard_sleeve_weights.json`) feeding `psychological/contemplation/confidence`; evaluated all legal moves as competing Action Bids on `psychological/will/execution` (`RIGHT_BRAIN_WILL_EDGES`); and distilled post-mortem Stockfish critiques during REM Sleep.
- **Result**: Verified `9/9` passing tests in `tests/test_chess_app.py` with a 100% pure pre-move air-gap.
