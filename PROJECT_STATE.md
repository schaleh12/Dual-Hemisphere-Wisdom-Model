# LMFT-Lab — Current Project State

*(Keep this file under 60 lines in plain, everyday English. Pre-reset Commit #36 history is archived in `docs/archive/PROJECT_STATE_PRE_RESET_COMMIT36.md`.)*

## Baseline Status
- Clean-slate foundation from Commit #36 (`b41e485` / `6a55393`) + Checkpoint `581baad`.
- All **1,318** automated tests pass (`1,296` baseline + `11` in `tests/test_reflection_classifier.py` + `3` in `tests/test_ftl_bridge_sim.py` + `8` in `tests/test_right_brain_equilibrium.py`).
- Frozen 40-node seed (`24` founding leaves) is 100% intact and untouched.

## Completed Milestones
1. **Core Fuzzy Rules, Multi-Tier Sub-Drawers, & Continuous Memory (`src/lmft_lab/reflection_classifier.py`)**:
   - Fuzzy `AND` (`min`) + shoulder curves (`>= 0.65`, `<= 0.30`) reached `5/5` on the 3-concept loop and `7/7` top-1 (`6/7` clear) on unseen 7-concept scaling (`1,000` rules in `32.0 ms`, `outputs/feasibility_scaling_report.json`).
   - Verified automatic `Tier 1 -> Tier 2 -> Tier 3` child-leaf splitting and decaying `ContinuousMemoryState` (`0.00 -> 0.96` follow-up carry; instant `0.31` resonance drop on sudden disruption, `outputs/continuous_tiered_report.json`).
2. **Head-to-Head Benchmarks vs. Standalone `Qwen3-4B`**:
   - **Executive Judgment**: **Hybrid `3/4`** vs. **Raw `Qwen3-4B` `0/4`** (`outputs/executive_hybrid_report.json`).
   - **Tactical Survival NPC**: **Veteran Hybrid `6/6` (`125 HP`, `0.177 ms`/tick)** vs. **Raw `4B` `4/6` (`85 HP`)** vs. **Rookie `2/6` (`30 HP`)** (`outputs/tactical_game_report.json`), plus 60 Hz clock-decay FTL reactor simulator (`src/lmft_lab/ftl_bridge_sim.py`).
   - **Silent Subconscious Stream**: Open Run #1 flop (`2/6`) fixed via shoulder curves & memory gating to **`4/6`** vs. Raw `4B` **`3/6`**, skipping **`50%` (`12/24`)** of child-drawer queries (`outputs/silent_subconscious_step1_report.json`).
3. **Right-Brain Dominant 3-Bridge Equilibrium, Self-Registry, & `Will -> Theory` Circuit (`src/lmft_lab/right_brain_equilibrium.py`)**:
   - Decoupled classification into `ThreeBridgeSensors` (competing **Do-er** `Will-Cultural`, **Thinker** `Truth-Philosophical`, and **Feeler** `Contemplation-Experiential` bids—with full Tier-2/Tier-3 complex-feeling sub-drawer growth unlocked under `experiential/*` for multi-bridge poetic chords), added `InternalSelfRegistry` separating `EXTERNAL_WORLD` (`Fact`) from `INTERNAL_LEFT_BRAIN_THEORY` (`Theory`), and routed outbound `Will` tugs into plain-English `QwenLeftBrain` `Theory` proposals.
   - Live 4-beat parable smoke check (`outputs/equilibrium_diary_smoke_report.json`): kept external `Fact` uncorrupted by diary thoughts, eliminated Beat 1's staircase & Beat 8's false earthquake alarm, and confirmed `Qwen3-4B`'s flat `Valence = 0.30` fallback scoring needs 3 dedicated neural classifiers.

## Next Step
- Replace `make_temporary_llm_bridge_sensors` with 3 fast (`~15 ms`), lightweight neural bridge classifiers trained on 9B-Teacher-calibrated scenarios (with a locked unseen holdout test vault), plus Sleep-Mode rule pruning so the Right Brain senses in milliseconds while `Qwen3-4B` stays pure Left Brain.
