# LMFT-Lab — Current Project State

*(Keep this file under 60 lines in plain, everyday English. Pre-reset Commit #36 history is archived in `docs/archive/PROJECT_STATE_PRE_RESET_COMMIT36.md`.)*

## Baseline Status
- Clean-slate foundation from Commit #36 (`b41e485` / `6a55393`) + Right-Brain Soil Checkpoint.
- All **1,322** automated tests pass (`1,296` baseline + `11` in `tests/test_reflection_classifier.py` + `3` in `tests/test_ftl_bridge_sim.py` + `8` in `tests/test_right_brain_equilibrium.py` + `3` in `tests/test_contemplation_ledger.py` + `1` in `tests/test_neural_bridge_classifiers.py`).
- Frozen 40-node seed (`24` founding leaves) is 100% intact and untouched.

## Completed Milestones
1. **Core Fuzzy Rules, Multi-Tier Sub-Drawers, & Continuous Memory (`src/lmft_lab/reflection_classifier.py`)**:
   - Fuzzy `AND` (`min`) + shoulder curves (`>= 0.65`, `<= 0.30`) reached `5/5` on the 3-concept loop and `7/7` top-1 (`6/7` clear) on unseen 7-concept scaling (`1,000` rules in `32.0 ms`, `outputs/feasibility_scaling_report.json`).
   - Verified automatic `Tier 1 -> Tier 2 -> Tier 3` child-leaf splitting and decaying `ContinuousMemoryState` (`outputs/continuous_tiered_report.json`).
2. **Head-to-Head Benchmarks vs. Standalone `Qwen3-4B`**:
   - **Executive Judgment**: **Hybrid `3/4`** vs. **Raw `Qwen3-4B` `0/4`** (`outputs/executive_hybrid_report.json`).
   - **Tactical Survival NPC**: **Veteran Hybrid `6/6` (`125 HP`, `0.177 ms`/tick)** vs. **Raw `4B` `4/6` (`85 HP`)** (`outputs/tactical_game_report.json`), plus 60 Hz FTL reactor simulator (`src/lmft_lab/ftl_bridge_sim.py`).
   - **Silent Subconscious Stream**: **`4/6`** vs. Raw `4B` **`3/6`**, skipping **`50%` (`12/24`)** of child-drawer queries (`outputs/silent_subconscious_step1_report.json`).
3. **Right-Brain Dominant Foundational Soil (`neural_bridge_classifiers.py`, `chord_memory.py`, `contemplation_ledger.py`, `right_brain_equilibrium.py`)**:
   - **3 Separate Neural Bridge Classifiers** (`models/right_brain_bridges/*`, `cross-encoder/ms-marco-MiniLM-L-12-v2`, `5.6–6.7 ms`/bridge on RTX 4080; `31–42 ms` full Tier-1+Tier-2 sensor pass vs. `~9,000 ms` LLM JSON fallback): calibrated offline by `Qwen3.5-9B` across 60 non-parable scenes with a locked 16-scene unseen holdout vault (`outputs/bridge_classifier_holdout_report.json`: unseen MAE/Pearson `r` = Do-er `0.169 / 0.686`, Thinker `0.134 / 0.830`, Feeler `0.167 / 0.752`).
   - **Tactical Hail, Felt Chords, & Chord-Rhyming Memory**: stripped dry debug tokens from `QwenLeftBrain` prompts, passing pure `FeltChord` notes (`austere_sanctuary`, `hollow_stillness`, `bittersweet_release`, `earned_return`) plus cross-scene Structural Rhymes and Chord Inversions.
   - **Two-Layer Contemplation Ledger**: archives raw self-talk passes to `.jsonl` and feeds `Qwen3-4B` a bounded `.json` ledger (`Settled Certainties` + `Active Uncertainties` with subliminal red-pen margin notes), keeping external `Fact` strictly separate from Left-Brain `Theory`.
   - **Live 8-Beat Parable Benchmark (`outputs/parable_soil_8beat_report.json`)**: eliminated flat `0.30` valence (`[0.34 -> 0.80]`), prevented Beat 8's false earthquake alarm, surfaced Beat 1/5 Rhyme+Inversion on Beat 8, and exposed clear next-step warts (Beat 6 pious irony missed as `solitary_autarky`; new child drawers need ~8s Sleep-Mode calibration examples before firing strongly).

## Next Step
- Build **Sleep-Mode Consolidation & Rule Forging**: review `.jsonl` unresolved tensions (`active_uncertainties`) offline with the 9B Teacher to forge multi-bridge poetic chord rules (e.g., *pious exclusionary hoarding*, *restorative fracture into flow*), fine-tune any newly minted child drawers in ~8 seconds, and prune redundant rules.
