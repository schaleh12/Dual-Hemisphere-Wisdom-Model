# LMFT-Lab — Current Project State

*(Keep this file under 60 lines in plain, everyday English. Pre-reset Commit #36 history is archived in `docs/archive/PROJECT_STATE_PRE_RESET_COMMIT36.md`.)*

## Baseline Status
- Clean-slate foundation from Commit #36 (`b41e485`) + Un-Rigged Dual-Hemisphere 3-Branch Soil + Offline 9B Sleep-Mode Consolidation + Stage 1 `<|subliminal_gap|>` Embedding Splice.
- All **1,331** automated tests pass (`1,296` baseline + `11` reflection + `3` FTL sim + `8` right-brain equilibrium + `3` contemplation ledger + `1` neural bridges + `3` 3-branch archives + `3` sleep consolidation + `3` subliminal gap splice).
- Frozen 40-node seed (`24` founding leaves) is 100% intact and untouched.

## Completed Milestones
1. **Core Fuzzy Rules, Multi-Tier Sub-Drawers, & Continuous Memory (`src/lmft_lab/reflection_classifier.py`)**:
   - Fuzzy `AND` (`min`) + shoulder curves (`>= 0.65`, `<= 0.30`) reached `5/5` on the 3-concept loop and `7/7` top-1 (`6/7` clear) on unseen 7-concept scaling (`1,000` rules in `32.0 ms`, `outputs/feasibility_scaling_report.json`).
   - Verified automatic `Tier 1 -> Tier 2 -> Tier 3` child-leaf splitting and decaying `ContinuousMemoryState` (`outputs/continuous_tiered_report.json`).
2. **Head-to-Head Benchmarks vs. Standalone `Qwen3-4B`**:
   - **Executive Judgment**: **Hybrid `3/4`** vs. **Raw `Qwen3-4B` `0/4`** (`outputs/executive_hybrid_report.json`).
   - **Tactical Survival NPC**: **Veteran Hybrid `6/6` (`125 HP`, `0.177 ms`/tick)** vs. **Raw `4B` `4/6` (`85 HP`)** (`outputs/tactical_game_report.json`), plus 60 Hz FTL reactor simulator (`src/lmft_lab/ftl_bridge_sim.py`).
   - **Silent Subconscious Stream**: **`4/6`** vs. Raw `4B` **`3/6`**, skipping **`50%` (`12/24`)** of child-drawer queries (`outputs/silent_subconscious_step1_report.json`).
3. **Right-Brain Neural Bridges, Tactical Hail, & Two-Layer Contemplation Ledger (`neural_bridge_classifiers.py`, `chord_memory.py`, `contemplation_ledger.py`)**:
   - **3 Separate Neural Bridge Classifiers** (`models/right_brain_bridges/*`, `5.6–6.7 ms`/bridge on RTX 4080; `31–42 ms` full sensor pass): calibrated offline by `Qwen3.5-9B` across 60 non-parable scenes with a locked 16-scene unseen holdout vault (`outputs/bridge_classifier_holdout_report.json`: unseen MAE/Pearson `r` = Do-er `0.169 / 0.686`, Thinker `0.134 / 0.830`, Feeler `0.167 / 0.752`).
   - **Tactical Hail, Felt Chords, & Contemplation Ledger**: passes pure `FeltChord` atmospheres and cross-scene Structural Rhymes / Inversions (requiring true polarity flips `<= 0.38` vs. `>= 0.55`), and separates `.jsonl` raw self-talk from the bounded `.json` ledger.
4. **Un-Rigged 3-Branch Archives & Offline 9B Sleep-Mode Consolidation (`branch_archives.py`, `sleep_consolidation.py`)**:
   - **3 Dedicated Branch Archives (`outputs/branch_archives/`)**: Executive (`cultural_history.json`), Legislative (`experiential_felt_diary.json`), and Judicial (`philosophical_lexicon.json`). Zero coordinate clamping, zero rule zeroing, and unsettled scenes are never harvested as ratified clauses.
   - **Pre-Sleep vs. Post-Sleep Live GPU Evaluation (`outputs/dual_hemisphere_soil_eval_results.json`)**:
     - **Epoch 1 (Cold Learning)**: Settled **`4/8` (`[1, 2, 3, 7]`)**; Beat 6 kept raw `Fair = 0.51` untouched while `cultural_history.json` caught `Social Drift = True` and the Executive branch vetoed (`[Exec=False, Leg=True, Jud=True]`, `CONTEST_EXCLUSION`).
     - **Pre-Sleep Consolidated (`0` LLM calls)**: Settled **`5/8` (`[1, 2, 3, 7, 8]`, `71.1 ms`)** on the Parable and **`1/4` (`[4]`, `53.9 ms`)** on the unseen Harbor Bakehouse (where unrefined `solitary_autarky` misfired on high chaos).
     - **Offline 9B Sleep Consolidation (`Qwen3.5-9B-Q4_K_M.gguf`, `outputs/sleep_consolidated_ruleset.json`)**: Refined `solitary_autarky` in JSON (`chaos: 0.10, needlessness: 0.75`) and forged two domain-general rules (`exclusionary_enclosure_cluster` and `shared_access_to_renewal`) plus 6 branch clauses with zero story-specific words.
     - **Post-Sleep Consolidated (`0` LLM calls)**: Improved to **`7/8` (`[1, 2, 3, 4, 5, 7, 8]`, `65.9 ms`/beat)** on the Parable (holding Beat 6 honestly in `CONTEST_EXCLUSION` via `social_drift`) and **`4/4` (`[1, 2, 3, 4]`, `46.7 ms`/beat)** on the unseen Harbor Bakehouse.

5. **Stage 1 of 'Three-Legged Race' `<|subliminal_gap|>` Embedding Splice (`src/lmft_lab/subliminal_gap_splice.py`, `docs/SUBLIMINAL_GAP_SPLICE_BLUEPRINT.md`)**:
   - Compiles the Right Brain's non-verbal state into a 32-float `SubliminalChordVector` (24 founding leaves + 4 complex feeling chord notes + 4 cadence/rupture signals including `cadence_rupture_drive`) and splices `K = 4` projected virtual token embeddings (`SubliminalGapProjector`) directly into `<|subliminal_gap|>` slots in `inputs_embeds`.

## Next Step
- **Stage 2 & 3 of `<|subliminal_gap|>` Three-Legged Race**: generate paired `<|subliminal_gap|>` thought trajectories offline with the 9B Teacher (`Qwen3.5-9B`) and train `SubliminalGapProjector` + LoRA on `Qwen3-4B` on the RTX 4080.
