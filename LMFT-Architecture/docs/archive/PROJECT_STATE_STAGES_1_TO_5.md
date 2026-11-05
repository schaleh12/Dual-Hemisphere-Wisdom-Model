# Archived Project State — Stages 1 to 5 Benchmark History

*(Archived from `PROJECT_STATE.md` on 2026-11-04 to keep startup context under 40 lines and 3,000 characters.)*

## Completed Foundation & Stages 1–5 Milestones
1. **Core Fuzzy Rules, Multi-Tier Sub-Drawers, & Continuous Memory (`src/lmft_lab/reflection_classifier.py`)**:
   - Fuzzy `AND` (`min`) + shoulder curves (`>= 0.65`, `<= 0.30`) reached `5/5` on the 3-concept loop and `7/7` top-1 (`6/7` clear) on unseen 7-concept scaling (`1,000` rules in `32.0 ms`, `outputs/feasibility_scaling_report.json`).
   - Verified automatic `Tier 1 -> Tier 2 -> Tier 3` child-leaf splitting and decaying `ContinuousMemoryState` (`outputs/continuous_tiered_report.json`).
2. **Head-to-Head Benchmarks vs. Standalone `Qwen3-4B`**:
   - **Executive Judgment**: **Hybrid `3/4`** vs. **Raw `Qwen3-4B` `0/4`** (`outputs/executive_hybrid_report.json`).
   - **Tactical Survival NPC**: **Veteran Hybrid `6/6` (`125 HP`, `0.177 ms`/tick)** vs. **Raw `4B` `4/6` (`85 HP`)** (`outputs/tactical_game_report.json`), plus 60 Hz FTL reactor simulator (`src/lmft_lab/ftl_bridge_sim.py`).
   - **Silent Subconscious Stream**: **`4/6`** vs. Raw `4B` **`3/6`**, skipping **`50%` (`12/24`)** of child-drawer queries (`outputs/silent_subconscious_step1_report.json`).
3. **Right-Brain Neural Bridges, Tactical Hail, & Two-Layer Contemplation Ledger (`neural_bridge_classifiers.py`, `chord_memory.py`, `contemplation_ledger.py`)**:
   - **3 Separate Neural Bridge Classifiers** (`models/right_brain_bridges/*`, `5.6–6.7 ms`/bridge on RTX 4080; `31–42 ms` full sensor pass): calibrated offline by `Qwen3.5-9B` across 60 non-parable scenes with a locked 16-scene unseen holdout vault (`outputs/bridge_classifier_holdout_report.json`: unseen MAE/Pearson `r` = Do-er `0.169 / 0.686`, Thinker `0.134 / 0.830`, Feeler `0.167 / 0.752`).
   - **Tactical Hail, Felt Chords, & Contemplation Ledger**: passes pure `FeltChord` atmospheres and cross-scene Structural Rhymes / Inversions, and separates `.jsonl` raw self-talk from the bounded `.json` ledger.
4. **Un-Rigged 3-Branch Archives & Offline 9B Sleep-Mode Consolidation (`branch_archives.py`, `sleep_consolidation.py`)**:
   - **3 Dedicated Branch Archives (`outputs/branch_archives/`)**: Executive (`cultural_history.json`), Legislative (`experiential_felt_diary.json`), and Judicial (`philosophical_lexicon.json`).
   - **Pre-Sleep vs. Post-Sleep Live GPU Evaluation (`outputs/dual_hemisphere_soil_eval_results.json`)**:
     - Epoch 1 settled `4/8`; Pre-Sleep settled `5/8` on the Parable and `1/4` on unseen Harbor Bakehouse; Post-Sleep consolidated (`0` LLM calls) improved to `7/8` (`65.9 ms`/beat) on the Parable and `4/4` (`46.7 ms`/beat) on unseen Harbor Bakehouse.
5. **Stages 1–5 of `<|subliminal_gap|>` Three-Legged Race (`subliminal_gap_splice.py`, `gap_trajectory_schema.py`, `gap_trajectory_generator.py`, `train_left_brain_lora.py`, `three_legged_runtime.py`, `rem_sleep_cycle.py`)**:
   - **Stage 1**: Compiles the Right Brain's non-verbal state into 4 disjoint Departmental Vectors (`doer_vector` 17 floats, `thinker_vector` 16 floats, `feeler_vector` 16 floats, `cadence_vector` 3 floats) and splices `K = 4` virtual token embeddings into `<|subliminal_gap|>`.
   - **Stage 2**: Generated `outputs/gap_trajectories_train.jsonl` (`17/18` accepted) and `outputs/gap_trajectories_holdout.jsonl` (`6/6` unseen seeds accepted).
   - **Stage 3**: Trained `Qwen/Qwen3-4B` (`bfloat16`, `8.127 GB` peak VRAM) with `LoRALinear` + `SubliminalGapProjector` (`models/left_brain_lora/`): Check A `5/6` positive shift (`mean_net_chord_swap_margin = +0.7775`, `3/6` zero-cal flip, `4/6` no-gap flip), Check B `0.0`, Check C `5/6` greedy flip.
   - **Stage 4**: Built the 3-Gear Closed-Loop Runtime (`run_three_legged_turn` + `SubliminalGapLeftBrain`) with live mid-sentence comma scoring (`D050`).
   - **Stage 5**: Executed the 4-Phase Daytime -> REM Sleep -> Next-Day loop (`sleep_cycle_index = 1 -> 2`, `models/left_brain_lora_cycle2/`), raising Daytime Parable settlement from `5/8` to `7/8`, Unseen Bakehouse from `3/4` to `4/4`, and unseen holdout `zero_cal_flip` from `3/6` to `4/6` (`mean_swap_margin: +0.7775 -> +0.8646`).
