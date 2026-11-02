# LMFT-Lab — Current Project State

*(Keep this file under 60 lines in plain, everyday English. Pre-reset Commit #36 history is archived in `docs/archive/PROJECT_STATE_PRE_RESET_COMMIT36.md`.)*

## Baseline Status
- Clean-slate foundation from Commit #36 (`b41e485`) + Un-Rigged Dual-Hemisphere 3-Branch Soil + Offline 9B Sleep-Mode Consolidation + Stages 1, 2 (`2A`/`2B`), & 3 (`<|subliminal_gap|>` Embedding Splice, Offline Contrastive Dataset, and `Qwen3-4B` LoRA + Corpus Callosum Training).
- All **1,345** automated tests pass (`1,296` baseline + `11` reflection + `3` FTL sim + `8` right-brain equilibrium + `3` contemplation ledger + `1` neural bridges + `3` 3-branch archives + `3` sleep consolidation + `3` subliminal gap splice + `5` gap trajectory schema + `5` gap trajectory generator + `4` left-brain LoRA).
- Frozen 40-node seed (`24` founding leaves, with leaf #18 updated from `Retention` to `Entrainment`) is 100% intact.

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
   - Compiles the Right Brain's non-verbal state into 4 strictly disjoint Departmental Vectors (`doer_vector` 17 floats, `thinker_vector` 16 floats, `feeler_vector` 16 floats, `cadence_vector` 3 floats, using `pool_departmental_child_chord` so sibling child leaves under the same parent branch like `austere_sanctuary`, `hollow_stillness`, and `bittersweet_release` stay 100% distinct without ever changing the department vector sizes) and splices `K = 4` compartmentalized virtual token embeddings (`SubliminalGapProjector`) directly into `<|subliminal_gap|>` slots in `inputs_embeds`.

6. **Stages 2 & 3 of `<|subliminal_gap|>` Three-Legged Race (`gap_trajectory_schema.py`, `gap_trajectory_generator.py`, `train_left_brain_lora.py`)**:
   - **Sub-Step 2A & 2B (`outputs/gap_trajectory_generation_report.json`)**: Generated `outputs/gap_trajectories_train.jsonl` (`17/18` accepted, `mean L2 = 1.9055`, `1` seed `train-tone-01` openly rejected) and `outputs/gap_trajectories_holdout.jsonl` (`6/6` unseen seeds accepted, `mean L2 = 2.1851`) with `Qwen3.5-9B` and the 3 Right-Brain MiniLM Bridge Classifiers.
   - **Sub-Step 3 (`src/lmft_lab/train_left_brain_lora.py`, `outputs/left_brain_lora_eval_report.json`, `models/left_brain_lora/`)**: Trained `Qwen/Qwen3-4B` (`bfloat16`, `8.127 GB` peak VRAM) with `LoRALinear` (`72` layers) + `SubliminalGapProjector` (`sleep_cycle_index = 1`) and `-100` pre-gap label masking. On the 6 unseen holdout pairs: **Check A** achieved `5/6` positive chord-swap log-prob shift (`mean_net_chord_swap_margin = +0.7775` vs. `+0.0435` pre-train; `0/6` raw un-normalized `both_branches_flipped` due to base LLM sentence-length/prior differences), **Check B** (`max_unplugged_chord_swap_shift`) was `0.000000`, and **Check C** changed greedy post-gap continuations on `5/6` unseen holdout prompts (`mean_generated_sensor_l2 = 0.9711`) while curing `Epoch 0`'s untrained mid-sentence Mandarin glitch on `holdout-tone-02`.

## Next Step
- **Stage 4 (Closed-Loop Dual-Hemisphere Runtime Integration & Sentence-Prior Length Normalization)** ([`docs/PROJECT_PIPELINE_AND_AUDIT_ROADMAP.md`](docs/PROJECT_PIPELINE_AND_AUDIT_ROADMAP.md)): wire the trained `models/left_brain_lora/` checkpoint into live Dual-Hemisphere equilibrium and Sleep-Cycle maturation.
