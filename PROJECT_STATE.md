# LMFT-Lab — Current Project State

*(Keep this file under 60 lines in plain, everyday English. Pre-reset Commit #36 history is archived in `docs/archive/PROJECT_STATE_PRE_RESET_COMMIT36.md`.)*

## Baseline Status
- Clean-slate foundation from Commit #36 (`b41e485`) + Dual-Hemisphere 3-Branch Soil Checkpoint.
- All **1,325** automated tests pass (`1,296` baseline + `11` reflection + `3` FTL sim + `8` right-brain equilibrium + `3` contemplation ledger + `1` neural bridges + `3` 3-branch archives).
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
4. **3-Branch Right-Brain Archives & Dual-Hemisphere Self-Talk Competition (`src/lmft_lab/branch_archives.py`, `src/lmft_lab/right_brain_equilibrium.py`)**:
   - **3 Dedicated Branch Archives (`outputs/branch_archives/`)**: Executive (`cultural_history.json` — tracks social history and ongoing enclosure drift), Legislative (`experiential_felt_diary.json` — stores felt atmospheres that push for new rules), and Judicial (`philosophical_lexicon.json` — stores structural Fact vs. Theory definitions and ratified chords).
   - **LLM-Free 3-Classifier Deliberation vs. Left-Brain `Proposer <-> Self-Critic` Dialogue (`outputs/dual_hemisphere_soil_eval_results.json`)**:
     - **Epoch 1 (Cold Learning on 8-Beat Parable)**: `Qwen3-4B` stayed silent (`0` calls) on 5 of 8 beats; Beat 2's false inversion was eliminated; Beat 6's quiet ongoing enclosure was caught as `CONTEST_EXCLUSION` (`75.8 ms`, `0` LLM calls) via `cultural_history.json` social-drift tracking; Beats 4 & 7 triggered genuine `1-to-2` and `2-to-1` branch deadlocks; Beat 8 triggered `Open-Thread Closure` in `Qwen3-4B`, resolving all open uncertainties in `contemplation_ledger.json` down to `0`.
     - **Epoch 2A (Consolidated 8-Beat Re-Read, `left_brain=None`)**: **`8/8` beats settled with `0` `Qwen3-4B` calls** at **`76.3 ms`** average latency.
     - **Epoch 2B (Unseen 4-Beat Out-of-Domain Story — *"The Harbor Bakehouse"*, `left_brain=None`)**: **`4/4` unseen beats settled with `0` `Qwen3-4B` calls** at **`71.5 ms`** average latency, transferring learned clauses across domains.

## Next Step
- Build **Sleep-Mode Consolidation & Offline Teacher Audits**: review `.jsonl` self-talk logs and the 3 branch archives offline with `Qwen3.5-9B` to prune redundant clauses, sharpen multi-branch chord definitions, and fine-tune the 3 neural bridge classifiers on newly discovered child drawers.
