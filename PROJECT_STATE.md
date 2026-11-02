# LMFT-Lab — Current Project State

*(Keep this file under 60 lines in plain, everyday English. Pre-reset Commit #36 history is archived in `docs/archive/PROJECT_STATE_PRE_RESET_COMMIT36.md`.)*

## Baseline Status
- Clean-slate foundation from Commit #36 (`b41e485`) + Un-Rigged Dual-Hemisphere 3-Branch Soil Checkpoint.
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
   - **3 Dedicated Branch Archives (`outputs/branch_archives/`)**: Executive (`cultural_history.json` — tracks social history and vetoes innocent open-access clauses during ongoing enclosure without clamping sensor coordinates), Legislative (`experiential_felt_diary.json`), and Judicial (`philosophical_lexicon.json`). Only settled scenes are harvested into reusable clauses.
   - **Un-Rigged Live Evaluation (`outputs/dual_hemisphere_soil_eval_results.json`)**:
     - **Epoch 1 (Cold Learning on 8-Beat Parable)**: Settled **`4/8` (`[1, 2, 3, 7]`)**; Beat 2's false inversion was eliminated; Beat 6 kept its raw `Fair = 0.51` untouched while `cultural_history.json` caught `Social Drift = True`, causing the Executive branch to veto (`Exec=False, Leg=True, Jud=True`, `Do-er Stance = CONTEST_EXCLUSION`) rather than misclassifying Beat 6 as `solitary_autarky`; Beats 4, 5, 6, 8 remained honestly `UNRESOLVED_TENSION` awaiting 9B Teacher rule forging.
     - **Epoch 2A (Consolidated 8-Beat Re-Read, `left_brain=None`)**: Settled **`5/8` (`[1, 2, 3, 7, 8]`)** with **`0` `Qwen3-4B` calls** at **`58.8 ms`** mean latency (including Beat 8 recombining Beat 1 + Beat 7 archive clauses), while Beats 4–6 stayed honestly unresolved (`CONTEST_EXCLUSION`) because no Teacher has taught an exclusionary rule yet.
     - **Epoch 2B (Unseen 4-Beat Harbor Bakehouse, `left_brain=None`)**: Settled **`0/4`** (`61.8 ms`/beat; Unseen Beats 1 & 4 ratified 3-to-0 across the 3 branches but tied on Tier-1 founding rules, and Unseen Beat 3 caught `Social Drift = True`), exposing the exact need for Sleep-Mode Teacher rule forging.

## Next Step
- Build **Sleep-Mode Consolidation & Offline Teacher Rule Forging**: have the 9B Teacher (`Qwen3.5-9B`) review `contemplation_ledger.json`'s unresolved scenes (Beats 4, 5, 6, 8) and the 3 branch archives offline to forge validated multi-branch rules (such as *exclusionary hoarding* and *restorative fracture into flow*) so Beats 4–6 and the unseen Harbor Bakehouse settle cleanly without shortcuts.
