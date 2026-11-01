# LMFT-Lab — Current Project State

*(Keep this file under 60 lines in plain, everyday English. Pre-reset Commit #36 history is archived in `docs/archive/PROJECT_STATE_PRE_RESET_COMMIT36.md`.)*

## Baseline Status
- Repository reset to clean Commit #36 (`b41e485` / `6a55393`).
- All **1,308** automated tests pass (`1,296` baseline + `9` in `tests/test_reflection_classifier.py` + `3` in `tests/test_ftl_bridge_sim.py`).
- All **24 founding leaves** of the 40-node seed are now active in `src/lmft_lab/reflection_classifier.py` (6 `Cultural`, 6 `Philosophical`, 6 `Experiential`, 6 `Psychological`).

## Completed Experiments (`src/lmft_lab/reflection_classifier.py`)
1. **3-Concept (5-Sentence) Loop (`experiments/run_step1_reflection_demo.py`)**:
   - Requiring all traits in a rule to match (`min` instead of `mean`) improved Round 2 from **`2/5` to `5/5` clear matches**.
2. **Two-Tier 7-Concept (14-Sentence) Scaling Study (`experiments/run_feasibility_scaling_study.py`)**:
   - Keeping Tier 1 (the 12 founding leaves) in an unchanging prompt and scoring Tier-2 child leaves (`LearnedLeaf`) in a separate pass reached **`6/6` Teacher rules accepted**, **`7/7` top-1 (`6/7` clear matches)** on unseen test sentences, and **1,000 rules evaluated in `32.0 ms`** (`outputs/feasibility_scaling_report.json`).
3. **Tier-3 Sub-Branching & Continuous Memory (`experiments/run_continuous_tiered_demo.py`)**:
   - **Tier 2 $\rightarrow$ Tier 3 Growth**: Automatically grew `philosophical/structure/tectonic_collapse/human_intervention_in_collapse` (`altruistic_structural_resilience`), matching training (`1.00` conf, `0.70` clar) and unseen (`1.00` conf, `0.50` clar) earthquake-rescue sentences while preserving the parent Tier-2 earthquake rule (`0.80`).
   - **3-Turn Continuous Memory (`ContinuousMemoryState`)**: Turn 2 (a quiet follow-up on the pier) improved from `0.00` (confused) statelessly to **`0.96` confidence/clarity** statefully (`Retention = 0.52`, `Resonance = 0.64`), while Turn 3 (a sudden earthquake) dropped `Resonance` to `0.31` and switched immediately to the earthquake rule (`0.80`).
4. **24-Leaf Executive Hybrid vs. Raw 4B Baseline (`experiments/run_executive_hybrid_eval.py`)**:
   - Wired `Psychological -> Truth (Theory / Fact)` and a 3-mode `ExecutiveDecision` controller (`COMMIT_FAST`, `VERIFY_HYPOTHESIS`, `ESCALATE_TO_TEACHER`) that steers `Qwen3-4B`'s spoken response (`outputs/executive_hybrid_report.json`):
     - **24-Leaf LM/FT Hybrid Score**: **`3 / 4`** (correctly chose `COMMIT_FAST` on the flood rescue at `Fact=1.00`, `VERIFY_HYPOTHESIS` on the quiet pier follow-up at `Fact=0.00, Theory=0.96`, and `ESCALATE_TO_TEACHER` on the unfamiliar duplicate-notarized-forms case, which the 9B Teacher resolved into `redundant_procedural_burden`; missed 1/4 only because apartment hallway chores scored `Fact=0.60` `COMMIT_FAST` instead of `VERIFY_HYPOTHESIS`).
     - **Raw `Qwen3-4B` Alone (No Fuzzy Tree) Score**: **`0 / 4`** (lost the rescue thread on Turn 2 and hallucinated `fair_structured_cooperation` on both the pier follow-up and the bureaucratic red-tape sentence instead of escalating).
5. **Autonomous NPC Tactical Survival Simulation (`experiments/run_tactical_game_benchmark.py`)**:
   - In a 6-turn survival expedition (`outputs/tactical_game_report.json`), after gating working-memory carryover to quiet follow-up turns (`salience_estimate < 0.70`) and letting the 9B Teacher learn the action policy (`outputs/npc_learned_game_policy.json`):
     - **Rookie NPC (Before Learning, 1 rule)**: **`2 / 6` optimal moves** (`30 HP`, `70 Supplies`).
     - **Standalone `Qwen3-4B` (No Fuzzy Tree)**: **`4 / 6` optimal moves** (`85 HP`, `120 Supplies`).
     - **Veteran 24-Leaf LM/FT Hybrid NPC (After Learning, Teacher OFF)**: **`6 / 6` optimal moves** (**`125 HP`, `150 Supplies`**) at **`0.177 ms` per game tick**.
6. **Interactive FTL Bridge Simulator (`src/lmft_lab/ftl_bridge_sim.py`, `experiments/run_ftl_bridge_server.py`)**:
   - Added clock-time memory half-life decay (`decay_memory_by_seconds`, verified across 1,800 frames at 60 Hz in `tests/test_ftl_bridge_sim.py`), sub-millisecond 8-bar reactor routing (`allocate_ship_power_fuzzy`), and a live browser dashboard at `http://127.0.0.1:8765` with unscripted custom beacon input and real-time 9B Teacher coaching.

## Next Step
- Add branch-gated Tier-2/Tier-3 querying so only child leaves under active parent branches are queried as the NPC's rulebook scales beyond 20+ rules.
