# Virgin Dual-Hemisphere Baseline Snapshot (`Cycle 0`)

This folder holds the **locked, SHA-256 verified snapshot of the completely untrained ("virgin") Dual-Hemisphere Wisdom Model (`sleep_cycle_index = 0`)**—before any parable/story REM sleep rules, 3-branch case-law precedents, or Left-Brain LoRA schooling are applied.

---

## Why This Folder Exists (Separation of Accountability)

Keeping a pristine `Cycle 0` snapshot in its own folder separates **Core Architecture ("Lab") accountability** from **Downstream Training / Curriculum ("School") accountability**:

1. **What Lives Inside the Virgin Baseline (`baselines/virgin_dual_hemisphere_v1/`)**:
   - **Frozen 40-Node Founding Seed (`24` founding leaves)**: Defined in [`src/lmft_lab/birth.py`](../../src/lmft_lab/birth.py).
   - **Calibrated 5-Tier Philosophical Abstraction Matrix**: Defined in [`src/lmft_lab/philosophical_abstraction_matrix.py`](../../src/lmft_lab/philosophical_abstraction_matrix.py), with **Cyan / Water / King (`philosophical/being/existence`)** anchored as **`"Wellspring of Being"`** so it never collides with the **Yellow / Rook (`philosophical/persistence/discreteness`)** **`"Ground"`** element.
   - **10 Built-in Structural Tier-2 Leaves (`virgin_ruleset.json`)**:
     - `4` Complex Feeling leaves (`austere_sanctuary`, `hollow_stillness`, `bittersweet_release`, `earned_return`)
     - `6` Shadow Pathology leaves (`psychosis`, `paranoia`, `prejudice`, `corruption`, `mania_greed`, `narcissism`)
     - **Zero** story-specific or REM-forged leaves.
   - **2 Foundational Equilibrium Rules (`virgin_ruleset.json`)**:
     - `rule_grounded_coordination` and `rule_sanctuary_equilibrium`.
     - **Zero** parable, story, or user-coached rules.
   - **Clean, Empty 3-Branch Archives (`branch_archives/`) & Contemplation Ledger (`contemplation_ledger.json`)**:
     - `0` social precedents, `0` harvested clauses, `0` ratified chords, and `0` prior scenes.

2. **How to Use It for Debugging & Time-Travel**:
   - Load and verify in Python via [`load_virgin_baseline_state()`](../../src/lmft_lab/virgin_baseline.py) or [`verify_virgin_baseline_snapshot()`](../../src/lmft_lab/virgin_baseline.py).
   - Or toggle **`Lineage: Virgin (Cycle 0)`** vs. **`Schooled (Cycle 2)`** right inside the top bar of the **LMFT Club & Sanctuary** desktop app (`Start-LMFT-Club.bat`).
   - **Diagnostic Rule**:
     - If an error or odd behavior already happens in `Virgin (Cycle 0)`, the cause is in the **Core Architecture** (tree propagation, sensor bridges, or comma-splice math).
     - If `Virgin (Cycle 0)` is clean and the issue only appears in `Cycle 1 / Cycle 2` or `club_user_state`, the cause is in **Downstream Schooling** (a bad REM-forged rule or LoRA drift).
