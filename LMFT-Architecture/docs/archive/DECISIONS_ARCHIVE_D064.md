

## D064: Six Core Evolution Upgrades (`reflection_classifier.py`, `right_brain_equilibrium.py`, `three_legged_runtime.py`, 2026-11-07)
- **Context**: Matching a `hazard_` rule boosted `Execution` instead of inhibiting it; colliding rules tied on `min_similarity`; `Experiential` sensors validated 3 of 6 leaves; and general dialogue lacked pre-gap rollout.
- **Design Choice**: Added `polarity` (`"attractive"` vs. `"repulsive"`) on `FuzzyConceptRule`, soft-min tie-breaking (`0.85 * min_sim + 0.15 * mean_sim`), `OPTIONAL_EXPERIENTIAL_SENSOR_LEAVES`, and `select_pre_gap_branch_chord_rollout`.
- **Result**: Verified `57/57` passing tests across core and app suites with Cycle-0 Virgin SHA-256 locks intact.
