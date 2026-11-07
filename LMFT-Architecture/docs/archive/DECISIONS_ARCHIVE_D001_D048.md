

## D057: Dual-Persona Adversarial Entrainment Sparring & Multi-Sentence Chaining (`entrainment_sparring.py`, `three_legged_runtime.py`, 2026-11-04)
- **Context**: Needed a self-play sparring mode where the model alternates between generating multi-sentence hypnotic induction scripts (`Induction Weaver`) and absorbing/grounding itself (`Lucid Resistor`).
- **Design Choice**: Added `SubliminalGapLeftBrain.generate_multi_sentence_script` (`1..6` sentences); built `entrainment_sparring.py` with Weaver and Resistor comma chords, un-clamped `SparringResistorFactSensors`, honest `RESISTED_LUCIDLY` vs. `SWEPT_BY_ENTRAINMENT` verdicts, and `outputs/impish_sparring_trajectories.jsonl` export.
- **Result**: Verified in `tests/test_bloodmoon_app.py`.
