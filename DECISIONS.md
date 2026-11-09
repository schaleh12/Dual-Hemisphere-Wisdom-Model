# LMFT-Architecture — Design Decisions (Rolling 5-Decision Window)

*(Keep only the 5 most recent decisions here, under 80 lines AND 5,000 characters in plain English. Earlier decisions `D001–D063` are archived in `LMFT-Architecture/docs/archive/`.)*

## D064: Six Core Evolution Upgrades (`reflection_classifier.py`, `right_brain_equilibrium.py`, `three_legged_runtime.py`, 2026-11-07)
- **Context**: Matching a `hazard_` rule boosted `Execution` instead of inhibiting it; colliding rules tied on `min_similarity`; `Experiential` sensors validated 3 of 6 leaves; and general dialogue lacked pre-gap rollout.
- **Design Choice**: Added `polarity` (`"attractive"` vs. `"repulsive"`) on `FuzzyConceptRule`, soft-min tie-breaking (`0.85 * min_sim + 0.15 * mean_sim`), `OPTIONAL_EXPERIENTIAL_SENSOR_LEAVES`, and `select_pre_gap_branch_chord_rollout`.
- **Result**: Verified `57/57` passing tests across core and app suites with Cycle-0 Virgin SHA-256 locks intact.

## D065: 24-Circuit Weight-Space Sculpting, Homeostatic Decay & Founding Cluster Naming (`chess_kibitz_bridge.py`, `sleep_consolidation.py`, 2026-11-07)
- **Context**: `BitboardIntuitionSleeve` saturated `order`/`chaos` at `±2.5` while 6 bitboard circuits stayed frozen, and `_cluster_key_for_unresolved` picked Tier-2 `hazard_*` leaves to produce recursive `hazard_high_hazard_` names.
- **Design Choice**: Threaded `played_bitboard_features` and `reference_bitboard_features` into `distill_post_mortem_scene`, restricted `_cluster_key_for_unresolved` to `FOUNDATIONAL_LEAVES`, and stripped recursive prefixes in `canonical_cluster_concept_name`.
- **Result**: Verified `56/56` passing tests across `test_chess_app.py`, `test_sleep_consolidation.py`, and `test_virgin_baseline.py`.

## D066: Core Weight-Space Sculpting (`subliminal_gap_splice.py`, `right_brain_equilibrium.py`, `sleep_consolidation.py`, 2026-11-07)
- **Context**: `entrain_existing_cluster_rule` ratcheted `high` above `0.90` and accumulated `4+` target leaves, `LearnedLeaf.anchor_weights` stayed frozen after birth, and `RIGHT_BRAIN_WILL_EDGES` never adapted during REM Sleep.
- **Design Choice**: Updated `entrain_existing_cluster_rule` with polarity preservation, top-3 leaf pruning, and Homeostatic Threshold Decay (`high <= 0.90`, `low <= 0.75`); added `sculpt_learned_leaf_anchor_weights`; and added `sculpt_will_edges_during_sleep`.
- **Result**: Normalized 7 cluster rules within homeostatic bounds and verified all core, chess, and Virgin baseline tests pass.

## D067: Semantic Birth Anchoring, Symmetric Updates, Unstamped Coordinates & `will_edges` Scoring (`subliminal_gap_splice.py`, `sleep_consolidation.py`, `chess_app.py`, `chess_kibitz_bridge.py`, 2026-11-08)
- **Context**: Audit and Red-Team review found 5 installer bugs, one-way ratchets, `0.25 / 0.53` coordinate stamps, a rigged `calm_baseline_score` gate, random static at `Cycle 0` birth, REM vow theater, and un-wired `will_edges` in move scoring.
- **Design Choice**: Fixed the installer bugs, replaced one-way ratchets with symmetric contrastive updates, removed `0.25 / 0.53` coordinate stamps, replaced `calm_baseline_score` with `calm_deviation`, cut Dream Vow / `apply_dream_reflection_to_archives` theater, wired `self.will_edges` into `evaluate_candidate_moves_with_right_brain`, and retrained all 3 checkpoints with `anchor_to_llm_embeddings` (`256`-dim projector).
- **Result**: Raw holdout flips reached `6/6` (`+6.4123` margin) on `Qwen/Qwen3-8B`, `5/6` (`6/6` calibrated, `+3.7225` margin) on `Impish_Bloodmoon_12B`, and `5/6` (`6/6` calibrated, `+4.7017` margin) on `Qwen/Qwen3-4B`.

## D068: Tier 2–4 Sub-Feature Leaf Genesis, Ply-5 Detective Walk & Two-Way Gear-2 Deliberation (`reflection_classifier.py`, `chess_kibitz_bridge.py`, `sleep_consolidation.py`, `chess_app.py`, 2026-11-09)
- **Context**: A 128-game live run showed all 29 learned rules in `club_ruleset.json` were Tier-1 rules on the 12 Cultural + Philosophical leaves, while 82 Tier-2 `LearnedLeaf` nodes went unused because multi-ply traps had nearly identical Tier-1 summary coordinates, and Gear-2 deliberation lacked two-way `Theory -> Intention -> Execution` suspect reweighting.
- **Design Choice**: Added a 16-float Sub-Sensor Bank (`SUB_SENSOR_FEATURE_NAMES`, `compute_move_sub_sensors`), Tier 2–4 `LearnedLeaf` genesis with parent-gated (`parent >= 0.50`) sub-sensor projections and `Tier 4 > Tier 3 > Tier 2 > Tier 1` inhibition, the `Ply 1 -> Ply 3 -> Ply 5` Intention Detective Walk (`_run_detective_ply5_walk_on_suspects`) on the top 3 suspects, and two-way Gear-2 priority steering (`reweight_suspects_with_left_brain_theory`) through `compute_will_circuit_activations`.
- **Result**: Verified `48/48` passing tests across `test_sleep_consolidation.py`, `test_chess_app.py`, `test_subliminal_gap_splice.py`, `test_right_brain_equilibrium.py`, and `test_virgin_baseline.py`.
