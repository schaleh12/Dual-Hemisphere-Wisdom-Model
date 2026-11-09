# LMFT-Architecture — Design Decisions (Rolling 5-Decision Window)

*(Keep only the 5 most recent decisions here, under 80 lines AND 5,000 characters in plain English. Earlier decisions `D001–D062` are archived in `LMFT-Architecture/docs/archive/`.)*

## D063: Ghost-Chord Masking, Stanza Synthesis, Pass-3 Pivot & Tier-2 Leaf Birth (`three_legged_runtime.py`, `sleep_consolidation.py`, 2026-11-07)
- **Context**: Prior virtual `<|subliminal_gap|>` tokens stayed unmasked in `past_kv`, Pass 3 reused Pass 2's comma chord, multi-stanza `Truth/Theory` diluted Stanza 2, and local REM sleep never minted Tier-2 `LearnedLeaf` nodes or harvested `ContrastiveGapPair` rows.
- **Design Choice**: Added `_mask_prior_virtual_gaps`, wired Pass-3 comma chords through `compute_stanza_residual_chord`, applied Synthesis Weighting (`0.70 * final + 0.30 * prior`), and added Tier-2 `LearnedLeaf` birth plus `ContrastiveGapPair` harvesting.
- **Result**: Verified `38/38` passing tests across core and chess test suites.

## D064: Six Core Evolution Upgrades (`reflection_classifier.py`, `right_brain_equilibrium.py`, `three_legged_runtime.py`, 2026-11-07)
- **Context**: Matching a `hazard_` rule boosted `Execution` instead of inhibiting it; colliding rules tied on `min_similarity`; `Experiential` sensors validated 3 of 6 leaves; general dialogue lacked pre-gap rollout; and live REM sleep lacked Callosum micro-plasticity.
- **Design Choice**: Added `polarity` (`"attractive"` vs. `"repulsive"`) on `FuzzyConceptRule`, soft-min tie-breaking (`0.85 * min_sim + 0.15 * mean_sim`), `OPTIONAL_EXPERIENTIAL_SENSOR_LEAVES`, `select_pre_gap_branch_chord_rollout`, and `update_projector_on_harvested_gap_pairs`.
- **Result**: Verified `57/57` passing tests across core and app suites with Cycle-0 Virgin SHA-256 locks intact.

## D065: 24-Circuit Weight-Space Sculpting, Homeostatic Decay & Founding Cluster Naming (`chess_kibitz_bridge.py`, `sleep_consolidation.py`, 2026-11-07)
- **Context**: `BitboardIntuitionSleeve` saturated `order`/`chaos` at `±2.5` while 6 bitboard circuits stayed frozen, and `_cluster_key_for_unresolved` picked Tier-2 `hazard_*` leaves to produce recursive `hazard_high_hazard_` names.
- **Design Choice**: Threaded `played_bitboard_features` and `reference_bitboard_features` into `distill_post_mortem_scene`, restricted `_cluster_key_for_unresolved` to `FOUNDATIONAL_LEAVES`, and stripped recursive prefixes in `canonical_cluster_concept_name`.
- **Result**: Verified `56/56` passing tests across `test_chess_app.py`, `test_sleep_consolidation.py`, `test_virgin_baseline.py`, and core suites.

## D066: Core Weight-Space Sculpting (`subliminal_gap_splice.py`, `right_brain_equilibrium.py`, `sleep_consolidation.py`, 2026-11-07)
- **Context**: `entrain_existing_cluster_rule` ratcheted `high` above `0.90` and accumulated `4+` target leaves, `LearnedLeaf.anchor_weights` stayed frozen after birth, and `RIGHT_BRAIN_WILL_EDGES` (`9` Will synapses) never adapted during REM Sleep.
- **Design Choice**: Updated `entrain_existing_cluster_rule` with polarity preservation, top-3 leaf pruning, and Homeostatic Threshold Decay (`high <= 0.90`, `low <= 0.75`); added `sculpt_learned_leaf_anchor_weights`; and added `sculpt_will_edges_during_sleep` (`will_edges.json`, keeping Virgin locked).
- **Result**: Normalized 7 cluster rules within homeostatic bounds and verified all core, chess, and Virgin baseline tests pass.

## D067: Semantic Birth Anchoring, Symmetric Updates, Unstamped Coordinates & `will_edges` Scoring (`subliminal_gap_splice.py`, `sleep_consolidation.py`, `chess_app.py`, `chess_kibitz_bridge.py`, 2026-11-08)
- **Context**: Audit and Red-Team review identified 5 installer bugs, one-way ratchets in `BitboardIntuitionSleeve` and `sculpt_will_edges_during_sleep`, `0.25 / 0.53` coordinate stamps, a rigged `calm_baseline_score` gate and `45%` raw-coordinate blend at Tier-2 leaf birth, random static at `Cycle 0` birth, REM vow/archive-chopping theater, `live_projector_weights.pt` overwrites, and un-wired `will_edges` in move scoring.
- **Design Choice**: Fixed all 5 installer bugs, replaced one-way ratchets with symmetric contrastive updates, removed `0.25 / 0.53` coordinate stamps, replaced `calm_baseline_score` with a genuine `calm_deviation` gate and unblended `score_leaf_anchor_weights_at_birth`, cut Dream Vow / `apply_dream_reflection_to_archives` theater and `live_projector_weights.pt` overwrites, wired `self.will_edges` into `evaluate_candidate_moves_with_right_brain`, defaulted `SubliminalGapProjector.intermediate_size = 256` with `anchor_to_llm_embeddings`, and retrained all 3 checkpoints on the 72-pair dataset.
- **Result**: Raw holdout flips jumped from `0/6` to `6/6` (`+6.4123` net swap margin) on `Qwen/Qwen3-8B`, `5/6` (`6/6` calibrated, `+3.7225` margin) on `Impish_Bloodmoon_12B`, and `5/6` (`6/6` calibrated, `+4.7017` margin) on `Qwen/Qwen3-4B`.
