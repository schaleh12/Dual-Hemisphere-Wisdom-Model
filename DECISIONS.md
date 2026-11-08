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
- **Context**: `BitboardIntuitionSleeve` saturated `order`/`chaos` at `±2.5` while 6 bitboard circuits stayed frozen at birth values, and `_cluster_key_for_unresolved` picked Tier-2 `hazard_*` leaves to produce recursive `hazard_high_hazard_` names.
- **Design Choice**: Threaded `played_bitboard_features` and `reference_bitboard_features` into `distill_post_mortem_scene` with `0.015 * lr` decay, restricted `_cluster_key_for_unresolved` to `FOUNDATIONAL_LEAVES`, and stripped recursive prefixes in `canonical_cluster_concept_name`.
- **Result**: Verified `56/56` passing tests across `test_chess_app.py`, `test_sleep_consolidation.py`, `test_virgin_baseline.py`, and core suites.

## D066: Core Weight-Space Sculpting (`subliminal_gap_splice.py`, `right_brain_equilibrium.py`, `sleep_consolidation.py`, 2026-11-07)
- **Context**: In the general `LMFT-Architecture` core, `entrain_existing_cluster_rule` ratcheted `high` thresholds above `0.90` and accumulated `4+` target leaves, `LearnedLeaf.anchor_weights` stayed frozen after birth, and `RIGHT_BRAIN_WILL_EDGES` (`9` Will synapses) never adapted during REM Sleep.
- **Design Choice**: Updated `entrain_existing_cluster_rule` with polarity preservation, top-3 leaf pruning, and Homeostatic Threshold Decay (`high <= 0.90`, `low <= 0.75`); added `sculpt_learned_leaf_anchor_weights` (preserving the 16 frozen baseline leaves); and added `sculpt_will_edges_during_sleep` (`will_edges.json` in `ChessKibitzerRuntime`, keeping Virgin locked to `RIGHT_BRAIN_WILL_EDGES`).
- **Result**: Normalized 7 cluster rules within homeostatic bounds and verified all core, chess, and Virgin baseline tests pass.

## D067: Semantic Birth Anchoring, Symmetric Sleeve/Will Updates & 72-Pair `<|subliminal_gap|>` Polarity Fix (`subliminal_gap_splice.py`, `train_left_brain_lora.py`, 2026-11-08)
- **Context**: Downstairs clean-machine audit and Red-Team review identified 5 installer bugs, one-way ratchets in `BitboardIntuitionSleeve` and `sculpt_will_edges_during_sleep`, hardcoded coordinate overwrites in `chess_app.py`, random Kaiming static at `Cycle 0` projector birth, and inconsistent `Chord A`/`Chord B` polarity (`0/6` raw holdout flips).
- **Design Choice**: Fixed all 5 installer bugs, replaced one-way ratchets in `distill_post_mortem_scene` and `sculpt_will_edges_during_sleep` with symmetric contrastive updates and reset `bitboard_sleeve_weights.json`, `will_edges.json`, and `club_ruleset.json` to clean birth defaults, defaulted `SubliminalGapProjector.intermediate_size = 256` with `anchor_to_llm_embeddings`, and retrained `Qwen3-4B` and `Qwen3-8B` on the 72-pair dataset.
- **Result**: Raw uncalibrated holdout flips jumped from `0/6` to `6/6` (`100%`) on `Qwen/Qwen3-8B` (`mean_net_chord_swap_margin = +6.4123`, `no_gap_calibrated = 6/6`, `greedy_text_changed = 6/6`) and `5/6` (`83.3%`) on `Qwen/Qwen3-4B` (`zero_calibrated = 6/6`, `mean_net_chord_swap_margin = +4.7017`).
