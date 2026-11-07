# LMFT-Architecture — Design Decisions (Rolling 5-Decision Window)

*(Keep only the 5 most recent decisions here, under 80 lines AND 5,000 characters in plain English. Earlier decisions `D001–D060` are archived in `LMFT-Architecture/docs/archive/`.)*

## D061: `Qwen/Qwen3-8B` 4-Bit Left Brain & Two-Tier REM Dream Contemplation (`chess_app.py`, `sleep_consolidation.py`, 2026-11-07)
- **Context**: `Qwen3-4B` (`2560`-d) had limited poetic depth during REM Sleep, and a 2-sentence reflection could not separately articulate felt tension, structural geometry, and execution for the 3-Branch Archives.
- **Design Choice**: Trained a 4-bit NF4 `Qwen/Qwen3-8B` (`4096`-d) `SubliminalGapProjector` + `LoRALinear` checkpoint (`models/left_brain_qwen3_8b/`); upgraded `_generate_rem_sleep_self_talk` to emit 2-sentence cat-naps and 4-sentence deep-sleep poems; and added `apply_dream_reflection_to_archives`.
- **Result**: Verified `5/6` holdout chord-swap shifts (`+0.88` margin, `0.0` unplugged shift) and `37/37` passing tests.

## D062: Persistent `past_kv` Streaming, Multi-Stanza Discourse & Longitudinal REM Vows (`three_legged_runtime.py`, `chess_app.py`, 2026-11-07)
- **Context**: Discarding `past_kv` at every `<|subliminal_gap|>` caused `O(N^2)` prefill overhead, and REM dreams lacked multi-stanza structure or cross-cycle vow tracking.
- **Design Choice**: Added `_PersistentTurnStreamState` so prompt prefill runs once per turn and `past_kv` persists across `<|subliminal_gap|>` chords and `\n\n` stanza breaks; added `generate_multi_stanza_discourse` (`max_theory_delta < 0.04`) and `compute_stanza_residual_chord`; and added `rem_dream_chronicles` (`HONORED` / `BROKEN` / `NEW_VOW`).
- **Result**: Verified `40/40` passing tests across `test_three_legged_runtime.py`, `test_chess_app.py`, `test_sleep_consolidation.py`, and `test_virgin_baseline.py`.

## D063: Ghost-Chord Masking, Stanza Synthesis, Pass-3 Residual Pivot & Tier-2 Leaf Birth (`three_legged_runtime.py`, `sleep_consolidation.py`, 2026-11-07)
- **Context**: Prior virtual `<|subliminal_gap|>` tokens stayed unmasked in `past_kv`, Pass 3 reused Pass 2's comma chord, multi-stanza `Truth/Theory` diluted Stanza 2 with Stanza 1, and local REM sleep never minted Tier-2 `LearnedLeaf` nodes or harvested `ContrastiveGapPair` rows in `LMFT-Chess.exe`.
- **Design Choice**: Added `_mask_prior_virtual_gaps`; wired Pass-3 comma chords through `compute_stanza_residual_chord`; applied Synthesis Weighting (`0.70 * final + 0.30 * prior`); added Tier-2 `LearnedLeaf` birth (`score_leaf_anchor_weights_at_birth`) in `sleep_consolidation.py`; and added live `ContrastiveGapPair` harvesting (`celestriad_gap_trajectories.jsonl`).
- **Result**: Verified `38/38` passing tests across core and chess test suites.

## D064: Six Core Evolution Upgrades (`reflection_classifier.py`, `right_brain_equilibrium.py`, `three_legged_runtime.py`, 2026-11-07)
- **Context**: Matching a `hazard_` rule boosted `Execution` via `rb-will-7` instead of inhibiting it; colliding rules tied on `min_similarity`; `Experiential` sensors validated 3 of 6 leaves; general dialogue lacked pre-gap rollout; and live REM sleep lacked Callosum micro-plasticity.
- **Design Choice**: Added `polarity` (`"attractive"` vs. `"repulsive"`) on `FuzzyConceptRule` so repulsive matches inhibit `Execution` and append `hazard_rule_warning(...)`; added soft-min tie-breaking (`0.85 * min_sim + 0.15 * mean_sim`); added `OPTIONAL_EXPERIENTIAL_SENSOR_LEAVES`; added `select_pre_gap_branch_chord_rollout`; wired Tier-2 leaf birth and gap harvesting into `bloodmoon_app.py`; and added `update_projector_on_harvested_gap_pairs`.
- **Result**: Verified `57/57` passing tests across core and app suites with Cycle-0 Virgin SHA-256 locks intact.

## D065: 24-Circuit Weight-Space Sculpting, Homeostatic Decay & Founding Cluster Naming (`chess_kibitz_bridge.py`, `sleep_consolidation.py`, 2026-11-07)
- **Context**: Across 30 games (`852` steps), `BitboardIntuitionSleeve` saturated `order`/`chaos` at `±2.5` while 6 bitboard circuits stayed frozen at birth values, and `_cluster_key_for_unresolved` picked Tier-2 `hazard_*` leaves to produce recursive `hazard_high_hazard_` names and cross-polarity `target_leaves`.
- **Design Choice**: Threaded `played_bitboard_features` and `reference_bitboard_features` into `distill_post_mortem_scene` to sculpt all 8 `value_weights` and `policy_weights` circuits with gentle homeostatic decay (`0.015 * lr`); restricted `_cluster_key_for_unresolved` to `FOUNDATIONAL_LEAVES`; stripped recursive prefixes in `canonical_cluster_concept_name`; and blocked cross-polarity Tier-2 leaf tangling.
- **Result**: Verified `56/56` passing tests across `test_chess_app.py`, `test_sleep_consolidation.py`, `test_virgin_baseline.py`, and core suites.
