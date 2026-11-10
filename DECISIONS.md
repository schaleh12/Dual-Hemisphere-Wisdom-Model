# LMFT-Architecture — Design Decisions (Rolling 5-Decision Window)

*(Keep only the 5 most recent decisions here, under 80 lines AND 5,000 characters in plain English. Earlier decisions `D001–D065` are archived in `LMFT-Architecture/docs/archive/`.)*

## D066: Core Weight-Space Sculpting (`subliminal_gap_splice.py`, `right_brain_equilibrium.py`, `sleep_consolidation.py`, 2026-11-07)
- **Context**: `entrain_existing_cluster_rule` ratcheted `high` above `0.90` and accumulated `4+` target leaves, `LearnedLeaf.anchor_weights` stayed frozen after birth, and `RIGHT_BRAIN_WILL_EDGES` never adapted during REM Sleep.
- **Design Choice**: Updated `entrain_existing_cluster_rule` with polarity preservation, top-3 leaf pruning, and Homeostatic Threshold Decay (`high <= 0.90`, `low <= 0.75`); added `sculpt_learned_leaf_anchor_weights`; and added `sculpt_will_edges_during_sleep`.
- **Result**: Normalized 7 cluster rules within homeostatic bounds and verified all core, chess, and Virgin baseline tests pass.

## D067: Semantic Birth Anchoring, Symmetric Updates & `will_edges` Scoring (`subliminal_gap_splice.py`, `sleep_consolidation.py`, `chess_app.py`, 2026-11-08)
- **Context**: Audit found 5 installer bugs, one-way ratchets, `0.25 / 0.53` coordinate stamps, a rigged `calm_baseline_score` gate, random static at `Cycle 0` birth, REM vow theater, and un-wired `will_edges` in move scoring.
- **Design Choice**: Fixed the installer bugs, replaced one-way ratchets with symmetric contrastive updates, removed `0.25 / 0.53` coordinate stamps, replaced `calm_baseline_score` with `calm_deviation`, cut Dream Vow theater, wired `self.will_edges` into move scoring, and anchored `SubliminalGapProjector` (`256`-dim) to LLM input embeddings at Birth.
- **Result**: Raw holdout flips reached `6/6` on `Qwen3-8B` and `5/6` (`6/6` calibrated) on `Impish_Bloodmoon_12B` and `Qwen3-4B`.

## D068: Tier 2–4 Sub-Feature Leaf Genesis, Ply-5 Detective Walk & Two-Way Gear-2 Deliberation (`reflection_classifier.py`, `chess_kibitz_bridge.py`, `sleep_consolidation.py`, `chess_app.py`, 2026-11-09)
- **Context**: A 128-game live run showed all 29 learned rules in `club_ruleset.json` were Tier-1 rules on the 12 Cultural + Philosophical leaves, while 82 Tier-2 `LearnedLeaf` nodes went unused because multi-ply traps had nearly identical Tier-1 summary coordinates.
- **Design Choice**: Added a 16-float Sub-Sensor Bank (`compute_move_sub_sensors`), Tier 2–4 `LearnedLeaf` genesis with parent-gated (`parent >= 0.50`) sub-sensor projections and `Tier 4 > Tier 3 > Tier 2 > Tier 1` inhibition, the `Ply 1 -> Ply 3 -> Ply 5` Intention Detective Walk, and two-way Gear-2 suspect reweighting.
- **Result**: Verified `48/48` passing tests across `test_sleep_consolidation.py`, `test_chess_app.py`, `test_subliminal_gap_splice.py`, `test_right_brain_equilibrium.py`, and `test_virgin_baseline.py`.

## D069: Universal Native 16-Bit `SicariusSicariiStuff/Impish_LLAMA_4B` Base Model & Full Clause-1 Supervision (`three_legged_runtime.py`, `impish_bloodmoon_adapter.py`, `train_left_brain_lora.py`, `chess_app.py`, `bloodmoon_app.py`, 2026-11-10)
- **Context**: `Qwen3-4B`/`8B` carry `<think>` reasoning priors and censorship filters, `Impish_Bloodmoon_12B` required 4-bit NF4 quantization, and `-100` pre-gap masking left Clause 1 un-trained up to the comma.
- **Design Choice**: Standardized `LMFT-Architecture`, `chess_app.py`, and `bloodmoon_app.py` on native 16-bit (`bfloat16`) `SicariusSicariiStuff/Impish_LLAMA_4B` (`hidden_size = 3072`, `32` layers, zero `<think>` tags, uncensored), removed `-100` pre-gap masking in `build_spliced_training_forward`, and organized checkpoints into `models/left_brain_baseline`, `models/left_brain_chess`, and `models/left_brain_bloodmoon`.
- **Result**: Native 16-bit `Impish_LLAMA_4B` achieved `6/6` raw uncalibrated holdout flips (`+4.4806` mean net chord-swap margin, `6/6` greedy text changed, `1.7440` sensor L2) in `25s/epoch` at `11.92 GB` VRAM.

## D070: `LMFT-Scholar` 12-Book Seminar, Canopy Inspector & Graduation (`scholar_app.py`, `scholar_ui.py`, 2026-11-10)
- **Context**: Training directly inside chess or hypnosis specialized rules and the Callosum too early without general domain wisdom across the 12 founding boughs.
- **Design Choice**: Built `LMFT-Scholar` (`scholar_app.py`, `scholar_ui.py`, `Launch-LMFT-Scholar.bat`, `models/left_brain_scholar/`, `outputs/scholar_state/`) with a 12-Book Literature Seminar (1 lesson per bough with REM sleep after each essay), local Tier-2/3 canopy genesis (`_grow_scholar_canopy_from_sleep_scenes`, zero 9B Teacher reliance), a Visual Tree & Canopy Inspector, and `graduate_brain_to_target` to export rules, leaves, archives, `will_edges.json`, and `live_projector_weights.pt`.
- **Result**: Verified `5/5` passing tests in `test_scholar_app.py` and full compatibility across `LMFT-Chess` and `LMFT-Bloodmoon`.
