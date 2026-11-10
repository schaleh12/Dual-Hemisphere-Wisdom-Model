# Archived Design Decision — D066

## D066: Core Weight-Space Sculpting (`subliminal_gap_splice.py`, `right_brain_equilibrium.py`, `sleep_consolidation.py`, 2026-11-07)
- **Context**: `entrain_existing_cluster_rule` ratcheted `high` above `0.90` and accumulated `4+` target leaves, `LearnedLeaf.anchor_weights` stayed frozen after birth, and `RIGHT_BRAIN_WILL_EDGES` never adapted during REM Sleep.
- **Design Choice**: Updated `entrain_existing_cluster_rule` with polarity preservation, top-3 leaf pruning, and Homeostatic Threshold Decay (`high <= 0.90`, `low <= 0.75`); added `sculpt_learned_leaf_anchor_weights`; and added `sculpt_will_edges_during_sleep`.
- **Result**: Normalized 7 cluster rules within homeostatic bounds and verified all core, chess, and Virgin baseline tests pass.
