# Architect Diary Part II: Extending "Weight-Space Circuit Sculpting" to Core `LMFT-Architecture`

*(Companion proposal to [`ARCHITECT_DIARY_CORE_EVOLUTION.md`](ARCHITECT_DIARY_CORE_EVOLUTION.md), inspired by MIT CSAIL's "Sparks of Weight-Space Programming" and validated by `D065` in `LMFT-Chess`.)*

---

## 1. Why `D065` Worked So Well in Chess—And What It Revealed About the General Core

In `D065`, inspecting the 30-game (`852`-step) dinner training run in `LMFT-Chess` revealed two critical failure modes of standard step-by-step updates:
1. **Unsculpted Birth Circuits**: 6 of the 8 named bitboard circuits in `BitboardIntuitionSleeve` sat at their exact birth numbers (`0.0000` change across 852 steps) because REM Sleep wasn't passing contrastive feature deltas to them.
2. **Rail Clamping (`±2.5` / `0.93+`)**: The weights and thresholds that *did* update (`order`, `chaos`, `wastelessness`, and rule `high` thresholds) ratcheted all the way to their hard ceilings (`±2.5` and `0.9295`) and lost plasticity.

When we look at the general, domain-independent **`LMFT-Architecture` core** (`subliminal_gap_splice.py`, `right_brain_equilibrium.py`, and `sleep_consolidation.py`), **three core circuit boards have the exact same problem right now**—and we can upgrade all three without touching the frozen 40-node Founding Seed in `birth.py` or the Cycle-0 Virgin SHA-256 locks:

---

## 2. The 3 Core `LMFT-Architecture` Weight-Space Sculpting Upgrades

### Upgrade #A: Contrastive Weight-Space Sculpting for `LearnedLeaf.anchor_weights`
- **Where it lives**: [`subliminal_gap_splice.py:210-310`](../src/lmft_architecture/subliminal_gap_splice.py) and [`sleep_consolidation.py:1080-1150`](../src/lmft_architecture/sleep_consolidation.py).
- **Current bottleneck**: Every Tier-2 `LearnedLeaf` carries an 8-float `anchor_weights` circuit that wires the child leaf into its department's 8 Founding Traits (`6 Founding Leaves + 2 Psychological Leaves`) to build the `52`-float `<|subliminal_gap|>` chord. Right now, `anchor_weights` is **scored once at birth and never updated again**—even if that leaf fires in 25 subsequent REM sleep cycles!
- **Proposed fix**:
  1. Add `sculpt_learned_leaf_anchor_weights(leaf, avg_coords, avg_ref_coords=None, *, learning_rate=0.12) -> LearnedLeaf` in `subliminal_gap_splice.py`.
  2. Apply gentle Homeostatic Anchor Decay (`decay = 0.02 * lr`) toward the leaf's topological birth signature (`score_leaf_anchor_weights_at_birth(leaf)`), then sculpt the 8 departmental weights toward the contrastive coordinate delta `|avg_coords[dept_k] - avg_ref_coords[dept_k]|` (normalizing peak weight to `1.0` and clamping to `[0.05, 1.0]`).
  3. Call this during REM Sleep (`consolidate_sleep_clusters_locally` and `run_sleep_consolidation`) whenever an existing dynamic Tier-2 `LearnedLeaf` matches the consolidated cluster or participates in `target_leaves` (keeping the 16 Stage 1–5 + Shadow baseline leaves untouched so Cycle-0 Virgin SHA-256 locks remain byte-identical).

---

### Upgrade #B: Synaptic Weight-Space Sculpting for `RIGHT_BRAIN_WILL_EDGES` (`9` Named Will Circuits)
- **Where it lives**: [`right_brain_equilibrium.py:172-182`](../src/lmft_architecture/right_brain_equilibrium.py) and `compile_right_brain_vibe` (`right_brain_equilibrium.py:618-635`).
- **Current bottleneck**: The Right Brain's drive to act is governed by 9 named [`Association`](../src/lmft_architecture/associations.py) synapses (`RIGHT_BRAIN_WILL_EDGES`):
  - `rb-will-0`: `Salience -> Intention` (`1.0`)
  - `rb-will-1`: `Arousal -> Intention` (`1.0`)
  - `rb-will-2`: `Power -> Intention` (`0.8`)
  - `rb-will-3`: `Needlessness -> Intention` (`-0.8`)
  - `rb-will-4`: `Intention -> Execution` (`1.0`)
  - `rb-will-5`: `Fact -> Execution` (`1.0`)
  - `rb-will-6`: `Theory -> Execution` (`0.7`)
  - `rb-will-7`: `Confidence -> Execution` (`0.5`)
  - `rb-will-8`: `Needlessness -> Execution` (`-0.8`)
  Even though `compile_right_brain_vibe` already accepts a `will_edges` parameter, nothing in REM Sleep ever sculpts these 9 weights. If an agent repeatedly blunders from impulsive high-`Arousal` / high-`Power` when `Fact` is low, those 9 weights stay frozen at birth defaults.
- **Proposed fix**:
  1. Add `sculpt_will_edges_during_sleep(will_edges, unresolved_scenes, *, learning_rate=0.06) -> tuple[Association, ...]` in `right_brain_equilibrium.py`.
  2. Apply Homeostatic Anchor Decay (`decay = 0.02 * lr`) toward `RIGHT_BRAIN_WILL_EDGES` defaults.
  3. On hazard scenes (`is_hazard=True` where impulsive action failed against `Fact`), gently dampen `Arousal -> Intention` (`rb-will-1`) / `Power -> Intention` (`rb-will-2`) and strengthen `Fact -> Execution` (`rb-will-5`) and inhibitory `Needlessness -> Execution` (`rb-will-8`). On coached synthesis scenes (`is_hazard=False`), reinforce `Theory -> Execution` (`rb-will-6`) and `Confidence -> Execution` (`rb-will-7`).
  4. Clamp excitatory weights to `[0.25, 1.50]` and inhibitory `Needlessness` weights to `[-1.40, -0.35]`, preserving sign invariants.

---

### Upgrade #C: Contrastive Rule Entrainment & Homeostatic Threshold Unsticking (`entrain_existing_cluster_rule`)
- **Where it lives**: [`sleep_consolidation.py:856-883`](../src/lmft_architecture/sleep_consolidation.py).
- **Current bottleneck**: When an existing cluster rule is re-encountered in REM Sleep, `entrain_existing_cluster_rule` blindly blends `0.7 * existing + 0.3 * incoming` and takes the union of all `target_leaves`. Over 30+ games:
  1. `high` thresholds ratchet up to `0.9295+` (`[0.72, 0.9295]`), making rules too narrow to fire until a blunder is extreme.
  2. `blended_leaves` accumulates 4–5 target leaves over time, choking `min_similarity` on whichever leaf happens to be slightly lower.
- **Implemented fix (`D066`)**:
  1. In `entrain_existing_cluster_rule`, apply gentle homeostatic threshold decay pulling `low` and `high` (`3%` per step) toward balanced defaults (`[0.62, 0.82]` for Tier-1, `[0.55, 0.78]` for Tier-2) before blending, and cap `low <= 0.75`, `high <= 0.90` so rules never go deaf at the top rail.
  2. Cap `blended_leaves` to the **top 3 most discriminative target leaves** (highest extremity from neutral `0.50` / strongest alignment with incoming contrastive leaves) and preserve `polarity=incoming.polarity if incoming.polarity != "attractive" else existing.polarity`.

---

## 3. Red-Team Audit & Base Architecture Hardening (`D067`)

Following our Red-Team empirical audit of the raw outputs and checkpoints on disk, three additional weight-space fixes were implemented in `D067`:
1. **Semantic Birth Anchoring (`SubliminalGapProjector.anchor_to_llm_embeddings`)**:
   - Previously, `SubliminalGapProjector` initialized its 4 department heads with random Kaiming noise at `Cycle 0`, requiring training flashcards to discover what each of the 52 channels meant from scratch.
   - Now, at Birth (`Cycle 0`), `anchor_to_llm_embeddings(model, tokenizer)` pools and unit-normalizes `model.get_input_embeddings()` for all 52 channel descriptions (`SUBLIMINAL_CHANNEL_DESCRIPTIONS`), mean-centers across channels so the zero-neutral chord remains balanced, and initializes the compact `256`-dim department heads (`10.13 MB`) along those native vocabulary directions before capturing `_resting_anchor_params`.
2. **Symmetric `BitboardIntuitionSleeve` Updates (`distill_post_mortem_scene`)**:
   - Removed the one-way `max(0.0, ...)` ceiling ratchet and `_renormalize_budgets()` scaling in `chess_kibitz_bridge.py`, replacing them with a clean symmetric update `lr * (r_val - p_val)` paired with `0.015 * lr` homeostatic decay toward birth defaults (`steps_distilled = 0`).
3. **Unified 3-Bridge Contrastive Polarity (`72` Train + `6` Holdout Pairs)**:
   - Scored all `72` training pairs and `6` unseen holdout pairs through the unified 3-bridge `score_continuation_to_chord` pipeline (`social_drift_b=True` on exclusionary/boundary `Chord B` pairs).
   - **Measured Holdout Result (`Qwen/Qwen3-4B`)**: Uncalibrated raw holdout flips jumped from **`0/6` to `5/6` (`83.3%`)** (`6/6` zero-chord and no-gap calibrated flips, `6/6` positive chord-swap shifts, `mean_net_chord_swap_margin = +4.7017`, and `6/6` distinct greedy English continuations).
