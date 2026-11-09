# Weight-Space Circuit Sculpting & Left-Brain "Magnifying Glass" Lenses (`D065–D067`)

> **Quick Navigation**: **[← Back to Main `README.md`](../../README.md)** · **[Watch the Research Inspiration (`Two Minute Papers` at `4:30`) →](https://youtu.be/GJ6PMUm5oTA?si=ps0-qMBm3-ZQOp3c&t=270)** · **[Companion Doc: Core Evolution (`D064`) →](ARCHITECT_DIARY_CORE_EVOLUTION.md)**

---

## 1. Research Attribution & How We Use MIT CSAIL / OpenAI Weight-Space Ideas

When visitors see `Weight-Space Circuit Sculpting` in our [`README.md`](../../README.md), we want to make it 100% clear **whose scientific research inspired these mechanisms**, **what those researchers discovered**, and **how our own original Python/PyTorch code adapts those concepts**.

### Primary Research Credits
1. **MIT CSAIL — *"Sparks of Weight-Space Programming"* & Direct Circuit Editing** (featured by **Dr. Károly Zsolnai-Fehér** on ***Two Minute Papers* — ["OpenAI's Astra Just Opened The Black Box Of AI" at `4:30`](https://youtu.be/GJ6PMUm5oTA?si=ps0-qMBm3-ZQOp3c&t=270)**):
   - Demonstrated that instead of treating neural networks as untouchable black boxes that can only be changed through massive text datasets or prompt engineering, small networks and sub-circuits can be **programmed and sculpted directly in weight space**.
2. **OpenAI & Apollo Research — *Weight-Sparse Transformers Have Interpretable Circuits*** (**Leo Gao et al., 2025**, also featured in [`youtu.be/GJ6PMUm5oTA`](https://youtu.be/GJ6PMUm5oTA?si=ps0-qMBm3-ZQOp3c&t=270)):
   - Showed that when you isolate sparse, named circuits inside a model rather than letting every weight entangle with every other weight, individual nodes and edges correspond to human-readable concepts that can be inspected, amplified, or dampened directly.
3. **University of Cincinnati / Psibernetix — Genetic Fuzzy Trees (`ALPHA`, Dr. Kelly Cohen & Dr. Nicholas Ernest)**:
   - Demonstrated that hierarchical fuzzy trees (`O(1)` branch depth) can make fast, verifiable tactical decisions and evolve their membership thresholds through sparring against an objective opponent.

### What Is Original Here (100% Clean-Room Implementation)
- **Zero external code or weights copied**: None of the code in `LMFT-Architecture` comes from MIT CSAIL, OpenAI Astra, or Psibernetix. Every module in [`LMFT-Architecture/src/lmft_architecture/`](../src/lmft_architecture/) is written from scratch in readable Python and PyTorch on top of our frozen **40-Node Founding Seed** ([`birth.py`](../src/lmft_architecture/birth.py)).
- **How we split the problem in two**:
  - **In the Left Brain (the local `4B` / `8B` / `12B` LLM)**, we do **not** try to teach the language model English or chess vocabulary from scratch. Pre-trained models (`Qwen/Qwen3-8B`, `Qwen/Qwen3-4B`, `Impish_Bloodmoon_12B`) already contain rich latent circuits for chess piece coordination, tactical urgency, and prose cadence inside their frozen weights. Instead, we use **3 Left-Brain "Magnifying Glass" Circuits** that shine a spotlight on the exact latent LLM circuits that match the Right Brain's 40-node state.
  - **In the Right Brain (the 40-Node Fuzzy Tree & Bitboard Sleeve)**, day-to-day learning happens in **3 Weight-Space Sculpting Circuits** stored as human-readable JSON files—using **Homeostatic Anchor Decay** so weights never ratchet to a ceiling or overwrite the frozen 40-node Founding Seed.

| Layer | Mechanism in `LMFT-Architecture` | Where It Lives in Code | Plain-English Role |
| :--- | :--- | :--- | :--- |
| **Left Brain Lens #1** | **Semantic Birth Anchoring (`anchor_to_llm_embeddings`)** | [`subliminal_gap_splice.py`](../src/lmft_architecture/subliminal_gap_splice.py) | Initializes the 52 Right-Brain channels directly from the LLM's own native word embeddings (`model.get_input_embeddings()`) at Birth (`Cycle 0`) instead of random noise. |
| **Left Brain Lens #2** | **4-Department `<|subliminal_gap|>` Virtual Token Splice (`K = 4`)** | [`subliminal_gap_splice.py`](../src/lmft_architecture/subliminal_gap_splice.py) | Splices `[Do-er (17), Thinker (16), Feeler (16), Cadence (3)]` as 4 separate virtual tokens at the first comma (`<|subliminal_gap|>`), masking prior commas so each sentence gets a clean spotlight. |
| **Left Brain Lens #3** | **Top-8-Layer Split `qk` (`0.5x`) vs. `vo` (`1.0x`) LoRA Lenses** | [`train_left_brain_lora.py`](../../LMFT-Training/experiments/train_left_brain_lora.py) | Freezes 99%+ of the LLM and tunes only the top 8 attention layers: `q_proj/k_proj` learn *where to look* at the 4 virtual tokens, and `v_proj/o_proj` learn *which existing LLM knowledge circuits to amplify*. |
| **Right Brain Circuit #1** | **Synaptic Will Sculpting (`sculpt_will_edges_during_sleep`)** | [`right_brain_equilibrium.py`](../src/lmft_architecture/right_brain_equilibrium.py) | Adjusts the 9 named Will synapses (`rb-will-0..8` in `will_edges.json`) with homeostatic decay, directly steering move selection in [`evaluate_candidate_moves_with_right_brain`](../src/lmft_architecture/chess_kibitz_bridge.py). |
| **Right Brain Circuit #2** | **Tier-2 Leaf Coupling (`sculpt_learned_leaf_anchor_weights`)** | [`subliminal_gap_splice.py`](../src/lmft_architecture/subliminal_gap_splice.py) | Adjusts each dynamic Tier-2 `LearnedLeaf`'s 8 departmental coupling weights (`anchor_weights`) toward the traits that actually separated good moves from blunders. |
| **Right Brain Circuit #3** | **Contrastive Rule & 24-Float Bitboard Sleeve Sculpting** | [`sleep_consolidation.py`](../src/lmft_architecture/sleep_consolidation.py) & [`chess_kibitz_bridge.py`](../src/lmft_architecture/chess_kibitz_bridge.py) | Sculpts `FuzzyConceptRule` firing bands (`entrain_existing_cluster_rule`) and the 24 named bitboard circuits (`distill_post_mortem_scene`) with homeostatic pull toward birth defaults. |

---

## 2. The 3 Left-Brain "Magnifying Glass" Circuits (Focusing Frozen LLM Knowledge)

A common misconception is that small adapter weights "teach" a language model domain knowledge from scratch. In reality, an 8-billion-parameter model like `Qwen/Qwen3-8B` has already read millions of chess games, tactical analyses, and stories during pre-training—it just doesn't know **which** of its internal circuits to activate on a given turn without getting distracted by surface chat wording.

Our three Left-Brain weight-space lenses act like **three magnifying glasses stacked in a telescope**, focusing the frozen LLM onto the exact internal concepts selected by the Right Brain:

### Magnifying Glass #1: Semantic Birth Anchoring (`SubliminalGapProjector.anchor_to_llm_embeddings`, `D067`)
- **Code**: [`SubliminalGapProjector.anchor_to_llm_embeddings`](../src/lmft_architecture/subliminal_gap_splice.py) in [`subliminal_gap_splice.py`](../src/lmft_architecture/subliminal_gap_splice.py).
- **How it works**:
  - Before a single training step runs (`Cycle 0`), we take the plain-English descriptions of all 52 Right-Brain channels (`SUBLIMINAL_CHANNEL_DESCRIPTIONS`—the 24 Founding Leaves, 24 Tier-2/3/4 slots, and 4 cadence dials) and look up their token vectors inside the LLM's own frozen input embedding matrix (`model.get_input_embeddings()`).
  - We mean-pool and unit-normalize each channel's native embedding vector, subtract the cross-channel mean so a neutral chord stays centered at zero, and project those native directions directly into the 4 department heads (`256`-dim bottleneck, `10.13 MB`).
  - **Why it acts as a magnifying glass**: Even at Birth (`Cycle 0`), when the Right Brain raises `philosophical/structure/order` or `cultural/efficiency/wastelessness`, the projector immediately pushes the LLM along the exact vocabulary directions the LLM *already uses* for order, structure, and material conservation.

### Magnifying Glass #2: 4-Department Virtual Token Injection at `<|subliminal_gap|>` (`K = 4`)
- **Code**: [`SubliminalGapProjector.forward`](../src/lmft_architecture/subliminal_gap_splice.py) and [`_PersistentTurnStreamState`](../src/lmft_architecture/three_legged_runtime.py) in [`three_legged_runtime.py`](../src/lmft_architecture/three_legged_runtime.py).
- **How it works**:
  - Instead of squashing all 52 Right-Brain numbers into a single blurry vector, `SubliminalGapProjector` keeps the four departments physically separate:
    1. **Token 0 (`Do-er`, `17` floats)**: `Cultural` leaves + `Will` (`Intention`, `Execution`) + Executive Tier-2/4 leaves.
    2. **Token 1 (`Thinker`, `16` floats)**: `Philosophical` leaves + `Truth` (`Theory`, `Fact`) + Judicial Tier-2/4 leaves.
    3. **Token 2 (`Feeler`, `16` floats)**: `Experiential` leaves + `Contemplation` (`Clarity`, `Confidence`) + Legislative Tier-2/4 leaves.
    4. **Token 3 (`Cadence`, `3` floats)**: `sentence_budget_norm`, `stanza_budget_norm`, and `contemplation_depth`.
  - When the LLM writes the opening clause of a sentence and hits the first comma, these `K = 4` virtual tokens are spliced directly into the transformer's KV cache (`past_kv`). Across multi-sentence responses, `_mask_prior_virtual_gaps` masks out earlier commas so each new sentence only sees its own fresh 4-token chord.

### Magnifying Glass #3: Top-8-Layer Split `qk` vs. `vo` LoRA Attention Lenses
- **Code**: [`LoRALinear`](../src/lmft_architecture/three_legged_runtime.py) and [`train_left_brain_lora.py`](../../LMFT-Training/experiments/train_left_brain_lora.py).
- **How it works**:
  - We freeze the bottom two-thirds of the transformer completely (`100%` untouched) and attach rank-16 `LoRALinear` lenses only to `q_proj`, `k_proj`, `v_proj`, and `o_proj` in the **top 8 layers** (`layers 28..35` on `Qwen3-8B` / `Qwen3-4B`; `layers 32..39` on `Impish_Bloodmoon_12B`).
  - During contrastive training (`72` 3-Bridge-scored pairs, holding out `6` unseen scenarios), we split the learning rates by role:
    - **`q_proj` & `k_proj` (`0.5x` learning rate — *Where to Look*)**: Gently aims the top-layer attention heads so tokens after the comma attend to the right virtual token (`Do-er` vs. `Thinker` vs. `Feeler`).
    - **`v_proj` & `o_proj` (`1.0x` learning rate — *What to Amplify*)**: Amplifies the frozen LLM's existing latent circuits into the output stream based on which virtual token is active.
- **Measured Unseen Holdout Result (`D067`)**:
  - On `Qwen/Qwen3-8B` (the default Chess App model), swapping only the 52-float chord at the comma (with zero prompt changes) flips **raw uncalibrated greedy English text on `6/6` (`100.0%`) unseen holdout scenarios** (`mean_net_chord_swap_margin = +6.4123`).
  - On both `Qwen/Qwen3-4B` and `Impish_Bloodmoon_12B`, raw uncalibrated greedy flips jumped from **`0/6` to `5/6` (`83.3%`)** (`6/6` calibrated flips).

---

## 3. The 3 Right-Brain Weight-Space Sculpting Circuits (Homeostatic REM Plasticity)

While the Left Brain's magnifying glasses stay locked to their verified `6/6` / `5/6` checkpoints during live app sessions so the voice never drifts or forgets English, the **Right Brain** updates its decision circuits after every REM Sleep cycle.

In [`D065–D067`](../../DECISIONS.md), inspecting multi-game training runs revealed two classic failure modes of naive weight updates:
1. **Unsculpted Birth Circuits**: Named circuits that never received contrastive `(reference - played)` deltas sat frozen at their birth numbers.
2. **Ceiling Saturation (Rail Clamping)**: One-way ratchets or un-anchored step updates pushed active weights (`wastelessness`, `material_balance_norm`, `move_net_safety`, and rule `high` thresholds) straight to their hard ceilings (`±2.5` or `0.93+`) within a few games, destroying nuance.

To fix both without touching the frozen 40-node Founding Seed in [`birth.py`](../src/lmft_architecture/birth.py), every Right-Brain weight update pairs **Symmetric Contrastive Deltas** with **Homeostatic Anchor Decay** toward birth defaults:

### Right-Brain Circuit #1: Synaptic Weight-Space Sculpting for `RIGHT_BRAIN_WILL_EDGES` (`9` Named Will Circuits)
- **Where it lives**: [`sculpt_will_edges_during_sleep`](../src/lmft_architecture/right_brain_equilibrium.py) in [`right_brain_equilibrium.py`](../src/lmft_architecture/right_brain_equilibrium.py), persisted to `will_edges.json`, and wired directly into [`evaluate_candidate_moves_with_right_brain`](../src/lmft_architecture/chess_kibitz_bridge.py).
- **The 9 Named Will Synapses**:
  - `rb-will-0`: `Salience -> Intention` (`1.00`)
  - `rb-will-1`: `Arousal -> Intention` (`1.00`)
  - `rb-will-2`: `Power -> Intention` (`0.80`)
  - `rb-will-3`: `Needlessness -> Intention` (`-0.80`)
  - `rb-will-4`: `Intention -> Execution` (`1.00`)
  - `rb-will-5`: `Fact -> Execution` (`1.00`)
  - `rb-will-6`: `Theory -> Execution` (`0.70`)
  - `rb-will-7`: `Confidence -> Execution` (`0.50`)
  - `rb-will-8`: `Needlessness -> Execution` (`-0.80`)
- **How REM Sleep Sculpts Them**:
  1. Applies gentle Homeostatic Anchor Decay (`decay = 0.02 * lr`) toward the birth defaults above.
  2. Computes symmetric contrastive deltas (`coord_val - ref_val`) from unresolved scenes: when impulsive high-`Arousal` / high-`Power` moves fail against `Fact` (`is_hazard=True`), it dampens `Arousal -> Intention` and `Power -> Intention` and strengthens `Fact -> Execution` and inhibitory `Needlessness -> Execution`. When a disciplined line succeeds (`is_hazard=False`), it reinforces `Theory -> Execution` and `Confidence -> Execution`.
  3. Directly drives `intention` and `will_execution_drive` when scoring legal candidate moves on the chessboard.

### Right-Brain Circuit #2: Contrastive Weight-Space Sculpting for `LearnedLeaf.anchor_weights` (`8` Dept. Circuits per Leaf)
- **Where it lives**: [`sculpt_learned_leaf_anchor_weights`](../src/lmft_architecture/subliminal_gap_splice.py) in [`subliminal_gap_splice.py`](../src/lmft_architecture/subliminal_gap_splice.py) and [`sleep_consolidation.py`](../src/lmft_architecture/sleep_consolidation.py).
- **How REM Sleep Sculpts Them**:
  1. Every dynamic Tier-2 `LearnedLeaf` carries an 8-float `anchor_weights` vector wiring that child concept into its department's 8 Founding Traits (`6 Founding + 2 Psychological`) inside the 52-float `<|subliminal_gap|>` chord.
  2. At birth, `anchor_weights` is initialized purely from the leaf's topological signature (`score_leaf_anchor_weights_at_birth`).
  3. Whenever that leaf participates in a subsequent REM Sleep consolidation, `sculpt_learned_leaf_anchor_weights` applies `0.02 * lr` homeostatic decay toward its topological birth signature and shifts the 8 weights toward the traits with the largest contrastive gap `|avg_coords[dept_k] - avg_ref_coords[dept_k]|` (keeping the 16 Stage 1–5 + Shadow baseline leaves untouched so Cycle-0 Virgin SHA-256 locks stay byte-identical).

### Right-Brain Circuit #3: Contrastive Rule Entrainment (`entrain_existing_cluster_rule`) & 24-Float `BitboardIntuitionSleeve` (`distill_post_mortem_scene`)
- **Where it lives**: [`entrain_existing_cluster_rule`](../src/lmft_architecture/sleep_consolidation.py) in [`sleep_consolidation.py`](../src/lmft_architecture/sleep_consolidation.py) and [`BitboardIntuitionSleeve.distill_post_mortem_scene`](../src/lmft_architecture/chess_kibitz_bridge.py) in [`chess_kibitz_bridge.py`](../src/lmft_architecture/chess_kibitz_bridge.py).
- **How REM Sleep Sculpts Them**:
  1. **Rule Threshold Unsticking**: Pulls `low` and `high` rule thresholds (`3%` per step) toward balanced defaults (`[0.62, 0.82]` for Tier-1, `[0.55, 0.78]` for Tier-2), caps `low <= 0.75` and `high <= 0.90` so rules never go deaf at the top rail, and prunes `target_leaves` to the **top 3 most discriminative leaves**.
  2. **Homeostatic Bitboard Sleeve Distillation (`24` Named Chess Circuits)**: Pulls `value_weights` (`8`), `policy_weights` (`8`), and `leaf_weights` (`8`) gently toward their birth defaults (`DEFAULT_VALUE_WEIGHTS`, `DEFAULT_POLICY_WEIGHTS`, `DEFAULT_LEAF_WEIGHTS`) instead of `0.0`, and updates only features with real contrastive separation (`|r_val - p_val| >= 0.15`) between Stockfish's stronger reference continuation and the played blunder.

---

## 4. Why We Separate Live Right-Brain Sculpting from Offline Left-Brain LoRA Training

At one point we explored whether live AFK sessions should also run continuous online LoRA/projector gradient updates (similar to `ReLoRA` merge-and-restart cycles) after every game. Our Red-Team audit showed why separating the two timescales works much better:
- **Fast Online Timescale (Every REM Sleep Cycle — Right Brain Only)**:
  - Updating `will_edges.json`, `bitboard_sleeve_weights.json`, `LearnedLeaf.anchor_weights`, and `club_ruleset.json` happens in pure coordinate space in milliseconds, cannot corrupt English syntax, and is 100% inspectable in plain JSON.
- **Slow Offline Timescale (Curated Multi-Cycle Curriculum — Left Brain Magnifying Glasses)**:
  - The Left-Brain `SubliminalGapProjector` + top-layer `LoRALinear` checkpoints (`models/left_brain_qwen3_8b/`, `models/left_brain_lora_cycle2/`, and `models/left_brain_impish_bloodmoon/`) are trained offline on verified, 3-Bridge-scored contrastive pairs ([`train_left_brain_lora.py`](../../LMFT-Training/experiments/train_left_brain_lora.py)) and gated on the 6 unseen holdout scenarios before being locked for live use.
