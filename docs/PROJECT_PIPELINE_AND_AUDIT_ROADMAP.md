# Project Pipeline, Plain-English Strategy, & Independent Audit Guide

This document lays out the step-by-step plan for taking the **Dual-Hemisphere Wisdom Model** from where it stands today (`Stage 1 Complete`) through live training and multi-cycle growth (`Stages 2–5`), written in plain English with zero buzzwords—plus a direct checklist for an independent Auditor AI window.

---

## 1. The Core Training Tactic: Same Starting Words, Different Inner Feeling

When people try to train a language model LoRA to respond to an internal emotional or structural vector, they usually hit a hidden trap:
- If you train on normal single sentences, the opening words of the sentence already give away where the sentence is going. So **the language model takes the lazy path: it just guesses the ending from the opening words and completely ignores the vector you spliced in.**

### How We Force the Left Brain to Actually Listen to the Right Brain
In **Stage 2**, we build our training examples in **matched pairs (`Pair A` and `Pair B`)**:
1. **Before the Subliminal Gap**: The user prompt and the first half of the model's sentence are **100% word-for-word identical** in both `Pair A` and `Pair B`.
2. **Inside the Subliminal Gap**:
   - `Pair A` gets **52-float Right-Brain Chord A** (for example: *Stoic Poker-Face Composure*, *Verified Fact*, or *Sincere Warmth*),
   - `Pair B` gets **52-float Right-Brain Chord B** (for example: *Contested Unfairness*, *Skeptical Doubt*, *Sarcastic Bluff Pivot*, or *High-Strain Cadence Rupture*).
3. **After the Subliminal Gap**: `Pair A` and `Pair B` finish the sentence in two completely different ways that match their respective Right-Brain chords.

Because the words before the gap are identical in both pairs, **it is mathematically impossible for `Qwen3-4B` to get both answers right during training unless its LoRA weights actually read the 4 non-verbal virtual tokens coming across the Corpus Callosum (`SubliminalGapProjector`).**

---

## 2. The "Chess Ice-Breaker": A Measurable Proving Ground for the 6 Philosophical Leaves

Alongside human social and emotional dilemmas, **Chess** serves as a clean, objective "ice-breaker" testbed where subconscious pattern recognition and conscious step-by-step calculation have to work together:

### A. Exact 1-to-1 Resonance Between the 6 Philosophical Leaves and the 6 Chess Pieces
In the Right Brain's **Thinker Department (`Philosophical` branch)**, the 6 founding leaves map directly onto the 6 archetypes on a chessboard:

| Philosophical Founding Leaf | Chess Piece Archetype | Positional & Structural Meaning on the Board |
| :--- | :--- | :--- |
| **`philosophical/being/agency`** | **Queen** (*Volition*) | Maximum multi-directional freedom of action and active intervention across the whole board. |
| **`philosophical/being/existence`** | **King** (*Ground of Being*) | Survival and physical shelter—if the King's safety collapses, existence on the board ends. |
| **`philosophical/structure/order`** | **Bishop** (*Prevailing Geometry*) | Long-range diagonal alignment, color-complex discipline, and structured lanes of control. |
| **`philosophical/structure/chaos`** | **Knight** (*Non-Linear Disruption*) | Leaping over walls, breaking rigid formations, forks, and unpredictable tactical disruption. |
| **`philosophical/persistence/continuity`** | **Pawn** (*Connected Flow*) | Linked pawn chains, steady territorial advance, and irreversible forward momentum toward promotion. |
| **`philosophical/persistence/discreteness`** | **Rook** (*Quantized Bastion*) | Heavy orthogonal walls, sealed files and ranks, isolated fortresses, and sharp structural boundaries. |

### B. How the Other Departments Join a Chess Turn
- **Do-er (`Cultural` + `Will`)**:
  - `Power` = initiative, forcing moves (checks, captures, threats).
  - `Wastelessness` = material conservation, not blundering pieces or wasting moves.
  - `Self-Sufficiency` & `Needlessness` = calm, Stoic "poker-face" defense and fortress building.
- **Feeler (`Experiential` + `Contemplation`)**:
  - `Salience` & `Arousal` = King-safety danger or sharp tactical tension.
  - `Clarity` vs. `Confidence` = knowing a tactical line actually works (`Clarity` high) vs. playing an overconfident "hope-chess" blunder without checking the opponent's reply (`Confidence > Clarity`).

---

## 3. The 5-Stage Project Pipeline

| Stage | Current Status | Plain-English Goal & Deliverable |
| :--- | :--- | :--- |
| **Stage 1: The 40-Node Right Brain & Corpus Callosum Bridge** | **DONE** | Frozen 40-node tree (`24` leaves incl. `Resonance` & `Entrainment`), 3 fast GPU classifiers (`~35 ms`), 3 persistent JSON branch archives, 4 separate Departmental Vectors (`17 + 16 + 16 + 3 = 52` floats), semantic child-drawer projection (`0` SHA-256 hashes), `SubliminalGapProjector`, and checkpointing (`1,333` passing tests). |
| **Stage 2: Building the Matched-Pair Training Examples** | **NEXT UP** | Build `src/lmft_lab/gap_trajectory_generator.py`. Use the 3 Right-Brain sensors and offline `Qwen3.5-9B` Teacher to create matched pairs (same opening clause before the gap, opposite 52-float chords inside the gap, different continuations after the gap) across both **human social/emotional scenes** and **chess-board intuition scenes**, with a locked **20% Unseen Holdout Split**. |
| **Stage 3: Training the Left-Brain LoRA on `Qwen3-4B` (RTX 4080)** | **Stage 3** | Build `src/lmft_lab/train_left_brain_lora.py`. Train `SubliminalGapProjector` + LoRA on `Qwen3-4B` (`models/left_brain_lora/`, `sleep_cycle_index = 1`). Run two honest pass/fail checks on unseen holdout prompts: **(1) The Chord-Swap Check** (keep the opening words frozen, swap Chord A vs. Chord B, measure how often the second half of the sentence flips to match) and **(2) The Unplug-the-Cord Check** (set the 52 floats to `0.0` and verify the steering disappears). |
| **Stage 4: Live 3-Gear Daytime Loop** | **Stage 4** | Build `src/lmft_lab/three_legged_runtime.py`: **Gear 1** (`~35 ms` silent Right-Brain reflex when all 3 branches agree, `0` LLM calls) $\rightarrow$ **Gear 2** (live clause-by-clause generation with `Qwen3-4B` pausing at the subliminal gap to read the Right Brain) $\rightarrow$ **Daytime Contemplation Ledger** (saving any turn where the branches clashed). |
| **Stage 5: Nighttime REM Sleep Growth (`Cycle 1 -> Cycle 2`)** | **Stage 5** | Feed daytime confusion logs to `Qwen3.5-9B` offline to **(a)** refine Right-Brain fuzzy rules or grow new Tier-2/Tier-3 child leaves and **(b)** train `models/left_brain_lora/` for one incremental sleep cycle (`sleep_cycle_index: 1 -> 2`), proving both hemispheres mature together over time. |

---

## 4. Independent Auditor AI Checklist (How to Verify Zero "Vibe Coding")

If you paste this repository or any future commit into a separate Auditor AI window, ask that AI to verify these **5 concrete code invariants**:

1. **Strict Departmental Separation ([`src/lmft_lab/subliminal_gap_splice.py`](../src/lmft_lab/subliminal_gap_splice.py))**:
   - Verify that `SubliminalChordVector` splits the 52 floats into 4 strictly separate departmental vectors (`doer_vector` 17, `thinker_vector` 16, `feeler_vector` 16, `cadence_vector` 3) with zero shared indices, and that each of the 4 neural heads in `SubliminalGapProjector` reads *only* its own department's slice.
2. **Zero Hashing & Zero `max()` Collisions on Sibling Child Leaves ([`src/lmft_lab/subliminal_gap_splice.py`](../src/lmft_lab/subliminal_gap_splice.py))**:
   - Verify that `semantic_child_drawer_signature()` and `pool_departmental_child_chord()` use zero `hashlib`/`sha256` calls, keep sibling leaves under the same parent branch (`austere_sanctuary`, `hollow_stillness`, `bittersweet_release`) distinct in 8-float space, and map synonymous new child leaves into the same semantic neighborhood.
3. **Pure Non-Verbal Embedding Splice (`splice_subliminal_gaps_into_embeds`)**:
   - Verify that the 52-float chord is spliced directly into PyTorch `inputs_embeds` as 4 virtual token vectors (`[4, hidden_size]`) with `-100` label masking—never injected as English stage directions in a text prompt.
4. **Zero Coordinate Clamping or Scoreboard Rigging ([`src/lmft_lab/right_brain_equilibrium.py`](../src/lmft_lab/right_brain_equilibrium.py), [`src/lmft_lab/branch_archives.py`](../src/lmft_lab/branch_archives.py))**:
   - Verify that raw sensor scores are never clamped or overwritten to force a scene to settle, competing rules are never zeroed out, and unsettled (`settled=False`) scenes are never harvested into the archives as ratified rules.
5. **Same-Prefix Chord-Swap Proof (Stages 2 & 3)**:
   - Verify that holdout evaluation keeps the opening text prefix before the gap 100% identical when testing Chord A vs. Chord B, so the LLM cannot pass without reading the spliced virtual tokens.
