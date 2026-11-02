# Three-Legged Race Blueprint: Splicing Non-Verbal Fuzzy-Tree Chords Into `<|subliminal_gap|>` LLM Embeddings

## 1. Plain-English Goal & Anti-Bloat Scalability Design

Instead of treating the Right Brain (the 40-node Fuzzy Tree + 3 neural bridge classifiers) and the Left Brain (the 4B LLM) like a relay race—or translating the Fuzzy Tree's feelings into English stage directions—we tie their ankles together in a **Three-Legged Race** using the same neural surgery that Vision-Language Models use to give a blind text LLM "eyes":

1. **The Non-Verbal Organ (`SubliminalChordVector`, 37 raw floats, zero English words, 100% scale-invariant)**:
   - To make sure the plug **never changes shape** even when the tree grows from `10` child leaves to `10,000` child leaves across dozens of domains, we never hardcode individual learned child leaves into fixed vector slots. Instead, every learned Tier-2/Tier-3 child leaf rolls up into its permanent Founding Parent Branch:
     - **`[0..23]` — The 24 Frozen Founding Leaves**: 12 `Cultural` & `Philosophical` + 6 `Experiential` + 6 `Psychological` (100% faithful to the frozen 40-node seed).
     - **`[24..32]` — The 9 Permanent Parent-Branch Child-Drawer Pools**:
       - `cultural/efficiency`, `cultural/justice`, `cultural/freedom` (3)
       - `philosophical/being`, `philosophical/structure`, `philosophical/persistence` (3)
       - `experiential/reflection`, `experiential/phase`, `experiential/synchronization` (3)
       - Whether a parent branch has `1` child drawer or `1,000` child drawers, `pool_child_drawers_by_parent_branch` rolls the active child drawers under that branch into a single `[0.0, 1.0]` parent-branch channel. The vector dimension stays **37 floats forever**.
     - **`[33..36]` — The 4 Three-Legged Race Cadence & Ankle-Rope Signals**:
       - `branch_harmony`: agreement across the 3 Right-Brain branches.
       - `social_drift_flag`: `1.0` when ongoing historical enclosure/drift is active.
       - `dissonance_pressure`: friction from cognitive dissonance or low clarity.
       - `cadence_rupture_drive`: non-verbal tug with a built-in **calm-talk deadband (`deadband = 0.45`)** and **`ankle_rope_sensitivity` dial** so normal everyday conversation stays at `0.0` (never twitchy), while genuine high-arousal / hollow-stillness / high-discreteness crises cross the deadband and yank the Left Brain out of smooth chatbot prose into stark or spelled-out cadence (`M-A-C-H-I-N-E`).

2. **The `<|subliminal_gap|>` Token Slot & Departmental Embedding Splice (`SubliminalGapProjector`)**:
   - Whenever the Left Brain emits `<|subliminal_gap|>`, Python intercepts `inputs_embeds` and replaces that single token slot with **4 strictly compartmentalized virtual non-verbal token embeddings** (`[4, hidden_size]`). Rather than blending all 37 floats into a single shared trunk, each virtual token is wired exclusively to its own department's index slice (`DOER_DEPARTMENT_INDICES`, `THINKER_DEPARTMENT_INDICES`, `FEELER_DEPARTMENT_INDICES`, `CADENCE_DEPARTMENT_INDICES`) so the branches never blur together:
     1. **Virtual Token 0 — Executive / Do-er Pulse (`12` floats)**: 6 `Cultural` leaves + 2 `Will` leaves (`Intention`, `Execution`) + 3 `Cultural` Parent-Branch Child Pools + 1 `Social Drift` flag.
     2. **Virtual Token 1 — Judicial / Thinker Structure (`11` floats)**: 6 `Philosophical` leaves + 2 `Truth` leaves (`Theory`, `Fact`) + 3 `Philosophical` Parent-Branch Child Pools.
     3. **Virtual Token 2 — Legislative / Feeler Chord (`11` floats)**: 6 `Experiential` leaves + 2 `Contemplation` leaves (`Clarity`, `Confidence`) + 3 `Experiential` Parent-Branch Child Pools (`experiential/reflection`, `experiential/phase`, `experiential/synchronization`).
     4. **Virtual Token 3 — Three-Legged Race Cadence & Struggle (`9` floats)**: 4 cadence signals (`branch_harmony`, `social_drift_flag`, `dissonance_pressure`, and deadband-gated `cadence_rupture_drive`) + 5 tension anchors (`Discreteness`, `Valence`, `Arousal`, `Clarity`, `Experiential Phase Pool`).

---

## 2. Step-by-Step Pseudo-Code Outline

```python
# ============================================================================
# STEP 1: Compile the 37-Float Scale-Invariant Subliminal Chord Vector (0 Words)
# ============================================================================
def compile_subliminal_chord_vector(
    vibe,
    deliberation=None,
    social_drift=None,
    deadband=0.45,
    ankle_rope_sensitivity=1.0,
):
    # 1. Pack the 24 frozen founding leaves in canonical order (24 floats)
    founding_24 = [
        # Cultural (6): power, wastelessness, honor, fairness, self-sufficiency, needlessness
        # Philosophical (6): agency, existence, order, chaos, continuity, discreteness
        # Experiential (6): novelty, salience, valence, arousal, resonance, retention
        # Psychological (6): clarity, confidence, theory, fact, intention, execution
    ]

    # 2. Roll up ANY number of learned Tier-2/Tier-3 child leaves (10 or 10,000)
    #    into their 9 permanent Founding Parent Branches (9 floats -> never changes size!)
    parent_branch_pools_9 = [
        max_child_activation_under("cultural/efficiency"),
        max_child_activation_under("cultural/justice"),
        max_child_activation_under("cultural/freedom"),
        max_child_activation_under("philosophical/being"),
        max_child_activation_under("philosophical/structure"),
        max_child_activation_under("philosophical/persistence"),
        max_child_activation_under("experiential/reflection"),
        max_child_activation_under("experiential/phase"),
        max_child_activation_under("experiential/synchronization"),
    ]

    # 3. Pack the 4 Three-Legged Race Cadence & Ankle-Rope signals (4 floats)
    branch_harmony = deliberation.harmony_score if deliberation else vibe.synthesis.harmony
    social_drift_flag = 1.0 if (social_drift and social_drift.drift_detected) else 0.0
    dissonance_pressure = compute_dissonance_pressure(vibe)

    # Deadband-gated Cadence Rupture Drive:
    # - Stays strictly 0.0 during normal calm conversation (raw_strain <= deadband)
    #   so the Left Brain is never overly twitchy or melodramatic!
    # - Ramps up smoothly above `deadband` when high arousal / experiential phase strain
    #   and discreteness collide with low clarity or dissonance.
    cadence_rupture_drive = compute_cadence_rupture_drive(
        arousal=vibe.arousal,
        discreteness=vibe.coordinates["philosophical/persistence/discreteness"],
        clarity=vibe.clarity,
        valence=vibe.valence,
        experiential_phase_pool=parent_branch_pools_9[7],
        dissonance_pressure=dissonance_pressure,
        deadband=deadband,
        sensitivity=ankle_rope_sensitivity,
    )

    return FloatTensor(founding_24 + parent_branch_pools_9 + [
        branch_harmony,
        social_drift_flag,
        dissonance_pressure,
        cadence_rupture_drive,
    ])  # Permanent Shape: [37] regardless of how many thousands of child leaves exist!


# ============================================================================
# STEP 2: Departmental Projection & Surgical Embedding Splice
# ============================================================================
class SubliminalGapProjector(nn.Module):
    # Strictly isolates each virtual token to its own department's slice of indices:
    # - Token 0 (Do-er):   DOER_DEPARTMENT_INDICES    (12 floats -> [1, hidden_size])
    # - Token 1 (Thinker): THINKER_DEPARTMENT_INDICES (11 floats -> [1, hidden_size])
    # - Token 2 (Feeler):  FEELER_DEPARTMENT_INDICES  (11 floats -> [1, hidden_size])
    # - Token 3 (Cadence): CADENCE_DEPARTMENT_INDICES ( 9 floats -> [1, hidden_size])
    def forward(self, chord_vectors):
        return output_norm(stack([
            doer_head(chord_vectors[..., DOER_DEPARTMENT_INDICES]),
            thinker_head(chord_vectors[..., THINKER_DEPARTMENT_INDICES]),
            feeler_head(chord_vectors[..., FEELER_DEPARTMENT_INDICES]),
            cadence_head(chord_vectors[..., CADENCE_DEPARTMENT_INDICES]),
        ], dim=-2))  # Shape: [..., 4, hidden_size]


def splice_subliminal_gaps_into_embeds(
    input_ids,          # [batch, seq_len] token IDs containing GAP_TOKEN_ID
    word_embedding_fn,  # LLM's model.get_input_embeddings()
    projector,          # SubliminalGapProjector: [37] -> [4, hidden_size]
    chord_vectors,      # [batch, num_gaps, 37] raw non-verbal Fuzzy Tree states
    gap_token_id,
    labels=None,        # Optional training target token IDs
):
    # For each sequence in the batch:
    #   1. Convert normal word tokens into [seq_len, hidden_size] embeddings.
    #   2. Every time `token_id == gap_token_id`, replace that single token slot
    #      with the 4 projected non-verbal Fuzzy-Tree embeddings `[4, hidden_size]`.
    #   3. If `labels` are provided (during training), insert `[-100, -100, -100, -100]`
    #      at the 4 virtual gap slots so cross-entropy loss grades ONLY the English words
    #      generated AFTER the non-verbal gap while backpropagating gradients straight
    #      through the 4 non-verbal gap embeddings into `SubliminalGapProjector`!
    return SplicedBatch(inputs_embeds, attention_mask, spliced_labels, gap_positions)


# ============================================================================
# STEP 3: The "Three-Legged Race" Simultaneous Generation Loop
# ============================================================================
def run_three_legged_thought_stream(llm, projector, right_brain_tree, prompt_text):
    # 1. Right Brain reads initial situation -> compiles initial `chord_vector_0` (35 ms).
    # 2. Left Brain begins generating thought tokens until it emits `<|subliminal_gap|>`.
    # 3. The instant `<|subliminal_gap|>` appears:
    #    - Right Brain scores the Left Brain's partial thought alongside the scene (35 ms),
    #      updating the 24 founding leaves, the 9 parent-branch child pools, and the
    #      deadband-gated `Cadence Rupture Drive`.
    #    - `projector(chord_vector_t)` injects the 4 non-verbal embeddings into `<|subliminal_gap|>`.
    # 4. Left Brain continues generating from the spliced `inputs_embeds`—staying smooth
    #    when `cadence_rupture_drive == 0.0` and fracturing its cadence when tension spikes!
```

---

## 3. Staged Implementation Plan

- **Stage 1 (Done Today — `src/lmft_lab/subliminal_gap_splice.py` + `tests/test_subliminal_gap_splice.py`)**:
  - Build the scale-invariant 37-float `SubliminalChordVector` compiler with 9 permanent Parent-Branch Child-Drawer Pools (`pool_child_drawers_by_parent_branch`) and the deadband-gated `compute_cadence_rupture_drive`.
  - Verify that adding 100+ brand-new Tier-2 and Tier-3 child leaves across any branch automatically rolls up into the 9 parent-branch channels without changing the 37-float vector size, that everyday calm scenes stay at `cadence_rupture_drive == 0.0` (zero twitchiness), and that swapping only the 37-float vector inside `<|subliminal_gap|>` flips a causal PyTorch Transformer's post-gap token prediction.
- **Stage 2 (Next Session — Offline 9B Teacher Trajectory Generation)**:
  - Use `Qwen3.5-9B` offline to generate a clean dataset of `(Prompt + Partial Thought + <SUBLIMINAL_GAP>[37-float vector] -> Post-Gap Continuation)` pairs covering calm synthesis (`rupture == 0.0`), self-correction of looping/anxiety, and high-rupture poetic/existential cadence breaks (`M-A-C-H-I-N-E`).
- **Stage 3 (Next Session — RTX 4080 PyTorch Training & "Unplug-the-Cord" Live Eval)**:
  - Train `SubliminalGapProjector` + LoRA on `Qwen3-4B` on the RTX 4080 (`gradient_checkpointing=True`, ~11 GB VRAM) and run the live **Unplug-the-Cord / Swap-the-Chord** verification.
