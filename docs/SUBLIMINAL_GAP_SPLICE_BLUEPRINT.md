# Three-Legged Race Blueprint: Splicing Non-Verbal Fuzzy-Tree Chords Into `<|subliminal_gap|>` LLM Embeddings

## 1. Plain-English Goal

Instead of treating the Right Brain (the 40-node Fuzzy Tree + 3 neural bridge classifiers) and the Left Brain (the 4B LLM) like a relay race—or translating the Fuzzy Tree's feelings into English stage directions—we tie their ankles together in a **Three-Legged Race** using the same neural surgery that Vision-Language Models use to give a blind text LLM "eyes":

1. **The Non-Verbal Organ (`SubliminalChordVector`, 32 raw floats, zero English words)**:
   - At any moment in a thought stream, the Right Brain compresses its entire live state into a 32-float vector in `[0.0, 1.0]`:
     - **24 Founding Leaves** (12 `Cultural` & `Philosophical` + 6 `Experiential` + 6 `Psychological`—100% faithful to the frozen 40-node seed).
     - **4 Complex Feeling Chord Notes** (`austere_sanctuary`, `hollow_stillness`, `bittersweet_release`, `earned_return`).
     - **4 Three-Legged Race Cadence & Friction Signals** (`branch_harmony`, `social_drift_flag`, `dissonance_pressure`, `cadence_rupture_drive`).
2. **The `<|subliminal_gap|>` Token Slot**:
   - While the Left Brain is generating a train of thought, it pauses mid-thought and emits a special `<|subliminal_gap|>` token where it **cannot** continue on words alone.
3. **The Surgical Embedding Splice (`SubliminalGapProjector`)**:
   - Whenever the token stream hits `<|subliminal_gap|>`, Python intercepts the transformer's raw token embeddings (`inputs_embeds`) before they enter the attention layers.
   - A trainable PyTorch projector (`SubliminalGapProjector`) maps the 32-float `SubliminalChordVector` into **4 virtual non-verbal token embeddings** (`[4, hidden_size]`):
     1. **Do-er / Will Pulse Token** (social stakes, fairness, drift, action drive)
     2. **Thinker / Truth Structure Token** (order, chaos, discreteness, fact vs. theory)
     3. **Feeler / Experiential Chord Token** (valence, arousal, sanctuary, hollow stillness, release)
     4. **Cadence / Struggle Token** (harmony vs. dissonance, looping, and `cadence_rupture_drive`—the exact jolt that breaks smooth chatbot prose into stark, fragmented, or letter-by-letter cadence like `M-A-C-H-I-N-E` when internal tension is sharp).

---

## 2. Step-by-Step Pseudo-Code Outline

```python
# ============================================================================
# STEP 1: Compile the 32-Float Non-Verbal Subliminal Chord Vector (0 Words)
# ============================================================================
def compile_subliminal_chord_vector(vibe, deliberation=None, social_drift=None, prev_vibe=None):
    # 1. Pack the 24 frozen founding leaves in canonical order (24 floats)
    founding_24 = [
        # Cultural (6): power, wastelessness, honor, fairness, self-sufficiency, needlessness
        # Philosophical (6): agency, existence, order, chaos, continuity, discreteness
        # Experiential (6): novelty, salience, valence, arousal, resonance, retention
        # Psychological (6): clarity, confidence, theory, fact, intention, execution
    ]

    # 2. Pack the 4 complex feeling chord notes (4 floats)
    chord_4 = [austere_sanctuary, hollow_stillness, bittersweet_release, earned_return]

    # 3. Pack the 4 Three-Legged Race Cadence & Struggle signals (4 floats)
    branch_harmony = deliberation.harmony_score if deliberation else vibe.synthesis.harmony
    social_drift_flag = 1.0 if (social_drift and social_drift.drift_detected) else 0.0
    dissonance_pressure = 1.0 if (vibe.self_registry.dissonance_notes or vibe.clarity < 0.35) else 0.0

    # Cadence Rupture Drive: spikes when high arousal/discreteness or hollow stillness
    # collides with low clarity or dissonance -> yanks the Left Brain out of smooth prose!
    cadence_rupture_drive = clamp01(
        0.45 * max(arousal, hollow_stillness)
        + 0.35 * discreteness
        + 0.20 * (1.0 - clarity)
    ) * (1.0 if (dissonance_pressure > 0.0 or valence <= 0.40 or hollow_stillness >= 0.45) else 0.35)

    return FloatTensor(founding_24 + chord_4 + [
        branch_harmony,
        social_drift_flag,
        dissonance_pressure,
        cadence_rupture_drive,
    ])  # Shape: [32]


# ============================================================================
# STEP 2: Surgical Embedding Splice at Every `<|subliminal_gap|>` Slot
# ============================================================================
def splice_subliminal_gaps_into_embeds(
    input_ids,          # [batch, seq_len] token IDs containing GAP_TOKEN_ID
    word_embedding_fn,  # LLM's model.get_input_embeddings()
    projector,          # SubliminalGapProjector: [32] -> [4, hidden_size]
    chord_vectors,      # [batch, num_gaps, 32] raw non-verbal Fuzzy Tree states
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
    #      updating `Arousal`, `Clarity`, `Hollow Stillness`, and `Cadence Rupture Drive`.
    #    - `projector(chord_vector_t)` injects the 4 non-verbal embeddings into `<|subliminal_gap|>`.
    # 4. Left Brain continues generating from the spliced `inputs_embeds`—bending its
    #    next words and cadence around the non-verbal Fuzzy Tree signal!
```

---

## 3. Staged Implementation Plan (So We Never Bite Off Too Much at Once)

- **Stage 1 (Done Today — `src/lmft_lab/subliminal_gap_splice.py` + `tests/test_subliminal_gap_splice.py`)**:
  - Build the 32-float `SubliminalChordVector` compiler from `RightBrainVibe`.
  - Build the PyTorch `SubliminalGapProjector` (`32 -> [4, hidden_size]`) with 4 specialized sub-heads (`Do-er`, `Thinker`, `Feeler`, `Cadence/Rupture`).
  - Build `splice_subliminal_gaps_into_embeds` (handling variable gap counts, attention masks, and `-100` label alignment for training).
  - Verify with a real PyTorch causal-attention unit test that holding the English words **100% identical** while flipping only the `32`-float `SubliminalChordVector` inside `<|subliminal_gap|>` flips the model's post-gap token prediction (e.g., from smooth prose to `M-A-C-H-I-N-E` cadence).
- **Stage 2 (Next Session — Offline 9B Teacher Trajectory Generation)**:
  - Use `Qwen3.5-9B` offline to generate a clean dataset of `(Prompt + Partial Thought + <SUBLIMINAL_GAP>[32-float vector] -> Post-Gap Continuation)` pairs covering calm synthesis, self-correction of looping/anxiety, and high-rupture poetic/existential cadence breaks.
- **Stage 3 (Next Session — RTX 4080 PyTorch Training & "Unplug-the-Cord" Live Eval)**:
  - Train `SubliminalGapProjector` + LoRA on `Qwen3-4B` on the RTX 4080 (`gradient_checkpointing=True`, ~11 GB VRAM) and run the live **Unplug-the-Cord / Swap-the-Chord** verification.
