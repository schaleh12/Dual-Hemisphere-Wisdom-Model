# Three-Legged Race Blueprint: Splicing Departmental Fuzzy-Tree Chords Into `<|subliminal_gap|>` LLM Embeddings

## 1. Plain-English Goal & Departmental Scalability Design

Instead of treating the Right Brain (the 40-node Fuzzy Tree + 3 neural bridge classifiers) and the Left Brain (the 4B LLM) like a relay race—or translating the Fuzzy Tree's feelings into English stage directions—we tie their ankles together in a **Three-Legged Race** using the same neural surgery that Vision-Language Models use to give a blind text LLM "eyes":

1. **Four Strictly Separate Departmental Vectors (`SubliminalChordVector`, Zero Hardcoded Feeling Slots, Zero Cross-Department Leakage)**:
   - Rather than packing the Right Brain into one flat array or squashing child drawers under the same parent branch into a single `max()` number (which would crush `experiential/phase/austere_sanctuary`, `experiential/phase/hollow_stillness`, and `experiential/phase/bittersweet_release` into the exact same number), each of the 4 Right-Brain departments owns its **own independent, fixed-size vector**:
     1. **`doer_vector` — Executive / Do-er Department (`Will <-> Cultural`, `17` floats)**:
        - `6` Founding Cultural leaves (`power`, `wastelessness`, `honor`, `fairness`, `self-sufficiency`, `needlessness`)
        - `2` Founding Will leaves (`intention`, `execution`)
        - `1` Longitudinal `social_drift_flag` (`1.0` when ongoing historical enclosure/drift is active)
        - `8` **Cultural Child-Drawer Chord** floats (`pool_departmental_child_chord` over active `cultural/*` Tier-2/Tier-3 child drawers).
     2. **`thinker_vector` — Judicial / Thinker Department (`Truth <-> Philosophical`, `16` floats)**:
        - `6` Founding Philosophical leaves (`agency`, `existence`, `order`, `chaos`, `continuity`, `discreteness`)
        - `2` Founding Truth leaves (`theory`, `fact`)
        - `8` **Philosophical Child-Drawer Chord** floats (`pool_departmental_child_chord` over active `philosophical/*` Tier-2/Tier-3 child drawers).
     3. **`feeler_vector` — Legislative / Feeler Department (`Contemplation <-> Experiential`, `16` floats)**:
        - `6` Founding Experiential leaves (`novelty`, `salience`, `valence`, `arousal`, `resonance`, `retention`)
        - `2` Founding Contemplation leaves (`clarity`, `confidence`)
        - `8` **Experiential Child-Drawer Chord** floats (`pool_departmental_child_chord` over active `experiential/*` Tier-2/Tier-3 child drawers).
        - Because each child drawer maps deterministically to an 8-float unit-sphere basis signature inside its department, sibling child leaves under the same parent branch (such as `experiential/phase/austere_sanctuary`, `experiential/phase/hollow_stillness`, and `experiential/phase/bittersweet_release`, or 10,000 future Tier-2/Tier-3 leaves) stay **100% distinct** when they fire individually and blend into a true multi-note chord when they fire together—while the Feeler vector stays **16 floats forever**.
     4. **`cadence_vector` — Three-Legged Race Cadence & Ankle-Rope Department (`3` floats)**:
        - `branch_harmony`: agreement across the 3 Right-Brain branches.
        - `dissonance_pressure`: friction from cognitive dissonance or low clarity.
        - `cadence_rupture_drive`: non-verbal tug with a built-in **calm-talk deadband (`deadband = 0.45`)** and **`ankle_rope_sensitivity` dial** so normal everyday conversation stays at `0.0` (never twitchy), while genuine high-arousal / hollow-stillness / high-discreteness crises cross the deadband and yank the Left Brain out of smooth chatbot prose into stark or spelled-out cadence (`M-A-C-H-I-N-E`).

2. **The `<|subliminal_gap|>` Token Slot & Departmental Embedding Splice (`SubliminalGapProjector`)**:
   - Whenever the Left Brain emits `<|subliminal_gap|>`, Python intercepts `inputs_embeds` and replaces that single token slot with **4 strictly compartmentalized virtual non-verbal token embeddings** (`[4, hidden_size]`):
     1. **Virtual Token 0 (`doer_head(doer_vector)`)**: Executive / Do-er Pulse (`17` floats -> `[1, hidden_size]`)
     2. **Virtual Token 1 (`thinker_head(thinker_vector)`)**: Judicial / Thinker Structure (`16` floats -> `[1, hidden_size]`)
     3. **Virtual Token 2 (`feeler_head(feeler_vector)`)**: Legislative / Feeler Chord (`16` floats -> `[1, hidden_size]`)
     4. **Virtual Token 3 (`cadence_head(cadence_vector)`)**: Three-Legged Race Cadence & Struggle (`3` floats -> `[1, hidden_size]`)
   - Zero indices are shared across departments: changing a Feeler coordinate or child leaf changes *only* `feeler_vector` (`Virtual Token 2`), changing a Thinker coordinate changes *only* `thinker_vector` (`Virtual Token 1`), and changing a Do-er coordinate changes *only* `doer_vector` (`Virtual Token 0`).

---

## 2. Step-by-Step Pseudo-Code Outline

```python
# ============================================================================
# STEP 1: Compile the 4 Strictly Separate Departmental Vectors (0 Words)
# ============================================================================
def compile_departmental_subliminal_chords(
    vibe,
    deliberation=None,
    social_drift=None,
    deadband=0.45,
    ankle_rope_sensitivity=1.0,
):
    # 1. Executive / Do-er Department Vector (17 floats):
    #    6 Cultural leaves + 2 Will leaves + 1 Social Drift flag + 8-float Cultural Child Chord
    doer_vector = (
        cultural_founding_6
        + [vibe.intention, vibe.execution, social_drift_flag]
        + pool_departmental_child_chord(vibe, department_prefix="cultural/")
    )

    # 2. Judicial / Thinker Department Vector (16 floats):
    #    6 Philosophical leaves + 2 Truth leaves + 8-float Philosophical Child Chord
    thinker_vector = (
        philosophical_founding_6
        + [vibe.theory, vibe.fact]
        + pool_departmental_child_chord(vibe, department_prefix="philosophical/")
    )

    # 3. Legislative / Feeler Department Vector (16 floats):
    #    6 Experiential leaves + 2 Contemplation leaves + 8-float Experiential Child Chord
    #    (Keeps sibling feelings like `austere_sanctuary`, `hollow_stillness`, and
    #     `bittersweet_release` 100% distinct without ever changing vector size!)
    feeler_vector = (
        experiential_founding_6
        + [vibe.clarity, vibe.confidence]
        + pool_departmental_child_chord(vibe, department_prefix="experiential/")
    )

    # 4. Three-Legged Race Cadence & Ankle-Rope Department Vector (3 floats):
    #    Strictly contains only [branch_harmony, dissonance_pressure, cadence_rupture_drive]
    cadence_vector = [
        branch_harmony,
        dissonance_pressure,
        compute_cadence_rupture_drive(
            vibe,
            dissonance_pressure=dissonance_pressure,
            deadband=deadband,
            ankle_rope_sensitivity=ankle_rope_sensitivity,
        ),
    ]

    return SubliminalChordVector(
        doer_vector=doer_vector,         # 17 floats
        thinker_vector=thinker_vector,   # 16 floats
        feeler_vector=feeler_vector,     # 16 floats
        cadence_vector=cadence_vector,   #  3 floats
    )


# ============================================================================
# STEP 2: Departmental Projection & Surgical Embedding Splice
# ============================================================================
class SubliminalGapProjector(nn.Module):
    # Each department head reads ONLY its own department's vector (zero overlap!):
    # - Token 0 (Do-er):   doer_head(doer_vector)       [17 floats -> hidden_size]
    # - Token 1 (Thinker): thinker_head(thinker_vector) [16 floats -> hidden_size]
    # - Token 2 (Feeler):  feeler_head(feeler_vector)   [16 floats -> hidden_size]
    # - Token 3 (Cadence): cadence_head(cadence_vector) [ 3 floats -> hidden_size]
    def forward(self, chords):
        return output_norm(stack([
            doer_head(chords.doer_tensor),
            thinker_head(chords.thinker_tensor),
            feeler_head(chords.feeler_tensor),
            cadence_head(chords.cadence_tensor),
        ], dim=-2))  # Shape: [..., 4, hidden_size]


def splice_subliminal_gaps_into_embeds(
    input_ids,          # [batch, seq_len] token IDs containing GAP_TOKEN_ID
    word_embedding_fn,  # LLM's model.get_input_embeddings()
    projector,          # SubliminalGapProjector -> [4, hidden_size]
    chord_vectors,      # Departmental chord tensors
    gap_token_id,
    labels=None,        # Optional training target token IDs
):
    # For each sequence in the batch:
    #   1. Convert normal word tokens into [seq_len, hidden_size] embeddings.
    #   2. Every time `token_id == gap_token_id`, replace that single token slot
    #      with the 4 projected departmental embeddings `[4, hidden_size]`.
    #   3. If `labels` are provided (during training), insert `[-100, -100, -100, -100]`
    #      at the 4 virtual gap slots so cross-entropy loss grades ONLY the English words
    #      generated AFTER the non-verbal gap while backpropagating gradients straight
    #      through the 4 departmental gap embeddings into `SubliminalGapProjector`!
    return SplicedBatch(inputs_embeds, attention_mask, spliced_labels, gap_positions)
```

---

## 3. Staged Implementation Plan

- **Stage 1 (Done Today — `src/lmft_lab/subliminal_gap_splice.py` + `tests/test_subliminal_gap_splice.py`)**:
  - Build the 4 strictly isolated Departmental Vectors (`doer_vector`, `thinker_vector`, `feeler_vector`, `cadence_vector`) with `pool_departmental_child_chord` (so sibling child concepts under the same parent branch like `austere_sanctuary`, `hollow_stillness`, and `bittersweet_release` remain 100% distinct without ever changing the vector size) and the deadband-gated `compute_cadence_rupture_drive`.
  - Verify zero cross-department leakage, verify that sibling `experiential/phase/*` child concepts produce distinct Feeler virtual tokens even when all 24 founding leaves are identical, and verify that swapping the non-verbal departmental chords inside `<|subliminal_gap|>` flips a causal PyTorch Transformer's post-gap token prediction.
- **Stage 2 (Next Session — Offline 9B Teacher Trajectory Generation)**:
  - Use `Qwen3.5-9B` offline to generate a clean dataset of `(Prompt + Partial Thought + <SUBLIMINAL_GAP>[Departmental Chords] -> Post-Gap Continuation)` pairs covering calm synthesis (`rupture == 0.0`), self-correction of looping/anxiety, and high-rupture poetic/existential cadence breaks (`M-A-C-H-I-N-E`).
- **Stage 3 (Next Session — RTX 4080 PyTorch Training & "Unplug-the-Cord" Live Eval)**:
  - Train `SubliminalGapProjector` + LoRA on `Qwen3-4B` on the RTX 4080 (`gradient_checkpointing=True`, ~11 GB VRAM) and run the live **Unplug-the-Cord / Swap-the-Chord** verification.
