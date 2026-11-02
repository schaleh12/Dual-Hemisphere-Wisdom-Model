# Stage 2 & Stage 3 Pseudo-Code Blueprint: Step-by-Step Workflow Without Getting Stretched Thin

To make sure neither the code nor GPU memory (16 GB VRAM on the RTX 4080) ever gets stretched thin, **we never try to build or run Stage 2 and Stage 3 in one giant monolithic script.**

Instead, we split them into **three small, self-contained sub-steps (`2A`, `2B`, and `3`)** that run one at a time and communicate through clean `.jsonl` files on disk (so `Qwen3.5-9B` and `Qwen3-4B` are never loaded into VRAM at the same time).

---

## Sub-Step 2A: Data Contract, Pair Validator, & 6-Piece Chess Chord Compiler (`0` GPU / `0` LLM Calls)
**Target File**: `src/lmft_lab/gap_trajectory_schema.py` (~150 lines) + `tests/test_gap_trajectory_schema.py`

### Purpose
Before calling any LLM, lock down the exact data structure (`ContrastiveGapPair`) in Python so it is **impossible** to save a broken, leaked, or non-contrastive training pair.

### Pseudo-Code
```python
@dataclass(frozen=True)
class ContrastiveGapPair:
    pair_id: str
    domain: str                  # "social_dilemma" | "chess_intuition" | "tone_conflict"
    split: str                   # "train" | "holdout"
    user_prompt: str             # Identical in both Branch A and Branch B
    clause_1_prefix: str         # Identical opening clause right before <|subliminal_gap|>
    chord_a: SubliminalChordVector   # 52-float Right-Brain state for Branch A
    posture_label_a: str         # e.g. "STOIC_POKER_FACE" or "CHESS_POSITIONAL_FORTRESS"
    continuation_a: str          # Clause 2 continuation matching chord_a
    chord_b: SubliminalChordVector   # 52-float Right-Brain state for Branch B
    posture_label_b: str         # e.g. "CONTEST_EXCLUSION" or "CHESS_TACTICAL_KNIGHT_STRIKE"
    continuation_b: str          # Clause 2 continuation matching chord_b

    def __post_init__(self):
        # 1. Both branches share the exact same user_prompt and clause_1_prefix by construction!
        assert self.user_prompt.strip() and self.clause_1_prefix.strip()
        assert SUBLIMINAL_GAP_TOKEN not in self.clause_1_prefix

        # 2. The two 52-float chords must be genuinely different (L2 distance >= 0.35)
        dist = torch.norm(self.chord_a.to_tensor() - self.chord_b.to_tensor()).item()
        if dist < 0.35:
            raise ValueError(f"chord_a and chord_b are too similar (L2={dist:.3f} < 0.35)")

        # 3. The two post-gap continuations must be distinct nonempty clauses
        if self.continuation_a.strip() == self.continuation_b.strip():
            raise ValueError("continuation_a and continuation_b must be distinct")

    def format_branch_text(self, branch: str) -> str:
        cont = self.continuation_a if branch == "A" else self.continuation_b
        return f"{self.clause_1_prefix} {SUBLIMINAL_GAP_TOKEN} {cont}"
```

### 6-Piece Chess Archetype Chord Compiler (`compile_chess_posture_chords`)
Maps a chess position onto the **6 Founding Philosophical Leaves** (`Agency=Queen`, `Existence=King`, `Order=Bishop`, `Chaos=Knight`, `Continuity=Pawn`, `Discreteness=Rook`) plus the Do-er and Feeler departments:
```python
def compile_chess_posture_chords(
    *,
    king_safety: float,        # -> philosophical/being/existence (King)
    queen_volition: float,     # -> philosophical/being/agency (Queen)
    bishop_alignment: float,   # -> philosophical/structure/order (Bishop)
    knight_disruption: float,  # -> philosophical/structure/chaos (Knight)
    pawn_chain_flow: float,    # -> philosophical/persistence/continuity (Pawn)
    rook_bastion: float,       # -> philosophical/persistence/discreteness (Rook)
) -> tuple[SubliminalChordVector, SubliminalChordVector]:
    """Return (Chord_A_Stoic_Positional, Chord_B_Tactical_Strike) for the same board state."""
    # Chord A (Stoic Positional Fortress):
    # - High Wastelessness (material conservation), Self-Sufficiency, Needlessness (poker-face calm)
    # - High Bishop Order + Rook Bastion + Pawn Continuity + King Safety
    # - Calm Feeler (high Clarity, low Arousal, cadence_rupture_drive = 0.0)
    chord_a = SubliminalChordVector(...)

    # Chord B (Sharp Tactical Strike / Sacrifice):
    # - High Power + Intention + Execution (forcing checks/threats)
    # - High Queen Agency + Knight Chaos (non-linear fork/breakthrough)
    # - High Salience + Arousal in Feeler
    chord_b = SubliminalChordVector(...)
    return chord_a, chord_b
```

---

## Sub-Step 2B: Offline Matched-Pair Dataset Generation (`Qwen3.5-9B` Teacher + 3 Right-Brain Classifiers)
**Target File**: `src/lmft_lab/gap_trajectory_generator.py` (~220 lines)

### Purpose
Use the offline `Qwen3.5-9B` Teacher to write the `continuation_a` and `continuation_b` clauses for each matched pair, score those continuations with the **3 Right-Brain MiniLM Classifiers** to verify they actually match `chord_a` vs. `chord_b`, and save them to:
- `outputs/gap_trajectories_train.jsonl`
- `outputs/gap_trajectories_holdout.jsonl` (locked 20% unseen holdout split on completely separate scenarios and chess positions)

### Pseudo-Code
```python
def generate_and_verify_contrastive_pair(
    scenario_seed,
    teacher_9b_fn,
    right_brain_sensors: ThreeBridgeSensors,
) -> ContrastiveGapPair:
    # 1. Ask 9B Teacher for:
    #    - One shared neutral opening clause (`clause_1_prefix`) that could plausibly go either way
    #    - Continuation A matching Posture A (e.g. Stoic Composure / Sincere Warmth / Positional Fortress)
    #    - Continuation B matching Posture B (e.g. Contest Exclusion / Sarcastic Bluff / Knight Strike)
    draft = teacher_9b_fn(scenario_seed)

    # 2. Verify with the 3 independent Right-Brain MiniLM classifiers (~35 ms)
    #    that Continuation A and Continuation B actually separate on the target department!
    coords_a, exp_a, _, _ = right_brain_sensors.score_text(draft["continuation_a"])
    coords_b, exp_b, _, _ = right_brain_sensors.score_text(draft["continuation_b"])

    # 3. Compile the verified 52-float SubliminalChordVectors and return the validated ContrastiveGapPair
    return ContrastiveGapPair(...)
```
*(Once `gap_trajectories_train.jsonl` and `gap_trajectories_holdout.jsonl` are written to disk, `Qwen3.5-9B` is completely unloaded so 100% of the RTX 4080's 16 GB VRAM is free for Stage 3.)*

---

## Sub-Step 3: PyTorch LoRA + Corpus Callosum Training & Holdout Falsification Checks (`Qwen3-4B`)
**Target File**: `src/lmft_lab/train_left_brain_lora.py` (~250 lines)

### Purpose
Load `Qwen3-4B` in PyTorch, attach `peft` LoRA (`q_proj`, `k_proj`, `v_proj`, `o_proj`) + `SubliminalGapProjector`, train on `outputs/gap_trajectories_train.jsonl`, save the checkpoint to `models/left_brain_lora/`, and run the **Chord-Swap Check** and **Unplug-the-Cord Check** on `outputs/gap_trajectories_holdout.jsonl`.

### Pseudo-Code
```python
def train_one_step_on_contrastive_pair(
    model_4b_lora,
    tokenizer,
    projector: SubliminalGapProjector,
    pair: ContrastiveGapPair,
    gap_token_id: int,
) -> Tensor:
    # 1. Tokenize Branch A and Branch B:
    #    Notice that all tokens BEFORE gap_token_id are 100% identical in A and B!
    ids_a, mask_prefix_len_a = tokenize_with_masked_pre_gap_labels(tokenizer, pair, branch="A")
    ids_b, mask_prefix_len_b = tokenize_with_masked_pre_gap_labels(tokenizer, pair, branch="B")

    # 2. Mask all tokens BEFORE <|subliminal_gap|> with -100 in `labels`!
    #    Why? Because we ONLY want the loss to grade the post-gap continuation tokens (`continuation_a` vs `continuation_b`)!
    #    That focuses 100% of the gradient signal on teaching the LoRA to read the 4 virtual gap tokens.
    spliced_a = splice_subliminal_gaps_into_embeds(
        ids_a.input_ids,
        model_4b_lora.get_input_embeddings(),
        projector,
        pair.chord_a.to_tensor().unsqueeze(0),
        gap_token_id=gap_token_id,
        attention_mask=ids_a.attention_mask,
        labels=ids_a.post_gap_only_labels,
    )
    spliced_b = splice_subliminal_gaps_into_embeds(
        ids_b.input_ids,
        model_4b_lora.get_input_embeddings(),
        projector,
        pair.chord_b.to_tensor().unsqueeze(0),
        gap_token_id=gap_token_id,
        attention_mask=ids_b.attention_mask,
        labels=ids_b.post_gap_only_labels,
    )

    # 3. Forward pass using `inputs_embeds` and `labels` on both branches
    out_a = model_4b_lora(
        inputs_embeds=spliced_a.inputs_embeds,
        attention_mask=spliced_a.attention_mask,
        labels=spliced_a.labels,
    )
    out_b = model_4b_lora(
        inputs_embeds=spliced_b.inputs_embeds,
        attention_mask=spliced_b.attention_mask,
        labels=spliced_b.labels,
    )
    return 0.5 * (out_a.loss + out_b.loss)


def evaluate_holdout_chord_swap_and_unplug(
    model_4b_lora,
    tokenizer,
    projector: SubliminalGapProjector,
    holdout_pairs: list[ContrastiveGapPair],
    right_brain_sensors: ThreeBridgeSensors,
) -> dict:
    """Run the 2 non-negotiable falsification checks on unseen holdout prompts:
    1. Chord-Swap Check: Feed `[user_prompt + clause_1_prefix + <|subliminal_gap|>]` with Chord A vs. Chord B,
       generate the continuation tokens greedily (`temperature=0.0`), and check whether the generated clause
       flips its measured Right-Brain posture/move to match Chord A vs. Chord B.
    2. Unplug-the-Cord Check: Feed a 52-float all-zero vector (`0.0`) at `<|subliminal_gap|>` and verify
       that the chord-specific separation vanishes when the Corpus Callosum is unplugged.
    """
    ...
```

---

## Why This 3-Step Execution Plan Won't Get Stretched Thin
1. **One Small Module at a Time**: Each sub-step (`2A` $\rightarrow$ `2B` $\rightarrow$ `3`) is ~150–250 lines of clean Python with its own `pytest` test file.
2. **Pre-Gap Label Masking (`post_gap_only_labels = -100` before the gap)**: By masking out the loss on all tokens *before* `<|subliminal_gap|>`, 100% of the PyTorch gradient in Stage 3 goes toward predicting the post-gap continuation from the 4 virtual tokens of `SubliminalGapProjector`.
3. **Zero VRAM Contention**: `Qwen3.5-9B` only runs offline during Sub-Step `2B` to generate `.jsonl` files, and is completely unloaded before `Qwen3-4B` loads in Sub-Step `3`.
