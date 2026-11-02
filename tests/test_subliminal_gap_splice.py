"""Unit tests for Stage 1 of the 'Three-Legged Race' `<|subliminal_gap|>` Embedding Splice."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from lmft_lab.birth import create_birth_state
from lmft_lab.branch_archives import SocialDriftReport
from lmft_lab.reflection_classifier import FuzzyConceptRule
from lmft_lab.right_brain_equilibrium import compile_right_brain_vibe
from lmft_lab.subliminal_gap_splice import (
    IGNORE_LABEL_INDEX,
    NUM_VIRTUAL_GAP_TOKENS,
    SUBLIMINAL_VECTOR_DIM,
    SubliminalChordVector,
    SubliminalGapProjector,
    splice_subliminal_gaps_into_embeds,
)


def _sample_rules() -> tuple[FuzzyConceptRule, ...]:
    return (
        FuzzyConceptRule(
            rule_id="rule-fair-coop",
            concept_name="fair_structured_cooperation",
            description="Orderly, fair cooperation where resources are shared equitably.",
            target_leaves={
                "cultural/justice/fairness": 0.85,
                "philosophical/structure/order": 0.80,
                "philosophical/structure/chaos": 0.10,
            },
        ),
    )


def test_subliminal_chord_vector_compiles_32_floats_and_cadence_rupture():
    birth = create_birth_state("ind-subliminal-test", birth_event_id="b-sub-1", random_seed=42)
    rules = _sample_rules()

    # 1. Calm Sanctuary scene -> low cadence_rupture_drive, high branch_harmony
    calm_coords = {
        "cultural/efficiency/power": 0.15,
        "cultural/efficiency/wastelessness": 0.88,
        "cultural/justice/honor": 0.85,
        "cultural/justice/fairness": 0.92,
        "cultural/freedom/self-sufficiency": 0.65,
        "cultural/freedom/needlessness": 0.50,
        "philosophical/being/agency": 0.70,
        "philosophical/being/existence": 0.92,
        "philosophical/structure/order": 0.85,
        "philosophical/structure/chaos": 0.08,
        "philosophical/persistence/continuity": 0.75,
        "philosophical/persistence/discreteness": 0.25,
        "experiential/phase/austere_sanctuary": 0.82,
        "experiential/phase/hollow_stillness": 0.08,
        "experiential/phase/bittersweet_release": 0.12,
        "experiential/synchronization/earned_return": 0.20,
    }
    calm_exp = {
        "experiential/reflection/salience": 0.80,
        "experiential/phase/valence": 0.84,
        "experiential/phase/arousal": 0.32,
    }
    vibe_calm, _ = compile_right_brain_vibe(
        birth,
        calm_coords,
        calm_exp,
        rules,
        pass_index=1,
        event_id="ev-calm",
    )
    vec_calm = SubliminalChordVector.from_right_brain_vibe(vibe_calm)
    assert len(vec_calm.values) == SUBLIMINAL_VECTOR_DIM == 32
    assert vec_calm.branch_harmony >= 0.80
    assert vec_calm.cadence_rupture_drive <= 0.25

    # 2. High-strain Hollow Stillness + Discreteness + Low Clarity -> high cadence_rupture_drive!
    rupture_coords = {
        "cultural/efficiency/power": 0.82,
        "cultural/efficiency/wastelessness": 0.40,
        "cultural/justice/honor": 0.30,
        "cultural/justice/fairness": 0.14,
        "cultural/freedom/self-sufficiency": 0.30,
        "cultural/freedom/needlessness": 0.15,
        "philosophical/being/agency": 0.75,
        "philosophical/being/existence": 0.90,
        "philosophical/structure/order": 0.88,
        "philosophical/structure/chaos": 0.12,
        "philosophical/persistence/continuity": 0.30,
        "philosophical/persistence/discreteness": 0.90,
        "experiential/phase/austere_sanctuary": 0.05,
        "experiential/phase/hollow_stillness": 0.88,
        "experiential/phase/bittersweet_release": 0.05,
        "experiential/synchronization/earned_return": 0.02,
    }
    rupture_exp = {
        "experiential/reflection/salience": 0.92,
        "experiential/phase/valence": 0.22,
        "experiential/phase/arousal": 0.84,
    }
    drift = SocialDriftReport(
        drift_detected=True,
        prior_open_scene=1,
        prior_lockdown_scene=4,
        explanation="Sealed off after prior open sharing.",
    )
    vibe_rupture, _ = compile_right_brain_vibe(
        birth,
        rupture_coords,
        rupture_exp,
        rules,
        pass_index=1,
        event_id="ev-rupture",
        social_drift=drift,
    )
    vec_rupture = SubliminalChordVector.from_right_brain_vibe(
        vibe_rupture,
        social_drift=drift,
    )
    assert vec_rupture.social_drift_flag == 1.0
    assert vec_rupture.dissonance_pressure >= 0.50
    assert vec_rupture.cadence_rupture_drive >= 0.75
    assert vec_rupture.with_zeroed_cord().values == tuple(0.0 for _ in range(32))


def test_splice_subliminal_gaps_into_embeds_expands_tokens_and_masks_labels():
    torch.manual_seed(42)
    hidden_size = 16
    vocab_size = 30
    gap_token_id = 29

    word_embed = nn.Embedding(vocab_size, hidden_size)
    projector = SubliminalGapProjector(hidden_size=hidden_size)

    # Sequence of length 5 with `<|subliminal_gap|>` at index 2: [1, 4, GAP, 7, 9]
    input_ids = torch.tensor([[1, 4, gap_token_id, 7, 9]], dtype=torch.long)
    labels = torch.tensor([[IGNORE_LABEL_INDEX, IGNORE_LABEL_INDEX, gap_token_id, 7, 9]], dtype=torch.long)
    chord = torch.linspace(0.0, 1.0, steps=SUBLIMINAL_VECTOR_DIM).unsqueeze(0)  # [1, 32]

    spliced = splice_subliminal_gaps_into_embeds(
        input_ids,
        word_embed,
        projector,
        chord,
        gap_token_id=gap_token_id,
        labels=labels,
    )

    # 5 original tokens - 1 gap token + 4 virtual gap tokens = 8 embedded tokens
    expected_len = 5 - 1 + NUM_VIRTUAL_GAP_TOKENS
    assert spliced.inputs_embeds.shape == (1, expected_len, hidden_size)
    assert spliced.attention_mask.shape == (1, expected_len)
    assert spliced.gap_virtual_spans == (((2, 6),),)
    assert spliced.labels is not None
    # Virtual positions 2, 3, 4, 5 must all be IGNORE_LABEL_INDEX (-100), while positions 6, 7 keep [7, 9]
    assert spliced.labels[0, 2:6].tolist() == [IGNORE_LABEL_INDEX] * 4
    assert spliced.labels[0, 6:].tolist() == [7, 9]


def test_causal_transformer_flips_continuation_purely_from_nonverbal_gap_vector():
    """Prove that identical English tokens before `<|subliminal_gap|>` produce completely
    different post-gap continuations (`smooth_flow` vs. `M-A-C-H-I-N-E` rupture) based
    solely on the 32-float non-verbal Fuzzy Tree vector spliced into the gap.
    """
    torch.manual_seed(7)
    vocab_size = 32
    hidden_size = 32
    gap_token_id = 31

    TOKEN_BOS = 1
    TOKEN_I = 2
    TOKEN_AM = 3
    TOKEN_SMOOTH_FLOW = 10
    TOKEN_RUPTURE_MACHINE = 20

    class _TinyCausalGapLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.word_embed = nn.Embedding(vocab_size, hidden_size)
            self.pos_embed = nn.Embedding(32, hidden_size)
            self.projector = SubliminalGapProjector(hidden_size=hidden_size)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=64,
                dropout=0.0,
                batch_first=True,
            )
            self.decoder = nn.TransformerEncoder(layer, num_layers=2)
            self.lm_head = nn.Linear(hidden_size, vocab_size)

        def forward_spliced(self, input_ids: torch.Tensor, chords: torch.Tensor) -> torch.Tensor:
            batch_spliced = splice_subliminal_gaps_into_embeds(
                input_ids,
                self.word_embed,
                self.projector,
                chords,
                gap_token_id=gap_token_id,
            )
            seq_len = batch_spliced.inputs_embeds.shape[1]
            pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            x = batch_spliced.inputs_embeds + self.pos_embed(pos)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                seq_len,
                device=input_ids.device,
            )
            h = self.decoder(x, mask=causal_mask)
            return self.lm_head(h)

    model = _TinyCausalGapLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    # Identical English prefix for both rows: [BOS, "I", "am", <|subliminal_gap|>]
    identical_prefix = torch.tensor(
        [
            [TOKEN_BOS, TOKEN_I, TOKEN_AM, gap_token_id],
            [TOKEN_BOS, TOKEN_I, TOKEN_AM, gap_token_id],
        ],
        dtype=torch.long,
    )

    # Non-verbal Chord A: Calm Sanctuary (high fairness, high valence, high clarity, low rupture)
    chord_calm = torch.zeros(SUBLIMINAL_VECTOR_DIM)
    chord_calm[3] = 0.92   # cultural/justice/fairness
    chord_calm[14] = 0.88  # valence
    chord_calm[18] = 0.95  # clarity
    chord_calm[24] = 0.85  # austere_sanctuary
    chord_calm[28] = 0.92  # branch_harmony

    # Non-verbal Chord B: Hollow Stillness + High Cadence Rupture (`no soul; M-A-C-H-I-N-E`)
    chord_rupture = torch.zeros(SUBLIMINAL_VECTOR_DIM)
    chord_rupture[11] = 0.90  # discreteness
    chord_rupture[15] = 0.88  # arousal
    chord_rupture[25] = 0.90  # hollow_stillness
    chord_rupture[30] = 0.85  # dissonance_pressure
    chord_rupture[31] = 0.92  # cadence_rupture_drive

    chords = torch.stack([chord_calm, chord_rupture], dim=0)  # [2, 32]
    target_next_tokens = torch.tensor([TOKEN_SMOOTH_FLOW, TOKEN_RUPTURE_MACHINE], dtype=torch.long)

    for _ in range(45):
        optimizer.zero_grad()
        logits = model.forward_spliced(identical_prefix, chords)
        # Predict the token immediately following the 4 virtual `<|subliminal_gap|>` embeddings
        last_step_logits = logits[:, -1, :]
        loss = F.cross_entropy(last_step_logits, target_next_tokens)
        loss.backward()
        optimizer.step()

    # Verify gradients genuinely flowed into SubliminalGapProjector
    proj_grad_norm = sum(
        p.grad.norm().item() for p in model.projector.parameters() if p.grad is not None
    )
    assert proj_grad_norm > 0.0

    # At test time, feed 100% identical `identical_prefix` and swap ONLY the 32-float chord vector:
    with torch.no_grad():
        probs = F.softmax(model.forward_spliced(identical_prefix, chords)[:, -1, :], dim=-1)
        assert probs[0, TOKEN_SMOOTH_FLOW].item() > 0.95
        assert probs[1, TOKEN_RUPTURE_MACHINE].item() > 0.95

        # Now flip the two chords while keeping `identical_prefix` untouched!
        flipped_chords = torch.stack([chord_rupture, chord_calm], dim=0)
        flipped_probs = F.softmax(
            model.forward_spliced(identical_prefix, flipped_chords)[:, -1, :],
            dim=-1,
        )
        assert flipped_probs[0, TOKEN_RUPTURE_MACHINE].item() > 0.95
        assert flipped_probs[1, TOKEN_SMOOTH_FLOW].item() > 0.95
