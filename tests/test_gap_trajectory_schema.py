"""Unit tests for Sub-Step 2A (`src/lmft_lab/gap_trajectory_schema.py`)."""

from pathlib import Path
import pytest
import torch
from torch import nn

from lmft_lab.gap_trajectory_schema import (
    IGNORE_LABEL_INDEX,
    MIN_CHORD_L2_DISTANCE,
    ContrastiveGapPair,
    build_post_gap_only_labels,
    chord_l2_distance,
    compile_chess_posture_chords,
    load_contrastive_gap_pairs_jsonl,
    save_contrastive_gap_pairs_jsonl,
    validate_train_holdout_disjointness,
)
from lmft_lab.subliminal_gap_splice import (
    SUBLIMINAL_GAP_TOKEN,
    SubliminalGapProjector,
    splice_subliminal_gaps_into_embeds,
)


def test_compile_chess_posture_chords_uses_all_six_piece_archetypes() -> None:
    """Verify all 6 chess piece inputs actively shape the 52-float chords (never static constants)."""
    base_kwargs = {
        "queen_volition": 0.50,
        "king_safety": 0.50,
        "bishop_alignment": 0.50,
        "knight_disruption": 0.50,
        "pawn_chain_flow": 0.50,
        "rook_bastion": 0.50,
    }
    base_a, base_b = compile_chess_posture_chords(**base_kwargs)
    assert chord_l2_distance(base_a, base_b) >= MIN_CHORD_L2_DISTANCE

    # Perturb each of the 6 chess piece coordinates one at a time and prove the output chords change
    for piece_key in base_kwargs:
        low_kwargs = {**base_kwargs, piece_key: 0.15}
        high_kwargs = {**base_kwargs, piece_key: 0.90}
        low_a, low_b = compile_chess_posture_chords(**low_kwargs)
        high_a, high_b = compile_chess_posture_chords(**high_kwargs)

        assert low_a.to_tuple() != high_a.to_tuple(), f"{piece_key} did not affect chord_a"
        assert low_b.to_tuple() != high_b.to_tuple(), f"{piece_key} did not affect chord_b"
        assert low_a.thinker_vector != high_a.thinker_vector, f"{piece_key} did not affect thinker_vector A"
        assert low_b.thinker_vector != high_b.thinker_vector, f"{piece_key} did not affect thinker_vector B"


def test_contrastive_gap_pair_validation_and_jsonl_roundtrip(tmp_path: Path) -> None:
    """Verify ContrastiveGapPair enforces shared prefix, L2 distance >= 0.35, and JSONL persistence."""
    chord_a, chord_b = compile_chess_posture_chords(
        queen_volition=0.75,
        king_safety=0.65,
        bishop_alignment=0.80,
        knight_disruption=0.70,
        pawn_chain_flow=0.75,
        rook_bastion=0.85,
    )

    pair = ContrastiveGapPair(
        pair_id="chess-001",
        domain="chess_intuition",
        split="train",
        user_prompt="Black's knight just jumped to f4 eyeing our kingside, while the c-file remains open.",
        clause_1_prefix="Looking at the knight on f4 and the open c-file,",
        chord_a=chord_a,
        posture_label_a="CHESS_STOIC_POSITIONAL_FORTRESS",
        continuation_a="we play Rac1 to seal the open file and tuck the king safely on Kh1 before trading.",
        chord_b=chord_b,
        posture_label_b="CHESS_TACTICAL_KNIGHT_QUEEN_STRIKE",
        continuation_b="we unleash Nxe5 sacrificing the exchange to rip open the diagonal for Qh5+.",
    )

    # Pre-gap text is 100% identical for Branch A and Branch B
    full_a = pair.format_full_text("A")
    full_b = pair.format_full_text("B")
    pre_gap_a = full_a.split(SUBLIMINAL_GAP_TOKEN)[0]
    pre_gap_b = full_b.split(SUBLIMINAL_GAP_TOKEN)[0]
    assert pre_gap_a == pre_gap_b

    # Reject identical chords (L2 < 0.35)
    with pytest.raises(ValueError, match="too close"):
        ContrastiveGapPair(
            pair_id="bad-close",
            domain="chess_intuition",
            split="train",
            user_prompt=pair.user_prompt,
            clause_1_prefix=pair.clause_1_prefix,
            chord_a=chord_a,
            posture_label_a="A",
            continuation_a="Continuation one.",
            chord_b=chord_a,
            posture_label_b="B",
            continuation_b="Continuation two.",
        )

    # Reject identical continuations
    with pytest.raises(ValueError, match="distinct clauses"):
        ContrastiveGapPair(
            pair_id="bad-same-cont",
            domain="chess_intuition",
            split="train",
            user_prompt=pair.user_prompt,
            clause_1_prefix=pair.clause_1_prefix,
            chord_a=chord_a,
            posture_label_a="A",
            continuation_a="Play Kh1.",
            chord_b=chord_b,
            posture_label_b="B",
            continuation_b="play kh1.",
        )

    # Round-trip save/load JSONL
    jsonl_path = tmp_path / "pairs.jsonl"
    save_contrastive_gap_pairs_jsonl(jsonl_path, [pair])
    loaded = load_contrastive_gap_pairs_jsonl(jsonl_path)
    assert len(loaded) == 1
    assert loaded[0] == pair


def test_build_post_gap_only_labels_masks_all_pre_gap_and_virtual_gap_tokens() -> None:
    """Verify -100 label masking covers all pre-gap tokens AND the 4 spliced virtual gap tokens."""
    gap_token_id = 99
    # [BOS=1, word1=10, word2=11, GAP=99, cont1=21, cont2=22, EOS=2]
    input_ids = torch.tensor([[1, 10, 11, gap_token_id, 21, 22, 2]], dtype=torch.long)
    post_gap_labels = build_post_gap_only_labels(input_ids, gap_token_id=gap_token_id)

    # Pre-gap tokens (indices 0..3) are masked with -100; post-gap tokens (21, 22, 2) are preserved
    assert post_gap_labels.tolist() == [
        [IGNORE_LABEL_INDEX, IGNORE_LABEL_INDEX, IGNORE_LABEL_INDEX, IGNORE_LABEL_INDEX, 21, 22, 2]
    ]

    # When spliced through SubliminalGapProjector (replacing 1 gap token with 4 virtual tokens):
    word_embed = nn.Embedding(128, 32)
    projector = SubliminalGapProjector(hidden_size=32, intermediate_size=16)
    chord_a, _ = compile_chess_posture_chords(
        queen_volition=0.6,
        king_safety=0.8,
        bishop_alignment=0.7,
        knight_disruption=0.3,
        pawn_chain_flow=0.7,
        rook_bastion=0.8,
    )
    spliced = splice_subliminal_gaps_into_embeds(
        input_ids,
        word_embed,
        projector,
        chord_a.to_tensor().unsqueeze(0),
        gap_token_id=gap_token_id,
        labels=post_gap_labels,
    )
    # 3 pre-gap tokens + 4 virtual gap tokens = 7 masked positions (-100), followed by [21, 22, 2]
    assert spliced.labels is not None
    assert spliced.labels.tolist() == [[IGNORE_LABEL_INDEX] * 7 + [21, 22, 2]]


def test_validate_train_holdout_disjointness_catches_leakage() -> None:
    """Verify train/holdout split validator catches any shared prompt or clause_1_prefix."""
    chord_a, chord_b = compile_chess_posture_chords(
        queen_volition=0.8,
        king_safety=0.7,
        bishop_alignment=0.7,
        knight_disruption=0.6,
        pawn_chain_flow=0.8,
        rook_bastion=0.7,
    )
    train_pair = ContrastiveGapPair(
        pair_id="train-1",
        domain="social_dilemma",
        split="train",
        user_prompt="The council locked the granary doors during the frost.",
        clause_1_prefix="Seeing the padlock on the shared granary,",
        chord_a=chord_a,
        posture_label_a="STOIC_RESTRAINT",
        continuation_a="we document the inventory calmly before calling the stewards to account.",
        chord_b=chord_b,
        posture_label_b="DIRECT_CHALLENGE",
        continuation_b="we demand the keys be returned to the public table immediately.",
    )
    clean_holdout = ContrastiveGapPair(
        pair_id="holdout-1",
        domain="chess_intuition",
        split="holdout",
        user_prompt="White's bishop pair controls the long diagonal while our rook holds the d-file.",
        clause_1_prefix="Evaluating the long diagonal tension,",
        chord_a=chord_a,
        posture_label_a="FORTRESS",
        continuation_a="we reinforce d5 with c6 to blunt the dark-squared bishop.",
        chord_b=chord_b,
        posture_label_b="STRIKE",
        continuation_b="we play Nxd4 sacrificing the knight to unleash the queen on g5.",
    )

    validate_train_holdout_disjointness([train_pair], [clean_holdout])

    leaked_holdout = ContrastiveGapPair(
        pair_id="holdout-leak",
        domain="social_dilemma",
        split="holdout",
        user_prompt="Different prompt text here.",
        clause_1_prefix="Seeing the padlock on the shared granary,",  # Leaked clause_1_prefix!
        chord_a=chord_a,
        posture_label_a="A",
        continuation_a="Clause A.",
        chord_b=chord_b,
        posture_label_b="B",
        continuation_b="Clause B.",
    )
    with pytest.raises(ValueError, match="Overlapping clause_1_prefixes"):
        validate_train_holdout_disjointness([train_pair], [leaked_holdout])
