"""Chess Kibitzing Board-to-Tree Bridge & Sanctuary Co-Regulation Lead (`chess_kibitz_bridge.py`).

Plain-English Purpose:
1. **Board-Anchored `Fact` vs. Chat `Theory` (Auditor Guardrail #2)**:
   - Uses `python-chess` (`chess.Board`) to read any live FEN position and compute the 6
     Philosophical `Fact` coordinates (`CHESS_PIECE_TO_LEAF_ID`):
       * `Queen`  -> `philosophical/being/agency`       (`Red / Lightning`)
       * `King`   -> `philosophical/being/existence`    (`Cyan / Water`)
       * `Bishop` -> `philosophical/structure/order`    (`Magenta / Wind`)
       * `Knight` -> `philosophical/structure/chaos`    (`Green / Ice`)
       * `Pawn`   -> `philosophical/persistence/continuity`   (`Blue / Fire`)
       * `Rook`   -> `philosophical/persistence/discreteness` (`Yellow / Ground`)
   - Wraps the 3 MiniLM bridges (`BoardAnchoredBridgeSensors`) so that on Pass 1 (`Fact`),
     the 6 Philosophical leaves are anchored to the real chessboard while the Cultural (`doer`)
     and Experiential (`feeler`) leaves come from the MiniLM bridges—and on Clause 1 (comma)
     and Pass 2+ (`Theory`), the MiniLM bridges score the proposed plan/chat text directly.
     If the board says the King is exposed (`existence = 0.18`) and someone proposes a reckless
     attack, the Right Brain catches `shadow_pathology_dissonance` (`Psychosis` / `Paranoia`)
     against the real board.

2. **Un-Rigged Sanctuary Pacing & Leading at the Comma (Auditor Guardrail #1)**:
   - Never overwrites or clamps `vibe.coordinates` (`Fact`) during Sanctuary Co-Regulation turns.
   - Instead, `apply_coregulation_lead_to_chord` steps the **spliced 52-float `SubliminalChordVector`
     right at the comma (`<|subliminal_gap|>`)** by `alpha = 0.25` toward the chosen Sanctuary
     target (`tilt_reset`, `deep_focus`, or `night_wind_down`) so Clause 1 honestly *Paces* the
     user's real state and Clause 2 *Leads* one small step toward equilibrium.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import subprocess
import re
from typing import Any, Mapping
import chess

from .neural_bridge_classifiers import DEFAULT_COMPLEX_FEELING_LEAVES
from .philosophical_abstraction_matrix import (
    CHESS_PIECE_TO_LEAF_ID,
    PHILOSOPHICAL_ABSTRACTION_MATRIX,
    ShadowPathologyWarning,
    SiblingColorBalance,
    compute_philosophical_color_harmony,
    detect_shadow_pathologies,
)
from .reflection_classifier import (
    FOUNDATIONAL_LEAVES,
    FuzzyConceptRule,
    LearnedLeaf,
    _evaluate_rules_with_inhibition,
    compute_contemplation,
)
from .right_brain_equilibrium import ThreeBridgeSensors
from .subliminal_gap_splice import (
    SubliminalChordVector,
    pool_departmental_child_chord,
)

SANCTUARY_MODES: tuple[str, ...] = (
    "tilt_reset",
    "deep_focus",
    "night_wind_down",
)


def _clamp01(value: float) -> float:
    return round(min(1.0, max(0.0, float(value))), 4)


def _lerp01(current: float, target: float, alpha: float) -> float:
    return _clamp01((1.0 - alpha) * float(current) + alpha * float(target))


_PIECE_VALUES: dict[int, float] = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.15,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}


def _static_centipawns_for_color(board: chess.Board, color: chess.Color) -> float:
    """Fast objective centipawn evaluation (`100.0 == 1 pawn`) from `color`'s perspective."""
    if board.is_checkmate():
        return -9999.0 if board.turn == color else 9999.0
    if board.is_stalemate() or board.is_insufficient_material() or board.is_repetition(3):
        return 0.0
    cp = 0.0
    for pt, val in _PIECE_VALUES.items():
        if val <= 0.0:
            continue
        cp += 100.0 * val * (len(board.pieces(pt, color)) - len(board.pieces(pt, not color)))
    for sq in (chess.D4, chess.E4, chess.D5, chess.E5):
        p = board.piece_at(sq)
        if p is not None:
            cp += 18.0 if p.color == color else -18.0
    return round(cp, 1)


def compute_board_valence(
    board: chess.Board,
    perspective: chess.Color | None = None,
) -> float:
    """Map the `+/-` centipawn board evaluation for `perspective` onto `experiential/phase/valence` in `[0.02, 0.98]`.

    - Equal board (`0.0 cp`) -> `0.50`
    - Winning advantage (`+300 cp`) -> `~0.74`, (`+900 cp`) -> `~0.93`, Checkmate delivered -> `0.98`
    - Losing deficit (`-300 cp`) -> `~0.26`, (`-900 cp`) -> `~0.07`, Checkmated -> `0.02`
    """
    side = board.turn if perspective is None else perspective
    if board.is_checkmate():
        return 0.02 if board.turn == side else 0.98
    if board.is_stalemate() or board.is_insufficient_material() or board.is_repetition(3):
        return 0.50
    cp = _static_centipawns_for_color(board, side)
    clamped_cp = max(-2500.0, min(2500.0, float(cp)))
    valence = 0.05 + 0.90 / (1.0 + math.exp(-clamped_cp / 250.0))
    king_sq = board.king(side)
    if king_sq is not None and board.is_attacked_by(not side, king_sq):
        valence -= 0.08
    return _clamp01(min(0.95, max(0.05, valence)))


@dataclass(frozen=True)
class ChessBoardPosture:
    """Real `python-chess` board evaluation mapped to the 6 Philosophical leaves & colors."""

    fen: str
    turn_color: str
    fullmove_number: int
    in_check: bool
    is_game_over: bool
    legal_moves_count: int
    philosophical_coordinates: Mapping[str, float]
    sibling_color_balances: tuple[SiblingColorBalance, ...]
    board_shadow_warnings: tuple[ShadowPathologyWarning, ...]
    dominant_piece: str
    dominant_color: str
    dominant_element: str
    board_summary: str
    board_valence: float = 0.50

    def to_dict(self) -> dict[str, object]:
        return {
            "fen": self.fen,
            "turn_color": self.turn_color,
            "fullmove_number": self.fullmove_number,
            "in_check": self.in_check,
            "is_game_over": self.is_game_over,
            "legal_moves_count": self.legal_moves_count,
            "philosophical_coordinates": dict(self.philosophical_coordinates),
            "sibling_color_balances": [b.to_dict() for b in self.sibling_color_balances],
            "board_shadow_warnings": [w.to_dict() for w in self.board_shadow_warnings],
            "dominant_piece": self.dominant_piece,
            "dominant_color": self.dominant_color,
            "dominant_element": self.dominant_element,
            "board_summary": self.board_summary,
            "board_valence": round(float(self.board_valence), 4),
        }


def evaluate_chess_board_posture(
    board_or_fen: chess.Board | str | None = None,
    *,
    perspective: chess.Color | None = None,
) -> ChessBoardPosture:
    """Compute the 6 Philosophical `Fact` coordinates directly from a live `python-chess` board."""
    if board_or_fen is None:
        board = chess.Board()
    elif isinstance(board_or_fen, str):
        board = chess.Board(board_or_fen.strip()) if board_or_fen.strip() else chess.Board()
    else:
        board = board_or_fen.copy()

    us = board.turn if perspective is None else perspective
    them = not us
    turn_label = "White" if board.turn == chess.WHITE else "Black"

    legal_moves = list(board.legal_moves)
    if board.turn != us:
        temp_b = board.copy()
        temp_b.turn = us
        our_moves = list(temp_b.legal_moves)
        check_board = temp_b
    else:
        our_moves = legal_moves
        check_board = board

    home_rank = 0 if us == chess.WHITE else 7
    our_bishops = list(board.pieces(chess.BISHOP, us))
    our_knights = list(board.pieces(chess.KNIGHT, us))
    developed_bishops = sum(1 for sq in our_bishops if chess.square_rank(sq) != home_rank)
    developed_knights = sum(1 for sq in our_knights if chess.square_rank(sq) != home_rank)
    minor_total = max(1, len(our_bishops) + len(our_knights))
    minor_dev_ratio = (developed_bishops + developed_knights) / minor_total

    # 1. Queen -> philosophical/being/agency (Red / Lightning: Volition & Attacking Drive)
    our_queens = list(board.pieces(chess.QUEEN, us))
    queen_moves = [m for m in our_moves if board.piece_type_at(m.from_square) == chess.QUEEN]
    capture_moves = [m for m in our_moves if board.is_capture(m)]
    checking_moves = [m for m in our_moves if check_board.gives_check(m)]
    enemy_half_ranks = range(4, 8) if us == chess.WHITE else range(0, 4)
    forward_attacks = [
        m
        for m in our_moves
        if chess.square_rank(m.to_square) in enemy_half_ranks
    ]
    queen_presence = (0.32 + 0.14 * min(1, len(our_queens) - 1)) if our_queens else 0.08
    queen_mob_scale = (0.008 + 0.025 * minor_dev_ratio) if board.fullmove_number <= 8 else 0.035
    agency = _clamp01(
        queen_presence
        + queen_mob_scale * min(12, len(queen_moves))
        + 0.06 * min(5, len(capture_moves))
        + 0.12 * min(3, len(checking_moves))
        + 0.015 * min(12, len(forward_attacks))
    )

    # 2. King -> philosophical/being/existence (Cyan / Water: Wellspring of Being & King Shelter)
    king_sq = board.king(us)
    if king_sq is None:
        existence = 0.10
        enemy_king_attackers = 0
        shield_pawns = 0
        in_check_now = False
    else:
        king_file = chess.square_file(king_sq)
        king_rank = chess.square_rank(king_sq)
        shield_rank = king_rank + (1 if us == chess.WHITE else -1)
        shield_pawns = 0
        if 0 <= shield_rank <= 7:
            for df in (-1, 0, 1):
                f = king_file + df
                if 0 <= f <= 7:
                    sq = chess.square(f, shield_rank)
                    p = board.piece_at(sq)
                    if p is not None and p.color == us and p.piece_type == chess.PAWN:
                        shield_pawns += 1

        king_ring = chess.SquareSet(chess.BB_KING_ATTACKS[king_sq])
        enemy_king_attackers = 0
        safe_escape_squares = 0
        for sq in king_ring:
            if board.is_attacked_by(them, sq):
                enemy_king_attackers += 1
            else:
                occupant = board.piece_at(sq)
                if occupant is None or occupant.color != us:
                    safe_escape_squares += 1

        in_check_now = board.is_attacked_by(them, king_sq)
        back_rank_home = ((king_rank in (0, 1)) if us == chess.WHITE else (king_rank in (6, 7))) and not in_check_now
        castled_shelter = 0.08 if (king_rank == home_rank and king_file in (2, 6) and not in_check_now) else 0.0
        existence = _clamp01(
            0.48
            + 0.11 * min(3, shield_pawns)
            + (0.10 if back_rank_home else -0.12)
            + castled_shelter
            + 0.03 * min(4, safe_escape_squares)
            - 0.11 * min(5, enemy_king_attackers)
            - (0.36 if in_check_now else 0.0)
        )

    # 3. Bishop -> philosophical/structure/order (Magenta / Wind: Alignment & Coordination)
    center_ring = {
        chess.C3, chess.D3, chess.E3, chess.F3,
        chess.C4, chess.D4, chess.E4, chess.F4,
        chess.C5, chess.D5, chess.E5, chess.F5,
        chess.C6, chess.D6, chess.E6, chess.F6,
    }
    active_knights = sum(1 for sq in our_knights if sq in center_ring)
    rim_knights = sum(1 for sq in our_knights if chess.square_file(sq) in (0, 7))
    bishop_moves = [m for m in our_moves if board.piece_type_at(m.from_square) == chess.BISHOP]
    our_non_king_sqs = [
        sq
        for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
        for sq in board.pieces(pt, us)
    ]
    defended_count = sum(1 for sq in our_non_king_sqs if board.is_attacked_by(us, sq))
    hanging_count = sum(
        1
        for sq in our_non_king_sqs
        if board.is_attacked_by(them, sq)
        and not board.is_attacked_by(us, sq)
        and not (
            board.piece_type_at(sq) == chess.PAWN
            and any(m.from_square == sq and board.is_capture(m) for m in our_moves)
        )
    )
    coord_ratio = defended_count / max(1, len(our_non_king_sqs))
    order = _clamp01(
        0.28
        + 0.08 * min(2, len(our_bishops))
        + 0.06 * min(2, developed_bishops)
        + 0.07 * min(2, active_knights)
        - 0.08 * min(2, rim_knights)
        + 0.03 * min(10, len(bishop_moves))
        + 0.35 * coord_ratio
        - 0.14 * min(3, hanging_count)
    )

    # 4. Knight -> philosophical/structure/chaos (Green / Ice: Nonlinear Forks & Tactical Shatter)
    knight_moves = [m for m in our_moves if board.piece_type_at(m.from_square) == chess.KNIGHT]
    chaos = _clamp01(
        0.12
        + 0.03 * min(2, active_knights)
        + 0.06 * min(2, rim_knights)
        + 0.015 * min(8, len(knight_moves))
        + 0.04 * min(5, len(capture_moves))
        + 0.06 * min(3, len(checking_moves))
        + 0.12 * min(3, hanging_count)
        + (0.22 if in_check_now else 0.0)
    )

    # 5. Pawn -> philosophical/persistence/continuity (Blue / Fire: Connected Phalanx & Flow)
    our_pawns = list(board.pieces(chess.PAWN, us))
    their_pawns = list(board.pieces(chess.PAWN, them))
    pawn_set = set(our_pawns)
    connected_pawns = 0
    passed_pawns = 0
    advanced_passed = 0
    for sq in our_pawns:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        neighbors = [
            chess.square(nf, nr)
            for nf in (f - 1, f + 1)
            for nr in (r - 1, r, r + 1)
            if 0 <= nf <= 7 and 0 <= nr <= 7
        ]
        if any(nsq in pawn_set for nsq in neighbors):
            connected_pawns += 1
        ahead_ranks = range(r + 1, 8) if us == chess.WHITE else range(0, r)
        is_passed = not any(
            abs(chess.square_file(ep) - f) <= 1 and chess.square_rank(ep) in ahead_ranks
            for ep in their_pawns
        )
        if is_passed:
            passed_pawns += 1
            rel_r = r if us == chess.WHITE else (7 - r)
            if rel_r >= 4:
                advanced_passed += (rel_r - 3)
    center_four = {chess.D4, chess.E4, chess.D5, chess.E5}
    center_pawns = sum(1 for sq in our_pawns if sq in center_four)
    continuity = _clamp01(
        0.24
        + 0.055 * min(8, connected_pawns)
        + 0.12 * min(4, center_pawns)
        + 0.06 * min(3, passed_pawns)
        + 0.08 * min(4, advanced_passed)
        + 0.02 * min(8, len(our_pawns))
    )

    # 6. Rook -> philosophical/persistence/discreteness (Yellow / Ground: Quantized Bastions & Open Files)
    our_rooks = list(board.pieces(chess.ROOK, us))
    rook_moves = [m for m in our_moves if board.piece_type_at(m.from_square) == chess.ROOK]
    open_file_rooks = 0
    seventh_rank = 6 if us == chess.WHITE else 1
    seventh_rank_rooks = 0
    for rsq in our_rooks:
        rf = chess.square_file(rsq)
        rr = chess.square_rank(rsq)
        if rr == seventh_rank:
            seventh_rank_rooks += 1
        friendly_pawn_on_file = any(chess.square_file(psq) == rf for psq in our_pawns)
        if not friendly_pawn_on_file:
            open_file_rooks += 1
    discreteness = _clamp01(
        0.28
        + 0.08 * min(2, len(our_rooks))
        + 0.14 * min(2, open_file_rooks)
        + 0.14 * min(2, seventh_rank_rooks)
        + 0.025 * min(10, len(rook_moves))
    )

    phil_coords = {
        CHESS_PIECE_TO_LEAF_ID["Queen"]: agency,
        CHESS_PIECE_TO_LEAF_ID["King"]: existence,
        CHESS_PIECE_TO_LEAF_ID["Bishop"]: order,
        CHESS_PIECE_TO_LEAF_ID["Knight"]: chaos,
        CHESS_PIECE_TO_LEAF_ID["Pawn"]: continuity,
        CHESS_PIECE_TO_LEAF_ID["Rook"]: discreteness,
    }

    balances = compute_philosophical_color_harmony(phil_coords)
    warnings = detect_shadow_pathologies(
        phil_coords,
        fact=0.25 if (in_check_now or hanging_count > 0 or existence <= 0.35) else 0.65,
        arousal=max(0.45, chaos),
        resonance=order,
    )

    ranked_rows = sorted(
        PHILOSOPHICAL_ABSTRACTION_MATRIX,
        key=lambda r: phil_coords.get(r.leaf_id, 0.0),
        reverse=True,
    )
    top_row = ranked_rows[0]

    board_val = compute_board_valence(board, perspective=us)
    check_tag = " (IN CHECK)" if board.is_check() else ""
    summary = (
        f"{turn_label} to move{check_tag} | Dominant: {top_row.chess_piece} "
        f"({top_row.color_name} / {top_row.element} = {phil_coords[top_row.leaf_id]:.2f}) | "
        f"Valence={board_val:.2f}, King Shelter={existence:.2f}, Attack Volition={agency:.2f}, Order={order:.2f}, Chaos={chaos:.2f}"
    )

    return ChessBoardPosture(
        fen=board.fen(),
        turn_color=turn_label,
        fullmove_number=int(board.fullmove_number),
        in_check=bool(board.is_check()),
        is_game_over=bool(board.is_game_over()),
        legal_moves_count=len(legal_moves),
        philosophical_coordinates=phil_coords,
        sibling_color_balances=balances,
        board_shadow_warnings=warnings,
        dominant_piece=top_row.chess_piece,
        dominant_color=top_row.color_name,
        dominant_element=top_row.element,
        board_summary=summary,
        board_valence=board_val,
    )


_SAN_TOKEN_REGEX = re.compile(
    r"(?<![A-Za-z0-9_-])"
    r"("
    r"(?:O-O-O|O-O|0-0-0|0-0)[+#]?"
    r"|[KQRBN][a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?"
    r"|[a-h]x[a-h][1-8](?:=[QRBN])?[+#]?"
    r"|[a-h][1-8](?:=[QRBN])?[+#]?"
    r")"
    r"(?=$|[\s,.;:!?)\]])(?![-A-Za-z0-9_])"
)

_BB_EXTENDED_CENTER: int = (
    chess.BB_C3 | chess.BB_D3 | chess.BB_E3 | chess.BB_F3
    | chess.BB_C4 | chess.BB_D4 | chess.BB_E4 | chess.BB_F4
    | chess.BB_C5 | chess.BB_D5 | chess.BB_E5 | chess.BB_F5
    | chess.BB_C6 | chess.BB_D6 | chess.BB_E6 | chess.BB_F6
)


def _sigmoid01(x: float) -> float:
    x_clamped = max(-12.0, min(12.0, float(x)))
    return _clamp01(1.0 / (1.0 + math.exp(-x_clamped)))


def _union_attacks_bb(board: chess.Board, color: chess.Color) -> int:
    mask = 0
    for sq in chess.scan_forward(board.occupied_co[color]):
        mask |= board.attacks_mask(sq)
    return mask


def extract_bitboard_features(
    board: chess.Board,
    move: chess.Move,
    *,
    perspective: chess.Color | None = None,
    next_board: chess.Board | None = None,
    blunder_penalty: float = 0.0,
    captured_val: float = 0.0,
    recapture_loss: float = 0.0,
    net_capture_gain: float | None = None,
) -> dict[str, float]:
    """Extract normalized 64-square bitboard features in O(1) without lookahead or engine calls."""
    us = board.turn if perspective is None else perspective
    them = not us
    nb = next_board if next_board is not None else board.copy()
    if next_board is None:
        nb.push(move)

    promo_bonus = (
        max(0.0, _PIECE_VALUES.get(move.promotion, 9.0) - 1.0 - float(recapture_loss))
        if move.promotion
        else 0.0
    )
    eff_capture_gain = (
        float(net_capture_gain)
        if net_capture_gain is not None
        else max(0.0, float(captured_val) + promo_bonus - float(recapture_loss) - float(blunder_penalty))
    )

    our_mat = sum(
        val * chess.popcount(nb.pieces_mask(pt, us))
        for pt, val in _PIECE_VALUES.items()
        if val > 0.0
    )
    their_mat = sum(
        val * chess.popcount(nb.pieces_mask(pt, them))
        for pt, val in _PIECE_VALUES.items()
        if val > 0.0
    )
    effective_mat_diff = (our_mat - their_mat) - float(recapture_loss) - max(0.0, float(blunder_penalty) - float(recapture_loss))
    material_balance_norm = max(-1.0, min(1.0, effective_mat_diff / 12.0))

    home_rank_bb = chess.BB_RANK_1 if us == chess.WHITE else chess.BB_RANK_8
    our_minors_bb = nb.pieces_mask(chess.KNIGHT, us) | nb.pieces_mask(chess.BISHOP, us)
    total_minors = max(1, chess.popcount(our_minors_bb))
    developed_minors = chess.popcount(our_minors_bb & ~home_rank_bb)
    piece_type = board.piece_type_at(move.from_square) or chess.PAWN

    our_attacks_bb = _union_attacks_bb(nb, us)
    their_attacks_bb = _union_attacks_bb(nb, them)
    our_att_cnt = chess.popcount(our_attacks_bb)
    their_att_cnt = chess.popcount(their_attacks_bb)
    attack_control_share = our_att_cnt / max(1, our_att_cnt + their_att_cnt)

    our_center_occ = chess.popcount(nb.occupied_co[us] & chess.BB_CENTER)
    their_center_occ = chess.popcount(nb.occupied_co[them] & chess.BB_CENTER)
    our_center_att = chess.popcount(our_attacks_bb & _BB_EXTENDED_CENTER)
    their_center_att = chess.popcount(their_attacks_bb & _BB_EXTENDED_CENTER)
    center_bitboard_control = max(
        -1.0,
        min(
            1.0,
            ((2.0 * our_center_occ + 0.35 * our_center_att) - (2.0 * their_center_occ + 0.35 * their_center_att))
            / 6.0,
        ),
    )

    king_sq = nb.king(us)
    if king_sq is None:
        king_shield_safety = 0.10
    else:
        king_ring_bb = chess.BB_KING_ATTACKS[king_sq]
        shield_pawns = chess.popcount(nb.pieces_mask(chess.PAWN, us) & king_ring_bb)
        ring_enemy_att = chess.popcount(their_attacks_bb & king_ring_bb)
        in_check_pen = 0.35 if nb.is_attacked_by(them, king_sq) else 0.0
        king_shield_safety = _clamp01(0.52 + 0.14 * min(3, shield_pawns) - 0.10 * min(4, ring_enemy_att) - in_check_pen)

    our_non_king_bb = nb.occupied_co[us] & ~nb.pieces_mask(chess.KING, us)
    non_king_cnt = max(1, chess.popcount(our_non_king_bb))
    defended_cnt = chess.popcount(our_non_king_bb & our_attacks_bb)
    piece_coordination_ratio = _clamp01(defended_cnt / non_king_cnt)

    castled_bonus = 0.25 if (king_sq is not None and chess.square_file(king_sq) in (2, 6)) else 0.0
    development_progress = _clamp01(0.75 * (developed_minors / total_minors) + castled_bonus)

    move_net_safety = max(-1.0, min(1.0, 0.45 + 0.28 * eff_capture_gain - 0.48 * blunder_penalty))

    to_bb = chess.BB_SQUARES[move.to_square]
    from_bb = chess.BB_SQUARES[move.from_square]
    pos_gain = 0.0
    if board.is_castling(move):
        pos_gain += 0.65
    if piece_type == chess.PAWN and (to_bb & chess.BB_CENTER):
        pos_gain += 0.55
    elif piece_type in (chess.KNIGHT, chess.BISHOP) and (from_bb & home_rank_bb) and not (to_bb & home_rank_bb):
        pos_gain += 0.45
    if to_bb & _BB_EXTENDED_CENTER:
        pos_gain += 0.22
    if not nb.is_attacked_by(them, move.to_square):
        pos_gain += 0.28
    elif nb.attackers_mask(them, move.to_square) & nb.pieces_mask(chess.PAWN, them):
        pos_gain -= 0.32
    move_positional_gain = max(-1.0, min(1.0, pos_gain))

    return {
        "material_balance_norm": round(float(material_balance_norm), 4),
        "attack_control_share": round(float(attack_control_share), 4),
        "center_bitboard_control": round(float(center_bitboard_control), 4),
        "king_shield_safety": round(float(king_shield_safety), 4),
        "piece_coordination_ratio": round(float(piece_coordination_ratio), 4),
        "development_progress": round(float(development_progress), 4),
        "move_net_safety": round(float(move_net_safety), 4),
        "move_positional_gain": round(float(move_positional_gain), 4),
    }


DEFAULT_SLEEVE_VALUE_WEIGHTS: dict[str, float] = {
    "material_balance_norm": 1.35,
    "attack_control_share": 0.90,
    "center_bitboard_control": 0.65,
    "king_shield_safety": 0.95,
    "piece_coordination_ratio": 0.70,
    "development_progress": 0.45,
    "move_net_safety": 1.55,
    "move_positional_gain": 0.40,
}

DEFAULT_SLEEVE_POLICY_WEIGHTS: dict[str, float] = {
    "material_balance_norm": 0.55,
    "attack_control_share": 0.85,
    "center_bitboard_control": 1.10,
    "king_shield_safety": 0.65,
    "piece_coordination_ratio": 0.80,
    "development_progress": 0.95,
    "move_net_safety": 1.15,
    "move_positional_gain": 1.45,
}

DEFAULT_SLEEVE_LEAF_WEIGHTS: dict[str, float] = {
    "philosophical/being/existence": 0.65,
    "philosophical/structure/order": 0.60,
    "philosophical/persistence/continuity": 0.55,
    "philosophical/persistence/discreteness": 0.45,
    "philosophical/being/agency": 0.35,
    "philosophical/structure/chaos": -0.35,
    "cultural/efficiency/wastelessness": 0.80,
    "cultural/freedom/self-sufficiency": 0.45,
}


@dataclass
class BitboardIntuitionSleeve:
    """Single-pass AlphaZero/Leela-style Bitboard Intuition Sleeve feeding `Confidence`, `Fact`, and `Theory`.

    Operates in O(1) strictly from 64-bit bitboard features and learned post-mortem weights
    (zero lookahead tree search and zero Stockfish queries during live move selection).
    """

    value_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SLEEVE_VALUE_WEIGHTS))
    policy_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SLEEVE_POLICY_WEIGHTS))
    leaf_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SLEEVE_LEAF_WEIGHTS))
    value_bias: float = -0.55
    policy_bias: float = -0.45
    steps_distilled: int = 0

    def evaluate_move(
        self,
        board: chess.Board,
        move: chess.Move,
        *,
        perspective: chess.Color | None = None,
        next_board: chess.Board | None = None,
        blunder_penalty: float = 0.0,
        captured_val: float = 0.0,
        recapture_loss: float = 0.0,
        net_capture_gain: float | None = None,
        leaf_coords: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        feats = extract_bitboard_features(
            board,
            move,
            perspective=perspective,
            next_board=next_board,
            blunder_penalty=blunder_penalty,
            captured_val=captured_val,
            recapture_loss=recapture_loss,
            net_capture_gain=net_capture_gain,
        )
        v_logit = float(self.value_bias) + sum(
            float(self.value_weights.get(k, 0.0)) * float(v) for k, v in feats.items()
        )
        p_logit = float(self.policy_bias) + sum(
            float(self.policy_weights.get(k, 0.0)) * float(v) for k, v in feats.items()
        )
        leaf_adj = 0.0
        if leaf_coords:
            for lk, lw in self.leaf_weights.items():
                if lk in leaf_coords:
                    leaf_adj += float(lw) * (float(leaf_coords[lk]) - 0.50)
        value_prior = _sigmoid01(v_logit + 0.55 * leaf_adj)
        policy_prior = _sigmoid01(p_logit + 0.75 * leaf_adj)
        safety_gate = 0.25 if blunder_penalty >= 1.5 else (0.55 if blunder_penalty >= 0.8 else 1.0)
        coherence = 1.0 - min(1.0, abs(value_prior - policy_prior))
        sleeve_confidence = _clamp01(
            safety_gate * (0.45 * value_prior + 0.35 * policy_prior + 0.20 * coherence)
        )
        return {
            "value_prior": value_prior,
            "policy_prior": policy_prior,
            "sleeve_confidence": sleeve_confidence,
            "leaf_adjustment": round(float(leaf_adj), 4),
            "features": feats,
        }

    def _renormalize_budgets(self) -> None:
        """Keep total value, policy, and positive leaf weight budgets anchored near birth sums so logits never saturate."""
        target_v_sum = sum(DEFAULT_SLEEVE_VALUE_WEIGHTS.values())
        for k in DEFAULT_SLEEVE_VALUE_WEIGHTS:
            self.value_weights[k] = max(0.20, min(2.20, float(self.value_weights.get(k, DEFAULT_SLEEVE_VALUE_WEIGHTS[k]))))
        cur_v_sum = sum(self.value_weights.get(k, 0.0) for k in DEFAULT_SLEEVE_VALUE_WEIGHTS)
        if cur_v_sum > target_v_sum * 1.15 or cur_v_sum < target_v_sum * 0.85:
            scale_v = target_v_sum / max(1e-6, cur_v_sum)
            for k in DEFAULT_SLEEVE_VALUE_WEIGHTS:
                self.value_weights[k] = round(max(0.20, min(2.20, self.value_weights[k] * scale_v)), 4)
        else:
            for k in DEFAULT_SLEEVE_VALUE_WEIGHTS:
                self.value_weights[k] = round(self.value_weights[k], 4)

        target_p_sum = sum(DEFAULT_SLEEVE_POLICY_WEIGHTS.values())
        for k in DEFAULT_SLEEVE_POLICY_WEIGHTS:
            self.policy_weights[k] = max(0.20, min(2.20, float(self.policy_weights.get(k, DEFAULT_SLEEVE_POLICY_WEIGHTS[k]))))
        cur_p_sum = sum(self.policy_weights.get(k, 0.0) for k in DEFAULT_SLEEVE_POLICY_WEIGHTS)
        if cur_p_sum > target_p_sum * 1.15 or cur_p_sum < target_p_sum * 0.85:
            scale_p = target_p_sum / max(1e-6, cur_p_sum)
            for k in DEFAULT_SLEEVE_POLICY_WEIGHTS:
                self.policy_weights[k] = round(max(0.20, min(2.20, self.policy_weights[k] * scale_p)), 4)
        else:
            for k in DEFAULT_SLEEVE_POLICY_WEIGHTS:
                self.policy_weights[k] = round(self.policy_weights[k], 4)

        pos_leaf_keys = [k for k, v in DEFAULT_SLEEVE_LEAF_WEIGHTS.items() if v > 0.0]
        target_l_sum = sum(DEFAULT_SLEEVE_LEAF_WEIGHTS[k] for k in pos_leaf_keys)
        for k in pos_leaf_keys:
            self.leaf_weights[k] = max(0.20, min(2.45, float(self.leaf_weights.get(k, DEFAULT_SLEEVE_LEAF_WEIGHTS[k]))))
        cur_l_sum = sum(self.leaf_weights.get(k, 0.0) for k in pos_leaf_keys)
        if cur_l_sum > target_l_sum * 1.35 or cur_l_sum < target_l_sum * 0.75:
            scale_l = (target_l_sum * 1.15 if cur_l_sum > target_l_sum * 1.35 else target_l_sum * 0.85) / max(1e-6, cur_l_sum)
            for k in pos_leaf_keys:
                self.leaf_weights[k] = round(max(0.20, min(2.45, self.leaf_weights[k] * scale_l)), 4)
        else:
            for k in pos_leaf_keys:
                self.leaf_weights[k] = round(self.leaf_weights[k], 4)
        if "philosophical/structure/chaos" in self.leaf_weights:
            self.leaf_weights["philosophical/structure/chaos"] = round(
                max(-2.45, min(-0.20, float(self.leaf_weights["philosophical/structure/chaos"]))),
                4,
            )

    def distill_post_mortem_scene(
        self,
        played_coordinates: Mapping[str, float],
        reference_coordinates: Mapping[str, float] | None,
        *,
        is_hazard: bool = True,
        learning_rate: float = 0.08,
        played_bitboard_features: Mapping[str, float] | None = None,
        reference_bitboard_features: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        """Distill a post-mortem Game Review critique (`played` vs `best_alternative`) into the sleeve weights.

        Uses two-way signed contrastive updates (rather than one-way `max(0, ...)` ratchets) plus
        homeostatic anchor decay and total-budget normalization so weights redistribute toward
        discriminative features without ever saturating at the rails.
        """
        lr = max(0.005, min(0.25, float(learning_rate)))
        decay = 0.08 * lr
        for vk, def_v in DEFAULT_SLEEVE_VALUE_WEIGHTS.items():
            cur_v = float(self.value_weights.get(vk, def_v))
            self.value_weights[vk] = round(
                max(0.20, min(2.45, cur_v + decay * (float(def_v) - cur_v))),
                4,
            )
        for pk, def_p in DEFAULT_SLEEVE_POLICY_WEIGHTS.items():
            cur_p = float(self.policy_weights.get(pk, def_p))
            self.policy_weights[pk] = round(
                max(0.20, min(2.45, cur_p + decay * (float(def_p) - cur_p))),
                4,
            )
        for lk, def_l in DEFAULT_SLEEVE_LEAF_WEIGHTS.items():
            cur_l = float(self.leaf_weights.get(lk, def_l))
            blended_l = cur_l + decay * (float(def_l) - cur_l)
            if lk == "philosophical/structure/chaos":
                self.leaf_weights[lk] = round(max(-2.45, min(-0.20, blended_l)), 4)
            else:
                self.leaf_weights[lk] = round(max(0.20, min(2.45, blended_l)), 4)

        updated_keys: list[str] = []
        updated_circuit_keys: list[str] = []
        ref_map = reference_coordinates or {}
        safety_virtues = {"philosophical/being/existence", "cultural/efficiency/wastelessness"}
        for lk in list(self.leaf_weights.keys()):
            if lk in played_coordinates and lk in ref_map:
                p_val = float(played_coordinates[lk])
                r_val = float(ref_map[lk])
                raw_delta = (r_val - p_val) if is_hazard else (p_val - r_val)
                if lk == "philosophical/structure/chaos":
                    delta = raw_delta if raw_delta < 0.0 else 0.50 * raw_delta
                    if abs(delta) >= 0.01:
                        new_w = max(-2.45, min(-0.20, float(self.leaf_weights[lk]) + lr * delta))
                        self.leaf_weights[lk] = round(new_w, 4)
                        updated_keys.append(lk)
                else:
                    # Two-way signed contrastive update (attenuating negative steps on core safety virtues)
                    delta = raw_delta if (raw_delta >= 0.0 or lk not in safety_virtues) else 0.35 * raw_delta
                    if abs(delta) >= 0.01:
                        new_w = max(0.20, min(2.45, float(self.leaf_weights[lk]) + lr * delta))
                        self.leaf_weights[lk] = round(new_w, 4)
                        updated_keys.append(lk)

        if played_bitboard_features and reference_bitboard_features:
            for fk in DEFAULT_SLEEVE_VALUE_WEIGHTS:
                if fk in played_bitboard_features and fk in reference_bitboard_features:
                    p_feat = float(played_bitboard_features[fk])
                    r_feat = float(reference_bitboard_features[fk])
                    feat_delta = (r_feat - p_feat) if is_hazard else (p_feat - r_feat)
                    if abs(feat_delta) >= 0.01:
                        eff_delta = feat_delta if feat_delta >= 0.0 else 0.65 * feat_delta
                        self.value_weights[fk] = round(
                            max(0.20, min(2.45, float(self.value_weights.get(fk, DEFAULT_SLEEVE_VALUE_WEIGHTS[fk])) + 0.45 * lr * eff_delta)),
                            4,
                        )
                        self.policy_weights[fk] = round(
                            max(0.20, min(2.45, float(self.policy_weights.get(fk, DEFAULT_SLEEVE_POLICY_WEIGHTS[fk])) + 0.55 * lr * eff_delta)),
                            4,
                        )
                        updated_circuit_keys.append(fk)

        fact_p = float(played_coordinates.get("psychological/truth/fact", 0.50))
        fact_r = float(ref_map.get("psychological/truth/fact", 0.70))
        if not (played_bitboard_features and reference_bitboard_features):
            if is_hazard and abs(fact_r - fact_p) >= 0.02:
                fact_delta = fact_r - fact_p
                self.value_weights["move_net_safety"] = round(
                    max(0.20, min(2.45, float(self.value_weights.get("move_net_safety", 1.55)) + 0.5 * lr * fact_delta)),
                    4,
                )
                self.policy_weights["move_net_safety"] = round(
                    max(0.20, min(2.45, float(self.policy_weights.get("move_net_safety", 1.15)) + 0.5 * lr * fact_delta)),
                    4,
                )
                if "move_net_safety" not in updated_circuit_keys:
                    updated_circuit_keys.append("move_net_safety")
            elif not is_hazard and abs(fact_p - fact_r) >= 0.02:
                pos_delta = fact_p - fact_r
                self.policy_weights["move_positional_gain"] = round(
                    max(0.20, min(2.45, float(self.policy_weights.get("move_positional_gain", 1.45)) + 0.35 * lr * pos_delta)),
                    4,
                )
                if "move_positional_gain" not in updated_circuit_keys:
                    updated_circuit_keys.append("move_positional_gain")
        self._renormalize_budgets()
        self.steps_distilled += 1
        return {
            "steps_distilled": self.steps_distilled,
            "updated_leaf_keys": updated_keys,
            "updated_circuit_keys": updated_circuit_keys,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "value_weights": dict(self.value_weights),
            "policy_weights": dict(self.policy_weights),
            "leaf_weights": dict(self.leaf_weights),
            "value_bias": round(float(self.value_bias), 4),
            "policy_bias": round(float(self.policy_bias), 4),
            "steps_distilled": int(self.steps_distilled),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> BitboardIntuitionSleeve:
        vw = dict(DEFAULT_SLEEVE_VALUE_WEIGHTS)
        pw = dict(DEFAULT_SLEEVE_POLICY_WEIGHTS)
        lw = dict(DEFAULT_SLEEVE_LEAF_WEIGHTS)
        if isinstance(raw.get("value_weights"), Mapping):
            for k, v in raw["value_weights"].items():
                vw[str(k)] = max(0.20, float(v))
        if isinstance(raw.get("policy_weights"), Mapping):
            for k, v in raw["policy_weights"].items():
                pw[str(k)] = max(0.20, float(v))
        if isinstance(raw.get("leaf_weights"), Mapping):
            for k, v in raw["leaf_weights"].items():
                key = str(k)
                val = float(v)
                if key == "philosophical/structure/chaos":
                    lw[key] = min(-0.20, max(-2.45, val))
                elif key in DEFAULT_SLEEVE_LEAF_WEIGHTS:
                    lw[key] = max(0.20, min(2.45, val))
                else:
                    lw[key] = val
        inst = cls(
            value_weights=vw,
            policy_weights=pw,
            leaf_weights=lw,
            value_bias=float(raw.get("value_bias", -0.55)),
            policy_bias=float(raw.get("policy_bias", -0.45)),
            steps_distilled=int(raw.get("steps_distilled", 0)),
        )
        inst._renormalize_budgets()
        return inst

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BitboardIntuitionSleeve:
        target = Path(path)
        if not target.exists():
            return cls()
        try:
            return cls.from_dict(json.loads(target.read_text(encoding="utf-8")))
        except Exception:
            return cls()


_DEFAULT_BITBOARD_SLEEVE = BitboardIntuitionSleeve()


@dataclass(frozen=True)
class CandidateMoveEvaluation:
    """Right-Brain 40-node evaluation of a legal candidate move Action Bid on `psychological/will/execution`."""

    san: str
    uci: str
    composite_score: float
    settled: bool
    clarity: float
    confidence: float
    matched_concept: str | None
    dominant_piece: str
    dominant_color: str
    is_blunder: bool
    is_repetition: bool
    shadow_warnings: tuple[str, ...]
    execution_bid: float = 0.0
    fact: float = 0.5
    theory: float = 0.5
    intention: float = 0.5
    vow_alignment_delta: float = 0.0
    guided_by_vow: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "san": self.san,
            "uci": self.uci,
            "composite_score": round(float(self.composite_score), 4),
            "execution_bid": round(float(self.execution_bid or self.composite_score), 4),
            "fact": round(float(self.fact), 4),
            "theory": round(float(self.theory), 4),
            "intention": round(float(self.intention), 4),
            "settled": bool(self.settled),
            "clarity": round(float(self.clarity), 4),
            "confidence": round(float(self.confidence), 4),
            "matched_concept": self.matched_concept,
            "dominant_piece": self.dominant_piece,
            "dominant_color": self.dominant_color,
            "is_blunder": bool(self.is_blunder),
            "is_repetition": bool(self.is_repetition),
            "shadow_warnings": list(self.shadow_warnings),
            "vow_alignment_delta": round(float(self.vow_alignment_delta), 4),
            "guided_by_vow": bool(self.guided_by_vow),
        }


def _compute_army_mobilization_and_sufficiency(
    board_before: chess.Board,
    move: chess.Move,
    next_board: chess.Board,
    *,
    us: chess.Color,
    existence: float,
    continuity: float,
    piece_coord_ratio: float,
    net_capture_gain: float,
    is_blunder: bool,
) -> tuple[float, bool, bool, float]:
    """Compute whole-army participation, premature-solo detection, and phase-aware `self-sufficiency`."""
    them = not us
    home_rank = 0 if us == chess.WHITE else 7
    seventh_rank = 6 if us == chess.WHITE else 1
    corner_sqs = {chess.A1, chess.H1} if us == chess.WHITE else {chess.A8, chess.H8}

    our_pawns = list(next_board.pieces(chess.PAWN, us))
    our_minors = list(next_board.pieces(chess.KNIGHT, us)) + list(next_board.pieces(chess.BISHOP, us))
    our_rooks = list(next_board.pieces(chess.ROOK, us))
    our_queens = list(next_board.pieces(chess.QUEEN, us))
    total_non_pawn = len(our_minors) + len(our_rooks) + len(our_queens)

    king_sq = next_board.king(us)
    is_castled_like = bool(
        king_sq is not None
        and chess.square_rank(king_sq) == home_rank
        and chess.square_file(king_sq) in (2, 6)
    )

    if total_non_pawn == 0:
        army_participation_ratio = 1.0
    else:
        active_units = 0.0
        for msq in our_minors:
            if chess.square_rank(msq) != home_rank:
                active_units += 1.0
        for qsq in our_queens:
            if (
                chess.square_rank(qsq) != home_rank
                or bool(next_board.attacks_mask(qsq) & _BB_EXTENDED_CENTER)
                or is_castled_like
            ):
                active_units += 1.0
            else:
                active_units += 0.5
        rook_set = set(our_rooks)
        for rsq in our_rooks:
            rf = chess.square_file(rsq)
            rr = chess.square_rank(rsq)
            has_friendly_pawn_on_file = any(chess.square_file(psq) == rf for psq in our_pawns)
            connects_other_rook = any(
                other_r != rsq and bool(next_board.attacks_mask(rsq) & chess.BB_SQUARES[other_r])
                for other_r in rook_set
            )
            if (
                rsq not in corner_sqs
                or rr == seventh_rank
                or not has_friendly_pawn_on_file
                or connects_other_rook
            ):
                active_units += 1.0
            elif is_castled_like:
                active_units += 0.5
        army_participation_ratio = _clamp01(active_units / total_non_pawn)

    minors_before = list(board_before.pieces(chess.KNIGHT, us)) + list(board_before.pieces(chess.BISHOP, us))
    undeveloped_minors_before = sum(
        1 for sq in minors_before if chess.square_rank(sq) == home_rank
    )
    piece_type = board_before.piece_type_at(move.from_square) or chess.PAWN
    from_rank = chess.square_rank(move.from_square)
    to_rank = chess.square_rank(move.to_square)
    was_attacked_before = board_before.is_attacked_by(them, move.from_square)

    is_tactical_conversion = bool(
        next_board.is_checkmate()
        or move.promotion
        or (net_capture_gain >= 1.0 and not is_blunder)
    )
    is_tactical_or_forced = bool(
        is_tactical_conversion
        or board_before.is_check()
        or board_before.is_castling(move)
        or (next_board.is_check() and not is_blunder and net_capture_gain >= 0.0)
        or was_attacked_before
    )

    premature_solo_move = False
    if (
        board_before.fullmove_number <= 10
        and undeveloped_minors_before >= 2
        and not is_tactical_or_forced
        and net_capture_gain < 0.5
    ):
        second_rank = home_rank + (1 if us == chess.WHITE else -1)
        if piece_type == chess.QUEEN and to_rank not in (home_rank, second_rank):
            premature_solo_move = True
        elif piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK) and from_rank != home_rank:
            premature_solo_move = True

    if next_board.is_checkmate():
        self_sufficiency = 0.95
    elif is_tactical_conversion:
        # Local strike force wins material or promotes right now without needing slow army buildup
        self_sufficiency = _clamp01(
            0.45 * existence
            + 0.40 * min(1.0, 0.68 + 0.10 * net_capture_gain)
            + 0.15 * piece_coord_ratio
        )
    elif total_non_pawn <= 2:
        # Simplified endgame: remaining pieces + King + pawn flow are self-sufficient
        self_sufficiency = _clamp01(
            0.40 * existence
            + 0.35 * continuity
            + 0.25 * army_participation_ratio
            - (0.22 if is_blunder else 0.0)
        )
    else:
        # Quiet opening/middlegame: full-army mobilization + King shelter + mutual defense are required
        self_sufficiency = _clamp01(
            0.40 * existence
            + 0.35 * army_participation_ratio
            + 0.25 * piece_coord_ratio
            - (0.18 if premature_solo_move else 0.0)
            - (0.22 if is_blunder else 0.0)
        )

    return army_participation_ratio, premature_solo_move, is_tactical_conversion, self_sufficiency


def compute_candidate_move_fuzzy_coordinates(
    board: chess.Board,
    move: chess.Move,
    *,
    perspective: chess.Color | None = None,
    learned_leaves: Sequence[LearnedLeaf] = (),
    bitboard_sleeve: BitboardIntuitionSleeve | None = None,
) -> tuple[dict[str, float], bool, float, ChessBoardPosture, tuple[str, ...]]:
    """Compute the 40-node Fuzzy Tree coordinate dictionary (`Fact`, `Theory`, `Confidence`, `Intention`) for `move`."""
    us = board.turn if perspective is None else perspective
    them = not us
    piece_type = board.piece_type_at(move.from_square) or chess.PAWN
    moving_val = _PIECE_VALUES.get(piece_type, 1.0)
    placed_val = _PIECE_VALUES.get(move.promotion, 9.0) if move.promotion else moving_val
    is_cap = board.is_capture(move)
    captured_val = 0.0
    if is_cap:
        if board.is_en_passant(move):
            captured_val = 1.0
        else:
            cap_pt = board.piece_type_at(move.to_square)
            captured_val = _PIECE_VALUES.get(cap_pt or chess.PAWN, 1.0)

    next_board = board.copy()
    next_board.push(move)

    enemy_atts = list(next_board.attackers(them, move.to_square))
    our_defs = list(next_board.attackers(us, move.to_square))
    dest_attacked_by_them = bool(enemy_atts)
    dest_defended_by_us = bool(our_defs)
    is_blunder = False
    blunder_penalty = 0.0
    recapture_loss = 0.0

    if dest_attacked_by_them:
        min_att_val = min(
            (
                _PIECE_VALUES.get(next_board.piece_type_at(a_sq) or chess.PAWN, 1.0)
                for a_sq in enemy_atts
                if next_board.piece_type_at(a_sq) != chess.KING
            ),
            default=placed_val,
        )
        if not dest_defended_by_us:
            recapture_loss = placed_val
            if moving_val > captured_val + 0.5:
                is_blunder = True
                blunder_penalty = moving_val - captured_val
        else:
            if len(enemy_atts) > len(our_defs):
                recapture_loss = placed_val
                if moving_val - captured_val >= 0.8:
                    is_blunder = True
                    blunder_penalty = moving_val - captured_val
            elif placed_val > min_att_val:
                recapture_loss = placed_val - min_att_val
                if (moving_val - captured_val) - min_att_val >= 0.8:
                    is_blunder = True
                    blunder_penalty = (moving_val - captured_val) - min_att_val
            elif captured_val > 0.0 and placed_val >= captured_val - 0.2:
                recapture_loss = min(placed_val, captured_val)

    promo_gain = max(0.0, placed_val - 1.0 - recapture_loss) if move.promotion else 0.0
    net_capture_gain = max(0.0, captured_val + promo_gain - recapture_loss - blunder_penalty)

    for sq, pc in next_board.piece_map().items():
        if pc.color != us or pc.piece_type == chess.KING or sq == move.to_square:
            continue
        p_val = _PIECE_VALUES.get(pc.piece_type, 1.0)
        sq_enemy_atts = list(next_board.attackers(them, sq))
        if not sq_enemy_atts:
            continue
        sq_our_defs = list(next_board.attackers(us, sq))
        if not sq_our_defs:
            net_unprotected = p_val - net_capture_gain
            if p_val >= 3.0 and net_unprotected >= 0.8:
                is_blunder = True
                blunder_penalty = max(blunder_penalty, net_unprotected)
            elif p_val == 1.0 and net_unprotected >= 0.8:
                blunder_penalty = max(blunder_penalty, 0.55)
        else:
            sq_min_att_val = min(
                (
                    _PIECE_VALUES.get(next_board.piece_type_at(a_sq) or chess.PAWN, 1.0)
                    for a_sq in sq_enemy_atts
                    if next_board.piece_type_at(a_sq) != chess.KING
                ),
                default=99.0,
            )
            if p_val - sq_min_att_val >= 1.0:
                sq_loss = (p_val - sq_min_att_val) - net_capture_gain
                if sq_loss >= 0.8:
                    is_blunder = True
                    blunder_penalty = max(blunder_penalty, sq_loss)
            elif p_val >= 3.0 and len(sq_enemy_atts) > len(sq_our_defs):
                sq_loss = (0.75 * p_val) - net_capture_gain
                if sq_loss >= 0.8:
                    is_blunder = True
                    blunder_penalty = max(blunder_penalty, sq_loss)

    # Detect immediate 1-ply mate threats after this move
    for reply in next_board.legal_moves:
        if next_board.gives_check(reply):
            next_board.push(reply)
            is_mate_reply = next_board.is_checkmate()
            next_board.pop()
            if is_mate_reply:
                is_blunder = True
                blunder_penalty = max(blunder_penalty, 9.5)
                break

    post_posture = evaluate_chess_board_posture(next_board, perspective=us)

    coords: dict[str, float] = {k: 0.52 for k in FOUNDATIONAL_LEAVES}
    for leaf in learned_leaves:
        coords[leaf.leaf_id] = 0.0
    for leaf_id, val in post_posture.philosophical_coordinates.items():
        coords[leaf_id] = float(val)
    coords["cultural/efficiency/wastelessness"] = (
        0.25 if is_blunder else round(min(0.95, 0.72 + 0.06 * net_capture_gain), 4)
    )
    coords["cultural/freedom/self-sufficiency"] = float(
        post_posture.philosophical_coordinates.get("philosophical/being/existence", 0.55)
    )

    sleeve = bitboard_sleeve if bitboard_sleeve is not None else _DEFAULT_BITBOARD_SLEEVE
    sleeve_out = sleeve.evaluate_move(
        board,
        move,
        perspective=us,
        next_board=next_board,
        blunder_penalty=blunder_penalty,
        captured_val=captured_val,
        recapture_loss=recapture_loss,
        net_capture_gain=net_capture_gain,
        leaf_coords=coords,
    )
    existence = float(coords.get("philosophical/being/existence", 0.50))
    order = float(coords.get("philosophical/structure/order", 0.50))
    continuity = float(coords.get("philosophical/persistence/continuity", 0.50))
    discreteness = float(coords.get("philosophical/persistence/discreteness", 0.50))
    agency = float(coords.get("philosophical/being/agency", 0.50))
    chaos = float(coords.get("philosophical/structure/chaos", 0.50))

    # 1. `psychological/truth/fact`: Concrete bitboard reality (material, attack control, king safety, piece safety)
    fact_coord = _clamp01(
        0.55 * float(sleeve_out["value_prior"])
        + 0.45 * existence
        + 0.12 * min(8.5, net_capture_gain)
        - 0.22 * min(3.0, blunder_penalty)
    )
    # 2. `psychological/truth/theory`: Projected positional hypothesis on the bitboard
    theory_coord = _clamp01(
        0.50 * float(sleeve_out["policy_prior"])
        + 0.25 * order
        + 0.15 * continuity
        + 0.10 * agency
    )
    # 3. `psychological/contemplation/confidence`: Bitboard Intuition Sleeve certainty
    sleeve_conf = float(sleeve_out["sleeve_confidence"])

    move_shadows = detect_shadow_pathologies(
        post_posture.philosophical_coordinates,
        fact=fact_coord,
        arousal=max(0.45, chaos),
        resonance=order,
    )
    all_warnings = list(post_posture.board_shadow_warnings)
    seen_shadows = {w.shadow_pathology for w in all_warnings}
    for mw in move_shadows:
        if mw.shadow_pathology not in seen_shadows:
            all_warnings.append(mw)
            seen_shadows.add(mw.shadow_pathology)
    for w in all_warnings:
        if w.shadow_leaf_id:
            coords[w.shadow_leaf_id] = float(w.severity)
    shadow_names = tuple(w.shadow_pathology for w in all_warnings)

    # 4. `psychological/contemplation/clarity`: Signal-to-noise alignment between `fact` and `theory` & low chaos/shadow noise
    clarity_coord = _clamp01(
        0.45 * (1.0 - abs(fact_coord - theory_coord))
        + 0.35 * (order * (1.0 - 0.4 * chaos))
        + 0.20 * (1.0 if (not is_blunder and not shadow_names) else 0.25)
    )

    coords["psychological/truth/fact"] = fact_coord
    coords["psychological/truth/theory"] = theory_coord
    coords["psychological/contemplation/clarity"] = clarity_coord
    coords["psychological/contemplation/confidence"] = sleeve_conf
    coords["cultural/efficiency/power"] = _clamp01(0.50 * agency + 0.50 * discreteness)

    sleeve_feats = sleeve_out.get("features") or {}
    dev_prog = float(sleeve_feats.get("development_progress", 0.50))
    piece_coord_ratio = float(sleeve_feats.get("piece_coordination_ratio", 0.50))
    is_rep_move = next_board.is_repetition(2)
    army_participation_ratio, premature_solo_move, is_tactical_conversion, self_sufficiency = (
        _compute_army_mobilization_and_sufficiency(
            board,
            move,
            next_board,
            us=us,
            existence=existence,
            continuity=continuity,
            piece_coord_ratio=piece_coord_ratio,
            net_capture_gain=net_capture_gain,
            is_blunder=is_blunder,
        )
    )
    coords["cultural/freedom/self-sufficiency"] = self_sufficiency
    coords["cultural/freedom/needlessness"] = _clamp01(
        0.08
        + (0.45 if is_blunder else 0.0)
        + (0.20 if shadow_names else 0.0)
        + (0.22 if premature_solo_move else 0.0)
        + (0.20 if is_rep_move else 0.0)
        - (0.03 if is_tactical_conversion and not is_blunder else 0.0)
    )
    coords["experiential/reflection/salience"] = _clamp01(
        0.45 + 0.08 * net_capture_gain + 0.25 * float(sleeve_out["policy_prior"])
    )
    coords["experiential/phase/arousal"] = _clamp01(0.35 + 0.40 * agency + 0.25 * chaos)
    raw_valence = (
        0.55 * float(post_posture.board_valence)
        + 0.45 * float(sleeve_out["value_prior"])
        + 0.06 * min(8.5, max(-3.5, float(net_capture_gain)))
        - 0.18 * min(3.5, float(blunder_penalty))
    )
    valence_coord = _clamp01(min(0.95, max(0.05, raw_valence)))
    if is_blunder:
        valence_coord = min(0.28, valence_coord)
    coords["experiential/phase/valence"] = valence_coord

    coords["cultural/justice/honor"] = _clamp01(
        0.24
        + 0.28 * order
        + 0.18 * dev_prog
        + 0.14 * army_participation_ratio
        + 0.10 * continuity
        + 0.08 * float(sleeve_out["policy_prior"])
        - (0.18 if premature_solo_move else 0.0)
        - (0.35 if is_rep_move else 0.0)
        - (0.22 if is_blunder else 0.0)
    )
    trade_parity = _clamp01(
        0.48
        + 0.10 * min(5.0, max(-3.5, float(net_capture_gain)))
        + 0.04 * order
        - 0.22 * min(3.5, float(blunder_penalty))
    )
    fairness_coord = _clamp01(0.55 * trade_parity + 0.45 * float(post_posture.board_valence))
    coords["cultural/justice/fairness"] = min(0.30, fairness_coord) if is_blunder else fairness_coord

    is_transforming = bool(
        is_cap or move.promotion or board.is_castling(move) or next_board.is_check()
    )
    coords["experiential/reflection/novelty"] = (
        0.08
        if is_rep_move
        else _clamp01(
            0.35
            + (0.24 if is_transforming else 0.0)
            + 0.22 * chaos
            + 0.19 * float(sleeve_out["policy_prior"])
        )
    )
    sleeve_coherence = 1.0 - min(
        1.0, abs(float(sleeve_out["value_prior"]) - float(sleeve_out["policy_prior"]))
    )
    coords["experiential/synchronization/resonance"] = _clamp01(
        0.32 * order
        + 0.24 * army_participation_ratio
        + 0.22 * continuity
        + 0.22 * sleeve_coherence
        - (0.15 if premature_solo_move else 0.0)
        - (0.25 if is_blunder else 0.0)
        - (0.20 if is_rep_move else 0.0)
    )
    coords["experiential/synchronization/entrainment"] = _clamp01(
        0.45 * continuity
        + 0.35 * order
        + 0.20 * float(sleeve_out["policy_prior"])
        - (0.25 if is_blunder else 0.0)
        - (0.30 if is_rep_move else 0.0)
    )

    # 4. `psychological/will/intention` via `RIGHT_BRAIN_WILL_EDGES` (`rb-will-0..3`)
    intention_coord = _clamp01(
        (
            1.0 * coords["experiential/reflection/salience"]
            + 1.0 * coords["experiential/phase/arousal"]
            + 0.8 * coords["cultural/efficiency/power"]
            - 0.8 * coords["cultural/freedom/needlessness"]
        )
        / 2.8
    )
    coords["psychological/will/intention"] = intention_coord
    coords["psychological/will/execution"] = _clamp01(
        (
            1.0 * intention_coord
            + 1.0 * fact_coord
            + 0.7 * theory_coord
            + 0.5 * sleeve_conf
            - 0.8 * coords["cultural/freedom/needlessness"]
        )
        / 3.2
    )
    active_shadow_ids = {w.shadow_leaf_id for w in all_warnings if w.shadow_leaf_id}
    for leaf in learned_leaves:
        if leaf.leaf_id in active_shadow_ids or not leaf.leaf_name.startswith(("hazard_", "coached_")):
            continue
        dept_root = leaf.parent_branch.split("/")[0]
        if dept_root == "cultural":
            d_keys = FOUNDATIONAL_LEAVES[:6] + ("psychological/will/intention", "psychological/will/execution")
        elif dept_root == "philosophical":
            d_keys = FOUNDATIONAL_LEAVES[6:12] + ("psychological/truth/theory", "psychological/truth/fact")
        else:
            d_keys = (
                "experiential/reflection/novelty",
                "experiential/reflection/salience",
                "experiential/phase/valence",
                "experiential/phase/arousal",
                "experiential/synchronization/resonance",
                "experiential/synchronization/entrainment",
                "psychological/contemplation/clarity",
                "psychological/contemplation/confidence",
            )
        aw = leaf.anchor_weights if (leaf.anchor_weights and len(leaf.anchor_weights) == 8) else (0.50,) * 8
        aw_sum = max(1e-6, sum(float(x) for x in aw))
        proj = sum(float(aw[i]) * float(coords.get(d_keys[i], 0.50)) for i in range(8)) / aw_sum
        if leaf.leaf_name.startswith("hazard_"):
            strain = (
                0.50 * (1.0 - float(coords.get("cultural/efficiency/wastelessness", 0.72)))
                + 0.30 * float(coords.get("cultural/freedom/needlessness", 0.08))
                + 0.20 * float(coords.get("philosophical/structure/chaos", 0.25))
            )
            coords[leaf.leaf_id] = _clamp01(0.65 * strain + 0.35 * proj + (0.22 if is_blunder else -0.10))
        else:
            coords[leaf.leaf_id] = _clamp01(proj * (0.45 if is_blunder else 1.0))
    return coords, is_blunder, blunder_penalty, post_posture, shadow_names


def _material_balance_for_side(board: chess.Board, side: chess.Color) -> float:
    """Return concrete non-king material advantage (`side` minus `not side`)."""
    opp = not side
    our_m = sum(
        val * chess.popcount(board.pieces_mask(pt, side))
        for pt, val in _PIECE_VALUES.items()
        if val > 0.0
    )
    opp_m = sum(
        val * chess.popcount(board.pieces_mask(pt, opp))
        for pt, val in _PIECE_VALUES.items()
        if val > 0.0
    )
    return float(our_m - opp_m)


def _select_rollout_branch_moves(
    board: chess.Board,
    *,
    max_quiet: int = 6,
) -> list[chess.Move]:
    """Select candidate moves at a rollout ply (all evasions if in check, else tacticals + top central/developing moves)."""
    legal = list(board.legal_moves)
    if len(legal) <= max_quiet + 4 or board.is_check():
        return legal
    tactical: list[chess.Move] = []
    quiet: list[chess.Move] = []
    for mv in legal:
        if board.is_capture(mv) or mv.promotion or board.gives_check(mv):
            tactical.append(mv)
        else:
            quiet.append(mv)
    quiet.sort(
        key=lambda mv: (
            -int(bool(chess.BB_SQUARES[mv.to_square] & _BB_EXTENDED_CENTER)),
            -int((board.piece_type_at(mv.from_square) or chess.PAWN) in (chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.ROOK)),
        )
    )
    return tactical + quiet[:max_quiet]


def _score_rollout_ply_impulse(
    board: chess.Board,
    move: chess.Move,
    *,
    perspective: chess.Color,
    sleeve: BitboardIntuitionSleeve,
) -> tuple[float, chess.Board, dict[str, Any], float, bool]:
    """Evaluate a single ply inside the `Intention -> Theory -> Confidence` rollout using the Bitboard Sleeve."""
    us = perspective
    them = not us
    nb = board.copy()
    moving_pt = board.piece_type_at(move.from_square) or chess.PAWN
    moving_val = _PIECE_VALUES.get(moving_pt, 1.0)
    placed_val = _PIECE_VALUES.get(move.promotion, moving_val) if move.promotion else moving_val
    captured_val = 0.0
    if board.is_capture(move):
        if board.is_en_passant(move):
            captured_val = 1.0
        else:
            cap_pc = board.piece_at(move.to_square)
            if cap_pc is not None:
                captured_val = _PIECE_VALUES.get(cap_pc.piece_type, 1.0)

    nb.push(move)
    if nb.is_checkmate():
        return (
            100.0,
            nb,
            {"value_prior": 1.0, "policy_prior": 1.0, "sleeve_confidence": 0.95},
            0.0,
            False,
        )

    enemy_atts = list(nb.attackers(them, move.to_square))
    our_defs = list(nb.attackers(us, move.to_square))
    recapture_loss = 0.0
    blunder_penalty = 0.0
    if enemy_atts:
        min_att_val = min(
            (
                _PIECE_VALUES.get(nb.piece_type_at(a_sq) or chess.PAWN, 1.0)
                for a_sq in enemy_atts
                if nb.piece_type_at(a_sq) != chess.KING
            ),
            default=placed_val,
        )
        if not our_defs:
            recapture_loss = placed_val
            if moving_val > captured_val + 0.5:
                blunder_penalty = moving_val - captured_val
        elif len(enemy_atts) > len(our_defs):
            recapture_loss = placed_val
            if moving_val - captured_val >= 0.8:
                blunder_penalty = moving_val - captured_val
        elif placed_val > min_att_val:
            recapture_loss = placed_val - min_att_val
            if (moving_val - captured_val) - min_att_val >= 0.8:
                blunder_penalty = (moving_val - captured_val) - min_att_val
        elif captured_val > 0.0 and placed_val >= captured_val - 0.2:
            recapture_loss = min(placed_val, captured_val)

    promo_gain = max(0.0, placed_val - 1.0 - recapture_loss) if move.promotion else 0.0
    net_gain = max(0.0, captured_val + promo_gain - recapture_loss - blunder_penalty)

    allows_mate = False
    for rep in nb.legal_moves:
        if nb.gives_check(rep):
            nb.push(rep)
            is_m = nb.is_checkmate()
            nb.pop()
            if is_m:
                allows_mate = True
                blunder_penalty = max(blunder_penalty, 9.5)
                break

    pending_piece_loss = 0.0
    if not allows_mate:
        for cap_mv in nb.generate_legal_captures():
            vic_pt = nb.piece_type_at(cap_mv.to_square)
            if vic_pt is None or vic_pt == chess.KING:
                continue
            vic_val = _PIECE_VALUES.get(vic_pt, 1.0)
            if vic_val < 3.0:
                continue
            att_pt = nb.piece_type_at(cap_mv.from_square) or chess.PAWN
            att_val = _PIECE_VALUES.get(att_pt, 1.0)
            if not nb.is_attacked_by(us, cap_mv.to_square):
                pending_piece_loss = max(pending_piece_loss, vic_val)
            elif vic_val - att_val >= 1.0:
                pending_piece_loss = max(pending_piece_loss, vic_val - att_val)

    sleeve_out = sleeve.evaluate_move(
        board,
        move,
        perspective=us,
        next_board=nb,
        blunder_penalty=blunder_penalty,
        captured_val=captured_val,
        recapture_loss=recapture_loss,
        net_capture_gain=net_gain,
    )
    impulse_score = (
        float(sleeve_out["value_prior"])
        + 0.75 * float(sleeve_out["policy_prior"])
        + 0.35 * net_gain
        - 0.55 * blunder_penalty
        - 0.45 * pending_piece_loss
        + (0.22 if board.gives_check(move) and blunder_penalty < 0.8 else 0.0)
    )
    return impulse_score, nb, sleeve_out, pending_piece_loss, allows_mate


def _rollout_intention_into_theory_and_confidence(
    board: chess.Board,
    next_board: chess.Board,
    *,
    perspective: chess.Color,
    rule_tuple: tuple[FuzzyConceptRule, ...],
    sleeve: BitboardIntuitionSleeve,
    base_coords: dict[str, float],
) -> tuple[float, float, float, float]:
    """Project a candidate `intention` through the opponent's best replies (`Ply 2`) and our best response (`Ply 3`) to ground `theory`, `confidence`, and `clarity`."""
    us = perspective
    them = not us
    reply_candidates = _select_rollout_branch_moves(next_board, max_quiet=6)
    if not reply_candidates:
        return (
            float(base_coords.get("psychological/truth/theory", 0.55)),
            float(base_coords.get("psychological/contemplation/confidence", 0.60)),
            float(base_coords.get("psychological/contemplation/clarity", 0.60)),
            0.0,
        )

    mat_before = _material_balance_for_side(board, us)
    scored_replies: list[tuple[float, chess.Move, chess.Board]] = []
    for r2 in reply_candidates:
        r2_score, board_2, _, _, r2_allows_mate = _score_rollout_ply_impulse(
            next_board,
            r2,
            perspective=them,
            sleeve=sleeve,
        )
        if not r2_allows_mate or len(reply_candidates) == 1:
            scored_replies.append((r2_score, r2, board_2))

    if not scored_replies:
        return (
            float(base_coords.get("psychological/truth/theory", 0.55)),
            float(base_coords.get("psychological/contemplation/confidence", 0.60)),
            float(base_coords.get("psychological/contemplation/clarity", 0.60)),
            0.0,
        )

    scored_replies.sort(key=lambda item: -item[0])
    top_opponent_replies = list(scored_replies[:2])
    seen_r2 = {item[1] for item in top_opponent_replies}
    for item in scored_replies[2:]:
        if next_board.gives_check(item[1]) and item[1] not in seen_r2:
            top_opponent_replies.append(item)
            seen_r2.add(item[1])
            if len(top_opponent_replies) >= 4:
                break

    worst_branch_utility = 999.0
    rollout_theory = float(base_coords.get("psychological/truth/theory", 0.55))
    rollout_conf = float(base_coords.get("psychological/contemplation/confidence", 0.60))
    rollout_clarity = float(base_coords.get("psychological/contemplation/clarity", 0.60))
    rollout_loss = 0.0
    fact_1ply = float(base_coords.get("psychological/truth/fact", 0.55))

    for _r2_score, _r2, board_2 in top_opponent_replies:
        if board_2.is_checkmate():
            return 0.02, 0.05, 0.10, 9.5

        our_responses = _select_rollout_branch_moves(board_2, max_quiet=6)
        if not our_responses:
            mat_2 = _material_balance_for_side(board_2, us)
            loss_2 = max(0.0, mat_before - mat_2)
            branch_th = 0.35 if loss_2 < 0.8 else 0.12
            branch_cf = 0.45 if loss_2 < 0.8 else 0.18
            branch_cl = 0.45 if loss_2 < 0.8 else 0.15
            util = branch_th - 0.25 * loss_2
            if util < worst_branch_utility:
                worst_branch_utility = util
                rollout_theory = branch_th
                rollout_conf = branch_cf
                rollout_clarity = branch_cl
                rollout_loss = max(rollout_loss, loss_2)
            continue

        best_m3_utility = -999.0
        second_m3_utility = -999.0
        best_board_3 = board_2
        best_sleeve_3: dict[str, Any] = {"value_prior": 0.50, "policy_prior": 0.50, "sleeve_confidence": 0.55}
        best_eff_mat_3 = _material_balance_for_side(board_2, us)

        for m3 in our_responses:
            m3_score, board_3, sleeve_3, pending_loss_3, allows_mate_4 = _score_rollout_ply_impulse(
                board_2,
                m3,
                perspective=us,
                sleeve=sleeve,
            )
            if board_3.is_checkmate():
                eff_mat_3 = mat_before + 9.0
                m3_util = 100.0
            else:
                mat_3 = _material_balance_for_side(board_3, us)
                eff_mat_3 = mat_3 - (9.0 if allows_mate_4 else 0.85 * pending_loss_3)
                m3_util = eff_mat_3 + 0.80 * m3_score
            if m3_util > best_m3_utility:
                second_m3_utility = best_m3_utility
                best_m3_utility = m3_util
                best_board_3 = board_3
                best_sleeve_3 = sleeve_3
                best_eff_mat_3 = eff_mat_3
            elif m3_util > second_m3_utility:
                second_m3_utility = m3_util

        net_mat_delta = best_eff_mat_3 - mat_before
        branch_loss = max(0.0, -net_mat_delta) if net_mat_delta <= -0.8 else 0.0

        posture_3 = evaluate_chess_board_posture(best_board_3, perspective=us)
        phil_3 = posture_3.philosophical_coordinates
        order_3 = float(phil_3.get("philosophical/structure/order", 0.50))
        cont_3 = float(phil_3.get("philosophical/persistence/continuity", 0.50))
        agency_3 = float(phil_3.get("philosophical/being/agency", 0.50))
        exist_3 = float(phil_3.get("philosophical/being/existence", 0.50))
        chaos_3 = float(phil_3.get("philosophical/structure/chaos", 0.50))

        branch_theory = _clamp01(
            0.35 * float(best_sleeve_3["policy_prior"])
            + 0.25 * float(best_sleeve_3["value_prior"])
            + 0.20 * order_3
            + 0.12 * cont_3
            + 0.08 * agency_3
            + 0.06 * min(4.0, max(0.0, net_mat_delta))
            - 0.18 * min(4.0, branch_loss)
        )
        structural_clarity_3 = _clamp01(
            0.55 * (1.0 - abs(fact_1ply - branch_theory))
            + 0.45 * (order_3 * (1.0 - 0.4 * chaos_3))
            - 0.20 * min(3.0, branch_loss)
        )
        sleeve_conf_3 = float(best_sleeve_3["sleeve_confidence"])
        if rule_tuple:
            coords_3 = dict(base_coords)
            for k, v in phil_3.items():
                coords_3[k] = float(v)
            coords_3["psychological/truth/fact"] = _clamp01(
                0.55 * float(best_sleeve_3["value_prior"])
                + 0.45 * exist_3
                - 0.20 * min(3.0, branch_loss)
            )
            coords_3["psychological/truth/theory"] = branch_theory
            coords_3["psychological/contemplation/clarity"] = structural_clarity_3
            coords_3["psychological/contemplation/confidence"] = sleeve_conf_3
            acts_3 = _evaluate_rules_with_inhibition(rule_tuple, coords_3)
            rule_cl_3, rule_conf_3, matched_3, _conf_3, _rsn_3 = compute_contemplation(acts_3)
            branch_clarity = _clamp01(0.50 * float(rule_cl_3) + 0.50 * structural_clarity_3)
            if matched_3 and matched_3.startswith(("hazard_", "blunder_", "avoid_")):
                branch_conf = _clamp01(0.35 * sleeve_conf_3 - 0.25 * float(rule_conf_3) - 0.15 * min(4.0, branch_loss))
            else:
                branch_conf = _clamp01(
                    0.50 * float(rule_conf_3)
                    + 0.50 * sleeve_conf_3
                    - 0.15 * min(4.0, branch_loss)
                )
        else:
            branch_clarity = structural_clarity_3
            branch_conf = _clamp01(sleeve_conf_3 - 0.15 * min(4.0, branch_loss))

        branch_util = branch_theory + 0.50 * branch_conf + 0.20 * branch_clarity - 0.30 * branch_loss
        if branch_util < worst_branch_utility:
            worst_branch_utility = branch_util
            rollout_theory = branch_theory
            rollout_conf = branch_conf
            rollout_clarity = branch_clarity
            rollout_loss = max(rollout_loss, branch_loss)

    return rollout_theory, rollout_conf, rollout_clarity, rollout_loss


def _enrich_move_coords_with_theory_rollout(
    board_before: chess.Board,
    move: chess.Move,
    coords: dict[str, float],
    is_blunder: bool,
    blunder_penalty: float,
    *,
    perspective: chess.Color,
    sleeve: BitboardIntuitionSleeve,
    rule_tuple: tuple[FuzzyConceptRule, ...] = (),
) -> tuple[dict[str, float], bool, float]:
    """Enrich 1-ply candidate coordinates with the `Ply 2-3` `Theory`, `Confidence`, and `Clarity` rollout."""
    nb = board_before.copy()
    nb.push(move)
    if nb.is_checkmate():
        return coords, False, 0.0
    r_theory, r_conf, r_clarity, r_loss = _rollout_intention_into_theory_and_confidence(
        board_before,
        nb,
        perspective=perspective,
        rule_tuple=rule_tuple,
        sleeve=sleeve,
        base_coords=coords,
    )
    coords["psychological/truth/theory"] = _clamp01(
        0.35 * float(coords.get("psychological/truth/theory", 0.50)) + 0.65 * r_theory
    )
    coords["psychological/contemplation/confidence"] = _clamp01(
        0.40 * float(coords.get("psychological/contemplation/confidence", 0.60)) + 0.60 * r_conf
    )
    coords["psychological/contemplation/clarity"] = _clamp01(
        0.40 * float(coords.get("psychological/contemplation/clarity", 0.60)) + 0.60 * r_clarity
    )
    if r_loss >= 0.8:
        was_blunder = is_blunder
        is_blunder = True
        blunder_penalty = max(float(blunder_penalty), r_loss)
        coords["cultural/efficiency/wastelessness"] = 0.25
        coords["cultural/freedom/needlessness"] = _clamp01(
            max(float(coords.get("cultural/freedom/needlessness", 0.10)), 0.53)
        )
        coords["experiential/phase/valence"] = min(
            0.28, float(coords.get("experiential/phase/valence", 0.25))
        )
        coords["cultural/justice/fairness"] = min(
            0.30, float(coords.get("cultural/justice/fairness", 0.30))
        )
        if not was_blunder:
            coords["cultural/justice/honor"] = _clamp01(
                float(coords.get("cultural/justice/honor", 0.50)) - 0.22
            )
            coords["experiential/synchronization/resonance"] = _clamp01(
                float(coords.get("experiential/synchronization/resonance", 0.50)) - 0.25
            )
            coords["experiential/synchronization/entrainment"] = _clamp01(
                float(coords.get("experiential/synchronization/entrainment", 0.50)) - 0.25
            )
    return coords, is_blunder, blunder_penalty


def evaluate_candidate_moves_with_right_brain(
    board: chess.Board,
    rules: Sequence[FuzzyConceptRule] = (),
    *,
    learned_leaves: Sequence[LearnedLeaf] = (),
    limit: int = 6,
    bitboard_sleeve: BitboardIntuitionSleeve | None = None,
    active_vow_leaves: Sequence[str] = (),
    branch_archives: Any = None,
) -> list[CandidateMoveEvaluation]:
    """Rank legal moves via `Intention (Ply 1) -> Theory (Ply 2-3 Rollout) -> Confidence -> Execution` (`RIGHT_BRAIN_WILL_EDGES`)."""
    legal = list(board.legal_moves)
    if not legal:
        return []

    us = board.turn
    rule_tuple = tuple(rules)
    learned_tuple = tuple(learned_leaves)
    vow_leaf_tuple = tuple(k for k in active_vow_leaves if k in FOUNDATIONAL_LEAVES)
    sleeve = bitboard_sleeve if bitboard_sleeve is not None else _DEFAULT_BITBOARD_SLEEVE
    evaluations: list[CandidateMoveEvaluation] = []
    pending_candidates: list[dict[str, Any]] = []

    for move in legal:
        san = board.san(move)
        uci = move.uci()

        next_board = board.copy()
        next_board.push(move)

        if next_board.is_checkmate():
            evaluations.append(
                CandidateMoveEvaluation(
                    san=san,
                    uci=uci,
                    composite_score=100.0,
                    settled=True,
                    clarity=1.0,
                    confidence=1.0,
                    matched_concept="decisive_checkmate",
                    dominant_piece="Queen",
                    dominant_color="Red",
                    is_blunder=False,
                    is_repetition=False,
                    shadow_warnings=(),
                    execution_bid=100.0,
                    fact=1.0,
                    theory=1.0,
                    intention=1.0,
                )
            )
            continue

        is_rep = next_board.is_repetition(2)
        coords, is_blunder, blunder_penalty, post_posture, shadow_names = (
            compute_candidate_move_fuzzy_coordinates(
                board,
                move,
                perspective=us,
                learned_leaves=learned_tuple,
                bitboard_sleeve=sleeve,
            )
        )
        for r in rule_tuple:
            for req_leaf in r.target_leaves:
                coords.setdefault(req_leaf, 0.0)

        intention_rank = (
            1.0 * float(coords.get("psychological/will/intention", 0.50))
            + 1.0 * float(coords.get("psychological/truth/fact", 0.50))
            + 0.5 * float(coords.get("psychological/truth/theory", 0.50))
            + 0.35 * float(coords.get("experiential/phase/valence", 0.50))
            - 1.2 * blunder_penalty
            - (1.5 if is_rep else 0.0)
        )
        pending_candidates.append(
            {
                "move": move,
                "san": san,
                "uci": uci,
                "next_board": next_board,
                "is_rep": is_rep,
                "coords": coords,
                "is_blunder": is_blunder,
                "blunder_penalty": blunder_penalty,
                "post_posture": post_posture,
                "shadow_names": shadow_names,
                "intention_rank": intention_rank,
                "rolled_out": False,
            }
        )

    pending_candidates.sort(key=lambda item: (-float(item["intention_rank"]), str(item["san"])))

    def _apply_rollout_to_candidate(cand: dict[str, Any]) -> None:
        if cand["rolled_out"]:
            return
        cand["rolled_out"] = True
        r_theory, r_conf, r_clarity, r_loss = _rollout_intention_into_theory_and_confidence(
            board,
            cand["next_board"],
            perspective=us,
            rule_tuple=rule_tuple,
            sleeve=sleeve,
            base_coords=cand["coords"],
        )
        c_dict = cand["coords"]
        c_dict["psychological/truth/theory"] = _clamp01(
            0.35 * float(c_dict.get("psychological/truth/theory", 0.50)) + 0.65 * r_theory
        )
        c_dict["psychological/contemplation/confidence"] = _clamp01(
            0.40 * float(c_dict.get("psychological/contemplation/confidence", 0.60)) + 0.60 * r_conf
        )
        c_dict["psychological/contemplation/clarity"] = _clamp01(
            0.40 * float(c_dict.get("psychological/contemplation/clarity", 0.60)) + 0.60 * r_clarity
        )
        if r_loss >= 0.8:
            was_blunder = bool(cand["is_blunder"])
            cand["is_blunder"] = True
            cand["blunder_penalty"] = max(float(cand["blunder_penalty"]), r_loss)
            c_dict["cultural/efficiency/wastelessness"] = 0.25
            c_dict["cultural/freedom/needlessness"] = _clamp01(
                max(float(c_dict.get("cultural/freedom/needlessness", 0.10)), 0.53)
            )
            c_dict["experiential/phase/valence"] = min(
                0.28, float(c_dict.get("experiential/phase/valence", 0.25))
            )
            c_dict["cultural/justice/fairness"] = min(
                0.30, float(c_dict.get("cultural/justice/fairness", 0.30))
            )
            if not was_blunder:
                c_dict["cultural/justice/honor"] = _clamp01(
                    float(c_dict.get("cultural/justice/honor", 0.50)) - 0.22
                )
                c_dict["experiential/synchronization/resonance"] = _clamp01(
                    float(c_dict.get("experiential/synchronization/resonance", 0.50)) - 0.25
                )
                c_dict["experiential/synchronization/entrainment"] = _clamp01(
                    float(c_dict.get("experiential/synchronization/entrainment", 0.50)) - 0.25
                )

    rollout_k = max(10, int(limit))
    for cand in pending_candidates[:rollout_k]:
        _apply_rollout_to_candidate(cand)
    # Also roll out any candidate with low 1-ply `clarity` (< 0.55) so murky positions are always tested by `theory`
    for cand in pending_candidates[rollout_k:]:
        if float(cand["coords"].get("psychological/contemplation/clarity", 0.60)) < 0.55:
            _apply_rollout_to_candidate(cand)

    if pending_candidates and all(bool(c["is_blunder"]) for c in pending_candidates[:rollout_k]):
        for cand in pending_candidates[rollout_k:]:
            _apply_rollout_to_candidate(cand)

    for cand in pending_candidates:
        san = cand["san"]
        uci = cand["uci"]
        is_rep = cand["is_rep"]
        coords = cand["coords"]
        is_blunder = cand["is_blunder"]
        blunder_penalty = float(cand["blunder_penalty"])
        post_posture = cand["post_posture"]
        shadow_names = cand["shadow_names"]

        sleeve_conf = float(coords.get("psychological/contemplation/confidence", 0.60))
        sensor_clarity = float(coords.get("psychological/contemplation/clarity", 0.60))
        is_hazard_match = False
        if rule_tuple:
            activations = _evaluate_rules_with_inhibition(rule_tuple, coords)
            rule_clarity, rule_confidence, matched, is_confused, _reason = compute_contemplation(activations)
            settled = not is_confused
            clarity = _clamp01(0.75 * float(rule_clarity) + 0.25 * sensor_clarity)
            hazard_peak = max(
                (
                    float(activations.get(r.concept_name, 0.0))
                    for r in rule_tuple
                    if r.concept_name.startswith(("hazard_", "blunder_", "avoid_"))
                ),
                default=0.0,
            )
            coached_peak = max(
                (
                    float(activations.get(r.concept_name, 0.0))
                    for r in rule_tuple
                    if r.rule_id not in (
                        "rule_1",
                        "rule_2",
                        "rule_grounded_coordination",
                        "rule_sanctuary_equilibrium",
                    )
                    and not r.concept_name.startswith(
                        (
                            "hazard_",
                            "blunder_",
                            "avoid_",
                            "balanced_",
                            "mutual_",
                            "grounded_",
                            "sanctuary_",
                        )
                    )
                ),
                default=0.0,
            )
            is_hazard_match = bool(
                (matched and matched.startswith(("hazard_", "blunder_", "avoid_")))
                or (hazard_peak >= 0.50 and hazard_peak >= coached_peak)
            )
            learned_bonus = 0.65 * coached_peak - 0.85 * hazard_peak
            if matched and matched.startswith(("hazard_", "blunder_", "avoid_")):
                learned_bonus = min(learned_bonus, -0.65 * float(rule_confidence))
            elif matched and not matched.startswith(("balanced_", "mutual_")) and not is_hazard_match:
                learned_bonus = max(learned_bonus, 0.55 * float(rule_confidence))
            confidence = _clamp01(0.55 * float(rule_confidence) + 0.45 * sleeve_conf)
        else:
            clarity, confidence, matched, settled, learned_bonus = (
                sensor_clarity,
                sleeve_conf,
                None,
                True,
                0.0,
            )
            coached_peak, hazard_peak = 0.0, 0.0

        coords["psychological/contemplation/clarity"] = float(clarity)
        coords["psychological/contemplation/confidence"] = float(confidence)

        fact_val = float(coords.get("psychological/truth/fact", 0.50))
        theory_val = _clamp01(
            float(coords.get("psychological/truth/theory", 0.50))
            + 0.18 * coached_peak
            - 0.22 * hazard_peak
        )
        intention_val = float(coords.get("psychological/will/intention", 0.50))
        needlessness_val = float(coords.get("cultural/freedom/needlessness", 0.10))

        # Action Bid on `psychological/will/execution` (`RIGHT_BRAIN_WILL_EDGES`: rb-will-4..8)
        will_execution_drive = (
            1.0 * intention_val
            + 1.0 * fact_val
            + 0.7 * theory_val
            + 0.5 * float(confidence)
            - 0.8 * needlessness_val
        )
        if is_hazard_match:
            contemplation_drive = (
                -0.45 * float(clarity)
                - 0.55 * float(confidence)
                + learned_bonus
                - 0.25
                - 0.45 * len(shadow_names)
            )
        else:
            contemplation_drive = (
                0.65 * float(clarity)
                + 0.55 * float(confidence)
                + learned_bonus
                + (0.30 if settled else -0.25)
                - 0.45 * len(shadow_names)
            )
        leaf_posture_score = (
            0.55 * float(coords.get("cultural/efficiency/wastelessness", 0.50))
            + 0.55 * float(coords.get("philosophical/being/existence", 0.50))
            + 0.45 * float(coords.get("philosophical/structure/order", 0.50))
            + 0.40 * float(coords.get("philosophical/persistence/continuity", 0.50))
            + 0.35 * float(coords.get("philosophical/persistence/discreteness", 0.50))
            + 0.25 * float(coords.get("philosophical/being/agency", 0.50))
            - 0.25 * float(coords.get("philosophical/structure/chaos", 0.50))
        )
        vow_delta = 0.0
        if vow_leaf_tuple:
            for vk in vow_leaf_tuple:
                c_val = float(coords.get(vk, 0.50))
                if vk in ("philosophical/structure/chaos", "cultural/freedom/needlessness"):
                    if c_val > 0.50:
                        vow_delta -= 0.35 * (c_val - 0.45)
                    elif c_val <= 0.35 and not is_blunder:
                        vow_delta += 0.16 * (0.45 - c_val)
                else:
                    if c_val >= 0.55 and not is_blunder:
                        vow_delta += 0.25 * (c_val - 0.45)
                    elif c_val < 0.42:
                        vow_delta -= 0.40 * (0.48 - c_val)
        if branch_archives is not None and hasattr(branch_archives, "recall_resonant_clauses"):
            res = branch_archives.recall_resonant_clauses(coords, min_similarity=0.70)
            exec_match = res.get("executive_match")
            if exec_match is not None and not is_blunder:
                vow_delta += 0.18 * (float(exec_match["similarity"]) - 0.65)
        guided_by_vow = bool(vow_delta >= 0.05 and not is_blunder)

        execution_bid = (
            will_execution_drive
            + contemplation_drive
            + leaf_posture_score
            + vow_delta
            - 1.35 * blunder_penalty
            - (1.80 if is_rep else 0.0)
        )
        evaluations.append(
            CandidateMoveEvaluation(
                san=san,
                uci=uci,
                composite_score=execution_bid,
                settled=settled,
                clarity=float(clarity),
                confidence=float(confidence),
                matched_concept=matched,
                dominant_piece=post_posture.dominant_piece,
                dominant_color=post_posture.dominant_color,
                is_blunder=is_blunder,
                is_repetition=is_rep,
                shadow_warnings=shadow_names,
                execution_bid=execution_bid,
                fact=fact_val,
                theory=theory_val,
                intention=intention_val,
                vow_alignment_delta=vow_delta,
                guided_by_vow=guided_by_vow,
            )
        )

    evaluations.sort(key=lambda item: (-item.composite_score, item.san))
    return evaluations[: max(1, int(limit))]


def format_move_history_pgn_snippet(move_history_san: Sequence[str], *, max_plies: int = 8) -> str:
    """Format recent SAN moves into standard numbered PGN notation (e.g., `'1. e4 e5 2. Nf3'`)."""
    if not move_history_san:
        return "starting position"
    total = len(move_history_san)
    start_idx = max(0, total - max(2, int(max_plies)))
    if start_idx % 2 == 1:
        start_idx = max(0, start_idx - 1)
    parts: list[str] = []
    for idx in range(start_idx, total, 2):
        move_num = (idx // 2) + 1
        white_san = move_history_san[idx]
        if idx + 1 < total:
            parts.append(f"{move_num}. {white_san} {move_history_san[idx + 1]}")
        else:
            parts.append(f"{move_num}. {white_san}")
    return " ".join(parts)


def build_natural_chess_scenario_prompt(
    board: chess.Board,
    move_history_san: Sequence[str] = (),
    *,
    user_note: str | None = None,
) -> str:
    """Build a natural-English chess scenario prompt with ZERO float telemetry and real legal move candidates."""
    turn_label = "White" if board.turn == chess.WHITE else "Black"
    check_clause = " (King is in check)" if board.is_check() else ""
    history_str = format_move_history_pgn_snippet(move_history_san)
    legal_candidates = [c.san for c in evaluate_candidate_moves_with_right_brain(board, limit=6)]
    legal_clause = (
        f"Legal replies include {', '.join(legal_candidates)}."
        if legal_candidates
        else "The game is over (no legal moves)."
    )

    if move_history_san:
        base = (
            f"Chess table — Move {int(board.fullmove_number)} ({history_str}): "
            f"{turn_label} to move{check_clause}. {legal_clause}"
        )
    else:
        base = (
            f"Chess table — Move {int(board.fullmove_number)}: "
            f"{turn_label} to move{check_clause}. {legal_clause}"
        )

    if user_note and user_note.strip():
        conversational_board = base.removeprefix("Chess table — ")
        return f"{user_note.strip()} On the board right now, {conversational_board}"
    return base


def _collect_past_moves_san(
    board: chess.Board,
    move_history_san: Sequence[str] = (),
) -> set[str]:
    past: set[str] = {s.rstrip("+#") for s in move_history_san if s}
    if board.move_stack:
        try:
            replay = board.root()
            for mv in board.move_stack:
                past.add(replay.san(mv).rstrip("+#"))
                replay.push(mv)
        except Exception:
            pass
    return past


def _is_descriptive_square_or_history_token(
    tok: str,
    core_san: str,
    board: chess.Board,
    past_moves_san: set[str],
) -> bool:
    """Return True when `tok` describes an already-played move or an existing piece/square on `board`."""
    # Explicit check/mate or capture claims (e.g. `Bf5+`, `Re8#`, `Nxe4`) are active tactical claims
    if tok.endswith(("+", "#")) or "x" in core_san:
        return False

    if core_san in past_moves_san:
        return True

    # Check if `core_san` is a bare square coordinate (`e4`, `e5`) that is already occupied on `board`
    if re.fullmatch(r"[a-h][1-8]", core_san):
        sq = chess.parse_square(core_san)
        if board.piece_at(sq) is not None:
            return True

    # Check if `core_san` is a piece+square (`Bc5`, `Nf3`) where that exact piece already sits on `sq`
    if re.fullmatch(r"[KQRBN][a-h][1-8]", core_san):
        sq = chess.parse_square(core_san[1:3])
        piece = board.piece_at(sq)
        if piece is not None and piece.symbol().upper() == core_san[0]:
            return True

    return False


def inspect_chess_moves_in_reply(
    reply_text: str,
    board: chess.Board,
    move_history_san: Sequence[str] = (),
) -> dict[str, Any]:
    """Inspect all SAN move tokens mentioned in `reply_text` against `board.parse_san()` and `board.gives_check()`."""
    detected: list[str] = []
    legal_moves_mentioned: list[str] = []
    illegal_moves_mentioned: list[str] = []
    false_check_tokens: list[tuple[str, str]] = []
    past_moves_san = _collect_past_moves_san(board, move_history_san)

    for match in _SAN_TOKEN_REGEX.finditer(reply_text):
        tok = match.group(1)
        detected.append(tok)
        core_san = tok.rstrip("+#").replace("0", "O")
        try:
            move = board.parse_san(core_san)
            canonical = board.san(move)
            if tok not in legal_moves_mentioned:
                legal_moves_mentioned.append(canonical)
            if tok.endswith(("+", "#")) and tok != canonical:
                false_check_tokens.append((tok, canonical))
        except ValueError:
            if _is_descriptive_square_or_history_token(
                tok,
                core_san,
                board,
                past_moves_san,
            ):
                continue
            if tok not in illegal_moves_mentioned:
                illegal_moves_mentioned.append(tok)

    return {
        "detected_tokens": detected,
        "legal_moves_mentioned": legal_moves_mentioned,
        "illegal_moves_mentioned": illegal_moves_mentioned,
        "false_check_tokens": false_check_tokens,
    }


def verify_or_ground_chess_moves_in_reply(
    reply_text: str,
    board: chess.Board,
    move_history_san: Sequence[str] = (),
) -> str:
    """Auditor Bug #3 & Guardrail Fix: Verify every SAN move token in `reply_text` against `python-chess` (`board.parse_san`).

    - Never overwrites or discards `Qwen3-4B`'s actual generated words with a canned sentence
      (`AGENTS.md` Rule #4 & #5), and never splices replacement move tokens into the middle of
      a sentence.
    - Strips false `+`/`#` check suffixes on legal moves when `board.gives_check(move)` is False.
    - Appends an honest `[Board Fact-Check: ...]` note at the end when `python-chess` detects
      an illegal move or ungrounded mate/opening-sacrifice claim.
    """
    if not reply_text or not reply_text.strip():
        return reply_text

    report = inspect_chess_moves_in_reply(reply_text, board, move_history_san=move_history_san)
    illegal = report["illegal_moves_mentioned"]
    false_checks = report["false_check_tokens"]

    has_legal_mate = False
    for m in board.legal_moves:
        if board.gives_check(m):
            board.push(m)
            is_mate = board.is_checkmate()
            board.pop()
            if is_mate:
                has_legal_mate = True
                break

    lower_reply = reply_text.lower()
    claims_ungrounded_mate = (not has_legal_mate) and (
        "checkmate" in lower_reply or "mate" in lower_reply
    )
    claims_ungrounded_opening_sacrifice = (
        board.fullmove_number <= 4
        and not board.is_check()
        and "sacrific" in lower_reply
    )

    if (
        not illegal
        and not false_checks
        and not claims_ungrounded_mate
        and not claims_ungrounded_opening_sacrifice
    ):
        return reply_text

    top_legal = [c.san for c in evaluate_candidate_moves_with_right_brain(board, limit=6)]
    turn_label = "White" if board.turn == chess.WHITE else "Black"

    grounded = reply_text.strip()
    # First fix any legal moves that had a false `+` or `#` suffix attached
    for raw_tok, canonical_san in false_checks:
        grounded = re.sub(
            r"(?<![A-Za-z0-9_])" + re.escape(raw_tok) + r"(?=$|[\s,.;:!?)\]])",
            canonical_san,
            grounded,
        )

    if top_legal:
        legal_list_str = ", ".join(top_legal[:5])
        if illegal:
            illegal_str = ", ".join(illegal)
            return (
                f"{grounded} [Board Fact-Check: {illegal_str} is not legal here; "
                f"legal moves for {turn_label} include {legal_list_str}.]"
            )
        if claims_ungrounded_mate:
            return (
                f"{grounded} [Board Fact-Check: no legal checkmate exists on the board here; "
                f"legal moves for {turn_label} include {legal_list_str}.]"
            )
        if claims_ungrounded_opening_sacrifice:
            return (
                f"{grounded} [Board Fact-Check: quiet opening position; "
                f"legal moves for {turn_label} include {legal_list_str}.]"
            )

    return grounded


class BoardAnchoredBridgeSensors:
    """Auditor Guardrail #2: Anchors Pass-1 Philosophical `Fact` to `python-chess` while scoring Chat/Theory via MiniLM."""

    def __init__(
        self,
        base_sensors: ThreeBridgeSensors,
        board_posture: ChessBoardPosture,
        *,
        ground_cultural_on_board: bool = False,
    ) -> None:
        self.base_sensors = base_sensors
        self.board_posture = board_posture
        self.ground_cultural_on_board = bool(ground_cultural_on_board)
        self.call_count = 0

    def score_text(
        self,
        text: str,
        *,
        learned_leaves: tuple[LearnedLeaf, ...] = (),
        branch_gating: bool = True,
        branch_gate_threshold: float = 0.50,
    ) -> tuple[dict[str, float], dict[str, float], tuple[str, ...], tuple[str, ...]]:
        self.call_count += 1
        # If board context was appended for the Left-Brain LLM, score the clean user/clause text via MiniLM
        clean_for_minilm = (
            text.split(" On the board right now, ")[0]
            .split(" [Board context:")[0]
            .strip()
            or text
        )
        coords, exp_raw, queried, skipped = self.base_sensors.score_text(
            clean_for_minilm,
            learned_leaves=learned_leaves,
            branch_gating=branch_gating,
            branch_gate_threshold=branch_gate_threshold,
        )
        # First call in run_three_legged_turn is Pass 1 (External Fact):
        # Anchor the 6 Philosophical coordinates to the real python-chess board!
        if self.call_count == 1:
            anchored = dict(coords)
            exp_out = dict(exp_raw)
            for leaf_id, board_val in self.board_posture.philosophical_coordinates.items():
                anchored[leaf_id] = float(board_val)
            # When scoring an automatic board state prompt (no user story text), ground
            # the 6 Cultural coordinates and 6 Experiential coordinates in the board's
            # structural and +/- evaluation state instead of MiniLM's dry reading of the
            # "Chess table — Move N..." header.
            if self.ground_cultural_on_board or clean_for_minilm.startswith("Chess table —"):
                phil = self.board_posture.philosophical_coordinates
                order_f = float(phil.get("philosophical/structure/order", 0.50))
                cont_f = float(phil.get("philosophical/persistence/continuity", 0.50))
                exist_f = float(phil.get("philosophical/being/existence", 0.50))
                chaos_f = float(phil.get("philosophical/structure/chaos", 0.30))
                agency_f = float(phil.get("philosophical/being/agency", 0.50))
                board_val_f = _clamp01(float(self.board_posture.board_valence))

                anchored["cultural/efficiency/power"] = _clamp01(
                    0.55 * agency_f
                    + 0.45 * float(phil.get("philosophical/persistence/discreteness", 0.50))
                )
                anchored["cultural/efficiency/wastelessness"] = 0.68
                anchored["cultural/justice/honor"] = _clamp01(
                    0.18
                    + 0.28 * order_f
                    + 0.14 * cont_f
                    + 0.08 * exist_f
                    - (0.15 if self.board_posture.board_shadow_warnings else 0.0)
                )
                anchored["cultural/justice/fairness"] = _clamp01(
                    0.15
                    + 0.55 * board_val_f
                    + 0.15 * order_f
                    - (0.12 if self.board_posture.in_check else 0.0)
                )
                anchored["cultural/freedom/self-sufficiency"] = _clamp01(
                    0.50 * exist_f + 0.30 * order_f + 0.20 * cont_f
                )
                anchored["cultural/freedom/needlessness"] = 0.08

                board_arousal = _clamp01(0.35 + 0.40 * agency_f + 0.25 * chaos_f)
                board_salience = _clamp01(
                    max(
                        float(exp_raw.get("experiential/reflection/salience", 0.45)),
                        0.48
                        + 0.30 * abs(board_val_f - 0.50)
                        + (0.18 if self.board_posture.board_shadow_warnings else 0.0),
                    )
                )
                board_novelty = _clamp01(
                    0.35
                    + 0.35 * chaos_f
                    + 0.20 * agency_f
                    + (0.10 if self.board_posture.in_check else 0.0)
                )
                board_resonance = _clamp01(
                    0.45 * order_f
                    + 0.35 * cont_f
                    + 0.20 * exist_f
                    - (0.15 if self.board_posture.board_shadow_warnings else 0.0)
                )
                board_entrainment = _clamp01(
                    0.50 * cont_f
                    + 0.35 * order_f
                    + 0.15 * exist_f
                    - (0.15 if self.board_posture.in_check else 0.0)
                )
                for exp_k, exp_v in (
                    ("experiential/phase/valence", board_val_f),
                    ("experiential/phase/arousal", board_arousal),
                    ("experiential/reflection/salience", board_salience),
                    ("experiential/reflection/novelty", board_novelty),
                    ("experiential/synchronization/resonance", board_resonance),
                    ("experiential/synchronization/entrainment", board_entrainment),
                ):
                    exp_out[exp_k] = exp_v
                    anchored[exp_k] = exp_v
            return anchored, exp_out, queried, skipped

        # Subsequent calls (Clause 1 at comma and full-sentence Pass 2+ Theory) score
        # the proposed plan/chat text directly so any mismatch with the board triggers dissonance!
        return coords, exp_raw, queried, skipped


def merge_board_fact_with_chat_sensors(
    board_posture: ChessBoardPosture,
    sensors: ThreeBridgeSensors,
    *,
    ground_cultural_on_board: bool = False,
) -> BoardAnchoredBridgeSensors:
    """Wrap `ThreeBridgeSensors` so Pass-1 `Fact` is anchored to `board_posture` while `Theory` is scored from text."""
    return BoardAnchoredBridgeSensors(
        sensors,
        board_posture,
        ground_cultural_on_board=ground_cultural_on_board,
    )


def apply_coregulation_lead_to_chord(
    chord: SubliminalChordVector,
    target_mode: str = "tilt_reset",
    *,
    alpha: float = 0.25,
) -> SubliminalChordVector:
    """Auditor Guardrail #1: Step the 52-float comma chord toward a Sanctuary target without touching `vibe.coordinates` (`Fact`).

    - Clause 1 (before the comma) has already been generated from the user's real state (**Pacing**).
    - Right at the comma (`<|subliminal_gap|>`), this helper steps the spliced 52-float chord
      by `alpha` (`0.25` by default) toward `target_mode` so Clause 2 **Leads** toward equilibrium.
    """
    mode = target_mode.strip().lower()
    if mode not in SANCTUARY_MODES:
        raise ValueError(f"Unknown sanctuary target_mode {target_mode!r}; expected one of {SANCTUARY_MODES}")
    step = _clamp01(alpha)
    if step <= 0.0:
        return chord

    doer = list(chord.doer_vector)
    thinker = list(chord.thinker_vector)
    feeler = list(chord.feeler_vector)
    cadence = list(chord.cadence_vector)

    austere_chord_8 = pool_departmental_child_chord(
        {"experiential/phase/austere_sanctuary": 0.85},
        department_prefix="experiential/",
        learned_leaves=DEFAULT_COMPLEX_FEELING_LEAVES,
    )

    if mode == "tilt_reset":
        # Cool down high Red/Agency & Green/Chaos -> complementary Cyan/Existence & Magenta/Order
        thinker[0] = _lerp01(thinker[0], 0.35, step)  # Agency (Red) cools down
        thinker[1] = _lerp01(thinker[1], 0.82, step)  # Existence (Cyan / Water) rises
        thinker[2] = _lerp01(thinker[2], 0.78, step)  # Order (Magenta / Wind) steadies
        thinker[3] = _lerp01(thinker[3], 0.20, step)  # Chaos (Green / Ice) settles
        # Feeler: raise Valence, Resonance, & Entrainment; lower Arousal
        feeler[2] = _lerp01(feeler[2], 0.72, step)  # Valence
        feeler[3] = _lerp01(feeler[3], 0.24, step)  # Arousal
        feeler[4] = _lerp01(feeler[4], 0.78, step)  # Resonance
        feeler[5] = _lerp01(feeler[5], 0.82, step)  # Entrainment (Leaf #18)
        for i in range(8):
            feeler[8 + i] = _lerp01(feeler[8 + i], austere_chord_8[i], step)
        cadence[0] = _lerp01(cadence[0], 0.80, step)  # Branch harmony
        cadence[1] = _lerp01(cadence[1], 0.15, step)  # Ease dissonance pressure
        cadence[2] = _lerp01(cadence[2], 0.0, step)   # Calm cadence rupture

    elif mode == "deep_focus":
        # Balance Magenta/Order (Bishop) + Yellow/Discreteness (Rook) + Wastelessness + Clarity
        doer[1] = _lerp01(doer[1], 0.84, step)        # Wastelessness
        doer[6] = _lerp01(doer[6], 0.76, step)        # Intention
        thinker[2] = _lerp01(thinker[2], 0.85, step)  # Order (Magenta / Wind)
        thinker[3] = _lerp01(thinker[3], 0.18, step)  # Chaos lowered
        thinker[4] = _lerp01(thinker[4], 0.76, step)  # Continuity (Blue)
        thinker[5] = _lerp01(thinker[5], 0.80, step)  # Discreteness (Yellow / Rook)
        feeler[3] = _lerp01(feeler[3], 0.36, step)    # Calm alert arousal
        feeler[5] = _lerp01(feeler[5], 0.85, step)    # Entrainment
        feeler[6] = _lerp01(feeler[6], 0.85, step)    # Clarity
        feeler[7] = _lerp01(feeler[7], 0.82, step)    # Confidence
        cadence[0] = _lerp01(cadence[0], 0.85, step)
        cadence[2] = _lerp01(cadence[2], 0.0, step)

    elif mode == "night_wind_down":
        # Deep rest induction: Needlessness + Cyan/Water + Blue/Continuity + Austere Sanctuary + Entrainment
        doer[0] = _lerp01(doer[0], 0.18, step)        # Power softens
        doer[5] = _lerp01(doer[5], 0.88, step)        # Needlessness rises
        doer[6] = _lerp01(doer[6], 0.15, step)        # Intention rests
        doer[7] = _lerp01(doer[7], 0.15, step)        # Execution rests
        thinker[0] = _lerp01(thinker[0], 0.22, step)  # Agency rests
        thinker[1] = _lerp01(thinker[1], 0.86, step)  # Existence (Cyan / Water)
        thinker[4] = _lerp01(thinker[4], 0.80, step)  # Continuity
        feeler[2] = _lerp01(feeler[2], 0.78, step)    # Warm Valence
        feeler[3] = _lerp01(feeler[3], 0.14, step)    # Low Arousal
        feeler[4] = _lerp01(feeler[4], 0.86, step)    # Resonance
        feeler[5] = _lerp01(feeler[5], 0.90, step)    # Deep Entrainment
        for i in range(8):
            feeler[8 + i] = _lerp01(feeler[8 + i], austere_chord_8[i], step)
        cadence[0] = _lerp01(cadence[0], 0.88, step)
        cadence[1] = _lerp01(cadence[1], 0.08, step)
        cadence[2] = _lerp01(cadence[2], 0.0, step)

    return SubliminalChordVector(
        doer_vector=tuple(doer),
        thinker_vector=tuple(thinker),
        feeler_vector=tuple(feeler),
        cadence_vector=tuple(cadence),
    )


def compute_fide_elo_update(
    player_elo: float,
    opponent_elo: float,
    score: float,
    *,
    k_factor: float = 24.0,
) -> tuple[float, float]:
    """Compute an un-rigged FIDE Elo rating update (`score`: `1.0` win, `0.5` draw, `0.0` loss).

    Returns `(new_player_elo, delta)`.
    """
    actual = min(1.0, max(0.0, float(score)))
    expected = 1.0 / (1.0 + 10.0 ** ((float(opponent_elo) - float(player_elo)) / 400.0))
    delta = round(float(k_factor) * (actual - expected), 2)
    new_elo = round(max(400.0, float(player_elo) + delta), 2)
    return new_elo, delta


class ObjectiveCalibratedSparringEngine:
    """Strength-toggled chess sparring opponent & post-move referee (`Stockfish UCI` + calibrated fallback).

    Anti-Cheat Air-Gap Invariant:
    - This engine is ONLY used to:
      1. Play the **Opponent's moves** on the opposite color from `LMFT` at `target_elo`.
      2. Measure post-move centipawn drops *after* `LMFT` has already committed its own move,
         so objective tactical blunders are queued for REM Sleep.
    - `LMFT` NEVER queries this engine to select `LMFT`'s own moves.
    """

    def __init__(
        self,
        *,
        stockfish_path: str | Any | None = None,
        use_stockfish_uci: bool = True,
    ) -> None:
        from pathlib import Path
        import chess.engine
        from .workspace_paths import resolve_training_root

        self.use_stockfish_uci = bool(use_stockfish_uci)
        self._chess_engine_mod = chess.engine
        if stockfish_path is not None:
            self.stockfish_path = Path(stockfish_path)
        else:
            self.stockfish_path = (
                resolve_training_root() / "models" / "chess_engine" / "stockfish.exe"
            )
        self.opponent_moves_played: int = 0
        self.post_move_audits_run: int = 0

    @property
    def engine_label(self) -> str:
        if self.use_stockfish_uci and self.stockfish_path.exists():
            return "Stockfish 19 UCI (Strength-Calibrated)"
        return "Calibrated Tactical Engine (Built-In)"

    def _popen_stockfish(self):
        """Spawn Stockfish UCI silently with `CREATE_NO_WINDOW` on Windows so no console window flashes."""
        popen_kwargs: dict[str, Any] = {}
        if os.name == "nt":
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        try:
            return self._chess_engine_mod.SimpleEngine.popen_uci(
                str(self.stockfish_path), **popen_kwargs
            )
        except TypeError:
            return self._chess_engine_mod.SimpleEngine.popen_uci(str(self.stockfish_path))

    def evaluate_position_centipawns(
        self,
        board: chess.Board,
        *,
        perspective: chess.Color,
    ) -> float:
        """Evaluate `board` in centipawns from `perspective`'s point of view (post-move referee only)."""
        self.post_move_audits_run += 1
        if self.use_stockfish_uci and self.stockfish_path.exists():
            try:
                with self._popen_stockfish() as eng:
                    info = eng.analyse(board, self._chess_engine_mod.Limit(depth=6, time=0.04))
                    pov_score = info["score"].pov(perspective)
                    cp = pov_score.score(mate_score=9999)
                    if cp is not None:
                        return float(cp)
            except Exception:
                pass
        return _static_centipawns_for_color(board, perspective)

    def select_sparring_opponent_move(
        self,
        board: chess.Board,
        *,
        target_elo: float = 1000.0,
    ) -> dict[str, Any]:
        """Select a move for the **Sparring Opponent** calibrated to `target_elo`."""
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError("No legal moves available for sparring opponent.")

        clamped_elo = min(2800.0, max(600.0, float(target_elo)))
        self.opponent_moves_played += 1

        if self.use_stockfish_uci and self.stockfish_path.exists():
            try:
                with self._popen_stockfish() as eng:
                    if clamped_elo >= 1320.0:
                        eng.configure(
                            {
                                "UCI_LimitStrength": True,
                                "UCI_Elo": int(round(clamped_elo)),
                            }
                        )
                        depth_lim = max(2, min(10, int((clamped_elo - 900.0) / 160.0)))
                        res = eng.play(board, self._chess_engine_mod.Limit(depth=depth_lim, time=0.06))
                        chosen_move = res.move
                    else:
                        # Stockfish UCI_Elo minimum is 1320; calibrate <1320 Elo via shallow MultiPV candidate pool
                        multipv_k = 2 if clamped_elo >= 1150.0 else (4 if clamped_elo >= 950.0 else 6)
                        infos = eng.analyse(
                            board,
                            self._chess_engine_mod.Limit(depth=1, time=0.02),
                            multipv=min(multipv_k, len(legal)),
                        )
                        if isinstance(infos, dict):
                            infos = [infos]
                        pv_moves = [
                            inf["pv"][0]
                            for inf in infos
                            if inf.get("pv") and inf["pv"][0] in board.legal_moves
                        ]
                        if pv_moves:
                            pick_idx = (len(board.move_stack) // 2 + board.fullmove_number) % len(pv_moves)
                            chosen_move = pv_moves[pick_idx]
                        else:
                            chosen_move = None
                    if chosen_move is not None and chosen_move in board.legal_moves:
                        san = board.san(chosen_move)
                        return {
                            "move_san": san,
                            "move_uci": chosen_move.uci(),
                            "target_elo": round(clamped_elo, 1),
                            "engine_backend": "stockfish_uci",
                        }
            except Exception:
                pass

        # Built-in strength-calibrated fallback when Stockfish subprocess is unavailable
        us = board.turn
        scored_moves: list[tuple[float, chess.Move, str]] = []
        for mv in legal:
            san = board.san(mv)
            nb = board.copy()
            nb.push(mv)
            cp = _static_centipawns_for_color(nb, us)
            if nb.is_repetition(2):
                cp -= 120.0
            scored_moves.append((cp, mv, san))
        scored_moves.sort(key=lambda item: (-item[0], item[2]))

        # Lower target_elo widens candidate pool slightly so play strength scales smoothly
        pool_width = 1 if clamped_elo >= 1400.0 else (2 if clamped_elo >= 1000.0 else min(4, len(scored_moves)))
        pick_idx = (len(board.move_stack) // 2) % pool_width
        chosen_cp, chosen_mv, chosen_san = scored_moves[pick_idx]
        return {
            "move_san": chosen_san,
            "move_uci": chosen_mv.uci(),
            "target_elo": round(clamped_elo, 1),
            "engine_backend": "calibrated_builtin",
            "estimated_cp": chosen_cp,
        }

    def audit_played_move_for_game_review(
        self,
        board_before: chess.Board,
        played_move_san: str | chess.Move,
        *,
        perspective: chess.Color | None = None,
        learned_leaves: Sequence[LearnedLeaf] = (),
        bitboard_sleeve: BitboardIntuitionSleeve | None = None,
    ) -> dict[str, Any]:
        """Post-move Game Review referee (`??` Blunder, `?` Mistake, `!!` Brilliancy, `!` Great Move).

        Strict Anti-Cheat Air-Gap:
        - Called ONLY AFTER `LMFT` has already committed `played_move_san` on `board_before`.
        - Never called during pre-move selection; never increments `self.opponent_moves_played`.
        - Computes both `played_coordinates` and `contrastive_reference_coordinates` (`best_move_coordinates`
          for blunders/mistakes, or `inferior_alternative_coordinates` for brilliancies/great moves)
          so REM Sleep and the 9B Teacher can normalize fuzzy rule `target_leaves` and `[low, high]`
          thresholds around the leaves that separated the two moves.
        """
        self.post_move_audits_run += 1
        us = board_before.turn if perspective is None else perspective
        legal = list(board_before.legal_moves)
        if not legal:
            raise ValueError("No legal moves on board_before for post-move review.")

        if isinstance(played_move_san, chess.Move):
            played_move = played_move_san if played_move_san in legal else legal[0]
            played_move_san = board_before.san(played_move)
        else:
            raw_mv = str(played_move_san).strip()
            try:
                played_move = board_before.parse_san(raw_mv)
                played_move_san = board_before.san(played_move)
            except Exception:
                try:
                    cand_mv = chess.Move.from_uci(raw_mv)
                    played_move = cand_mv if cand_mv in legal else legal[0]
                    played_move_san = board_before.san(played_move)
                except Exception:
                    played_move = legal[0]
                    played_move_san = board_before.san(played_move)

        learned_tuple = tuple(learned_leaves)
        sleeve = bitboard_sleeve if bitboard_sleeve is not None else _DEFAULT_BITBOARD_SLEEVE
        played_coords, played_is_blunder, played_b_pen, _p_post, _p_shad = (
            compute_candidate_move_fuzzy_coordinates(
                board_before,
                played_move,
                perspective=us,
                learned_leaves=learned_tuple,
                bitboard_sleeve=sleeve,
            )
        )
        played_coords, played_is_blunder, played_b_pen = _enrich_move_coords_with_theory_rollout(
            board_before,
            played_move,
            played_coords,
            played_is_blunder,
            played_b_pen,
            perspective=us,
            sleeve=sleeve,
        )

        cp_before = _static_centipawns_for_color(board_before, us)
        board_after_played = board_before.copy()
        board_after_played.push(played_move)
        cp_after = _static_centipawns_for_color(board_after_played, us) - (100.0 * played_b_pen)

        scored_alternatives: list[tuple[float, chess.Move, str, bool]] = []
        for mv in legal:
            san = board_before.san(mv)
            _c, is_b, b_pen, _post, _sh = compute_candidate_move_fuzzy_coordinates(
                board_before,
                mv,
                perspective=us,
                learned_leaves=learned_tuple,
                bitboard_sleeve=sleeve,
            )
            nb = board_before.copy()
            nb.push(mv)
            est_cp = _static_centipawns_for_color(nb, us) - (100.0 * b_pen)
            if nb.is_repetition(2):
                est_cp -= 120.0
            scored_alternatives.append((est_cp, mv, san, is_b))
        scored_alternatives.sort(key=lambda item: (-item[0], item[2]))

        best_cp, best_move, best_san, best_is_blunder = scored_alternatives[0]
        if self.use_stockfish_uci and self.stockfish_path.exists():
            try:
                with self._popen_stockfish() as eng:
                    info_before = eng.analyse(board_before, self._chess_engine_mod.Limit(depth=6, time=0.04))
                    pv_moves = info_before.get("pv") or []
                    if pv_moves and pv_moves[0] in board_before.legal_moves:
                        best_move = pv_moves[0]
                        best_san = board_before.san(best_move)
                    cp_b = info_before["score"].pov(us).score(mate_score=9999)
                    if cp_b is not None:
                        cp_before = float(cp_b)
                        best_cp = float(cp_b)
                    info_after = eng.analyse(board_after_played, self._chess_engine_mod.Limit(depth=6, time=0.04))
                    cp_a = info_after["score"].pov(us).score(mate_score=9999)
                    if cp_a is not None:
                        cp_after = float(cp_a)
            except Exception:
                pass

        best_coords, best_is_blunder, _best_pen, _b_post, _b_shad = (
            compute_candidate_move_fuzzy_coordinates(
                board_before,
                best_move,
                perspective=us,
                learned_leaves=learned_tuple,
                bitboard_sleeve=sleeve,
            )
        )
        best_coords, best_is_blunder, _best_pen = _enrich_move_coords_with_theory_rollout(
            board_before,
            best_move,
            best_coords,
            best_is_blunder,
            _best_pen,
            perspective=us,
            sleeve=sleeve,
        )

        # Find the second-best move (for `!`/`!!` edge detection) AND an inferior move (for REM Sleep coordinate contrast)
        static_played_cp = _static_centipawns_for_color(board_after_played, us) - (100.0 * played_b_pen)
        second_best_cp = static_played_cp
        for est_cp, mv, _san, _is_b in scored_alternatives:
            if mv != played_move:
                second_best_cp = est_cp
                break

        alt_move = scored_alternatives[-1][1] if len(scored_alternatives) > 1 else best_move
        for _est_cp, mv, _san, _is_b in reversed(scored_alternatives):
            if mv != played_move:
                alt_move = mv
                break
        alt_coords, _alt_b, _alt_pen, _a_post, _a_shad = compute_candidate_move_fuzzy_coordinates(
            board_before,
            alt_move,
            perspective=us,
            learned_leaves=learned_tuple,
            bitboard_sleeve=sleeve,
        )
        alt_coords, _alt_b, _alt_pen = _enrich_move_coords_with_theory_rollout(
            board_before,
            alt_move,
            alt_coords,
            _alt_b,
            _alt_pen,
            perspective=us,
            sleeve=sleeve,
        )

        cp_drop = 0.0 if played_move == best_move else round(max(0.0, best_cp - cp_after), 1)
        cp_gain = round(cp_after - cp_before, 1)
        cp_edge_over_alt = round(static_played_cp - second_best_cp, 1)
        chaos_val = float(played_coords.get("philosophical/structure/chaos", 0.0))

        if played_move != best_move and (cp_drop >= 150.0 or (played_is_blunder and not best_is_blunder)):
            classification = "BLUNDER_??"
            glyph = "??"
            is_hazard = True
            should_queue = True
            ref_coords = best_coords
        elif played_move != best_move and cp_drop >= 90.0:
            classification = "MISTAKE_?"
            glyph = "?"
            is_hazard = True
            should_queue = True
            ref_coords = best_coords
        elif not played_is_blunder and (
            cp_gain >= 110.0
            or (played_move == best_move and cp_edge_over_alt >= 140.0 and chaos_val >= 0.35)
        ):
            classification = "BRILLIANCY_!!"
            glyph = "!!"
            is_hazard = False
            should_queue = True
            ref_coords = alt_coords
        elif not played_is_blunder and played_move == best_move and (
            cp_gain >= 55.0 or cp_edge_over_alt >= 95.0
        ):
            classification = "GREAT_MOVE_!"
            glyph = "!"
            is_hazard = False
            should_queue = True
            ref_coords = alt_coords
        else:
            classification = "SOLID"
            glyph = ""
            is_hazard = False
            should_queue = False
            ref_coords = best_coords

        if is_hazard:
            played_coords["cultural/efficiency/wastelessness"] = 0.25
            best_coords["cultural/efficiency/wastelessness"] = max(
                0.72,
                float(best_coords.get("cultural/efficiency/wastelessness", 0.72)),
            )
            ref_coords = best_coords

        played_bitboard_features = extract_bitboard_features(
            board_before,
            played_move,
            perspective=us,
            blunder_penalty=played_b_pen,
        )
        reference_bitboard_features = extract_bitboard_features(
            board_before,
            best_move if is_hazard else alt_move,
            perspective=us,
            blunder_penalty=_best_pen if is_hazard else _alt_pen,
        )
        bitboard_feature_delta: dict[str, float] = {}
        for fk, fv in played_bitboard_features.items():
            f_diff = round(float(fv) - float(reference_bitboard_features.get(fk, fv)), 4)
            if abs(f_diff) >= 0.02:
                bitboard_feature_delta[fk] = f_diff

        coord_delta: dict[str, float] = {}
        for k, v in played_coords.items():
            diff = round(float(v) - float(ref_coords.get(k, v)), 4)
            if abs(diff) >= 0.03:
                coord_delta[k] = diff

        top_deltas = sorted(coord_delta.items(), key=lambda kv: (-abs(kv[1]), kv[0]))[:3]
        delta_str = ", ".join(
            f"{k.split('/')[-1]} {('+' if d >= 0 else '')}{d:.2f}" for k, d in top_deltas
        ) or "minimal posture delta"

        if is_hazard:
            summary = (
                f"Game Review [{glyph} {classification}]: Played {played_move_san} "
                f"(-{cp_drop / 100.0:.2f} pawns) instead of best move {best_san} ({delta_str})."
            )
        elif glyph:
            summary = (
                f"Game Review [{glyph} {classification}]: Found {played_move_san} "
                f"(+{max(cp_gain, cp_edge_over_alt) / 100.0:.2f} pawns vs. alternatives; {delta_str})."
            )
        else:
            summary = f"Game Review [SOLID]: Played {played_move_san} (best: {best_san})."

        return {
            "classification": classification,
            "glyph": glyph,
            "is_hazard": is_hazard,
            "should_queue_for_sleep": should_queue,
            "played_san": played_move_san,
            "best_san": best_san,
            "cp_before": round(cp_before, 1),
            "cp_after": round(cp_after, 1),
            "cp_best": round(best_cp, 1),
            "cp_drop": cp_drop,
            "cp_gain": cp_gain,
            "played_coordinates": played_coords,
            "best_move_coordinates": best_coords,
            "contrastive_reference_coordinates": ref_coords,
            "coordinate_delta": coord_delta,
            "played_bitboard_features": played_bitboard_features,
            "reference_bitboard_features": reference_bitboard_features,
            "bitboard_feature_delta": bitboard_feature_delta,
            "annotation_summary": summary,
        }


def compute_move_accuracy_percent(cp_drop: float) -> float:
    """Convert centipawn loss (`cp_drop >= 0`) into a `[0.0, 100.0]` move accuracy percentage."""
    loss = max(0.0, float(cp_drop))
    raw_acc = 103.1668 * math.exp(-0.04354 * loss) - 3.1669
    return round(min(100.0, max(0.0, raw_acc)), 1)


def compute_estimated_game_elo(
    mean_accuracy_pct: float,
    *,
    reference_elo: float = 1000.0,
) -> float:
    """Estimate single-game performance Elo from running move accuracy (`0..100%`) relative to `reference_elo`."""
    acc = min(100.0, max(0.0, float(mean_accuracy_pct)))
    est = float(reference_elo) + (acc - 75.0) * 14.0
    return round(min(2600.0, max(400.0, est)), 1)


def build_human_move_coaching_report(
    review_audit: Mapping[str, Any],
    *,
    rules: Sequence[FuzzyConceptRule] = (),
    player_elo: float = 1000.0,
    reference_elo: float = 1000.0,
    prior_move_accuracies: Sequence[float] = (),
) -> dict[str, Any]:
    """Build a custom Fuzzy-Tree coaching card and live rating/accuracy report for a human player's move."""
    cp_drop = float(review_audit.get("cp_drop", 0.0))
    move_acc = compute_move_accuracy_percent(cp_drop)
    all_accs = [*prior_move_accuracies, move_acc]
    game_acc = round(sum(all_accs) / max(1, len(all_accs)), 1)
    est_game_elo = compute_estimated_game_elo(game_acc, reference_elo=reference_elo)

    played_san = str(review_audit.get("played_san", "?"))
    best_san = str(review_audit.get("best_san", played_san))
    classification = str(review_audit.get("classification", "SOLID"))
    glyph = str(review_audit.get("glyph", ""))
    is_hazard = bool(review_audit.get("is_hazard", False))
    played_coords: Mapping[str, float] = review_audit.get("played_coordinates", {})
    best_coords: Mapping[str, float] = review_audit.get("best_move_coordinates", {})
    coord_delta: Mapping[str, float] = review_audit.get("coordinate_delta", {})

    foundational_ids = {
        "rule_1",
        "rule_2",
        "rule_grounded_coordination",
        "rule_sanctuary_equilibrium",
    }
    foundational_prefixes = (
        "balanced_",
        "mutual_",
        "grounded_",
        "sanctuary_",
    )
    matched_wisdom_rule: str | None = None
    matched_rule_score: float = 0.0
    for r in rules:
        if r.rule_id in foundational_ids or r.concept_name.startswith(foundational_prefixes):
            continue
        is_rule_haz = r.concept_name.startswith(("hazard_", "blunder_", "avoid_"))
        if is_hazard and is_rule_haz:
            sc = float(r.evaluate(played_coords))
            if sc > matched_rule_score and sc >= 0.35:
                matched_rule_score = sc
                matched_wisdom_rule = r.concept_name
        elif not is_hazard and not is_rule_haz:
            sc = float(r.evaluate(played_coords))
            if sc > matched_rule_score and sc >= 0.35:
                matched_rule_score = sc
                matched_wisdom_rule = r.concept_name

    if is_hazard and matched_wisdom_rule is None:
        for r in rules:
            if r.rule_id in foundational_ids or r.concept_name.startswith(foundational_prefixes):
                continue
            if not r.concept_name.startswith(("hazard_", "blunder_", "avoid_")):
                sc = float(r.evaluate(best_coords))
                if sc > matched_rule_score and sc >= 0.35:
                    matched_rule_score = sc
                    matched_wisdom_rule = r.concept_name

    top_shifts = sorted(coord_delta.items(), key=lambda kv: (-abs(float(kv[1])), kv[0]))[:2]
    shift_phrases = [
        f"{k.split('/')[-1]} {('+' if float(d) >= 0 else '')}{float(d):.2f}"
        for k, d in top_shifts
    ]
    shift_str = ", ".join(shift_phrases) if shift_phrases else "balanced tree posture"
    rule_clause = (
        f" [Triggered REM Rule: {matched_wisdom_rule} ({matched_rule_score:.2f})]"
        if matched_wisdom_rule
        else ""
    )

    if is_hazard:
        coaching_advice = (
            f"{played_san}{glyph} ({move_acc:.0f}% accuracy) lost {cp_drop / 100.0:.2f} pawns "
            f"and shifted your tree posture ({shift_str}). Stronger was {best_san}, "
            f"which preserves structure and safety.{rule_clause}"
        )
    elif glyph in ("!!", "!"):
        coaching_advice = (
            f"{played_san}{glyph} ({move_acc:.0f}% accuracy) was a standout move "
            f"with strong tree harmony ({shift_str}).{rule_clause}"
        )
    else:
        alt_note = f" (best engine line: {best_san})" if best_san != played_san else " (top engine choice)"
        coaching_advice = (
            f"{played_san} ({move_acc:.0f}% accuracy) keeps solid equilibrium{alt_note}.{rule_clause}"
        )

    return {
        "played_san": played_san,
        "best_san": best_san,
        "classification": classification,
        "glyph": glyph,
        "is_hazard": is_hazard,
        "move_accuracy_pct": move_acc,
        "game_accuracy_pct": game_acc,
        "player_elo": round(float(player_elo), 1),
        "estimated_performance_elo": est_game_elo,
        "matched_wisdom_rule": matched_wisdom_rule,
        "matched_rule_score": round(float(matched_rule_score), 3),
        "coordinate_shift_summary": shift_str,
        "coaching_advice": coaching_advice,
    }

