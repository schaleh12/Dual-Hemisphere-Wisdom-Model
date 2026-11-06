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
from dataclasses import dataclass
import math
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
    turn_label = "White" if us == chess.WHITE else "Black"

    legal_moves = list(board.legal_moves) if board.turn == us else []
    if board.turn != us:
        temp_b = board.copy()
        temp_b.turn = us
        our_moves = list(temp_b.legal_moves)
    else:
        our_moves = legal_moves

    # 1. Queen -> philosophical/being/agency (Red / Lightning: Volition & Attacking Drive)
    our_queens = list(board.pieces(chess.QUEEN, us))
    queen_moves = [m for m in our_moves if board.piece_type_at(m.from_square) == chess.QUEEN]
    capture_moves = [m for m in our_moves if board.is_capture(m)]
    checking_moves = [m for m in our_moves if board.gives_check(m)]
    enemy_half_ranks = range(4, 8) if us == chess.WHITE else range(0, 4)
    forward_attacks = [
        m for m in our_moves if chess.square_rank(m.to_square) in enemy_half_ranks
    ]
    queen_presence = 0.32 if our_queens else 0.08
    agency = _clamp01(
        queen_presence
        + 0.035 * min(12, len(queen_moves))
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
        existence = _clamp01(
            0.48
            + 0.11 * min(3, shield_pawns)
            + (0.10 if back_rank_home else -0.12)
            + 0.03 * min(4, safe_escape_squares)
            - 0.11 * min(5, enemy_king_attackers)
            - (0.36 if in_check_now else 0.0)
        )

    # 3. Bishop -> philosophical/structure/order (Magenta / Wind: Alignment & Coordination)
    our_bishops = list(board.pieces(chess.BISHOP, us))
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
        + 0.03 * min(10, len(bishop_moves))
        + 0.35 * coord_ratio
        - 0.14 * min(3, hanging_count)
    )

    # 4. Knight -> philosophical/structure/chaos (Green / Ice: Nonlinear Forks & Tactical Shatter)
    knight_moves = [m for m in our_moves if board.piece_type_at(m.from_square) == chess.KNIGHT]
    center_ring = {
        chess.C3, chess.D3, chess.E3, chess.F3,
        chess.C4, chess.D4, chess.E4, chess.F4,
        chess.C5, chess.D5, chess.E5, chess.F5,
        chess.C6, chess.D6, chess.E6, chess.F6,
    }
    active_knights = sum(1 for sq in board.pieces(chess.KNIGHT, us) if sq in center_ring)
    chaos = _clamp01(
        0.12
        + 0.06 * min(2, active_knights)
        + 0.02 * min(8, len(knight_moves))
        + 0.04 * min(5, len(capture_moves))
        + 0.06 * min(3, len(checking_moves))
        + 0.12 * min(3, hanging_count)
        + (0.22 if board.is_check() else 0.0)
    )

    # 5. Pawn -> philosophical/persistence/continuity (Blue / Fire: Connected Phalanx & Flow)
    our_pawns = list(board.pieces(chess.PAWN, us))
    pawn_set = set(our_pawns)
    connected_pawns = 0
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
    center_four = {chess.D4, chess.E4, chess.D5, chess.E5}
    center_pawns = sum(1 for sq in our_pawns if sq in center_four)
    continuity = _clamp01(
        0.24
        + 0.055 * min(8, connected_pawns)
        + 0.12 * min(4, center_pawns)
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
        fact=0.25 if (board.is_check() or hanging_count > 0 or existence <= 0.35) else 0.65,
        arousal=max(0.45, chaos),
        resonance=order,
    )

    ranked_rows = sorted(
        PHILOSOPHICAL_ABSTRACTION_MATRIX,
        key=lambda r: phil_coords.get(r.leaf_id, 0.0),
        reverse=True,
    )
    top_row = ranked_rows[0]

    check_tag = " (IN CHECK)" if board.is_check() else ""
    summary = (
        f"{turn_label} to move{check_tag} | Dominant: {top_row.chess_piece} "
        f"({top_row.color_name} / {top_row.element} = {phil_coords[top_row.leaf_id]:.2f}) | "
        f"King Shelter={existence:.2f}, Attack Volition={agency:.2f}, Order={order:.2f}, Chaos={chaos:.2f}"
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


_PIECE_VALUES: dict[int, float] = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.15,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}


@dataclass(frozen=True)
class CandidateMoveEvaluation:
    """Right-Brain 40-node evaluation of a legal candidate move on `chess.Board`."""

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "san": self.san,
            "uci": self.uci,
            "composite_score": round(float(self.composite_score), 4),
            "settled": bool(self.settled),
            "clarity": round(float(self.clarity), 4),
            "confidence": round(float(self.confidence), 4),
            "matched_concept": self.matched_concept,
            "dominant_piece": self.dominant_piece,
            "dominant_color": self.dominant_color,
            "is_blunder": bool(self.is_blunder),
            "is_repetition": bool(self.is_repetition),
            "shadow_warnings": list(self.shadow_warnings),
        }


def compute_candidate_move_fuzzy_coordinates(
    board: chess.Board,
    move: chess.Move,
    *,
    perspective: chess.Color | None = None,
    learned_leaves: Sequence[LearnedLeaf] = (),
) -> tuple[dict[str, float], bool, float, ChessBoardPosture, tuple[str, ...]]:
    """Compute the 40-node Fuzzy Tree coordinate dictionary and 1-ply blunder check for `move` on `board`."""
    us = board.turn if perspective is None else perspective
    them = not us
    piece_type = board.piece_type_at(move.from_square) or chess.PAWN
    moving_val = _PIECE_VALUES.get(piece_type, 1.0)
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

    dest_attacked_by_them = next_board.is_attacked_by(them, move.to_square)
    dest_defended_by_us = next_board.is_attacked_by(us, move.to_square)
    is_blunder = False
    blunder_penalty = 0.0
    if dest_attacked_by_them:
        if not dest_defended_by_us and moving_val > captured_val + 0.5:
            is_blunder = True
            blunder_penalty = moving_val - captured_val
        elif dest_defended_by_us:
            min_att_val = min(
                (
                    _PIECE_VALUES.get(next_board.piece_type_at(a_sq) or chess.PAWN, 1.0)
                    for a_sq in next_board.attackers(them, move.to_square)
                    if next_board.piece_type_at(a_sq) != chess.KING
                ),
                default=99.0,
            )
            if (moving_val - captured_val) - min_att_val >= 1.5:
                is_blunder = True
                blunder_penalty = 0.65 * ((moving_val - captured_val) - min_att_val)

    for sq, pc in next_board.piece_map().items():
        if pc.color != us or pc.piece_type == chess.KING or sq == move.to_square:
            continue
        p_val = _PIECE_VALUES.get(pc.piece_type, 1.0)
        if p_val >= 3.0 and next_board.is_attacked_by(them, sq) and not next_board.is_attacked_by(us, sq):
            net_unprotected = p_val - captured_val
            if net_unprotected >= 1.5 and net_unprotected > blunder_penalty:
                is_blunder = True
                blunder_penalty = net_unprotected

    post_posture = evaluate_chess_board_posture(next_board, perspective=us)
    shadow_names = tuple(w.shadow_pathology for w in post_posture.board_shadow_warnings)

    coords: dict[str, float] = {k: 0.52 for k in FOUNDATIONAL_LEAVES}
    for leaf in learned_leaves:
        coords[leaf.leaf_id] = 0.35
    for leaf_id, val in post_posture.philosophical_coordinates.items():
        coords[leaf_id] = float(val)
    for w in post_posture.board_shadow_warnings:
        if w.shadow_leaf_id:
            coords[w.shadow_leaf_id] = float(w.severity)
    coords["cultural/efficiency/wastelessness"] = (
        0.25 if is_blunder else round(min(0.95, 0.72 + 0.06 * captured_val), 4)
    )
    coords["cultural/freedom/self-sufficiency"] = float(
        post_posture.philosophical_coordinates.get("philosophical/being/existence", 0.55)
    )
    return coords, is_blunder, blunder_penalty, post_posture, shadow_names


def evaluate_candidate_moves_with_right_brain(
    board: chess.Board,
    rules: Sequence[FuzzyConceptRule] = (),
    *,
    learned_leaves: Sequence[LearnedLeaf] = (),
    limit: int = 6,
) -> list[CandidateMoveEvaluation]:
    """Rank legal moves by combining pure `python-chess` 1-ply board geometry with learned Right-Brain `rules`."""
    legal = list(board.legal_moves)
    if not legal:
        return []

    us = board.turn
    them = not us
    rule_tuple = tuple(rules)
    learned_tuple = tuple(learned_leaves)
    evaluations: list[CandidateMoveEvaluation] = []

    for move in legal:
        san = board.san(move)
        uci = move.uci()
        piece_type = board.piece_type_at(move.from_square) or chess.PAWN
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
                )
            )
            continue

        dest_attacked_by_them = next_board.is_attacked_by(them, move.to_square)
        is_rep = next_board.is_repetition(2)
        coords, is_blunder, blunder_penalty, post_posture, shadow_names = (
            compute_candidate_move_fuzzy_coordinates(
                board,
                move,
                perspective=us,
                learned_leaves=learned_tuple,
            )
        )
        for r in rule_tuple:
            for req_leaf in r.target_leaves:
                coords.setdefault(req_leaf, 0.0)

        is_hazard_match = False
        if rule_tuple:
            activations = _evaluate_rules_with_inhibition(rule_tuple, coords)
            clarity, confidence, matched, is_confused, _reason = compute_contemplation(activations)
            settled = not is_confused
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
                learned_bonus = min(learned_bonus, -0.65 * float(confidence))
            elif matched and not matched.startswith(("balanced_", "mutual_")) and not is_hazard_match:
                learned_bonus = max(learned_bonus, 0.55 * float(confidence))
        else:
            clarity, confidence, matched, settled, learned_bonus = 0.60, 0.60, None, True, 0.0

        if is_hazard_match:
            right_brain_score = (
                -0.45 * float(clarity)
                - 0.55 * float(confidence)
                + learned_bonus
                - 0.25
                - 0.45 * len(shadow_names)
            )
        else:
            right_brain_score = (
                0.85 * float(clarity)
                + 0.65 * float(confidence)
                + learned_bonus
                + (0.35 if settled else -0.25)
                - 0.45 * len(shadow_names)
            )
        leaf_posture_score = (
            float(coords.get("cultural/efficiency/wastelessness", 0.50))
            + 0.85 * float(coords.get("philosophical/being/existence", 0.50))
            + 0.75 * float(coords.get("philosophical/structure/order", 0.50))
            + 0.65 * float(coords.get("philosophical/persistence/continuity", 0.50))
            + 0.55 * float(coords.get("philosophical/persistence/discreteness", 0.50))
            + 0.45 * float(coords.get("philosophical/being/agency", 0.50))
            - 0.35 * float(coords.get("philosophical/structure/chaos", 0.50))
        )
        composite = (
            right_brain_score
            + leaf_posture_score
            - 1.35 * blunder_penalty
            - (1.80 if is_rep else 0.0)
        )
        evaluations.append(
            CandidateMoveEvaluation(
                san=san,
                uci=uci,
                composite_score=composite,
                settled=settled,
                clarity=float(clarity),
                confidence=float(confidence),
                matched_concept=matched,
                dominant_piece=post_posture.dominant_piece,
                dominant_color=post_posture.dominant_color,
                is_blunder=is_blunder,
                is_repetition=is_rep,
                shadow_warnings=shadow_names,
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
            f"Chess table — Move {int(board.fullmove_number)} (FEN {board.fen()}): "
            f"{turn_label} to move{check_clause}. {legal_clause}"
        )

    if user_note and user_note.strip():
        return f"{user_note.strip()} [Board context: {base}]"
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
    ) -> None:
        self.base_sensors = base_sensors
        self.board_posture = board_posture
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
        clean_for_minilm = text.split(" [Board context:")[0].strip() or text
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
            for leaf_id, board_val in self.board_posture.philosophical_coordinates.items():
                anchored[leaf_id] = float(board_val)
            return anchored, dict(exp_raw), queried, skipped

        # Subsequent calls (Clause 1 at comma and full-sentence Pass 2+ Theory) score
        # the proposed plan/chat text directly so any mismatch with the board triggers dissonance!
        return coords, exp_raw, queried, skipped


def merge_board_fact_with_chat_sensors(
    board_posture: ChessBoardPosture,
    sensors: ThreeBridgeSensors,
) -> BoardAnchoredBridgeSensors:
    """Wrap `ThreeBridgeSensors` so Pass-1 `Fact` is anchored to `board_posture` while `Theory` is scored from text."""
    return BoardAnchoredBridgeSensors(sensors, board_posture)


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
                with self._chess_engine_mod.SimpleEngine.popen_uci(str(self.stockfish_path)) as eng:
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
                with self._chess_engine_mod.SimpleEngine.popen_uci(str(self.stockfish_path)) as eng:
                    if clamped_elo >= 1320.0:
                        eng.configure(
                            {
                                "UCI_LimitStrength": True,
                                "UCI_Elo": int(round(clamped_elo)),
                            }
                        )
                        depth_lim = max(2, min(10, int((clamped_elo - 900.0) / 160.0)))
                        res = eng.play(board, self._chess_engine_mod.Limit(depth=depth_lim, time=0.06))
                    else:
                        skill = max(0, min(6, int((clamped_elo - 600.0) / 120.0)))
                        eng.configure({"Skill Level": skill})
                        nodes_lim = max(40, int((clamped_elo - 500.0) * 2.0))
                        depth_lim = max(1, min(4, int((clamped_elo - 500.0) / 220.0)))
                        res = eng.play(
                            board,
                            self._chess_engine_mod.Limit(nodes=nodes_lim, depth=depth_lim, time=0.04),
                        )
                    if res.move is not None and res.move in board.legal_moves:
                        san = board.san(res.move)
                        return {
                            "move_san": san,
                            "move_uci": res.move.uci(),
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
        played_coords, played_is_blunder, played_b_pen, _p_post, _p_shad = (
            compute_candidate_move_fuzzy_coordinates(
                board_before,
                played_move,
                perspective=us,
                learned_leaves=learned_tuple,
            )
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
                with self._chess_engine_mod.SimpleEngine.popen_uci(str(self.stockfish_path)) as eng:
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
            )
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

