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
from .reflection_classifier import LearnedLeaf
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
        + 0.08 * min(4, center_pawns)
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


_CORE_CENTER_SQUARES = frozenset({chess.D4, chess.E4, chess.D5, chess.E5})
_EXTENDED_CENTER_SQUARES = frozenset(
    {
        chess.C4,
        chess.F4,
        chess.C5,
        chess.F5,
        chess.C3,
        chess.F3,
        chess.C6,
        chess.F6,
        chess.D3,
        chess.E3,
        chess.D6,
        chess.E6,
    }
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


def rank_principled_legal_moves_san(board: chess.Board, *, limit: int = 6) -> list[str]:
    """Return up to `limit` legal moves in SAN ordered by natural chess principles (checks, captures, center, development)."""
    legal = list(board.legal_moves)
    if not legal:
        return []

    them = not board.turn
    scored: list[tuple[float, str]] = []
    for move in legal:
        san = board.san(move)
        score = 0.0
        piece_type = board.piece_type_at(move.from_square)
        is_cap = board.is_capture(move)
        is_chk = board.gives_check(move)
        dest_attacked = board.is_attacked_by(them, move.to_square)

        if is_chk:
            if is_cap or board.fullmove_number > 4 or not dest_attacked:
                score += 0.6 if (board.fullmove_number <= 4 and not is_cap) else 3.0
        if is_cap:
            score += 2.5
        if board.is_castling(move):
            score += 2.2
        if move.to_square in _CORE_CENTER_SQUARES:
            score += 1.8
        elif move.to_square in _EXTENDED_CENTER_SQUARES:
            score += 1.2
        if piece_type in (chess.KNIGHT, chess.BISHOP):
            score += 1.05
        elif piece_type == chess.PAWN and chess.square_file(move.to_square) in (2, 3, 4):
            score += 0.90
        elif piece_type == chess.KING and not board.is_castling(move) and not board.is_check():
            score -= 1.5
        if not dest_attacked:
            score += 0.45
        scored.append((score, san))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [san for _score, san in scored[: max(1, int(limit))]]


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
    legal_candidates = rank_principled_legal_moves_san(board, limit=6)
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

    top_legal = rank_principled_legal_moves_san(board, limit=6)
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
