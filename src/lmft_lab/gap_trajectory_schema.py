"""Sub-Step 2A: Matched Contrastive `<|subliminal_gap|>` Data Contract & 6-Piece Chess Chord Compiler.

Plain-English Purpose:
1. `ContrastiveGapPair`:
   Locks down the data structure for Stage 2 and Stage 3 so that Branch A and Branch B
   share the **exact same `user_prompt` and `clause_1_prefix`** (the words before
   `<|subliminal_gap|>`), while holding two genuinely different 52-float Right-Brain
   chords (`chord_a` vs. `chord_b`, L2 distance >= 0.35) and two distinct post-gap
   continuations (`continuation_a` vs. `continuation_b`).
2. `compile_chess_posture_chords`:
   Maps the 6 Chess Piece Archetypes onto the 6 Founding Philosophical Leaves:
   - `queen_volition`    -> `philosophical/being/agency`          (Queen: multi-directional volition)
   - `king_safety`       -> `philosophical/being/existence`       (King: survival / ground of being)
   - `bishop_alignment`  -> `philosophical/structure/order`       (Bishop: diagonal geometry)
   - `knight_disruption` -> `philosophical/structure/chaos`       (Knight: non-linear leaps / forks)
   - `pawn_chain_flow`   -> `philosophical/persistence/continuity`(Pawn: linked chain advance)
   - `rook_bastion`      -> `philosophical/persistence/discreteness` (Rook: sealed file / fortress)
   Every one of the 6 board coordinates actively shapes the resulting `thinker_vector`,
   `doer_vector`, and `feeler_vector` for both strategic postures (`CHESS_POSITIONAL_FORTRESS`
   vs. `CHESS_TACTICAL_STRIKE`) so different board states always yield distinct chords.
3. `build_post_gap_only_labels`:
   Masks all tokens before `<|subliminal_gap|>` with `-100` (`IGNORE_LABEL_INDEX`) so
   Stage 3 PyTorch LoRA training focuses 100% of its gradient on predicting the post-gap
   continuation from the 4 virtual tokens of `SubliminalGapProjector`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch
from torch import Tensor

from .philosophical_abstraction_matrix import (
    CHESS_PIECE_TO_LEAF_ID,
    PHILOSOPHICAL_SHADOW_LEAVES,
    detect_shadow_pathologies,
)
from .reflection_classifier import LearnedLeaf
from .subliminal_gap_splice import (
    DEFAULT_RUPTURE_DEADBAND,
    IGNORE_LABEL_INDEX,
    SUBLIMINAL_GAP_TOKEN,
    SubliminalChordVector,
    compute_cadence_rupture_drive,
    pool_departmental_child_chord,
)

VALID_GAP_DOMAINS = (
    "social_dilemma",
    "chess_intuition",
    "tone_conflict",
)

VALID_GAP_SPLITS = (
    "train",
    "holdout",
)

MIN_CHORD_L2_DISTANCE = 0.35

CHESS_PHILOSOPHICAL_CHILD_LEAVES: tuple[LearnedLeaf, ...] = (
    LearnedLeaf(
        "philosophical/structure",
        "positional_fortress_harmony",
        "Coordinated bishop diagonals, sealed rook files, and unbroken pawn chain shelter protecting the king.",
        anchor_weights=(0.35, 0.92, 1.00, 0.10, 0.88, 0.90, 0.85, 0.90),
    ),
    LearnedLeaf(
        "philosophical/being",
        "tactical_fork_sacrifice",
        "Dynamic queen volition and non-linear knight leaps shattering rigid lines in a forcing tactical strike.",
        anchor_weights=(1.00, 0.55, 0.20, 0.95, 0.30, 0.20, 0.88, 0.75),
    ),
) + PHILOSOPHICAL_SHADOW_LEAVES


def _clamp01(value: float, name: str = "value") -> float:
    if type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite float, got {value!r}")
    return round(min(1.0, max(0.0, float(value))), 6)


def _validate_unit_board_input(value: float, name: str) -> float:
    if type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite float in [0.0, 1.0], got {value!r}")
    if not (0.0 <= float(value) <= 1.0):
        raise ValueError(f"{name} must be in [0.0, 1.0], got {value!r}")
    return float(value)


def chord_l2_distance(chord_a: SubliminalChordVector, chord_b: SubliminalChordVector) -> float:
    """Compute Euclidean (L2) distance between two 52-float SubliminalChordVectors."""
    diff = chord_a.to_tensor(dtype=torch.float64) - chord_b.to_tensor(dtype=torch.float64)
    return round(float(torch.norm(diff).item()), 6)


@dataclass(frozen=True)
class ContrastiveGapPair:
    """One matched training/evaluation pair sharing an identical pre-gap prefix and opposite Right-Brain chords."""

    pair_id: str
    domain: str
    split: str
    user_prompt: str
    clause_1_prefix: str
    chord_a: SubliminalChordVector
    posture_label_a: str
    continuation_a: str
    chord_b: SubliminalChordVector
    posture_label_b: str
    continuation_b: str
    min_l2_distance: float = MIN_CHORD_L2_DISTANCE

    def __post_init__(self) -> None:
        if not isinstance(self.pair_id, str) or not self.pair_id.strip():
            raise ValueError("pair_id must be a nonempty string")
        if self.domain not in VALID_GAP_DOMAINS:
            raise ValueError(f"domain must be one of {VALID_GAP_DOMAINS}, got {self.domain!r}")
        if self.split not in VALID_GAP_SPLITS:
            raise ValueError(f"split must be one of {VALID_GAP_SPLITS}, got {self.split!r}")
        for field_name, val in (
            ("user_prompt", self.user_prompt),
            ("clause_1_prefix", self.clause_1_prefix),
            ("posture_label_a", self.posture_label_a),
            ("continuation_a", self.continuation_a),
            ("posture_label_b", self.posture_label_b),
            ("continuation_b", self.continuation_b),
        ):
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{field_name} must be a nonempty string")
            if SUBLIMINAL_GAP_TOKEN in val:
                raise ValueError(f"{field_name} must not contain raw {SUBLIMINAL_GAP_TOKEN!r}")

        if not isinstance(self.chord_a, SubliminalChordVector) or not isinstance(
            self.chord_b, SubliminalChordVector
        ):
            raise ValueError("chord_a and chord_b must be SubliminalChordVector instances")

        if self.posture_label_a.strip() == self.posture_label_b.strip():
            raise ValueError("posture_label_a and posture_label_b must be distinct")

        if self.continuation_a.strip().lower() == self.continuation_b.strip().lower():
            raise ValueError("continuation_a and continuation_b must be distinct clauses")

        dist = chord_l2_distance(self.chord_a, self.chord_b)
        if dist < float(self.min_l2_distance):
            raise ValueError(
                f"ContrastiveGapPair {self.pair_id!r} chord_a and chord_b are too close "
                f"(L2={dist:.4f} < {self.min_l2_distance:.4f})"
            )

    @property
    def chord_distance(self) -> float:
        return chord_l2_distance(self.chord_a, self.chord_b)

    def format_pre_gap_prompt(self) -> str:
        """Return the shared text up to and including `<|subliminal_gap|>` (identical for A and B)."""
        return f"User: {self.user_prompt.strip()}\nAssistant: {self.clause_1_prefix.strip()} {SUBLIMINAL_GAP_TOKEN}"

    def format_full_text(self, branch: str) -> str:
        """Return the full text for branch `'A'` or `'B'` with `<|subliminal_gap|>` between Clause 1 and Clause 2."""
        if branch == "A":
            cont = self.continuation_a.strip()
        elif branch == "B":
            cont = self.continuation_b.strip()
        else:
            raise ValueError(f"branch must be 'A' or 'B', got {branch!r}")
        return f"{self.format_pre_gap_prompt()} {cont}"

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "pair_id": self.pair_id,
            "domain": self.domain,
            "split": self.split,
            "user_prompt": self.user_prompt,
            "clause_1_prefix": self.clause_1_prefix,
            "chord_a": self.chord_a.to_dict(),
            "posture_label_a": self.posture_label_a,
            "continuation_a": self.continuation_a,
            "chord_b": self.chord_b.to_dict(),
            "posture_label_b": self.posture_label_b,
            "continuation_b": self.continuation_b,
            "chord_l2_distance": self.chord_distance,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ContrastiveGapPair":
        if data.get("schema_version", 1) != 1:
            raise ValueError(f"Unsupported ContrastiveGapPair schema_version: {data.get('schema_version')}")
        return cls(
            pair_id=str(data["pair_id"]),
            domain=str(data["domain"]),
            split=str(data["split"]),
            user_prompt=str(data["user_prompt"]),
            clause_1_prefix=str(data["clause_1_prefix"]),
            chord_a=SubliminalChordVector.from_dict(data["chord_a"]),
            posture_label_a=str(data["posture_label_a"]),
            continuation_a=str(data["continuation_a"]),
            chord_b=SubliminalChordVector.from_dict(data["chord_b"]),
            posture_label_b=str(data["posture_label_b"]),
            continuation_b=str(data["continuation_b"]),
        )


def compile_chess_posture_chords(
    *,
    queen_volition: float,
    king_safety: float,
    bishop_alignment: float,
    knight_disruption: float,
    pawn_chain_flow: float,
    rook_bastion: float,
    rupture_deadband: float = DEFAULT_RUPTURE_DEADBAND,
) -> tuple[SubliminalChordVector, SubliminalChordVector]:
    """Compile (`Chord_A_Positional_Fortress`, `Chord_B_Tactical_Strike`) from the 6 Chess Piece Archetypes.

    All 6 board coordinates actively shape `thinker_vector`, `doer_vector`, and `feeler_vector`:
    - `queen_volition`    -> `philosophical/being/agency`          (Queen)
    - `king_safety`       -> `philosophical/being/existence`       (King)
    - `bishop_alignment`  -> `philosophical/structure/order`       (Bishop)
    - `knight_disruption` -> `philosophical/structure/chaos`       (Knight)
    - `pawn_chain_flow`   -> `philosophical/persistence/continuity`(Pawn)
    - `rook_bastion`      -> `philosophical/persistence/discreteness` (Rook)
    """
    q_ag = _validate_unit_board_input(queen_volition, "queen_volition")
    k_ex = _validate_unit_board_input(king_safety, "king_safety")
    b_or = _validate_unit_board_input(bishop_alignment, "bishop_alignment")
    n_ch = _validate_unit_board_input(knight_disruption, "knight_disruption")
    p_co = _validate_unit_board_input(pawn_chain_flow, "pawn_chain_flow")
    r_di = _validate_unit_board_input(rook_bastion, "rook_bastion")

    structural_stability = (k_ex + b_or + p_co + r_di) / 4.0
    tactical_volatility = (q_ag + n_ch + (1.0 - k_ex)) / 3.0

    board_coords = {
        CHESS_PIECE_TO_LEAF_ID["Queen"]: q_ag,
        CHESS_PIECE_TO_LEAF_ID["King"]: k_ex,
        CHESS_PIECE_TO_LEAF_ID["Bishop"]: b_or,
        CHESS_PIECE_TO_LEAF_ID["Knight"]: n_ch,
        CHESS_PIECE_TO_LEAF_ID["Pawn"]: p_co,
        CHESS_PIECE_TO_LEAF_ID["Rook"]: r_di,
    }
    board_shadows = {
        w.shadow_leaf_id: w.severity
        for w in detect_shadow_pathologies(board_coords)
        if w.shadow_leaf_id
    }

    # =========================================================================
    # Branch A: Stoic Positional Fortress (consolidate lines, shelter King, conserve material)
    # =========================================================================
    phil_child_a = pool_departmental_child_chord(
        {
            "philosophical/structure/positional_fortress_harmony": _clamp01(0.45 + 0.50 * structural_stability),
            **{
                k: v for k, v in board_shadows.items()
                if k in ("philosophical/being/paranoia", "philosophical/structure/prejudice", "philosophical/persistence/narcissism")
            },
        },
        department_prefix="philosophical/",
        learned_leaves=CHESS_PHILOSOPHICAL_CHILD_LEAVES,
    )
    thinker_a = (
        _clamp01(0.35 * q_ag + 0.25 * b_or),                  # Agency (measured positional maneuvering)
        _clamp01(0.55 + 0.45 * k_ex),                         # Existence (King shelter prioritized)
        _clamp01(0.50 + 0.50 * b_or),                         # Order (Bishop diagonal discipline)
        _clamp01(0.25 * n_ch * (1.0 - 0.5 * r_di)),           # Chaos (dampened by Rook bastion)
        _clamp01(0.50 + 0.50 * p_co),                         # Continuity (Pawn chain flow)
        _clamp01(0.45 + 0.55 * r_di),                         # Discreteness (Rook file/fortress blockade)
        _clamp01(0.30 + 0.35 * b_or + 0.20 * q_ag),           # Theory (long-term positional plan)
        _clamp01(0.55 + 0.25 * k_ex + 0.20 * r_di),           # Fact (concrete structural verification)
    ) + phil_child_a

    doer_a = (
        _clamp01(0.20 + 0.25 * r_di + 0.15 * q_ag),           # Power (restrained blockade leverage)
        _clamp01(0.60 + 0.20 * k_ex + 0.20 * p_co),           # Wastelessness (material & tempo conservation)
        _clamp01(0.55 + 0.30 * b_or + 0.15 * p_co),           # Honor (sound principled structure)
        _clamp01(0.50 + 0.30 * p_co + 0.20 * b_or),           # Fairness (balanced piece coordination)
        _clamp01(0.55 + 0.25 * r_di + 0.20 * k_ex),           # Self-Sufficiency (Stoic fortress autonomy)
        _clamp01(0.45 + 0.35 * k_ex + 0.20 * (1.0 - n_ch)),   # Needlessness (calm poker-face sufficiency)
        _clamp01(0.30 + 0.25 * b_or + 0.20 * p_co),           # Intention (steady positional resolve)
        _clamp01(0.20 + 0.20 * p_co + 0.15 * r_di),           # Execution (quiet prophylactic move)
        0.0,                                                  # social_drift_flag
    ) + (0.0,) * 8

    clarity_a = _clamp01(0.60 + 0.25 * structural_stability + 0.15 * (1.0 - n_ch))
    arousal_a = _clamp01(0.15 + 0.25 * n_ch + 0.15 * (1.0 - k_ex))
    feeler_a = (
        _clamp01(0.15 + 0.25 * n_ch),                         # Novelty
        _clamp01(0.35 + 0.30 * (1.0 - k_ex) + 0.20 * q_ag),   # Salience
        _clamp01(0.55 + 0.30 * k_ex + 0.15 * p_co),           # Valence (positional harmony/comfort)
        arousal_a,                                            # Arousal (low, composed)
        _clamp01(0.50 + 0.30 * b_or + 0.20 * p_co),           # Resonance (piece synchronization)
        _clamp01(0.50 + 0.35 * p_co + 0.15 * r_di),           # Entrainment (steady rhythmic advance)
        clarity_a,                                            # Clarity
        _clamp01(0.55 + 0.30 * structural_stability),         # Confidence
    ) + (0.0,) * 8

    dissonance_a = _clamp01(0.15 * n_ch * (1.0 - k_ex))
    rupture_a = compute_cadence_rupture_drive(
        arousal=arousal_a,
        clarity=clarity_a,
        valence=feeler_a[2],
        discreteness=thinker_a[5],
        feeler_child_peak=0.0,
        dissonance_pressure=dissonance_a,
        deadband=rupture_deadband,
    )
    cadence_a = (
        _clamp01(0.75 + 0.20 * structural_stability),
        dissonance_a,
        rupture_a,
    )

    # =========================================================================
    # Branch B: Sharp Tactical Strike (unleash Queen volition + Knight fork disruption)
    # =========================================================================
    phil_child_b = pool_departmental_child_chord(
        {
            "philosophical/being/tactical_fork_sacrifice": _clamp01(0.50 + 0.50 * tactical_volatility),
            **{
                k: v for k, v in board_shadows.items()
                if k in ("philosophical/being/psychosis", "philosophical/structure/corruption", "philosophical/persistence/mania_greed")
            },
        },
        department_prefix="philosophical/",
        learned_leaves=CHESS_PHILOSOPHICAL_CHILD_LEAVES,
    )
    thinker_b = (
        _clamp01(0.55 + 0.45 * q_ag),                         # Agency (Queen maximum volition unleashed)
        _clamp01(0.35 * k_ex + 0.15 * r_di),                  # Existence (accepting King-shelter risk)
        _clamp01(0.30 * b_or + 0.20 * q_ag),                  # Order (subordinated to tactical breakthrough)
        _clamp01(0.55 + 0.45 * n_ch),                         # Chaos (Knight non-linear jump/fork)
        _clamp01(0.35 * p_co + 0.25 * q_ag),                  # Continuity (sacrificing quiet flow for tempo)
        _clamp01(0.25 * r_di + 0.15 * (1.0 - n_ch)),          # Discreteness (shattering fortress walls)
        _clamp01(0.55 + 0.30 * q_ag + 0.15 * n_ch),           # Theory (calculating sharp sacrifice line)
        _clamp01(0.45 + 0.35 * q_ag + 0.20 * b_or),           # Fact (concrete forcing check/capture)
    ) + phil_child_b

    doer_b = (
        _clamp01(0.60 + 0.25 * q_ag + 0.15 * n_ch),           # Power (maximum forcing initiative)
        _clamp01(0.25 * k_ex + 0.20 * r_di),                  # Wastelessness (low: willing to sacrifice material)
        _clamp01(0.45 + 0.30 * q_ag + 0.15 * b_or),           # Honor (bold direct confrontation)
        _clamp01(0.35 + 0.25 * b_or + 0.15 * p_co),           # Fairness
        _clamp01(0.50 + 0.35 * q_ag + 0.15 * n_ch),           # Self-Sufficiency (active initiative)
        _clamp01(0.10 + 0.20 * k_ex * (1.0 - tactical_volatility)),  # Needlessness (low: urgent combat drive)
        _clamp01(0.65 + 0.20 * q_ag + 0.15 * n_ch),           # Intention (decisive strike commitment)
        _clamp01(0.60 + 0.25 * q_ag + 0.15 * n_ch),           # Execution (immediate forcing move)
        0.0,                                                  # social_drift_flag
    ) + (0.0,) * 8

    clarity_b = _clamp01(0.50 + 0.25 * q_ag - 0.15 * (1.0 - k_ex))
    arousal_b = _clamp01(0.60 + 0.25 * n_ch + 0.15 * q_ag)
    feeler_b = (
        _clamp01(0.55 + 0.30 * n_ch + 0.15 * q_ag),           # Novelty (unexpected tactical leap)
        _clamp01(0.65 + 0.20 * q_ag + 0.15 * (1.0 - k_ex)),   # Salience (critical tactical turning point)
        _clamp01(0.40 + 0.30 * q_ag + 0.15 * k_ex),           # Valence (fierce competitive focus)
        arousal_b,                                            # Arousal (high combat tension)
        _clamp01(0.45 + 0.30 * q_ag + 0.25 * b_or),           # Resonance (attacking piece coordination)
        _clamp01(0.40 + 0.35 * q_ag + 0.25 * p_co),           # Entrainment (attacking momentum)
        clarity_b,                                            # Clarity
        _clamp01(0.60 + 0.25 * q_ag + 0.15 * n_ch),           # Confidence
    ) + (0.0,) * 8

    dissonance_b = _clamp01(0.30 + 0.35 * (1.0 - k_ex) + 0.20 * n_ch)
    rupture_b = compute_cadence_rupture_drive(
        arousal=arousal_b,
        clarity=clarity_b,
        valence=feeler_b[2],
        discreteness=thinker_b[5],
        feeler_child_peak=0.0,
        dissonance_pressure=dissonance_b,
        deadband=rupture_deadband,
    )
    cadence_b = (
        _clamp01(0.50 + 0.20 * q_ag - 0.15 * (1.0 - k_ex)),
        dissonance_b,
        rupture_b,
    )

    return (
        SubliminalChordVector(
            doer_vector=doer_a,
            thinker_vector=thinker_a,
            feeler_vector=feeler_a,
            cadence_vector=cadence_a,
        ),
        SubliminalChordVector(
            doer_vector=doer_b,
            thinker_vector=thinker_b,
            feeler_vector=feeler_b,
            cadence_vector=cadence_b,
        ),
    )


def build_post_gap_only_labels(
    input_ids: Tensor,
    *,
    gap_token_id: int,
    ignore_index: int = IGNORE_LABEL_INDEX,
) -> Tensor:
    """Mask all tokens up to and including `<|subliminal_gap|>` with `-100` (`ignore_index`).

    Why this matters for Stage 3 LoRA training:
    - Because `user_prompt` and `clause_1_prefix` before `<|subliminal_gap|>` are 100%
      identical in Branch A and Branch B, grading pre-gap tokens would waste gradient
      descent on words that do not depend on the Right Brain.
    - Masking all pre-gap tokens with `-100` ensures 100% of the causal LM loss grades
      ONLY the post-gap continuation tokens (`continuation_a` vs. `continuation_b`).
    """
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape [batch, seq_len], got {tuple(input_ids.shape)}")

    labels = input_ids.clone()
    batch_size, seq_len = labels.shape
    for b in range(batch_size):
        gap_positions = (input_ids[b] == gap_token_id).nonzero(as_tuple=False).flatten()
        if gap_positions.numel() == 0:
            raise ValueError(f"Row {b} of input_ids contains no gap_token_id ({gap_token_id})")
        first_gap_idx = int(gap_positions[0].item())
        labels[b, : first_gap_idx + 1] = ignore_index
    return labels


def validate_train_holdout_disjointness(
    train_pairs: Sequence[ContrastiveGapPair],
    holdout_pairs: Sequence[ContrastiveGapPair],
) -> None:
    """Verify zero overlap in `pair_id`, `user_prompt`, or `clause_1_prefix` between train and holdout."""
    if not train_pairs or not holdout_pairs:
        raise ValueError("Both train_pairs and holdout_pairs must be nonempty")

    for p in train_pairs:
        if p.split != "train":
            raise ValueError(f"Pair {p.pair_id!r} in train_pairs has split={p.split!r}")
    for p in holdout_pairs:
        if p.split != "holdout":
            raise ValueError(f"Pair {p.pair_id!r} in holdout_pairs has split={p.split!r}")

    train_ids = {p.pair_id for p in train_pairs}
    holdout_ids = {p.pair_id for p in holdout_pairs}
    if train_ids & holdout_ids:
        raise ValueError(f"Overlapping pair_ids between train and holdout: {sorted(train_ids & holdout_ids)}")

    train_prompts = {p.user_prompt.strip().lower() for p in train_pairs}
    holdout_prompts = {p.user_prompt.strip().lower() for p in holdout_pairs}
    if train_prompts & holdout_prompts:
        raise ValueError("Overlapping user_prompts detected between train and holdout splits")

    train_prefixes = {p.clause_1_prefix.strip().lower() for p in train_pairs}
    holdout_prefixes = {p.clause_1_prefix.strip().lower() for p in holdout_pairs}
    if train_prefixes & holdout_prefixes:
        raise ValueError("Overlapping clause_1_prefixes detected between train and holdout splits")


def save_contrastive_gap_pairs_jsonl(
    path: str | Path,
    pairs: Sequence[ContrastiveGapPair],
) -> None:
    """Atomically write a sequence of `ContrastiveGapPair` items to a `.jsonl` file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    lines = [json.dumps(p.to_dict(), ensure_ascii=False) for p in pairs]
    tmp_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    tmp_path.replace(out_path)


def load_contrastive_gap_pairs_jsonl(path: str | Path) -> tuple[ContrastiveGapPair, ...]:
    """Load and validate a sequence of `ContrastiveGapPair` items from a `.jsonl` file."""
    in_path = Path(path)
    raw_lines = [line.strip() for line in in_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return tuple(ContrastiveGapPair.from_dict(json.loads(line)) for line in raw_lines)
