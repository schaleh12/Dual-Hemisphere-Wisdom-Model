"""Stage 1 of the 'Three-Legged Race' Architecture: Departmental `<|subliminal_gap|>` Embedding Splice.

Plain-English Purpose:
Instead of translating the Right Brain's state into English stage directions in a prompt—or
crushing child leaves under the same parent branch into a single `max()` number—this module
organizes the Right Brain's live state into **4 strictly separate Departmental Vectors**
(zero English words, zero hardcoded feeling slots, zero cross-department index leakage)
and surgically splices them directly into a PyTorch language model's hidden token embeddings
(`inputs_embeds`) whenever the Left Brain emits a `<|subliminal_gap|>` token:

1. Four Strictly Disjoint Departmental Vectors (`SubliminalChordVector`):
   - **Department 0 — `doer_vector` (Executive / Do-er: `Will <-> Cultural`, 17 floats)**:
     * `6` Founding Cultural leaves (`Power`, `Wastelessness`, `Honor`, `Fairness`,
       `Self-Sufficiency`, `Needlessness`)
     * `2` Founding Will leaves (`Intention`, `Execution`)
     * `1` Longitudinal `social_drift_flag`
     * `8` Cultural Child-Drawer Chord floats (`pool_departmental_child_chord` over active
       `cultural/*` Tier-2/Tier-3 child leaves).
   - **Department 1 — `thinker_vector` (Judicial / Thinker: `Truth <-> Philosophical`, 16 floats)**:
     * `6` Founding Philosophical leaves (`Agency`, `Existence`, `Order`, `Chaos`,
       `Continuity`, `Discreteness`)
     * `2` Founding Truth leaves (`Theory`, `Fact`)
     * `8` Philosophical Child-Drawer Chord floats (`pool_departmental_child_chord` over active
       `philosophical/*` Tier-2/Tier-3 child leaves).
   - **Department 2 — `feeler_vector` (Legislative / Feeler: `Contemplation <-> Experiential`, 16 floats)**:
     * `6` Founding Experiential leaves (`Novelty`, `Salience`, `Valence`, `Arousal`,
       `Resonance`, `Retention`)
     * `2` Founding Contemplation leaves (`Clarity`, `Confidence`)
     * `8` Experiential Child-Drawer Chord floats (`pool_departmental_child_chord` over active
       `experiential/*` Tier-2/Tier-3 child leaves).
       Because every child leaf maps deterministically to an 8-float basis signature inside its
       own department, sibling child concepts under the same parent branch (such as
       `experiential/phase/austere_sanctuary`, `experiential/phase/hollow_stillness`, and
       `experiential/phase/bittersweet_release`, or 10,000 future Tier-2/Tier-3 child leaves)
       produce **completely distinct Feeler vectors** without ever changing the 16-float Feeler
       department size.
   - **Department 3 — `cadence_vector` (Three-Legged Race Cadence & Ankle-Rope, 3 floats)**:
     * `branch_harmony`: agreement across the 3 Right-Brain branches.
     * `dissonance_pressure`: friction from cognitive dissonance or low clarity.
     * `cadence_rupture_drive`: deadband-gated non-verbal tug (`deadband = 0.45`) that stays
       at `0.0` during normal calm speech (preventing twitchiness) and spikes when high
       arousal/discreteness or experiential strain collides with low clarity—pushing the
       Left Brain to break smooth chatbot syntax into stark or spelled-out cadence (`M-A-C-H-I-N-E`).

2. `SubliminalGapProjector` (`nn.Module`):
   - Projects each of the 4 disjoint Departmental Vectors through its own dedicated neural head
     into `K = 4` virtual non-verbal token embeddings of shape `[4, hidden_size]`
     (`Do-er Pulse`, `Thinker Structure`, `Feeler Chord`, and `Cadence/Rupture`).

3. `splice_subliminal_gaps_into_embeds`:
   - Replaces every `<|subliminal_gap|>` token ID in `input_ids` with the 4 projected
     departmental Fuzzy Tree embeddings in `inputs_embeds` and pads `labels` with `-100`
     at those 4 virtual positions so causal LM loss grades only the words while flowing
     gradients straight back through the non-verbal Fuzzy Tree splice.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import math
import torch
from torch import Tensor, nn

from .branch_archives import SocialDriftReport, ThreeClassifierDeliberation
from .reflection_classifier import (
    AROUSAL_LEAF,
    CLARITY_LEAF,
    CONFIDENCE_LEAF,
    EXECUTION_LEAF,
    FACT_LEAF,
    FOUNDATIONAL_LEAVES,
    INTENTION_LEAF,
    NOVELTY_LEAF,
    RESONANCE_LEAF,
    RETENTION_LEAF,
    SALIENCE_LEAF,
    THEORY_LEAF,
    VALENCE_LEAF,
    leaf_tier,
)
from .right_brain_equilibrium import RightBrainVibe

SUBLIMINAL_GAP_TOKEN = "<|subliminal_gap|>"
NUM_VIRTUAL_GAP_TOKENS = 4
IGNORE_LABEL_INDEX = -100
DEFAULT_RUPTURE_DEADBAND = 0.45
DEPARTMENT_CHILD_CHORD_DIM = 8

CULTURAL_FOUNDING_ORDER: tuple[str, ...] = FOUNDATIONAL_LEAVES[:6]
PHILOSOPHICAL_FOUNDING_ORDER: tuple[str, ...] = FOUNDATIONAL_LEAVES[6:12]

EXPERIENTIAL_FOUNDING_ORDER: tuple[str, ...] = (
    NOVELTY_LEAF,
    SALIENCE_LEAF,
    VALENCE_LEAF,
    AROUSAL_LEAF,
    RESONANCE_LEAF,
    RETENTION_LEAF,
)

DOER_FEATURE_NAMES: tuple[str, ...] = (
    CULTURAL_FOUNDING_ORDER
    + (INTENTION_LEAF, EXECUTION_LEAF, "doer/social_drift_flag")
    + tuple(f"doer/child_chord_{i}" for i in range(DEPARTMENT_CHILD_CHORD_DIM))
)  # 17 floats

THINKER_FEATURE_NAMES: tuple[str, ...] = (
    PHILOSOPHICAL_FOUNDING_ORDER
    + (THEORY_LEAF, FACT_LEAF)
    + tuple(f"thinker/child_chord_{i}" for i in range(DEPARTMENT_CHILD_CHORD_DIM))
)  # 16 floats

FEELER_FEATURE_NAMES: tuple[str, ...] = (
    EXPERIENTIAL_FOUNDING_ORDER
    + (CLARITY_LEAF, CONFIDENCE_LEAF)
    + tuple(f"feeler/child_chord_{i}" for i in range(DEPARTMENT_CHILD_CHORD_DIM))
)  # 16 floats

CADENCE_FEATURE_NAMES: tuple[str, ...] = (
    "cadence/branch_harmony",
    "cadence/dissonance_pressure",
    "cadence/cadence_rupture_drive",
)  # 3 floats

DOER_DEPT_DIM = len(DOER_FEATURE_NAMES)        # 17
THINKER_DEPT_DIM = len(THINKER_FEATURE_NAMES)  # 16
FEELER_DEPT_DIM = len(FEELER_FEATURE_NAMES)    # 16
CADENCE_DEPT_DIM = len(CADENCE_FEATURE_NAMES)  # 3

SUBLIMINAL_FEATURE_NAMES: tuple[str, ...] = (
    DOER_FEATURE_NAMES
    + THINKER_FEATURE_NAMES
    + FEELER_FEATURE_NAMES
    + CADENCE_FEATURE_NAMES
)
SUBLIMINAL_VECTOR_DIM = len(SUBLIMINAL_FEATURE_NAMES)  # 52 total across the 4 disjoint departments

# Strictly disjoint departmental index slices (zero overlap between any two departments!):
DOER_DEPARTMENT_INDICES: tuple[int, ...] = tuple(range(0, 17))
THINKER_DEPARTMENT_INDICES: tuple[int, ...] = tuple(range(17, 33))
FEELER_DEPARTMENT_INDICES: tuple[int, ...] = tuple(range(33, 49))
CADENCE_DEPARTMENT_INDICES: tuple[int, ...] = tuple(range(49, 52))


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"subliminal coordinate must be finite, got {value!r}")
    return round(min(1.0, max(0.0, float(value))), 6)


def child_drawer_basis_signature(leaf_id: str) -> tuple[float, ...]:
    """Compute a deterministic 8-float unit-peak basis signature in `[0.15, 1.0]^8` for a child leaf.

    Why this matters:
    - Sibling child leaves under the same parent branch (e.g., `experiential/phase/austere_sanctuary`,
      `experiential/phase/hollow_stillness`, and `experiential/phase/bittersweet_release`) have
      completely distinct 8-float signatures inside their department, without hardcoding any
      leaf names and without ever changing the department vector's size as 10,000 new leaves grow.
    """
    digest = hashlib.sha256(leaf_id.strip().encode("utf-8")).digest()
    raw = [0.15 + 0.85 * (digest[i] / 255.0) for i in range(DEPARTMENT_CHILD_CHORD_DIM)]
    peak = max(raw)
    return tuple(round(v / peak, 6) for v in raw)


def pool_departmental_child_chord(
    coordinates: Mapping[str, float],
    *,
    department_prefix: str,
    extra_notes: Mapping[str, float] | None = None,
) -> tuple[float, ...]:
    """Pool all active Tier-2/Tier-3 child leaves in a single department into an 8-float chord.

    - Only reads child leaves (`leaf_tier >= 2`) belonging to `department_prefix`
      (`"cultural/"`, `"philosophical/"`, or `"experiential/"`).
    - Preserves the distinct identity of sibling child leaves under the same parent branch
      (unlike a single `max()` number, which would crush `austere_sanctuary` and `hollow_stillness`
      into the same scalar).
    - Blends multiple simultaneously active child leaves via activation-weighted superposition
      scaled by the strongest active child leaf in that department.
    """
    if department_prefix not in ("cultural/", "philosophical/", "experiential/"):
        raise ValueError(f"Invalid department_prefix: {department_prefix!r}")

    merged: dict[str, float] = dict(coordinates)
    if extra_notes:
        for k, v in extra_notes.items():
            merged[k] = max(float(merged.get(k, 0.0)), float(v))

    active_children: list[tuple[str, float]] = []
    for leaf_id, val in sorted(merged.items()):
        if leaf_tier(leaf_id) >= 2 and leaf_id.startswith(department_prefix):
            clamped = _clamp01(float(val))
            if clamped > 0.0:
                active_children.append((leaf_id, clamped))

    if not active_children:
        return (0.0,) * DEPARTMENT_CHILD_CHORD_DIM

    total_weight = sum(weight for _, weight in active_children)
    peak_activation = max(weight for _, weight in active_children)

    blended = [0.0] * DEPARTMENT_CHILD_CHORD_DIM
    for leaf_id, weight in active_children:
        basis = child_drawer_basis_signature(leaf_id)
        for i in range(DEPARTMENT_CHILD_CHORD_DIM):
            blended[i] += weight * basis[i]

    return tuple(
        _clamp01(peak_activation * (blended[i] / total_weight))
        for i in range(DEPARTMENT_CHILD_CHORD_DIM)
    )


def compute_cadence_rupture_drive(
    *,
    arousal: float,
    discreteness: float,
    clarity: float,
    valence: float,
    feeler_child_peak: float,
    dissonance_pressure: float,
    deadband: float = DEFAULT_RUPTURE_DEADBAND,
    ankle_rope_sensitivity: float = 1.0,
) -> float:
    """Compute the deadband-gated non-verbal cadence-rupture drive in [0.0, 1.0].

    - Calm-talk deadband (`deadband = 0.45`): when internal strain is below `deadband`
      or when the scene feels harmonious and positive (`dissonance_pressure < 0.35` and
      `valence > 0.42`), this returns `0.0` so normal conversation never feels twitchy.
    - Above `deadband` under genuine strain, it scales smoothly with `ankle_rope_sensitivity`
      to yank the Left Brain out of smooth chatbot prose into stark or spelled-out cadence.
    """
    if not (0.0 <= deadband < 0.95):
        raise ValueError("deadband must be in [0.0, 0.95)")
    if ankle_rope_sensitivity <= 0.0 or not math.isfinite(ankle_rope_sensitivity):
        raise ValueError("ankle_rope_sensitivity must be a positive finite float")

    under_strain = (
        float(dissonance_pressure) >= 0.35
        or float(valence) <= 0.42
    )
    if not under_strain:
        return 0.0

    raw_intensity = (
        0.45 * max(float(arousal), float(feeler_child_peak))
        + 0.35 * float(discreteness)
        + 0.20 * (1.0 - float(clarity))
    )
    if raw_intensity <= deadband:
        return 0.0

    normalized_above_deadband = (raw_intensity - deadband) / (1.0 - deadband)
    return _clamp01(normalized_above_deadband * float(ankle_rope_sensitivity))


@dataclass(frozen=True)
class SubliminalChordVector:
    """Four strictly isolated Departmental Vectors injected into `<|subliminal_gap|>`."""

    doer_vector: tuple[float, ...]     # 17 floats (Executive / Do-er: Will <-> Cultural)
    thinker_vector: tuple[float, ...]  # 16 floats (Judicial / Thinker: Truth <-> Philosophical)
    feeler_vector: tuple[float, ...]   # 16 floats (Legislative / Feeler: Contemplation <-> Experiential)
    cadence_vector: tuple[float, ...]  #  3 floats (Three-Legged Race Cadence & Ankle-Rope)

    def __post_init__(self) -> None:
        expected_specs = (
            ("doer_vector", self.doer_vector, DOER_DEPT_DIM, DOER_FEATURE_NAMES),
            ("thinker_vector", self.thinker_vector, THINKER_DEPT_DIM, THINKER_FEATURE_NAMES),
            ("feeler_vector", self.feeler_vector, FEELER_DEPT_DIM, FEELER_FEATURE_NAMES),
            ("cadence_vector", self.cadence_vector, CADENCE_DEPT_DIM, CADENCE_FEATURE_NAMES),
        )
        for field_name, vec, expected_len, names in expected_specs:
            if len(vec) != expected_len:
                raise ValueError(
                    f"{field_name} requires {expected_len} floats, got {len(vec)}"
                )
            cleaned: list[float] = []
            for idx, v in enumerate(vec):
                if type(v) not in (int, float) or not math.isfinite(float(v)):
                    raise ValueError(f"{field_name}[{idx}] ({names[idx]}) must be finite")
                fv = float(v)
                if not (0.0 <= fv <= 1.0):
                    raise ValueError(
                        f"{field_name}[{idx}] ({names[idx]}) must be in [0, 1], got {fv}"
                    )
                cleaned.append(round(fv, 6))
            object.__setattr__(self, field_name, tuple(cleaned))

    @property
    def values(self) -> tuple[float, ...]:
        """Concatenation of the 4 disjoint departmental vectors (`17 + 16 + 16 + 3 = 52`)."""
        return (
            self.doer_vector
            + self.thinker_vector
            + self.feeler_vector
            + self.cadence_vector
        )

    def to_dict(self) -> dict[str, float]:
        vals = self.values
        return {
            name: vals[idx]
            for idx, name in enumerate(SUBLIMINAL_FEATURE_NAMES)
        }

    def to_tensor(
        self,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        return torch.tensor(self.values, device=device, dtype=dtype)

    def with_zeroed_cord(self) -> "SubliminalChordVector":
        """Return all-zero departmental vectors for the 'Unplug the Cord' diagnostic test."""
        return SubliminalChordVector(
            doer_vector=(0.0,) * DOER_DEPT_DIM,
            thinker_vector=(0.0,) * THINKER_DEPT_DIM,
            feeler_vector=(0.0,) * FEELER_DEPT_DIM,
            cadence_vector=(0.0,) * CADENCE_DEPT_DIM,
        )

    @property
    def social_drift_flag(self) -> float:
        return self.doer_vector[8]

    @property
    def branch_harmony(self) -> float:
        return self.cadence_vector[0]

    @property
    def dissonance_pressure(self) -> float:
        return self.cadence_vector[1]

    @property
    def cadence_rupture_drive(self) -> float:
        return self.cadence_vector[2]

    @classmethod
    def from_flat_values(cls, flat_values: tuple[float, ...] | list[float]) -> "SubliminalChordVector":
        if len(flat_values) != SUBLIMINAL_VECTOR_DIM:
            raise ValueError(
                f"Expected {SUBLIMINAL_VECTOR_DIM} flat values across the 4 departments, got {len(flat_values)}"
            )
        t = tuple(float(x) for x in flat_values)
        return cls(
            doer_vector=t[0:17],
            thinker_vector=t[17:33],
            feeler_vector=t[33:49],
            cadence_vector=t[49:52],
        )

    @classmethod
    def from_feature_mapping(cls, mapping: Mapping[str, float]) -> "SubliminalChordVector":
        vals = [
            _clamp01(float(mapping.get(name, 0.0)))
            for name in SUBLIMINAL_FEATURE_NAMES
        ]
        return cls.from_flat_values(vals)

    @classmethod
    def from_right_brain_vibe(
        cls,
        vibe: RightBrainVibe,
        *,
        deliberation: ThreeClassifierDeliberation | None = None,
        social_drift: SocialDriftReport | None = None,
        deadband: float = DEFAULT_RUPTURE_DEADBAND,
        ankle_rope_sensitivity: float = 1.0,
    ) -> "SubliminalChordVector":
        """Compile the 4 strictly isolated Departmental Vectors directly from a live `RightBrainVibe`."""
        coords = vibe.coordinates
        fc_notes = vibe.felt_chord.complex_notes if vibe.felt_chord is not None else {}

        # 1. Executive / Do-er Department Vector (17 floats):
        #    6 Cultural leaves + 2 Will leaves + 1 Social Drift flag + 8-float Cultural Child Chord
        cult_6 = tuple(_clamp01(float(coords.get(k, 0.0))) for k in CULTURAL_FOUNDING_ORDER)
        will_2 = (_clamp01(vibe.intention), _clamp01(vibe.execution))
        drift_flag = (1.0 if (social_drift is not None and social_drift.drift_detected) else 0.0,)
        cult_child_chord_8 = pool_departmental_child_chord(
            coords,
            department_prefix="cultural/",
            extra_notes=fc_notes,
        )
        doer_vec = cult_6 + will_2 + drift_flag + cult_child_chord_8

        # 2. Judicial / Thinker Department Vector (16 floats):
        #    6 Philosophical leaves + 2 Truth leaves + 8-float Philosophical Child Chord
        phil_6 = tuple(_clamp01(float(coords.get(k, 0.0))) for k in PHILOSOPHICAL_FOUNDING_ORDER)
        truth_2 = (_clamp01(vibe.theory), _clamp01(vibe.fact))
        phil_child_chord_8 = pool_departmental_child_chord(
            coords,
            department_prefix="philosophical/",
            extra_notes=fc_notes,
        )
        thinker_vec = phil_6 + truth_2 + phil_child_chord_8

        # 3. Legislative / Feeler Department Vector (16 floats):
        #    6 Experiential leaves + 2 Contemplation leaves + 8-float Experiential Child Chord
        exp_6 = (
            _clamp01(vibe.novelty),
            _clamp01(vibe.salience),
            _clamp01(vibe.valence),
            _clamp01(vibe.arousal),
            _clamp01(vibe.resonance),
            _clamp01(vibe.retention),
        )
        contemp_2 = (_clamp01(vibe.clarity), _clamp01(vibe.confidence))
        exp_child_chord_8 = pool_departmental_child_chord(
            coords,
            department_prefix="experiential/",
            extra_notes=fc_notes,
        )
        feeler_vec = exp_6 + contemp_2 + exp_child_chord_8

        # 4. Three-Legged Race Cadence & Ankle-Rope Department Vector (3 floats):
        harmony = (
            _clamp01(deliberation.harmony_score)
            if deliberation is not None
            else _clamp01(vibe.synthesis.harmony)
        )
        has_dissonance = bool(vibe.self_registry.dissonance_notes)
        conflict_count = len(vibe.synthesis.conflicts)
        dissonance_pressure = _clamp01(
            (0.65 if has_dissonance else 0.0)
            + 0.25 * min(2, conflict_count)
            + max(0.0, 0.50 - vibe.clarity)
        )
        discreteness = _clamp01(float(coords.get("philosophical/persistence/discreteness", 0.0)))
        feeler_child_peak = max(exp_child_chord_8, default=0.0)
        rupture_drive = compute_cadence_rupture_drive(
            arousal=vibe.arousal,
            discreteness=discreteness,
            clarity=vibe.clarity,
            valence=vibe.valence,
            feeler_child_peak=feeler_child_peak,
            dissonance_pressure=dissonance_pressure,
            deadband=deadband,
            ankle_rope_sensitivity=ankle_rope_sensitivity,
        )
        cadence_vec = (harmony, dissonance_pressure, rupture_drive)

        return cls(
            doer_vector=doer_vec,
            thinker_vector=thinker_vec,
            feeler_vector=feeler_vec,
            cadence_vector=cadence_vec,
        )


class SubliminalGapProjector(nn.Module):
    """Projects the 4 strictly disjoint Departmental Vectors into 4 virtual token embeddings `[4, hidden_size]`.

    Each virtual token is wired exclusively to its own department's disjoint vector with
    **zero shared or overlapping indices**:
      - Virtual Token 0 (`doer_head`): Executive / Do-er (`doer_vector`, 17 floats)
      - Virtual Token 1 (`thinker_head`): Judicial / Thinker (`thinker_vector`, 16 floats)
      - Virtual Token 2 (`feeler_head`): Legislative / Feeler (`feeler_vector`, 16 floats)
      - Virtual Token 3 (`cadence_head`): Three-Legged Race Cadence & Rupture (`cadence_vector`, 3 floats)
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        input_dim: int = SUBLIMINAL_VECTOR_DIM,
        num_virtual_tokens: int = NUM_VIRTUAL_GAP_TOKENS,
        intermediate_size: int | None = None,
    ) -> None:
        super().__init__()
        if type(hidden_size) is not int or hidden_size < 8:
            raise ValueError("hidden_size must be an int >= 8")
        if type(input_dim) is not int or input_dim != SUBLIMINAL_VECTOR_DIM:
            raise ValueError(f"input_dim must equal {SUBLIMINAL_VECTOR_DIM}")
        if type(num_virtual_tokens) is not int or num_virtual_tokens != NUM_VIRTUAL_GAP_TOKENS:
            raise ValueError(f"num_virtual_tokens must equal {NUM_VIRTUAL_GAP_TOKENS}")

        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.num_virtual_tokens = num_virtual_tokens
        mid = intermediate_size or max(64, hidden_size)

        dept_slices = (
            DOER_DEPARTMENT_INDICES,
            THINKER_DEPARTMENT_INDICES,
            FEELER_DEPARTMENT_INDICES,
            CADENCE_DEPARTMENT_INDICES,
        )
        self.department_indices = dept_slices
        self.department_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(len(indices), mid),
                    nn.SiLU(),
                    nn.LayerNorm(mid),
                    nn.Linear(mid, hidden_size),
                )
                for indices in dept_slices
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, chord_vectors: Tensor) -> Tensor:
        """Map `[..., 52]` disjoint departmental tensor -> `[..., 4, hidden_size]` virtual token embeddings."""
        if chord_vectors.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last dimension {self.input_dim}, got {tuple(chord_vectors.shape)}"
            )
        per_token: list[Tensor] = []
        for idx_tuple, head in zip(self.department_indices, self.department_heads):
            dept_slice = chord_vectors[..., list(idx_tuple)]
            per_token.append(head(dept_slice))
        stacked = torch.stack(per_token, dim=-2)  # [..., 4, hidden_size]
        return self.output_norm(stacked)


@dataclass(frozen=True)
class SplicedEmbeddingBatch:
    """Result of splicing non-verbal Fuzzy Tree embeddings into `<|subliminal_gap|>` token slots."""

    inputs_embeds: Tensor
    attention_mask: Tensor
    labels: Tensor | None
    gap_virtual_spans: tuple[tuple[tuple[int, int], ...], ...]


def splice_subliminal_gaps_into_embeds(
    input_ids: Tensor,
    word_embedding_fn: Callable[[Tensor], Tensor] | nn.Module,
    projector: SubliminalGapProjector,
    chord_vectors: Tensor,
    *,
    gap_token_id: int,
    attention_mask: Tensor | None = None,
    labels: Tensor | None = None,
) -> SplicedEmbeddingBatch:
    """Replace every `gap_token_id` in `input_ids` with 4 projected departmental Fuzzy Tree embeddings."""
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape [batch, seq_len], got {tuple(input_ids.shape)}")
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    if chord_vectors.ndim == 2:
        chord_vectors = chord_vectors.unsqueeze(1)  # [batch, 1, 52]
    if chord_vectors.ndim != 3 or chord_vectors.shape[0] != batch_size:
        raise ValueError(
            f"chord_vectors must have shape [batch, num_gaps, {SUBLIMINAL_VECTOR_DIM}] or [batch, {SUBLIMINAL_VECTOR_DIM}], got {tuple(chord_vectors.shape)}"
        )

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    if labels is not None and labels.shape != input_ids.shape:
        raise ValueError("labels must have the same shape as input_ids")

    safe_input_ids = input_ids.clone()
    is_gap_mask = safe_input_ids == int(gap_token_id)
    safe_input_ids[is_gap_mask] = 0

    base_embeds = word_embedding_fn(safe_input_ids)  # [batch, seq_len, hidden_size]
    hidden_size = base_embeds.shape[-1]
    projected_gaps = projector(chord_vectors.to(device=device, dtype=base_embeds.dtype))
    k_tokens = projected_gaps.shape[-2]

    spliced_embeds_list: list[Tensor] = []
    spliced_mask_list: list[Tensor] = []
    spliced_labels_list: list[Tensor] = []
    all_spans: list[tuple[tuple[int, int], ...]] = []

    for b in range(batch_size):
        row_embeds: list[Tensor] = []
        row_mask: list[Tensor] = []
        row_labels: list[Tensor] = []
        row_spans: list[tuple[int, int]] = []
        gap_counter = 0
        cur_pos = 0

        for t in range(seq_len):
            if bool(is_gap_mask[b, t].item()):
                if gap_counter >= projected_gaps.shape[1]:
                    raise ValueError(
                        f"Sequence {b} has more <|subliminal_gap|> tokens than provided chord_vectors ({projected_gaps.shape[1]})"
                    )
                virt_emb = projected_gaps[b, gap_counter]  # [4, hidden_size]
                row_embeds.append(virt_emb)
                mask_val = attention_mask[b, t].unsqueeze(0).expand(k_tokens)
                row_mask.append(mask_val)
                if labels is not None:
                    ignore_chunk = torch.full(
                        (k_tokens,),
                        IGNORE_LABEL_INDEX,
                        dtype=labels.dtype,
                        device=device,
                    )
                    row_labels.append(ignore_chunk)
                row_spans.append((cur_pos, cur_pos + k_tokens))
                cur_pos += k_tokens
                gap_counter += 1
            else:
                row_embeds.append(base_embeds[b, t : t + 1])
                row_mask.append(attention_mask[b, t : t + 1])
                if labels is not None:
                    row_labels.append(labels[b, t : t + 1])
                cur_pos += 1

        spliced_embeds_list.append(torch.cat(row_embeds, dim=0))
        spliced_mask_list.append(torch.cat(row_mask, dim=0))
        if labels is not None:
            spliced_labels_list.append(torch.cat(row_labels, dim=0))
        all_spans.append(tuple(row_spans))

    max_len = max(x.shape[0] for x in spliced_embeds_list)
    padded_embeds = torch.zeros(
        (batch_size, max_len, hidden_size),
        dtype=base_embeds.dtype,
        device=device,
    )
    padded_mask = torch.zeros(
        (batch_size, max_len),
        dtype=attention_mask.dtype,
        device=device,
    )
    padded_labels = (
        torch.full(
            (batch_size, max_len),
            IGNORE_LABEL_INDEX,
            dtype=labels.dtype,
            device=device,
        )
        if labels is not None
        else None
    )

    for b in range(batch_size):
        length = spliced_embeds_list[b].shape[0]
        padded_embeds[b, :length] = spliced_embeds_list[b]
        padded_mask[b, :length] = spliced_mask_list[b]
        if padded_labels is not None:
            padded_labels[b, :length] = spliced_labels_list[b]

    return SplicedEmbeddingBatch(
        inputs_embeds=padded_embeds,
        attention_mask=padded_mask,
        labels=padded_labels,
        gap_virtual_spans=tuple(all_spans),
    )
