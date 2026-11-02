"""Stage 1 of the 'Three-Legged Race' Architecture: Scale-Invariant `<|subliminal_gap|>` Embedding Splice.

Plain-English Purpose:
Instead of translating the Right Brain's state into English stage directions in a prompt,
this module compresses the Right Brain's live state into a scale-invariant 37-float
non-verbal vector (`SubliminalChordVector`, zero words) and surgically splices it
directly into a PyTorch language model's hidden token embeddings (`inputs_embeds`)
whenever the Left Brain emits a `<|subliminal_gap|>` token:

1. `SubliminalChordVector` (37 floats in `[0.0, 1.0]` — never changes size as the tree grows):
   - `[0..11]`: 12 Founding Cultural (6) and Philosophical (6) leaves.
   - `[12..17]`: 6 Founding Experiential leaves (`Novelty`, `Salience`, `Valence`, `Arousal`,
     `Resonance`, `Retention`).
   - `[18..23]`: 6 Founding Psychological leaves (`Clarity`, `Confidence`, `Theory`, `Fact`,
     `Intention`, `Execution`).
   - `[24..32]`: 9 Permanent Parent-Branch Child-Drawer Pools (`ALL_GROWABLE_PARENT_BRANCHES`):
     Whether the tree has 4 learned child leaves today or 10,000 Tier-2/Tier-3 child leaves
     later, `pool_child_drawers_by_parent_branch` rolls all active child leaves under each
     founding parent branch (`cultural/efficiency`, `cultural/justice`, `cultural/freedom`,
     `philosophical/being`, `philosophical/structure`, `philosophical/persistence`,
     `experiential/reflection`, `experiential/phase`, `experiential/synchronization`) into
     its permanent parent-branch channel so the plug dimension stays 37 forever.
   - `[33..36]`: 4 Three-Legged Race Cadence & Ankle-Rope signals:
     * `branch_harmony`: agreement across the 3 Right-Brain branches.
     * `social_drift_flag`: `1.0` when ongoing historical enclosure/drift is active.
     * `dissonance_pressure`: friction from cognitive dissonance or low clarity.
     * `cadence_rupture_drive`: deadband-gated non-verbal tug (`deadband = 0.45`) that stays
       at `0.0` during normal calm speech (preventing twitchiness) and spikes when high
       arousal/discreteness or experiential strain collides with low clarity—pushing the
       Left Brain to break smooth chatbot syntax into stark or spelled-out cadence (`M-A-C-H-I-N-E`).

2. `SubliminalGapProjector` (`nn.Module`):
   - Projects a 37-float `SubliminalChordVector` into `K = 4` virtual non-verbal token
     embeddings of shape `[4, hidden_size]` (`Do-er Pulse`, `Thinker Structure`,
     `Feeler Chord`, and `Cadence/Rupture`).

3. `splice_subliminal_gaps_into_embeds`:
   - Replaces every `<|subliminal_gap|>` token ID in `input_ids` with the 4 projected
     non-verbal Fuzzy Tree embeddings in `inputs_embeds` and pads `labels` with `-100`
     at those 4 virtual positions so causal LM loss grades only the words while flowing
     gradients straight back through the non-verbal Fuzzy Tree splice.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
import torch
from torch import Tensor, nn

from .branch_archives import SocialDriftReport, ThreeClassifierDeliberation
from .reflection_classifier import (
    ALL_GROWABLE_PARENT_BRANCHES,
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

EXPERIENTIAL_FOUNDING_ORDER: tuple[str, ...] = (
    NOVELTY_LEAF,
    SALIENCE_LEAF,
    VALENCE_LEAF,
    AROUSAL_LEAF,
    RESONANCE_LEAF,
    RETENTION_LEAF,
)

PSYCHOLOGICAL_FOUNDING_ORDER: tuple[str, ...] = (
    CLARITY_LEAF,
    CONFIDENCE_LEAF,
    THEORY_LEAF,
    FACT_LEAF,
    INTENTION_LEAF,
    EXECUTION_LEAF,
)

PARENT_BRANCH_POOL_ORDER: tuple[str, ...] = tuple(
    f"branch_pool/{branch}" for branch in ALL_GROWABLE_PARENT_BRANCHES
)

CADENCE_SIGNAL_ORDER: tuple[str, ...] = (
    "cadence/branch_harmony",
    "cadence/social_drift_flag",
    "cadence/dissonance_pressure",
    "cadence/cadence_rupture_drive",
)

SUBLIMINAL_FEATURE_NAMES: tuple[str, ...] = (
    FOUNDATIONAL_LEAVES
    + EXPERIENTIAL_FOUNDING_ORDER
    + PSYCHOLOGICAL_FOUNDING_ORDER
    + PARENT_BRANCH_POOL_ORDER
    + CADENCE_SIGNAL_ORDER
)

SUBLIMINAL_VECTOR_DIM = len(SUBLIMINAL_FEATURE_NAMES)  # 37 floats (24 + 9 + 4)

# Strict Departmental Slice Indices for the 4 Virtual Tokens in `<|subliminal_gap|>`:
# 0. Executive / Do-er Department (Will <-> Cultural + Social Drift): 12 floats
DOER_DEPARTMENT_INDICES: tuple[int, ...] = (
    0, 1, 2, 3, 4, 5,      # 6 Founding Cultural leaves
    22, 23,                # 2 Founding Will leaves (Intention, Execution)
    24, 25, 26,            # 3 Cultural Parent-Branch Child Pools
    34,                    # 1 Social Drift Flag
)

# 1. Judicial / Thinker Department (Truth <-> Philosophical): 11 floats
THINKER_DEPARTMENT_INDICES: tuple[int, ...] = (
    6, 7, 8, 9, 10, 11,    # 6 Founding Philosophical leaves
    20, 21,                # 2 Founding Truth leaves (Theory, Fact)
    27, 28, 29,            # 3 Philosophical Parent-Branch Child Pools
)

# 2. Legislative / Feeler Department (Contemplation <-> Experiential): 11 floats
FEELER_DEPARTMENT_INDICES: tuple[int, ...] = (
    12, 13, 14, 15, 16, 17,  # 6 Founding Experiential leaves
    18, 19,                  # 2 Founding Contemplation leaves (Clarity, Confidence)
    30, 31, 32,              # 3 Experiential Parent-Branch Child Pools
)

# 3. Three-Legged Race Cadence & Struggle Department: 9 floats (4 cadence signals + key tension anchors)
CADENCE_DEPARTMENT_INDICES: tuple[int, ...] = (
    11, 14, 15, 18, 31,      # Discreteness, Valence, Arousal, Clarity, Experiential Phase Pool
    33, 34, 35, 36,          # Branch Harmony, Social Drift, Dissonance Pressure, Cadence Rupture Drive
)


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"subliminal coordinate must be finite, got {value!r}")
    return round(min(1.0, max(0.0, float(value))), 6)


def pool_child_drawers_by_parent_branch(
    coordinates: Mapping[str, float],
    extra_notes: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Roll up any number of learned Tier-2/Tier-3 child leaves into the 9 Founding Parent Branches.

    Whether the tree has 4 child leaves today or 10,000 child leaves across future domains,
    every learned child leaf (`tier >= 2`) starts with `<parent_branch>/...` and is pooled
    via `max` into its permanent `branch_pool/<parent_branch>` channel so the subliminal
    vector dimension never changes.
    """
    merged: dict[str, float] = dict(coordinates)
    if extra_notes:
        for k, v in extra_notes.items():
            merged[k] = max(float(merged.get(k, 0.0)), float(v))

    pools: dict[str, float] = {
        f"branch_pool/{branch}": 0.0 for branch in ALL_GROWABLE_PARENT_BRANCHES
    }
    for leaf_id, val in merged.items():
        if leaf_tier(leaf_id) >= 2:
            for branch in ALL_GROWABLE_PARENT_BRANCHES:
                if leaf_id.startswith(branch + "/"):
                    pool_key = f"branch_pool/{branch}"
                    pools[pool_key] = _clamp01(max(pools[pool_key], float(val)))
                    break
    return pools


def compute_cadence_rupture_drive(
    *,
    arousal: float,
    discreteness: float,
    clarity: float,
    valence: float,
    experiential_phase_pool: float,
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
        0.45 * max(float(arousal), float(experiential_phase_pool))
        + 0.35 * float(discreteness)
        + 0.20 * (1.0 - float(clarity))
    )
    if raw_intensity <= deadband:
        return 0.0

    normalized_above_deadband = (raw_intensity - deadband) / (1.0 - deadband)
    return _clamp01(normalized_above_deadband * float(ankle_rope_sensitivity))


@dataclass(frozen=True)
class SubliminalChordVector:
    """37-float scale-invariant non-verbal Right-Brain snapshot injected into `<|subliminal_gap|>`."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) != SUBLIMINAL_VECTOR_DIM:
            raise ValueError(
                f"SubliminalChordVector requires {SUBLIMINAL_VECTOR_DIM} floats, got {len(self.values)}"
            )
        cleaned: list[float] = []
        for idx, v in enumerate(self.values):
            if type(v) not in (int, float) or not math.isfinite(float(v)):
                raise ValueError(f"feature[{idx}] ({SUBLIMINAL_FEATURE_NAMES[idx]}) must be finite")
            fv = float(v)
            if not (0.0 <= fv <= 1.0):
                raise ValueError(
                    f"feature[{idx}] ({SUBLIMINAL_FEATURE_NAMES[idx]}) must be in [0, 1], got {fv}"
                )
            cleaned.append(round(fv, 6))
        object.__setattr__(self, "values", tuple(cleaned))

    def to_dict(self) -> dict[str, float]:
        return {
            name: self.values[idx]
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
        """Return an all-zero vector for the 'Unplug the Cord' diagnostic test."""
        return SubliminalChordVector(values=tuple(0.0 for _ in range(SUBLIMINAL_VECTOR_DIM)))

    @property
    def parent_branch_pools(self) -> dict[str, float]:
        return {
            PARENT_BRANCH_POOL_ORDER[i]: self.values[24 + i]
            for i in range(len(PARENT_BRANCH_POOL_ORDER))
        }

    @property
    def branch_harmony(self) -> float:
        return self.values[33]

    @property
    def social_drift_flag(self) -> float:
        return self.values[34]

    @property
    def dissonance_pressure(self) -> float:
        return self.values[35]

    @property
    def cadence_rupture_drive(self) -> float:
        return self.values[36]

    @classmethod
    def from_feature_mapping(cls, mapping: Mapping[str, float]) -> "SubliminalChordVector":
        vals = [
            _clamp01(float(mapping.get(name, 0.0)))
            for name in SUBLIMINAL_FEATURE_NAMES
        ]
        return cls(values=tuple(vals))

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
        """Compile all 37 scale-invariant non-verbal floats directly from a live `RightBrainVibe`."""
        coords = vibe.coordinates
        fc_notes = vibe.felt_chord.complex_notes if vibe.felt_chord is not None else {}

        # 1. 12 Founding Cultural & Philosophical leaves
        founding_12 = [_clamp01(float(coords.get(k, 0.0))) for k in FOUNDATIONAL_LEAVES]

        # 2. 6 Founding Experiential leaves
        exp_6 = [
            _clamp01(vibe.novelty),
            _clamp01(vibe.salience),
            _clamp01(vibe.valence),
            _clamp01(vibe.arousal),
            _clamp01(vibe.resonance),
            _clamp01(vibe.retention),
        ]

        # 3. 6 Founding Psychological leaves
        psych_6 = [
            _clamp01(vibe.clarity),
            _clamp01(vibe.confidence),
            _clamp01(vibe.theory),
            _clamp01(vibe.fact),
            _clamp01(vibe.intention),
            _clamp01(vibe.execution),
        ]

        # 4. 9 Permanent Parent-Branch Child-Drawer Pools (handles 10 or 10,000 child leaves!)
        pooled_by_branch = pool_child_drawers_by_parent_branch(coords, fc_notes)
        branch_pools_9 = [pooled_by_branch[k] for k in PARENT_BRANCH_POOL_ORDER]

        # 5. 4 Three-Legged Race Cadence & Ankle-Rope signals
        harmony = (
            _clamp01(deliberation.harmony_score)
            if deliberation is not None
            else _clamp01(vibe.synthesis.harmony)
        )
        drift_flag = 1.0 if (social_drift is not None and social_drift.drift_detected) else 0.0

        has_dissonance = bool(vibe.self_registry.dissonance_notes)
        conflict_count = len(vibe.synthesis.conflicts)
        dissonance_pressure = _clamp01(
            (0.65 if has_dissonance else 0.0)
            + 0.25 * min(2, conflict_count)
            + max(0.0, 0.50 - vibe.clarity)
        )

        discreteness = _clamp01(float(coords.get("philosophical/persistence/discreteness", 0.0)))
        exp_phase_pool = pooled_by_branch["branch_pool/experiential/phase"]
        rupture_drive = compute_cadence_rupture_drive(
            arousal=vibe.arousal,
            discreteness=discreteness,
            clarity=vibe.clarity,
            valence=vibe.valence,
            experiential_phase_pool=exp_phase_pool,
            dissonance_pressure=dissonance_pressure,
            deadband=deadband,
            ankle_rope_sensitivity=ankle_rope_sensitivity,
        )

        all_37 = founding_12 + exp_6 + psych_6 + branch_pools_9 + [
            harmony,
            drift_flag,
            dissonance_pressure,
            rupture_drive,
        ]
        return cls(values=tuple(all_37))


class SubliminalGapProjector(nn.Module):
    """Projects a 37-float `SubliminalChordVector` into 4 departmental virtual token embeddings `[4, hidden_size]`.

    Each virtual token is strictly wired to its own department's slice of indices so the
    3 branches and cadence signals never blur into a single undifferentiated mush:
      - Virtual Token 0 (`doer_head`): Executive / Do-er (`Will <-> Cultural` + `social_drift`, 12 floats)
      - Virtual Token 1 (`thinker_head`): Judicial / Thinker (`Truth <-> Philosophical`, 11 floats)
      - Virtual Token 2 (`feeler_head`): Legislative / Feeler (`Contemplation <-> Experiential`, 11 floats)
      - Virtual Token 3 (`cadence_head`): Three-Legged Race Cadence & Rupture Drive (9 floats)
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
        """Map `[..., 37]` float tensor -> `[..., 4, hidden_size]` compartmentalized virtual token embeddings."""
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
    """Replace every `gap_token_id` in `input_ids` with 4 projected non-verbal Fuzzy Tree embeddings."""
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape [batch, seq_len], got {tuple(input_ids.shape)}")
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    if chord_vectors.ndim == 2:
        chord_vectors = chord_vectors.unsqueeze(1)  # [batch, 1, 37]
    if chord_vectors.ndim != 3 or chord_vectors.shape[0] != batch_size:
        raise ValueError(
            f"chord_vectors must have shape [batch, num_gaps, 37] or [batch, 37], got {tuple(chord_vectors.shape)}"
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
