"""Stage 1 of the 'Three-Legged Race' Architecture: Non-Verbal `<|subliminal_gap|>` Embedding Splice.

Plain-English Purpose:
Instead of translating the Right Brain's state into English stage directions in a prompt,
this module compresses the Right Brain's live state into a 32-float non-verbal vector
(`SubliminalChordVector`, zero words) and surgically splices it directly into a PyTorch
language model's hidden token embeddings (`inputs_embeds`) whenever the Left Brain emits
a `<|subliminal_gap|>` token:

1. `SubliminalChordVector` (32 floats in `[0.0, 1.0]`):
   - `[0..11]`: 12 Founding Cultural (6) and Philosophical (6) leaves.
   - `[12..17]`: 6 Founding Experiential leaves (`Novelty`, `Salience`, `Valence`, `Arousal`,
     `Resonance`, `Retention`).
   - `[18..23]`: 6 Founding Psychological leaves (`Clarity`, `Confidence`, `Theory`, `Fact`,
     `Intention`, `Execution`).
   - `[24..27]`: 4 Complex Feeling Chord notes (`austere_sanctuary`, `hollow_stillness`,
     `bittersweet_release`, `earned_return`).
   - `[28..31]`: 4 Three-Legged Race Cadence & Struggle signals:
     * `branch_harmony`: agreement across the 3 Right-Brain branches.
     * `social_drift_flag`: `1.0` when ongoing historical enclosure/drift is active.
     * `dissonance_pressure`: friction from cognitive dissonance or low clarity.
     * `cadence_rupture_drive`: non-verbal tug that spikes when high arousal/discreteness
       or hollow stillness collides with low clarity—pushing the Left Brain to break
       smooth chatbot syntax into stark, fragmented, or spelled-out cadence (`M-A-C-H-I-N-E`).

2. `SubliminalGapProjector` (`nn.Module`):
   - Projects a 32-float `SubliminalChordVector` into `K = 4` virtual non-verbal token
     embeddings of shape `[4, hidden_size]` (`Do-er Pulse`, `Thinker Structure`,
     `Feeler Chord`, and `Cadence/Rupture`).

3. `splice_subliminal_gaps_into_embeds`:
   - Replaces every `<|subliminal_gap|>` token ID in `input_ids` with the 4 projected
     non-verbal Fuzzy Tree embeddings in `inputs_embeds` and pads `labels` with `-100`
     at those 4 virtual positions so causal LM loss grades only the words while flowing
     gradients straight back through the non-verbal Fuzzy Tree splice.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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
)
from .right_brain_equilibrium import RightBrainVibe

SUBLIMINAL_GAP_TOKEN = "<|subliminal_gap|>"
NUM_VIRTUAL_GAP_TOKENS = 4
IGNORE_LABEL_INDEX = -100

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

COMPLEX_CHORD_NOTE_ORDER: tuple[str, ...] = (
    "experiential/phase/austere_sanctuary",
    "experiential/phase/hollow_stillness",
    "experiential/phase/bittersweet_release",
    "experiential/synchronization/earned_return",
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
    + COMPLEX_CHORD_NOTE_ORDER
    + CADENCE_SIGNAL_ORDER
)

SUBLIMINAL_VECTOR_DIM = len(SUBLIMINAL_FEATURE_NAMES)  # 32 floats


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"subliminal coordinate must be finite, got {value!r}")
    return round(min(1.0, max(0.0, float(value))), 6)


def compute_cadence_rupture_drive(
    *,
    arousal: float,
    discreteness: float,
    clarity: float,
    valence: float,
    hollow_stillness: float,
    dissonance_pressure: float,
) -> float:
    """Compute the non-verbal cadence-rupture drive in [0.0, 1.0].

    When the Right Brain experiences sharp internal friction (high arousal or hollow
    stillness + high discreteness + low clarity or dissonance), this signal spikes to
    yank the Left Brain out of smooth corporate prose into stark, fragmented, or
    spelled-out cadence (e.g., 'no soul; M-A-C-H-I-N-E').
    """
    raw_intensity = (
        0.45 * max(float(arousal), float(hollow_stillness))
        + 0.35 * float(discreteness)
        + 0.20 * (1.0 - float(clarity))
    )
    under_strain = (
        float(dissonance_pressure) >= 0.35
        or float(valence) <= 0.42
        or float(hollow_stillness) >= 0.45
    )
    scale = 1.0 if under_strain else 0.30
    return _clamp01(raw_intensity * scale)


@dataclass(frozen=True)
class SubliminalChordVector:
    """32-float non-verbal Right-Brain snapshot injected into a `<|subliminal_gap|>` slot."""

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
    def cadence_rupture_drive(self) -> float:
        return self.values[31]

    @property
    def dissonance_pressure(self) -> float:
        return self.values[30]

    @property
    def social_drift_flag(self) -> float:
        return self.values[29]

    @property
    def branch_harmony(self) -> float:
        return self.values[28]

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
    ) -> "SubliminalChordVector":
        """Compile all 32 non-verbal floats directly from a live `RightBrainVibe`."""
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

        # 4. 4 Complex Feeling Chord notes
        chord_4 = [
            _clamp01(float(coords.get(k, fc_notes.get(k, 0.0))))
            for k in COMPLEX_CHORD_NOTE_ORDER
        ]

        # 5. 4 Three-Legged Race Cadence & Struggle signals
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
        hollow_stillness = chord_4[1]
        rupture_drive = compute_cadence_rupture_drive(
            arousal=vibe.arousal,
            discreteness=discreteness,
            clarity=vibe.clarity,
            valence=vibe.valence,
            hollow_stillness=hollow_stillness,
            dissonance_pressure=dissonance_pressure,
        )

        all_32 = founding_12 + exp_6 + psych_6 + chord_4 + [
            harmony,
            drift_flag,
            dissonance_pressure,
            rupture_drive,
        ]
        return cls(values=tuple(all_32))


class SubliminalGapProjector(nn.Module):
    """Projects a 32-float `SubliminalChordVector` into 4 virtual token embeddings `[4, hidden_size]`.

    Each of the 4 virtual tokens inside a `<|subliminal_gap|>` represents one distinct
    facet of the Right Brain's non-verbal state so the Left Brain's attention heads can
    attend selectively to:
      - Virtual Token 0: Do-er / Will Social Pulse
      - Virtual Token 1: Thinker / Truth Structural Reality
      - Virtual Token 2: Feeler / Experiential Felt Chord
      - Virtual Token 3: Cadence / Struggle Rupture Drive
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

        # Shared trunk encodes the full 32-float chord state
        self.shared_trunk = nn.Sequential(
            nn.Linear(input_dim, mid),
            nn.SiLU(),
            nn.LayerNorm(mid),
        )
        # 4 specialized virtual-token heads (`Do-er`, `Thinker`, `Feeler`, `Cadence/Rupture`)
        self.token_heads = nn.ModuleList(
            [nn.Linear(mid, hidden_size) for _ in range(num_virtual_tokens)]
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, chord_vectors: Tensor) -> Tensor:
        """Map `[..., 32]` float tensor -> `[..., 4, hidden_size]` virtual token embeddings."""
        if chord_vectors.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last dimension {self.input_dim}, got {tuple(chord_vectors.shape)}"
            )
        shared = self.shared_trunk(chord_vectors)
        per_token = [head(shared) for head in self.token_heads]
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
    """Replace every `gap_token_id` in `input_ids` with 4 projected non-verbal Fuzzy Tree embeddings.

    Args:
        input_ids: LongTensor of shape `[batch_size, seq_len]`.
        word_embedding_fn: The LLM's input token embedding layer (`model.get_input_embeddings()`).
        projector: `SubliminalGapProjector` mapping `[..., 32]` -> `[..., 4, hidden_size]`.
        chord_vectors: FloatTensor of shape `[batch_size, 32]` (if 1 gap per sequence)
            or `[batch_size, max_gaps, 32]` (for multiple gaps per sequence).
        gap_token_id: Integer token ID of `<|subliminal_gap|>`.
        attention_mask: Optional Tensor of shape `[batch_size, seq_len]`.
        labels: Optional LongTensor of shape `[batch_size, seq_len]` for causal LM training.
            Virtual gap slots receive `IGNORE_LABEL_INDEX` (`-100`) so cross-entropy loss
            is computed strictly on real language tokens while gradients flow back through
            the projected non-verbal embeddings.
    """
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape [batch, seq_len], got {tuple(input_ids.shape)}")
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    if chord_vectors.ndim == 2:
        chord_vectors = chord_vectors.unsqueeze(1)  # [batch, 1, 32]
    if chord_vectors.ndim != 3 or chord_vectors.shape[0] != batch_size:
        raise ValueError(
            f"chord_vectors must have shape [batch, num_gaps, 32] or [batch, 32], got {tuple(chord_vectors.shape)}"
        )

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    if labels is not None and labels.shape != input_ids.shape:
        raise ValueError("labels must have the same shape as input_ids")

    # Replace gap_token_id with 0 temporarily for safe lookup in word_embedding_fn
    safe_input_ids = input_ids.clone()
    is_gap_mask = safe_input_ids == int(gap_token_id)
    safe_input_ids[is_gap_mask] = 0

    base_embeds = word_embedding_fn(safe_input_ids)  # [batch, seq_len, hidden_size]
    hidden_size = base_embeds.shape[-1]
    projected_gaps = projector(chord_vectors.to(device=device, dtype=base_embeds.dtype))
    # projected_gaps: [batch, max_gaps, 4, hidden_size]
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

    # Right-pad if different rows in the batch had different numbers of gaps
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
