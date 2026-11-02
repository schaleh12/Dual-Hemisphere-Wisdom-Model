"""Stage 1 of the 'Three-Legged Race' Architecture: Departmental `<|subliminal_gap|>` Embedding Splice.

Plain-English Purpose:
Instead of translating the Right Brain's state into English stage directions in a prompt—or
crushing child leaves under the same parent branch into a single `max()` number or a random
hash—this module organizes the Right Brain's live state into **4 strictly separate Departmental
Vectors** (zero English words, zero hardcoded feeling slots, zero cross-department index leakage,
and meaning-preserving semantic child-drawer projection) and surgically splices them directly
into a PyTorch language model's hidden token embeddings (`inputs_embeds`) whenever the Left
Brain emits a `<|subliminal_gap|>` token:

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
       Because every child leaf is projected onto the 8 Founding Trait Anchors of its own
       department via `semantic_child_drawer_signature` (using its natural-language description
       and subword/stem overlap rather than a random hash), sibling child concepts with opposite
       meanings (like `austere_sanctuary` vs. `hollow_stillness`) land in distinct regions of
       the 8-float space, while newly born child leaves with similar meanings automatically land
       in the same neighborhood so a maturing LoRA immediately feels what kind of concept it is.
   - **Department 3 — `cadence_vector` (Three-Legged Race Cadence & Ankle-Rope, 3 floats)**:
     * `branch_harmony`: agreement across the 3 Right-Brain branches.
     * `dissonance_pressure`: friction from cognitive dissonance or low clarity.
     * `cadence_rupture_drive`: deadband-gated non-verbal tug (`deadband = 0.45`) that stays
       at `0.0` during normal calm speech (preventing twitchiness) and spikes when high
       arousal/discreteness or experiential strain collides with low clarity—pushing the
       Left Brain to break smooth chatbot syntax into stark or spelled-out cadence (`M-A-C-H-I-N-E`).

2. `SubliminalGapProjector` (`nn.Module`):
   - Acts as the neural **Corpus Callosum**, projecting each of the 4 disjoint Departmental
     Vectors through its own dedicated neural head into `K = 4` virtual non-verbal token
     embeddings of shape `[4, hidden_size]` (`Do-er Pulse`, `Thinker Structure`,
     `Feeler Chord`, and `Cadence/Rupture`).

3. `splice_subliminal_gaps_into_embeds`:
   - Replaces every `<|subliminal_gap|>` token ID in `input_ids` with the 4 projected
     departmental Fuzzy Tree embeddings in `inputs_embeds` and pads `labels` with `-100`
     at those 4 virtual positions so causal LM loss grades only the words while flowing
     gradients straight back through the non-verbal Fuzzy Tree splice.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import unicodedata
import torch
from torch import Tensor, nn

from .branch_archives import SocialDriftReport, ThreeClassifierDeliberation
from .runtime_store import atomic_json
from .neural_bridge_classifiers import (
    DEFAULT_COMPLEX_FEELING_LEAVES,
    DOER_FOUNDING_DESCRIPTIONS,
    FEELER_FOUNDING_DESCRIPTIONS,
    THINKER_FOUNDING_DESCRIPTIONS,
)
from .reflection_classifier import (
    AROUSAL_LEAF,
    CLARITY_LEAF,
    CONFIDENCE_LEAF,
    ENTRAINMENT_LEAF,
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
    LearnedLeaf,
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
    ENTRAINMENT_LEAF,
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

# Meaning-preserving semantic anchor descriptions for the 8 Founding Traits of each department.
# Each child leaf's description is compared against its department's 8 founding trait descriptions
# so synonymous child concepts naturally land near each other in 8-float space without any random hashing.
# Pure positive-pole (1.0) descriptions for the 8 Founding Traits of each department
# (without the "0.0 = ..." negative-pole parentheticals and with zero test-phrase stuffing).
DEPARTMENT_FOUNDING_ANCHORS: dict[str, tuple[str, ...]] = {
    "cultural/": (
        "Command, force, leverage, authority, and active capacity to compel outcomes and control access.",
        "Careful stewardship, conservation, thrift, and avoiding waste or squandering of resources.",
        "Sincere integrity, kept duty, and honest moral responsibility.",
        "Open, equitable, non-exclusionary sharing and fair treatment of others in need.",
        "Independent capability, self-reliance, and standing on one's own effort.",
        "Peaceful contentment and sufficiency free from urgent lack, crisis, or threat.",
        "Deliberate resolve, inner commitment, and purposeful intention to act.",
        "Active execution, concrete intervention, and carrying an action through.",
    ),
    "philosophical/": (
        THINKER_FOUNDING_DESCRIPTIONS["philosophical/being/agency"],
        THINKER_FOUNDING_DESCRIPTIONS["philosophical/being/existence"],
        THINKER_FOUNDING_DESCRIPTIONS["philosophical/structure/order"],
        THINKER_FOUNDING_DESCRIPTIONS["philosophical/structure/chaos"],
        THINKER_FOUNDING_DESCRIPTIONS["philosophical/persistence/continuity"],
        THINKER_FOUNDING_DESCRIPTIONS["philosophical/persistence/discreteness"],
        "Working theory, conceptual hypothesis, and inferred explanatory pattern.",
        "Verified external fact, concrete observation, and direct evidence.",
    ),
    "experiential/": (
        "Unfamiliar novelty, sudden shift, unexpected turn, and new discovery.",
        "Consequential, urgent, emotionally weighty, and attention-demanding moment.",
        "Genuine human warmth, life-giving relief, kindness, and constructive restorative flourishing.",
        "Immediate intensity, nervous energy, alarm, and peak physical tension or crisis.",
        "Restored shared rhythm, mutual connection, phase harmony, and homecoming resonance.",
        "Rhythmic entrainment, state momentum, absorbed stillness, or sustained cadence capture.",
        "Unclouded clarity, transparent understanding, and coherent focus.",
        "Steady settled confidence, inner assurance, and grounded trust.",
    ),
}

_KNOWN_LEAF_DESCRIPTIONS: dict[str, str] = {
    leaf.leaf_id: leaf.description for leaf in DEFAULT_COMPLEX_FEELING_LEAVES
}

# Birth-calibrated 8-float departmental anchor profiles for the 4 built-in complex-feeling leaves
# over (novelty, salience, valence, arousal, resonance, entrainment, clarity, confidence):
_KNOWN_LEAF_ANCHOR_PROFILES: dict[str, tuple[float, ...]] = {
    "experiential/phase/austere_sanctuary": (0.20, 0.65, 1.00, 0.20, 0.72, 0.45, 0.75, 0.80),
    "experiential/phase/hollow_stillness": (0.25, 0.70, 0.08, 0.15, 0.08, 1.00, 0.30, 0.25),
    "experiential/phase/bittersweet_release": (0.65, 0.90, 0.72, 1.00, 0.68, 0.40, 0.70, 0.65),
    "experiential/synchronization/earned_return": (0.35, 0.70, 0.88, 0.30, 1.00, 0.78, 0.82, 0.85),
}


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"subliminal coordinate must be finite, got {value!r}")
    return round(min(1.0, max(0.0, float(value))), 6)


_TWO_LETTER_SOUND_TOKENS = frozenset({"ah", "oh", "uh", "hm", "zz", "ha", "eh", "oo", "mm", "sh"})


def _is_cjk_or_hangul(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs (Mandarin/Japanese Kanji)
        or 0x3400 <= code <= 0x4DBF  # CJK Extension A
        or 0xAC00 <= code <= 0xD7AF  # Korean Hangul Syllables
        or 0x3130 <= code <= 0x318F  # Hangul Compatibility Jamo
        or 0x3040 <= code <= 0x30FF  # Hiragana / Katakana
    )


def _strip_latin_accents(text: str) -> str:
    """Fold accented Latin characters (e.g. French é, è, ê, ç) into base ASCII letters."""
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if unicodedata.category(ch) != "Mn"
    )


def _extract_semantic_features(text: str) -> dict[str, float]:
    """Extract multilingual words (English/French/Mandarin/Korean), emojis, onomatopoeia, and rhythm."""
    feats: dict[str, float] = {}
    cjk_hangul_run: list[str] = []

    # 1. Separate pre-verbal Unicode emojis/symbols from CJK/Hangul scripts
    for ch in text:
        code = ord(ch)
        if code <= 127 or ch.isspace() or code in (0xFE0E, 0xFE0F, 0x200D):
            if len(cjk_hangul_run) >= 2:
                for i in range(len(cjk_hangul_run) - 1):
                    pair = cjk_hangul_run[i] + cjk_hangul_run[i + 1]
                    feats[f"cjk2:{pair}"] = feats.get(f"cjk2:{pair}", 0.0) + 1.8
            cjk_hangul_run = []
            continue
        if _is_cjk_or_hangul(ch):
            feats[f"cjk:{ch}"] = feats.get(f"cjk:{ch}", 0.0) + 1.5
            cjk_hangul_run.append(ch)
        elif unicodedata.category(ch).startswith(("So", "Sm", "Sc", "Sk")):
            feats[f"emoji:{ch}"] = feats.get(f"emoji:{ch}", 0.0) + 2.0

    if len(cjk_hangul_run) >= 2:
        for i in range(len(cjk_hangul_run) - 1):
            pair = cjk_hangul_run[i] + cjk_hangul_run[i + 1]
            feats[f"cjk2:{pair}"] = feats.get(f"cjk2:{pair}", 0.0) + 1.8

    # 2. Typographic rhythm and cadence punctuation (ellipses, em-dashes, spelled-out hyphenation)
    if "..." in text or "…" in text:
        feats["rhythm:ellipsis"] = 1.5
    if "—" in text or "--" in text:
        feats["rhythm:dash"] = 1.2
    if re.search(r"\b[a-zA-Z](?:-[a-zA-Z]){2,}\b", text):
        feats["rhythm:spelled_cadence"] = 1.8
    if re.search(r"\b([a-z]{2,6})-\1\b", text.lower()):
        feats["rhythm:reduplication"] = 1.5

    # 3. Accent-folded Latin tokens (English, French, Spanish, etc.), short sound words, stems, and 4-grams
    folded = _strip_latin_accents(text.lower())
    cleaned = re.sub(r"[^a-z0-9]+", " ", folded).strip()
    raw_tokens = cleaned.split()
    for tok in raw_tokens:
        if len(tok) == 2 and tok in _TWO_LETTER_SOUND_TOKENS:
            feats[f"sound:{tok}"] = feats.get(f"sound:{tok}", 0.0) + 1.5
            continue
        if len(tok) < 3:
            continue
        # Compress stretched onomatopoeic tails (e.g. "whirrrr" -> "whirr", "shhhh" -> "shh")
        canonical = re.sub(r"(.)\1{2,}", r"\1\1", tok)
        feats[f"w:{canonical}"] = feats.get(f"w:{canonical}", 0.0) + 1.5
        if canonical != tok:
            feats["rhythm:stretched_sound"] = feats.get("rhythm:stretched_sound", 0.0) + 1.0
        if len(canonical) >= 5:
            stem = canonical[:5]
            feats[f"s:{stem}"] = feats.get(f"s:{stem}", 0.0) + 1.2
        padded = f"<{canonical}>"
        for i in range(len(padded) - 3):
            gram = padded[i : i + 4]
            feats[f"g:{gram}"] = feats.get(f"g:{gram}", 0.0) + 0.4
    return feats


def _cosine_similarity(feats_a: Mapping[str, float], feats_b: Mapping[str, float]) -> float:
    if not feats_a or not feats_b:
        return 0.0
    dot = sum(v * feats_b.get(k, 0.0) for k, v in feats_a.items())
    norm_a = math.sqrt(sum(v * v for v in feats_a.values()))
    norm_b = math.sqrt(sum(v * v for v in feats_b.values()))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def semantic_child_drawer_signature(
    leaf_id: str,
    *,
    department_prefix: str,
    description: str | None = None,
    anchor_weights: Sequence[float] | None = None,
) -> tuple[float, ...]:
    """Project a child leaf onto the 8 Founding Trait Anchors of its department (`[0.10, 1.0]^8`).

    - Primary path (`anchor_weights` provided at leaf birth by the 9B Teacher or MiniLM neural
      bridge): normalizes the 8 birth-scored departmental trait weights in `O(1)` (`0.0 ms` CPU).
    - Fallback path (when `anchor_weights is None`): compares the leaf's English description
      against the 8 un-stuffed Founding Trait descriptions in `DEPARTMENT_FOUNDING_ANCHORS`.
    """
    if department_prefix not in DEPARTMENT_FOUNDING_ANCHORS:
        raise ValueError(f"Unknown department_prefix: {department_prefix!r}")

    if anchor_weights is not None:
        if len(anchor_weights) != DEPARTMENT_CHILD_CHORD_DIM:
            raise ValueError(
                f"anchor_weights must have length {DEPARTMENT_CHILD_CHORD_DIM}, got {len(anchor_weights)}"
            )
        clamped = [max(0.05, _clamp01(float(w))) for w in anchor_weights]
        peak_w = max(clamped)
        return tuple(round(v / peak_w, 6) for v in clamped)

    short_name = leaf_id.split("/")[-1].replace("_", " ").replace("-", " ")
    resolved_desc = description or _KNOWN_LEAF_DESCRIPTIONS.get(leaf_id, "")
    full_leaf_text = f"{short_name} {resolved_desc}".strip()
    leaf_feats = _extract_semantic_features(full_leaf_text)

    anchors = DEPARTMENT_FOUNDING_ANCHORS[department_prefix]
    raw_scores: list[float] = []
    for anchor_text in anchors:
        anchor_feats = _extract_semantic_features(anchor_text)
        sim = _cosine_similarity(leaf_feats, anchor_feats)
        raw_scores.append(0.10 + 0.90 * sim)

    # Also blend similarity against already-born exemplar leaves in the same department
    # so newly minted synonymous leaves inherit their exemplar neighborhood naturally.
    for ex_id, ex_profile in _KNOWN_LEAF_ANCHOR_PROFILES.items():
        if ex_id.startswith(department_prefix):
            ex_desc = _KNOWN_LEAF_DESCRIPTIONS.get(ex_id, "")
            ex_short = ex_id.split("/")[-1].replace("_", " ")
            ex_feats = _extract_semantic_features(f"{ex_short} {ex_desc}")
            ex_sim = _cosine_similarity(leaf_feats, ex_feats)
            if ex_sim > 0.0:
                for i in range(DEPARTMENT_CHILD_CHORD_DIM):
                    raw_scores[i] += 1.25 * ex_sim * ex_profile[i]

    # Add gentle structural parent-branch grounding so even bare leaf IDs carry branch topology
    parts = leaf_id.split("/")
    parent_branch = "/".join(parts[:2]) if len(parts) >= 2 else ""
    branch_boost_map: dict[str, tuple[int, int]] = {
        "cultural/efficiency": (0, 1),
        "cultural/justice": (2, 3),
        "cultural/freedom": (4, 5),
        "philosophical/being": (0, 1),
        "philosophical/structure": (2, 3),
        "philosophical/persistence": (4, 5),
        "experiential/reflection": (0, 1),
        "experiential/phase": (2, 3),
        "experiential/synchronization": (4, 5),
    }
    if parent_branch in branch_boost_map:
        b0, b1 = branch_boost_map[parent_branch]
        raw_scores[b0] += 0.06
        raw_scores[b1] += 0.04

    peak = max(raw_scores)
    return tuple(round(v / peak, 6) for v in raw_scores)


def score_leaf_anchor_weights_at_birth(
    leaf: LearnedLeaf,
    *,
    pair_scorer: Callable[[str, str], float] | None = None,
) -> LearnedLeaf:
    """Score a new `LearnedLeaf` once at birth against its department's 8 founding traits.

    - When `pair_scorer` (such as a neural cross-encoder bridge or 9B Teacher scorer) is supplied,
      it scores `(founding_anchor_description, leaf.description)` across all 8 departmental traits
      and attaches the resulting 8-float `anchor_weights` tuple to `leaf`.
    - When `pair_scorer` is None, it uses `semantic_child_drawer_signature` as an offline fallback.
    """
    dept_prefix = leaf.leaf_id.split("/")[0] + "/"
    if dept_prefix not in DEPARTMENT_FOUNDING_ANCHORS:
        raise ValueError(f"Unknown department for leaf: {leaf.leaf_id!r}")

    if pair_scorer is not None:
        anchors = DEPARTMENT_FOUNDING_ANCHORS[dept_prefix]
        raw = [max(0.05, _clamp01(float(pair_scorer(anchor, leaf.description)))) for anchor in anchors]
        peak = max(raw)
        weights = tuple(round(v / peak, 6) for v in raw)
    else:
        weights = semantic_child_drawer_signature(
            leaf.leaf_id,
            department_prefix=dept_prefix,
            description=leaf.description,
        )

    return LearnedLeaf(
        parent_branch=leaf.parent_branch,
        leaf_name=leaf.leaf_name,
        description=leaf.description,
        anchor_weights=weights,
    )


def pool_departmental_child_chord(
    coordinates: Mapping[str, float],
    *,
    department_prefix: str,
    extra_notes: Mapping[str, float] | None = None,
    learned_leaves: Sequence[LearnedLeaf] = (),
    leaf_descriptions: Mapping[str, str] | None = None,
    leaf_anchor_weights: Mapping[str, Sequence[float]] | None = None,
) -> tuple[float, ...]:
    """Pool all active Tier-2/Tier-3 child leaves in a single department into an 8-float chord.

    - Only reads child leaves (`leaf_tier >= 2`) belonging to `department_prefix`
      (`"cultural/"`, `"philosophical/"`, or `"experiential/"`).
    - Uses `LearnedLeaf.anchor_weights` (scored once at leaf birth) when available, or
      `semantic_child_drawer_signature` as an English fallback.
    """
    if department_prefix not in ("cultural/", "philosophical/", "experiential/"):
        raise ValueError(f"Invalid department_prefix: {department_prefix!r}")

    merged: dict[str, float] = dict(coordinates)
    if extra_notes:
        for k, v in extra_notes.items():
            merged[k] = max(float(merged.get(k, 0.0)), float(v))

    desc_lookup: dict[str, str] = dict(_KNOWN_LEAF_DESCRIPTIONS)
    weights_lookup: dict[str, Sequence[float]] = {}
    for ll in learned_leaves:
        desc_lookup[ll.leaf_id] = ll.description
        if ll.anchor_weights is not None:
            weights_lookup[ll.leaf_id] = ll.anchor_weights
    if leaf_descriptions:
        desc_lookup.update(leaf_descriptions)
    if leaf_anchor_weights:
        weights_lookup.update(leaf_anchor_weights)

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
        basis = semantic_child_drawer_signature(
            leaf_id,
            department_prefix=department_prefix,
            description=desc_lookup.get(leaf_id),
            anchor_weights=weights_lookup.get(leaf_id),
        )
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

    @classmethod
    def from_dict(cls, data: Mapping[str, float]) -> "SubliminalChordVector":
        flat = [float(data[name]) for name in SUBLIMINAL_FEATURE_NAMES]
        return cls.from_flat_values(flat)

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
        learned_leaves: Sequence[LearnedLeaf] = (),
        leaf_descriptions: Mapping[str, str] | None = None,
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
            learned_leaves=learned_leaves,
            leaf_descriptions=leaf_descriptions,
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
            learned_leaves=learned_leaves,
            leaf_descriptions=leaf_descriptions,
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
            learned_leaves=learned_leaves,
            leaf_descriptions=leaf_descriptions,
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
    """Neural Corpus Callosum: projects the 4 strictly disjoint Departmental Vectors into 4 virtual tokens `[4, hidden_size]`.

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
    proj_param = next(projector.parameters(), None)
    proj_dtype = proj_param.dtype if proj_param is not None else base_embeds.dtype
    projected_gaps = projector(chord_vectors.to(device=device, dtype=proj_dtype)).to(
        dtype=base_embeds.dtype
    )
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


@dataclass(frozen=True)
class CorpusCallosumMetadata:
    """Versioned maturation record for `SubliminalGapProjector` across Sleep Cycles."""

    schema_version: int
    hidden_size: int
    intermediate_size: int
    sleep_cycle_index: int
    trained_trajectories_count: int
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "input_dim": SUBLIMINAL_VECTOR_DIM,
            "department_dims": {
                "doer": DOER_DEPT_DIM,
                "thinker": THINKER_DEPT_DIM,
                "feeler": FEELER_DEPT_DIM,
                "cadence": CADENCE_DEPT_DIM,
            },
            "sleep_cycle_index": self.sleep_cycle_index,
            "trained_trajectories_count": self.trained_trajectories_count,
            "notes": self.notes,
        }


def save_corpus_callosum_checkpoint(
    checkpoint_dir: str | Path,
    projector: SubliminalGapProjector,
    *,
    sleep_cycle_index: int = 0,
    trained_trajectories_count: int = 0,
    notes: str = "",
) -> CorpusCallosumMetadata:
    """Persist `SubliminalGapProjector` weights and Sleep-Cycle maturation metadata to disk."""
    if sleep_cycle_index < 0 or trained_trajectories_count < 0:
        raise ValueError("sleep_cycle_index and trained_trajectories_count must be >= 0")

    out_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    first_linear = projector.department_heads[0][0]
    inter_size = int(first_linear.out_features)

    meta = CorpusCallosumMetadata(
        schema_version=1,
        hidden_size=projector.hidden_size,
        intermediate_size=inter_size,
        sleep_cycle_index=int(sleep_cycle_index),
        trained_trajectories_count=int(trained_trajectories_count),
        notes=str(notes),
    )
    atomic_json(out_dir / "corpus_callosum_meta.json", meta.to_dict())
    torch.save(projector.state_dict(), out_dir / "projector_weights.pt")
    return meta


def load_corpus_callosum_checkpoint(
    checkpoint_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[SubliminalGapProjector, CorpusCallosumMetadata]:
    """Reload a saved `SubliminalGapProjector` and its Sleep-Cycle maturation metadata."""
    in_dir = Path(checkpoint_dir)
    meta_path = in_dir / "corpus_callosum_meta.json"
    weights_path = in_dir / "projector_weights.pt"
    if not meta_path.exists() or not weights_path.exists():
        raise FileNotFoundError(f"Missing Corpus Callosum checkpoint files in {in_dir}")

    raw_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if raw_meta.get("schema_version") != 1:
        raise ValueError(f"Unsupported Corpus Callosum schema_version: {raw_meta.get('schema_version')}")

    meta = CorpusCallosumMetadata(
        schema_version=1,
        hidden_size=int(raw_meta["hidden_size"]),
        intermediate_size=int(raw_meta["intermediate_size"]),
        sleep_cycle_index=int(raw_meta["sleep_cycle_index"]),
        trained_trajectories_count=int(raw_meta["trained_trajectories_count"]),
        notes=str(raw_meta.get("notes", "")),
    )
    projector = SubliminalGapProjector(
        hidden_size=meta.hidden_size,
        intermediate_size=meta.intermediate_size,
    )
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    projector.load_state_dict(state_dict)
    projector.to(device)
    return projector, meta

