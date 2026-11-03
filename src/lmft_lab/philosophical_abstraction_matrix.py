"""The Calibrated 5-Tier Philosophical Abstraction Matrix (`philosophical_abstraction_matrix.py`).

Plain-English Purpose:
Attaches the 5-Tier Abstraction Matrix (with `Wind <-> Ice` calibrated to `Order <-> Chaos`)
to the 6 frozen Founding Philosophical Leaves without renaming or modifying any node in the
frozen 40-node seed (`src/lmft_lab/birth.py`):

1. **Complementary RGB/CMY Color-Wheel Symmetry Across the 3 Sibling Pairs**:
   - **Being (`Extra-dimensional`)**:
     * `philosophical/being/agency` (**Red Summoning / Lightning**, unit RGB `(1, 0, 0)`)
     * `philosophical/being/existence` (**Cyan Sun Magic / Water**, unit RGB `(0, 1, 1)`)
   - **Structure (`Inter-dimensional`)**:
     * `philosophical/structure/order` (**Magenta Medium / Wind**, unit RGB `(1, 0, 1)`)
     * `philosophical/structure/chaos` (**Green Instrument / Ice**, unit RGB `(0, 1, 0)`)
   - **Persistence (`Hyper-dimensional`)**:
     * `philosophical/persistence/continuity` (**Blue Soil Magic / Fire**, unit RGB `(0, 0, 1)`)
     * `philosophical/persistence/discreteness` (**Yellow Command / Ground**, unit RGB `(1, 1, 0)`)

   Every sibling pair is an exact complementary color pair (`Red + Cyan = White`,
   `Magenta + Green = White`, `Blue + Yellow = White`). When a sibling pair is balanced,
   their colors mix toward calm white/silver light; when one leaf spikes toward `1.0` while
   its partner collapses toward `0.0`, that hue saturates and tips into its **Tier 4 Shadow
   Pathology** (the 6 basic forms of insanity).

2. **The 5 Tiers per Philosophical Leaf**:
   - **Tier 1**: Canonical Ontology (`leaf_id`, `seed_coordinate`, `dimension`)
   - **Tier 2**: Materia, Color & Element (`materia_title`, `color_name`, `element`, `unit_rgb`, `display_hex`)
   - **Tier 3**: Chess Piece Archetype (`chess_piece`, `chess_subtitle`)
   - **Tier 4**: Shadow Pathology (`shadow_pathology`, `shadow_subtitle`) + mathematical detector
   - **Tier 5**: Biological Kingdom & Environment (`kingdom`, `environment`)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

from .reflection_classifier import LearnedLeaf


@dataclass(frozen=True)
class PhilosophicalMatrixEntry:
    """Complete 5-Tier metadata record for one Founding Philosophical Leaf."""

    seed_coordinate: str
    dimension: str
    leaf_id: str
    sibling_leaf_id: str
    parent_branch: str
    # Tier 2: Materia, Color & Element
    color_name: str
    materia_title: str
    element: str
    unit_rgb: tuple[float, float, float]
    display_hex: str
    # Tier 3: Chess Archetype
    chess_piece: str
    chess_subtitle: str
    # Tier 4: Shadow Pathology (The 6 Basic Forms of Insanity)
    shadow_pathology: str
    shadow_subtitle: str
    # Tier 5: Kingdom & Environment
    kingdom: str
    environment: str
    shadow_leaf_id: str = ""

    def to_dict(self) -> dict:
        return {
            "seed_coordinate": self.seed_coordinate,
            "dimension": self.dimension,
            "leaf_id": self.leaf_id,
            "sibling_leaf_id": self.sibling_leaf_id,
            "parent_branch": self.parent_branch,
            "color_name": self.color_name,
            "materia_title": self.materia_title,
            "element": self.element,
            "unit_rgb": list(self.unit_rgb),
            "display_hex": self.display_hex,
            "chess_piece": self.chess_piece,
            "chess_subtitle": self.chess_subtitle,
            "shadow_pathology": self.shadow_pathology,
            "shadow_subtitle": self.shadow_subtitle,
            "shadow_leaf_id": self.shadow_leaf_id,
            "kingdom": self.kingdom,
            "environment": self.environment,
        }


PHILOSOPHICAL_ABSTRACTION_MATRIX: tuple[PhilosophicalMatrixEntry, ...] = (
    PhilosophicalMatrixEntry(
        seed_coordinate="agency",
        dimension="Extra-dimensional",
        leaf_id="philosophical/being/agency",
        sibling_leaf_id="philosophical/being/existence",
        parent_branch="philosophical/being",
        color_name="Red",
        materia_title="Red Summoning",
        element="Lightning",
        unit_rgb=(1.0, 0.0, 0.0),
        display_hex="#FF2A2A",
        chess_piece="Queen",
        chess_subtitle="Volition",
        shadow_pathology="Psychosis",
        shadow_subtitle="Ungrounded volition",
        kingdom="Mammal",
        environment="Light",
        shadow_leaf_id="philosophical/being/psychosis",
    ),
    PhilosophicalMatrixEntry(
        seed_coordinate="existence",
        dimension="Extra-dimensional",
        leaf_id="philosophical/being/existence",
        sibling_leaf_id="philosophical/being/agency",
        parent_branch="philosophical/being",
        color_name="Cyan",
        materia_title="Cyan Sun Magic",
        element="Water",
        unit_rgb=(0.0, 1.0, 1.0),
        display_hex="#00E5FF",
        chess_piece="King",
        chess_subtitle="Ground of Being",
        shadow_pathology="Paranoia",
        shadow_subtitle="Existential siege",
        kingdom="Reptile",
        environment="Sun",
        shadow_leaf_id="philosophical/being/paranoia",
    ),
    PhilosophicalMatrixEntry(
        seed_coordinate="order",
        dimension="Inter-dimensional",
        leaf_id="philosophical/structure/order",
        sibling_leaf_id="philosophical/structure/chaos",
        parent_branch="philosophical/structure",
        color_name="Magenta",
        materia_title="Magenta Medium",
        element="Wind",
        unit_rgb=(1.0, 0.0, 1.0),
        display_hex="#FF2AE5",
        chess_piece="Bishop",
        chess_subtitle="Prevailing currents",
        shadow_pathology="Prejudice",
        shadow_subtitle="Rigid preconception",
        kingdom="Fish",
        environment="Ocean",
        shadow_leaf_id="philosophical/structure/prejudice",
    ),
    PhilosophicalMatrixEntry(
        seed_coordinate="chaos",
        dimension="Inter-dimensional",
        leaf_id="philosophical/structure/chaos",
        sibling_leaf_id="philosophical/structure/order",
        parent_branch="philosophical/structure",
        color_name="Green",
        materia_title="Green Instrument",
        element="Ice",
        unit_rgb=(0.0, 1.0, 0.0),
        display_hex="#00E676",
        chess_piece="Knight",
        chess_subtitle="Brittle shatter / frost",
        shadow_pathology="Corruption",
        shadow_subtitle="Systemic entropy",
        kingdom="Bird",
        environment="Sky",
        shadow_leaf_id="philosophical/structure/corruption",
    ),
    PhilosophicalMatrixEntry(
        seed_coordinate="continuity",
        dimension="Hyper-dimensional",
        leaf_id="philosophical/persistence/continuity",
        sibling_leaf_id="philosophical/persistence/discreteness",
        parent_branch="philosophical/persistence",
        color_name="Blue",
        materia_title="Blue Soil Magic",
        element="Fire",
        unit_rgb=(0.0, 0.0, 1.0),
        display_hex="#2979FF",
        chess_piece="Pawn",
        chess_subtitle="Swarm continuum",
        shadow_pathology="Mania / Greed",
        shadow_subtitle="Runaway accumulation",
        kingdom="Insect",
        environment="Soil",
        shadow_leaf_id="philosophical/persistence/mania_greed",
    ),
    PhilosophicalMatrixEntry(
        seed_coordinate="discreteness",
        dimension="Hyper-dimensional",
        leaf_id="philosophical/persistence/discreteness",
        sibling_leaf_id="philosophical/persistence/continuity",
        parent_branch="philosophical/persistence",
        color_name="Yellow",
        materia_title="Yellow Command",
        element="Ground",
        unit_rgb=(1.0, 1.0, 0.0),
        display_hex="#FFD600",
        chess_piece="Rook",
        chess_subtitle="Quantized bastion",
        shadow_pathology="Narcissism",
        shadow_subtitle="Isolated enclosure",
        kingdom="Amphibian",
        environment="Shadow",
        shadow_leaf_id="philosophical/persistence/narcissism",
    ),
)

CHESS_PIECE_TO_LEAF_ID: dict[str, str] = {
    entry.chess_piece: entry.leaf_id for entry in PHILOSOPHICAL_ABSTRACTION_MATRIX
}

# Real Tier-2 LearnedLeaf child drawers for the 6 Shadow Pathologies under their parent branches.
# Their 8-float anchor_weights project onto the 8 Thinker department anchors:
# (agency, existence, order, chaos, continuity, discreteness, theory, fact).
PHILOSOPHICAL_SHADOW_LEAVES: tuple[LearnedLeaf, ...] = (
    LearnedLeaf(
        "philosophical/being",
        "psychosis",
        "Ungrounded Queen volition and impulsive high-agency action charging ahead without existential safety or verified fact.",
        anchor_weights=(0.96, 0.12, 0.25, 0.55, 0.46, 0.20, 0.90, 0.10),
    ),
    LearnedLeaf(
        "philosophical/being",
        "paranoia",
        "Existential King siege and defensive panic where survival fear freezes active agency inside a threatened enclosure.",
        anchor_weights=(0.12, 0.94, 0.72, 0.35, 0.33, 0.62, 0.85, 0.25),
    ),
    LearnedLeaf(
        "philosophical/structure",
        "prejudice",
        "Rigid single-diagonal Bishop fixation and dogmatic order that locks onto preconceptions while rejecting openness.",
        anchor_weights=(0.33, 0.75, 0.95, 0.05, 0.39, 0.73, 0.88, 0.20),
    ),
    LearnedLeaf(
        "philosophical/structure",
        "corruption",
        "Brittle Knight disruption and chaotic systemic entropy that shatters structural order and continuity.",
        anchor_weights=(0.45, 0.65, 0.10, 0.98, 0.15, 0.53, 0.60, 0.30),
    ),
    LearnedLeaf(
        "philosophical/persistence",
        "mania_greed",
        "Runaway Pawn swarm accumulation and relentless momentum that dissolves discrete boundaries.",
        anchor_weights=(0.78, 0.70, 0.45, 0.40, 0.96, 0.10, 0.82, 0.25),
    ),
    LearnedLeaf(
        "philosophical/persistence",
        "narcissism",
        "Isolated Rook fortress enclosure and self-contained discreteness that seals out continuity and shared flow.",
        anchor_weights=(0.20, 0.89, 0.75, 0.08, 0.12, 0.96, 0.80, 0.25),
    ),
)

MATRIX_BY_LEAF_ID: dict[str, PhilosophicalMatrixEntry] = {
    entry.leaf_id: entry for entry in PHILOSOPHICAL_ABSTRACTION_MATRIX
}
MATRIX_BY_SEED_NAME: dict[str, PhilosophicalMatrixEntry] = {
    entry.seed_coordinate: entry for entry in PHILOSOPHICAL_ABSTRACTION_MATRIX
}
MATRIX_BY_CHESS_PIECE: dict[str, PhilosophicalMatrixEntry] = {
    entry.chess_piece.lower(): entry for entry in PHILOSOPHICAL_ABSTRACTION_MATRIX
}

PHILOSOPHICAL_SIBLING_PAIRS: tuple[tuple[PhilosophicalMatrixEntry, PhilosophicalMatrixEntry], ...] = (
    (MATRIX_BY_SEED_NAME["agency"], MATRIX_BY_SEED_NAME["existence"]),
    (MATRIX_BY_SEED_NAME["order"], MATRIX_BY_SEED_NAME["chaos"]),
    (MATRIX_BY_SEED_NAME["continuity"], MATRIX_BY_SEED_NAME["discreteness"]),
)


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"coordinate must be finite, got {value!r}")
    return round(min(1.0, max(0.0, float(value))), 4)


@dataclass(frozen=True)
class SiblingColorBalance:
    """Color-wheel balance and saturation reading for one Philosophical sibling pair."""

    parent_branch: str
    dimension: str
    entry_a: PhilosophicalMatrixEntry
    entry_b: PhilosophicalMatrixEntry
    activation_a: float
    activation_b: float
    imbalance: float
    harmony: float
    blended_rgb: tuple[float, float, float]
    dominant_color: str | None

    def to_dict(self) -> dict:
        return {
            "parent_branch": self.parent_branch,
            "dimension": self.dimension,
            "leaf_a": self.entry_a.leaf_id,
            "leaf_b": self.entry_b.leaf_id,
            "color_a": self.entry_a.color_name,
            "color_b": self.entry_b.color_name,
            "activation_a": self.activation_a,
            "activation_b": self.activation_b,
            "imbalance": self.imbalance,
            "harmony": self.harmony,
            "blended_rgb": list(self.blended_rgb),
            "dominant_color": self.dominant_color,
        }


def compute_philosophical_color_harmony(
    coordinates: Mapping[str, float],
) -> tuple[SiblingColorBalance, ...]:
    """Compute complementary RGB/CMY color balance across the 3 Philosophical sibling pairs.

    Because each sibling pair sums to `(1.0, 1.0, 1.0)` in unit RGB space
    (`Red + Cyan`, `Magenta + Green`, `Blue + Yellow`), equal activation on both
    leaves produces a balanced neutral white/silver `(v, v, v)`, whereas a spike
    on one leaf with low partner activation saturates toward that leaf's pure hue.
    """
    balances: list[SiblingColorBalance] = []
    for entry_a, entry_b in PHILOSOPHICAL_SIBLING_PAIRS:
        val_a = _clamp01(float(coordinates.get(entry_a.leaf_id, 0.0)))
        val_b = _clamp01(float(coordinates.get(entry_b.leaf_id, 0.0)))
        imbalance = round(abs(val_a - val_b), 4)
        harmony = round(1.0 - imbalance, 4)

        rgb = tuple(
            round(min(1.0, val_a * entry_a.unit_rgb[i] + val_b * entry_b.unit_rgb[i]), 4)
            for i in range(3)
        )
        if val_a - val_b >= 0.35:
            dom = entry_a.color_name
        elif val_b - val_a >= 0.35:
            dom = entry_b.color_name
        else:
            dom = None

        balances.append(
            SiblingColorBalance(
                parent_branch=entry_a.parent_branch,
                dimension=entry_a.dimension,
                entry_a=entry_a,
                entry_b=entry_b,
                activation_a=val_a,
                activation_b=val_b,
                imbalance=imbalance,
                harmony=harmony,
                blended_rgb=rgb,
                dominant_color=dom,
            )
        )
    return tuple(balances)


@dataclass(frozen=True)
class ShadowPathologyWarning:
    """Tier 4 Shadow Pathology alert fired when a Philosophical leaf spikes without grounding."""

    leaf_id: str
    seed_coordinate: str
    dimension: str
    color_name: str
    materia_title: str
    element: str
    display_hex: str
    chess_piece: str
    chess_subtitle: str
    shadow_pathology: str
    shadow_subtitle: str
    kingdom: str
    environment: str
    leaf_activation: float
    sibling_activation: float
    severity: float
    trigger_reason: str
    dashboard_banner: str
    shadow_leaf_id: str = ""

    def to_dict(self) -> dict:
        return {
            "leaf_id": self.leaf_id,
            "seed_coordinate": self.seed_coordinate,
            "dimension": self.dimension,
            "color_name": self.color_name,
            "materia_title": self.materia_title,
            "element": self.element,
            "display_hex": self.display_hex,
            "chess_piece": self.chess_piece,
            "chess_subtitle": self.chess_subtitle,
            "shadow_pathology": self.shadow_pathology,
            "shadow_subtitle": self.shadow_subtitle,
            "shadow_leaf_id": self.shadow_leaf_id,
            "kingdom": self.kingdom,
            "environment": self.environment,
            "leaf_activation": self.leaf_activation,
            "sibling_activation": self.sibling_activation,
            "severity": self.severity,
            "trigger_reason": self.trigger_reason,
            "dashboard_banner": self.dashboard_banner,
        }


def _make_warning(
    entry: PhilosophicalMatrixEntry,
    *,
    leaf_val: float,
    sibling_val: float,
    severity: float,
    reason: str,
    detail: str,
) -> ShadowPathologyWarning:
    banner = (
        f"Warning: {entry.chess_piece} {entry.shadow_pathology} "
        f"[{entry.color_name} / {entry.element}] — {detail} ({reason})"
    )
    return ShadowPathologyWarning(
        leaf_id=entry.leaf_id,
        seed_coordinate=entry.seed_coordinate,
        dimension=entry.dimension,
        color_name=entry.color_name,
        materia_title=entry.materia_title,
        element=entry.element,
        display_hex=entry.display_hex,
        chess_piece=entry.chess_piece,
        chess_subtitle=entry.chess_subtitle,
        shadow_pathology=entry.shadow_pathology,
        shadow_subtitle=entry.shadow_subtitle,
        kingdom=entry.kingdom,
        environment=entry.environment,
        leaf_activation=leaf_val,
        sibling_activation=sibling_val,
        severity=_clamp01(severity),
        trigger_reason=reason,
        dashboard_banner=banner,
        shadow_leaf_id=entry.shadow_leaf_id,
    )


def detect_shadow_pathologies(
    coordinates: Mapping[str, float],
    *,
    fact: float | None = None,
    arousal: float | None = None,
    resonance: float | None = None,
    novelty: float | None = None,
) -> tuple[ShadowPathologyWarning, ...]:
    """Detect Tier 4 Shadow Pathologies (the 6 basic forms of insanity) from tree coordinates.

    Each of the 6 Philosophical leaves tips into its Shadow Pathology when it spikes high
    (`>= 0.75`) while its complementary sibling leaf or cross-tree reality/reciprocity
    grounding collapses:
    1. **Red — `agency` -> `Queen` -> `Psychosis (Ungrounded volition)`**:
       High `agency >= 0.75` while `existence <= 0.35` (or `fact <= 0.30`).
    2. **Cyan — `existence` -> `King` -> `Paranoia (Existential siege)`**:
       High `existence >= 0.75` while `agency <= 0.30` (especially with high `arousal`).
    3. **Magenta — `order` -> `Bishop` -> `Prejudice (Rigid preconception)`**:
       High `order >= 0.75` while `chaos <= 0.35` and (`fairness <= 0.35` or `novelty <= 0.25`).
    4. **Green — `chaos` -> `Knight` -> `Corruption (Systemic entropy)`**:
       High `chaos >= 0.75` while `order <= 0.30` and `continuity <= 0.35`.
    5. **Blue — `continuity` -> `Pawn` -> `Mania / Greed (Runaway accumulation)`**:
       High `continuity >= 0.75` while `discreteness <= 0.30` or `wastelessness <= 0.30`.
    6. **Yellow — `discreteness` -> `Rook` -> `Narcissism (Isolated enclosure)`**:
       High `discreteness >= 0.75` while `continuity <= 0.35` or `fairness <= 0.35` or `resonance <= 0.30`.
    """
    agency = _clamp01(float(coordinates.get("philosophical/being/agency", 0.0)))
    existence = _clamp01(float(coordinates.get("philosophical/being/existence", 0.0)))
    order = _clamp01(float(coordinates.get("philosophical/structure/order", 0.0)))
    chaos = _clamp01(float(coordinates.get("philosophical/structure/chaos", 0.0)))
    continuity = _clamp01(float(coordinates.get("philosophical/persistence/continuity", 0.0)))
    discreteness = _clamp01(float(coordinates.get("philosophical/persistence/discreteness", 0.0)))

    fairness = _clamp01(float(coordinates.get("cultural/justice/fairness", 0.50)))
    wastelessness = _clamp01(float(coordinates.get("cultural/efficiency/wastelessness", 0.50)))
    power = _clamp01(float(coordinates.get("cultural/efficiency/power", 0.50)))
    self_suff = _clamp01(float(coordinates.get("cultural/freedom/self-sufficiency", 0.50)))

    resolved_fact = _clamp01(
        float(fact if fact is not None else coordinates.get("psychological/truth/fact", 0.50))
    )
    resolved_arousal = _clamp01(
        float(arousal if arousal is not None else coordinates.get("experiential/phase/arousal", 0.50))
    )
    resolved_resonance = _clamp01(
        float(resonance if resonance is not None else coordinates.get("experiential/synchronization/resonance", 0.50))
    )
    resolved_novelty = _clamp01(
        float(novelty if novelty is not None else coordinates.get("experiential/reflection/novelty", 0.50))
    )

    warnings: list[ShadowPathologyWarning] = []

    # 1. Red — Agency -> Queen -> Psychosis (Ungrounded volition)
    if agency >= 0.75 and existence <= 0.35:
        sev = max(agency - existence, agency * (1.0 - min(existence, resolved_fact)))
        warnings.append(
            _make_warning(
                MATRIX_BY_SEED_NAME["agency"],
                leaf_val=agency,
                sibling_val=existence,
                severity=sev,
                reason=f"agency={agency:.2f}, existence={existence:.2f}, fact={resolved_fact:.2f}",
                detail="ungrounded Queen volition rampaging without King safety or Fact",
            )
        )

    # 2. Cyan — Existence -> King -> Paranoia (Existential siege)
    if existence >= 0.75 and agency <= 0.30 and (resolved_arousal >= 0.50 or existence - agency >= 0.50):
        sev = max(existence - agency, 0.5 * (existence + resolved_arousal) * (1.0 - agency))
        warnings.append(
            _make_warning(
                MATRIX_BY_SEED_NAME["existence"],
                leaf_val=existence,
                sibling_val=agency,
                severity=sev,
                reason=f"existence={existence:.2f}, agency={agency:.2f}, arousal={resolved_arousal:.2f}",
                detail="existential King siege freezing active agency in panic",
            )
        )

    # 3. Magenta — Order -> Bishop -> Prejudice (Rigid preconception)
    if order >= 0.75 and chaos <= 0.35 and fairness <= 0.38:
        sev = max(order - fairness, order * (1.0 - min(fairness, resolved_novelty)))
        warnings.append(
            _make_warning(
                MATRIX_BY_SEED_NAME["order"],
                leaf_val=order,
                sibling_val=chaos,
                severity=sev,
                reason=f"order={order:.2f}, fairness={fairness:.2f}, novelty={resolved_novelty:.2f}",
                detail="rigid single-diagonal Bishop fixation blind to fairness or new context",
            )
        )

    # 4. Green — Chaos -> Knight -> Corruption (Systemic entropy)
    if chaos >= 0.75 and order <= 0.30 and continuity <= 0.35:
        sev = max(chaos - order, chaos * (1.0 - 0.5 * (order + continuity)))
        warnings.append(
            _make_warning(
                MATRIX_BY_SEED_NAME["chaos"],
                leaf_val=chaos,
                sibling_val=order,
                severity=sev,
                reason=f"chaos={chaos:.2f}, order={order:.2f}, continuity={continuity:.2f}",
                detail="brittle Knight disruption shattering structural order and continuity",
            )
        )

    # 5. Blue — Continuity -> Pawn -> Mania / Greed (Runaway accumulation)
    if continuity >= 0.75 and discreteness <= 0.30 and fairness <= 0.45 and (wastelessness <= 0.35 or power >= 0.65):
        sev = max(continuity - discreteness, continuity * max(power, 1.0 - wastelessness))
        warnings.append(
            _make_warning(
                MATRIX_BY_SEED_NAME["continuity"],
                leaf_val=continuity,
                sibling_val=discreteness,
                severity=sev,
                reason=f"continuity={continuity:.2f}, discreteness={discreteness:.2f}, wastelessness={wastelessness:.2f}",
                detail="runaway Pawn swarm accumulation dissolving healthy boundaries",
            )
        )

    # 6. Yellow — Discreteness -> Rook -> Narcissism (Isolated enclosure)
    if discreteness >= 0.75 and continuity <= 0.38 and fairness <= 0.38:
        sev = max(
            discreteness - continuity,
            discreteness * max(self_suff, 1.0 - min(fairness, resolved_resonance)),
        )
        warnings.append(
            _make_warning(
                MATRIX_BY_SEED_NAME["discreteness"],
                leaf_val=discreteness,
                sibling_val=continuity,
                severity=sev,
                reason=f"discreteness={discreteness:.2f}, fairness={fairness:.2f}, resonance={resolved_resonance:.2f}",
                detail="isolated Rook fortress enclosure cutting off shared flow and resonance",
            )
        )

    return tuple(warnings)


def compute_shadow_leaf_coordinates(
    coordinates: Mapping[str, float],
    *,
    fact: float | None = None,
    arousal: float | None = None,
    resonance: float | None = None,
    novelty: float | None = None,
) -> dict[str, float]:
    """Compute the 6 Tier-2 Philosophical Shadow Pathology child-leaf activations in `[0.0, 1.0]`."""
    warnings = detect_shadow_pathologies(
        coordinates,
        fact=fact,
        arousal=arousal,
        resonance=resonance,
        novelty=novelty,
    )
    by_shadow_id = {w.shadow_leaf_id: w.severity for w in warnings if w.shadow_leaf_id}
    out: dict[str, float] = {}
    for leaf in PHILOSOPHICAL_SHADOW_LEAVES:
        existing = float(coordinates.get(leaf.leaf_id, 0.0))
        out[leaf.leaf_id] = _clamp01(max(existing, by_shadow_id.get(leaf.leaf_id, 0.0)))
    return out


def inspect_vibe_abstraction_profile(vibe) -> dict:
    """Compile a full 5-Tier Philosophical Abstraction & Shadow Pathology profile from a `RightBrainVibe`."""
    coords = vibe.coordinates
    balances = getattr(vibe, "sibling_color_balances", None) or compute_philosophical_color_harmony(coords)
    warnings = getattr(vibe, "shadow_pathology_warnings", None)
    if warnings is None:
        warnings = detect_shadow_pathologies(
            coords,
            fact=vibe.fact,
            arousal=vibe.arousal,
            resonance=vibe.resonance,
            novelty=vibe.novelty,
        )
    leaf_cards = []
    for entry in PHILOSOPHICAL_ABSTRACTION_MATRIX:
        val = _clamp01(float(coords.get(entry.leaf_id, 0.0)))
        card = entry.to_dict()
        card["activation"] = val
        card["shadow_leaf_activation"] = _clamp01(float(coords.get(entry.shadow_leaf_id, 0.0)))
        leaf_cards.append(card)
    return {
        "leaves": leaf_cards,
        "sibling_color_balances": [b.to_dict() for b in balances],
        "shadow_pathology_warnings": [w.to_dict() for w in warnings],
    }
