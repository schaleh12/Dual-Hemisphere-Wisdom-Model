"""Right-Brain Dominant 3-Bridge Equilibrium Loop, Self-Registry, and Theory Routing.

Plain-English architecture:
1. Three Competing Right-Brain Classifiers (decoupled from Qwen3-4B):
   - Do-er (`Will <-> Cultural`): reads social/value stakes (`Power`, `Wastelessness`,
     `Honor`, `Fairness`, `Self-Sufficiency`, `Needlessness`, plus `cultural/*` drawers)
     and puts forward an action/stance bid.
   - Thinker (`Truth <-> Philosophical`): reads structural reality (`Agency`, `Existence`,
     `Order`, `Chaos`, `Continuity`, `Discreteness`, plus `philosophical/*` drawers)
     and separates direct external `Fact` from candidate `Theory`.
   - Feeler (`Contemplation <-> Experiential`): reads felt weight (`Salience`, `Valence`,
     `Arousal`) plus working-memory `Retention` and `Resonance`, putting forward an
     emotional/coherence bid.
2. Internal Self & Source-Tagging Registry (`InternalSelfRegistry`):
   - Explicitly separates `EXTERNAL_WORLD` sensory facts from `INTERNAL_LEFT_BRAIN_THEORY`
     thoughts so the Right Brain never mistakes its own inner monologue or diary memory
     for a live outside event, while tracking its own bounded rule inventory and blind spots.
3. Directional Cross-Hemisphere Circuit:
   - Outbound (`Will` -> Left Brain): When the Do-er, Thinker, and Feeler clash or hit a
     high-salience blind spot, `Will` sends a `WillTugMessage` carrying its action opinion
     and the Feeler's interjected emotion, plus the `SettledDiary` and `InternalSelfRegistry`.
   - Inbound (Left Brain -> `Truth / Theory` ONLY): `Qwen3-4B` replies in plain English
     (zero JSON float scoring). Its reply is scored strictly into `theory_coordinates`
     (`Truth / Theory`), NEVER overwriting raw `external_coordinates` (`Fact`). Candidate
     theories can synthesize ambiguous scenes (`THEORY_SYNTHESIS`, keeping `Fact` honestly
     low and `Theory` high), while stark raw injustice vetoes pretty rationalizations
     (`SELF_RATIONALIZATION_DISSONANCE`).
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import inspect
import json
import math
from pathlib import Path
import re

from .birth import BirthState
from .branch_archives import (
    OpenThreadTrigger,
    SocialDriftReport,
    ThreeBranchArchives,
    ThreeClassifierDeliberation,
    build_three_branch_chord_rule,
    check_open_thread_trigger,
    deliberate_across_three_branches,
    harvest_into_branch_archives,
)
from .chord_memory import (
    ChordEchoReport,
    ChordMemoryBank,
    FeltChord,
    compile_felt_chord,
    retrieve_chord_echoes,
)
from .contemplation_ledger import (
    ContemplationLedger,
    RawThoughtRecord,
    SubliminalMarginTag,
    append_raw_thought_record,
    build_subliminal_margin_tag,
    save_contemplation_ledger,
)
from .local_models import strict_json
from .model_assets import ModelError
from .philosophical_abstraction_matrix import (
    PHILOSOPHICAL_SHADOW_LEAVES,
    ShadowPathologyWarning,
    SiblingColorBalance,
    compute_philosophical_color_harmony,
    detect_shadow_pathologies,
    inspect_vibe_abstraction_profile,
)
from .propagation import Association, PropagationConfig, PropagationResult, propagate_activation
from .reflection_classifier import (
    AROUSAL_LEAF,
    BRANCH_GATE_THRESHOLD,
    CLARITY_LEAF,
    CONFIDENCE_LEAF,
    CONFUSION_THRESHOLD,
    EXECUTION_LEAF,
    FACT_LEAF,
    FOUNDATIONAL_LEAVES,
    INTENTION_LEAF,
    NOVELTY_LEAF,
    RESONANCE_LEAF,
    RETENTION_LEAF,
    SALIENCE_LEAF,
    SILENCE_SALIENCE_THRESHOLD,
    THEORY_LEAF,
    VALENCE_LEAF,
    ContinuousMemoryState,
    FuzzyConceptRule,
    LearnedLeaf,
    _evaluate_rules_with_inhibition,
    compute_contemplation,
    is_valid_leaf_id,
    leaf_tier,
)
from .runtime_store import atomic_json

WILL_CULTURAL_LEAVES = (
    "cultural/efficiency/power",
    "cultural/efficiency/wastelessness",
    "cultural/justice/honor",
    "cultural/justice/fairness",
    "cultural/freedom/self-sufficiency",
    "cultural/freedom/needlessness",
)

TRUTH_PHILOSOPHICAL_LEAVES = (
    "philosophical/being/agency",
    "philosophical/being/existence",
    "philosophical/structure/order",
    "philosophical/structure/chaos",
    "philosophical/persistence/continuity",
    "philosophical/persistence/discreteness",
)

EXPERIENTIAL_SENSOR_LEAVES = (
    SALIENCE_LEAF,
    VALENCE_LEAF,
    AROUSAL_LEAF,
)

OPTIONAL_EXPERIENTIAL_SENSOR_LEAVES = (
    NOVELTY_LEAF,
    RESONANCE_LEAF,
    RETENTION_LEAF,
)

CONSTRUCTIVE_THEORY_LEAVES = (
    "cultural/justice/fairness",
    "cultural/justice/honor",
    "cultural/efficiency/wastelessness",
    "cultural/freedom/self-sufficiency",
    "cultural/freedom/needlessness",
    "philosophical/structure/order",
    "philosophical/persistence/continuity",
    "philosophical/being/agency",
)

RIGHT_BRAIN_POSTURES = (
    "AT_EASE",
    "WORDLESS_REFLEX",
    "THEORY_SYNTHESIS",
    "TUG_LEFT_BRAIN",
)

SETTLED_OUTCOMES = (
    "AT_EASE",
    "WORDLESS_REFLEX",
    "THEORY_SYNTHESIS",
    "UNRESOLVED_TENSION",
)

EPISTEMIC_SELF_STATES = (
    "QUIET_EQUILIBRIUM",
    "GROUNDED_EXTERNAL_FACT",
    "THEORY_GUIDED_SYNTHESIS",
    "KNOWN_INSTINCT_BLIND_SPOT",
    "SELF_RATIONALIZATION_DISSONANCE",
)

RIGHT_BRAIN_WILL_VERSION = "right-brain-action-will-v2"
LEFT_BRAIN_PROMPT_VERSION = "left-brain-theory-reply-v2"
ACTION_EXECUTION_THRESHOLD = 0.25
DEFAULT_EQUILIBRIUM_TOLERANCE = 0.08

# Untangled Will wiring: Will measures drive to act (salience, arousal, power,
# verified fact, or synthesized theory), quieted by needlessness/contentment.
RIGHT_BRAIN_WILL_EDGES = (
    Association("rb-will-0", SALIENCE_LEAF, INTENTION_LEAF, 1.0, 0.5, 0.5, RIGHT_BRAIN_WILL_VERSION, RIGHT_BRAIN_WILL_VERSION),
    Association("rb-will-1", AROUSAL_LEAF, INTENTION_LEAF, 1.0, 0.5, 0.5, RIGHT_BRAIN_WILL_VERSION, RIGHT_BRAIN_WILL_VERSION),
    Association("rb-will-2", "cultural/efficiency/power", INTENTION_LEAF, 0.8, 0.5, 0.5, RIGHT_BRAIN_WILL_VERSION, RIGHT_BRAIN_WILL_VERSION),
    Association("rb-will-3", "cultural/freedom/needlessness", INTENTION_LEAF, -0.8, 0.5, 0.5, RIGHT_BRAIN_WILL_VERSION, RIGHT_BRAIN_WILL_VERSION),
    Association("rb-will-4", INTENTION_LEAF, EXECUTION_LEAF, 1.0, 0.5, 0.5, RIGHT_BRAIN_WILL_VERSION, RIGHT_BRAIN_WILL_VERSION),
    Association("rb-will-5", FACT_LEAF, EXECUTION_LEAF, 1.0, 0.5, 0.5, RIGHT_BRAIN_WILL_VERSION, RIGHT_BRAIN_WILL_VERSION),
    Association("rb-will-6", THEORY_LEAF, EXECUTION_LEAF, 0.7, 0.5, 0.5, RIGHT_BRAIN_WILL_VERSION, RIGHT_BRAIN_WILL_VERSION),
    Association("rb-will-7", CONFIDENCE_LEAF, EXECUTION_LEAF, 0.5, 0.5, 0.5, RIGHT_BRAIN_WILL_VERSION, RIGHT_BRAIN_WILL_VERSION),
    Association("rb-will-8", "cultural/freedom/needlessness", EXECUTION_LEAF, -0.8, 0.5, 0.5, RIGHT_BRAIN_WILL_VERSION, RIGHT_BRAIN_WILL_VERSION),
)


def _validate_unit_map(
    raw: Mapping[str, float],
    expected_keys: tuple[str, ...],
    label: str,
    *,
    optional_keys: tuple[str, ...] = (),
) -> dict[str, float]:
    raw_keys = set(raw.keys()) if isinstance(raw, Mapping) else set()
    req_keys = set(expected_keys)
    allowed_keys = req_keys | set(optional_keys)
    if not isinstance(raw, Mapping) or not req_keys.issubset(raw_keys) or not raw_keys.issubset(allowed_keys):
        missing = sorted(req_keys - raw_keys)
        extra = sorted(raw_keys - allowed_keys)
        raise ValueError(f"{label} key mismatch (missing={missing}, extra={extra})")
    cleaned: dict[str, float] = {}
    for key in expected_keys + tuple(k for k in optional_keys if k in raw):
        val = raw[key]
        if type(val) not in (int, float) or isinstance(val, bool) or not math.isfinite(val) or not (0.0 <= val <= 1.0):
            raise ValueError(f"{label}[{key!r}] must be a finite float in [0, 1]")
        cleaned[key] = round(float(val), 4)
    return cleaned


@dataclass(frozen=True)
class ThreeBridgeSensors:
    """Three dedicated Right-Brain classifiers decoupled from the Left-Brain LLM.

    - `will_cultural` (The Do-er): scores the 6 Cultural leaves + active `cultural/*` drawers.
    - `truth_philosophical` (The Thinker): scores the 6 Philosophical leaves + active `philosophical/*` drawers.
    - `contemplation_experiential` (The Feeler): scores raw felt `salience`, `valence`, `arousal`,
      plus active `experiential/*` complex-feeling drawers (e.g. bittersweet release, austere sanctuary).
    """

    will_cultural: Callable[[str, tuple[LearnedLeaf, ...]], Mapping[str, float]]
    truth_philosophical: Callable[[str, tuple[LearnedLeaf, ...]], Mapping[str, float]]
    contemplation_experiential: Callable[..., Mapping[str, float]]

    def _call_feeler(self, text: str, children: tuple[LearnedLeaf, ...]) -> Mapping[str, float]:
        sig = inspect.signature(self.contemplation_experiential)
        params = [
            p for p in sig.parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
        if has_varargs or len(params) >= 2:
            return self.contemplation_experiential(text, children)
        if children:
            raise ValueError("contemplation_experiential callable must accept (text, child_leaves) to score experiential/* drawers")
        return self.contemplation_experiential(text)

    def score_text(
        self,
        text: str,
        *,
        learned_leaves: tuple[LearnedLeaf, ...] = (),
        branch_gating: bool = True,
        branch_gate_threshold: float = BRANCH_GATE_THRESHOLD,
    ) -> tuple[dict[str, float], dict[str, float], tuple[str, ...], tuple[str, ...]]:
        """Score text across all 3 Right-Brain bridges with branch-gated child drawers."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a nonempty string")

        exp_raw = _validate_unit_map(
            self._call_feeler(text, ()),
            EXPERIENTIAL_SENSOR_LEAVES,
            "contemplation_experiential(tier1)",
            optional_keys=OPTIONAL_EXPERIENTIAL_SENSOR_LEAVES,
        )
        cult_base = _validate_unit_map(
            self.will_cultural(text, ()),
            WILL_CULTURAL_LEAVES,
            "will_cultural(tier1)",
        )
        phil_base = _validate_unit_map(
            self.truth_philosophical(text, ()),
            TRUTH_PHILOSOPHICAL_LEAVES,
            "truth_philosophical(tier1)",
        )

        coords: dict[str, float] = {**cult_base, **phil_base, **exp_raw}
        queried: list[str] = []
        skipped: list[str] = []

        if learned_leaves:
            for tier_level in (2, 3):
                tier_leaves = tuple(l for l in learned_leaves if l.tier == tier_level)
                if not tier_leaves:
                    continue
                open_cult: list[LearnedLeaf] = []
                open_phil: list[LearnedLeaf] = []
                open_exp: list[LearnedLeaf] = []
                for leaf in tier_leaves:
                    parent_active = True
                    if branch_gating:
                        if tier_level == 2:
                            if leaf.leaf_id.startswith("experiential/"):
                                parent_vals = [
                                    v for k, v in exp_raw.items()
                                    if k.startswith(leaf.parent_branch + "/")
                                ]
                                # Synchronization has no tier-1 raw sensor key yet, so open when salience >= threshold
                                exp_signal = max(parent_vals, default=exp_raw[SALIENCE_LEAF])
                                parent_active = exp_signal >= branch_gate_threshold
                            else:
                                parent_vals = [
                                    v for k, v in coords.items()
                                    if k in FOUNDATIONAL_LEAVES and k.startswith(leaf.parent_branch + "/")
                                ]
                                max_founding = max((coords[k] for k in FOUNDATIONAL_LEAVES), default=0.0)
                                blind_spot_fallback = max_founding < branch_gate_threshold and exp_raw[SALIENCE_LEAF] >= 0.45
                                parent_active = blind_spot_fallback or (max(parent_vals, default=0.0) >= branch_gate_threshold)
                        else:
                            parent_active = float(coords.get(leaf.parent_branch, 0.0)) >= CONFUSION_THRESHOLD
                    if not parent_active:
                        coords[leaf.leaf_id] = 0.0
                        skipped.append(leaf.leaf_id)
                    elif leaf.leaf_id.startswith("cultural/"):
                        open_cult.append(leaf)
                    elif leaf.leaf_id.startswith("experiential/"):
                        open_exp.append(leaf)
                    else:
                        open_phil.append(leaf)

                if open_cult:
                    cult_t = tuple(open_cult)
                    expected = WILL_CULTURAL_LEAVES + tuple(l.leaf_id for l in cult_t)
                    scored = _validate_unit_map(
                        self.will_cultural(text, cult_t),
                        expected,
                        f"will_cultural(tier{tier_level})",
                    )
                    for l in cult_t:
                        coords[l.leaf_id] = scored[l.leaf_id]
                        queried.append(l.leaf_id)

                if open_phil:
                    phil_t = tuple(open_phil)
                    expected = TRUTH_PHILOSOPHICAL_LEAVES + tuple(l.leaf_id for l in phil_t)
                    scored = _validate_unit_map(
                        self.truth_philosophical(text, phil_t),
                        expected,
                        f"truth_philosophical(tier{tier_level})",
                    )
                    for l in phil_t:
                        coords[l.leaf_id] = scored[l.leaf_id]
                        queried.append(l.leaf_id)

                if open_exp:
                    exp_t = tuple(open_exp)
                    expected = EXPERIENTIAL_SENSOR_LEAVES + tuple(l.leaf_id for l in exp_t)
                    scored = _validate_unit_map(
                        self._call_feeler(text, exp_t),
                        expected,
                        f"contemplation_experiential(tier{tier_level})",
                        optional_keys=OPTIONAL_EXPERIENTIAL_SENSOR_LEAVES,
                    )
                    for l in exp_t:
                        coords[l.leaf_id] = scored[l.leaf_id]
                        queried.append(l.leaf_id)

        return coords, exp_raw, tuple(queried), tuple(skipped)


@dataclass(frozen=True)
class BridgeBid:
    """One bridge persona's competing bid (Do-er, Thinker, or Feeler)."""

    persona: str
    stance: str
    strength: float
    summary: str


@dataclass(frozen=True)
class ThreeWaySynthesis:
    """Competing bids from the Do-er, Thinker, and Feeler bridges plus overall harmony."""

    doer: BridgeBid
    thinker: BridgeBid
    feeler: BridgeBid
    harmony: float
    conflicts: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "doer_will_cultural": {
                "stance": self.doer.stance,
                "strength": self.doer.strength,
                "summary": self.doer.summary,
            },
            "thinker_truth_philosophical": {
                "stance": self.thinker.stance,
                "strength": self.thinker.strength,
                "summary": self.thinker.summary,
            },
            "feeler_contemplation_experiential": {
                "stance": self.feeler.stance,
                "strength": self.feeler.strength,
                "summary": self.feeler.summary,
            },
            "harmony": self.harmony,
            "conflicts": list(self.conflicts),
        }


@dataclass(frozen=True)
class InternalSelfRegistry:
    """Operational self-registry separating outside world Fact from internal Left-Brain Theory."""

    architecture_id: str = "lmft-v0.1-40node-right-brain-dominant"
    known_instinct_count: int = 0
    learned_drawer_count: int = 0
    external_source_tag: str = "EXTERNAL_WORLD"
    internal_thought_tag: str = "NONE"
    epistemic_state: str = "QUIET_EQUILIBRIUM"
    dissonance_notes: tuple[str, ...] = ()

    def __post_init__(self):
        if self.epistemic_state not in EPISTEMIC_SELF_STATES:
            raise ValueError(f"invalid epistemic_state: {self.epistemic_state!r}")

    def to_dict(self) -> dict:
        return {
            "architecture_id": self.architecture_id,
            "known_instinct_count": self.known_instinct_count,
            "learned_drawer_count": self.learned_drawer_count,
            "external_source_tag": self.external_source_tag,
            "internal_thought_tag": self.internal_thought_tag,
            "epistemic_state": self.epistemic_state,
            "dissonance_notes": list(self.dissonance_notes),
        }


@dataclass(frozen=True)
class DiaryEntry:
    """One compact settled-vibe record carried forward for Left-Brain continuity."""

    scene_index: int
    outcome: str
    active_concept: str | None
    key_coordinates: Mapping[str, float]
    conscious_note: str | None = None
    epistemic_state: str = "QUIET_EQUILIBRIUM"

    def __post_init__(self):
        if type(self.scene_index) is not int or self.scene_index < 1:
            raise ValueError("scene_index must be a positive int")
        if self.outcome not in SETTLED_OUTCOMES:
            raise ValueError(f"invalid diary outcome: {self.outcome!r}")
        if self.epistemic_state not in EPISTEMIC_SELF_STATES:
            raise ValueError(f"invalid diary epistemic_state: {self.epistemic_state!r}")
        if self.conscious_note is not None and (not isinstance(self.conscious_note, str) or not self.conscious_note.strip()):
            raise ValueError("conscious_note must be None or a nonempty string")
        for k, v in self.key_coordinates.items():
            if not is_valid_leaf_id(k):
                raise ValueError(f"invalid diary leaf_id: {k!r}")
            if type(v) not in (int, float) or not math.isfinite(v) or not (0.0 <= v <= 1.0):
                raise ValueError(f"diary coordinate {k!r} must be in [0, 1]")

    def to_dict(self) -> dict:
        return {
            "scene_index": self.scene_index,
            "outcome": self.outcome,
            "active_concept": self.active_concept,
            "key_coordinates": dict(self.key_coordinates),
            "conscious_note": self.conscious_note,
            "epistemic_state": self.epistemic_state,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DiaryEntry":
        return cls(
            scene_index=data["scene_index"],
            outcome=data["outcome"],
            active_concept=data.get("active_concept"),
            key_coordinates=dict(data.get("key_coordinates", {})),
            conscious_note=data.get("conscious_note"),
            epistemic_state=data.get("epistemic_state", "QUIET_EQUILIBRIUM"),
        )


@dataclass(frozen=True)
class SettledDiary:
    """Bounded rolling diary of settled scenes fed into the Left Brain when tugged."""

    entries: tuple[DiaryEntry, ...] = ()
    max_entries: int = 5

    def __post_init__(self):
        if type(self.max_entries) is not int or self.max_entries < 1:
            raise ValueError("max_entries must be a positive int")

    def append(self, entry: DiaryEntry) -> "SettledDiary":
        kept = (self.entries + (entry,))[-self.max_entries:]
        return SettledDiary(entries=kept, max_entries=self.max_entries)

    def format_for_left_brain(self) -> list[dict]:
        items = []
        for e in self.entries:
            row = {
                "scene": e.scene_index,
                "posture": e.outcome,
                "self_status": e.epistemic_state,
                "dominant_signals": {k.split("/")[-1]: v for k, v in e.key_coordinates.items()},
            }
            if e.active_concept:
                row["instinct"] = e.active_concept
            if e.conscious_note:
                first_clause = e.conscious_note.split(".")[0].strip()
                row["brief_gist"] = (first_clause[:110] + "...") if len(first_clause) > 110 else first_clause
            items.append(row)
        return items


def save_diary(path: str | Path, diary: SettledDiary) -> None:
    payload = {
        "schema_version": 1,
        "max_entries": diary.max_entries,
        "entries": [e.to_dict() for e in diary.entries],
    }
    atomic_json(Path(path), payload)


def load_diary(path: str | Path) -> SettledDiary:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("unsupported diary schema_version")
    return SettledDiary(
        entries=tuple(DiaryEntry.from_dict(d) for d in data.get("entries", ())),
        max_entries=int(data.get("max_entries", 5)),
    )


@dataclass(frozen=True)
class RightBrainVibe:
    """Compiled 24-leaf Right-Brain snapshot separating external Fact from internal Theory."""

    pass_index: int
    posture: str
    active_concept: str | None
    tug_reasons: tuple[str, ...]
    coordinates: dict[str, float]
    theory_coordinates: dict[str, float]
    rule_activations: dict[str, float]
    theory_rule_activations: dict[str, float]
    synthesis: ThreeWaySynthesis
    self_registry: InternalSelfRegistry
    novelty: float
    salience: float
    valence: float
    arousal: float
    resonance: float
    retention: float
    clarity: float
    confidence: float
    theory: float
    fact: float
    intention: float
    execution: float
    propagation: PropagationResult
    queried_child_leaves: tuple[str, ...] = ()
    skipped_child_leaves: tuple[str, ...] = ()
    felt_chord: FeltChord | None = None
    sibling_color_balances: tuple[SiblingColorBalance, ...] = ()
    shadow_pathology_warnings: tuple[ShadowPathologyWarning, ...] = ()

    def __post_init__(self):
        if self.posture not in RIGHT_BRAIN_POSTURES:
            raise ValueError(f"invalid Right-Brain posture: {self.posture!r}")

    @property
    def entrainment(self) -> float:
        """Synchronization -> Entrainment leaf activation (alias for retention field)."""
        return self.retention


def _detect_structural_value_tensions(coords: Mapping[str, float]) -> list[str]:
    """Detect general cross-branch polarity clashes between high control/authority and depressed reciprocity."""
    tensions: list[str] = []
    chaos = float(coords.get("philosophical/structure/chaos", 0.0))
    if chaos >= 0.45:
        return tensions

    control_axes = (
        ("philosophical/structure/order", 0.60, 0.25),
        ("cultural/efficiency/power", 0.65, 0.30),
        ("cultural/justice/honor", 0.60, 0.25),
    )
    reciprocity_val = float(coords.get("cultural/justice/fairness", 0.50))

    for ctrl_leaf, ctrl_min, low_thresh in control_axes:
        ctrl_val = float(coords.get(ctrl_leaf, 0.0))
        if ctrl_val >= ctrl_min and reciprocity_val <= low_thresh:
            c_short = ctrl_leaf.split("/")[-1]
            tensions.append(
                f"polarity_clash({c_short}={ctrl_val:.2f}, fairness={reciprocity_val:.2f})"
            )
    return tensions


def _compile_three_way_synthesis(
    external_coords: Mapping[str, float],
    salience: float,
    valence: float,
    arousal: float,
    resonance: float,
    fact: float,
    theory: float,
    clarity: float,
    novelty: float,
    intention: float,
    execution: float,
    raw_tensions: list[str],
    dissonance_notes: list[str],
    *,
    matched_ratified_chord: bool = False,
) -> ThreeWaySynthesis:
    """Evaluate competing Do-er, Thinker, and Feeler bids and their harmony."""
    power = float(external_coords.get("cultural/efficiency/power", 0.0))
    fairness = float(external_coords.get("cultural/justice/fairness", 0.0))

    if raw_tensions:
        doer_stance = "CONTEST_EXCLUSION"
        doer_strength = round(min(1.0, max(intention, power * (1.0 - fairness), 0.55)), 4)
        doer_summary = f"Value/power friction active ({', '.join(raw_tensions)})."
    elif execution >= ACTION_EXECUTION_THRESHOLD:
        doer_stance = "ACT_DECISIVELY"
        doer_strength = round(execution, 4)
        doer_summary = f"Action drive engaged (execution={execution:.2f}, intention={intention:.2f})."
    else:
        doer_stance = "HOLD_STEADY"
        doer_strength = round(max(0.0, 1.0 - intention), 4)
        doer_summary = f"No urgent intervention needed (execution={execution:.2f})."

    if fact >= CONFUSION_THRESHOLD and clarity >= CONFUSION_THRESHOLD and (not raw_tensions or matched_ratified_chord):
        thinker_stance = "VERIFIED_EXTERNAL_FACT"
        thinker_strength = round(fact, 4)
        thinker_summary = f"Outside world directly matches a known instinct (fact={fact:.2f})."
    elif (
        theory >= CONFUSION_THRESHOLD
        and clarity >= CONFUSION_THRESHOLD
        and not dissonance_notes
        and (not raw_tensions or matched_ratified_chord)
    ):
        thinker_stance = "CANDIDATE_THEORY_FIT"
        thinker_strength = round(theory, 4)
        thinker_summary = f"Left-Brain Theory hypothesis resolves ambiguous context (theory={theory:.2f}, fact={fact:.2f})."
    elif novelty >= 0.50:
        thinker_stance = "UNRECOGNIZED_STRUCTURE"
        thinker_strength = round(novelty, 4)
        thinker_summary = f"No current rule fits the structure (novelty={novelty:.2f})."
    else:
        thinker_stance = "AMBIGUOUS_RULE_TIE"
        thinker_strength = round(1.0 - clarity, 4)
        thinker_summary = f"Competing rules overlap without clear winner (clarity={clarity:.2f})."

    if salience < SILENCE_SALIENCE_THRESHOLD and arousal < 0.50:
        feeler_stance = "CALM_ABSORPTION"
        feeler_strength = round(1.0 - salience, 4)
        feeler_summary = f"Low-salience background flow (salience={salience:.2f}, valence={valence:.2f})."
    elif valence <= 0.35 and salience >= SILENCE_SALIENCE_THRESHOLD:
        feeler_stance = "FELT_UNEASE"
        feeler_strength = round(min(1.0, salience * (1.0 - valence) + 0.25), 4)
        feeler_summary = f"High-salience low-valence discomfort (valence={valence:.2f}, arousal={arousal:.2f})."
    else:
        feeler_stance = "ENGAGED_RESONANCE"
        feeler_strength = round(min(1.0, 0.5 * valence + 0.5 * max(resonance, salience)), 4)
        feeler_summary = f"Constructive or resonant engagement (valence={valence:.2f}, resonance={resonance:.2f})."

    conflicts: list[str] = []
    if doer_stance == "CONTEST_EXCLUSION" and thinker_stance in ("UNRECOGNIZED_STRUCTURE", "AMBIGUOUS_RULE_TIE"):
        conflicts.append("doer_contests_exclusion_without_matching_thinker_rule")
    if feeler_stance == "FELT_UNEASE" and thinker_stance in ("UNRECOGNIZED_STRUCTURE", "AMBIGUOUS_RULE_TIE"):
        conflicts.append("feeler_unease_unresolved_by_thinker")
    if dissonance_notes:
        conflicts.extend(dissonance_notes)

    if conflicts:
        harmony = round(max(0.0, 0.35 - 0.10 * len(conflicts)), 4)
    elif thinker_stance == "CANDIDATE_THEORY_FIT" and feeler_stance == "FELT_UNEASE" and not matched_ratified_chord:
        conflicts.append("feeler_unease_vetoes_tone_deaf_theory")
        harmony = 0.25
    elif thinker_stance in ("VERIFIED_EXTERNAL_FACT", "CANDIDATE_THEORY_FIT"):
        harmony = round(min(1.0, 0.5 * clarity + 0.5 * max(fact, theory)), 4)
    elif feeler_stance == "CALM_ABSORPTION" and doer_stance == "HOLD_STEADY":
        harmony = 0.85
    else:
        harmony = round(0.5 * clarity + 0.25 * (1.0 - novelty), 4)

    return ThreeWaySynthesis(
        doer=BridgeBid("DOER_WILL_CULTURAL", doer_stance, doer_strength, doer_summary),
        thinker=BridgeBid("THINKER_TRUTH_PHILOSOPHICAL", thinker_stance, thinker_strength, thinker_summary),
        feeler=BridgeBid("FEELER_CONTEMPLATION_EXPERIENTIAL", feeler_stance, feeler_strength, feeler_summary),
        harmony=harmony,
        conflicts=tuple(conflicts),
    )


def compile_right_brain_vibe(
    birth_state: BirthState,
    external_coords: Mapping[str, float],
    external_experiential: Mapping[str, float],
    rules: tuple[FuzzyConceptRule, ...],
    *,
    pass_index: int,
    event_id: str,
    prior_memory: ContinuousMemoryState | None = None,
    learned_leaves: tuple[LearnedLeaf, ...] = (),
    internal_theory_coords: Mapping[str, float] | None = None,
    internal_theory_experiential: Mapping[str, float] | None = None,
    queried_child_leaves: tuple[str, ...] = (),
    skipped_child_leaves: tuple[str, ...] = (),
    silence_salience_threshold: float = SILENCE_SALIENCE_THRESHOLD,
    social_drift: SocialDriftReport | None = None,
    ratified_chords: tuple[FuzzyConceptRule, ...] = (),
) -> tuple[RightBrainVibe, ContinuousMemoryState]:
    """Compile all 24 founding leaves while keeping external Fact separate from internal Theory."""
    locked_external = dict(external_coords)
    raw_rule_activations = _evaluate_rules_with_inhibition(rules, locked_external)

    salience = round(float(external_experiential[SALIENCE_LEAF]), 6)
    raw_valence = float(external_experiential[VALENCE_LEAF])
    arousal = round(float(external_experiential[AROUSAL_LEAF]), 6)

    memory_coords = dict(locked_external)
    retention = round(float(external_experiential.get(RETENTION_LEAF, 0.0)), 6)
    resonance = round(float(external_experiential.get(RESONANCE_LEAF, 0.0)), 6)

    if prior_memory is not None and prior_memory.retained_coordinates:
        mem_retention = round(min(1.0, max(prior_memory.retained_coordinates.values(), default=0.0)), 6)
        retention = (
            round(0.50 * retention + 0.50 * mem_retention, 6)
            if RETENTION_LEAF in external_experiential
            else mem_retention
        )
        decay = prior_memory.retention_decay
        prev_unscaled = {
            k: min(1.0, v / decay)
            for k, v in prior_memory.retained_coordinates.items()
        }
        active_keys = [
            k for k in locked_external
            if float(locked_external.get(k, 0.0)) >= 0.20 or float(prev_unscaled.get(k, 0.0)) >= 0.20
        ]
        if active_keys:
            base_sim = sum(
                1.0 - abs(float(locked_external.get(k, 0.0)) - float(prev_unscaled.get(k, 0.0)))
                for k in active_keys
            ) / len(active_keys)
        else:
            base_sim = 0.50
        new_spike_penalty = max(
            (
                float(locked_external.get(k, 0.0)) - float(prev_unscaled.get(k, 0.0))
                for k in locked_external
                if k != "philosophical/persistence/continuity"
            ),
            default=0.0,
        )
        continuity_bonus = 0.15 * float(locked_external.get("philosophical/persistence/continuity", 0.0))
        mem_resonance = round(min(1.0, max(0.0, base_sim - 0.65 * max(0.0, new_spike_penalty - 0.20) + continuity_bonus)), 6)
        resonance = (
            round(0.50 * resonance + 0.50 * mem_resonance, 6)
            if RESONANCE_LEAF in external_experiential
            else mem_resonance
        )

        raw_max = max(raw_rule_activations.values(), default=0.0)
        if (
            raw_max < CONFUSION_THRESHOLD
            and resonance >= 0.55
            and 0.35 <= salience < 0.70
            and float(locked_external.get("philosophical/persistence/continuity", 0.0)) >= 0.35
        ):
            for k, prev_val in prev_unscaled.items():
                if k in memory_coords:
                    cur_val = float(locked_external.get(k, 0.0))
                    memory_coords[k] = round(max(cur_val, 0.20 * cur_val + 0.80 * prev_val), 4)

    # Fact comes STRICTLY from external_coords (outside world only—never Left-Brain words!)
    raw_clarity, raw_confidence, raw_matched, raw_confused, raw_reason = compute_contemplation(raw_rule_activations)
    fact = (
        round(min(raw_confidence, raw_clarity), 6)
        if not raw_confused
        else round(0.30 * min(raw_confidence, raw_clarity), 6)
    )

    raw_tensions = _detect_structural_value_tensions(locked_external)
    if social_drift is not None and social_drift.drift_detected:
        raw_tensions.append("social_drift_ongoing_enclosure")

    dissonance_notes: list[str] = []
    theory_candidate_coords = dict(memory_coords)

    if internal_theory_coords is not None:
        theory_warmer = bool(
            internal_theory_experiential
            and float(internal_theory_experiential[VALENCE_LEAF]) > raw_valence
        )
        for k, l_raw in internal_theory_coords.items():
            r_val = float(locked_external.get(k, 0.0))
            l_val = float(l_raw)
            if (
                raw_tensions
                and k in WILL_CULTURAL_LEAVES
                and r_val <= 0.25
                and (l_val - r_val) >= 0.45
            ):
                dissonance_notes.append(
                    f"cognitive_dissonance({k.split('/')[-1]}:external={r_val:.2f},theory={l_val:.2f})"
                )
            elif not raw_tensions and k in CONSTRUCTIVE_THEORY_LEAVES:
                # Theory can lift under-specified constructive context without dragging down raw facts
                # when the Left Brain mentions past diary conflicts.
                lifted = round(0.40 * r_val + 0.60 * l_val, 4)
                theory_candidate_coords[k] = max(float(memory_coords.get(k, 0.0)), lifted)
            elif not raw_tensions and k in FOUNDATIONAL_LEAVES and theory_warmer:
                theory_candidate_coords[k] = round(0.50 * r_val + 0.50 * l_val, 4)
            elif k not in FOUNDATIONAL_LEAVES and not dissonance_notes:
                theory_candidate_coords[k] = max(
                    float(theory_candidate_coords.get(k, 0.0)),
                    round(0.50 * r_val + 0.50 * l_val, 4),
                )

        if fact <= 0.30 and not theory_warmer:
            ext_pre_shadows = {
                w.shadow_leaf_id
                for w in detect_shadow_pathologies(
                    locked_external,
                    fact=fact,
                    arousal=arousal,
                    resonance=resonance,
                )
            }
            th_shadows = detect_shadow_pathologies(
                internal_theory_coords,
                fact=fact,
                arousal=arousal,
                resonance=resonance,
            )
            for th_w in th_shadows:
                ext_val = float(locked_external.get(th_w.leaf_id, 0.0))
                if (
                    th_w.shadow_leaf_id not in ext_pre_shadows
                    and ext_val <= 0.55
                    and (th_w.leaf_activation - ext_val) >= 0.35
                ):
                    dissonance_notes.append(
                        f"shadow_pathology_dissonance({th_w.shadow_pathology.lower()}:{th_w.trigger_reason})"
                    )

    ratified_chord_names: set[str] = set()
    tension_resolving_chord_names: set[str] = set()
    active_theory_rules = list(rules)
    if ratified_chords:
        existing_names = {r.concept_name for r in active_theory_rules}
        for rc in ratified_chords:
            ratified_chord_names.add(rc.concept_name)
            has_low_or_specific_target = any(
                float(t_val) <= 0.35 or (leaf_tier(t_key) >= 2 and float(t_val) >= 0.40)
                for t_key, t_val in rc.target_leaves.items()
            )
            if has_low_or_specific_target:
                tension_resolving_chord_names.add(rc.concept_name)
            if internal_theory_coords is not None:
                for req_leaf in rc.target_leaves:
                    if req_leaf not in theory_candidate_coords:
                        theory_candidate_coords[req_leaf] = float(locked_external.get(req_leaf, 0.0))
                if rc.concept_name not in existing_names:
                    active_theory_rules.append(rc)
                    existing_names.add(rc.concept_name)

    theory_rule_activations = _evaluate_rules_with_inhibition(tuple(active_theory_rules), theory_candidate_coords)

    th_clarity, th_confidence, th_matched, th_confused, th_reason = compute_contemplation(theory_rule_activations)
    allowed_chord_names = tension_resolving_chord_names if raw_tensions else ratified_chord_names
    matched_ratified_chord = bool(
        not dissonance_notes
        and (
            (not raw_confused and raw_matched in allowed_chord_names)
            or (internal_theory_coords is not None and not th_confused and th_matched in allowed_chord_names)
        )
    )

    if dissonance_notes:
        # Veto rationalizing theories that contradict stark raw sensory deficits
        theory = 0.0
        clarity = raw_clarity
        confidence = raw_confidence
        matched_concept = None
        is_confused = True
        confusion_reason = raw_reason
        valence = round(raw_valence, 6)
    elif not raw_confused:
        # External Fact already proved a match on its own
        partial_hyp = round(raw_confidence * (1.0 - fact), 6) if raw_clarity >= 0.35 else 0.0
        theory = round(min(1.0, partial_hyp), 6)
        clarity = raw_clarity
        confidence = raw_confidence
        matched_concept = raw_matched
        is_confused = False
        confusion_reason = "clear_match"
        valence = round(raw_valence, 6)
    elif not th_confused and (not raw_tensions or matched_ratified_chord):
        # Left-Brain Theory / 3-Branch Deliberation resolved an ambiguous or ratified scene!
        # Fact stays honestly low while Theory carries the synthesized match.
        theory_gain = max(th_confidence - raw_confidence, min(th_confidence, th_clarity))
        theory = round(min(1.0, max(0.0, theory_gain)), 6)
        clarity = th_clarity
        confidence = th_confidence
        matched_concept = th_matched
        is_confused = False
        confusion_reason = "theory_synthesis_match"
        if internal_theory_experiential is not None and not raw_tensions:
            th_val = float(internal_theory_experiential[VALENCE_LEAF])
            valence = round(max(raw_valence, 0.45 * raw_valence + 0.55 * th_val), 6)
        else:
            valence = round(raw_valence, 6)
    else:
        theory = round(min(1.0, max(0.0, th_confidence - raw_confidence)), 6)
        clarity = th_clarity
        confidence = th_confidence
        matched_concept = None
        is_confused = True
        confusion_reason = th_reason if internal_theory_coords is not None else raw_reason
        valence = round(raw_valence, 6)

    max_rule = max(
        max(raw_rule_activations.values(), default=0.0),
        0.0 if dissonance_notes else max(theory_rule_activations.values(), default=0.0),
    )
    rule_novelty = round(min(1.0, max(0.0, 1.0 - max_rule)), 6)
    novelty = (
        round(0.50 * float(external_experiential[NOVELTY_LEAF]) + 0.50 * rule_novelty, 6)
        if NOVELTY_LEAF in external_experiential
        else rule_novelty
    )

    matched_rule_obj = next(
        (r for r in active_theory_rules if r.concept_name == matched_concept),
        None,
    )
    repulsive_rule_matched = bool(
        matched_rule_obj is not None and getattr(matched_rule_obj, "polarity", "attractive") == "repulsive"
    )

    # Founding tree propagation uses locked external coordinates + separate Fact & Theory leaves
    founding_external = {k: v for k, v in locked_external.items() if k in FOUNDATIONAL_LEAVES}
    proposed_memberships = {
        **founding_external,
        VALENCE_LEAF: valence,
        AROUSAL_LEAF: arousal,
        RESONANCE_LEAF: resonance,
        RETENTION_LEAF: retention,
        NOVELTY_LEAF: novelty,
        SALIENCE_LEAF: salience,
        CLARITY_LEAF: clarity,
        CONFIDENCE_LEAF: confidence,
        THEORY_LEAF: theory,
        FACT_LEAF: fact,
        INTENTION_LEAF: 0.0,
        EXECUTION_LEAF: 0.0,
    }
    propagation = propagate_activation(
        birth_state,
        proposed_memberships,
        RIGHT_BRAIN_WILL_EDGES,
        event_id=event_id,
        source_instance=f"pass-{pass_index}",
        relationship_source=RIGHT_BRAIN_WILL_VERSION,
        config=PropagationConfig(passes=2, gain=0.25),
        source="developer",
    )
    final_act = propagation.activations
    intention = round(final_act[INTENTION_LEAF], 6)
    execution = round(final_act[EXECUTION_LEAF], 6)
    if repulsive_rule_matched:
        # Repulsive / hazard rules act as a brake pedal on blind execution rather than a gas pedal
        execution = round(max(0.0, execution - 0.50 * confidence), 6)

    synthesis = _compile_three_way_synthesis(
        locked_external,
        salience,
        valence,
        arousal,
        resonance,
        fact,
        theory,
        clarity,
        novelty,
        intention,
        execution,
        raw_tensions,
        dissonance_notes,
        matched_ratified_chord=matched_ratified_chord,
    )

    chaos = float(locked_external.get("philosophical/structure/chaos", 0.0))
    learned_active = any(
        l.tier >= 2 and float(locked_external.get(l.leaf_id, 0.0)) >= CONFUSION_THRESHOLD
        for l in learned_leaves
    )
    urgent_drive = chaos >= 0.45 or learned_active or intention >= 0.25

    tug_reasons: list[str] = []
    if repulsive_rule_matched and matched_concept:
        tug_reasons.append(f"hazard_rule_warning({matched_concept})")
    if raw_tensions and not matched_ratified_chord and (is_confused or salience >= silence_salience_threshold):
        tug_reasons.extend(raw_tensions)
    if is_confused and (salience >= silence_salience_threshold or urgent_drive):
        tug_reasons.append(confusion_reason)
    if "feeler_unease_vetoes_tone_deaf_theory" in synthesis.conflicts:
        tug_reasons.append("feeler_unease_vetoes_tone_deaf_theory")
    if dissonance_notes:
        tug_reasons.extend(dissonance_notes)
    tug_reasons = list(dict.fromkeys(tug_reasons))

    if tug_reasons:
        posture = "TUG_LEFT_BRAIN"
        active_concept = None if dissonance_notes else matched_concept
        epistemic_state = (
            "SELF_RATIONALIZATION_DISSONANCE"
            if dissonance_notes
            else "KNOWN_INSTINCT_BLIND_SPOT"
        )
    elif not raw_confused and execution >= ACTION_EXECUTION_THRESHOLD and (urgent_drive or salience >= silence_salience_threshold):
        posture = "WORDLESS_REFLEX"
        active_concept = matched_concept
        epistemic_state = "GROUNDED_EXTERNAL_FACT"
    elif raw_confused and not is_confused and theory >= CONFUSION_THRESHOLD:
        posture = "THEORY_SYNTHESIS"
        active_concept = matched_concept
        epistemic_state = "THEORY_GUIDED_SYNTHESIS"
    else:
        posture = "AT_EASE"
        active_concept = matched_concept
        epistemic_state = (
            "GROUNDED_EXTERNAL_FACT"
            if (not raw_confused and salience >= silence_salience_threshold)
            else "QUIET_EQUILIBRIUM"
        )

    self_registry = InternalSelfRegistry(
        known_instinct_count=len(rules),
        learned_drawer_count=len(learned_leaves),
        external_source_tag="EXTERNAL_WORLD",
        internal_thought_tag="INTERNAL_LEFT_BRAIN_THEORY" if internal_theory_coords is not None else "NONE",
        epistemic_state=epistemic_state,
        dissonance_notes=tuple(dissonance_notes),
    )

    decay = prior_memory.retention_decay if prior_memory is not None else 0.65
    # Memory retains external world coordinates (never polluted by Left-Brain rationalizations)
    next_memory = ContinuousMemoryState(
        step_count=(prior_memory.step_count + 1) if prior_memory is not None else 1,
        retained_coordinates={k: round(float(v) * decay, 4) for k, v in locked_external.items()},
        last_matched_concept=active_concept,
        retention_decay=decay,
    )

    felt_chord = compile_felt_chord(
        locked_external,
        salience=salience,
        valence=valence,
        arousal=arousal,
        resonance=resonance,
        learned_leaves=learned_leaves,
    )

    sibling_color_balances = compute_philosophical_color_harmony(locked_external)
    shadow_warnings = detect_shadow_pathologies(
        locked_external,
        fact=fact,
        arousal=arousal,
        resonance=resonance,
        novelty=novelty,
    )
    vibe_coords = dict(locked_external)
    for w in shadow_warnings:
        if w.shadow_leaf_id:
            vibe_coords[w.shadow_leaf_id] = max(
                float(vibe_coords.get(w.shadow_leaf_id, 0.0)),
                float(w.severity),
            )

    theory_coords_out = dict(internal_theory_coords) if internal_theory_coords is not None else {}
    if internal_theory_coords is not None:
        for th_w in detect_shadow_pathologies(
            internal_theory_coords,
            fact=fact,
            arousal=arousal,
            resonance=resonance,
            novelty=novelty,
        ):
            if th_w.shadow_leaf_id:
                theory_coords_out[th_w.shadow_leaf_id] = max(
                    float(theory_coords_out.get(th_w.shadow_leaf_id, 0.0)),
                    float(th_w.severity),
                )

    vibe = RightBrainVibe(
        pass_index=pass_index,
        posture=posture,
        active_concept=active_concept,
        tug_reasons=tuple(tug_reasons),
        coordinates=vibe_coords,
        theory_coordinates=theory_coords_out,
        rule_activations=raw_rule_activations,
        theory_rule_activations=theory_rule_activations,
        synthesis=synthesis,
        self_registry=self_registry,
        novelty=novelty,
        salience=salience,
        valence=valence,
        arousal=arousal,
        resonance=resonance,
        retention=retention,
        clarity=clarity,
        confidence=confidence,
        theory=theory,
        fact=fact,
        intention=intention,
        execution=execution,
        propagation=propagation,
        queried_child_leaves=queried_child_leaves,
        skipped_child_leaves=skipped_child_leaves,
        felt_chord=felt_chord,
        sibling_color_balances=sibling_color_balances,
        shadow_pathology_warnings=shadow_warnings,
    )
    return vibe, next_memory


@dataclass(frozen=True)
class TacticalHail:
    """Commander hail from Will to the Left Brain carrying pure Felt Chords and Memory Echoes."""

    pass_index: int
    commander_directive: str
    felt_chord: FeltChord
    chord_echoes: ChordEchoReport
    contemplation_ledger: ContemplationLedger
    prior_self_talk_hunch: str | None = None
    subliminal_margin_tag: SubliminalMarginTag | None = None

    def to_dict(self) -> dict:
        return {
            "pass_index": self.pass_index,
            "commander_directive": self.commander_directive,
            "felt_chord": self.felt_chord.to_dict(),
            "chord_echoes": self.chord_echoes.to_dict(),
            "contemplation_ledger": self.contemplation_ledger.to_dict(),
            "prior_self_talk_hunch": self.prior_self_talk_hunch,
            "subliminal_margin_tag": (
                self.subliminal_margin_tag.to_dict() if self.subliminal_margin_tag else None
            ),
        }

    def render_for_left_brain(self, scenario_text: str) -> str:
        """Render clean human-readable tactical prompt with zero dry bureaucratic debug labels."""
        blocks = [
            f"CURRENT SCENE:\n{scenario_text}",
            f"TACTICAL HAIL:\n{self.commander_directive}",
            f"FELT CHORD:\n{self.felt_chord.render_for_left_brain()}",
            f"PRE-VERBAL MEMORY ECHO:\n{self.chord_echoes.render_for_left_brain()}",
            f"CONTEMPLATION LEDGER:\n{self.contemplation_ledger.render_for_left_brain()}",
        ]
        if self.prior_self_talk_hunch is not None:
            revision_lines = [
                "YOUR PRIOR HUNCH ON THIS SCENE (debate and revise—do not repeat verbatim):",
                f"\"{self.prior_self_talk_hunch}\"",
            ]
            if self.subliminal_margin_tag is not None:
                revision_lines.append(self.subliminal_margin_tag.render_for_left_brain())
            blocks.append("\n".join(revision_lines))
        elif self.subliminal_margin_tag is not None:
            blocks.append(f"SUBLIMINAL MARGIN STAMP:\n{self.subliminal_margin_tag.render_for_left_brain()}")
        return "\n\n".join(blocks)


def _make_margin_tag_for_vibe(
    vibe: RightBrainVibe,
    *,
    prior_hunch_exists: bool,
) -> SubliminalMarginTag:
    fc = vibe.felt_chord or compile_felt_chord(
        vibe.coordinates,
        salience=vibe.salience,
        valence=vibe.valence,
        arousal=vibe.arousal,
        resonance=vibe.resonance,
    )
    has_dissonance = bool(vibe.self_registry.dissonance_notes)
    has_feeler_veto = "feeler_unease_vetoes_tone_deaf_theory" in vibe.synthesis.conflicts
    has_exclusion = vibe.synthesis.doer.stance == "CONTEST_EXCLUSION"
    return build_subliminal_margin_tag(
        clarity=vibe.clarity,
        confidence=vibe.confidence,
        felt_chord=fc,
        has_dissonance=has_dissonance,
        has_feeler_veto=has_feeler_veto,
        has_exclusion_tension=has_exclusion,
        prior_hunch_exists=prior_hunch_exists,
    )


def build_tactical_hail(
    vibe: RightBrainVibe,
    *,
    chord_echoes: ChordEchoReport | None = None,
    contemplation_ledger: ContemplationLedger | None = None,
    prior_self_talk_hunch: str | None = None,
    subliminal_margin_tag: SubliminalMarginTag | None = None,
) -> TacticalHail:
    """Build Will's explicit Tactical Hail carrying pure Felt Chords and Memory Echoes."""
    fc = vibe.felt_chord or compile_felt_chord(
        vibe.coordinates,
        salience=vibe.salience,
        valence=vibe.valence,
        arousal=vibe.arousal,
        resonance=vibe.resonance,
    )
    echoes = chord_echoes if chord_echoes is not None else ChordEchoReport()
    ledger = contemplation_ledger if contemplation_ledger is not None else ContemplationLedger()
    margin = (
        subliminal_margin_tag
        if subliminal_margin_tag is not None
        else _make_margin_tag_for_vibe(vibe, prior_hunch_exists=prior_self_talk_hunch is not None)
    )

    if prior_self_talk_hunch is not None:
        directive = (
            "Your first hunch did not settle the room's tension. Question your previous reading "
            "against the Red-Pen Margin Note and Felt Chord below, then offer a sharper 2-sentence revision."
        )
    elif vibe.synthesis.doer.stance == "CONTEST_EXCLUSION":
        directive = (
            "High structural control or force is colliding with a depressed value coordinate. "
            "Weigh the Felt Chord and Memory Echo below, then articulate the concrete tension in this scene."
        )
    elif vibe.synthesis.feeler.stance == "FELT_UNEASE":
        directive = (
            "A heavy chill in the room refuses to settle into routine. Read the concrete scene alongside "
            "the Felt Chord and Memory Echo, then state in 2 grounded sentences what shifted here."
        )
    else:
        directive = (
            "Our fast instincts do not harmonize on this moment alone. Use the Felt Chord and Memory Echo "
            "as perspective to articulate a grounded 2-sentence theory of what is unfolding right now."
        )

    return TacticalHail(
        pass_index=vibe.pass_index,
        commander_directive=directive,
        felt_chord=fc,
        chord_echoes=echoes,
        contemplation_ledger=ledger,
        prior_self_talk_hunch=prior_self_talk_hunch,
        subliminal_margin_tag=margin,
    )


def build_will_tug_message(
    vibe: RightBrainVibe,
    diary: SettledDiary,
    *,
    prior_reply: str | None = None,
    tactical_hail: TacticalHail | None = None,
) -> dict:
    """Outbound message authored by Will (with Tactical Hail + telemetry for internal audit)."""
    hail = tactical_hail if tactical_hail is not None else build_tactical_hail(
        vibe,
        prior_self_talk_hunch=prior_reply,
    )
    strong_external = {
        k: v for k, v in sorted(vibe.coordinates.items(), key=lambda kv: (-kv[1], kv[0]))
        if v >= 0.50
    }
    contrasts = {
        k: v
        for k, v in vibe.coordinates.items()
        if k in FOUNDATIONAL_LEAVES and float(v) <= 0.30 and strong_external
    }

    packet = {
        "pass": vibe.pass_index,
        "sender_branch": "psychological/will",
        "target_channel": "psychological/truth/theory",
        "tactical_hail": hail,
        "will_opinion": {
            "stance": vibe.synthesis.doer.stance,
            "intention": round(vibe.intention, 3),
            "execution": round(vibe.execution, 3),
            "summary": vibe.synthesis.doer.summary,
        },
        "interjected_emotion": {
            "stance": vibe.synthesis.feeler.stance,
            "valence": round(vibe.valence, 3),
            "arousal": round(vibe.arousal, 3),
            "resonance": round(vibe.resonance, 3),
            "summary": vibe.synthesis.feeler.summary,
        },
        "competing_bids": vibe.synthesis.to_dict(),
        "self_registry": vibe.self_registry.to_dict(),
        "unresolved_tensions": list(vibe.tug_reasons),
        "external_world_signals": strong_external,
        "contrasting_low_signals": contrasts,
        "settled_diary": diary.format_for_left_brain(),
    }
    if prior_reply is not None:
        packet["previous_theory_proposal"] = prior_reply
    return packet


# Backward-compatible alias
build_non_leading_tug_packet = build_will_tug_message


def validate_plain_english_left_brain_reply(raw: str) -> str:
    """Validate that the Left Brain replied in plain English, not a JSON score dump."""
    if not isinstance(raw, str):
        raise ModelError("Left Brain returned non-string output")
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if not cleaned:
        raise ModelError("Left Brain returned empty reply")
    if cleaned.startswith("{") and cleaned.endswith("}"):
        raise ModelError("Left Brain returned JSON instead of plain-English thought")
    if len(cleaned) > 3000:
        cleaned = cleaned[:3000].strip()
    return cleaned


class QwenLeftBrain:
    """Qwen3-4B acting strictly as the plain-English conscious Left Brain feeding Truth/Theory."""

    def __init__(self, models, *, role: str = "contemplation", max_tokens: int = 160):
        self.models = models
        self.role = role
        self.max_tokens = max_tokens

    def build_prompt_messages(
        self,
        scenario_text: str,
        tug_packet_or_hail: Mapping | TacticalHail,
    ) -> list[dict[str, str]]:
        """Build pure human-readable Left-Brain messages stripped of dry debug tokens."""
        if isinstance(tug_packet_or_hail, TacticalHail):
            hail = tug_packet_or_hail
        elif isinstance(tug_packet_or_hail, Mapping) and isinstance(
            tug_packet_or_hail.get("tactical_hail"), TacticalHail
        ):
            hail = tug_packet_or_hail["tactical_hail"]
        else:
            raise TypeError("QwenLeftBrain requires a TacticalHail or packet containing tactical_hail")

        system = (
            "You are the conscious Left Brain answering a Tactical Hail from your Right Brain. "
            "Read CURRENT SCENE alongside the Felt Chord, any Pre-Verbal Memory Echo, and your "
            "Contemplation Ledger. If a prior hunch and Red-Pen Margin Note appear, actively "
            "debate and revise that earlier hunch rather than repeating it. Write 2 concise, "
            "grounded plain-English sentences about what is concretely happening in CURRENT SCENE "
            "and what it means for the people or creatures involved. Never copy wording from "
            "earlier ledger entries, and never output JSON, bullet lists, or raw numbers. /no_think"
        )
        user_text = hail.render_for_left_brain(scenario_text)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ]

    def respond_to_tug(
        self,
        scenario_text: str,
        tug_packet: Mapping | TacticalHail,
    ) -> tuple[str, dict]:
        """Answer a Will Tactical Hail with a fresh 2-sentence plain-English Theory proposal."""
        gen = self.models.config["generation"]
        messages = self.build_prompt_messages(scenario_text, tug_packet)
        request = {
            "messages": messages,
            "temperature": gen["temperature"],
            "seed": gen["seed"],
            "max_tokens": min(self.max_tokens, gen["max_tokens"]),
            "stream": False,
            "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        return self.models.generate(
            self.role,
            request,
            LEFT_BRAIN_PROMPT_VERSION,
            validate_plain_english_left_brain_reply,
        )


@dataclass(frozen=True)
class EquilibriumPassTrace:
    """Record of one internal Right-Brain <-> Left-Brain settling pass."""

    pass_index: int
    vibe: RightBrainVibe
    left_brain_reply: str | None = None
    max_theory_delta: float = 0.0
    tactical_hail: TacticalHail | None = None


@dataclass(frozen=True)
class EquilibriumResult:
    """Final settled (or genuinely unresolved) outcome for one external scenario."""

    scene_index: int
    scenario_text: str
    settled: bool
    outcome: str
    final_vibe: RightBrainVibe
    passes: tuple[EquilibriumPassTrace, ...]
    left_brain_calls: int
    final_conscious_note: str | None
    needs_new_instinct: bool
    next_memory: ContinuousMemoryState
    next_diary: SettledDiary
    felt_chord: FeltChord | None = None
    chord_echoes: ChordEchoReport | None = None
    last_tactical_hail: TacticalHail | None = None
    raw_thoughts: tuple[RawThoughtRecord, ...] = ()
    next_chord_memory: ChordMemoryBank = ChordMemoryBank()
    next_ledger: ContemplationLedger = ContemplationLedger()
    deliberation: ThreeClassifierDeliberation | None = None
    social_drift: SocialDriftReport | None = None
    open_thread: OpenThreadTrigger | None = None
    audience_mode: str = "PASS1_REFLEX_OR_EASE"
    next_archives: ThreeBranchArchives = ThreeBranchArchives()


def _top_key_coordinates(coords: Mapping[str, float], limit: int = 4) -> dict[str, float]:
    ranked = sorted(coords.items(), key=lambda kv: (-kv[1], kv[0]))
    selected = {k: round(float(v), 3) for k, v in ranked[:limit] if v > 0.0}
    if max(coords.values(), default=0.0) >= 0.55:
        lows = sorted(
            ((k, float(v)) for k, v in coords.items() if k in FOUNDATIONAL_LEAVES and float(v) <= 0.30),
            key=lambda kv: (kv[1], kv[0]),
        )
        if lows:
            selected[lows[0][0]] = round(lows[0][1], 3)
    return selected


def settle_scenario(
    birth_state: BirthState,
    scenario_text: str,
    sensors: ThreeBridgeSensors,
    rules: tuple[FuzzyConceptRule, ...],
    *,
    scene_index: int = 1,
    left_brain=None,
    learned_leaves: tuple[LearnedLeaf, ...] = (),
    prior_memory: ContinuousMemoryState | None = None,
    diary: SettledDiary | None = None,
    chord_memory: ChordMemoryBank | None = None,
    contemplation_ledger: ContemplationLedger | None = None,
    branch_archives: ThreeBranchArchives | None = None,
    raw_thought_stream_path: str | Path | None = None,
    contemplation_ledger_path: str | Path | None = None,
    branch_archives_dir: str | Path | None = None,
    require_self_critique: bool = False,
    max_passes: int = 3,
    equilibrium_tolerance: float = DEFAULT_EQUILIBRIUM_TOLERANCE,
    branch_gating: bool = True,
) -> EquilibriumResult:
    """Run internal 3-Branch Deliberation <-> Tactical Hail -> Left-Brain Self-Talk while keeping Fact immutable."""
    if type(max_passes) is not int or not (1 <= max_passes <= 10):
        raise ValueError("max_passes must be an int in [1, 10]")

    active_diary = diary if diary is not None else SettledDiary()
    active_chord_memory = chord_memory if chord_memory is not None else ChordMemoryBank()
    active_ledger = (
        contemplation_ledger if contemplation_ledger is not None else ContemplationLedger()
    )
    active_archives = branch_archives if branch_archives is not None else ThreeBranchArchives()

    # Pass 1: Pure EXTERNAL_WORLD sensory & memory read (Fact locked here for the whole scene)
    external_coords, external_exp, queried_1, skipped_1 = sensors.score_text(
        scenario_text,
        learned_leaves=learned_leaves,
        branch_gating=branch_gating,
    )
    social_drift = active_archives.executive.detect_social_drift(external_coords)

    vibe_1, next_mem = compile_right_brain_vibe(
        birth_state,
        external_coords,
        external_exp,
        rules,
        pass_index=1,
        event_id=f"scene-{scene_index}-pass-1",
        prior_memory=prior_memory,
        learned_leaves=learned_leaves,
        internal_theory_coords=None,
        internal_theory_experiential=None,
        queried_child_leaves=queried_1,
        skipped_child_leaves=skipped_1,
        social_drift=social_drift,
        ratified_chords=active_archives.judicial.ratified_chords,
    )

    initial_felt_chord = vibe_1.felt_chord or compile_felt_chord(
        external_coords,
        salience=vibe_1.salience,
        valence=vibe_1.valence,
        arousal=vibe_1.arousal,
        resonance=vibe_1.resonance,
        learned_leaves=learned_leaves,
    )
    chord_echoes = retrieve_chord_echoes(
        external_coords,
        initial_felt_chord,
        active_chord_memory,
        current_scene_index=scene_index,
    )
    open_thread = check_open_thread_trigger(
        active_ledger,
        chord_echoes,
        initial_felt_chord,
        external_coords,
        social_drift=social_drift,
    )

    traces: list[EquilibriumPassTrace] = [EquilibriumPassTrace(pass_index=1, vibe=vibe_1)]
    current_vibe = vibe_1
    prev_theory_coords: dict[str, float] = {}
    left_calls = 0
    latest_reply: str | None = None
    last_hail: TacticalHail | None = None
    raw_thoughts: list[RawThoughtRecord] = []
    deliberation: ThreeClassifierDeliberation | None = None
    audience_mode = "PASS1_REFLEX_OR_EASE"

    if current_vibe.posture == "TUG_LEFT_BRAIN" or open_thread.triggered:
        # Step 2A: Right-Brain Audience With Itself (~15-30 ms, 0 LLM calls)
        deliberation = deliberate_across_three_branches(
            birth_state,
            external_coords,
            external_exp,
            initial_felt_chord,
            sensors,
            active_archives,
            RIGHT_BRAIN_WILL_EDGES,
            RIGHT_BRAIN_WILL_VERSION,
            scene_index=scene_index,
            learned_leaves=learned_leaves,
            social_drift=social_drift,
            branch_gating=branch_gating,
        )

        if deliberation.ratified and (not open_thread.triggered or left_brain is None) and max_passes >= 2:
            # 3-Branch Harmony achieved with no open Left-Brain uncertainty (or LLM disabled): settle in milliseconds!
            audience_mode = "RIGHT_BRAIN_3_CLASSIFIER_DELIBERATION"
            active_ratified = (
                (deliberation.newly_ratified_chord,)
                if deliberation.newly_ratified_chord is not None
                else active_archives.judicial.ratified_chords
            )

            vibe_delib, next_mem = compile_right_brain_vibe(
                birth_state,
                external_coords,
                external_exp,
                rules,
                pass_index=2,
                event_id=f"scene-{scene_index}-pass-2-delib",
                prior_memory=prior_memory,
                learned_leaves=learned_leaves,
                internal_theory_coords=deliberation.bid_coordinates,
                internal_theory_experiential=deliberation.bid_experiential,
                queried_child_leaves=queried_1,
                skipped_child_leaves=skipped_1,
                social_drift=social_drift,
                ratified_chords=active_ratified,
            )
            max_delta_delib = round(
                max(
                    (
                        abs(float(deliberation.bid_coordinates.get(k, 0.0)) - float(external_coords.get(k, 0.0)))
                        for k in deliberation.bid_coordinates
                    ),
                    default=0.0,
                ),
                6,
            )
            traces.append(
                EquilibriumPassTrace(
                    pass_index=2,
                    vibe=vibe_delib,
                    max_theory_delta=max_delta_delib,
                )
            )
            current_vibe = vibe_delib
        elif left_brain is not None and max_passes >= 2:
            # Step 2B: Left-Brain Audience With Itself (Proposer <-> Self-Critic Dialogue)
            audience_mode = (
                "LEFT_BRAIN_OPEN_THREAD_CLOSURE"
                if open_thread.triggered
                else "LEFT_BRAIN_DEADLOCK_HAIL"
            )
            for p in range(2, max_passes + 1):
                prior_hunch_for_pass = latest_reply
                pre_pass_margin = _make_margin_tag_for_vibe(
                    current_vibe,
                    prior_hunch_exists=prior_hunch_for_pass is not None,
                )
                hail = build_tactical_hail(
                    current_vibe,
                    chord_echoes=chord_echoes,
                    contemplation_ledger=active_ledger,
                    prior_self_talk_hunch=prior_hunch_for_pass,
                    subliminal_margin_tag=pre_pass_margin,
                )
                last_hail = hail
                tug_msg = build_will_tug_message(
                    current_vibe,
                    active_diary,
                    prior_reply=prior_hunch_for_pass,
                    tactical_hail=hail,
                )
                reply_result = left_brain.respond_to_tug(scenario_text, tug_msg)
                reply_text = reply_result[0] if isinstance(reply_result, tuple) else str(reply_result)
                reply_text = validate_plain_english_left_brain_reply(reply_text)
                left_calls += 1
                latest_reply = reply_text

                theory_coords, theory_exp, _q_p, _s_p = sensors.score_text(
                    reply_text,
                    learned_leaves=learned_leaves,
                    branch_gating=branch_gating,
                )

                pass_ratified_chords = (
                    (deliberation.newly_ratified_chord,)
                    if (deliberation is not None and deliberation.newly_ratified_chord is not None)
                    else active_archives.judicial.ratified_chords
                )

                vibe_p, next_mem = compile_right_brain_vibe(
                    birth_state,
                    external_coords,
                    external_exp,
                    rules,
                    pass_index=p,
                    event_id=f"scene-{scene_index}-pass-{p}",
                    prior_memory=prior_memory,
                    learned_leaves=learned_leaves,
                    internal_theory_coords=theory_coords,
                    internal_theory_experiential=theory_exp,
                    queried_child_leaves=queried_1,
                    skipped_child_leaves=skipped_1,
                    social_drift=social_drift,
                    ratified_chords=pass_ratified_chords,
                )

                if prev_theory_coords:
                    max_delta = round(
                        max(
                            (abs(float(theory_coords.get(k, 0.0)) - float(prev_theory_coords.get(k, 0.0))) for k in theory_coords),
                            default=0.0,
                        ),
                        6,
                    )
                else:
                    max_delta = round(
                        max(
                            (abs(float(theory_coords.get(k, 0.0)) - float(external_coords.get(k, 0.0))) for k in theory_coords),
                            default=0.0,
                        ),
                        6,
                    )

                must_continue_for_critique = (
                    (require_self_critique or open_thread.triggered) and p == 2 and max_passes >= 3
                )
                settled_this_pass = (
                    vibe_p.posture in ("AT_EASE", "WORDLESS_REFLEX", "THEORY_SYNTHESIS")
                    and not must_continue_for_critique
                )
                post_pass_fc = vibe_p.felt_chord or initial_felt_chord
                post_pass_margin = _make_margin_tag_for_vibe(vibe_p, prior_hunch_exists=True)
                thought_record = RawThoughtRecord(
                    scene_index=scene_index,
                    pass_index=p,
                    clarity=vibe_p.clarity,
                    confidence=vibe_p.confidence,
                    raw_self_talk_draft=reply_text,
                    prior_hunch_debated=prior_hunch_for_pass,
                    felt_chord_stamp=post_pass_fc.short_stamp(),
                    subliminal_margin_tag=post_pass_margin,
                    epistemic_state=vibe_p.self_registry.epistemic_state,
                    settled_on_this_pass=settled_this_pass,
                )
                raw_thoughts.append(thought_record)
                if raw_thought_stream_path is not None:
                    append_raw_thought_record(raw_thought_stream_path, thought_record)

                if not settled_this_pass:
                    active_ledger = active_ledger.record_uncertainty(
                        scene_index=scene_index,
                        raw_hunch=reply_text,
                        open_question=post_pass_margin.red_pen_prompt,
                        margin_tag=post_pass_margin,
                    )

                traces[-1] = EquilibriumPassTrace(
                    pass_index=traces[-1].pass_index,
                    vibe=traces[-1].vibe,
                    left_brain_reply=reply_text,
                    max_theory_delta=traces[-1].max_theory_delta,
                    tactical_hail=hail,
                )
                traces.append(
                    EquilibriumPassTrace(
                        pass_index=p,
                        vibe=vibe_p,
                        max_theory_delta=max_delta,
                    )
                )
                current_vibe = vibe_p
                prev_theory_coords = dict(theory_coords)

                if settled_this_pass:
                    break
                if p >= 3 and max_delta <= equilibrium_tolerance:
                    break

    final_fc = current_vibe.felt_chord or initial_felt_chord
    if current_vibe.posture in ("AT_EASE", "WORDLESS_REFLEX", "THEORY_SYNTHESIS"):
        settled = True
        outcome = current_vibe.posture
        needs_new_instinct = False
        resolved_scenes = (
            (open_thread.target_scene_index,)
            if (open_thread.triggered and open_thread.target_scene_index is not None)
            else ()
        )
        takeaway_src = (
            latest_reply
            if latest_reply is not None
            else (
                deliberation.candidate_bid_text
                if (deliberation is not None and deliberation.ratified and deliberation.candidate_bid_text)
                else scenario_text
            )
        )
        active_ledger = active_ledger.record_settled(
            scene_index=scene_index,
            takeaway_source=takeaway_src,
            felt_chord=final_fc,
            clarity=current_vibe.clarity,
            confidence=current_vibe.confidence,
            resolved_uncertainty_scenes=resolved_scenes,
        )
    else:
        settled = False
        outcome = "UNRESOLVED_TENSION"
        needs_new_instinct = True
        final_margin = _make_margin_tag_for_vibe(
            current_vibe,
            prior_hunch_exists=latest_reply is not None,
        )
        active_ledger = active_ledger.record_uncertainty(
            scene_index=scene_index,
            raw_hunch=(
                latest_reply
                if latest_reply is not None
                else f"Unresolved scene tension: {scenario_text}"
            ),
            open_question=final_margin.red_pen_prompt,
            margin_tag=final_margin,
        )

    if contemplation_ledger_path is not None:
        save_contemplation_ledger(contemplation_ledger_path, active_ledger)

    next_archives = harvest_into_branch_archives(
        active_archives,
        scene_index=scene_index,
        scenario_text=scenario_text,
        coordinates=current_vibe.coordinates,
        felt_chord=final_fc,
        left_brain_reply=latest_reply,
        settled=settled,
        deliberation=deliberation,
        social_drift=social_drift,
    )
    if branch_archives_dir is not None:
        next_archives.save_all(branch_archives_dir)

    updated_chord_memory = active_chord_memory.append(
        scene_index=scene_index,
        scene_text_or_gist=scenario_text,
        coordinates=current_vibe.coordinates,
        felt_chord=final_fc,
        settled=settled,
    )

    diary_entry = DiaryEntry(
        scene_index=scene_index,
        outcome=outcome,
        active_concept=current_vibe.active_concept,
        key_coordinates=_top_key_coordinates(current_vibe.coordinates),
        conscious_note=latest_reply,
        epistemic_state=current_vibe.self_registry.epistemic_state,
    )
    updated_diary = active_diary.append(diary_entry)

    return EquilibriumResult(
        scene_index=scene_index,
        scenario_text=scenario_text,
        settled=settled,
        outcome=outcome,
        final_vibe=current_vibe,
        passes=tuple(traces),
        left_brain_calls=left_calls,
        final_conscious_note=latest_reply,
        needs_new_instinct=needs_new_instinct,
        next_memory=next_mem,
        next_diary=updated_diary,
        felt_chord=final_fc,
        chord_echoes=chord_echoes,
        last_tactical_hail=last_hail,
        raw_thoughts=tuple(raw_thoughts),
        next_chord_memory=updated_chord_memory,
        next_ledger=active_ledger,
        deliberation=deliberation,
        social_drift=social_drift,
        open_thread=open_thread,
        audience_mode=audience_mode,
        next_archives=next_archives,
    )
