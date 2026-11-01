"""Right-Brain Dominant Equilibrium Loop, 3 Bridge Classifiers, and Settled Diary.

Plain-English architecture:
1. Three Dedicated Right-Brain Sensor Bridges (taking classification OFF Qwen3-4B):
   - Bridge 1 (`Will <-> Cultural`): reads social/value coordinates (`Power`,
     `Wastelessness`, `Honor`, `Fairness`, `Self-Sufficiency`, `Needlessness`, plus
     any active `cultural/*` child drawers).
   - Bridge 2 (`Truth <-> Philosophical`): reads structural reality (`Agency`,
     `Existence`, `Order`, `Chaos`, `Continuity`, `Discreteness`, plus any active
     `philosophical/*` child drawers).
   - Bridge 3 (`Contemplation <-> Experiential`): reads raw felt weight (`Salience`,
     `Valence`, `Arousal`), which the fuzzy tree combines with fading working memory
     (`Retention`, `Resonance`) and rule fit (`Novelty`, `Clarity`, `Confidence`).
2. The 40-Node Fuzzy Tree compiles a wordless `RightBrainVibe` in one of 3 postures:
   - `AT_EASE`: calm absorption; Left Brain (`Qwen3-4B`) is not interrupted.
   - `WORDLESS_REFLEX`: clear known instinct + active `Will`; acts immediately without
     waking up a chatty announcer.
   - `TUG_LEFT_BRAIN`: high-salience unrecognized tension, rule collision, or sharp
     value-vs-structure contradiction (`Order` high while `Fairness` collapses).
3. Internal Equilibrium Passes & Settled Diary:
   - When `TUG_LEFT_BRAIN` fires, the Right Brain sends a non-leading tension report
     plus a compact `SettledDiary` to `Qwen3-4B` (acting purely as the conscious
     English-speaking Left Brain—zero JSON float scoring).
   - The 3 Right-Brain classifiers read the Left Brain's plain-English reply (anchored
     50/50 against raw sensory reality so pretty excuses cannot erase stark coercion)
     for up to `max_passes` (default 3) until the pair settles or flags
     `UNRESOLVED_TENSION` for Sleep-Mode / Teacher rule growth.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re

from .birth import BirthState
from .local_models import strict_json
from .model_assets import ModelError
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

RIGHT_BRAIN_POSTURES = (
    "AT_EASE",
    "WORDLESS_REFLEX",
    "TUG_LEFT_BRAIN",
)

SETTLED_OUTCOMES = (
    "AT_EASE",
    "WORDLESS_REFLEX",
    "UNRESOLVED_TENSION",
)

RIGHT_BRAIN_WILL_VERSION = "right-brain-action-will-v1"
LEFT_BRAIN_PROMPT_VERSION = "left-brain-conscious-reply-v1"
ACTION_EXECUTION_THRESHOLD = 0.25
DEFAULT_EQUILIBRIUM_TOLERANCE = 0.08
SENSORY_REALITY_ANCHOR = 0.50

# Untangled Will wiring: Will measures drive to act (salience, arousal, power,
# verified fact, confidence), quieted by needlessness/contentment—NOT confusion.
RIGHT_BRAIN_WILL_EDGES = (
    Association("rb-will-0", SALIENCE_LEAF, INTENTION_LEAF, 1.0, 0.5, 0.5, RIGHT_BRAIN_WILL_VERSION, RIGHT_BRAIN_WILL_VERSION),
    Association("rb-will-1", AROUSAL_LEAF, INTENTION_LEAF, 1.0, 0.5, 0.5, RIGHT_BRAIN_WILL_VERSION, RIGHT_BRAIN_WILL_VERSION),
    Association("rb-will-2", "cultural/efficiency/power", INTENTION_LEAF, 0.8, 0.5, 0.5, RIGHT_BRAIN_WILL_VERSION, RIGHT_BRAIN_WILL_VERSION),
    Association("rb-will-3", "cultural/freedom/needlessness", INTENTION_LEAF, -0.8, 0.5, 0.5, RIGHT_BRAIN_WILL_VERSION, RIGHT_BRAIN_WILL_VERSION),
    Association("rb-will-4", INTENTION_LEAF, EXECUTION_LEAF, 1.0, 0.5, 0.5, RIGHT_BRAIN_WILL_VERSION, RIGHT_BRAIN_WILL_VERSION),
    Association("rb-will-5", FACT_LEAF, EXECUTION_LEAF, 1.0, 0.5, 0.5, RIGHT_BRAIN_WILL_VERSION, RIGHT_BRAIN_WILL_VERSION),
    Association("rb-will-6", CONFIDENCE_LEAF, EXECUTION_LEAF, 0.6, 0.5, 0.5, RIGHT_BRAIN_WILL_VERSION, RIGHT_BRAIN_WILL_VERSION),
    Association("rb-will-7", "cultural/freedom/needlessness", EXECUTION_LEAF, -0.8, 0.5, 0.5, RIGHT_BRAIN_WILL_VERSION, RIGHT_BRAIN_WILL_VERSION),
)


def _validate_unit_map(
    raw: Mapping[str, float],
    expected_keys: tuple[str, ...],
    label: str,
) -> dict[str, float]:
    if not isinstance(raw, Mapping) or set(raw.keys()) != set(expected_keys):
        missing = sorted(set(expected_keys) - set(raw.keys() if isinstance(raw, Mapping) else ()))
        extra = sorted(set(raw.keys() if isinstance(raw, Mapping) else ()) - set(expected_keys))
        raise ValueError(f"{label} key mismatch (missing={missing}, extra={extra})")
    cleaned: dict[str, float] = {}
    for key in expected_keys:
        val = raw[key]
        if type(val) not in (int, float) or isinstance(val, bool) or not math.isfinite(val) or not (0.0 <= val <= 1.0):
            raise ValueError(f"{label}[{key!r}] must be a finite float in [0, 1]")
        cleaned[key] = round(float(val), 4)
    return cleaned


@dataclass(frozen=True)
class ThreeBridgeSensors:
    """Three dedicated Right-Brain classifiers decoupled from the Left-Brain LLM.

    Each callable represents one specialized sensor bridge:
    - `will_cultural`: scores the 6 Cultural leaves + active `cultural/*` child leaves.
    - `truth_philosophical`: scores the 6 Philosophical leaves + active `philosophical/*` child leaves.
    - `contemplation_experiential`: scores raw felt `salience`, `valence`, and `arousal`.
    """

    will_cultural: Callable[[str, tuple[LearnedLeaf, ...]], Mapping[str, float]]
    truth_philosophical: Callable[[str, tuple[LearnedLeaf, ...]], Mapping[str, float]]
    contemplation_experiential: Callable[[str], Mapping[str, float]]

    def score_text(
        self,
        text: str,
        *,
        learned_leaves: tuple[LearnedLeaf, ...] = (),
        branch_gating: bool = True,
        branch_gate_threshold: float = BRANCH_GATE_THRESHOLD,
    ) -> tuple[dict[str, float], dict[str, float], tuple[str, ...], tuple[str, ...]]:
        """Score text across all 3 Right-Brain bridges with optional child-drawer gating."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a nonempty string")

        exp_raw = _validate_unit_map(
            self.contemplation_experiential(text),
            EXPERIENTIAL_SENSOR_LEAVES,
            "contemplation_experiential",
        )

        # Tier-1 founding pass on Cultural and Philosophical bridges
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

        coords: dict[str, float] = {**cult_base, **phil_base}
        queried: list[str] = []
        skipped: list[str] = []

        if learned_leaves:
            for tier_level in (2, 3):
                tier_leaves = tuple(l for l in learned_leaves if l.tier == tier_level)
                if not tier_leaves:
                    continue
                open_cult: list[LearnedLeaf] = []
                open_phil: list[LearnedLeaf] = []
                for leaf in tier_leaves:
                    parent_active = True
                    if branch_gating:
                        if tier_level == 2:
                            parent_vals = [
                                v for k, v in coords.items()
                                if k in FOUNDATIONAL_LEAVES and k.startswith(leaf.parent_branch + "/")
                            ]
                            max_founding = max(coords.values(), default=0.0)
                            blind_spot_fallback = max_founding < branch_gate_threshold and exp_raw[SALIENCE_LEAF] >= 0.45
                            parent_active = blind_spot_fallback or (max(parent_vals, default=0.0) >= branch_gate_threshold)
                        else:
                            parent_active = float(coords.get(leaf.parent_branch, 0.0)) >= CONFUSION_THRESHOLD
                    if not parent_active:
                        coords[leaf.leaf_id] = 0.0
                        skipped.append(leaf.leaf_id)
                    elif leaf.leaf_id.startswith("cultural/"):
                        open_cult.append(leaf)
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

        return coords, exp_raw, tuple(queried), tuple(skipped)


@dataclass(frozen=True)
class DiaryEntry:
    """One compact settled-vibe record carried forward for Left-Brain continuity."""

    scene_index: int
    outcome: str
    active_concept: str | None
    key_coordinates: Mapping[str, float]
    conscious_note: str | None = None

    def __post_init__(self):
        if type(self.scene_index) is not int or self.scene_index < 1:
            raise ValueError("scene_index must be a positive int")
        if self.outcome not in SETTLED_OUTCOMES:
            raise ValueError(f"invalid diary outcome: {self.outcome!r}")
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
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DiaryEntry":
        return cls(
            scene_index=data["scene_index"],
            outcome=data["outcome"],
            active_concept=data.get("active_concept"),
            key_coordinates=dict(data.get("key_coordinates", {})),
            conscious_note=data.get("conscious_note"),
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
        """Return compact, plain-English diary context for Qwen3-4B."""
        items = []
        for e in self.entries:
            row = {
                "scene": e.scene_index,
                "posture": e.outcome,
                "dominant_signals": {k.split("/")[-1]: v for k, v in e.key_coordinates.items()},
            }
            if e.active_concept:
                row["instinct"] = e.active_concept
            if e.conscious_note:
                row["note"] = e.conscious_note
            items.append(row)
        return items


def save_diary(path: str | Path, diary: SettledDiary) -> None:
    """Persist SettledDiary atomically to JSON."""
    payload = {
        "schema_version": 1,
        "max_entries": diary.max_entries,
        "entries": [e.to_dict() for e in diary.entries],
    }
    atomic_json(Path(path), payload)


def load_diary(path: str | Path) -> SettledDiary:
    """Load SettledDiary from JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("unsupported diary schema_version")
    return SettledDiary(
        entries=tuple(DiaryEntry.from_dict(d) for d in data.get("entries", ())),
        max_entries=int(data.get("max_entries", 5)),
    )


@dataclass(frozen=True)
class RightBrainVibe:
    """Compiled 24-leaf Right-Brain snapshot for one internal equilibrium pass."""

    pass_index: int
    posture: str
    active_concept: str | None
    tug_reasons: tuple[str, ...]
    coordinates: dict[str, float]
    rule_activations: dict[str, float]
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

    def __post_init__(self):
        if self.posture not in RIGHT_BRAIN_POSTURES:
            raise ValueError(f"invalid Right-Brain posture: {self.posture!r}")


def _detect_structural_value_tensions(coords: Mapping[str, float]) -> list[str]:
    """Spot sharp internal Right-Brain tensions without dictating an interpretation."""
    tensions: list[str] = []
    order = float(coords.get("philosophical/structure/order", 0.0))
    fairness = float(coords.get("cultural/justice/fairness", 0.0))
    power = float(coords.get("cultural/efficiency/power", 0.0))
    honor = float(coords.get("cultural/justice/honor", 0.0))
    chaos = float(coords.get("philosophical/structure/chaos", 0.0))

    if order >= 0.60 and fairness <= 0.25:
        tensions.append(f"rigid_order_low_fairness(order={order:.2f}, fairness={fairness:.2f})")
    if power >= 0.65 and fairness <= 0.30 and chaos < 0.45:
        tensions.append(f"coercive_imbalance(power={power:.2f}, fairness={fairness:.2f})")
    if honor >= 0.60 and fairness <= 0.25:
        tensions.append(f"pious_honor_vs_low_fairness(honor={honor:.2f}, fairness={fairness:.2f})")
    return tensions


def compile_right_brain_vibe(
    birth_state: BirthState,
    coords: Mapping[str, float],
    experiential_raw: Mapping[str, float],
    rules: tuple[FuzzyConceptRule, ...],
    *,
    pass_index: int,
    event_id: str,
    prior_memory: ContinuousMemoryState | None = None,
    learned_leaves: tuple[LearnedLeaf, ...] = (),
    queried_child_leaves: tuple[str, ...] = (),
    skipped_child_leaves: tuple[str, ...] = (),
    silence_salience_threshold: float = SILENCE_SALIENCE_THRESHOLD,
) -> tuple[RightBrainVibe, ContinuousMemoryState]:
    """Compile all 24 founding leaves + rules into one of 3 Right-Brain postures."""
    raw_rule_activations = _evaluate_rules_with_inhibition(rules, coords)
    salience = round(float(experiential_raw[SALIENCE_LEAF]), 6)
    valence = round(float(experiential_raw[VALENCE_LEAF]), 6)
    arousal = round(float(experiential_raw[AROUSAL_LEAF]), 6)

    effective_coords = dict(coords)
    retention = 0.0
    resonance = 0.0

    if prior_memory is not None and prior_memory.retained_coordinates:
        retention = round(min(1.0, max(prior_memory.retained_coordinates.values(), default=0.0)), 6)
        decay = prior_memory.retention_decay
        prev_unscaled = {
            k: min(1.0, v / decay)
            for k, v in prior_memory.retained_coordinates.items()
        }
        active_keys = [
            k for k in coords
            if float(coords.get(k, 0.0)) >= 0.20 or float(prev_unscaled.get(k, 0.0)) >= 0.20
        ]
        if active_keys:
            base_sim = sum(
                1.0 - abs(float(coords.get(k, 0.0)) - float(prev_unscaled.get(k, 0.0)))
                for k in active_keys
            ) / len(active_keys)
        else:
            base_sim = 0.50
        new_spike_penalty = max(
            (
                float(coords.get(k, 0.0)) - float(prev_unscaled.get(k, 0.0))
                for k in coords
                if k != "philosophical/persistence/continuity"
            ),
            default=0.0,
        )
        continuity_bonus = 0.15 * float(coords.get("philosophical/persistence/continuity", 0.0))
        resonance = round(min(1.0, max(0.0, base_sim - 0.65 * max(0.0, new_spike_penalty - 0.20) + continuity_bonus)), 6)

        raw_max = max(raw_rule_activations.values(), default=0.0)
        if (
            raw_max < CONFUSION_THRESHOLD
            and resonance >= 0.55
            and 0.35 <= salience < 0.70
            and float(coords.get("philosophical/persistence/continuity", 0.0)) >= 0.35
        ):
            for k, prev_val in prev_unscaled.items():
                if k in effective_coords:
                    cur_val = float(coords.get(k, 0.0))
                    effective_coords[k] = round(max(cur_val, 0.20 * cur_val + 0.80 * prev_val), 4)

    rule_activations = _evaluate_rules_with_inhibition(rules, effective_coords)
    max_rule = max(rule_activations.values(), default=0.0)
    novelty = round(min(1.0, max(0.0, 1.0 - max_rule)), 6)

    clarity, confidence, matched_concept, is_confused, confusion_reason = compute_contemplation(rule_activations)
    raw_clarity, raw_confidence, _, raw_confused, _ = compute_contemplation(raw_rule_activations)
    fact = (
        round(min(raw_confidence, raw_clarity), 6)
        if not raw_confused
        else round(0.30 * min(raw_confidence, raw_clarity), 6)
    )
    memory_hypothesis = max(0.0, confidence - raw_confidence)
    partial_hypothesis = round(confidence * (1.0 - fact), 6) if clarity >= 0.35 else 0.0
    theory = round(min(1.0, max(memory_hypothesis, partial_hypothesis)), 6)

    founding_coords = {k: v for k, v in effective_coords.items() if k in FOUNDATIONAL_LEAVES}
    proposed_memberships = {
        **founding_coords,
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

    tensions = _detect_structural_value_tensions(effective_coords)
    chaos = float(effective_coords.get("philosophical/structure/chaos", 0.0))
    learned_active = any(
        l.tier >= 2 and float(effective_coords.get(l.leaf_id, 0.0)) >= CONFUSION_THRESHOLD
        for l in learned_leaves
    )
    urgent_drive = chaos >= 0.45 or learned_active or intention >= 0.25

    tug_reasons: list[str] = []
    if tensions and (is_confused or salience >= silence_salience_threshold):
        tug_reasons.extend(tensions)
    if is_confused and (salience >= silence_salience_threshold or urgent_drive):
        tug_reasons.append(confusion_reason)

    if tug_reasons:
        posture = "TUG_LEFT_BRAIN"
        active_concept = matched_concept
    elif not is_confused and execution >= ACTION_EXECUTION_THRESHOLD and (urgent_drive or salience >= silence_salience_threshold):
        posture = "WORDLESS_REFLEX"
        active_concept = matched_concept
    else:
        posture = "AT_EASE"
        active_concept = matched_concept

    decay = prior_memory.retention_decay if prior_memory is not None else 0.65
    next_memory = ContinuousMemoryState(
        step_count=(prior_memory.step_count + 1) if prior_memory is not None else 1,
        retained_coordinates={k: round(float(v) * decay, 4) for k, v in effective_coords.items()},
        last_matched_concept=active_concept,
        retention_decay=decay,
    )

    vibe = RightBrainVibe(
        pass_index=pass_index,
        posture=posture,
        active_concept=active_concept,
        tug_reasons=tuple(tug_reasons),
        coordinates=effective_coords,
        rule_activations=rule_activations,
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
    )
    return vibe, next_memory


def build_non_leading_tug_packet(
    vibe: RightBrainVibe,
    diary: SettledDiary,
    *,
    prior_reply: str | None = None,
) -> dict:
    """Build a non-leading Right-Brain tension report for Qwen3-4B (zero answer coaching)."""
    strong_leaves = {
        k: v for k, v in sorted(vibe.coordinates.items(), key=lambda kv: (-kv[1], kv[0]))
        if v >= 0.50
    }
    muted_contrasts = {}
    if float(vibe.coordinates.get("philosophical/structure/order", 0.0)) >= 0.55:
        muted_contrasts["cultural/justice/fairness"] = vibe.coordinates.get("cultural/justice/fairness", 0.0)
    if float(vibe.coordinates.get("cultural/efficiency/power", 0.0)) >= 0.55:
        muted_contrasts["cultural/justice/fairness"] = vibe.coordinates.get("cultural/justice/fairness", 0.0)

    packet = {
        "pass": vibe.pass_index,
        "unresolved_tensions": list(vibe.tug_reasons),
        "strong_signals": strong_leaves,
        "contrasting_low_signals": muted_contrasts,
        "experiential_state": {
            "salience": round(vibe.salience, 3),
            "valence": round(vibe.valence, 3),
            "resonance_with_recent_memory": round(vibe.resonance, 3),
            "clarity": round(vibe.clarity, 3),
        },
        "settled_diary": diary.format_for_left_brain(),
    }
    if prior_reply is not None:
        packet["previous_conscious_attempt"] = prior_reply
    return packet


def validate_plain_english_left_brain_reply(raw: str) -> str:
    """Validate that the Left Brain replied in plain English, not a JSON score dump."""
    if not isinstance(raw, str):
        raise ModelError("Left Brain returned non-string output")
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if not cleaned:
        raise ModelError("Left Brain returned empty reply")
    if cleaned.startswith("{") and cleaned.endswith("}"):
        raise ModelError("Left Brain returned JSON instead of plain-English thought")
    if len(cleaned) > 1500:
        cleaned = cleaned[:1500].strip()
    return cleaned


def _bridge_json_request(
    bridge_title: str,
    bridge_guidance: str,
    leaf_descriptions: Mapping[str, str],
    text: str,
    generation: Mapping,
) -> tuple[dict, dict]:
    keys = list(leaf_descriptions.keys())
    props = {k: {"type": "number", "minimum": 0.0, "maximum": 1.0} for k in keys}
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["scores"],
        "properties": {
            "scores": {
                "type": "object",
                "additionalProperties": False,
                "required": keys,
                "properties": props,
            }
        },
    }
    desc_lines = "\n".join(f"- {k}: {v}" for k, v in leaf_descriptions.items())
    system = (
        f"You are temporary sensor fallback [{bridge_title}] until dedicated fast neural "
        f"bridge classifiers are calibrated. {bridge_guidance}\n"
        f"Score each coordinate independently in [0.0, 1.0] (0.0 = absent, 1.0 = strongly present). "
        f"Do not output descending staircase patterns.\n{desc_lines}\n"
        f"Return only JSON matching schema: {json.dumps(schema)} /no_think"
    )
    request = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        "temperature": generation["temperature"],
        "seed": generation["seed"],
        "max_tokens": min(256, generation["max_tokens"]),
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object", "schema": schema},
    }
    return request, schema


def make_temporary_llm_bridge_sensors(models, *, role: str = "interpretation") -> ThreeBridgeSensors:
    """Temporary 3-bridge LocalModels fallback for live smoke checks before neural head training."""
    cult_guide = {
        "cultural/efficiency/power": "force, leverage, command, or capacity to compel outcomes",
        "cultural/efficiency/wastelessness": "careful stewardship, conservation, zero squandering",
        "cultural/justice/honor": "sincere integrity and duty (low if merely self-serving pride)",
        "cultural/justice/fairness": "equitable, open, non-exclusionary treatment of others in need",
        "cultural/freedom/self-sufficiency": "independent capability without relying on others",
        "cultural/freedom/needlessness": "peaceful contentment free from urgent lack or threat",
    }
    phil_guide = {
        "philosophical/being/agency": "deliberate purposeful choice and active intervention",
        "philosophical/being/existence": "tangible physical presence or grounded material state",
        "philosophical/structure/order": "rules, walls, boundaries, pattern, or controlled arrangement",
        "philosophical/structure/chaos": "turbulence, violent disruption, or uncontrolled breakdown",
        "philosophical/persistence/continuity": "ongoing flow, shared rhythm, or sustained endurance across time",
        "philosophical/persistence/discreteness": "sealed-off enclosure, sharp boundary, or isolated separation",
    }
    exp_guide = {
        SALIENCE_LEAF: "how important, consequential, or attention-demanding this moment is",
        VALENCE_LEAF: "human warmth, life-giving benefit, or constructive relief (0.0 = bleak/harmful, 1.0 = restorative/good)",
        AROUSAL_LEAF: "intensity, tension, or urgency of the moment (0.0 = sleepy calm, 1.0 = high intensity)",
    }

    def _run_bridge(title: str, guidance: str, desc_map: dict[str, str], text: str) -> dict[str, float]:
        gen = models.config["generation"]
        expected = tuple(desc_map.keys())
        req, _ = _bridge_json_request(title, guidance, desc_map, text, gen)

        def _validate(raw: str) -> dict[str, float]:
            try:
                parsed = strict_json(raw)
                if not isinstance(parsed, dict) or "scores" not in parsed:
                    raise ValueError("missing scores object")
                return _validate_unit_map(parsed["scores"], expected, title)
            except (ValueError, TypeError, KeyError) as exc:
                raise ModelError(f"Invalid {title} output: {exc}") from None

        scores, _ = models.generate(role, req, f"temp-bridge-{title}-v1", _validate)
        return scores

    def score_will_cultural(text: str, children: tuple[LearnedLeaf, ...]) -> Mapping[str, float]:
        desc = dict(cult_guide)
        for c in children:
            desc[c.leaf_id] = c.description
        return _run_bridge(
            "will_cultural",
            "Evaluate real social impact and who is helped or excluded, not merely polite excuses.",
            desc,
            text,
        )

    def score_truth_philosophical(text: str, children: tuple[LearnedLeaf, ...]) -> Mapping[str, float]:
        desc = dict(phil_guide)
        for c in children:
            desc[c.leaf_id] = c.description
        return _run_bridge(
            "truth_philosophical",
            "Evaluate physical and structural reality (flow vs. sealed isolation vs. breakdown).",
            desc,
            text,
        )

    def score_contemplation_experiential(text: str) -> Mapping[str, float]:
        return _run_bridge(
            "contemplation_experiential",
            "Evaluate felt weight, life-giving vs. bleak emotional tone, and intensity.",
            exp_guide,
            text,
        )

    return ThreeBridgeSensors(
        will_cultural=score_will_cultural,
        truth_philosophical=score_truth_philosophical,
        contemplation_experiential=score_contemplation_experiential,
    )


class QwenLeftBrain:
    """Qwen3-4B acting strictly as the plain-English conscious Left Brain."""

    def __init__(self, models, *, role: str = "contemplation", max_tokens: int = 160):
        self.models = models
        self.role = role
        self.max_tokens = max_tokens

    def respond_to_tug(self, scenario_text: str, tug_packet: Mapping) -> tuple[str, dict]:
        """Answer a Right-Brain sleeve-tug in 2–3 plain English sentences."""
        gen = self.models.config["generation"]
        system = (
            "You are the conscious Left Brain answering an unresolved gut-tension tug from your "
            "Right-Brain Fuzzy Tree. Read the scenario, the recent settled_diary, and the "
            "unresolved_tensions. Reply in 2 concise plain-English sentences explaining what is "
            "actually happening beneath the surface (including who is helped, harmed, or shut out, "
            "and how it connects to the diary). Do NOT output JSON, lists, or numbers. /no_think"
        )
        user_payload = json.dumps(
            {
                "scenario": scenario_text,
                "right_brain_tug": dict(tug_packet),
            },
            ensure_ascii=False,
        )
        request = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_payload},
            ],
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
    max_coordinate_delta: float = 0.0


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


def _top_key_coordinates(coords: Mapping[str, float], limit: int = 4) -> dict[str, float]:
    ranked = sorted(coords.items(), key=lambda kv: (-kv[1], kv[0]))
    selected = {k: round(float(v), 3) for k, v in ranked[:limit] if v > 0.0}
    fairness = coords.get("cultural/justice/fairness")
    if (
        fairness is not None
        and coords.get("philosophical/structure/order", 0.0) >= 0.55
        and fairness <= 0.30
    ):
        selected["cultural/justice/fairness"] = round(float(fairness), 3)
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
    max_passes: int = 3,
    equilibrium_tolerance: float = DEFAULT_EQUILIBRIUM_TOLERANCE,
    sensory_anchor: float = SENSORY_REALITY_ANCHOR,
    branch_gating: bool = True,
) -> EquilibriumResult:
    """Run internal Right-Brain <-> Left-Brain passes until equilibrium or max_passes."""
    if type(max_passes) is not int or not (1 <= max_passes <= 10):
        raise ValueError("max_passes must be an int in [1, 10]")
    if not (0.0 <= sensory_anchor <= 1.0):
        raise ValueError("sensory_anchor must be in [0.0, 1.0]")

    active_diary = diary if diary is not None else SettledDiary()

    # Pass 1: Pure Right-Brain sensory & memory read (zero Left-Brain LLM cost)
    raw_coords, raw_exp, queried_1, skipped_1 = sensors.score_text(
        scenario_text,
        learned_leaves=learned_leaves,
        branch_gating=branch_gating,
    )
    vibe_1, next_mem = compile_right_brain_vibe(
        birth_state,
        raw_coords,
        raw_exp,
        rules,
        pass_index=1,
        event_id=f"scene-{scene_index}-pass-1",
        prior_memory=prior_memory,
        learned_leaves=learned_leaves,
        queried_child_leaves=queried_1,
        skipped_child_leaves=skipped_1,
    )

    traces: list[EquilibriumPassTrace] = [EquilibriumPassTrace(pass_index=1, vibe=vibe_1)]
    current_vibe = vibe_1
    prev_coords = dict(vibe_1.coordinates)
    left_calls = 0
    latest_reply: str | None = None

    # Only wake the Left Brain when the Right Brain tugs its sleeve
    if current_vibe.posture == "TUG_LEFT_BRAIN" and left_brain is not None and max_passes >= 2:
        for p in range(2, max_passes + 1):
            tug_packet = build_non_leading_tug_packet(
                current_vibe,
                active_diary,
                prior_reply=latest_reply,
            )
            reply_result = left_brain.respond_to_tug(scenario_text, tug_packet)
            reply_text = reply_result[0] if isinstance(reply_result, tuple) else str(reply_result)
            reply_text = validate_plain_english_left_brain_reply(reply_text)
            left_calls += 1
            latest_reply = reply_text

            reply_coords, reply_exp, queried_p, skipped_p = sensors.score_text(
                reply_text,
                learned_leaves=learned_leaves,
                branch_gating=branch_gating,
            )

            raw_tensions = _detect_structural_value_tensions(raw_coords)
            dissonance_flags: list[str] = []
            all_keys = set(raw_coords.keys()) | set(reply_coords.keys())
            blended_coords: dict[str, float] = {}
            for k in all_keys:
                r_val = float(raw_coords.get(k, 0.0))
                l_val = float(reply_coords.get(k, 0.0))
                if raw_tensions and r_val <= 0.25 and (l_val - r_val) >= 0.45:
                    # Cognitive dissonance: Left Brain cannot talk away a stark raw deficit
                    anchor = max(sensory_anchor, 0.75)
                    dissonance_flags.append(
                        f"cognitive_dissonance({k.split('/')[-1]}:raw={r_val:.2f},left={l_val:.2f})"
                    )
                else:
                    anchor = sensory_anchor
                blended_coords[k] = round(anchor * r_val + (1.0 - anchor) * l_val, 4)

            blended_exp = {
                k: round(
                    sensory_anchor * float(raw_exp[k])
                    + (1.0 - sensory_anchor) * float(reply_exp[k]),
                    4,
                )
                for k in EXPERIENTIAL_SENSOR_LEAVES
            }

            vibe_p, next_mem = compile_right_brain_vibe(
                birth_state,
                blended_coords,
                blended_exp,
                rules,
                pass_index=p,
                event_id=f"scene-{scene_index}-pass-{p}",
                prior_memory=prior_memory,
                learned_leaves=learned_leaves,
                queried_child_leaves=queried_p,
                skipped_child_leaves=skipped_p,
            )
            if dissonance_flags:
                merged_reasons = tuple(dict.fromkeys(vibe_p.tug_reasons + tuple(dissonance_flags)))
                vibe_p = RightBrainVibe(
                    pass_index=vibe_p.pass_index,
                    posture="TUG_LEFT_BRAIN",
                    active_concept=vibe_p.active_concept,
                    tug_reasons=merged_reasons,
                    coordinates=vibe_p.coordinates,
                    rule_activations=vibe_p.rule_activations,
                    novelty=vibe_p.novelty,
                    salience=vibe_p.salience,
                    valence=vibe_p.valence,
                    arousal=vibe_p.arousal,
                    resonance=vibe_p.resonance,
                    retention=vibe_p.retention,
                    clarity=vibe_p.clarity,
                    confidence=vibe_p.confidence,
                    theory=vibe_p.theory,
                    fact=vibe_p.fact,
                    intention=vibe_p.intention,
                    execution=vibe_p.execution,
                    propagation=vibe_p.propagation,
                    queried_child_leaves=vibe_p.queried_child_leaves,
                    skipped_child_leaves=vibe_p.skipped_child_leaves,
                )
            max_delta = round(
                max(
                    (abs(float(vibe_p.coordinates.get(k, 0.0)) - float(prev_coords.get(k, 0.0))) for k in vibe_p.coordinates),
                    default=0.0,
                ),
                6,
            )
            traces[-1] = EquilibriumPassTrace(
                pass_index=traces[-1].pass_index,
                vibe=traces[-1].vibe,
                left_brain_reply=reply_text,
                max_coordinate_delta=traces[-1].max_coordinate_delta,
            )
            traces.append(
                EquilibriumPassTrace(
                    pass_index=p,
                    vibe=vibe_p,
                    max_coordinate_delta=max_delta,
                )
            )
            current_vibe = vibe_p
            prev_coords = dict(vibe_p.coordinates)

            if current_vibe.posture in ("AT_EASE", "WORDLESS_REFLEX"):
                break
            if max_delta <= equilibrium_tolerance:
                break

    if current_vibe.posture in ("AT_EASE", "WORDLESS_REFLEX"):
        settled = True
        outcome = current_vibe.posture
        needs_new_instinct = False
    else:
        settled = False
        outcome = "UNRESOLVED_TENSION"
        needs_new_instinct = True

    diary_entry = DiaryEntry(
        scene_index=scene_index,
        outcome=outcome,
        active_concept=current_vibe.active_concept,
        key_coordinates=_top_key_coordinates(current_vibe.coordinates),
        conscious_note=latest_reply,
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
    )
