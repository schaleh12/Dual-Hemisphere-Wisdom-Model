"""Steps 1–4: Two-Tier Fast SLM Coordinate Mapper + Fuzzy Ruleset + Will Execution + Teacher Growth.

How the full loop works in plain English:
1. Step 1 (`Experiential -> Reflection` with Two-Tier Routing):
   - Tier 1: Qwen3-4B always maps incoming text onto the 12 foundational Cultural (6) and
     Philosophical (6) leaves in [0, 1] using a fixed, unchanging prompt so Tier-1 readings
     never shift as the tree grows.
   - Tier 2: When the tree has grown learned child leaves (`LearnedLeaf`) under founding
     branches, a separate focused Tier-2 pass scores those child leaves in [0, 1].
   - Specificity inhibition: When a specific Tier-2 child rule activates (>= 0.50), broad
     Tier-1 trunk rules are discounted by `1 - 0.5 * max_tier2` so specific child concepts
     cleanly supersede broad trunk rules.
2. Step 2 (`Psychological -> Contemplation`):
   - `confidence` = strongest matching rule activation.
   - `clarity` = strongest rule activation minus second-strongest rule activation.
   - When 0 rules match or 2 rules collide, `clarity` / `confidence` drop below 0.50.
3. Step 3 (`Psychological -> Will -> Intention / Execution` & Teacher Request):
   - Synchronous propagation (`propagate_activation`) passes activations from
     `Novelty` (+) and `Salience` (+) into `Intention` and `Execution`, while
     `Clarity` (-) and `Confidence` (-) inhibit `Execution`.
   - When `Execution` exceeds `EXECUTION_THRESHOLD` (0.20) (or when two rules collide
     and `is_confused` is True), the tree asks the Teacher LLM (Qwen3.5-9B) to propose
     a new rule—stepping down from Tier 1 to grow a Tier-2 `LearnedLeaf` whenever Tier 1
     hits a sensor blind spot (`max_leaf < 0.50`) or a Tier-1 rule collision.
4. Step 4 (Persistent Growth):
   - Validated child leaves and rules are saved to disk as JSON and reloaded so future
     similar situations are recognized directly by Qwen3-4B + the fuzzy ruleset without
     needing the Teacher LLM.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
import numpy as np
import skfuzzy as fuzz

from .activation import ActivationFrame
from .birth import BirthState
from .local_models import strict_json
from .model_assets import ModelError
from .propagation import Association, PropagationConfig, PropagationResult, propagate_activation
from .runtime_store import atomic_json

FOUNDATIONAL_LEAVES = (
    # Cultural (top-down values & social context)
    "cultural/efficiency/power",
    "cultural/efficiency/wastelessness",
    "cultural/justice/honor",
    "cultural/justice/fairness",
    "cultural/freedom/self-sufficiency",
    "cultural/freedom/needlessness",
    # Philosophical (cross-cutting structural framing)
    "philosophical/being/agency",
    "philosophical/being/existence",
    "philosophical/structure/order",
    "philosophical/structure/chaos",
    "philosophical/persistence/continuity",
    "philosophical/persistence/discreteness",
)

FOUNDING_PARENT_BRANCHES = (
    "cultural/efficiency",
    "cultural/justice",
    "cultural/freedom",
    "philosophical/being",
    "philosophical/structure",
    "philosophical/persistence",
)

EXPERIENTIAL_PARENT_BRANCHES = (
    "experiential/reflection",
    "experiential/phase",
    "experiential/synchronization",
)

ALL_GROWABLE_PARENT_BRANCHES = FOUNDING_PARENT_BRANCHES + EXPERIENTIAL_PARENT_BRANCHES

NOVELTY_LEAF = "experiential/reflection/novelty"
SALIENCE_LEAF = "experiential/reflection/salience"
VALENCE_LEAF = "experiential/phase/valence"
AROUSAL_LEAF = "experiential/phase/arousal"
RESONANCE_LEAF = "experiential/synchronization/resonance"
ENTRAINMENT_LEAF = "experiential/synchronization/entrainment"
RETENTION_LEAF = ENTRAINMENT_LEAF
CLARITY_LEAF = "psychological/contemplation/clarity"
CONFIDENCE_LEAF = "psychological/contemplation/confidence"
THEORY_LEAF = "psychological/truth/theory"
FACT_LEAF = "psychological/truth/fact"
INTENTION_LEAF = "psychological/will/intention"
EXECUTION_LEAF = "psychological/will/execution"

CONSTRUCTIVE_LEAVES = (
    "cultural/justice/fairness",
    "cultural/justice/honor",
    "cultural/efficiency/wastelessness",
    "cultural/freedom/self-sufficiency",
    "philosophical/structure/order",
    "philosophical/persistence/continuity",
)

CONFUSION_THRESHOLD = 0.50
EXECUTION_THRESHOLD = 0.20
BRANCH_GATE_THRESHOLD = 0.25
SILENCE_SALIENCE_THRESHOLD = 0.48
PROMPT_VERSION = "reflection-leaf-mapper-step1-v2"
TIER2_PROMPT_VERSION = "reflection-tier2-child-mapper-v1"
TEACHER_PROMPT_VERSION = "desperation-teacher-step3-v1"
WILL_RULE_VERSION = "reflection-contemplation-will-1"

WILL_EDGES = (
    Association("will-0", NOVELTY_LEAF, INTENTION_LEAF, 1.0, 0.5, 0.5, WILL_RULE_VERSION, WILL_RULE_VERSION),
    Association("will-1", SALIENCE_LEAF, INTENTION_LEAF, 1.0, 0.5, 0.5, WILL_RULE_VERSION, WILL_RULE_VERSION),
    Association("will-2", INTENTION_LEAF, EXECUTION_LEAF, 1.0, 0.5, 0.5, WILL_RULE_VERSION, WILL_RULE_VERSION),
    Association("will-3", NOVELTY_LEAF, EXECUTION_LEAF, 1.0, 0.5, 0.5, WILL_RULE_VERSION, WILL_RULE_VERSION),
    Association("will-4", CLARITY_LEAF, EXECUTION_LEAF, -1.0, 0.5, 0.5, WILL_RULE_VERSION, WILL_RULE_VERSION),
    Association("will-5", CONFIDENCE_LEAF, EXECUTION_LEAF, -1.0, 0.5, 0.5, WILL_RULE_VERSION, WILL_RULE_VERSION),
)


def leaf_tier(leaf_id: str) -> int:
    """Return 1 for founding leaves, 2 for child leaves under founding branches, 3 for Tier-3 sub-leaves."""
    if not isinstance(leaf_id, str):
        return 0
    if leaf_id in FOUNDATIONAL_LEAVES or leaf_id in (
        NOVELTY_LEAF,
        SALIENCE_LEAF,
        VALENCE_LEAF,
        AROUSAL_LEAF,
        RESONANCE_LEAF,
        RETENTION_LEAF,
    ):
        return 1
    for branch in ALL_GROWABLE_PARENT_BRANCHES:
        prefix = branch + "/"
        if leaf_id.startswith(prefix):
            rest = leaf_id[len(prefix):]
            parts = rest.split("/")
            if 1 <= len(parts) <= 2 and all(
                re.fullmatch(r"[a-z][a-z0-9_-]{2,40}", p) for p in parts
            ):
                return len(parts) + 1
    return 0


def is_valid_leaf_id(leaf_id: str) -> bool:
    """Return True if leaf_id is a founding leaf or a learned Tier-2/Tier-3 child leaf."""
    return leaf_tier(leaf_id) > 0


@dataclass(frozen=True)
class LearnedLeaf:
    """A non-founding Tier-2 or Tier-3 child classification leaf grown under a parent branch."""

    parent_branch: str
    leaf_name: str
    description: str
    anchor_weights: tuple[float, ...] | None = None

    def __post_init__(self):
        if self.parent_branch not in ALL_GROWABLE_PARENT_BRANCHES and leaf_tier(self.parent_branch) != 2:
            raise ValueError(
                f"parent_branch must be a founding branch or Tier-2 learned leaf: {self.parent_branch!r}"
            )
        if not isinstance(self.leaf_name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{2,40}", self.leaf_name):
            raise ValueError(f"invalid leaf_name: {self.leaf_name!r}")
        if leaf_tier(self.leaf_id) == 1:
            raise ValueError(f"learned leaf cannot overwrite founding leaf: {self.leaf_id!r}")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be a nonempty string")
        if self.anchor_weights is not None:
            weights = tuple(round(float(w), 6) for w in self.anchor_weights)
            if len(weights) != 8 or any(not math.isfinite(w) or not (0.0 <= w <= 1.0) for w in weights):
                raise ValueError("anchor_weights must be a sequence of 8 finite floats in [0.0, 1.0]")
            object.__setattr__(self, "anchor_weights", weights)

    @property
    def leaf_id(self) -> str:
        return f"{self.parent_branch}/{self.leaf_name}"

    @property
    def tier(self) -> int:
        return leaf_tier(self.leaf_id)

    def to_dict(self) -> dict:
        out = {
            "parent_branch": self.parent_branch,
            "leaf_name": self.leaf_name,
            "description": self.description,
            "leaf_id": self.leaf_id,
        }
        if self.anchor_weights is not None:
            out["anchor_weights"] = list(self.anchor_weights)
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "LearnedLeaf":
        raw_weights = data.get("anchor_weights")
        return cls(
            parent_branch=data["parent_branch"],
            leaf_name=data["leaf_name"],
            description=data["description"],
            anchor_weights=tuple(float(w) for w in raw_weights) if raw_weights is not None else None,
        )


@dataclass(frozen=True)
class FuzzyConceptRule:
    """A fuzzy IF-THEN rule defined as a coordinate pattern over foundational and learned leaves."""

    rule_id: str
    concept_name: str
    description: str
    target_leaves: Mapping[str, float]
    low: float = 0.55
    high: float = 0.80
    polarity: str = ""

    def __post_init__(self):
        if not isinstance(self.rule_id, str) or not self.rule_id.strip():
            raise ValueError("rule_id must be a nonempty string")
        if not isinstance(self.concept_name, str) or not self.concept_name.strip():
            raise ValueError("concept_name must be a nonempty string")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be a nonempty string")
        if not isinstance(self.target_leaves, Mapping) or not self.target_leaves:
            raise ValueError("target_leaves must be a nonempty mapping of leaf_id -> float")
        for leaf_id, val in self.target_leaves.items():
            if not is_valid_leaf_id(leaf_id):
                raise ValueError(f"unknown foundational leaf: {leaf_id!r}")
            if type(val) not in (int, float) or not math.isfinite(val) or not (0.0 <= val <= 1.0):
                raise ValueError(f"target value for {leaf_id!r} must be in [0, 1]")
        if (type(self.low) not in (int, float) or type(self.high) not in (int, float)
                or not (0.0 <= self.low < self.high <= 1.0)):
            raise ValueError("rule thresholds must satisfy 0.0 <= low < high <= 1.0")
        resolved_polarity = (
            self.polarity.strip().lower()
            if isinstance(self.polarity, str) and self.polarity.strip()
            else ("repulsive" if self.concept_name.startswith("hazard_") else "attractive")
        )
        if resolved_polarity not in ("attractive", "repulsive"):
            raise ValueError("polarity must be 'attractive' or 'repulsive'")
        object.__setattr__(self, "polarity", resolved_polarity)

    @property
    def tier(self) -> int:
        """Return the deepest leaf tier (1, 2, or 3) used in target_leaves."""
        return max(leaf_tier(leaf_id) for leaf_id in self.target_leaves)

    def evaluate(self, leaf_coordinates: Mapping[str, float], *, soft_min: bool = False) -> float:
        """Compute fuzzy rule firing strength in [0, 1] using shoulder/triangular leaf similarity."""
        similarities = []
        for leaf_id, target_val in self.target_leaves.items():
            if leaf_id not in leaf_coordinates:
                raise ValueError(f"missing coordinate for leaf {leaf_id!r}")
            actual = float(leaf_coordinates[leaf_id])
            t_val = float(target_val)
            if t_val >= 0.65 and actual >= t_val:
                sim = 1.0
            elif t_val <= 0.30 and actual <= t_val:
                sim = 1.0
            else:
                sim = 1.0 - abs(actual - t_val)
            similarities.append(sim)
        min_similarity = min(similarities)
        mean_similarity = sum(similarities) / len(similarities)
        effective_sim = (
            (0.85 * min_similarity + 0.15 * mean_similarity)
            if (soft_min and min_similarity >= self.low)
            else min_similarity
        )
        universe = np.array([0.0, self.low, self.high, 1.0], dtype=float)
        curve = fuzz.trapmf(universe, [self.low, self.high, 1.0, 1.0])
        return round(float(np.interp(effective_sim, universe, curve)), 6)

    def to_dict(self) -> dict:
        out = {
            "rule_id": self.rule_id,
            "concept_name": self.concept_name,
            "description": self.description,
            "target_leaves": dict(self.target_leaves),
            "low": self.low,
            "high": self.high,
        }
        if self.polarity != "attractive":
            out["polarity"] = self.polarity
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "FuzzyConceptRule":
        return cls(
            rule_id=data["rule_id"],
            concept_name=data["concept_name"],
            description=data["description"],
            target_leaves=dict(data["target_leaves"]),
            low=float(data.get("low", 0.55)),
            high=float(data.get("high", 0.80)),
            polarity=str(data.get("polarity", "")),
        )


def save_ruleset(
    path: str | Path,
    rules: tuple[FuzzyConceptRule, ...],
    learned_leaves: tuple[LearnedLeaf, ...] = (),
) -> None:
    """Step 4: Atomically persist the learned fuzzy ruleset (and any learned child leaves) to disk."""
    payload = {
        "schema_version": 1,
        "rules": [r.to_dict() for r in rules],
    }
    if learned_leaves:
        payload["learned_leaves"] = [l.to_dict() for l in learned_leaves]
    atomic_json(Path(path), payload)


def load_ruleset(path: str | Path) -> tuple[FuzzyConceptRule, ...]:
    """Step 4: Load and validate a persisted fuzzy ruleset from disk."""
    raw = strict_json(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1 or not isinstance(raw.get("rules"), list):
        raise ValueError("invalid ruleset file format")
    return tuple(FuzzyConceptRule.from_dict(item) for item in raw["rules"])


def load_learned_leaves(path: str | Path) -> tuple[LearnedLeaf, ...]:
    """Load any learned child classification leaves stored alongside the ruleset."""
    raw = strict_json(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("invalid ruleset file format")
    items = raw.get("learned_leaves", [])
    return tuple(LearnedLeaf.from_dict(item) for item in items)


def slm_schema() -> dict:
    """Constant-size Tier-1 JSON schema for the 4B SLM: 12 foundational leaves + salience + summary."""
    leaf_props = {
        leaf_id: {"type": "number", "minimum": 0.0, "maximum": 1.0}
        for leaf_id in FOUNDATIONAL_LEAVES
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["summary", "salience_estimate", "leaf_coordinates"],
        "properties": {
            "summary": {"type": "string"},
            "salience_estimate": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "leaf_coordinates": {
                "type": "object",
                "additionalProperties": False,
                "required": list(FOUNDATIONAL_LEAVES),
                "properties": leaf_props,
            },
        },
    }


def make_slm_request(observation: str, generation: dict) -> dict:
    """Build the invariant Tier-1 chat-completion payload for Qwen3-4B (always the 12 founding leaves)."""
    if not isinstance(observation, str) or not observation.strip():
        raise ValueError("observation must be a nonempty string")
    schema = slm_schema()
    system_prompt = (
        "You are the fast perception mapper for a Language Model Fuzzy Tree. "
        "Read the input situation and score how strongly each of the 12 foundational "
        "Cultural and Philosophical leaves is present in [0.0, 1.0] (0.0 = absent/low, 1.0 = strongly present). "
        "Pairs (such as order and chaos, or agency and existence) are independent fuzzy dimensions and do NOT need to sum to 1. "
        "Also provide a one-sentence summary and a salience_estimate in [0.0, 1.0] for how important or consequential the situation is. "
        "Return JSON only. /no_think"
    )
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps({"observation": observation}, ensure_ascii=False)},
        ],
        "temperature": generation["temperature"],
        "seed": generation["seed"],
        "max_tokens": generation["max_tokens"],
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object", "schema": schema},
    }


def validate_slm_output(raw: str, learned_leaves: tuple[LearnedLeaf, ...] = ()) -> dict:
    """Validate that the 4B SLM returned finite [0, 1] coordinates for the 12 founding leaves."""
    allowed_extra = {l.leaf_id for l in learned_leaves}
    try:
        data = strict_json(raw)
        if not isinstance(data, dict):
            raise ValueError("response must be a JSON object")
        if set(data) != {"summary", "salience_estimate", "leaf_coordinates"}:
            raise ValueError("unexpected or missing top-level JSON fields")
        if not isinstance(data["summary"], str) or not data["summary"].strip():
            raise ValueError("summary must be a nonempty string")
        salience = data["salience_estimate"]
        if type(salience) not in (int, float) or not math.isfinite(salience) or not (0.0 <= salience <= 1.0):
            raise ValueError("salience_estimate must be a finite number in [0, 1]")
        coords = data["leaf_coordinates"]
        if not isinstance(coords, dict):
            raise ValueError("leaf_coordinates must be a JSON object")
        if not set(FOUNDATIONAL_LEAVES).issubset(set(coords)) or not set(coords).issubset(
            set(FOUNDATIONAL_LEAVES) | allowed_extra
        ):
            raise ValueError("leaf_coordinates must contain all 12 foundational leaves")
        cleaned_coords = {}
        for leaf_id, val in coords.items():
            if type(val) not in (int, float) or not math.isfinite(val) or not (0.0 <= val <= 1.0):
                raise ValueError(f"coordinate for {leaf_id!r} must be a finite number in [0, 1]")
            cleaned_coords[leaf_id] = round(float(val), 4)
        return {
            "summary": data["summary"].strip(),
            "salience_estimate": round(float(salience), 4),
            "leaf_coordinates": cleaned_coords,
        }
    except (ValueError, TypeError, KeyError) as exc:
        raise ModelError(f"Invalid reflection classifier output: {exc}") from None


def make_tier2_child_request(
    observation: str,
    generation: dict,
    learned_leaves: tuple[LearnedLeaf, ...],
) -> dict:
    """Build the focused Tier-2 prompt to score only learned child leaves without perturbing Tier 1."""
    leaf_ids = [l.leaf_id for l in learned_leaves]
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["child_leaf_coordinates"],
        "properties": {
            "child_leaf_coordinates": {
                "type": "object",
                "additionalProperties": False,
                "required": leaf_ids,
                "properties": {
                    lid: {"type": "number", "minimum": 0.0, "maximum": 1.0}
                    for lid in leaf_ids
                },
            }
        },
    }
    descriptions = "\n".join(f"- {l.leaf_id}: {l.description}" for l in learned_leaves)
    system_prompt = (
        "You are the Tier-2 branch classifier for a Language Model Fuzzy Tree. "
        "Read the input situation and score ONLY how strongly each specific child classification applies "
        "in [0.0, 1.0] (0.0 = does not apply at all, 1.0 = directly describes the core phenomenon):\n"
        f"{descriptions}\n"
        "Be selective: give 0.0 if a child classification does not specifically apply. Return JSON only. /no_think"
    )
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps({"observation": observation}, ensure_ascii=False)},
        ],
        "temperature": generation["temperature"],
        "seed": generation["seed"],
        "max_tokens": generation["max_tokens"],
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object", "schema": schema},
    }


def validate_tier2_child_output(raw: str, learned_leaves: tuple[LearnedLeaf, ...]) -> dict[str, float]:
    """Validate the Tier-2 child leaf coordinates."""
    expected = {l.leaf_id for l in learned_leaves}
    try:
        data = strict_json(raw)
        if not isinstance(data, dict) or set(data) != {"child_leaf_coordinates"}:
            raise ValueError("unexpected tier2 JSON fields")
        coords = data["child_leaf_coordinates"]
        if not isinstance(coords, dict) or set(coords) != expected:
            raise ValueError("missing tier2 child_leaf_coordinates")
        cleaned = {}
        for lid in expected:
            val = coords[lid]
            if type(val) not in (int, float) or not math.isfinite(val) or not (0.0 <= val <= 1.0):
                raise ValueError("invalid tier2 coordinate value")
            cleaned[lid] = round(float(val), 4)
        return cleaned
    except (ValueError, TypeError, KeyError) as exc:
        raise ModelError(f"Invalid tier2 child classifier output: {exc}") from None


def compute_contemplation(rule_activations: Mapping[str, float]) -> tuple[float, float, str | None, bool, str]:
    """Step 2: Compute Clarity, Confidence, winning concept, and confusion status."""
    if not rule_activations:
        return 0.0, 0.0, None, True, "unfamiliar_no_rules_in_notebook"

    ranked = sorted(rule_activations.items(), key=lambda kv: (-kv[1], kv[0]))
    top_name, top_val = ranked[0]
    second_val = ranked[1][1] if len(ranked) > 1 else 0.0

    confidence = round(min(1.0, max(0.0, float(top_val))), 6)
    clarity = round(min(1.0, max(0.0, float(top_val) - float(second_val))), 6)

    if confidence < CONFUSION_THRESHOLD:
        return clarity, confidence, None, True, "unfamiliar_pattern_no_matching_rule"
    if clarity < CONFUSION_THRESHOLD:
        runner_up = ranked[1][0]
        return clarity, confidence, None, True, f"ambiguous_rule_collision:{top_name}_vs_{runner_up}"
    return clarity, confidence, top_name, False, "clear_match"


@dataclass(frozen=True)
class ContinuousMemoryState:
    """Continuous working-memory trace carried across sequential observations."""

    step_count: int = 0
    retained_coordinates: Mapping[str, float] = field(default_factory=dict)
    last_matched_concept: str | None = None
    retention_decay: float = 0.65

    def __post_init__(self):
        if type(self.step_count) is not int or self.step_count < 0:
            raise ValueError("step_count must be a non-negative int")
        if type(self.retention_decay) not in (int, float) or not (0.0 < self.retention_decay <= 1.0):
            raise ValueError("retention_decay must be in (0, 1]")
        for k, v in self.retained_coordinates.items():
            if not is_valid_leaf_id(k):
                raise ValueError(f"invalid retained leaf_id: {k!r}")
            if type(v) not in (int, float) or not math.isfinite(v) or not (0.0 <= v <= 1.0):
                raise ValueError(f"retained coordinate for {k!r} must be in [0, 1]")


EXECUTIVE_MODES = ("STAY_SILENT", "COMMIT_FAST", "VERIFY_HYPOTHESIS", "ESCALATE_TO_TEACHER")
ACTIVE_EXECUTIVE_MODES = ("COMMIT_FAST", "VERIFY_HYPOTHESIS", "ESCALATE_TO_TEACHER")
EXECUTIVE_PROMPT_VERSION = "executive-hybrid-response-v1"


@dataclass(frozen=True)
class ExecutiveDecision:
    """Executive control decision produced by the Psychological branch (Truth, Contemplation, Will)."""

    mode: str = "ESCALATE_TO_TEACHER"
    active_concept: str | None = None
    fact: float = 0.0
    theory: float = 0.0
    clarity: float = 0.0
    confidence: float = 0.0
    execution: float = 0.0
    rationale: str = ""

    def __post_init__(self):
        if self.mode not in EXECUTIVE_MODES:
            raise ValueError(f"invalid executive mode: {self.mode!r}")


@dataclass(frozen=True)
class ReflectionEvaluation:
    """Inspectable result of Steps 1, 2, and 3 (Will propagation, continuous memory, Truth, & Executive decision)."""

    observation: str
    slm_output: dict
    rule_activations: dict[str, float]
    max_rule_activation: float
    novelty: float
    salience: float
    clarity: float
    confidence: float
    intention: float
    execution: float
    matched_concept: str | None
    is_confused: bool
    confusion_reason: str
    needs_teacher: bool
    propagation_result: PropagationResult
    model_trace: dict
    valence: float = 0.0
    arousal: float = 0.0
    resonance: float = 0.0
    retention: float = 0.0
    theory: float = 0.0
    fact: float = 0.0
    executive_decision: ExecutiveDecision = field(default_factory=ExecutiveDecision)
    next_memory_state: ContinuousMemoryState = field(default_factory=ContinuousMemoryState)
    queried_child_leaves: tuple[str, ...] = ()
    skipped_child_leaves: tuple[str, ...] = ()

    @property
    def activation_frame(self) -> ActivationFrame:
        return self.propagation_result.direct_frame


def _evaluate_rules_with_inhibition(
    rules: tuple[FuzzyConceptRule, ...],
    coords: Mapping[str, float],
) -> dict[str, float]:
    """Evaluate rules and apply multi-tier Specificity Inhibition (Tier 3 > Tier 2 > Tier 1)."""
    raw_activations = {
        r.concept_name: r.evaluate(coords)
        for r in rules
    }
    active_tiers = [r.tier for r in rules if raw_activations[r.concept_name] >= CONFUSION_THRESHOLD]
    deepest_active_tier = max(active_tiers, default=1)
    max_deeper = max(
        (raw_activations[r.concept_name] for r in rules if r.tier == deepest_active_tier),
        default=0.0,
    )
    rule_activations = {}
    for r in rules:
        val = raw_activations[r.concept_name]
        if r.tier < deepest_active_tier and max_deeper >= CONFUSION_THRESHOLD:
            val = round(val * (1.0 - 0.5 * max_deeper), 6)
        rule_activations[r.concept_name] = val

    # Compensated Soft-Min tie-breaker (Upgrade #2): when 2+ active rules in the deepest tier
    # collide within CONFUSION_THRESHOLD on pure min_similarity, re-evaluate those colliding rules
    # with `soft_min=True` (0.85 * min_sim + 0.15 * mean_sim) BEFORE structural/lateral inhibition
    # so rules with stronger supporting leaves break false ties without overwriting attenuated runner-ups.
    sorted_vals = sorted(rule_activations.values(), reverse=True)
    if len(sorted_vals) >= 2 and sorted_vals[0] >= CONFUSION_THRESHOLD and (sorted_vals[0] - sorted_vals[1]) < CONFUSION_THRESHOLD:
        top_val = sorted_vals[0]
        colliding = [
            r for r in rules
            if r.tier == deepest_active_tier
            and rule_activations[r.concept_name] >= CONFUSION_THRESHOLD
            and (top_val - rule_activations[r.concept_name]) < CONFUSION_THRESHOLD
        ]
        if len(colliding) >= 2:
            for r in colliding:
                rule_activations[r.concept_name] = r.evaluate(coords, soft_min=True)

    # Broad-vs-Specific Intra-Tier Inhibition: when a 3+-leaf specific rule in `deepest_active_tier`
    # fires at `>= CONFUSION_THRESHOLD`, attenuate broader 1–2 leaf posture rules in the same tier
    # so background 2-leaf rules (like `grounded_coordination`) do not tie 3-leaf learned rules.
    max_specific = max(
        (
            rule_activations[r.concept_name]
            for r in rules
            if r.tier == deepest_active_tier and len(r.target_leaves) >= 3
        ),
        default=0.0,
    )
    if max_specific >= CONFUSION_THRESHOLD:
        max_broad_allowed = round(max(0.0, max_specific - CONFUSION_THRESHOLD - 0.05), 6)
        for r in rules:
            if r.tier == deepest_active_tier and len(r.target_leaves) < 3:
                attenuated_broad = round(
                    rule_activations[r.concept_name] * (1.0 - 0.55 * max_specific),
                    6,
                )
                rule_activations[r.concept_name] = min(attenuated_broad, max_broad_allowed)

    # Same-Family Cluster Archetype Winner-Take-All (`hazard_*_cluster` and `coached_*_cluster`):
    # Multiple `hazard_*_cluster` rules firing on the same blunder (or multiple `coached_*_cluster`
    # rules firing on the same sound move) are sub-facets of the same verdict, not opposing concepts.
    # Keep the closest-centroid archetype at full strength and laterally attenuate secondary co-firing
    # sub-facets of the same family, while preserving genuine cross-family collisions (`hazard_` vs `coached_`).
    for fam_prefix in ("hazard_", "coached_"):
        fam_rules = [
            r
            for r in rules
            if r.tier == deepest_active_tier
            and r.concept_name.startswith(fam_prefix)
            and re.sub(r"_s\d+$", "", r.concept_name).endswith("_cluster")
            and rule_activations[r.concept_name] > 0.0
        ]
        if len(fam_rules) >= 2 and max(rule_activations[r.concept_name] for r in fam_rules) >= CONFUSION_THRESHOLD:
            def _cluster_rank_key(rule_obj: FuzzyConceptRule) -> tuple[float, int, float, float]:
                exact_sims = [
                    1.0 - abs(float(coords.get(k, 0.0)) - float(tv))
                    for k, tv in rule_obj.target_leaves.items()
                ]
                mean_exact = sum(exact_sims) / max(1, len(exact_sims))
                min_exact = min(exact_sims, default=0.0)
                return (
                    rule_activations[rule_obj.concept_name],
                    len(rule_obj.target_leaves),
                    round(mean_exact, 6),
                    round(min_exact, 6),
                )

            ranked_fam = sorted(fam_rules, key=_cluster_rank_key, reverse=True)
            winner = ranked_fam[0]
            w_val = rule_activations[winner.concept_name]
            max_sub_allowed = round(max(0.0, w_val - CONFUSION_THRESHOLD - 0.05), 6)
            for ru in ranked_fam[1:]:
                ru_val = rule_activations[ru.concept_name]
                attenuated_ru = round(ru_val * (1.0 - 0.55 * w_val), 6)
                rule_activations[ru.concept_name] = min(attenuated_ru, max_sub_allowed)

    return rule_activations


def _select_active_tier2_leaves(
    coords: Mapping[str, float],
    missing_tier2: tuple[LearnedLeaf, ...],
    salience_estimate: float,
    branch_gate_threshold: float,
) -> tuple[tuple[LearnedLeaf, ...], tuple[LearnedLeaf, ...]]:
    """Open only Tier-2 child leaves under active founding parent branches (with high-salience blind-spot fallback)."""
    if not missing_tier2:
        return (), ()
    max_founding = max((float(coords.get(k, 0.0)) for k in FOUNDATIONAL_LEAVES), default=0.0)
    if max_founding < CONFUSION_THRESHOLD and float(salience_estimate) >= 0.45:
        return missing_tier2, ()
    branch_max = {
        b: max(
            (float(coords.get(k, 0.0)) for k in FOUNDATIONAL_LEAVES if k.startswith(b + "/")),
            default=0.0,
        )
        for b in FOUNDING_PARENT_BRANCHES
    }
    to_query = []
    skipped = []
    for leaf in missing_tier2:
        if branch_max.get(leaf.parent_branch, 0.0) >= branch_gate_threshold:
            to_query.append(leaf)
        else:
            skipped.append(leaf)
    return tuple(to_query), tuple(skipped)


def _has_urgent_or_disruptive_signal(
    effective_coords: Mapping[str, float],
    learned_leaves: tuple[LearnedLeaf, ...],
) -> bool:
    """Check 24-leaf coordinates for disruption, coercive imbalance, or active Tier-2/3 exception leaves."""
    chaos = float(effective_coords.get("philosophical/structure/chaos", 0.0))
    power = float(effective_coords.get("cultural/efficiency/power", 0.0))
    fairness = float(effective_coords.get("cultural/justice/fairness", 0.0))
    learned_active = any(
        leaf.tier >= 2 and float(effective_coords.get(leaf.leaf_id, 0.0)) >= CONFUSION_THRESHOLD
        for leaf in learned_leaves
    )
    if chaos >= 0.45:
        return True
    if power >= 0.65 and fairness <= 0.40:
        return True
    if learned_active:
        return True
    return False


def evaluate_reflection(
    birth_state: BirthState,
    observation: str,
    rules: tuple[FuzzyConceptRule, ...],
    models,
    *,
    event_id: str,
    learned_leaves: tuple[LearnedLeaf, ...] = (),
    memory_state: ContinuousMemoryState | None = None,
    allow_silence: bool = False,
    branch_gating: bool = True,
    branch_gate_threshold: float = BRANCH_GATE_THRESHOLD,
    silence_salience_threshold: float = SILENCE_SALIENCE_THRESHOLD,
) -> ReflectionEvaluation:
    """Run Steps 1–3: map Tier-1/2/3 leaves, update continuous memory, evaluate rules, and drive Will."""
    if not isinstance(rules, tuple):
        raise ValueError("rules must be a tuple of FuzzyConceptRule")
    seen_ids = set()
    for r in rules:
        if not isinstance(r, FuzzyConceptRule):
            raise ValueError("all rules must be FuzzyConceptRule instances")
        if r.rule_id in seen_ids:
            raise ValueError(f"duplicate rule_id: {r.rule_id}")
        seen_ids.add(r.rule_id)

    generation = models.config["generation"]
    # Tier 1: Always score the 12 founding leaves with the invariant Tier-1 prompt
    request = make_slm_request(observation, generation)
    slm_output, trace = models.generate(
        "interpretation",
        request,
        PROMPT_VERSION,
        lambda raw: validate_slm_output(raw, learned_leaves=learned_leaves),
    )

    coords = dict(slm_output["leaf_coordinates"])
    queried_ids: list[str] = []
    skipped_ids: list[str] = []

    missing_learned = tuple(l for l in learned_leaves if l.leaf_id not in coords)
    if missing_learned:
        if not branch_gating:
            tier2_req = make_tier2_child_request(observation, generation, missing_learned)
            tier2_coords, _ = models.generate(
                "interpretation",
                tier2_req,
                TIER2_PROMPT_VERSION,
                lambda raw: validate_tier2_child_output(raw, missing_learned),
            )
            coords.update(tier2_coords)
            queried_ids.extend(l.leaf_id for l in missing_learned)
        else:
            # Stage 2: Branch-gated Tier-2 child leaves
            missing_tier2 = tuple(l for l in missing_learned if l.tier == 2)
            tier2_open, tier2_skip = _select_active_tier2_leaves(
                coords,
                missing_tier2,
                float(slm_output["salience_estimate"]),
                branch_gate_threshold,
            )
            for l in tier2_skip:
                coords[l.leaf_id] = 0.0
                skipped_ids.append(l.leaf_id)
            if tier2_open:
                tier2_req = make_tier2_child_request(observation, generation, tier2_open)
                tier2_coords, _ = models.generate(
                    "interpretation",
                    tier2_req,
                    TIER2_PROMPT_VERSION,
                    lambda raw: validate_tier2_child_output(raw, tier2_open),
                )
                coords.update(tier2_coords)
                queried_ids.extend(l.leaf_id for l in tier2_open)

            # Stage 3: Parent-gated Tier-3 sub-leaves (only open when Tier-2 parent >= CONFUSION_THRESHOLD)
            missing_tier3 = tuple(l for l in missing_learned if l.tier == 3)
            tier3_open = tuple(
                l for l in missing_tier3 if float(coords.get(l.parent_branch, 0.0)) >= CONFUSION_THRESHOLD
            )
            tier3_skip = tuple(
                l for l in missing_tier3 if float(coords.get(l.parent_branch, 0.0)) < CONFUSION_THRESHOLD
            )
            for l in tier3_skip:
                coords[l.leaf_id] = 0.0
                skipped_ids.append(l.leaf_id)
            if tier3_open:
                tier3_req = make_tier2_child_request(observation, generation, tier3_open)
                tier3_coords, _ = models.generate(
                    "interpretation",
                    tier3_req,
                    TIER2_PROMPT_VERSION,
                    lambda raw: validate_tier2_child_output(raw, tier3_open),
                )
                coords.update(tier3_coords)
                queried_ids.extend(l.leaf_id for l in tier3_open)

        slm_output = {
            **slm_output,
            "leaf_coordinates": coords,
        }

    # Continuous-State Memory (Experiential -> Phase & Synchronization)
    retention = 0.0
    resonance = 0.0
    effective_coords = dict(coords)
    raw_rule_activations = _evaluate_rules_with_inhibition(rules, coords)
    raw_max_rule = max(raw_rule_activations.values(), default=0.0)

    if memory_state is not None and memory_state.retained_coordinates:
        retention = round(min(1.0, max(float(v) for v in memory_state.retained_coordinates.values())), 6)
        prev_unscaled = {
            k: min(1.0, float(v) / memory_state.retention_decay)
            for k, v in memory_state.retained_coordinates.items()
        }
        shared_keys = [
            k
            for k in prev_unscaled
            if k in coords
            and k != "philosophical/persistence/continuity"
            and max(float(coords.get(k, 0.0)), prev_unscaled[k]) >= 0.20
        ]
        if shared_keys:
            diffs = [abs(float(coords.get(k, 0.0)) - prev_unscaled[k]) for k in shared_keys]
            continuity_bonus = 0.15 * float(coords.get("philosophical/persistence/continuity", 0.0))
            new_spike_penalty = 0.50 * max(
                (
                    float(coords.get(k, 0.0)) - prev_unscaled.get(k, 0.0)
                    for k in coords
                    if k != "philosophical/persistence/continuity"
                ),
                default=0.0,
            )
            resonance = round(
                min(1.0, max(0.0, 1.0 - (sum(diffs) / len(diffs)) + continuity_bonus - new_spike_penalty)),
                6,
            )
        # If the current observation is a quiet/brief follow-up (raw_max_rule < CONFUSION_THRESHOLD,
        # 0.35 <= salience_estimate < 0.70, and continuity >= 0.35) and resonates with retained working
        # memory (resonance >= 0.55), blend retained coordinates without bleeding onto low-salience chatter.
        if (
            raw_max_rule < CONFUSION_THRESHOLD
            and resonance >= 0.55
            and 0.35 <= float(slm_output["salience_estimate"]) < 0.70
            and float(coords.get("philosophical/persistence/continuity", 0.0)) >= 0.35
        ):
            for k, prev_val in prev_unscaled.items():
                cur_val = float(coords.get(k, 0.0))
                effective_coords[k] = round(max(cur_val, 0.20 * cur_val + 0.80 * prev_val), 4)

    rule_activations = _evaluate_rules_with_inhibition(rules, effective_coords)
    max_rule = max(rule_activations.values(), default=0.0)

    # Step 1: Experiential -> Reflection (Novelty / Salience) & Phase (Valence / Arousal)
    novelty = round(min(1.0, max(0.0, 1.0 - max_rule)), 6)
    max_leaf = max(coords.values(), default=0.0)
    salience = round(
        min(1.0, max(0.0, 0.7 * slm_output["salience_estimate"] + 0.3 * max_leaf)),
        6,
    )
    valence = round(
        min(
            1.0,
            max(
                0.0,
                sum(float(coords.get(k, 0.0)) for k in CONSTRUCTIVE_LEAVES) / len(CONSTRUCTIVE_LEAVES),
            ),
        ),
        6,
    )
    arousal = round(
        min(1.0, max(0.0, 0.5 * slm_output["salience_estimate"] + 0.5 * max_leaf)),
        6,
    )

    # Step 2: Psychological -> Contemplation (Clarity / Confidence) & Truth (Theory / Fact)
    clarity, confidence, matched_concept, is_confused, confusion_reason = compute_contemplation(
        rule_activations
    )
    raw_clarity, raw_confidence, _, raw_confused, _ = compute_contemplation(raw_rule_activations)
    fact = (
        round(min(raw_confidence, raw_clarity), 6)
        if not raw_confused
        else round(0.30 * min(raw_confidence, raw_clarity), 6)
    )
    memory_hypothesis = max(0.0, confidence - raw_confidence)
    partial_hypothesis = round(confidence * (1.0 - fact), 6) if clarity >= 0.35 else 0.0
    theory = round(min(1.0, max(memory_hypothesis, partial_hypothesis)), 6)

    decay = memory_state.retention_decay if memory_state is not None else 0.65
    next_retained = {
        k: round(float(v) * decay, 4)
        for k, v in effective_coords.items()
    }
    next_memory_state = ContinuousMemoryState(
        step_count=(memory_state.step_count + 1) if memory_state is not None else 1,
        retained_coordinates=next_retained,
        last_matched_concept=matched_concept,
        retention_decay=decay,
    )

    # Step 3: Propagate to Psychological -> Will (Intention / Execution) across all 24 founding leaves
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
        WILL_EDGES,
        event_id=event_id,
        source_instance=trace.get("request_sha256", "mock"),
        relationship_source=WILL_RULE_VERSION,
        config=PropagationConfig(passes=2, gain=0.25),
        source="llm",
    )
    final_activations = propagation.activations
    intention = round(final_activations[INTENTION_LEAF], 6)
    execution = round(final_activations[EXECUTION_LEAF], 6)
    urgent_signal = _has_urgent_or_disruptive_signal(effective_coords, learned_leaves)

    if allow_silence and not urgent_signal and (
        salience < silence_salience_threshold or execution < EXECUTION_THRESHOLD
    ):
        needs_teacher = False
    else:
        needs_teacher = execution >= EXECUTION_THRESHOLD or is_confused

    top_concept = (
        max(rule_activations.items(), key=lambda kv: (kv[1], kv[0]))[0]
        if rule_activations and max_rule > 0.0
        else None
    )
    if allow_silence and not urgent_signal and (
        salience < silence_salience_threshold
        or execution < EXECUTION_THRESHOLD
        or (not is_confused and (fact >= CONFUSION_THRESHOLD or theory >= CONFUSION_THRESHOLD))
        or (clarity >= 0.25 and confidence >= 0.40)
    ):
        quiet_concept = matched_concept or (
            top_concept if confidence >= CONFUSION_THRESHOLD or (clarity >= 0.25 and confidence >= 0.40) else None
        )
        executive_decision = ExecutiveDecision(
            mode="STAY_SILENT",
            active_concept=quiet_concept,
            fact=fact,
            theory=theory,
            clarity=clarity,
            confidence=confidence,
            execution=execution,
            rationale=(
                f"Silent background observation (salience={salience:.2f}, urgent=False, "
                f"noted_concept={quiet_concept}, execution={execution:.2f})."
            ),
        )
    elif not is_confused and fact >= CONFUSION_THRESHOLD:
        executive_decision = ExecutiveDecision(
            mode="COMMIT_FAST",
            active_concept=matched_concept,
            fact=fact,
            theory=theory,
            clarity=clarity,
            confidence=confidence,
            execution=execution,
            rationale=f"Direct verified match on {matched_concept} (fact={fact:.2f}, clarity={clarity:.2f}).",
        )
    elif (not is_confused and theory >= CONFUSION_THRESHOLD) or (
        clarity >= 0.25 and confidence >= 0.40
    ):
        hyp_concept = matched_concept or top_concept
        executive_decision = ExecutiveDecision(
            mode="VERIFY_HYPOTHESIS",
            active_concept=hyp_concept,
            fact=fact,
            theory=theory,
            clarity=clarity,
            confidence=confidence,
            execution=execution,
            rationale=f"Working hypothesis on {hyp_concept} (theory={theory:.2f}, fact={fact:.2f}, clarity={clarity:.2f}).",
        )
    else:
        executive_decision = ExecutiveDecision(
            mode="ESCALATE_TO_TEACHER",
            active_concept=None,
            fact=fact,
            theory=theory,
            clarity=clarity,
            confidence=confidence,
            execution=execution,
            rationale=f"Escalate due to {confusion_reason} (clarity={clarity:.2f}, confidence={confidence:.2f}, execution={execution:.2f}).",
        )

    return ReflectionEvaluation(
        observation=observation,
        slm_output=slm_output,
        rule_activations=rule_activations,
        max_rule_activation=max_rule,
        novelty=novelty,
        salience=salience,
        clarity=clarity,
        confidence=confidence,
        intention=intention,
        execution=execution,
        matched_concept=matched_concept,
        is_confused=is_confused,
        confusion_reason=confusion_reason,
        needs_teacher=needs_teacher,
        propagation_result=propagation,
        model_trace=trace,
        valence=valence,
        arousal=arousal,
        resonance=resonance,
        retention=retention,
        theory=theory,
        fact=fact,
        executive_decision=executive_decision,
        next_memory_state=next_memory_state,
        queried_child_leaves=tuple(queried_ids),
        skipped_child_leaves=tuple(skipped_ids),
    )


def teacher_schema(
    allow_child_leaf: bool = False,
    force_child_leaf: bool = False,
    learned_leaves: tuple[LearnedLeaf, ...] = (),
    allowed_parent_branches: tuple[str, ...] = FOUNDING_PARENT_BRANCHES,
) -> dict:
    """JSON schema for the Teacher LLM (Step 3 desperation feedback response)."""
    available_leaves = list(FOUNDATIONAL_LEAVES) + [l.leaf_id for l in learned_leaves]
    props: dict = {
        "concept_name": {"type": "string"},
        "description": {"type": "string"},
        "defining_leaves": {
            "type": "array",
            "minItems": 1 if allow_child_leaf else 2,
            "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["leaf_id", "target_value"],
                "properties": {
                    "leaf_id": {"type": "string", "enum": available_leaves},
                    "target_value": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
            },
        },
    }
    required = ["concept_name", "description", "defining_leaves"]
    if allow_child_leaf:
        required.append("new_child_leaves")
        props["new_child_leaves"] = {
            "type": "array",
            "minItems": 1 if force_child_leaf else 0,
            "maxItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["parent_branch", "leaf_name", "description"],
                "properties": {
                    "parent_branch": {"type": "string", "enum": list(allowed_parent_branches)},
                    "leaf_name": {"type": "string"},
                    "description": {"type": "string"},
                },
            },
        }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": props,
    }


def make_teacher_request(
    evaluation: ReflectionEvaluation,
    existing_rules: tuple[FuzzyConceptRule, ...],
    generation: dict,
    *,
    allow_child_leaf: bool = False,
    force_child_leaf: bool = False,
    learned_leaves: tuple[LearnedLeaf, ...] = (),
    allowed_parent_branches: tuple[str, ...] = FOUNDING_PARENT_BRANCHES,
) -> dict:
    """Build the desperation feedback request when Execution >= EXECUTION_THRESHOLD."""
    coords = evaluation.slm_output["leaf_coordinates"]
    blind_spot = max(coords.values(), default=0.0) < CONFUSION_THRESHOLD
    schema = teacher_schema(
        allow_child_leaf=allow_child_leaf,
        force_child_leaf=force_child_leaf or (allow_child_leaf and blind_spot),
        learned_leaves=learned_leaves,
        allowed_parent_branches=allowed_parent_branches,
    )
    existing_summary = [
        {"concept_name": r.concept_name, "description": r.description, "target_leaves": dict(r.target_leaves)}
        for r in existing_rules
    ]
    child_leaf_instructions = ""
    if allow_child_leaf:
        child_leaf_instructions = (
            "4. new_child_leaves: Propose ONE new specific child classification leaf in `new_child_leaves` "
            "with `parent_branch`, a snake_case `leaf_name` (2-3 words), and a clear one-sentence `description` "
            "that specifically distinguishes this phenomenon from all existing_rules. "
        )
    system_prompt = (
        "You are the Teacher LLM assisting a Language Model Fuzzy Tree that is confused by an unfamiliar situation. "
        "Review the observation, the 12 foundational leaf coordinates measured by the fast SLM, and the existing rules. "
        "Propose ONE new reusable concept rule that characterizes this situation: "
        "1. concept_name: a concise snake_case name (2 to 4 words) distinct from existing concepts. "
        "2. description: a one-sentence explanation of the general concept. "
        "3. defining_leaves: pick 3 or 4 foundational leaves that best distinguish this concept from existing_rules "
        "(choosing leaves where the situation has a clear signature in observed_leaf_coordinates) and set their "
        "target_value in [0.0, 1.0] close to observed_leaf_coordinates so this pattern is recognized next time. "
        f"{child_leaf_instructions}"
        "Return JSON only. /no_think"
    )
    user_payload = {
        "observation": evaluation.observation,
        "confusion_reason": evaluation.confusion_reason,
        "observed_leaf_coordinates": coords,
        "existing_rules": existing_summary,
    }
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "temperature": generation["temperature"],
        "seed": generation["seed"],
        "max_tokens": generation["max_tokens"],
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object", "schema": schema},
    }


def validate_teacher_proposal(
    raw: str,
    evaluation: ReflectionEvaluation,
    existing_rules: tuple[FuzzyConceptRule, ...],
) -> FuzzyConceptRule:
    """Validate the Teacher LLM's proposed rule (without child leaf growth)."""
    rule, _ = validate_teacher_growth_proposal(
        raw,
        evaluation,
        existing_rules,
        learned_leaves=(),
        allow_child_leaf=False,
    )
    return rule


def validate_teacher_growth_proposal(
    raw: str,
    evaluation: ReflectionEvaluation,
    existing_rules: tuple[FuzzyConceptRule, ...],
    learned_leaves: tuple[LearnedLeaf, ...] = (),
    *,
    allow_child_leaf: bool = True,
) -> tuple[FuzzyConceptRule, tuple[LearnedLeaf, ...]]:
    """Validate the Teacher LLM's proposed rule and optional new LearnedLeaf child classification."""
    available_leaves = set(FOUNDATIONAL_LEAVES) | {l.leaf_id for l in learned_leaves}
    try:
        data = strict_json(raw)
        expected_keys = {"concept_name", "description", "defining_leaves"}
        if allow_child_leaf:
            expected_keys.add("new_child_leaves")
        if not isinstance(data, dict) or set(data) != expected_keys:
            raise ValueError("unexpected teacher JSON fields")
        name = data["concept_name"].strip().lower()
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,60}", name):
            raise ValueError(f"concept_name must be snake_case: {name!r}")
        existing_names = {r.concept_name for r in existing_rules}
        if name in existing_names:
            raise ValueError(f"concept_name {name!r} already exists in ruleset")
        desc = data["description"].strip()
        if not desc or len(desc) > 500:
            raise ValueError("invalid description")

        updated_leaves = list(learned_leaves)
        created_leaf: LearnedLeaf | None = None
        if allow_child_leaf:
            new_leaf_items = data["new_child_leaves"]
            if not isinstance(new_leaf_items, list) or len(new_leaf_items) > 1:
                raise ValueError("new_child_leaves must be a list of 0 or 1 item")
            if new_leaf_items:
                item = new_leaf_items[0]
                if not isinstance(item, dict) or set(item) != {"parent_branch", "leaf_name", "description"}:
                    raise ValueError("invalid new_child_leaves item")
                leaf_name_clean = item["leaf_name"].strip().lower().replace(" ", "_")
                created_leaf = LearnedLeaf(
                    parent_branch=item["parent_branch"],
                    leaf_name=leaf_name_clean,
                    description=item["description"].strip(),
                )
                if created_leaf.leaf_id not in available_leaves:
                    updated_leaves.append(created_leaf)
                    available_leaves.add(created_leaf.leaf_id)

        items = data["defining_leaves"]
        min_items = 1 if created_leaf is not None else 2
        if not isinstance(items, list) or not (min_items <= len(items) <= 4):
            raise ValueError(f"defining_leaves must contain {min_items} to 4 leaves")
        target_leaves = {}
        for item in items:
            if not isinstance(item, dict) or set(item) != {"leaf_id", "target_value"}:
                raise ValueError("invalid defining_leaves item")
            leaf_id = item["leaf_id"]
            val = item["target_value"]
            if leaf_id not in available_leaves or leaf_id in target_leaves:
                raise ValueError(f"invalid or duplicate leaf_id: {leaf_id!r}")
            if type(val) not in (int, float) or not math.isfinite(val) or not (0.0 <= val <= 1.0):
                raise ValueError("invalid target_value")
            target_leaves[leaf_id] = round(float(val), 4)

        if created_leaf is not None:
            # Anchor the new Tier-2 child leaf (0.80) with 1 contrasting low founding leaf (<= 0.30)
            # from the Teacher's defining_leaves that matches the observed sensor reading.
            observed = evaluation.slm_output["leaf_coordinates"]
            low_contrasting = {}
            for k, v in target_leaves.items():
                obs_v = float(observed.get(k, 1.0))
                if obs_v <= 0.30 and abs(v - obs_v) <= 0.25:
                    low_contrasting[k] = round(obs_v, 4)
                    break
            target_leaves = {
                created_leaf.leaf_id: 0.80,
                **low_contrasting,
            }

        if max(target_leaves.values()) < 0.50:
            raise ValueError("defining_leaves must include at least one elevated leaf (>= 0.50)")

        candidate = FuzzyConceptRule(
            rule_id=f"rule-{name.replace('_', '-')}",
            concept_name=name,
            description=desc,
            target_leaves=target_leaves,
            low=0.60,
            high=0.85,
        )
        # Verify the proposed rule actually fires on the current confused observation
        eval_coords = dict(evaluation.slm_output["leaf_coordinates"])
        if created_leaf is not None:
            eval_coords[created_leaf.leaf_id] = 0.85
        firing = candidate.evaluate(eval_coords)
        if firing < CONFUSION_THRESHOLD:
            raise ValueError(
                f"teacher proposed rule only fired at {firing} (< {CONFUSION_THRESHOLD}) on the current coordinates"
            )
        return candidate, tuple(updated_leaves)
    except (ValueError, TypeError, KeyError) as exc:
        raise ModelError(f"Invalid teacher rule proposal: {exc}") from None


def request_teacher_rule(
    evaluation: ReflectionEvaluation,
    existing_rules: tuple[FuzzyConceptRule, ...],
    teacher_models,
    *,
    role: str = "teacher",
) -> tuple[FuzzyConceptRule, dict]:
    """Step 3: If `evaluation.needs_teacher` is True, ask the Teacher LLM for a new Tier-1 rule."""
    if not evaluation.needs_teacher:
        raise ValueError("Execution activation is below threshold; teacher request not warranted")
    generation = teacher_models.config["generation"]
    req = make_teacher_request(evaluation, existing_rules, generation, allow_child_leaf=False)
    new_rule, trace = teacher_models.generate(
        role,
        req,
        TEACHER_PROMPT_VERSION,
        lambda raw: validate_teacher_proposal(raw, evaluation, existing_rules),
    )
    return new_rule, trace


def request_teacher_growth(
    evaluation: ReflectionEvaluation,
    existing_rules: tuple[FuzzyConceptRule, ...],
    learned_leaves: tuple[LearnedLeaf, ...],
    teacher_models,
    *,
    role: str = "teacher",
) -> tuple[FuzzyConceptRule, tuple[LearnedLeaf, ...], dict]:
    """Two-Tier Step 3: Try Tier-1 founding rule first; step down to Tier-2 child leaf on blind spot or collision."""
    if not evaluation.needs_teacher:
        raise ValueError("Execution activation is below threshold; teacher request not warranted")
    coords = evaluation.slm_output["leaf_coordinates"]
    founding_coords = {k: v for k, v in coords.items() if k in FOUNDATIONAL_LEAVES}
    blind_spot = max(founding_coords.values(), default=0.0) < CONFUSION_THRESHOLD
    collision = evaluation.confusion_reason.startswith("ambiguous_rule_collision")
    existing_tier1_fires = any(
        r.tier == 1 and r.evaluate(founding_coords) >= CONFUSION_THRESHOLD
        for r in existing_rules
    )

    active_tier2_branches = tuple(
        l.leaf_id
        for l in learned_leaves
        if l.tier == 2 and float(coords.get(l.leaf_id, 0.0)) >= CONFUSION_THRESHOLD
    )
    # If the 12 founding leaves have a clear signal and no existing Tier-1 or Tier-2 rule already fires,
    # try a Tier-1 rule first.
    if not blind_spot and not collision and not existing_tier1_fires and not active_tier2_branches:
        try:
            tier1_rule, trace = request_teacher_rule(
                evaluation,
                existing_rules,
                teacher_models,
                role=role,
            )
            return tier1_rule, learned_leaves, trace
        except ModelError:
            pass

    # Step down one tier: if a Tier-2 leaf is already active, split it into a Tier-3 sub-leaf;
    # otherwise grow a Tier-2 child leaf under one of the 6 founding parent branches.
    allowed_parent_branches = active_tier2_branches if active_tier2_branches else FOUNDING_PARENT_BRANCHES
    generation = teacher_models.config["generation"]
    req = make_teacher_request(
        evaluation,
        existing_rules,
        generation,
        allow_child_leaf=True,
        force_child_leaf=True,
        learned_leaves=learned_leaves,
        allowed_parent_branches=allowed_parent_branches,
    )
    (new_rule, updated_leaves), trace = teacher_models.generate(
        role,
        req,
        TEACHER_PROMPT_VERSION + "-tier2-growth",
        lambda raw: validate_teacher_growth_proposal(
            raw,
            evaluation,
            existing_rules,
            learned_leaves=learned_leaves,
            allow_child_leaf=True,
        ),
    )
    return new_rule, updated_leaves, trace


def make_executive_response_request(
    evaluation: ReflectionEvaluation,
    generation: dict,
) -> dict:
    """Build the closed-loop prompt where the Fuzzy Tree's ExecutiveDecision governs the 4B SLM response."""
    dec = evaluation.executive_decision
    expected_concept = dec.active_concept or "unclassified"
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["executive_mode", "active_concept", "spoken_response"],
        "properties": {
            "executive_mode": {"type": "string", "enum": [dec.mode]},
            "active_concept": {"type": "string", "enum": [expected_concept]},
            "spoken_response": {"type": "string"},
        },
    }
    system_prompt = (
        "You are the language expression module governed by a Language Model Fuzzy Tree's Executive Decision. "
        "You must obey the tree's `executive_mode` strictly:\n"
        "- COMMIT_FAST (high Fact & Clarity): State the recognized `active_concept` directly as an established "
        "fact in the current observation and recommend a direct action.\n"
        "- VERIFY_HYPOTHESIS (high Theory, low standalone Fact): Frame `active_concept` explicitly as a tentative "
        "working hypothesis (inferred from recent context or partial evidence) and recommend verifying before committing.\n"
        "- ESCALATE_TO_TEACHER (low Clarity or low Confidence): Do NOT guess a concept; state plainly that the "
        "situation is unfamiliar or ambiguous and is being escalated to the Teacher LLM.\n"
        "Keep `spoken_response` to 1 or 2 clear English sentences. Return JSON only. /no_think"
    )
    user_payload = {
        "observation": evaluation.observation,
        "executive_decision": {
            "mode": dec.mode,
            "active_concept": expected_concept,
            "fact": dec.fact,
            "theory": dec.theory,
            "clarity": dec.clarity,
            "confidence": dec.confidence,
            "rationale": dec.rationale,
        },
    }
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "temperature": generation["temperature"],
        "seed": generation["seed"],
        "max_tokens": generation["max_tokens"],
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object", "schema": schema},
    }


def validate_executive_response(raw: str, evaluation: ReflectionEvaluation) -> dict:
    """Validate that the 4B SLM response obeys the Fuzzy Tree's ExecutiveDecision."""
    dec = evaluation.executive_decision
    expected_concept = dec.active_concept or "unclassified"
    try:
        data = strict_json(raw)
        if not isinstance(data, dict) or set(data) != {"executive_mode", "active_concept", "spoken_response"}:
            raise ValueError("unexpected executive response JSON fields")
        if data["executive_mode"] != dec.mode:
            raise ValueError("executive_mode mismatch")
        if data["active_concept"] != expected_concept:
            raise ValueError("active_concept mismatch")
        spoken = data["spoken_response"]
        if not isinstance(spoken, str) or not spoken.strip():
            raise ValueError("spoken_response must be a nonempty string")
        return {
            "executive_mode": data["executive_mode"],
            "active_concept": data["active_concept"],
            "spoken_response": spoken.strip(),
        }
    except (ValueError, TypeError, KeyError) as exc:
        raise ModelError(f"Invalid executive response output: {exc}") from None


def generate_executive_response(
    evaluation: ReflectionEvaluation,
    slm_models,
) -> tuple[dict, dict]:
    """Generate a 4B spoken response governed by the Fuzzy Tree's ExecutiveDecision (0 chars on STAY_SILENT)."""
    if evaluation.executive_decision.mode == "STAY_SILENT":
        return (
            {
                "executive_mode": "STAY_SILENT",
                "active_concept": evaluation.executive_decision.active_concept or "unclassified",
                "spoken_response": "",
            },
            {"role": "silent_gate", "prompt_version": EXECUTIVE_PROMPT_VERSION, "llm_calls": 0},
        )
    generation = slm_models.config["generation"]
    req = make_executive_response_request(evaluation, generation)
    return slm_models.generate(
        "interpretation",
        req,
        EXECUTIVE_PROMPT_VERSION,
        lambda raw: validate_executive_response(raw, evaluation),
    )

