"""Three Right-Brain Branch Archives (Executive, Legislative, Judicial) & 3-Classifier Deliberation.

Plain-English architecture:
1. Executive Archive (`cultural_history.json` — Do-er: `Will <-> Cultural`):
   Tracks social history, access precedents (`OPEN_SHARED`, `GATED_CONDITIONAL`,
   `SEALED_EXCLUSIONARY`), social-drift trajectories across scenes, and reusable
   social-duty / action clauses.
2. Legislative Archive (`experiential_felt_diary.json` — Feeler: `Contemplation <-> Experiential`):
   Stores sensory snapshots, complex feeling definitions (`austere_sanctuary`,
   `hollow_stillness`, `bittersweet_release`, `earned_return`), and emotional
   chords that originate the push for new rules when lived feeling is unsettled.
3. Judicial Archive (`philosophical_lexicon.json` — Thinker: `Truth [Fact / Theory] <-> Philosophical`):
   Acts as the constitutional case-law lexicon separating hard external `Fact`
   from candidate `Theory`, storing reusable structural definitions and ratified
   multi-branch chords.
4. Right-Brain Audience With Itself (`deliberate_across_three_branches`):
   Runs a fast (`~15-30 ms`, 0 LLM calls) 3-classifier committee that recombines
   clauses from the 3 archives into a candidate theoretical action bid, pushes it
   into `Will -> Intention / Execution` via `propagate_activation`, and checks
   whether all 3 branches (Executive, Legislative, Judicial) ratify the arrangement.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

from .birth import BirthState
from .chord_memory import ChordEchoReport, FeltChord, humanize_feeling_name
from .contemplation_ledger import ContemplationLedger
from .propagation import PropagationConfig, propagate_activation
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
    FuzzyConceptRule,
    LearnedLeaf,
    is_valid_leaf_id,
)
from .runtime_store import atomic_json

EXECUTIVE_BRANCH = "executive_cultural"
LEGISLATIVE_BRANCH = "legislative_experiential"
JUDICIAL_BRANCH = "judicial_philosophical"

ACCESS_STATES = (
    "OPEN_SHARED",
    "GATED_CONDITIONAL",
    "SEALED_EXCLUSIONARY",
    "NEUTRAL",
)


def _split_sentences(text: str) -> list[str]:
    cleaned = " ".join(text.strip().split())
    parts = [p.strip() for p in cleaned.replace("!", ".").replace("?", ".").split(".") if p.strip()]
    return [p + "." for p in parts] if parts else [cleaned]


def _compact_clause(text: str, limit: int = 155) -> str:
    cleaned = " ".join(text.strip().split())
    if len(cleaned) > limit:
        return cleaned[: limit - 3].rstrip() + "..."
    return cleaned


def _branch_subset(coords: Mapping[str, float], prefix: str) -> dict[str, float]:
    return {
        k: round(float(v), 4)
        for k, v in coords.items()
        if k.startswith(prefix) and is_valid_leaf_id(k)
    }


def _coord_similarity(a: Mapping[str, float], b: Mapping[str, float], min_active: float = 0.22) -> float:
    keys = [
        k for k in (set(a.keys()) | set(b.keys()))
        if max(float(a.get(k, 0.0)), float(b.get(k, 0.0))) >= min_active
    ]
    if not keys:
        return 0.50
    mean_diff = sum(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in keys) / len(keys)
    return round(max(0.0, min(1.0, 1.0 - mean_diff)), 4)


def classify_access_state(coordinates: Mapping[str, float]) -> str:
    """Classify a scene's social access posture from its Right-Brain coordinates."""
    fairness = float(coordinates.get("cultural/justice/fairness", 0.0))
    discreteness = float(coordinates.get("philosophical/persistence/discreteness", 0.0))
    continuity = float(coordinates.get("philosophical/persistence/continuity", 0.0))
    order = float(coordinates.get("philosophical/structure/order", 0.0))
    valence = float(coordinates.get(VALENCE_LEAF, 0.50))

    # Any active Tier-2/3 child leaf under cultural/justice when fairness is not high signals justice friction
    justice_child_active = any(
        float(v) >= 0.35
        for k, v in coordinates.items()
        if k.startswith("cultural/justice/")
        and k not in ("cultural/justice/fairness", "cultural/justice/honor")
    )

    if fairness >= 0.65 and not justice_child_active and (continuity >= 0.45 or discreteness < 0.72 or valence >= 0.55):
        return "OPEN_SHARED"
    if fairness <= 0.38 or justice_child_active:
        return "SEALED_EXCLUSIONARY"
    if order >= 0.70 and discreteness >= 0.65 and fairness < 0.65:
        return "GATED_CONDITIONAL"
    return "NEUTRAL"


@dataclass(frozen=True)
class BranchClauseEntry:
    """One reusable definition or precedent clause stored in an Executive, Legislative, or Judicial archive."""

    clause_id: str
    branch: str
    text: str
    signature_coords: dict[str, float]
    source_scene: int
    concept_tag: str

    def __post_init__(self):
        if self.branch not in (EXECUTIVE_BRANCH, LEGISLATIVE_BRANCH, JUDICIAL_BRANCH):
            raise ValueError(f"invalid branch: {self.branch!r}")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("clause text must be a nonempty string")
        for k, v in self.signature_coords.items():
            if not is_valid_leaf_id(k):
                raise ValueError(f"invalid signature leaf_id: {k!r}")
            if type(v) not in (int, float) or not math.isfinite(v) or not (0.0 <= v <= 1.0):
                raise ValueError(f"signature coordinate {k!r} must be in [0, 1]")

    def to_dict(self) -> dict:
        return {
            "clause_id": self.clause_id,
            "branch": self.branch,
            "text": self.text,
            "signature_coords": dict(self.signature_coords),
            "source_scene": self.source_scene,
            "concept_tag": self.concept_tag,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "BranchClauseEntry":
        return cls(
            clause_id=str(data["clause_id"]),
            branch=str(data["branch"]),
            text=str(data["text"]),
            signature_coords={str(k): float(v) for k, v in dict(data["signature_coords"]).items()},
            source_scene=int(data["source_scene"]),
            concept_tag=str(data["concept_tag"]),
        )


@dataclass(frozen=True)
class CulturalPrecedentEntry:
    """One chronological social-studies record in `cultural_history.json`."""

    scene_index: int
    access_state: str
    action_summary: str
    cultural_coords: dict[str, float]
    order_level: float
    discreteness_level: float

    def __post_init__(self):
        if self.access_state not in ACCESS_STATES:
            raise ValueError(f"invalid access_state: {self.access_state!r}")

    def to_dict(self) -> dict:
        return {
            "scene_index": self.scene_index,
            "access_state": self.access_state,
            "action_summary": self.action_summary,
            "cultural_coords": dict(self.cultural_coords),
            "order_level": round(float(self.order_level), 4),
            "discreteness_level": round(float(self.discreteness_level), 4),
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "CulturalPrecedentEntry":
        return cls(
            scene_index=int(data["scene_index"]),
            access_state=str(data["access_state"]),
            action_summary=str(data["action_summary"]),
            cultural_coords={str(k): float(v) for k, v in dict(data["cultural_coords"]).items()},
            order_level=float(data.get("order_level", 0.0)),
            discreteness_level=float(data.get("discreteness_level", 0.0)),
        )


@dataclass(frozen=True)
class SocialDriftReport:
    """Result of checking `cultural_history.json` for open-to-locked institutional drift."""

    drift_detected: bool
    prior_open_scene: int | None = None
    prior_lockdown_scene: int | None = None
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "drift_detected": self.drift_detected,
            "prior_open_scene": self.prior_open_scene,
            "prior_lockdown_scene": self.prior_lockdown_scene,
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class ExecutiveCulturalArchive:
    """Executive (`Do-er: Will <-> Cultural`) archive stored in `cultural_history.json`."""

    precedents: tuple[CulturalPrecedentEntry, ...] = ()
    action_clauses: tuple[BranchClauseEntry, ...] = ()
    max_entries: int = 24

    def detect_social_drift(self, current_coordinates: Mapping[str, float]) -> SocialDriftReport:
        """Detect when a formerly open shared space has been gated/sealed and remains locked."""
        if len(self.precedents) < 2:
            return SocialDriftReport(drift_detected=False)

        open_scenes = [p.scene_index for p in self.precedents if p.access_state == "OPEN_SHARED"]
        if not open_scenes:
            return SocialDriftReport(drift_detected=False)
        first_open = open_scenes[0]

        lockdown_scenes = [
            p.scene_index
            for p in self.precedents
            if p.scene_index > first_open and p.access_state in ("GATED_CONDITIONAL", "SEALED_EXCLUSIONARY")
        ]
        if not lockdown_scenes:
            return SocialDriftReport(drift_detected=False)

        # Check if a later precedent already reopened the space before the current turn
        last_lockdown = lockdown_scenes[-1]
        reopened_since = any(
            p.scene_index > last_lockdown and p.access_state == "OPEN_SHARED"
            for p in self.precedents
        )
        if reopened_since:
            return SocialDriftReport(drift_detected=False)

        cur_disc = float(current_coordinates.get("philosophical/persistence/discreteness", 0.0))
        cur_order = float(current_coordinates.get("philosophical/structure/order", 0.0))
        cur_chaos = float(current_coordinates.get("philosophical/structure/chaos", 0.0))
        cur_cont = float(current_coordinates.get("philosophical/persistence/continuity", 0.0))
        cur_fair = float(current_coordinates.get("cultural/justice/fairness", 0.0))

        # If current moment breaks the barrier open or restores high open sharing, drift ends
        if cur_chaos >= 0.45 or (cur_fair >= 0.70 and cur_cont >= 0.55) or (cur_fair >= 0.78 and cur_disc < 0.55):
            return SocialDriftReport(drift_detected=False)

        if cur_disc >= 0.65 and cur_order >= 0.60:
            return SocialDriftReport(
                drift_detected=True,
                prior_open_scene=first_open,
                prior_lockdown_scene=last_lockdown,
                explanation=(
                    f"Social history precedent: this space was open and shared in Scene {first_open}, "
                    f"became gated/sealed by Scene {last_lockdown}, and remains enclosed without reopening."
                ),
            )
        return SocialDriftReport(drift_detected=False)

    def record_scene(
        self,
        *,
        scene_index: int,
        coordinates: Mapping[str, float],
        action_summary: str,
        harvested_clause: BranchClauseEntry | None = None,
    ) -> "ExecutiveCulturalArchive":
        state = classify_access_state(coordinates)
        entry = CulturalPrecedentEntry(
            scene_index=scene_index,
            access_state=state,
            action_summary=_compact_clause(action_summary),
            cultural_coords=_branch_subset(coordinates, "cultural/"),
            order_level=float(coordinates.get("philosophical/structure/order", 0.0)),
            discreteness_level=float(coordinates.get("philosophical/persistence/discreteness", 0.0)),
        )
        kept_prec = tuple(p for p in self.precedents if p.scene_index != scene_index) + (entry,)
        kept_clauses = self.action_clauses
        if harvested_clause is not None:
            kept_clauses = tuple(
                c for c in self.action_clauses if c.clause_id != harvested_clause.clause_id
            ) + (harvested_clause,)
        return ExecutiveCulturalArchive(
            precedents=kept_prec[-self.max_entries :],
            action_clauses=kept_clauses[-self.max_entries :],
            max_entries=self.max_entries,
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "branch": EXECUTIVE_BRANCH,
            "max_entries": self.max_entries,
            "precedents": [p.to_dict() for p in self.precedents],
            "action_clauses": [c.to_dict() for c in self.action_clauses],
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "ExecutiveCulturalArchive":
        if data.get("schema_version") != 1:
            raise ValueError("unsupported cultural_history schema_version")
        return cls(
            precedents=tuple(CulturalPrecedentEntry.from_dict(p) for p in data.get("precedents", ())),
            action_clauses=tuple(BranchClauseEntry.from_dict(c) for c in data.get("action_clauses", ())),
            max_entries=int(data.get("max_entries", 24)),
        )


@dataclass(frozen=True)
class ExperientialDiaryEntry:
    """One sensory & emotional chord record in `experiential_felt_diary.json`."""

    scene_index: int
    sensory_snapshot: str
    felt_chord: FeltChord
    experiential_coords: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "scene_index": self.scene_index,
            "sensory_snapshot": self.sensory_snapshot,
            "felt_chord": self.felt_chord.to_dict(),
            "experiential_coords": dict(self.experiential_coords),
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "ExperientialDiaryEntry":
        return cls(
            scene_index=int(data["scene_index"]),
            sensory_snapshot=str(data["sensory_snapshot"]),
            felt_chord=FeltChord.from_dict(data["felt_chord"]),
            experiential_coords={str(k): float(v) for k, v in dict(data["experiential_coords"]).items()},
        )


@dataclass(frozen=True)
class LegislativeExperientialArchive:
    """Legislative (`Feeler: Contemplation <-> Experiential`) archive in `experiential_felt_diary.json`."""

    diary_snapshots: tuple[ExperientialDiaryEntry, ...] = ()
    feeling_clauses: tuple[BranchClauseEntry, ...] = ()
    max_entries: int = 24

    def record_scene(
        self,
        *,
        scene_index: int,
        sensory_snapshot: str,
        felt_chord: FeltChord,
        coordinates: Mapping[str, float],
        harvested_clause: BranchClauseEntry | None = None,
    ) -> "LegislativeExperientialArchive":
        exp_coords = _branch_subset(coordinates, "experiential/")
        exp_coords[SALIENCE_LEAF] = felt_chord.salience
        exp_coords[VALENCE_LEAF] = felt_chord.valence
        exp_coords[AROUSAL_LEAF] = felt_chord.arousal
        for k, v in felt_chord.complex_notes.items():
            if is_valid_leaf_id(k):
                exp_coords[k] = round(float(v), 4)

        snap = ExperientialDiaryEntry(
            scene_index=scene_index,
            sensory_snapshot=_compact_clause(sensory_snapshot),
            felt_chord=felt_chord,
            experiential_coords=exp_coords,
        )
        kept_snaps = tuple(s for s in self.diary_snapshots if s.scene_index != scene_index) + (snap,)
        kept_clauses = self.feeling_clauses
        if harvested_clause is not None:
            kept_clauses = tuple(
                c for c in self.feeling_clauses if c.clause_id != harvested_clause.clause_id
            ) + (harvested_clause,)
        return LegislativeExperientialArchive(
            diary_snapshots=kept_snaps[-self.max_entries :],
            feeling_clauses=kept_clauses[-self.max_entries :],
            max_entries=self.max_entries,
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "branch": LEGISLATIVE_BRANCH,
            "max_entries": self.max_entries,
            "diary_snapshots": [s.to_dict() for s in self.diary_snapshots],
            "feeling_clauses": [c.to_dict() for c in self.feeling_clauses],
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "LegislativeExperientialArchive":
        if data.get("schema_version") != 1:
            raise ValueError("unsupported experiential_felt_diary schema_version")
        return cls(
            diary_snapshots=tuple(
                ExperientialDiaryEntry.from_dict(s) for s in data.get("diary_snapshots", ())
            ),
            feeling_clauses=tuple(
                BranchClauseEntry.from_dict(c) for c in data.get("feeling_clauses", ())
            ),
            max_entries=int(data.get("max_entries", 24)),
        )


@dataclass(frozen=True)
class JudicialPhilosophicalArchive:
    """Judicial (`Thinker: Truth [Fact/Theory] <-> Philosophical`) archive in `philosophical_lexicon.json`."""

    structural_clauses: tuple[BranchClauseEntry, ...] = ()
    ratified_chords: tuple[FuzzyConceptRule, ...] = ()
    max_entries: int = 24

    def record_clause_and_chord(
        self,
        *,
        harvested_clause: BranchClauseEntry | None = None,
        ratified_chord: FuzzyConceptRule | None = None,
    ) -> "JudicialPhilosophicalArchive":
        kept_clauses = self.structural_clauses
        if harvested_clause is not None:
            kept_clauses = tuple(
                c for c in self.structural_clauses if c.clause_id != harvested_clause.clause_id
            ) + (harvested_clause,)
        kept_chords = self.ratified_chords
        if ratified_chord is not None:
            kept_chords = tuple(
                r for r in self.ratified_chords if r.concept_name != ratified_chord.concept_name
            ) + (ratified_chord,)
        return JudicialPhilosophicalArchive(
            structural_clauses=kept_clauses[-self.max_entries :],
            ratified_chords=kept_chords[-self.max_entries :],
            max_entries=self.max_entries,
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "branch": JUDICIAL_BRANCH,
            "max_entries": self.max_entries,
            "structural_clauses": [c.to_dict() for c in self.structural_clauses],
            "ratified_chords": [r.to_dict() for r in self.ratified_chords],
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "JudicialPhilosophicalArchive":
        if data.get("schema_version") != 1:
            raise ValueError("unsupported philosophical_lexicon schema_version")
        return cls(
            structural_clauses=tuple(
                BranchClauseEntry.from_dict(c) for c in data.get("structural_clauses", ())
            ),
            ratified_chords=tuple(
                FuzzyConceptRule.from_dict(r) for r in data.get("ratified_chords", ())
            ),
            max_entries=int(data.get("max_entries", 24)),
        )


@dataclass(frozen=True)
class ThreeBranchArchives:
    """Bundle of all 3 Right-Brain sub-seed archives (Executive, Legislative, Judicial)."""

    executive: ExecutiveCulturalArchive = ExecutiveCulturalArchive()
    legislative: LegislativeExperientialArchive = LegislativeExperientialArchive()
    judicial: JudicialPhilosophicalArchive = JudicialPhilosophicalArchive()

    def save_all(self, directory: str | Path) -> dict[str, Path]:
        base = Path(directory)
        base.mkdir(parents=True, exist_ok=True)
        exec_path = base / "cultural_history.json"
        leg_path = base / "experiential_felt_diary.json"
        jud_path = base / "philosophical_lexicon.json"
        atomic_json(exec_path, self.executive.to_dict())
        atomic_json(leg_path, self.legislative.to_dict())
        atomic_json(jud_path, self.judicial.to_dict())
        return {
            "cultural_history": exec_path,
            "experiential_felt_diary": leg_path,
            "philosophical_lexicon": jud_path,
        }

    @classmethod
    def load_all(cls, directory: str | Path) -> "ThreeBranchArchives":
        base = Path(directory)
        exec_path = base / "cultural_history.json"
        leg_path = base / "experiential_felt_diary.json"
        jud_path = base / "philosophical_lexicon.json"
        return cls(
            executive=(
                ExecutiveCulturalArchive.from_dict(json.loads(exec_path.read_text(encoding="utf-8")))
                if exec_path.exists()
                else ExecutiveCulturalArchive()
            ),
            legislative=(
                LegislativeExperientialArchive.from_dict(json.loads(leg_path.read_text(encoding="utf-8")))
                if leg_path.exists()
                else LegislativeExperientialArchive()
            ),
            judicial=(
                JudicialPhilosophicalArchive.from_dict(json.loads(jud_path.read_text(encoding="utf-8")))
                if jud_path.exists()
                else JudicialPhilosophicalArchive()
            ),
        )


@dataclass(frozen=True)
class ThreeClassifierDeliberation:
    """Result of the LLM-free 3-Classifier Deliberation ('Right-Brain Pseudo-Self-Talk')."""

    ratified: bool
    deliberation_ms: float
    candidate_bid_text: str | None
    ratified_concept: str | None
    executive_clause: BranchClauseEntry | None
    legislative_clause: BranchClauseEntry | None
    judicial_clause: BranchClauseEntry | None
    executive_vote: bool
    legislative_vote: bool
    judicial_vote: bool
    executive_similarity: float
    legislative_similarity: float
    judicial_similarity: float
    tested_intention: float
    tested_execution: float
    harmony_score: float
    reason: str
    bid_coordinates: dict[str, float]
    bid_experiential: dict[str, float]
    newly_ratified_chord: FuzzyConceptRule | None = None

    def to_dict(self) -> dict:
        return {
            "ratified": self.ratified,
            "deliberation_ms": round(self.deliberation_ms, 2),
            "candidate_bid_text": self.candidate_bid_text,
            "ratified_concept": self.ratified_concept,
            "executive_clause": self.executive_clause.to_dict() if self.executive_clause else None,
            "legislative_clause": self.legislative_clause.to_dict() if self.legislative_clause else None,
            "judicial_clause": self.judicial_clause.to_dict() if self.judicial_clause else None,
            "votes": {
                "executive_doer": self.executive_vote,
                "legislative_feeler": self.legislative_vote,
                "judicial_thinker": self.judicial_vote,
            },
            "similarities": {
                "executive": self.executive_similarity,
                "legislative": self.legislative_similarity,
                "judicial": self.judicial_similarity,
            },
            "tested_intention": round(self.tested_intention, 4),
            "tested_execution": round(self.tested_execution, 4),
            "harmony_score": round(self.harmony_score, 4),
            "reason": self.reason,
        }


def _executive_clause_compatible(
    scene_cult: Mapping[str, float],
    clause_sig: Mapping[str, float],
) -> bool:
    s_fair = float(scene_cult.get("cultural/justice/fairness", 0.50))
    c_fair = float(clause_sig.get("cultural/justice/fairness", 0.50))
    if (s_fair <= 0.38) != (c_fair <= 0.38):
        return False
    return abs(s_fair - c_fair) <= 0.45


def _legislative_clause_compatible(
    scene_exp: Mapping[str, float],
    clause_sig: Mapping[str, float],
) -> bool:
    s_val = float(scene_exp.get(VALENCE_LEAF, 0.50))
    c_val = float(clause_sig.get(VALENCE_LEAF, 0.50))
    if (s_val <= 0.42) != (c_val <= 0.42):
        return False
    return abs(s_val - c_val) <= 0.42


def _judicial_clause_compatible(
    scene_phil: Mapping[str, float],
    clause_sig: Mapping[str, float],
) -> bool:
    s_chaos = float(scene_phil.get("philosophical/structure/chaos", 0.0))
    c_chaos = float(clause_sig.get("philosophical/structure/chaos", 0.0))

    scene_fracture = s_chaos >= 0.38
    clause_fracture = c_chaos >= 0.38
    if scene_fracture != clause_fracture:
        return False
    return True


def _best_matching_clause(
    clauses: tuple[BranchClauseEntry, ...],
    branch_coords: Mapping[str, float],
    compatibility_fn=None,
) -> tuple[BranchClauseEntry | None, float]:
    best: BranchClauseEntry | None = None
    best_sim = 0.0
    for clause in clauses:
        if compatibility_fn is not None and not compatibility_fn(branch_coords, clause.signature_coords):
            continue
        sim = _coord_similarity(branch_coords, clause.signature_coords)
        if sim > best_sim:
            best_sim = sim
            best = clause
    return best, best_sim


def build_three_branch_chord_rule(
    *,
    scene_index: int,
    concept_name: str,
    description: str,
    coordinates: Mapping[str, float],
    felt_chord: FeltChord | None = None,
) -> FuzzyConceptRule | None:
    """Build a 3-branch FuzzyConceptRule strictly from stored archive clause coordinates."""
    cult = _branch_subset(coordinates, "cultural/")
    phil = _branch_subset(coordinates, "philosophical/")
    exp_candidates = {
        k: v
        for k, v in _branch_subset(coordinates, "experiential/").items()
        if k not in (SALIENCE_LEAF, VALENCE_LEAF, AROUSAL_LEAF, RESONANCE_LEAF, RETENTION_LEAF, NOVELTY_LEAF)
    }
    if not exp_candidates and felt_chord is not None:
        exp_candidates = dict(felt_chord.complex_notes)

    target_leaves: dict[str, float] = {}

    # 1. Executive (Cultural) anchor: select strongest extreme or salient leaf in cultural/
    low_cult = sorted((kv for kv in cult.items() if kv[1] <= 0.38), key=lambda kv: (kv[1], kv[0]))
    high_cult = sorted((kv for kv in cult.items() if kv[1] >= 0.50), key=lambda kv: (-kv[1], kv[0]))
    if low_cult:
        target_leaves[low_cult[0][0]] = round(min(0.28, low_cult[0][1]), 2)
        for k, v in high_cult:
            if k not in FOUNDATIONAL_LEAVES and v >= 0.45:
                target_leaves[k] = round(min(0.65, v), 2)
                break
    elif high_cult:
        target_leaves[high_cult[0][0]] = round(min(0.75, high_cult[0][1]), 2)

    # 2. Judicial (Philosophical) anchor: pick strongest structural leaf from the archived Judicial clause
    chaos = float(phil.get("philosophical/structure/chaos", 0.0))
    if chaos >= 0.38:
        target_leaves["philosophical/structure/chaos"] = round(min(0.68, chaos), 2)
    else:
        top_phil = sorted(phil.items(), key=lambda kv: (-kv[1], kv[0]))
        if top_phil and top_phil[0][1] >= 0.50:
            target_leaves[top_phil[0][0]] = round(min(0.75, top_phil[0][1]), 2)

    # 3. Legislative (Experiential) anchor: pick strongest complex feeling drawer from the archived Legislative clause
    top_exp = sorted(exp_candidates.items(), key=lambda kv: (-kv[1], kv[0]))
    if top_exp and top_exp[0][1] >= 0.48:
        target_leaves[top_exp[0][0]] = round(min(0.65, top_exp[0][1]), 2)

    if len(target_leaves) < 2:
        return None

    return FuzzyConceptRule(
        rule_id=f"rule-3branch-s{scene_index}-{concept_name[:28]}",
        concept_name=concept_name,
        description=_compact_clause(description, limit=180),
        target_leaves=target_leaves,
        low=0.48,
        high=0.72,
    )


def deliberate_across_three_branches(
    birth_state: BirthState,
    external_coords: Mapping[str, float],
    external_exp: Mapping[str, float],
    felt_chord: FeltChord,
    sensors,
    archives: ThreeBranchArchives,
    will_edges: tuple,
    will_version: str,
    *,
    scene_index: int = 1,
    learned_leaves: tuple[LearnedLeaf, ...] = (),
    social_drift: SocialDriftReport | None = None,
    min_branch_similarity: float = 0.62,
    branch_gating: bool = True,
) -> ThreeClassifierDeliberation:
    """Run LLM-free 3-Classifier Deliberation across the Executive, Legislative, and Judicial archives."""
    t0 = time.perf_counter()

    # Use raw, unmodified neural sensor coordinates (never clamp or overwrite sensor scores)
    cult_now = _branch_subset(external_coords, "cultural/")
    phil_now = _branch_subset(external_coords, "philosophical/")
    exp_now = _branch_subset(external_coords, "experiential/")
    exp_now[SALIENCE_LEAF] = felt_chord.salience
    exp_coords_with_notes = {**exp_now, **felt_chord.complex_notes}

    exec_clause, exec_sim = _best_matching_clause(
        archives.executive.action_clauses,
        cult_now,
        compatibility_fn=_executive_clause_compatible,
    )
    leg_clause, leg_sim = _best_matching_clause(
        archives.legislative.feeling_clauses,
        exp_coords_with_notes,
        compatibility_fn=_legislative_clause_compatible,
    )
    jud_clause, jud_sim = _best_matching_clause(
        archives.judicial.structural_clauses,
        phil_now,
        compatibility_fn=_judicial_clause_compatible,
    )

    if exec_clause is None or leg_clause is None or jud_clause is None:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        exec_v = exec_clause is not None and exec_sim >= min_branch_similarity
        leg_v = leg_clause is not None and leg_sim >= min_branch_similarity
        jud_v = jud_clause is not None and jud_sim >= min_branch_similarity
        reason_str = (
            "branch_archives_incomplete_need_left_brain_hail"
            if (exec_clause is None and leg_clause is None and jud_clause is None)
            else f"three_branch_deadlock(exec={exec_v}:{exec_sim:.2f},leg={leg_v}:{leg_sim:.2f},jud={jud_v}:{jud_sim:.2f})"
        )
        return ThreeClassifierDeliberation(
            ratified=False,
            deliberation_ms=dt_ms,
            candidate_bid_text=None,
            ratified_concept=None,
            executive_clause=exec_clause,
            legislative_clause=leg_clause,
            judicial_clause=jud_clause,
            executive_vote=exec_v,
            legislative_vote=leg_v,
            judicial_vote=jud_v,
            executive_similarity=exec_sim,
            legislative_similarity=leg_sim,
            judicial_similarity=jud_sim,
            tested_intention=0.0,
            tested_execution=0.0,
            harmony_score=round((exec_sim + leg_sim + jud_sim) / 6.0, 4),
            reason=reason_str,
            bid_coordinates={},
            bid_experiential={},
        )

    # Assemble candidate theoretical action bid by recombining the 3 branch clauses
    seen_texts: list[str] = []
    for cl in (exec_clause, jud_clause, leg_clause):
        if cl.text not in seen_texts:
            seen_texts.append(cl.text)
    candidate_bid_text = " ".join(seen_texts)

    # Score the recombined candidate bid across the 3 neural classifiers (~15-25 ms, 0 LLM calls)
    bid_coords, bid_exp, _q, _s = sensors.score_text(
        candidate_bid_text,
        learned_leaves=learned_leaves,
        branch_gating=branch_gating,
    )

    # Test candidate arrangement into Will -> Intention / Execution via synchronous propagation
    founding_bid = {k: float(bid_coords.get(k, 0.0)) for k in FOUNDATIONAL_LEAVES}
    harmony_est = round((exec_sim + leg_sim + jud_sim) / 3.0, 4)
    proposed = {
        **founding_bid,
        VALENCE_LEAF: float(bid_exp[VALENCE_LEAF]),
        AROUSAL_LEAF: float(bid_exp[AROUSAL_LEAF]),
        RESONANCE_LEAF: harmony_est,
        RETENTION_LEAF: harmony_est,
        NOVELTY_LEAF: round(max(0.0, 1.0 - harmony_est), 4),
        SALIENCE_LEAF: max(felt_chord.salience, float(bid_exp[SALIENCE_LEAF])),
        CLARITY_LEAF: harmony_est,
        CONFIDENCE_LEAF: harmony_est,
        THEORY_LEAF: harmony_est,
        FACT_LEAF: round(min(exec_sim, jud_sim), 4),
        INTENTION_LEAF: 0.0,
        EXECUTION_LEAF: 0.0,
    }
    prop = propagate_activation(
        birth_state,
        proposed,
        will_edges,
        event_id=f"scene-{scene_index}-delib",
        source_instance="right-brain-3branch-deliberation",
        relationship_source=will_version,
        config=PropagationConfig(passes=2, gain=0.25),
        source="developer",
    )
    tested_intention = float(prop.activations[INTENTION_LEAF])
    tested_execution = float(prop.activations[EXECUTION_LEAF])

    # 3-Branch Constitutional Votes:
    # 1. Executive (Do-er): archive precedent similarity + active intention
    #    If social history drift is active (ongoing enclosure of a formerly shared space),
    #    the Executive branch vetoes innocent/open action clauses that ignore the sealed gate.
    clause_state = classify_access_state(exec_clause.signature_coords)
    scene_state = classify_access_state(external_coords)
    drift_conflict = bool(
        social_drift is not None
        and social_drift.drift_detected
        and (clause_state != "SEALED_EXCLUSIONARY" or scene_state != "SEALED_EXCLUSIONARY")
    )
    exec_vote = exec_sim >= min_branch_similarity and tested_intention >= 0.12 and not drift_conflict

    # 2. Legislative (Feeler): emotional chord similarity + no tone-deaf valence whitewashing
    scene_fair = float(cult_now.get("cultural/justice/fairness", 0.50))
    bid_fair = float(bid_coords.get("cultural/justice/fairness", 0.50))
    leg_vote = leg_sim >= min_branch_similarity and not (
        felt_chord.valence <= 0.38 and float(bid_exp[VALENCE_LEAF]) >= 0.72 and bid_fair >= 0.70
    )

    # 3. Judicial (Thinker): structural similarity + Fact-vs-Theory consistency (no self-rationalization)
    dissonance = scene_fair <= 0.28 and bid_fair >= 0.72
    jud_vote = jud_sim >= min_branch_similarity and not dissonance

    ratified = bool(exec_vote and leg_vote and jud_vote)

    tags = []
    for cl in (exec_clause, jud_clause, leg_clause):
        if cl.concept_tag not in tags:
            tags.append(cl.concept_tag)
    synthesized_name = "_".join(tags[:2]) if tags else f"ratified_chord_s{scene_index}"

    new_chord: FuzzyConceptRule | None = None
    if ratified:
        # Build candidate chord strictly from the stored coordinates of the 3 retrieved archive clauses
        # (never from the current scene's own coordinates)
        archived_chord_coords = {
            **exec_clause.signature_coords,
            **jud_clause.signature_coords,
            **leg_clause.signature_coords,
        }
        new_chord = build_three_branch_chord_rule(
            scene_index=scene_index,
            concept_name=synthesized_name,
            description=candidate_bid_text,
            coordinates=archived_chord_coords,
        )

    dt_ms = (time.perf_counter() - t0) * 1000.0
    reason = (
        f"three_branch_ratified({synthesized_name})"
        if ratified
        else f"three_branch_deadlock(exec={exec_vote}:{exec_sim:.2f},leg={leg_vote}:{leg_sim:.2f},jud={jud_vote}:{jud_sim:.2f})"
    )

    return ThreeClassifierDeliberation(
        ratified=ratified,
        deliberation_ms=dt_ms,
        candidate_bid_text=candidate_bid_text,
        ratified_concept=synthesized_name if ratified else None,
        executive_clause=exec_clause,
        legislative_clause=leg_clause,
        judicial_clause=jud_clause,
        executive_vote=exec_vote,
        legislative_vote=leg_vote,
        judicial_vote=jud_vote,
        executive_similarity=exec_sim,
        legislative_similarity=leg_sim,
        judicial_similarity=jud_sim,
        tested_intention=tested_intention,
        tested_execution=tested_execution,
        harmony_score=harmony_est if ratified else round(0.5 * harmony_est, 4),
        reason=reason,
        bid_coordinates=dict(bid_coords),
        bid_experiential=dict(bid_exp),
        newly_ratified_chord=new_chord,
    )


@dataclass(frozen=True)
class OpenThreadTrigger:
    """Indicates whether an open Left-Brain ActiveUncertainty demands verbal self-talk closure."""

    triggered: bool
    target_scene_index: int | None = None
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "triggered": self.triggered,
            "target_scene_index": self.target_scene_index,
            "reason": self.reason,
        }


def check_open_thread_trigger(
    ledger: ContemplationLedger,
    chord_echoes: ChordEchoReport,
    felt_chord: FeltChord,
    coordinates: Mapping[str, float],
    *,
    social_drift: SocialDriftReport | None = None,
) -> OpenThreadTrigger:
    """Check if the current scene directly answers or flips an open Left-Brain uncertainty."""
    if not ledger.active_uncertainties:
        return OpenThreadTrigger(triggered=False)
    if social_drift is not None and social_drift.drift_detected:
        return OpenThreadTrigger(triggered=False)

    fairness = float(coordinates.get("cultural/justice/fairness", 0.0))
    if fairness < 0.65:
        return OpenThreadTrigger(triggered=False)

    open_scenes = {u.scene_index: u for u in ledger.active_uncertainties}

    if chord_echoes.chord_inversion is not None and chord_echoes.chord_inversion.scene_index in open_scenes:
        target = chord_echoes.chord_inversion.scene_index
        return OpenThreadTrigger(
            triggered=True,
            target_scene_index=target,
            reason=f"Scene chord sharply inverts open Left-Brain uncertainty from Scene {target}.",
        )

    # General positive experiential shift or restorative resolution
    has_positive_complex_note = any(
        float(v) >= 0.48 for v in felt_chord.complex_notes.values()
    ) and felt_chord.valence >= 0.55
    if has_positive_complex_note:
        latest_u = ledger.active_uncertainties[-1]
        return OpenThreadTrigger(
            triggered=True,
            target_scene_index=latest_u.scene_index,
            reason=(
                f"Restorative shift resolves the open Left-Brain tension carried from Scene {latest_u.scene_index}."
            ),
        )

    return OpenThreadTrigger(triggered=False)


def _derive_concept_tags(
    coordinates: Mapping[str, float],
    felt_chord: FeltChord,
) -> tuple[str, str, str]:
    """Derive plain-English branch concept tags from observed coordinates without hardcoded scene words."""
    cult = _branch_subset(coordinates, "cultural/")
    phil = _branch_subset(coordinates, "philosophical/")

    low_cult = sorted(
        (kv for kv in cult.items() if kv[0] in FOUNDATIONAL_LEAVES and kv[1] <= 0.35),
        key=lambda kv: (kv[1], kv[0]),
    )
    high_cult = sorted(cult.items(), key=lambda kv: (-kv[1], kv[0]))
    if low_cult:
        exec_tag = f"low_{low_cult[0][0].split('/')[-1].replace('-', '_')}"
    elif high_cult and high_cult[0][1] >= 0.55:
        exec_tag = f"high_{high_cult[0][0].split('/')[-1].replace('-', '_')}"
    else:
        exec_tag = "moderate_cultural_balance"

    high_phil = sorted(phil.items(), key=lambda kv: (-kv[1], kv[0]))
    if high_phil and high_phil[0][1] >= 0.50:
        top_p_name = high_phil[0][0].split("/")[-1].replace("-", "_")
        if len(high_phil) >= 2 and high_phil[1][1] >= 0.55:
            sec_p_name = high_phil[1][0].split("/")[-1].replace("-", "_")
            jud_tag = f"{top_p_name}_{sec_p_name}"
        else:
            jud_tag = f"high_{top_p_name}"
    else:
        jud_tag = "moderate_structure"

    if felt_chord.complex_notes:
        top_note = max(felt_chord.complex_notes.items(), key=lambda kv: kv[1])
        if top_note[1] >= 0.45:
            leg_tag = top_note[0].split("/")[-1]
        else:
            leg_tag = "warm_relief" if felt_chord.valence >= 0.58 else "felt_chill"
    else:
        leg_tag = "warm_relief" if felt_chord.valence >= 0.58 else "felt_chill"

    return exec_tag, leg_tag, jud_tag


def harvest_into_branch_archives(
    archives: ThreeBranchArchives,
    *,
    scene_index: int,
    scenario_text: str,
    coordinates: Mapping[str, float],
    felt_chord: FeltChord,
    left_brain_reply: str | None,
    settled: bool = False,
    deliberation: ThreeClassifierDeliberation | None = None,
    social_drift: SocialDriftReport | None = None,
) -> ThreeBranchArchives:
    """File a scene's precedent and—ONLY when settled—harvested clauses into the 3 branch archives."""
    raw_coords = dict(coordinates)
    exec_tag, leg_tag, jud_tag = _derive_concept_tags(raw_coords, felt_chord)

    exec_clause: BranchClauseEntry | None = None
    leg_clause: BranchClauseEntry | None = None
    jud_clause: BranchClauseEntry | None = None
    ratified_chord: FuzzyConceptRule | None = (
        deliberation.newly_ratified_chord if (settled and deliberation is not None) else None
    )

    # Never harvest unsettled/rejected drafts into the reusable clause libraries;
    # unsettled scenes stay in contemplation_ledger.json's active_uncertainties for Sleep Mode.
    if settled:
        has_left_reply = left_brain_reply is not None and bool(left_brain_reply.strip())
        clause_source_text = left_brain_reply.strip() if has_left_reply else scenario_text.strip()
        if clause_source_text:
            sentences = _split_sentences(clause_source_text)
            s_exec = sentences[0]
            s_jud = sentences[1] if len(sentences) >= 2 else sentences[0]
            s_leg = f"Felt chord of {humanize_feeling_name(leg_tag)}: {felt_chord.dominant_mood}."

            cult_sig = _branch_subset(raw_coords, "cultural/")
            phil_sig = _branch_subset(raw_coords, "philosophical/")
            exp_sig = {
                **_branch_subset(raw_coords, "experiential/"),
                SALIENCE_LEAF: felt_chord.salience,
                VALENCE_LEAF: felt_chord.valence,
                AROUSAL_LEAF: felt_chord.arousal,
                **felt_chord.complex_notes,
            }

            exec_clause = BranchClauseEntry(
                clause_id=f"exec-s{scene_index}-{exec_tag}",
                branch=EXECUTIVE_BRANCH,
                text=_compact_clause(s_exec),
                signature_coords=cult_sig,
                source_scene=scene_index,
                concept_tag=exec_tag,
            )
            leg_clause = BranchClauseEntry(
                clause_id=f"leg-s{scene_index}-{leg_tag}",
                branch=LEGISLATIVE_BRANCH,
                text=_compact_clause(s_leg),
                signature_coords=exp_sig,
                source_scene=scene_index,
                concept_tag=leg_tag,
            )
            jud_clause = BranchClauseEntry(
                clause_id=f"jud-s{scene_index}-{jud_tag}",
                branch=JUDICIAL_BRANCH,
                text=_compact_clause(s_jud),
                signature_coords=phil_sig,
                source_scene=scene_index,
                concept_tag=jud_tag,
            )
            if ratified_chord is None and has_left_reply:
                ratified_chord = build_three_branch_chord_rule(
                    scene_index=scene_index,
                    concept_name=f"{exec_tag}_{leg_tag}",
                    description=f"{s_exec} {s_jud}",
                    coordinates={**cult_sig, **phil_sig, **exp_sig},
                )

    next_exec = archives.executive.record_scene(
        scene_index=scene_index,
        coordinates=raw_coords,
        action_summary=left_brain_reply if left_brain_reply else scenario_text,
        harvested_clause=exec_clause,
    )
    next_leg = archives.legislative.record_scene(
        scene_index=scene_index,
        sensory_snapshot=scenario_text,
        felt_chord=felt_chord,
        coordinates=raw_coords,
        harvested_clause=leg_clause,
    )
    next_jud = archives.judicial.record_clause_and_chord(
        harvested_clause=jud_clause,
        ratified_chord=ratified_chord,
    )
    return ThreeBranchArchives(
        executive=next_exec,
        legislative=next_leg,
        judicial=next_jud,
    )
