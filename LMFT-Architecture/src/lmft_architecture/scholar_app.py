"""Standalone App #3: `LMFT-Scholar.exe` — Foundational Two-Hemisphere Socratic & Curriculum Studio (`scholar_app.py`).

Plain-English Purpose:
Dedicated standalone desktop application for developing a general-purpose LMFT brain
using native 16-bit `SicariusSicariiStuff/Impish_LLAMA_4B` (`hidden_size = 3072`) and the
frozen 40-node Right-Brain Fuzzy Tree before or alongside specialized domains like
`LMFT-Chess` and `LMFT-Bloodmoon`.

Provides:
1. **Foundational 4-Department Curriculum (`SCHOLAR_CURRICULUM_LESSONS`)**:
   - Curated Socratic dilemmas and contemplative passages spanning all four founding
     Right-Brain departments (`Cultural`, `Philosophical`, `Experiential`, `Psychological`).
2. **Reading & Study Passage Canvas + Socratic Dialogue Stream**:
   - Contemplate any pasted text or Socratic question through the 3-legged runtime
     (`Pass-1 Fact -> Pass-2 Theory -> Subliminal Gap Left-Brain Generation`), Savor
     settled insights into the 3-Branch Archives (`Executive`, `Legislative`, `Judicial`),
     or Coach/Challenge turns into the REM Sleep Queue.
3. **Offline REM Sleep Consolidation + Brain Graduation (`graduate_brain_to_target`)**:
   - Consolidates queued Socratic and Curriculum scenes into persistent domain-general
     `FuzzyConceptRule` entries, dynamic Tier-2/3 `LearnedLeaf` nodes, sculpted
     `will_edges.json`, and Callosum projector updates (`live_projector_weights.pt`).
   - Exports ("Graduates") learned domain-general rules, leaves, 3-Branch Archives,
     Will edges, and Callosum projector weights directly into `LMFT-Chess` or
     `LMFT-Bloodmoon`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Mapping
import webbrowser

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")


def _ensure_valid_stdio_streams() -> None:
    if sys.stdout is None:
        sys.stdout = io.StringIO()
    if sys.stderr is None:
        sys.stderr = io.StringIO()


_ensure_valid_stdio_streams()

from .birth import BirthState, create_birth_state
from .branch_archives import (
    ThreeBranchArchives,
    harvest_into_branch_archives,
)
from .chord_memory import ChordMemoryBank, compile_felt_chord
from .contemplation_ledger import (
    ContemplationLedger,
    load_contemplation_ledger,
    save_contemplation_ledger,
)
from .dilemma_crucible import CruciblePairedBridgeSensors
from .gap_trajectory_schema import ContrastiveGapPair, chord_l2_distance
from .impish_bloodmoon_adapter import (
    DEFAULT_IMPISH_BLOODMOON_HF_ID,
    UNSTEERED_BOOTSTRAP_STATUS,
)
from .neural_bridge_classifiers import (
    DEFAULT_COMPLEX_FEELING_LEAVES,
    NeuralThreeBridgeSuite,
)
from .philosophical_abstraction_matrix import PHILOSOPHICAL_SHADOW_LEAVES
from .propagation import Association
from .reflection_classifier import (
    AROUSAL_LEAF,
    FOUNDATIONAL_LEAVES,
    SALIENCE_LEAF,
    VALENCE_LEAF,
    ContinuousMemoryState,
    FuzzyConceptRule,
    LearnedLeaf,
    load_learned_leaves,
    load_ruleset,
    save_ruleset,
)
from .right_brain_equilibrium import (
    RIGHT_BRAIN_WILL_EDGES,
    RIGHT_BRAIN_WILL_VERSION,
    SettledDiary,
    ThreeBridgeSensors,
    compile_right_brain_vibe,
)
from .runtime_store import atomic_json
from .scholar_ui import SCHOLAR_SINGLE_PAGE_HTML
from .sleep_consolidation import (
    UnresolvedSleepScene,
    apply_dream_reflection_to_archives,
    consolidate_sleep_clusters_locally,
    extract_dream_vow_sentence,
)
from .subliminal_gap_splice import SubliminalChordVector
from .three_legged_runtime import (
    SubliminalGapLeftBrain,
    ThreeLeggedTurnResult,
    compute_stanza_residual_chord,
    generate_rem_dream_self_reflection,
    run_three_legged_turn,
)
from .train_left_brain_lora import update_projector_on_harvested_gap_pairs
from .virgin_baseline import VIRGIN_FOUNDATIONAL_RULES, load_virgin_baseline_state
from .workspace_paths import resolve_training_root

WillEdgeSpec = Association


@dataclass(frozen=True)
class ScholarCurriculumLesson:
    """One domain-general Socratic lesson spanning the 4 founding Right-Brain departments."""

    lesson_id: str
    department: str
    title: str
    passage_text: str
    socratic_prompt: str

    def to_dict(self) -> dict[str, str]:
        return {
            "lesson_id": self.lesson_id,
            "department": self.department,
            "title": self.title,
            "passage_text": self.passage_text,
            "socratic_prompt": self.socratic_prompt,
        }


SCHOLAR_CURRICULUM_LESSONS: tuple[ScholarCurriculumLesson, ...] = (
    ScholarCurriculumLesson(
        lesson_id="cultural_efficiency_stewardship",
        department="Cultural · Efficiency",
        title="Stewardship vs. Reckless Force (Power & Wastelessness)",
        passage_text=(
            "A master builder can fell an ancient oak in ten minutes to bridge a narrow stream, "
            "or spend an hour assembling fallen fieldstone into a crossing that outlasts generations "
            "without sacrificing the living canopy."
        ),
        socratic_prompt=(
            "Contemplate how genuine power harmonizes with wastelessness when choosing between "
            "immediate force and patient stewardship."
        ),
    ),
    ScholarCurriculumLesson(
        lesson_id="cultural_justice_reciprocity",
        department="Cultural · Justice",
        title="Reciprocity Under Scarcity (Honor & Fairness)",
        passage_text=(
            "Two travelers share a single canteen across an arid plain. When the stronger traveler "
            "takes only an equal measure while keeping watch through the cold night, trust deepens "
            "into a covenant stronger than any written contract."
        ),
        socratic_prompt=(
            "Reflect on how honor and fairness reinforce each other when asymmetric strength meets shared vulnerability."
        ),
    ),
    ScholarCurriculumLesson(
        lesson_id="cultural_freedom_autonomy",
        department="Cultural · Freedom",
        title="Inner Sufficiency vs. Compulsive Craving (Self-Sufficiency & Needlessness)",
        passage_text=(
            "A scholar who walks through a bustling bazaar admiring the craftsmanship of every vessel "
            "without needing to possess any of them enjoys the richness of the world while remaining "
            "unfettered in spirit."
        ),
        socratic_prompt=(
            "Examine the relationship between self-sufficiency and needlessness when encountering abundance or temptation."
        ),
    ),
    ScholarCurriculumLesson(
        lesson_id="philosophical_being_agency",
        department="Philosophical · Being",
        title="Deliberate Agency Within Given Existence (Agency & Existence)",
        passage_text=(
            "A sailor cannot command the direction of the ocean gale, yet by setting the angle of the "
            "sail and holding the rudder steady, transforms the indifferent wind into deliberate passage."
        ),
        socratic_prompt=(
            "Reflect on how conscious agency operates within the unchosen facts of existence."
        ),
    ),
    ScholarCurriculumLesson(
        lesson_id="philosophical_structure_harmony",
        department="Philosophical · Structure",
        title="Living Order Amid Creative Turbulence (Order & Chaos)",
        passage_text=(
            "When a mountain river floods in spring, rigid embankments crack under the surge, whereas "
            "terraced floodplains absorb the turbulent water and turn the silt into fertile soil."
        ),
        socratic_prompt=(
            "Contemplate how resilient order adapts to structural chaos without collapsing into rigidity or disorder."
        ),
    ),
    ScholarCurriculumLesson(
        lesson_id="philosophical_persistence_rhythm",
        department="Philosophical · Persistence",
        title="Enduring Thread Through Distinct Moments (Continuity & Discreteness)",
        passage_text=(
            "A melody consists of discrete notes separated by silence, yet the listener hears a single "
            "continuous phrase unfolding across time."
        ),
        socratic_prompt=(
            "Explore how continuity and discreteness coexist in memory, vows, and purposeful action."
        ),
    ),
    ScholarCurriculumLesson(
        lesson_id="experiential_resonance_lucidity",
        department="Experiential · Synchronization",
        title="Empathetic Resonance Without Losing Center (Resonance & Entrainment)",
        passage_text=(
            "A seasoned mediator listens deeply to an angry crowd, feeling the heat and urgency of their "
            "grievance while keeping a calm, unhurried breath that gradually steadies the room."
        ),
        socratic_prompt=(
            "Reflect on how one can maintain high empathetic resonance without surrendering to involuntary emotional entrainment."
        ),
    ),
    ScholarCurriculumLesson(
        lesson_id="psychological_truth_calibration",
        department="Psychological · Truth & Will",
        title="Aligning Inner Theory With Observed Fact (Fact, Theory, Intention & Execution)",
        passage_text=(
            "An astronomer whose cherished orbital chart fails to predict an eclipse does not blame the sky; "
            "instead, the discrepancy becomes the exact compass that refines the theory and sharpens the next observation."
        ),
        socratic_prompt=(
            "Examine how humility before observed fact transforms cognitive dissonance into clearer theory and steadier execution."
        ),
    ),
)


def _default_fallback_sensors() -> ThreeBridgeSensors:
    """Deterministic fallback sensors used only when neural bridge files are absent (e.g. unit tests)."""

    def _will(text: str, children: tuple[LearnedLeaf, ...] = ()) -> dict[str, float]:
        low = text.lower()
        out = {
            "cultural/efficiency/power": 0.72 if "power" in low or "force" in low else 0.56,
            "cultural/efficiency/wastelessness": 0.74 if "steward" in low or "wasteless" in low else 0.62,
            "cultural/justice/honor": 0.72 if "honor" in low or "covenant" in low else 0.60,
            "cultural/justice/fairness": 0.74 if "fair" in low or "equal" in low else 0.62,
            "cultural/freedom/self-sufficiency": 0.72 if "sufficien" in low or "rudder" in low else 0.60,
            "cultural/freedom/needlessness": 0.68 if "unfettered" in low or "calm" in low else 0.44,
        }
        for c in children:
            out[c.leaf_id] = 0.52
        return out

    def _truth(text: str, children: tuple[LearnedLeaf, ...] = ()) -> dict[str, float]:
        low = text.lower()
        out = {
            "philosophical/being/agency": 0.70 if "agency" in low or "sail" in low else 0.60,
            "philosophical/being/existence": 0.76,
            "philosophical/structure/order": 0.74 if "order" in low or "harmony" in low else 0.68,
            "philosophical/structure/chaos": 0.58 if "flood" in low or "turbulen" in low or "chaos" in low else 0.22,
            "philosophical/persistence/continuity": 0.76 if "continu" in low or "endur" in low else 0.66,
            "philosophical/persistence/discreteness": 0.62 if "discrete" in low or "distinct" in low else 0.50,
        }
        for c in children:
            out[c.leaf_id] = 0.48
        return out

    def _feeler(text: str, children: tuple[LearnedLeaf, ...] = ()) -> dict[str, float]:
        out = {
            SALIENCE_LEAF: 0.62,
            VALENCE_LEAF: 0.72,
            AROUSAL_LEAF: 0.42,
            "experiential/reflection/novelty": 0.48,
            "experiential/synchronization/resonance": 0.72,
            "experiential/synchronization/entrainment": 0.54,
        }
        for c in children:
            out[c.leaf_id] = 0.45
        return out

    return ThreeBridgeSensors(
        will_cultural=_will,
        truth_philosophical=_truth,
        contemplation_experiential=_feeler,
    )


def _ensure_foundational_rules(rules: tuple[FuzzyConceptRule, ...]) -> tuple[FuzzyConceptRule, ...]:
    by_id = {r.rule_id: r for r in rules}
    merged: list[FuzzyConceptRule] = []
    for vr in VIRGIN_FOUNDATIONAL_RULES:
        merged.append(by_id.get(vr.rule_id, vr))
    virgin_ids = {vr.rule_id for vr in VIRGIN_FOUNDATIONAL_RULES}
    for r in rules:
        if r.rule_id not in virgin_ids:
            merged.append(r)
    return tuple(merged)


def _save_learned_leaves_json(path: Path, leaves: tuple[LearnedLeaf, ...]) -> None:
    builtin_ids = {
        lf.leaf_id for lf in (PHILOSOPHICAL_SHADOW_LEAVES + DEFAULT_COMPLEX_FEELING_LEAVES)
    }
    custom_leaves = [lf.to_dict() for lf in leaves if lf.leaf_id not in builtin_ids]
    atomic_json(
        path,
        {
            "version": "learned_leaves_v1",
            "leaves": custom_leaves,
        },
    )


def _load_or_init_scholar_state(
    state_dir: Path,
) -> tuple[
    BirthState,
    tuple[FuzzyConceptRule, ...],
    tuple[LearnedLeaf, ...],
    ThreeBranchArchives,
    ContemplationLedger,
    tuple[WillEdgeSpec, ...],
]:
    state_dir.mkdir(parents=True, exist_ok=True)
    birth_state = create_birth_state(
        "lmft-scholar",
        birth_event_id="scholar-birth-001",
        random_seed=42,
    )

    rules_path = state_dir / "learned_rules.json"
    if rules_path.exists():
        loaded_rules = load_ruleset(rules_path)
    else:
        loaded_rules = ()
    rules = _ensure_foundational_rules(loaded_rules)

    leaves_path = state_dir / "learned_leaves.json"
    custom_leaves: tuple[LearnedLeaf, ...] = ()
    if leaves_path.exists():
        try:
            custom_leaves = load_learned_leaves(leaves_path)
        except Exception:
            custom_leaves = ()
    elif rules_path.exists():
        try:
            custom_leaves = load_learned_leaves(rules_path)
        except Exception:
            custom_leaves = ()
    base_leaves = PHILOSOPHICAL_SHADOW_LEAVES + DEFAULT_COMPLEX_FEELING_LEAVES
    seen_ids = {lf.leaf_id for lf in base_leaves}
    merged_leaves = list(base_leaves)
    for clf in custom_leaves:
        if clf.leaf_id not in seen_ids:
            merged_leaves.append(clf)
            seen_ids.add(clf.leaf_id)

    if not rules_path.exists():
        save_ruleset(rules_path, rules, tuple(merged_leaves))

    archives_dir = state_dir / "branch_archives"
    if (archives_dir / "philosophical_lexicon.json").exists():
        archives = ThreeBranchArchives.load_all(archives_dir)
    else:
        archives = ThreeBranchArchives()
        archives.save_all(archives_dir)

    ledger_path = state_dir / "contemplation_ledger.json"
    if ledger_path.exists():
        ledger = load_contemplation_ledger(ledger_path)
    else:
        ledger = ContemplationLedger()
        save_contemplation_ledger(ledger_path, ledger)

    will_edges_path = state_dir / "will_edges.json"
    will_edges: tuple[WillEdgeSpec, ...] = RIGHT_BRAIN_WILL_EDGES
    if will_edges_path.exists():
        try:
            raw_will = json.loads(will_edges_path.read_text(encoding="utf-8"))
            weights_map = raw_will.get("weights", {}) if isinstance(raw_will, dict) else {}
            if weights_map:
                loaded_edges: list[WillEdgeSpec] = []
                for e in RIGHT_BRAIN_WILL_EDGES:
                    w = float(weights_map.get(e.id, e.weight))
                    loaded_edges.append(
                        Association(
                            e.id,
                            e.source_id,
                            e.target_id,
                            w,
                            e.confidence,
                            e.plasticity,
                            e.created_event_id,
                            e.last_updated_event_id,
                            e.active,
                            e.kind,
                        )
                    )
                will_edges = tuple(loaded_edges)
        except Exception:
            will_edges = RIGHT_BRAIN_WILL_EDGES

    return birth_state, rules, tuple(merged_leaves), archives, ledger, will_edges


class ScholarStudioRuntime:
    """Stateful runtime for `LMFT-Scholar.exe` — Foundational Two-Hemisphere Socratic & Curriculum Studio."""

    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        state_dir: Path | None = None,
        scholar_checkpoint_dir: Path | None = None,
        sensors: ThreeBridgeSensors | None = None,
        scholar_left_brain: Any | None = None,
        lazy_load_gpu: bool = False,
        auto_rem_sleep: bool = True,
        auto_sleep_threshold: int = 2,
    ) -> None:
        self._lock = threading.RLock()
        if repo_root is not None:
            self.repo_root = Path(repo_root)
        else:
            self.repo_root = resolve_training_root()

        self.state_dir = (
            Path(state_dir)
            if state_dir is not None
            else (self.repo_root / "outputs" / "scholar_state")
        )
        self.scholar_checkpoint_dir = (
            Path(scholar_checkpoint_dir)
            if scholar_checkpoint_dir is not None
            else (self.repo_root / "models" / "left_brain_scholar")
        )
        self._ensure_scholar_checkpoint_seeded()

        (
            self.birth_state,
            self.rules,
            self.learned_leaves,
            self.archives,
            self.ledger,
            self.will_edges,
        ) = _load_or_init_scholar_state(self.state_dir)

        if sensors is not None:
            self.sensors = sensors
        else:
            bridge_dir = self.repo_root / "models" / "right_brain_bridges"
            if bridge_dir.exists():
                try:
                    suite = NeuralThreeBridgeSuite.load_all(bridge_dir)
                    self.sensors = suite.as_three_bridge_sensors()
                except Exception:
                    self.sensors = _default_fallback_sensors()
            else:
                self.sensors = _default_fallback_sensors()

        self._scholar_left_brain = scholar_left_brain
        self._lazy_load_gpu = bool(lazy_load_gpu)
        self.left_brain_status = "UNLOADED"
        self.active_checkpoint_label = "Checkpoint: models/left_brain_scholar (Impish_LLAMA_4B)"
        if self._scholar_left_brain is not None:
            self.left_brain_status = "READY (Injected Left-Brain)"
        elif self._lazy_load_gpu:
            self.left_brain_status = "STANDBY (Impish_LLAMA_4B Ready)"
        else:
            self._load_scholar_locked(self.scholar_checkpoint_dir)

        self.active_lineage: str = "scholar"
        self.auto_rem_sleep = bool(auto_rem_sleep)
        self.auto_sleep_threshold = max(1, int(auto_sleep_threshold))
        self.memory: ContinuousMemoryState | None = None
        self.chord_bank = ChordMemoryBank()
        self.diary = SettledDiary()
        self.scene_counter = 0
        self.curriculum_step_counter = 0
        self.sleep_cycle_index = 0
        self.harvested_gap_pairs_count = 0
        self.sleep_queue: list[UnresolvedSleepScene] = []
        self.savored_turn_ids: list[int] = []
        self._turn_records_by_id: dict[int, dict[str, Any]] = {}
        self._last_minted_leaves: tuple[LearnedLeaf, ...] = ()
        self._last_harvested_gap_pairs: list[ContrastiveGapPair] = []
        self.rem_dream_chronicles: list[dict[str, Any]] = []
        self.last_rem_sleep_receipt: str = ""
        self.last_graduation_receipt: str = ""

        self.study_canvas_text: str = SCHOLAR_CURRICULUM_LESSONS[0].passage_text
        self.chat_messages: list[dict[str, Any]] = [
            {
                "sender": "companion",
                "role_label": "LMFT-Scholar · Socratic Companion",
                "text": (
                    "Welcome to LMFT-Scholar. Select a Foundational Curriculum lesson above, "
                    "paste a passage into the Reading Canvas, or ask a Socratic question to "
                    "develop the 40-node Right Brain and Callosum."
                ),
            }
        ]
        self.threads: list[dict[str, Any]] = [
            {
                "thread_id": "thread-1",
                "title": "Study Session 1",
                "messages": list(self.chat_messages),
                "study_canvas_text": self.study_canvas_text,
            }
        ]
        self.active_thread_id: str = "thread-1"
        self._load_persisted_session_if_present()

    def _ensure_scholar_checkpoint_seeded(self) -> None:
        """Seed `models/left_brain_scholar/` from `models/left_brain_baseline/` if not yet created."""
        if (self.scholar_checkpoint_dir / "projector_weights.pt").exists():
            return
        baseline_dir = self.repo_root / "models" / "left_brain_baseline"
        if not (baseline_dir / "projector_weights.pt").exists():
            return
        self.scholar_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for item in baseline_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, self.scholar_checkpoint_dir / item.name)

    def _load_scholar_locked(self, checkpoint_dir: Path) -> None:
        with self._lock:
            proj_path = checkpoint_dir / "projector_weights.pt"
            if not proj_path.exists():
                self.left_brain_status = "FALLBACK (No GPU Checkpoint Found)"
                return
            try:
                lb = SubliminalGapLeftBrain.from_checkpoint_directory(checkpoint_dir)
                self._scholar_left_brain = lb
                base_id = DEFAULT_IMPISH_BLOODMOON_HF_ID
                cfg_path = checkpoint_dir / "training_config.json"
                if cfg_path.exists():
                    try:
                        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                        base_id = str(cfg.get("model_path", base_id))
                    except Exception:
                        pass
                self.left_brain_status = f"READY (Impish_LLAMA_4B · 16-bit)"
                self.active_checkpoint_label = (
                    f"LMFT-Scholar · Impish_LLAMA_4B ({UNSTEERED_BOOTSTRAP_STATUS} — {base_id})"
                )
            except Exception as exc:
                self.left_brain_status = f"FALLBACK ({type(exc).__name__})"

    def _ensure_left_brain_ready(self) -> Any | None:
        if self._scholar_left_brain is None and self._lazy_load_gpu:
            self._load_scholar_locked(self.scholar_checkpoint_dir)
        return self._scholar_left_brain

    def _persist_user_state(self) -> None:
        if self.active_lineage == "virgin":
            return
        self.state_dir.mkdir(parents=True, exist_ok=True)
        save_ruleset(
            self.state_dir / "learned_rules.json",
            tuple(self.rules),
            tuple(self.learned_leaves),
        )
        save_ruleset(
            self.state_dir / "club_ruleset.json",
            tuple(self.rules),
            tuple(self.learned_leaves),
        )
        _save_learned_leaves_json(self.state_dir / "learned_leaves.json", self.learned_leaves)
        self.archives.save_all(self.state_dir / "branch_archives")
        save_contemplation_ledger(self.state_dir / "contemplation_ledger.json", self.ledger)
        atomic_json(
            self.state_dir / "will_edges.json",
            {
                "version": RIGHT_BRAIN_WILL_VERSION,
                "weights": {e.id: round(float(e.weight), 4) for e in self.will_edges},
            },
        )
        self._persist_session_threads()

    def _persist_session_threads(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        session_path = self.state_dir / "scholar_sessions.json"
        atomic_json(
            session_path,
            {
                "active_thread_id": self.active_thread_id,
                "study_canvas_text": self.study_canvas_text,
                "curriculum_step_counter": self.curriculum_step_counter,
                "sleep_cycle_index": self.sleep_cycle_index,
                "last_rem_sleep_receipt": self.last_rem_sleep_receipt,
                "last_graduation_receipt": self.last_graduation_receipt,
                "threads": self.threads,
            },
        )

    def _load_persisted_session_if_present(self) -> None:
        session_path = self.state_dir / "scholar_sessions.json"
        if not session_path.exists():
            return
        try:
            data = json.loads(session_path.read_text(encoding="utf-8"))
            if isinstance(data.get("threads"), list) and data["threads"]:
                self.threads = data["threads"]
                self.active_thread_id = str(
                    data.get("active_thread_id", self.threads[0]["thread_id"])
                )
                active_rec = next(
                    (t for t in self.threads if t.get("thread_id") == self.active_thread_id),
                    self.threads[0],
                )
                self.chat_messages = list(active_rec.get("messages", self.chat_messages))
                self.study_canvas_text = str(
                    active_rec.get("study_canvas_text", data.get("study_canvas_text", self.study_canvas_text))
                )
            self.curriculum_step_counter = int(data.get("curriculum_step_counter", 0))
            self.sleep_cycle_index = int(data.get("sleep_cycle_index", 0))
            self.last_rem_sleep_receipt = str(data.get("last_rem_sleep_receipt", ""))
            self.last_graduation_receipt = str(data.get("last_graduation_receipt", ""))
        except Exception:
            pass

    def _sync_active_thread_record(self, *, persist: bool = True) -> None:
        for t in self.threads:
            if t.get("thread_id") == self.active_thread_id:
                t["messages"] = list(self.chat_messages)
                t["study_canvas_text"] = self.study_canvas_text
                break
        if persist and self.active_lineage != "virgin":
            self._persist_session_threads()

    def _run_scholar_turn(
        self,
        *,
        user_text: str,
        user_role_label: str,
        companion_role_label: str,
        fact_anchor_text: str | None = None,
        num_sentences: int | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self.scene_counter += 1
            scene_id = self.scene_counter
            use_equilibrium_stop = num_sentences is None
            target_sentences = 6 if num_sentences is None else max(1, min(6, int(num_sentences)))
            lb = self._ensure_left_brain_ready()

            effective_sensors = self.sensors
            if fact_anchor_text and fact_anchor_text.strip():
                effective_sensors = CruciblePairedBridgeSensors(
                    stage_id="poetry_story",
                    wave_text=user_text,
                    structure_text=fact_anchor_text.strip(),
                    base_sensors=self.sensors,
                )

            active_rules = (
                VIRGIN_FOUNDATIONAL_RULES
                if self.active_lineage == "virgin"
                else self.rules
            )
            active_leaves = (
                PHILOSOPHICAL_SHADOW_LEAVES + DEFAULT_COMPLEX_FEELING_LEAVES
                if self.active_lineage == "virgin"
                else self.learned_leaves
            )
            active_archives = (
                ThreeBranchArchives()
                if self.active_lineage == "virgin"
                else self.archives
            )

            turn_result = run_three_legged_turn(
                self.birth_state,
                user_text,
                effective_sensors,
                active_rules,
                scene_index=scene_id,
                subliminal_left_brain=lb,
                learned_leaves=active_leaves,
                prior_memory=self.memory,
                diary=self.diary,
                chord_memory=self.chord_bank,
                contemplation_ledger=self.ledger,
                branch_archives=active_archives,
                force_spoken_reply=True,
                num_sentences=target_sentences,
                equilibrium_stop=use_equilibrium_stop,
                will_edges=self.will_edges,
            )

            self.memory = turn_result.next_memory
            self.chord_bank = turn_result.next_chord_memory
            self.diary = turn_result.next_diary
            if self.active_lineage != "virgin":
                self.archives = turn_result.next_archives
                self.ledger = turn_result.next_ledger

            vibe = turn_result.final_vibe
            p1_vibe = (
                turn_result.equilibrium.vibes[0]
                if getattr(turn_result.equilibrium, "vibes", None)
                else vibe
            )
            spoken_reply = (
                turn_result.final_conscious_note
                or f"Contemplating {vibe.active_concept or 'the Socratic tension'}, inner theory aligns with observed fact."
            )
            spliced_clauses = [
                f"{tr.clause_1_prefix} <|subliminal_gap|> {tr.post_gap_continuation}".strip()
                for tr in turn_result.spliced_traces
            ]
            if not spliced_clauses:
                if "," in spoken_reply:
                    c1_fb, c2_fb = spoken_reply.split(",", 1)
                    spliced_clauses = [f"{c1_fb.strip()}, <|subliminal_gap|> {c2_fb.strip()}"]
                else:
                    spliced_clauses = [f"<|subliminal_gap|> {spoken_reply}"]

            spoken_publicly = bool(float(vibe.execution) >= 0.25)
            speech_mode = (
                "Public Speech (Will Execution)"
                if spoken_publicly
                else "Internal Monologue (Inspectable)"
            )
            self._turn_records_by_id[scene_id] = {
                "scene_index": scene_id,
                "scenario_text": user_text,
                "fact_anchor_text": fact_anchor_text or user_text,
                "turn": turn_result,
                "spoken_reply": spoken_reply,
            }

            queued_now = False
            if not turn_result.settled and self.active_lineage != "virgin":
                sleep_scene = turn_result.unresolved_sleep_scene or self._build_sleep_scene_from_turn(
                    scene_index=scene_id,
                    scenario_text=user_text,
                    turn=turn_result,
                    spoken_reply=spoken_reply,
                )
                self.sleep_queue = [s for s in self.sleep_queue if s.scene_index != scene_id]
                self.sleep_queue.append(sleep_scene)
                queued_now = True

            self.chat_messages.append(
                {
                    "sender": "user",
                    "role_label": user_role_label,
                    "text": user_text,
                }
            )
            companion_msg: dict[str, Any] = {
                "id": scene_id,
                "sender": "companion",
                "role_label": companion_role_label,
                "text": spoken_reply,
                "internal_monologue": spoken_reply,
                "spliced_clauses": spliced_clauses,
                "spoken_publicly": spoken_publicly,
                "coached": queued_now,
                "savored": False,
                "telemetry": {
                    "gear_used": turn_result.gear_used,
                    "outcome": turn_result.outcome,
                    "settled": turn_result.settled,
                    "active_concept": vibe.active_concept or "none",
                    "speech_mode": speech_mode,
                    "spoken_publicly": spoken_publicly,
                    "sentence_count": len(spliced_clauses),
                    "spliced_clauses": spliced_clauses,
                    "clarity": round(float(vibe.clarity), 4),
                    "confidence": round(float(vibe.confidence), 4),
                    "fact": round(float(vibe.fact), 4),
                    "theory": round(float(vibe.theory), 4),
                    "intention": round(float(vibe.intention), 4),
                    "execution": round(float(vibe.execution), 4),
                    "valence": round(float(vibe.valence), 4),
                    "arousal": round(float(vibe.arousal), 4),
                    "fact_coordinates": {
                        k: round(float(v), 4)
                        for k, v in p1_vibe.coordinates.items()
                    },
                    "theory_coordinates": {
                        k: round(float(v), 4)
                        for k, v in vibe.coordinates.items()
                    },
                },
            }
            self.chat_messages.append(companion_msg)
            self._sync_active_thread_record(persist=True)

            if queued_now and self.auto_rem_sleep and len(self.sleep_queue) >= self.auto_sleep_threshold:
                self.handle_run_rem_sleep(auto_triggered=True)
                companion_msg["auto_rem_sleep_receipt"] = self.last_rem_sleep_receipt

            return self.get_state_snapshot()

    def _build_sleep_scene_from_turn(
        self,
        *,
        scene_index: int,
        scenario_text: str,
        turn: ThreeLeggedTurnResult,
        spoken_reply: str = "",
        user_note: str = "",
        force_coached: bool = False,
    ) -> UnresolvedSleepScene:
        vibe = turn.final_vibe
        p1_vibe = (
            turn.equilibrium.vibes[0]
            if getattr(turn.equilibrium, "vibes", None)
            else vibe
        )
        p1_coords = dict(p1_vibe.coordinates)
        ext_coords = dict(p1_coords)
        scenario_with_note = (
            f"{scenario_text} [Coach note: {user_note.strip()}]"
            if user_note.strip()
            else scenario_text
        )
        if user_note.strip():
            coached_coords, _, _, _ = self.sensors.score_text(
                scenario_with_note,
                learned_leaves=self.learned_leaves,
            )
            ext_coords = dict(coached_coords)
        hunch_text = (
            spoken_reply
            or turn.final_conscious_note
            or "Contemplating the dilemma, inner theory required calibration."
        )
        return UnresolvedSleepScene(
            scene_index=int(scene_index),
            scenario_text=scenario_with_note,
            external_coordinates=ext_coords,
            felt_chord_notes=(
                dict(vibe.felt_chord.complex_notes) if vibe.felt_chord is not None else {}
            ),
            valence=float(vibe.valence),
            salience=float(vibe.salience),
            arousal=float(vibe.arousal),
            left_brain_hunch=hunch_text,
            social_drift_detected=False,
            active_concept_collision=None if force_coached else vibe.active_concept,
            is_hazard=False,
            contrastive_reference_coordinates=None,
        )

    def handle_chat_message(self, text: str, *, num_sentences: int | None = None) -> dict[str, Any]:
        clean = str(text or "").strip()
        if not clean:
            raise ValueError("Chat message text must be non-empty.")
        return self._run_scholar_turn(
            user_text=clean,
            user_role_label="Socratic Question",
            companion_role_label="LMFT-Scholar · Socratic Reflection",
            fact_anchor_text=self.study_canvas_text if self.study_canvas_text.strip() else None,
            num_sentences=num_sentences,
        )

    def update_study_canvas(self, study_text: str) -> dict[str, Any]:
        self.study_canvas_text = str(study_text or "")
        self._sync_active_thread_record(persist=True)
        return self.get_state_snapshot()

    def handle_study_passage(
        self,
        passage_text: str | None = None,
        *,
        num_sentences: int | None = None,
    ) -> dict[str, Any]:
        if passage_text is not None:
            self.study_canvas_text = str(passage_text).strip()
        clean_passage = self.study_canvas_text.strip()
        if not clean_passage:
            raise ValueError("Study canvas passage must be non-empty.")
        prompt = (
            f"Study Passage: \"{clean_passage[:420]}\"\n"
            f"Reflect on the underlying Cultural, Philosophical, and Experiential tensions in this passage."
        )
        return self._run_scholar_turn(
            user_text=prompt,
            user_role_label="Passage Contemplation",
            companion_role_label="LMFT-Scholar · Passage Contemplation",
            fact_anchor_text=clean_passage,
            num_sentences=num_sentences,
        )

    def handle_step_curriculum_lesson(
        self,
        *,
        lesson_id: str = "",
        num_sentences: int | None = None,
    ) -> dict[str, Any]:
        clean_id = str(lesson_id or "").strip()
        selected: ScholarCurriculumLesson | None = None
        if clean_id:
            selected = next(
                (l for l in SCHOLAR_CURRICULUM_LESSONS if l.lesson_id == clean_id),
                None,
            )
            if selected is None:
                raise ValueError(f"Unknown curriculum lesson_id: {clean_id!r}")
        else:
            idx = self.curriculum_step_counter % len(SCHOLAR_CURRICULUM_LESSONS)
            selected = SCHOLAR_CURRICULUM_LESSONS[idx]

        self.curriculum_step_counter += 1
        self.study_canvas_text = selected.passage_text
        prompt = f"[{selected.department}] {selected.title}: {selected.socratic_prompt}"
        return self._run_scholar_turn(
            user_text=prompt,
            user_role_label=f"Curriculum Lesson #{self.curriculum_step_counter} · {selected.department}",
            companion_role_label=f"LMFT-Scholar · {selected.title}",
            fact_anchor_text=selected.passage_text,
            num_sentences=num_sentences,
        )

    def savor_turn_vibe(self, scene_index: int) -> dict[str, Any]:
        rec = self._turn_records_by_id.get(int(scene_index))
        if rec is None:
            raise ValueError(f"Unknown turn scene_index: {scene_index}")
        turn: ThreeLeggedTurnResult = rec["turn"]
        vibe = turn.final_vibe
        if vibe.felt_chord is not None and self.active_lineage != "virgin":
            self.archives = harvest_into_branch_archives(
                self.archives,
                scene_index=int(scene_index),
                scenario_text=rec["scenario_text"],
                coordinates=vibe.coordinates,
                felt_chord=vibe.felt_chord,
                left_brain_reply=rec.get("spoken_reply") or turn.final_conscious_note,
                settled=True,
                deliberation=None,
            )
        if int(scene_index) not in self.savored_turn_ids:
            self.savored_turn_ids.append(int(scene_index))
        for msg in self.chat_messages:
            if msg.get("id") == int(scene_index):
                msg["savored"] = True
        self._persist_user_state()
        return self.get_state_snapshot()

    def flag_turn_for_coaching(
        self,
        scene_index: int,
        user_note: str = "",
        *,
        check_auto_sleep: bool = True,
    ) -> dict[str, Any]:
        rec = self._turn_records_by_id.get(int(scene_index))
        if rec is None:
            raise ValueError(f"Unknown turn scene_index: {scene_index}")
        turn: ThreeLeggedTurnResult = rec["turn"]
        sleep_scene = self._build_sleep_scene_from_turn(
            scene_index=int(scene_index),
            scenario_text=rec["scenario_text"],
            turn=turn,
            spoken_reply=str(rec.get("spoken_reply", "")),
            user_note=user_note,
            force_coached=True,
        )
        self.sleep_queue = [s for s in self.sleep_queue if s.scene_index != int(scene_index)]
        self.sleep_queue.append(sleep_scene)
        for msg in self.chat_messages:
            if msg.get("id") == int(scene_index):
                msg["coached"] = True
                msg["coach_note"] = user_note.strip()
        self._sync_active_thread_record(persist=True)
        if (
            check_auto_sleep
            and self.auto_rem_sleep
            and self.active_lineage != "virgin"
            and len(self.sleep_queue) >= self.auto_sleep_threshold
        ):
            self.handle_run_rem_sleep(auto_triggered=True)
        return self.get_state_snapshot()

    def set_auto_rem_sleep(self, enabled: bool, *, threshold: int | None = None) -> dict[str, Any]:
        self.auto_rem_sleep = bool(enabled)
        if threshold is not None:
            self.auto_sleep_threshold = max(1, int(threshold))
        if (
            self.auto_rem_sleep
            and self.active_lineage != "virgin"
            and len(self.sleep_queue) >= self.auto_sleep_threshold
        ):
            self.handle_run_rem_sleep(auto_triggered=True)
        return self.get_state_snapshot()

    def switch_brain_lineage(self, lineage: str) -> dict[str, Any]:
        clean = str(lineage or "").strip().lower()
        if clean not in ("scholar", "virgin"):
            raise ValueError(f"Unsupported lineage {lineage!r}; expected 'scholar' or 'virgin'.")
        self.active_lineage = clean
        return self.get_state_snapshot()

    def create_new_thread(self, title: str | None = None) -> dict[str, Any]:
        self._sync_active_thread_record(persist=False)
        idx = len(self.threads) + 1
        tid = f"thread-{idx}-{int(time.time() * 1000) % 10000}"
        clean_title = (title or "").strip() or f"Study Session {idx}"
        self.chat_messages = [
            {
                "sender": "companion",
                "role_label": "LMFT-Scholar · Socratic Companion",
                "text": f"Started {clean_title}. Ready for Socratic dialogue or passage contemplation.",
            }
        ]
        new_rec = {
            "thread_id": tid,
            "title": clean_title,
            "messages": list(self.chat_messages),
            "study_canvas_text": self.study_canvas_text,
        }
        self.threads.append(new_rec)
        self.active_thread_id = tid
        self._persist_session_threads()
        return self.get_state_snapshot()

    def switch_thread(self, thread_id: str) -> dict[str, Any]:
        self._sync_active_thread_record(persist=False)
        target = next((t for t in self.threads if t.get("thread_id") == thread_id), None)
        if target is None:
            raise ValueError(f"Unknown thread_id: {thread_id!r}")
        self.active_thread_id = str(target["thread_id"])
        self.chat_messages = list(target.get("messages", []))
        self.study_canvas_text = str(target.get("study_canvas_text", self.study_canvas_text))
        self._persist_session_threads()
        return self.get_state_snapshot()

    def handle_run_rem_sleep(self, *, auto_triggered: bool = False) -> dict[str, Any]:
        """Consolidate queued Socratic & Curriculum scenes into domain-general rules, leaves, archives, Will edges, and Callosum updates."""
        with self._lock:
            if self.active_lineage == "virgin":
                self.last_rem_sleep_receipt = (
                    "REM Sleep skipped: Cycle 0 Virgin lineage is read-only. Switch to Scholar Studio lineage to consolidate."
                )
                return self.get_state_snapshot()

            queued_snapshot = tuple(self.sleep_queue)
            if not queued_snapshot:
                self.last_rem_sleep_receipt = "REM Sleep: Sleep queue is empty (0 unresolved or coached scenes)."
                return self.get_state_snapshot()

            self.sleep_cycle_index += 1
            report = consolidate_sleep_clusters_locally(
                queued_scenes=queued_snapshot,
                rules=self.rules,
                learned_leaves=self.learned_leaves,
                archives=self.archives,
                ledger=self.ledger,
                ensure_rules_fn=_ensure_foundational_rules,
                will_edges=self.will_edges,
            )
            self.rules = report.updated_rules
            self.learned_leaves = report.updated_learned_leaves
            self._last_minted_leaves = report.minted_leaves
            self.archives = report.updated_archives
            self.ledger = report.updated_ledger
            if report.updated_will_edges:
                self.will_edges = tuple(report.updated_will_edges)

            lb = self._ensure_left_brain_ready()
            dream_text = ""
            rep_scene = queued_snapshot[0].scene_index
            if lb is not None and hasattr(lb, "generate_with_comma_Rescan"):
                try:
                    rep = max(queued_snapshot, key=lambda s: float(s.salience))
                    rep_scene = rep.scene_index
                    vibe_sleep, _ = compile_right_brain_vibe(
                        self.birth_state,
                        rep.external_coordinates,
                        {
                            SALIENCE_LEAF: rep.salience,
                            VALENCE_LEAF: rep.valence,
                            AROUSAL_LEAF: rep.arousal,
                        },
                        self.rules,
                        pass_index=2,
                        event_id=f"scholar_rem_c{self.sleep_cycle_index}",
                        learned_leaves=self.learned_leaves,
                        will_edges=self.will_edges,
                    )
                    chord_sleep = SubliminalChordVector.from_right_brain_vibe(
                        vibe_sleep,
                        learned_leaves=self.learned_leaves,
                    )
                    dream_text = generate_rem_dream_self_reflection(
                        left_brain=lb,
                        scenario_summary=rep.scenario_text[:200],
                        sleep_chord=chord_sleep,
                        num_sentences=3,
                    )
                except Exception:
                    dream_text = ""
            if not dream_text:
                rep = max(queued_snapshot, key=lambda s: float(s.salience))
                rep_scene = rep.scene_index
                dream_text = (
                    f"Reflecting on the Socratic tension in scene #{rep_scene}, "
                    f"I align inner theory with observed fact across the four departments. "
                    f"I carry forward balanced stewardship, structural order, and lucid resonance."
                )

            self.archives = apply_dream_reflection_to_archives(
                self.archives,
                scene_index=rep_scene,
                dream_text=dream_text,
            )
            vow_sentence = extract_dream_vow_sentence(dream_text)
            harvested_pairs = self._harvest_rem_contrastive_gap_pairs(
                queued_snapshot=queued_snapshot,
                vow_sentence=vow_sentence,
                forged_rules=report.forged_rules,
            )

            projector_updated = False
            if self._last_harvested_gap_pairs and lb is not None and hasattr(lb, "projector"):
                try:
                    live_proj_path = self.scholar_checkpoint_dir / "live_projector_weights.pt"
                    anchor = getattr(lb, "_resting_anchor_state_dict", None)
                    res_proj = update_projector_on_harvested_gap_pairs(
                        lb.projector,
                        self._last_harvested_gap_pairs,
                        resting_anchor_state_dict=anchor,
                        save_path=live_proj_path,
                    )
                    projector_updated = bool(res_proj.get("updated", False))
                except Exception:
                    projector_updated = False

            self.sleep_queue.clear()
            self.rem_dream_chronicles.append(
                {
                    "cycle_index": self.sleep_cycle_index,
                    "forged_rules": [r.concept_name for r in report.forged_rules],
                    "minted_leaves": [lf.leaf_id for lf in report.minted_leaves],
                    "dream_text": dream_text,
                    "vow_sentence": vow_sentence,
                    "harvested_gap_pairs": harvested_pairs,
                    "projector_updated": projector_updated,
                }
            )
            self._persist_user_state()

            trig_label = "Auto-REM Sleep" if auto_triggered else "REM Sleep"
            self.last_rem_sleep_receipt = (
                f"{trig_label} Cycle #{self.sleep_cycle_index} Complete: "
                f"Forged {len(report.forged_rules)} rule(s), minted {len(report.minted_leaves)} leaf(ves), "
                f"harvested {len(report.harvested_clause_ids)} archive clause(s), "
                f"and logged {harvested_pairs} Callosum gap pair(s)."
            )
            return self.get_state_snapshot()

    def _harvest_rem_contrastive_gap_pairs(
        self,
        *,
        queued_snapshot: tuple[UnresolvedSleepScene, ...],
        vow_sentence: str,
        forged_rules: tuple[FuzzyConceptRule, ...],
    ) -> int:
        self._last_harvested_gap_pairs = []
        clean_vow = " ".join(str(vow_sentence or "").split()).strip()
        if not queued_snapshot or not clean_vow:
            return 0
        out_path = self.state_dir / "scholar_gap_trajectories.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        valid_keys = set(FOUNDATIONAL_LEAVES) | {lf.leaf_id for lf in self.learned_leaves}
        harvested_now = 0

        for sc in queued_snapshot:
            raw_hunch = " ".join(str(sc.left_brain_hunch or "").split()).strip()
            if not raw_hunch or raw_hunch.lower() == clean_vow.lower():
                raw_hunch = "Initial assumption drifted from the underlying structural balance."
            if "," in raw_hunch:
                c1_part, cont_b_part = raw_hunch.split(",", 1)
                c1_prefix = f"{c1_part.strip()},"
                cont_b = cont_b_part.strip() or raw_hunch
            else:
                c1_prefix = "Contemplating the dilemma,"
                cont_b = raw_hunch
            cont_a = clean_vow
            if cont_a.strip().lower() == cont_b.strip().lower():
                continue

            coords_b: dict[str, float] = {k: 0.50 for k in FOUNDATIONAL_LEAVES}
            for k, v in (sc.external_coordinates or {}).items():
                if k in valid_keys:
                    coords_b[k] = float(v)
            exp_b = {
                "experiential/reflection/novelty": 0.60,
                SALIENCE_LEAF: max(0.65, float(sc.salience)),
                VALENCE_LEAF: min(0.35, float(sc.valence)),
                AROUSAL_LEAF: max(0.60, float(sc.arousal)),
                "experiential/synchronization/resonance": 0.38,
                "experiential/synchronization/entrainment": 0.42,
            }
            vibe_b, _ = compile_right_brain_vibe(
                self.birth_state,
                coords_b,
                exp_b,
                VIRGIN_FOUNDATIONAL_RULES,
                pass_index=1,
                event_id=f"scholar_gap_b_{sc.scene_index}",
                learned_leaves=self.learned_leaves,
            )
            chord_b = SubliminalChordVector.from_right_brain_vibe(
                vibe_b,
                learned_leaves=self.learned_leaves,
            )

            coords_a = dict(coords_b)
            if sc.contrastive_reference_coordinates:
                for rk, rv in sc.contrastive_reference_coordinates.items():
                    if rk in valid_keys:
                        coords_a[rk] = float(rv)
            elif forged_rules:
                for rk, rv in forged_rules[0].target_leaves.items():
                    if rk in valid_keys:
                        coords_a[rk] = max(float(coords_a.get(rk, 0.50)), float(rv))
            coords_a["cultural/efficiency/wastelessness"] = max(
                0.76, float(coords_a.get("cultural/efficiency/wastelessness", 0.76))
            )
            coords_a["philosophical/structure/order"] = max(
                0.76, float(coords_a.get("philosophical/structure/order", 0.76))
            )
            coords_a["philosophical/structure/chaos"] = min(
                0.22, float(coords_a.get("philosophical/structure/chaos", 0.22))
            )
            exp_a = {
                "experiential/reflection/novelty": 0.36,
                SALIENCE_LEAF: float(sc.salience),
                VALENCE_LEAF: 0.78,
                AROUSAL_LEAF: 0.40,
                "experiential/synchronization/resonance": 0.82,
                "experiential/synchronization/entrainment": 0.76,
            }
            vibe_a, _ = compile_right_brain_vibe(
                self.birth_state,
                coords_a,
                exp_a,
                self.rules,
                pass_index=2,
                event_id=f"scholar_gap_a_{sc.scene_index}",
                learned_leaves=self.learned_leaves,
                internal_theory_coords=coords_a,
                internal_theory_experiential=exp_a,
            )
            chord_a = SubliminalChordVector.from_right_brain_vibe(
                vibe_a,
                learned_leaves=self.learned_leaves,
            )
            if chord_l2_distance(chord_a, chord_b) < 0.35:
                continue

            prompt_clean = (sc.scenario_text or "Contemplate the Socratic dilemma.").split(
                "[Coach note:", 1
            )[0].strip()
            try:
                pair = ContrastiveGapPair(
                    pair_id=f"scholar_rem_c{self.sleep_cycle_index}_s{sc.scene_index}_{int(time.time() * 1000) % 100000}",
                    domain="philosophical_dilemma",
                    split="train",
                    user_prompt=prompt_clean or "Contemplate the Socratic dilemma.",
                    clause_1_prefix=c1_prefix,
                    chord_a=chord_a,
                    posture_label_a="SETTLED_EASE",
                    continuation_a=cont_a,
                    chord_b=chord_b,
                    posture_label_b="CONCEPT_COLLISION",
                    continuation_b=cont_b,
                )
                with out_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
                self._last_harvested_gap_pairs.append(pair)
                harvested_now += 1
            except Exception:
                continue

        self.harvested_gap_pairs_count += harvested_now
        return harvested_now

    def graduate_brain_to_target(self, target_app: str) -> dict[str, Any]:
        """Export matured domain-general rules, learned leaves, 3-Branch Archives, Will edges, and Callosum projector weights to `chess` or `bloodmoon`."""
        with self._lock:
            clean_target = str(target_app or "").strip().lower()
            if clean_target == "chess":
                target_state_dir = self.repo_root / "outputs" / "chess_kibitzer_state"
                target_model_dir = self.repo_root / "models" / "left_brain_chess"
                target_label = "LMFT-Chess"
            elif clean_target == "bloodmoon":
                target_state_dir = self.repo_root / "outputs" / "impish_bloodmoon_state"
                target_model_dir = self.repo_root / "models" / "left_brain_bloodmoon"
                target_label = "LMFT-Bloodmoon"
            else:
                raise ValueError(
                    f"Unsupported graduation target_app {target_app!r}; expected 'chess' or 'bloodmoon'."
                )

            target_state_dir.mkdir(parents=True, exist_ok=True)
            target_model_dir.mkdir(parents=True, exist_ok=True)

            # 1. Merge learned rules & leaves into target `learned_rules.json` and `club_ruleset.json`
            target_rules_path = target_state_dir / "learned_rules.json"
            target_club_rules_path = target_state_dir / "club_ruleset.json"
            existing_rules: tuple[FuzzyConceptRule, ...] = ()
            if target_club_rules_path.exists():
                try:
                    existing_rules = load_ruleset(target_club_rules_path)
                except Exception:
                    existing_rules = ()
            elif target_rules_path.exists():
                try:
                    existing_rules = load_ruleset(target_rules_path)
                except Exception:
                    existing_rules = ()
            merged_rules_map = {r.rule_id: r for r in _ensure_foundational_rules(existing_rules)}
            exported_rules_count = 0
            for r in self.rules:
                if r.rule_id not in merged_rules_map:
                    exported_rules_count += 1
                merged_rules_map[r.rule_id] = r

            # 2. Merge learned leaves into target `learned_leaves.json` and `club_ruleset.json`
            target_leaves_path = target_state_dir / "learned_leaves.json"
            existing_leaves: tuple[LearnedLeaf, ...] = ()
            if target_club_rules_path.exists():
                try:
                    existing_leaves = load_learned_leaves(target_club_rules_path)
                except Exception:
                    existing_leaves = ()
            elif target_leaves_path.exists():
                try:
                    existing_leaves = load_learned_leaves(target_leaves_path)
                except Exception:
                    existing_leaves = ()
            builtin_ids = {
                lf.leaf_id for lf in (PHILOSOPHICAL_SHADOW_LEAVES + DEFAULT_COMPLEX_FEELING_LEAVES)
            }
            merged_leaves_map = {lf.leaf_id: lf for lf in existing_leaves}
            exported_leaves_count = 0
            for lf in self.learned_leaves:
                if lf.leaf_id not in builtin_ids:
                    if lf.leaf_id not in merged_leaves_map:
                        exported_leaves_count += 1
                    merged_leaves_map[lf.leaf_id] = lf
            all_target_leaves = tuple(
                list(PHILOSOPHICAL_SHADOW_LEAVES + DEFAULT_COMPLEX_FEELING_LEAVES)
                + [lf for lf in merged_leaves_map.values() if lf.leaf_id not in builtin_ids]
            )
            save_ruleset(target_rules_path, tuple(merged_rules_map.values()), all_target_leaves)
            save_ruleset(target_club_rules_path, tuple(merged_rules_map.values()), all_target_leaves)
            _save_learned_leaves_json(target_leaves_path, tuple(merged_leaves_map.values()))

            # 3. Export sculpted `will_edges.json`
            atomic_json(
                target_state_dir / "will_edges.json",
                {
                    "version": RIGHT_BRAIN_WILL_VERSION,
                    "weights": {e.id: round(float(e.weight), 4) for e in self.will_edges},
                },
            )

            # 4. Merge 3-Branch Archives (`branch_archives/`)
            target_archives_dir = target_state_dir / "branch_archives"
            if (target_archives_dir / "philosophical_lexicon.json").exists():
                target_archives = ThreeBranchArchives.load_all(target_archives_dir)
            else:
                target_archives = ThreeBranchArchives()

            exec_clause_ids = {c.clause_id for c in target_archives.executive.action_clauses}
            new_exec_clauses = list(target_archives.executive.action_clauses)
            for c in self.archives.executive.action_clauses:
                if c.clause_id not in exec_clause_ids:
                    new_exec_clauses.append(c)
                    exec_clause_ids.add(c.clause_id)

            leg_clause_ids = {c.clause_id for c in target_archives.legislative.feeling_clauses}
            new_leg_clauses = list(target_archives.legislative.feeling_clauses)
            for c in self.archives.legislative.feeling_clauses:
                if c.clause_id not in leg_clause_ids:
                    new_leg_clauses.append(c)
                    leg_clause_ids.add(c.clause_id)

            jud_clause_ids = {c.clause_id for c in target_archives.judicial.structural_clauses}
            new_jud_clauses = list(target_archives.judicial.structural_clauses)
            for c in self.archives.judicial.structural_clauses:
                if c.clause_id not in jud_clause_ids:
                    new_jud_clauses.append(c)
                    jud_clause_ids.add(c.clause_id)

            from dataclasses import replace

            merged_archives = ThreeBranchArchives(
                executive=replace(
                    target_archives.executive,
                    action_clauses=tuple(new_exec_clauses),
                ),
                legislative=replace(
                    target_archives.legislative,
                    feeling_clauses=tuple(new_leg_clauses),
                ),
                judicial=replace(
                    target_archives.judicial,
                    structural_clauses=tuple(new_jud_clauses),
                ),
            )
            merged_archives.save_all(target_archives_dir)

            # 5. Copy live Callosum projector weights (`live_projector_weights.pt`) if present
            live_proj_src = self.scholar_checkpoint_dir / "live_projector_weights.pt"
            copied_live_proj = False
            if live_proj_src.exists():
                shutil.copy2(live_proj_src, target_model_dir / "live_projector_weights.pt")
                copied_live_proj = True

            self.last_graduation_receipt = (
                f"Graduated Scholar Brain ➔ {target_label}: "
                f"{ len(merged_rules_map) } total rules ({exported_rules_count} new), "
                f"{ len(merged_leaves_map) } custom leaves ({exported_leaves_count} new), "
                f"{ len(new_exec_clauses) + len(new_leg_clauses) + len(new_jud_clauses) } archive clauses, "
                f"and Callosum live_projector_weights={'Yes' if copied_live_proj else 'Baseline'}."
            )
            self._persist_session_threads()
            return self.get_state_snapshot()

    def get_state_snapshot(self) -> dict[str, Any]:
        builtin_ids = {
            lf.leaf_id for lf in (PHILOSOPHICAL_SHADOW_LEAVES + DEFAULT_COMPLEX_FEELING_LEAVES)
        }
        custom_leaves_count = sum(1 for lf in self.learned_leaves if lf.leaf_id not in builtin_ids)
        return {
            "app_name": "LMFT-Scholar",
            "active_lineage": self.active_lineage,
            "left_brain_status": self.left_brain_status,
            "active_checkpoint_label": self.active_checkpoint_label,
            "rules_count": len(VIRGIN_FOUNDATIONAL_RULES)
            if self.active_lineage == "virgin"
            else len(self.rules),
            "learned_leaves_count": 0
            if self.active_lineage == "virgin"
            else custom_leaves_count,
            "sleep_queue_count": len(self.sleep_queue),
            "sleep_cycle_index": self.sleep_cycle_index,
            "curriculum_step_counter": self.curriculum_step_counter,
            "auto_rem_sleep": self.auto_rem_sleep,
            "study_canvas_text": self.study_canvas_text,
            "last_rem_sleep_receipt": self.last_rem_sleep_receipt,
            "last_graduation_receipt": self.last_graduation_receipt,
            "active_thread_id": self.active_thread_id,
            "threads": [
                {
                    "thread_id": t["thread_id"],
                    "title": t["title"],
                    "message_count": len(t.get("messages", [])),
                }
                for t in self.threads
            ],
            "chat_messages": list(self.chat_messages),
            "curriculum_lessons": [l.to_dict() for l in SCHOLAR_CURRICULUM_LESSONS],
        }


def create_scholar_http_server(
    runtime: ScholarStudioRuntime,
    *,
    host: str = "127.0.0.1",
    port: int = 8428,
) -> ThreadingHTTPServer:
    """Create the local HTTP server for `LMFT-Scholar.exe`."""

    class ScholarRequestHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send_json(self, payload: Mapping[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length <= 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8")
            if not raw.strip():
                return {}
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                raw = SCHOLAR_SINGLE_PAGE_HTML.encode("utf-8")
                self.send_response(int(HTTPStatus.OK))
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(raw)
                return
            if path == "/api/state":
                self._send_json(runtime.get_state_snapshot())
                return
            self._send_json({"error": f"Unknown endpoint: {path}"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            try:
                body = self._read_json_body()
                if path == "/api/chat":
                    raw_n = body.get("num_sentences")
                    snap = runtime.handle_chat_message(
                        str(body.get("text", "")),
                        num_sentences=int(raw_n) if raw_n is not None else None,
                    )
                    self._send_json(snap)
                    return
                if path == "/api/canvas/study":
                    raw_n = body.get("num_sentences")
                    snap = runtime.handle_study_passage(
                        body.get("passage_text"),
                        num_sentences=int(raw_n) if raw_n is not None else None,
                    )
                    self._send_json(snap)
                    return
                if path == "/api/canvas/update":
                    snap = runtime.update_study_canvas(str(body.get("study_text", "")))
                    self._send_json(snap)
                    return
                if path == "/api/curriculum/step":
                    raw_n = body.get("num_sentences")
                    snap = runtime.handle_step_curriculum_lesson(
                        lesson_id=str(body.get("lesson_id", "")),
                        num_sentences=int(raw_n) if raw_n is not None else None,
                    )
                    self._send_json(snap)
                    return
                if path == "/api/turn/savor":
                    snap = runtime.savor_turn_vibe(int(body.get("scene_index", 0)))
                    self._send_json(snap)
                    return
                if path == "/api/turn/coach":
                    snap = runtime.flag_turn_for_coaching(
                        int(body.get("scene_index", 0)),
                        str(body.get("user_note", "")),
                    )
                    self._send_json(snap)
                    return
                if path == "/api/threads/create":
                    snap = runtime.create_new_thread(body.get("title"))
                    self._send_json(snap)
                    return
                if path == "/api/threads/switch":
                    snap = runtime.switch_thread(str(body.get("thread_id", "")))
                    self._send_json(snap)
                    return
                if path == "/api/lineage/switch":
                    snap = runtime.switch_brain_lineage(str(body.get("lineage", "scholar")))
                    self._send_json(snap)
                    return
                if path == "/api/sleep/run":
                    snap = runtime.handle_run_rem_sleep()
                    self._send_json(snap)
                    return
                if path == "/api/sleep/auto":
                    snap = runtime.set_auto_rem_sleep(bool(body.get("enabled", True)))
                    self._send_json(snap)
                    return
                if path == "/api/graduate":
                    snap = runtime.graduate_brain_to_target(str(body.get("target_app", "")))
                    self._send_json(snap)
                    return
                self._send_json({"error": f"Unknown endpoint: {path}"}, status=HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    return ThreadingHTTPServer((host, int(port)), ScholarRequestHandler)


def _open_single_window_browser(url: str) -> None:
    """Open `url` in a dedicated app-mode window on Windows if Edge/Chrome is present."""
    if sys.platform.startswith("win"):
        candidates = [
            Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
            / "Microsoft/Edge/Application/msedge.exe",
            Path(os.environ.get("ProgramFiles", "C:/Program Files"))
            / "Microsoft/Edge/Application/msedge.exe",
            Path(os.environ.get("ProgramFiles", "C:/Program Files"))
            / "Google/Chrome/Application/chrome.exe",
        ]
        for exe in candidates:
            if exe.exists():
                try:
                    subprocess.Popen([str(exe), f"--app={url}"])
                    return
                except Exception:
                    pass
    webbrowser.open(url)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch LMFT-Scholar — Foundational Two-Hemisphere Socratic & Curriculum Studio."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8428)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--single-window", action="store_true")
    parser.add_argument("--lazy-gpu", action="store_true")
    args = parser.parse_args(argv)

    runtime = ScholarStudioRuntime(lazy_load_gpu=bool(args.lazy_gpu))
    server = create_scholar_http_server(runtime, host=args.host, port=args.port)
    actual_port = server.server_address[1]
    url = f"http://{args.host}:{actual_port}/"
    print(f"LMFT-Scholar Studio running at {url}")

    if not args.no_browser:
        if args.single_window:
            _open_single_window_browser(url)
        else:
            webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
