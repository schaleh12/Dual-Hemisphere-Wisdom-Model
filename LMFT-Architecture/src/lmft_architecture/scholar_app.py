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
    ALL_GROWABLE_PARENT_BRANCHES,
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
    assert_domain_general_text,
    consolidate_sleep_clusters_locally,
    entrain_existing_cluster_rule,
)
from .subliminal_gap_splice import (
    _DEFAULT_LEAF_ANCHOR_WEIGHTS,
    SubliminalChordVector,
    score_leaf_anchor_weights_at_birth,
    sculpt_learned_leaf_anchor_weights,
)
from .three_legged_runtime import (
    SubliminalGapLeftBrain,
    ThreeLeggedTurnResult,
    compute_stanza_residual_chord,
    generate_rem_dream_self_reflection,
    run_three_legged_turn,
)
from .train_left_brain_lora import LoRALinear, update_projector_on_harvested_gap_pairs
from .virgin_baseline import VIRGIN_FOUNDATIONAL_RULES, load_virgin_baseline_state
from .workspace_paths import resolve_training_root

WillEdgeSpec = Association
DEFAULT_SCHOLAR_HF_ID = "SicariusSicariiStuff/Impish_LLAMA_4B"


@dataclass(frozen=True)
class _PassageAnchoredSensors:
    """Ground Pass-1 external Fact scoring in `fact_anchor_text + wave_text` while scoring Left-Brain Theory drafts directly."""

    wave_text: str
    fact_anchor_text: str
    base_sensors: ThreeBridgeSensors

    def score_text(
        self,
        text: str,
        *,
        learned_leaves: tuple[LearnedLeaf, ...] = (),
        branch_gating: bool = True,
        branch_gate_threshold: float = 0.50,
    ) -> tuple[dict[str, float], dict[str, float], tuple[str, ...], tuple[str, ...]]:
        clean = text.strip()
        anchor = self.fact_anchor_text.strip()
        if clean == self.wave_text.strip() and anchor and anchor not in clean:
            combined = f"{anchor}\n\n{clean}"
            return self.base_sensors.score_text(
                combined,
                learned_leaves=learned_leaves,
                branch_gating=branch_gating,
                branch_gate_threshold=branch_gate_threshold,
            )
        return self.base_sensors.score_text(
            text,
            learned_leaves=learned_leaves,
            branch_gating=branch_gating,
            branch_gate_threshold=branch_gate_threshold,
        )


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
        title="Moby-Dick — Prudent Stewardship vs. Monomaniacal Force (Power & Wastelessness)",
        passage_text=(
            "In Herman Melville's Moby-Dick, First Mate Starbuck pleads that the voyage exists to gather oil "
            "and bring the crew safely home to Nantucket, whereas Captain Ahab welds the crew's will into a "
            "single harpoon of vengeance that ultimately consumes the Pequod and all aboard."
        ),
        socratic_prompt=(
            "Write a contemplative essay examining how genuine power must remain tethered to wastelessness and "
            "stewardship so that focused ambition does not devour the vessel that carries it."
        ),
    ),
    ScholarCurriculumLesson(
        lesson_id="cultural_justice_reciprocity",
        department="Cultural · Justice",
        title="Antigone & Les Misérables — Sacred Covenant vs. Rigid Decree (Honor & Fairness)",
        passage_text=(
            "In Sophocles' Antigone and Victor Hugo's Les Misérables, an unyielding legal decree demands "
            "punishment without mercy, while Antigone and Bishop Myriel uphold a deeper covenant of honor "
            "and restorative fairness that recognizes shared human dignity under scarcity and fault."
        ),
        socratic_prompt=(
            "Reflect on how honor and fairness harmonize when strict procedural rules collide with restorative justice."
        ),
    ),
    ScholarCurriculumLesson(
        lesson_id="cultural_freedom_autonomy",
        department="Cultural · Freedom",
        title="Siddhartha & Walden — Inner Sufficiency Amid Abundance (Self-Sufficiency & Needlessness)",
        passage_text=(
            "In Hermann Hesse's Siddhartha and Henry David Thoreau's Walden, the seeker walks through the "
            "bustling marketplace and the quiet woods alike, discovering that true autonomy arises not from "
            "accumulating possessions, but from self-sufficiency and needlessness before compulsive craving."
        ),
        socratic_prompt=(
            "Examine how self-sufficiency and needlessness sustain inner freedom when surrounded by status or temptation."
        ),
    ),
    ScholarCurriculumLesson(
        lesson_id="philosophical_being_agency",
        department="Philosophical · Being",
        title="Frankenstein — Moral Agency Within Given Existence (Agency & Existence)",
        passage_text=(
            "In Mary Shelley's Frankenstein, Victor claims the creative agency to bring a sentient being into "
            "existence, yet flees the moment his creation opens its eyes, abandoning the creature to navigate "
            "an unchosen existence in solitude."
        ),
        socratic_prompt=(
            "Reflect on how conscious agency carries ongoing stewardship for what it brings into existence."
        ),
    ),
    ScholarCurriculumLesson(
        lesson_id="philosophical_structure_harmony",
        department="Philosophical · Structure",
        title="The Tempest & Lord of the Flies — Living Order Amid Upheaval (Order & Chaos)",
        passage_text=(
            "When a sudden tempest shatters familiar hierarchies on an isolated island, brittle command "
            "fractures into factional chaos, whereas resilient order absorbs the storm and weaves reconciliation "
            "out of turbulence."
        ),
        socratic_prompt=(
            "Contemplate how living structural order adapts to chaos without hardening into tyranny or collapsing into disorder."
        ),
    ),
    ScholarCurriculumLesson(
        lesson_id="philosophical_persistence_rhythm",
        department="Philosophical · Persistence",
        title="The Odyssey — Enduring Identity Across Discrete Trials (Continuity & Discreteness)",
        passage_text=(
            "Across twenty years of discrete islands, shipwrecks, and shifting disguises in Homer's Odyssey, "
            "Odysseus and Penelope preserve a single unbroken thread of continuity and memory that links "
            "departure to return."
        ),
        socratic_prompt=(
            "Explore how enduring continuity of purpose connects discrete, changing episodes across time."
        ),
    ),
    ScholarCurriculumLesson(
        lesson_id="experiential_reflection_wonder",
        department="Experiential · Reflection",
        title="Plato's Cave & Alice in Wonderland — Discerning What Matters (Novelty & Salience)",
        passage_text=(
            "When stepping out of Plato's cave into open sunlight or descending into an unfamiliar wonderland, "
            "every shadow and spectacle dazzles with immediate novelty, requiring patient reflection to discern "
            "true structural salience from passing distraction."
        ),
        socratic_prompt=(
            "Examine how contemplative reflection distinguishes enduring salience from loud sensory novelty."
        ),
    ),
    ScholarCurriculumLesson(
        lesson_id="experiential_phase_catharsis",
        department="Experiential · Phase",
        title="The Iliad (Book 24) — Transforming Wrath Into Shared Catharsis (Valence & Arousal)",
        passage_text=(
            "In the final book of Homer's Iliad, aged King Priam enters Achilles' tent at night to ransom "
            "Hector's body; as they weep together, consuming high-arousal rage softens into quiet, shared "
            "sorrow and restorative emotional valence."
        ),
        socratic_prompt=(
            "Reflect on how overwhelming emotional arousal is metabolized into grounded empathy and balanced valence."
        ),
    ),
    ScholarCurriculumLesson(
        lesson_id="experiential_resonance_lucidity",
        department="Experiential · Synchronization",
        title="Julius Caesar & 1984 — Empathetic Resonance Without Crowd Entrainment (Resonance & Entrainment)",
        passage_text=(
            "In Shakespeare's Forum scene in Julius Caesar and Orwell's 1984, rhythmic rhetoric sweeps an "
            "entire square into involuntary emotional entrainment, testing whether a listener can feel genuine "
            "human resonance while keeping an unhurried, independent mind."
        ),
        socratic_prompt=(
            "Reflect on how one maintains compassionate resonance with others without surrendering to involuntary entrainment."
        ),
    ),
    ScholarCurriculumLesson(
        lesson_id="psychological_will_resolution",
        department="Psychological · Will",
        title="Hamlet — Bridging Deliberate Intention and Timed Execution (Intention & Execution)",
        passage_text=(
            "Throughout Shakespeare's Hamlet, the prince oscillates between endless solitary rumination that "
            "paralyzes execution and sudden unexamined impulse, until intention and execution finally align "
            "in calm readiness."
        ),
        socratic_prompt=(
            "Examine how clear moral intention matures into steady, timely execution without hesitation or recklessness."
        ),
    ),
    ScholarCurriculumLesson(
        lesson_id="psychological_truth_calibration",
        department="Psychological · Truth",
        title="Crime and Punishment — Calibrating Theory Against Lived Fact (Theory & Fact)",
        passage_text=(
            "In Dostoevsky's Crime and Punishment, Raskolnikov constructs an airtight abstract theory that "
            "extraordinary individuals stand above moral law, only to discover that lived human fact and "
            "conscience refuse to bend to his rationalization."
        ),
        socratic_prompt=(
            "Examine how humility before observed fact dissolves self-rationalizing theory and restores inner truth."
        ),
    ),
    ScholarCurriculumLesson(
        lesson_id="psychological_contemplation_stillness",
        department="Psychological · Contemplation",
        title="Marcus Aurelius' Meditations — Unclouded Stillness Under Pressure (Clarity & Confidence)",
        passage_text=(
            "Writing in his campaign tent along the frontier, Marcus Aurelius reminds himself that external "
            "tumult cannot muddy the mind unless judgment consents, pairing unclouded perceptual clarity with "
            "quiet, unboastful confidence."
        ),
        socratic_prompt=(
            "Contemplate how genuine clarity and grounded confidence reinforce each other during uncertainty."
        ),
    ),
)

_SCHOLAR_BANNED_STORY_NOUNS: tuple[str, ...] = (
    "ahab",
    "starbuck",
    "pequod",
    "melville",
    "moby",
    "antigone",
    "creon",
    "valjean",
    "javert",
    "myriel",
    "siddhartha",
    "thoreau",
    "walden",
    "frankenstein",
    "victor",
    "shelley",
    "prospero",
    "golding",
    "odysseus",
    "penelope",
    "homer",
    "plato",
    "alice",
    "achilles",
    "priam",
    "hector",
    "caesar",
    "brutus",
    "antony",
    "orwell",
    "hamlet",
    "shakespeare",
    "raskolnikov",
    "dostoevsky",
    "aurelius",
)

_FOUNDING_TREE_STRUCTURE: tuple[tuple[str, str, tuple[tuple[str, str, tuple[str, str]], ...]], ...] = (
    (
        "cultural",
        "Cultural · The Do-er (Action & Stewardship)",
        (
            ("cultural/efficiency", "Efficiency", ("cultural/efficiency/power", "cultural/efficiency/wastelessness")),
            ("cultural/justice", "Justice", ("cultural/justice/honor", "cultural/justice/fairness")),
            ("cultural/freedom", "Freedom", ("cultural/freedom/self-sufficiency", "cultural/freedom/needlessness")),
        ),
    ),
    (
        "philosophical",
        "Philosophical · The Thinker (Structure & Being)",
        (
            ("philosophical/being", "Being", ("philosophical/being/agency", "philosophical/being/existence")),
            ("philosophical/structure", "Structure", ("philosophical/structure/order", "philosophical/structure/chaos")),
            ("philosophical/persistence", "Persistence", ("philosophical/persistence/continuity", "philosophical/persistence/discreteness")),
        ),
    ),
    (
        "experiential",
        "Experiential · The Feeler (Phase & Resonance)",
        (
            ("experiential/reflection", "Reflection", ("experiential/reflection/novelty", "experiential/reflection/salience")),
            ("experiential/phase", "Phase", ("experiential/phase/valence", "experiential/phase/arousal")),
            ("experiential/synchronization", "Synchronization", ("experiential/synchronization/resonance", "experiential/synchronization/entrainment")),
        ),
    ),
    (
        "psychological",
        "Psychological · The Regulator (Will, Truth & Contemplation)",
        (
            ("psychological/will", "Will", ("psychological/will/intention", "psychological/will/execution")),
            ("psychological/truth", "Truth", ("psychological/truth/theory", "psychological/truth/fact")),
            ("psychological/contemplation", "Contemplation", ("psychological/contemplation/clarity", "psychological/contemplation/confidence")),
        ),
    ),
)

_DEPT_ANCHOR_LEAF_NAMES: dict[str, tuple[str, ...]] = {
    "cultural": FOUNDATIONAL_LEAVES[:6] + ("psychological/will/intention", "psychological/will/execution"),
    "philosophical": FOUNDATIONAL_LEAVES[6:12] + ("psychological/truth/theory", "psychological/truth/fact"),
    "experiential": (
        "experiential/reflection/novelty",
        SALIENCE_LEAF,
        VALENCE_LEAF,
        AROUSAL_LEAF,
        "experiential/synchronization/resonance",
        "experiential/synchronization/entrainment",
        "psychological/contemplation/clarity",
        "psychological/contemplation/confidence",
    ),
    "psychological": (
        "psychological/will/intention",
        "psychological/will/execution",
        "psychological/truth/theory",
        "psychological/truth/fact",
        "psychological/contemplation/clarity",
        "psychological/contemplation/confidence",
        VALENCE_LEAF,
        AROUSAL_LEAF,
    ),
}

_SHARED_NEURAL_BRIDGES: NeuralThreeBridgeSuite | None = None


def _get_shared_neural_bridges(repo_root: Path | None = None) -> NeuralThreeBridgeSuite | None:
    """Load the 3 MiniLM neural bridges once per process from `repo_root` or workspace training root."""
    global _SHARED_NEURAL_BRIDGES
    if _SHARED_NEURAL_BRIDGES is not None:
        return _SHARED_NEURAL_BRIDGES
    candidates: list[Path] = []
    if repo_root is not None:
        candidates.append(Path(repo_root) / "models" / "right_brain_bridges")
    try:
        candidates.append(resolve_training_root() / "models" / "right_brain_bridges")
    except Exception:
        pass
    for bridge_dir in candidates:
        if bridge_dir.exists():
            try:
                _SHARED_NEURAL_BRIDGES = NeuralThreeBridgeSuite.load_all(bridge_dir)
                return _SHARED_NEURAL_BRIDGES
            except Exception:
                continue
    return None


def _evaluate_learned_leaf_from_anchors(
    leaf: LearnedLeaf,
    coords: Mapping[str, float],
) -> float:
    """Compute a `LearnedLeaf`'s live coordinate activation from its 8 departmental `anchor_weights` over `coords`."""
    if leaf.leaf_id in coords:
        return round(min(1.0, max(0.0, float(coords[leaf.leaf_id]))), 6)
    dept_prefix = leaf.leaf_id.split("/", 1)[0]
    anchor_keys = _DEPT_ANCHOR_LEAF_NAMES.get(dept_prefix, FOUNDATIONAL_LEAVES[:8])
    weights = leaf.anchor_weights if (leaf.anchor_weights and len(leaf.anchor_weights) == 8) else (0.5,) * 8
    w_sum = sum(float(w) for w in weights)
    if w_sum <= 1e-6:
        base_val = 0.50
    else:
        base_val = sum(
            float(w) * float(coords.get(k, 0.50))
            for k, w in zip(anchor_keys, weights)
        ) / w_sum
    # Parent-branch gating for Tier 3 and Tier 4 leaves
    if leaf.tier >= 3 and leaf.parent_branch in coords:
        parent_val = float(coords[leaf.parent_branch])
        base_val = 0.65 * base_val + 0.35 * parent_val
    return round(min(1.0, max(0.0, base_val)), 6)


def _default_fallback_sensors() -> ThreeBridgeSensors:
    """Zero-keyword continuous structural fallback sensors used only if MiniLM bridge files are unavailable."""

    def _continuous_profile(text: str, salt: int) -> tuple[float, ...]:
        raw = (text or "").strip()
        if not raw:
            return (0.50,) * 6
        chars = [ord(c) for c in raw]
        n = float(len(chars))
        mean_c = sum(chars) / n
        var_c = sum((c - mean_c) ** 2 for c in chars) / n
        punct_ratio = sum(1 for c in raw if c in ",.;:!?—-") / max(1.0, n)
        upper_ratio = sum(1 for c in raw if c.isupper()) / max(1.0, n)
        space_ratio = sum(1 for c in raw if c.isspace()) / max(1.0, n)
        vals: list[float] = []
        for idx in range(6):
            window = chars[idx::6] or chars
            w_mean = (sum(window) / float(len(window))) / 122.0
            trig = (
                0.38 * min(1.0, max(0.0, w_mean))
                + 0.24 * min(1.0, var_c / 900.0)
                + 0.18 * min(1.0, punct_ratio * (6.0 + idx + salt))
                + 0.12 * min(1.0, space_ratio * 4.0)
                + 0.08 * min(1.0, upper_ratio * 8.0)
            )
            vals.append(round(min(0.88, max(0.18, trig)), 4))
        return tuple(vals)

    def _will(text: str, children: tuple[LearnedLeaf, ...] = ()) -> dict[str, float]:
        p = _continuous_profile(text, 1)
        out = {k: p[i] for i, k in enumerate(FOUNDATIONAL_LEAVES[:6])}
        for c in children:
            if c.leaf_id.startswith("cultural/"):
                out[c.leaf_id] = _evaluate_learned_leaf_from_anchors(c, out)
        return out

    def _truth(text: str, children: tuple[LearnedLeaf, ...] = ()) -> dict[str, float]:
        p = _continuous_profile(text, 2)
        out = {k: p[i] for i, k in enumerate(FOUNDATIONAL_LEAVES[6:12])}
        for c in children:
            if c.leaf_id.startswith("philosophical/"):
                out[c.leaf_id] = _evaluate_learned_leaf_from_anchors(c, out)
        return out

    def _feeler(text: str, children: tuple[LearnedLeaf, ...] = ()) -> dict[str, float]:
        p = _continuous_profile(text, 3)
        exp_keys = (
            "experiential/reflection/novelty",
            SALIENCE_LEAF,
            VALENCE_LEAF,
            AROUSAL_LEAF,
            "experiential/synchronization/resonance",
            "experiential/synchronization/entrainment",
        )
        out = {k: p[i] for i, k in enumerate(exp_keys)}
        for c in children:
            if c.leaf_id.startswith("experiential/"):
                out[c.leaf_id] = _evaluate_learned_leaf_from_anchors(c, out)
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
            suite = _get_shared_neural_bridges(self.repo_root)
            if suite is not None:
                self.sensors = suite.as_three_bridge_sensors()
            else:
                self.sensors = _default_fallback_sensors()

        self._scholar_left_brain = scholar_left_brain
        self._lazy_load_gpu = bool(lazy_load_gpu)
        self.sleep_cycle_index = 0
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

    def _resolve_live_projector_path(self) -> Path | None:
        for cand in (
            self.state_dir / "live_projector_weights.pt",
            self.scholar_checkpoint_dir / "live_projector_weights.pt",
        ):
            if cand.is_file():
                return cand
        return None

    def _resolve_live_lora_rank_scales_path(self) -> Path | None:
        for cand in (
            self.state_dir / "live_lora_rank_scales.pt",
            self.scholar_checkpoint_dir / "live_lora_rank_scales.pt",
        ):
            if cand.is_file():
                return cand
        return None

    def _load_scholar_locked(self, checkpoint_dir: Path) -> None:
        with self._lock:
            proj_path = checkpoint_dir / "projector_weights.pt"
            if not proj_path.exists():
                self.left_brain_status = "UNAVAILABLE (No GPU Checkpoint Found)"
                return
            try:
                lb = SubliminalGapLeftBrain.from_checkpoint(
                    checkpoint_dir,
                    max_clause_1_tokens=28,
                    max_post_gap_tokens=96,
                )
                if hasattr(lb, "projector") and hasattr(lb.projector, "capture_resting_anchor"):
                    lb.projector.capture_resting_anchor()

                loaded_live_proj = False
                live_proj = self._resolve_live_projector_path()
                if live_proj is not None and hasattr(lb, "projector"):
                    import torch

                    raw_state = torch.load(live_proj, map_location=lb.device, weights_only=True)
                    lb.projector.load_state_dict(raw_state, strict=True)
                    loaded_live_proj = True

                live_scales = self._resolve_live_lora_rank_scales_path()
                if live_scales is not None and hasattr(lb, "model"):
                    try:
                        import torch

                        scale_map = torch.load(live_scales, map_location=lb.device, weights_only=True)
                        if isinstance(scale_map, dict):
                            for mod_name, mod in lb.model.named_modules():
                                if isinstance(mod, LoRALinear) and mod_name in scale_map:
                                    saved_vec = scale_map[mod_name]
                                    if tuple(saved_vec.shape) == tuple(mod.rank_scale.shape):
                                        mod.rank_scale.copy_(
                                            saved_vec.to(device=mod.rank_scale.device, dtype=mod.rank_scale.dtype)
                                        )
                    except Exception:
                        pass

                self._scholar_left_brain = lb
                base_id = DEFAULT_SCHOLAR_HF_ID
                callosum_status = "TRAINED_STEERED_CALLOSUM"
                if getattr(lb, "callosum_metadata", None) is not None:
                    base_id = str(lb.callosum_metadata.base_model_id or base_id)
                    callosum_status = str(lb.callosum_metadata.callosum_status or callosum_status)
                    self.sleep_cycle_index = max(
                        self.sleep_cycle_index,
                        int(lb.callosum_metadata.sleep_cycle_index),
                    )
                else:
                    meta_path = checkpoint_dir / "corpus_callosum_meta.json"
                    if meta_path.exists():
                        try:
                            meta = json.loads(meta_path.read_text(encoding="utf-8"))
                            base_id = str(meta.get("base_model_id", base_id))
                            callosum_status = str(meta.get("callosum_status", callosum_status))
                        except Exception:
                            pass
                if loaded_live_proj:
                    callosum_status = f"{callosum_status}+LIVE_SCULPTED"
                self.left_brain_status = "READY (Impish_LLAMA_4B · 16-bit)"
                self.active_checkpoint_label = (
                    f"LMFT-Scholar · Impish_LLAMA_4B ({callosum_status} — {base_id})"
                )
            except Exception as exc:
                self.left_brain_status = f"UNAVAILABLE ({type(exc).__name__}: {exc})"

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

    def _extract_conversation_history_pairs(self) -> list[tuple[str, str]]:
        """Extract prior `(user_text, companion_text)` turns from `self.chat_messages`, skipping non-speech diagnostics."""
        pairs: list[tuple[str, str]] = []
        pending_user: str | None = None
        for msg in self.chat_messages:
            sender = str(msg.get("sender", ""))
            txt = str(msg.get("text", "")).strip()
            if sender == "user":
                pending_user = txt if txt else None
            elif sender == "companion" and pending_user is not None:
                if txt and not txt.startswith("[Offline Right-Brain Telemetry"):
                    pairs.append((pending_user, txt))
                pending_user = None
        return pairs

    def _run_scholar_turn(
        self,
        *,
        user_text: str,
        user_display_text: str | None = None,
        user_role_label: str,
        companion_role_label: str,
        fact_anchor_text: str | None = None,
        num_sentences: int | None = None,
        include_conversation_history: bool = True,
        lead_with_gap: bool | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self.scene_counter += 1
            scene_id = self.scene_counter
            use_equilibrium_stop = num_sentences is None
            target_sentences = 6 if num_sentences is None else max(1, min(6, int(num_sentences)))
            lb = self._ensure_left_brain_ready()
            if lb is None and not self._lazy_load_gpu:
                raise RuntimeError(
                    f"Left-Brain LLM is not loaded ({self.left_brain_status})."
                )

            effective_sensors: Any = self.sensors
            if fact_anchor_text and fact_anchor_text.strip():
                effective_sensors = _PassageAnchoredSensors(
                    wave_text=user_text,
                    fact_anchor_text=fact_anchor_text.strip(),
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
            history_pairs = (
                self._extract_conversation_history_pairs()
                if include_conversation_history
                else ()
            )
            resolved_lead = (
                bool(lead_with_gap)
                if lead_with_gap is not None
                else bool(getattr(self, "lead_with_gap", False))
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
                conversation_history=history_pairs,
                lead_with_gap=resolved_lead,
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
            if lb is None:
                spoken_reply = (
                    f"[Offline Right-Brain Telemetry — No Left-Brain Checkpoint Loaded | "
                    f"posture={vibe.posture}, active_concept={vibe.active_concept or 'none'}, "
                    f"clarity={vibe.clarity:.2f}]"
                )
            else:
                spoken_reply = (turn_result.final_conscious_note or "").strip()
                if not spoken_reply:
                    raise RuntimeError("Left-Brain returned empty generation output.")

            shown_user_text = (user_display_text or user_text).strip()
            self._turn_records_by_id[scene_id] = {
                "scene_index": scene_id,
                "scenario_text": user_text,
                "fact_anchor_text": fact_anchor_text or shown_user_text,
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
                    "text": shown_user_text,
                }
            )
            companion_msg: dict[str, Any] = {
                "id": scene_id,
                "sender": "companion",
                "role_label": companion_role_label,
                "text": spoken_reply,
                "coached": queued_now,
                "savored": False,
                "telemetry": {
                    "gear_used": turn_result.gear_used,
                    "outcome": turn_result.outcome,
                    "settled": turn_result.settled,
                    "active_concept": vibe.active_concept or "none",
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
        reference_text: str = "",
        force_coached: bool = False,
    ) -> UnresolvedSleepScene:
        vibe = turn.final_vibe
        p1_vibe = (
            turn.equilibrium.vibes[0]
            if getattr(turn.equilibrium, "vibes", None)
            else vibe
        )
        p1_coords = dict(p1_vibe.coordinates)
        ext_coords = dict(vibe.coordinates)
        ext_coords["psychological/truth/fact"] = round(float(vibe.fact), 6)

        scenario_with_note = (
            f"{scenario_text} [Coach note: {user_note.strip()}]"
            if user_note.strip()
            else scenario_text
        )
        if user_note.strip():
            ref_coords, _, _, _ = self.sensors.score_text(
                user_note.strip(),
                learned_leaves=self.learned_leaves,
            )
            contrastive_ref = dict(ref_coords)
        elif reference_text.strip():
            ref_coords, _, _, _ = self.sensors.score_text(
                reference_text.strip(),
                learned_leaves=self.learned_leaves,
            )
            contrastive_ref = dict(ref_coords)
        else:
            contrastive_ref = dict(p1_coords)

        raw_spoken = (spoken_reply or turn.final_conscious_note or "").strip()
        if raw_spoken.startswith("[Offline Right-Brain Telemetry"):
            raw_spoken = ""
        hunch_text = raw_spoken or scenario_text.strip()
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
            contrastive_reference_coordinates=contrastive_ref,
        )

    def handle_chat_message(
        self,
        text: str,
        *,
        num_sentences: int | None = None,
        anchor_to_canvas: bool = False,
    ) -> dict[str, Any]:
        clean = str(text or "").strip()
        if not clean:
            raise ValueError("Chat message text must be non-empty.")
        canvas_clean = self.study_canvas_text.strip() if anchor_to_canvas else ""
        if canvas_clean:
            llm_prompt = f"Reading Passage: \"{canvas_clean[:800]}\"\n\n{clean}"
            anchor_txt: str | None = canvas_clean
        else:
            llm_prompt = clean
            anchor_txt = None
        return self._run_scholar_turn(
            user_text=llm_prompt,
            user_display_text=clean,
            user_role_label="Socratic Dialogue",
            companion_role_label="LMFT-Scholar · Socratic Reflection",
            fact_anchor_text=anchor_txt,
            num_sentences=num_sentences,
            include_conversation_history=True,
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
            f"Study Passage: \"{clean_passage[:800]}\"\n"
            f"Reflect on the underlying Cultural, Philosophical, and Experiential tensions in this passage."
        )
        return self._run_scholar_turn(
            user_text=prompt,
            user_role_label="Passage Contemplation",
            companion_role_label="LMFT-Scholar · Passage Contemplation",
            fact_anchor_text=clean_passage,
            num_sentences=num_sentences,
            include_conversation_history=True,
        )

    def handle_step_curriculum_lesson(
        self,
        *,
        lesson_id: str = "",
        num_sentences: int | None = None,
        sleep_after_essay: bool = False,
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
        prompt = (
            f"[{selected.department}] {selected.title}\n"
            f"Passage: \"{selected.passage_text}\"\n\n"
            f"{selected.socratic_prompt}"
        )
        snap = self._run_scholar_turn(
            user_text=prompt,
            user_display_text=f"[{selected.department}] {selected.title}: {selected.socratic_prompt}",
            user_role_label=f"Curriculum Lesson #{self.curriculum_step_counter} · {selected.department}",
            companion_role_label=f"LMFT-Scholar · {selected.title}",
            fact_anchor_text=selected.passage_text,
            num_sentences=num_sentences,
            include_conversation_history=False,
        )
        if sleep_after_essay and self.active_lineage != "virgin":
            latest_id = self.scene_counter
            rec = self._turn_records_by_id.get(latest_id)
            if rec is not None and not any(s.scene_index == latest_id for s in self.sleep_queue):
                seminar_sleep_scene = self._build_sleep_scene_from_turn(
                    scene_index=latest_id,
                    scenario_text=prompt,
                    turn=rec["turn"],
                    spoken_reply=str(rec.get("spoken_reply", "")),
                    reference_text=f"{selected.passage_text} {selected.socratic_prompt}",
                    force_coached=False,
                )
                self.sleep_queue.append(seminar_sleep_scene)
            snap = self.handle_run_rem_sleep(auto_triggered=True)
            for msg in reversed(self.chat_messages):
                if msg.get("id") == latest_id:
                    msg["auto_rem_sleep_receipt"] = self.last_rem_sleep_receipt
                    break
            self._sync_active_thread_record(persist=True)
            snap = self.get_state_snapshot()
        return snap

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

            canopy_rules, canopy_leaves = self._grow_scholar_canopy_from_sleep_scenes(queued_snapshot)
            all_forged_rules = tuple(list(report.forged_rules) + list(canopy_rules))
            all_minted_leaves = tuple(list(report.minted_leaves) + list(canopy_leaves))
            self._last_minted_leaves = all_minted_leaves

            rep = max(queued_snapshot, key=lambda s: float(s.salience))
            rep_scene = rep.scene_index
            target_summary = (
                all_forged_rules[0].description
                if all_forged_rules
                else (rep.scenario_text or "").strip()
            )
            if target_summary:
                self.archives = apply_dream_reflection_to_archives(
                    self.archives,
                    scene_index=rep_scene,
                    dream_text=target_summary,
                )

            harvested_pairs = self._harvest_rem_contrastive_gap_pairs(
                queued_snapshot=queued_snapshot,
                vow_sentence=target_summary,
                forged_rules=all_forged_rules,
            )

            lb = self._ensure_left_brain_ready()
            projector_updated = False
            if self._last_harvested_gap_pairs and lb is not None and hasattr(lb, "projector"):
                try:
                    res_proj = update_projector_on_harvested_gap_pairs(
                        lb.projector,
                        self._last_harvested_gap_pairs,
                        state_dir=self.state_dir,
                        model=getattr(lb, "model", None),
                        tokenizer=getattr(lb, "tokenizer", None),
                        prompt_format=getattr(lb, "prompt_format", "chatml"),
                    )
                    projector_updated = bool(res_proj.get("updated", False))
                except Exception:
                    projector_updated = False

            self.sleep_queue.clear()
            self.rem_dream_chronicles.append(
                {
                    "cycle_index": self.sleep_cycle_index,
                    "forged_rules": [r.concept_name for r in all_forged_rules],
                    "minted_leaves": [lf.leaf_id for lf in all_minted_leaves],
                    "harvested_gap_pairs": harvested_pairs,
                    "projector_updated": projector_updated,
                }
            )
            self._persist_user_state()

            trig_label = "Auto-REM Sleep" if auto_triggered else "REM Sleep"
            self.last_rem_sleep_receipt = (
                f"{trig_label} Cycle #{self.sleep_cycle_index} Complete: "
                f"Forged {len(all_forged_rules)} rule(s), minted {len(all_minted_leaves)} leaf(ves), "
                f"harvested {len(report.harvested_clause_ids)} archive clause(s), "
                f"and logged {harvested_pairs} Callosum gap pair(s)."
            )
            return self.get_state_snapshot()

    def _grow_scholar_canopy_from_sleep_scenes(
        self,
        queued_snapshot: tuple[UnresolvedSleepScene, ...],
    ) -> tuple[tuple[FuzzyConceptRule, ...], tuple[LearnedLeaf, ...]]:
        """Grow coordinate-driven Tier-2, Tier-3, and Tier-4 `LearnedLeaf` nodes and `FuzzyConceptRule`s from `ext_coords` vs `ref_coords` with zero 9B Teacher dependency."""
        if not queued_snapshot or self.active_lineage == "virgin":
            return (), ()

        active_leaves = list(self.learned_leaves)
        existing_leaf_ids = {lf.leaf_id for lf in active_leaves}
        active_rules = list(self.rules)
        existing_rule_names = {r.concept_name for r in active_rules}
        minted_leaves: list[LearnedLeaf] = []
        forged_rules: list[FuzzyConceptRule] = []
        builtin_ids = {
            lf.leaf_id for lf in (PHILOSOPHICAL_SHADOW_LEAVES + DEFAULT_COMPLEX_FEELING_LEAVES)
        }

        growable_boughs_map: dict[str, tuple[str, str]] = {
            "cultural/efficiency": ("cultural/efficiency/power", "cultural/efficiency/wastelessness"),
            "cultural/justice": ("cultural/justice/honor", "cultural/justice/fairness"),
            "cultural/freedom": ("cultural/freedom/self-sufficiency", "cultural/freedom/needlessness"),
            "philosophical/being": ("philosophical/being/agency", "philosophical/being/existence"),
            "philosophical/structure": ("philosophical/structure/order", "philosophical/structure/chaos"),
            "philosophical/persistence": ("philosophical/persistence/continuity", "philosophical/persistence/discreteness"),
            "experiential/reflection": ("experiential/reflection/novelty", SALIENCE_LEAF),
            "experiential/phase": (VALENCE_LEAF, AROUSAL_LEAF),
            "experiential/synchronization": ("experiential/synchronization/resonance", "experiential/synchronization/entrainment"),
        }
        all_growable_leaf_keys: tuple[str, ...] = tuple(
            lf_key for pair in growable_boughs_map.values() for lf_key in pair
        )

        for sc in queued_snapshot:
            ext_coords = dict(sc.external_coordinates or {})
            ref_coords = dict(sc.contrastive_reference_coordinates or ext_coords)
            full_ext = dict(ext_coords)
            full_ext.setdefault(SALIENCE_LEAF, float(sc.salience))
            full_ext.setdefault(VALENCE_LEAF, float(sc.valence))
            full_ext.setdefault(AROUSAL_LEAF, float(sc.arousal))
            full_ref = dict(ref_coords)
            full_ref.setdefault(SALIENCE_LEAF, float(sc.salience))
            full_ref.setdefault(VALENCE_LEAF, float(sc.valence))
            full_ref.setdefault(AROUSAL_LEAF, float(sc.arousal))

            # Score all 9 growable boughs from the actual coordinate deviation + contrastive delta
            low_scen = (sc.scenario_text or "").lower()
            best_bough = "cultural/efficiency"
            best_bough_score = -1.0
            for bough_id, (f1, f2) in growable_boughs_map.items():
                r1 = float(full_ref.get(f1, 0.50))
                r2 = float(full_ref.get(f2, 0.50))
                e1 = float(full_ext.get(f1, 0.50))
                e2 = float(full_ext.get(f2, 0.50))
                score = (
                    abs(r1 - 0.50)
                    + abs(r2 - 0.50)
                    + 1.5 * (abs(r1 - e1) + abs(r2 - e2))
                )
                b_dept, b_sub = bough_id.split("/", 1)
                if f"[{b_dept} · {b_sub}]" in low_scen or bough_id in low_scen:
                    score += 0.35
                if score > best_bough_score:
                    best_bough_score = score
                    best_bough = bough_id

            target_bough = best_bough
            b_leaves = growable_boughs_map[target_bough]
            primary_leaf = max(
                b_leaves,
                key=lambda k: (
                    abs(float(full_ref.get(k, 0.50)) - 0.50)
                    + abs(float(full_ref.get(k, 0.50)) - float(full_ext.get(k, 0.50))),
                    float(full_ref.get(k, 0.50)),
                ),
            )
            cross_candidates = sorted(
                (k for k in all_growable_leaf_keys if k not in b_leaves),
                key=lambda k: (
                    abs(float(full_ref.get(k, 0.50)) - 0.50)
                    + abs(float(full_ref.get(k, 0.50)) - float(full_ext.get(k, 0.50))),
                    float(full_ref.get(k, 0.50)),
                ),
                reverse=True,
            )
            c1_leaf = cross_candidates[0] if len(cross_candidates) >= 1 else "philosophical/structure/order"
            c2_leaf = cross_candidates[1] if len(cross_candidates) >= 2 else "philosophical/persistence/continuity"
            c3_leaf = cross_candidates[2] if len(cross_candidates) >= 3 else "cultural/efficiency/wastelessness"

            p_short = primary_leaf.split("/")[-1].replace("-", "_")
            c1_short = c1_leaf.split("/")[-1].replace("-", "_")
            c2_short = c2_leaf.split("/")[-1].replace("-", "_")
            c3_short = c3_leaf.split("/")[-1].replace("-", "_")

            p_pole = "elevated" if float(full_ref.get(primary_leaf, 0.50)) >= 0.50 else "tempered"
            t2_slug = assert_domain_general_text(
                f"{p_short}_{c1_short}_synthesis"[:40],
                "leaf_name",
                banned_substrings=_SCHOLAR_BANNED_STORY_NOUNS,
            )
            t2_desc = assert_domain_general_text(
                f"Coordinate synthesis of {p_pole} {p_short} ({float(full_ref.get(primary_leaf, 0.50)):.2f}) with {c1_short} ({float(full_ref.get(c1_leaf, 0.50)):.2f}).",
                "description",
                banned_substrings=_SCHOLAR_BANNED_STORY_NOUNS,
            )
            t2_id = f"{target_bough}/{t2_slug}"

            chosen_leaf: LearnedLeaf | None = None
            partner_leaf: str = c1_leaf

            existing_t2_under_bough = [
                lf for lf in active_leaves
                if lf.tier == 2 and lf.parent_branch == target_bough and lf.leaf_id not in builtin_ids
            ]
            if t2_id not in existing_leaf_ids and not existing_t2_under_bough:
                raw_leaf = score_leaf_anchor_weights_at_birth(
                    LearnedLeaf(
                        parent_branch=target_bough,
                        leaf_name=t2_slug,
                        description=t2_desc,
                    )
                )
                chosen_leaf = sculpt_learned_leaf_anchor_weights(raw_leaf, full_ext, full_ref)
                partner_leaf = c1_leaf
            else:
                parent_t2 = next((lf for lf in existing_t2_under_bough if lf.leaf_id == t2_id), existing_t2_under_bough[0])
                existing_t3_under_t2 = [
                    lf for lf in active_leaves
                    if lf.tier == 3 and lf.parent_branch == parent_t2.leaf_id and lf.leaf_id not in builtin_ids
                ]
                t3_slug = assert_domain_general_text(
                    f"{p_short}_{c2_short}_accord"[:40],
                    "leaf_name",
                    banned_substrings=_SCHOLAR_BANNED_STORY_NOUNS,
                )
                t3_desc = assert_domain_general_text(
                    f"Tier-3 sub-branch refining {parent_t2.leaf_name} along {c2_short} ({float(full_ref.get(c2_leaf, 0.50)):.2f}).",
                    "description",
                    banned_substrings=_SCHOLAR_BANNED_STORY_NOUNS,
                )
                t3_id = f"{parent_t2.leaf_id}/{t3_slug}"
                if t3_id not in existing_leaf_ids and not existing_t3_under_t2:
                    raw_leaf = score_leaf_anchor_weights_at_birth(
                        LearnedLeaf(
                            parent_branch=parent_t2.leaf_id,
                            leaf_name=t3_slug,
                            description=t3_desc,
                        )
                    )
                    chosen_leaf = sculpt_learned_leaf_anchor_weights(raw_leaf, full_ext, full_ref)
                    partner_leaf = c2_leaf
                else:
                    parent_t3 = next((lf for lf in existing_t3_under_t2 if lf.leaf_id == t3_id), existing_t3_under_t2[0])
                    t4_slug = assert_domain_general_text(
                        f"{p_short}_{c3_short}_focus"[:40],
                        "leaf_name",
                        banned_substrings=_SCHOLAR_BANNED_STORY_NOUNS,
                    )
                    t4_desc = assert_domain_general_text(
                        f"Tier-4 detective leaf refining {parent_t3.leaf_name} along {c3_short} ({float(full_ref.get(c3_leaf, 0.50)):.2f}).",
                        "description",
                        banned_substrings=_SCHOLAR_BANNED_STORY_NOUNS,
                    )
                    t4_id = f"{parent_t3.leaf_id}/{t4_slug}"
                    if t4_id not in existing_leaf_ids:
                        raw_leaf = score_leaf_anchor_weights_at_birth(
                            LearnedLeaf(
                                parent_branch=parent_t3.leaf_id,
                                leaf_name=t4_slug,
                                description=t4_desc,
                            )
                        )
                        chosen_leaf = sculpt_learned_leaf_anchor_weights(raw_leaf, full_ext, full_ref)
                        partner_leaf = c3_leaf

            if chosen_leaf is not None:
                active_leaves.append(chosen_leaf)
                minted_leaves.append(chosen_leaf)
                existing_leaf_ids.add(chosen_leaf.leaf_id)

            # Contrastive weight-space sculpting for all custom leaves under this bough
            for idx_lf, lf in enumerate(active_leaves):
                if (
                    lf.leaf_id not in builtin_ids
                    and lf.leaf_id not in _DEFAULT_LEAF_ANCHOR_WEIGHTS
                    and lf.leaf_id.startswith(target_bough + "/")
                ):
                    active_leaves[idx_lf] = sculpt_learned_leaf_anchor_weights(
                        lf,
                        full_ext,
                        full_ref,
                    )

            # Evaluate learned leaves on full_ref in tier order so rule target values come from real coordinates
            for tier_val in (2, 3, 4):
                for lf in active_leaves:
                    if lf.tier == tier_val:
                        full_ref[lf.leaf_id] = _evaluate_learned_leaf_from_anchors(lf, full_ref)

            rule_targets: dict[str, float] = {}
            if primary_leaf in FOUNDATIONAL_LEAVES:
                rule_targets[primary_leaf] = round(float(full_ref.get(primary_leaf, 0.50)), 4)
            if partner_leaf in FOUNDATIONAL_LEAVES and partner_leaf != primary_leaf:
                rule_targets[partner_leaf] = round(float(full_ref.get(partner_leaf, 0.50)), 4)
            target_leaf_obj = chosen_leaf or (existing_t2_under_bough[-1] if existing_t2_under_bough else None)
            if target_leaf_obj is not None:
                rule_targets[target_leaf_obj.leaf_id] = round(
                    float(full_ref.get(target_leaf_obj.leaf_id, 0.50)), 4
                )

            if rule_targets:
                min_t = min(rule_targets.values())
                max_t = max(rule_targets.values())
                rule_low = round(min(0.72, max(0.35, min_t - 0.12)), 4)
                rule_high = round(min(0.88, max(rule_low + 0.12, max_t + 0.04)), 4)
                concept_name = (
                    chosen_leaf.leaf_name
                    if chosen_leaf is not None
                    else (target_leaf_obj.leaf_name if target_leaf_obj is not None else f"{p_short}_accord")
                )
                desc_text = (
                    chosen_leaf.description
                    if chosen_leaf is not None
                    else f"Coordinate equilibrium across {primary_leaf} and {partner_leaf}."
                )
                candidate_rule = FuzzyConceptRule(
                    rule_id=f"rule-scholar-{concept_name[:28]}",
                    concept_name=concept_name,
                    description=desc_text,
                    target_leaves=rule_targets,
                    low=rule_low,
                    high=rule_high,
                    polarity="attractive",
                )
                if concept_name not in existing_rule_names:
                    active_rules.append(candidate_rule)
                    forged_rules.append(candidate_rule)
                    existing_rule_names.add(concept_name)
                else:
                    for r_idx, ex_r in enumerate(active_rules):
                        if ex_r.concept_name == concept_name:
                            entrained = entrain_existing_cluster_rule(
                                ex_r,
                                candidate_rule,
                                concept_name,
                            )
                            active_rules[r_idx] = entrained
                            forged_rules.append(entrained)
                            break

        self.learned_leaves = tuple(active_leaves)
        self.rules = tuple(active_rules)
        return tuple(forged_rules), tuple(minted_leaves)

    def _harvest_rem_contrastive_gap_pairs(
        self,
        *,
        queued_snapshot: tuple[UnresolvedSleepScene, ...],
        vow_sentence: str,
        forged_rules: tuple[FuzzyConceptRule, ...],
    ) -> int:
        self._last_harvested_gap_pairs = []
        clean_vow = " ".join(str(vow_sentence or "").split()).strip()
        if not queued_snapshot:
            return 0
        out_path = self.state_dir / "scholar_gap_trajectories.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        valid_keys = set(FOUNDATIONAL_LEAVES) | {lf.leaf_id for lf in self.learned_leaves}
        harvested_now = 0

        for sc in queued_snapshot:
            raw_hunch = " ".join(str(sc.left_brain_hunch or "").split()).strip()
            target_text = clean_vow or " ".join(str(sc.scenario_text or "").split()).strip()
            if not raw_hunch or not target_text or raw_hunch.lower() == target_text.lower():
                continue
            if "," in raw_hunch:
                c1_part, cont_b_part = raw_hunch.split(",", 1)
                c1_prefix = f"{c1_part.strip()},"
                cont_b = cont_b_part.strip() or raw_hunch
            else:
                c1_prefix = ""
                cont_b = raw_hunch
            cont_a = target_text
            if cont_a.strip().lower() == cont_b.strip().lower():
                continue

            # Un-rigged coordinates: coords_b is the raw scene external_coordinates; coords_a is the raw contrastive_reference_coordinates
            coords_b: dict[str, float] = {k: 0.50 for k in FOUNDATIONAL_LEAVES}
            for k, v in (sc.external_coordinates or {}).items():
                if k in valid_keys:
                    coords_b[k] = float(v)
            for tier_val in (2, 3, 4):
                for lf in self.learned_leaves:
                    if lf.tier == tier_val and lf.leaf_id not in coords_b:
                        coords_b[lf.leaf_id] = _evaluate_learned_leaf_from_anchors(lf, coords_b)

            _, exp_b_raw, _, _ = self.sensors.score_text(cont_b, learned_leaves=self.learned_leaves)
            exp_b = dict(exp_b_raw)
            exp_b[SALIENCE_LEAF] = float(sc.salience)
            exp_b[VALENCE_LEAF] = float(sc.valence)
            exp_b[AROUSAL_LEAF] = float(sc.arousal)

            vibe_b, _ = compile_right_brain_vibe(
                self.birth_state,
                coords_b,
                exp_b,
                VIRGIN_FOUNDATIONAL_RULES,
                pass_index=1,
                event_id=f"scholar_gap_b_{sc.scene_index}",
                learned_leaves=self.learned_leaves,
                will_edges=self.will_edges,
            )
            chord_b = SubliminalChordVector.from_right_brain_vibe(
                vibe_b,
                learned_leaves=self.learned_leaves,
            )

            coords_a: dict[str, float] = {k: float(coords_b.get(k, 0.50)) for k in FOUNDATIONAL_LEAVES}
            if sc.contrastive_reference_coordinates:
                for rk, rv in sc.contrastive_reference_coordinates.items():
                    if rk in valid_keys:
                        coords_a[rk] = float(rv)
            for tier_val in (2, 3, 4):
                for lf in self.learned_leaves:
                    if lf.tier == tier_val:
                        coords_a[lf.leaf_id] = _evaluate_learned_leaf_from_anchors(lf, coords_a)

            _, exp_a_raw, _, _ = self.sensors.score_text(cont_a, learned_leaves=self.learned_leaves)
            exp_a = dict(exp_a_raw)

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
                will_edges=self.will_edges,
            )
            chord_a = SubliminalChordVector.from_right_brain_vibe(
                vibe_a,
                learned_leaves=self.learned_leaves,
            )
            if chord_l2_distance(chord_a, chord_b) < 0.15:
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
                    min_l2_distance=0.15,
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
            live_proj_src = self._resolve_live_projector_path()
            copied_live_proj = False
            if live_proj_src is not None and live_proj_src.is_file():
                shutil.copy2(live_proj_src, target_model_dir / "live_projector_weights.pt")
                shutil.copy2(live_proj_src, target_state_dir / "live_projector_weights.pt")
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

    def _build_tree_inspector_snapshot(self) -> dict[str, Any]:
        builtin_ids = {
            lf.leaf_id for lf in (PHILOSOPHICAL_SHADOW_LEAVES + DEFAULT_COMPLEX_FEELING_LEAVES)
        }
        visible_leaves = (
            ()
            if self.active_lineage == "virgin"
            else tuple(lf for lf in self.learned_leaves if lf.leaf_id not in builtin_ids)
        )
        visible_rules = (
            VIRGIN_FOUNDATIONAL_RULES
            if self.active_lineage == "virgin"
            else self.rules
        )

        latest_coords: dict[str, float] = {}
        for msg in reversed(self.chat_messages):
            tel = msg.get("telemetry")
            if isinstance(tel, dict) and isinstance(tel.get("theory_coordinates"), dict):
                latest_coords.update({str(k): float(v) for k, v in tel["theory_coordinates"].items()})
                if "fact" in tel:
                    latest_coords["psychological/truth/fact"] = float(tel["fact"])
                if "theory" in tel:
                    latest_coords["psychological/truth/theory"] = float(tel["theory"])
                if "intention" in tel:
                    latest_coords["psychological/will/intention"] = float(tel["intention"])
                if "execution" in tel:
                    latest_coords["psychological/will/execution"] = float(tel["execution"])
                if "clarity" in tel:
                    latest_coords["psychological/contemplation/clarity"] = float(tel["clarity"])
                if "confidence" in tel:
                    latest_coords["psychological/contemplation/confidence"] = float(tel["confidence"])
                if "valence" in tel:
                    latest_coords["experiential/phase/valence"] = float(tel["valence"])
                if "arousal" in tel:
                    latest_coords["experiential/phase/arousal"] = float(tel["arousal"])
                break

        # Evaluate any newly sprouted Tier 2/3/4 leaves over `latest_coords` in tier order
        for tier_val in (2, 3, 4):
            for lf in visible_leaves:
                if lf.tier == tier_val:
                    latest_coords[lf.leaf_id] = _evaluate_learned_leaf_from_anchors(lf, latest_coords)

        tier_counts = {
            "tier_1": 24,
            "tier_2": sum(1 for lf in visible_leaves if lf.tier == 2),
            "tier_3": sum(1 for lf in visible_leaves if lf.tier == 3),
            "tier_4": sum(1 for lf in visible_leaves if lf.tier >= 4),
        }

        departments_out: list[dict[str, Any]] = []
        for dept_id, dept_label, boughs in _FOUNDING_TREE_STRUCTURE:
            boughs_out: list[dict[str, Any]] = []
            anchor_names_for_dept = _DEPT_ANCHOR_LEAF_NAMES.get(dept_id, FOUNDATIONAL_LEAVES[:8])
            for bough_id, bough_label, f_leaves in boughs:
                founding_items = [
                    {
                        "leaf_id": fl_id,
                        "short_name": fl_id.split("/")[-1],
                        "tier": 1,
                        "activation": round(float(latest_coords.get(fl_id, 0.50)), 3),
                    }
                    for fl_id in f_leaves
                ]
                grown_items: list[dict[str, Any]] = []
                for lf in visible_leaves:
                    if lf.leaf_id.startswith(bough_id + "/"):
                        anchor_map: dict[str, float] = {}
                        if lf.anchor_weights:
                            paired = sorted(
                                zip(anchor_names_for_dept, lf.anchor_weights),
                                key=lambda kv: abs(float(kv[1])),
                                reverse=True,
                            )[:3]
                            anchor_map = {str(k): round(float(v), 2) for k, v in paired}
                        live_act = _evaluate_learned_leaf_from_anchors(lf, latest_coords)
                        grown_items.append(
                            {
                                "leaf_id": lf.leaf_id,
                                "short_name": lf.leaf_name,
                                "leaf_name": lf.leaf_name,
                                "parent_branch": lf.parent_branch,
                                "tier": lf.tier,
                                "description": lf.description,
                                "activation": round(float(live_act), 3),
                                "anchor_weights": anchor_map,
                            }
                        )
                bough_act = (
                    sum(float(item["activation"]) for item in founding_items) / max(1, len(founding_items))
                )
                boughs_out.append(
                    {
                        "bough_id": bough_id,
                        "title": bough_label,
                        "bough_label": bough_label,
                        "activation": round(bough_act, 3),
                        "founding_leaves": founding_items,
                        "learned_leaves": grown_items,
                        "grown_leaves": grown_items,
                    }
                )
            dept_act = (
                sum(float(b["activation"]) for b in boughs_out) / max(1, len(boughs_out))
            )
            departments_out.append(
                {
                    "department_id": dept_id,
                    "title": dept_label,
                    "department_label": dept_label,
                    "activation": round(dept_act, 3),
                    "boughs": boughs_out,
                }
            )

        return {
            "founding_node_count": 40,
            "total_tree_nodes": 40 + len(visible_leaves),
            "total_node_count": 40 + len(visible_leaves),
            "learned_leaves_count": len(visible_leaves),
            "rules_count": len(visible_rules),
            "tier_counts": tier_counts,
            "departments": departments_out,
            "rules": [
                {
                    "rule_id": r.rule_id,
                    "concept_name": r.concept_name,
                    "target_leaf": f"Tier {r.tier} ({getattr(r, 'polarity', 'attractive')})",
                    "top_clauses": {k: round(float(v), 2) for k, v in r.target_leaves.items()},
                    "source": r.rule_id,
                    "confidence": round(float(r.high), 2),
                    "description": r.description,
                }
                for r in visible_rules
            ],
            "will_synapses": [
                {
                    "id": e.id,
                    "edge_id": f"{e.source_id.split('/')[-1]} ➔ {e.target_id.split('/')[-1]}",
                    "source": e.source_id.split("/")[-1],
                    "target": e.target_id.split("/")[-1],
                    "weight": round(float(e.weight), 4),
                }
                for e in self.will_edges
            ],
            "archive_counts": {
                "executive": len(self.archives.executive.action_clauses),
                "legislative": len(self.archives.legislative.feeling_clauses),
                "judicial": len(self.archives.judicial.structural_clauses),
            },
            "has_live_callosum_projector": self._resolve_live_projector_path() is not None,
            "callosum_live_weights": self._resolve_live_projector_path() is not None,
        }

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
            "tree_inspector": self._build_tree_inspector_snapshot(),
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
                        anchor_to_canvas=bool(body.get("anchor_to_canvas", False)),
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
                        sleep_after_essay=bool(body.get("sleep_after_essay", False)),
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
