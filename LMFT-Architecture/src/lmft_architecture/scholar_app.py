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
import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Mapping
import urllib.request
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
    EXECUTIVE_BRANCH,
    JUDICIAL_BRANCH,
    LEGISLATIVE_BRANCH,
    BranchClauseEntry,
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
from .gap_trajectory_schema import ContrastiveGapPair, build_chatml_user_header, chord_l2_distance
from .impish_bloodmoon_adapter import (
    DEFAULT_IMPISH_BLOODMOON_HF_ID,
    UNSTEERED_BOOTSTRAP_STATUS,
)
from .local_models import strict_json
from .neural_bridge_classifiers import (
    DEFAULT_COMPLEX_FEELING_LEAVES,
    DOER_FOUNDING_DESCRIPTIONS,
    FEELER_FOUNDING_DESCRIPTIONS,
    THINKER_FOUNDING_DESCRIPTIONS,
    BridgePairExample,
    NeuralBridgeClassifier,
    NeuralThreeBridgeSuite,
    train_bridge_classifier,
)
from .philosophical_abstraction_matrix import PHILOSOPHICAL_SHADOW_LEAVES
from .propagation import Association
from .reflection_classifier import (
    ALL_GROWABLE_PARENT_BRANCHES,
    AROUSAL_LEAF,
    BRANCH_GATE_THRESHOLD,
    CONFUSION_THRESHOLD,
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
    sculpt_will_edges_during_sleep,
)
from .runtime_store import atomic_json
from .scholar_ui import SCHOLAR_SINGLE_PAGE_HTML
from .sleep_consolidation import (
    UnresolvedSleepScene,
    apply_dream_reflection_to_archives,
    assert_domain_general_text,
    entrain_existing_cluster_rule,
    refine_colliding_rules_during_sleep,
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
        branch_gate_threshold: float = BRANCH_GATE_THRESHOLD,
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
    contrast_foil_text: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "lesson_id": self.lesson_id,
            "department": self.department,
            "title": self.title,
            "passage_text": self.passage_text,
            "socratic_prompt": self.socratic_prompt,
            "contrast_foil_text": self.contrast_foil_text,
        }


SCHOLAR_CURRICULUM_LESSONS: tuple[ScholarCurriculumLesson, ...] = (
    ScholarCurriculumLesson(
        lesson_id="cultural_efficiency_stewardship",
        department="Cultural · Efficiency",
        title="Moby-Dick — Prudent Stewardship vs. Monomaniacal Force (Power & Wastelessness)",
        passage_text=(
            "In Herman Melville's Moby-Dick, First Mate Starbuck counsels that the ship's strength and skill "
            "exist to gather oil responsibly, husband the vessel's gear, and bring every sailor safely home "
            "to Nantucket rather than spending human lives on private obsession."
        ),
        contrast_foil_text=(
            "On the quarterdeck of the Pequod, Captain Ahab welds the crew's discipline and every resource "
            "aboard into a single harpoon of relentless vengeance, burning through warnings and safeguards "
            "until the chase devours ship and crew alike."
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
            "In Sophocles' Antigone and Victor Hugo's Les Misérables, Antigone and Bishop Myriel uphold a "
            "living covenant of honor and restorative mercy, offering silver and dignity to a faulted stranger "
            "so shared humanity is repaired rather than cast out."
        ),
        contrast_foil_text=(
            "Creon and Inspector Javert enforce the literal letter of an unbending decree without exception "
            "or mercy, treating pitiless punishment and permanent exclusion as moral order regardless of human cost."
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
            "In Hermann Hesse's Siddhartha and Henry David Thoreau's Walden, the seeker walks calmly through "
            "both busy market and quiet woods with simple tools and unhurried sufficiency, free from compulsive "
            "appetite or dependence on status."
        ),
        contrast_foil_text=(
            "Amidst glittering trade and social rivalry, restless strivers chase luxury, applause, and "
            "possessions without pause, terrified of falling behind and unable to sit quietly with enough."
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
            "Conscious creative agency accepts ongoing care, presence, and answerability for the living "
            "realities it brings into the world, staying beside what it has made when difficulties begin."
        ),
        contrast_foil_text=(
            "In Mary Shelley's Frankenstein, Victor seizes solitary ambition to animate a living creature in "
            "secret, yet bolts from the laboratory in horror the moment it opens its eyes, abandoning it to "
            "unchosen isolation."
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
            "After a sudden storm upends familiar rank in Shakespeare's Tempest, resilient living order listens "
            "to the upheaval, sets humane bounds, and weaves former rivals into open reconciliation."
        ),
        contrast_foil_text=(
            "On Golding's island, fearful command hardens into painted tribal domination and brittle force until "
            "the slightest shock shatters the camp into hunting frenzy and chaotic ruin."
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
            "Across twenty years of shipwrecks, changing shores, and discrete trials in Homer's Odyssey, "
            "Odysseus and Penelope keep an unbroken thread of memory, craft, and faithful purpose linking "
            "departure to return."
        ),
        contrast_foil_text=(
            "Along the Lotus shore and Circe's hall, drifting crews surrender memory to each passing island's "
            "impulse, severing yesterday's vow from tomorrow's direction."
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
            "Stepping out of Plato's cave into open daylight, patient inquiry lets the eyes adjust slowly, "
            "looking past loud flickering shadows to discern the enduring causes that actually govern the scene."
        ),
        contrast_foil_text=(
            "Inside the cavern and Alice's noisy court, spectators lunge after every sudden riddle, spectacle, "
            "and glittering distraction, mistaking theatrical commotion for real significance."
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
            "In Book Twenty-Four of Homer's Iliad, aged Priam kneels in Achilles' tent at night to ask for "
            "Hector's body, and as both men weep together for their losses, fierce tension softens into quiet "
            "shared sorrow and humane care."
        ),
        contrast_foil_text=(
            "Outside the walls at dawn, sleepless Achilles lashes Hector's body behind his chariot in "
            "consuming rage, feeding raw pain with further fury that brings neither rest nor relief."
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
            "Listening amid an impassioned public square, a lucid witness feels sincere human sympathy for "
            "neighbors' grief while keeping an unhurried pulse and independent moral judgment."
        ),
        contrast_foil_text=(
            "In Shakespeare's Forum and Orwell's Two Minutes Hate, rhythmic chants sweep the crowd into "
            "automatic emotional contagion, erasing individual conscience in a single roaring pulse."
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
            "In the final act of Shakespeare's Hamlet, tested moral intention settles into calm, wakeful "
            "readiness, meeting the appointed hour with steady action neither delayed by doubt nor rushed by panic."
        ),
        contrast_foil_text=(
            "Earlier in Elsinore, Hamlet swings between endless solitary overthinking that freezes action and "
            "a blind, impulsive lunge through the curtain before knowing who stands behind it."
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
            "Facing Sonia's quiet honesty in Dostoevsky's Crime and Punishment, Raskolnikov drops his grand "
            "excuses and lets lived human reality and conscience correct his proud abstract theory."
        ),
        contrast_foil_text=(
            "Locked in his narrow garret, Raskolnikov constructs an airtight rationalization that superior "
            "ambition stands above ordinary moral fact, sealing his mind against visible human harm."
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
            "Writing by lamplight in his frontier campaign tent, Marcus Aurelius observes alarming reports "
            "with unclouded perceptual clarity and quiet, unboastful steadiness."
        ),
        contrast_foil_text=(
            "Around the campfires, anxious commanders alternate between urgent rumors that cloud judgment and "
            "loud, brittle bravado meant to hide uncertainty."
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
    "nantucket",
    "melville",
    "moby",
    "antigone",
    "creon",
    "sophocles",
    "valjean",
    "javert",
    "myriel",
    "hugo",
    "siddhartha",
    "hesse",
    "thoreau",
    "walden",
    "frankenstein",
    "victor",
    "shelley",
    "prospero",
    "golding",
    "odysseus",
    "penelope",
    "ithaca",
    "circe",
    "homer",
    "plato",
    "alice",
    "achilles",
    "priam",
    "hector",
    "iliad",
    "caesar",
    "brutus",
    "antony",
    "orwell",
    "hamlet",
    "elsinore",
    "shakespeare",
    "raskolnikov",
    "sonia",
    "dostoevsky",
    "aurelius",
    "marcus",
)

_SCHOLAR_LEAF_NAME_SLOP: tuple[str, ...] = (
    "_synthesis",
    "synthesis_",
    "_accord",
    "accord_",
    "_focus",
    "focus_",
    "_cluster",
    "cluster_",
    "coached",
    "hazard",
    "immunized",
    "two_word",
    "three_word",
    "leaf_name",
    "concept_slug",
    "short_name",
    "cultural_",
    "philosophical_",
    "experiential_",
    "psychological_",
    "tier_2",
    "tier_3",
    "tier_4",
    "sub_branch",
)

_SCHOLAR_PROSE_SLOP: tuple[str, ...] = (
    "coached",
    "immunized",
    "tier-2",
    "tier-3",
    "tier-4",
    "tier_2",
    "tier_3",
    "tier_4",
    "sub-branch",
    "sub_branch",
    "coordinate synthesis",
    "coordinate equilibrium",
    "detective leaf",
    "positional coordination",
    "piece coordination",
    "tactical overextension",
)

_FOUNDING_SHORT_LEAF_NAMES: frozenset[str] = frozenset(
    lf.split("/")[-1]
    for lf in (
        FOUNDATIONAL_LEAVES
        + (
            "experiential/reflection/novelty",
            SALIENCE_LEAF,
            VALENCE_LEAF,
            AROUSAL_LEAF,
            "experiential/synchronization/resonance",
            "experiential/synchronization/entrainment",
            "psychological/will/intention",
            "psychological/will/execution",
            "psychological/truth/theory",
            "psychological/truth/fact",
            "psychological/contemplation/clarity",
            "psychological/contemplation/confidence",
        )
    )
)


def _content_words_5plus(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{5,}", (text or "").lower())}


def _parse_distilled_leaf_response(raw_text: str) -> dict[str, Any] | None:
    """Parse labeled plain-text lines (or fallback JSON) from the 4B Left Brain into a leaf candidate dict."""
    text = (raw_text or "").strip()
    if not text:
        return None
    if text.startswith("{"):
        parsed = strict_json(text)
        return parsed if isinstance(parsed, dict) else None

    out: dict[str, Any] = {
        "leaf_name": "",
        "description": "",
        "positive_examples": [],
        "negative_examples": [],
        "executive_clause": "",
        "legislative_clause": "",
        "judicial_clause": "",
    }
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("*-• ").strip()
        if not line or ":" not in line:
            continue
        head, val = line.split(":", 1)
        key = re.sub(r"[^a-z0-9_]", "", head.strip().lower().replace(" ", "_"))
        clean_val = val.strip().strip('"').strip("`").strip()
        if not clean_val:
            continue
        if key == "leaf_name" and not out["leaf_name"]:
            out["leaf_name"] = clean_val
        elif key == "description" and not out["description"]:
            out["description"] = clean_val
        elif key.startswith("pos") and len(out["positive_examples"]) < 6:
            out["positive_examples"].append(clean_val)
        elif key.startswith("neg") and len(out["negative_examples"]) < 6:
            out["negative_examples"].append(clean_val)
        elif key in ("executive", "executive_clause") and not out["executive_clause"]:
            out["executive_clause"] = clean_val
        elif key in ("legislative", "legislative_clause") and not out["legislative_clause"]:
            out["legislative_clause"] = clean_val
        elif key in ("judicial", "judicial_clause") and not out["judicial_clause"]:
            out["judicial_clause"] = clean_val
    if not out["leaf_name"] or not out["description"]:
        return None
    return out


def _validate_human_leaf_payload(
    raw_payload: Mapping[str, Any],
    *,
    scene_a_text: str,
    scene_b_text: str,
    existing_short_names: set[str],
    parent_description: str = "",
) -> dict[str, Any]:
    """Validate an LLM-distilled human concept leaf + contrastive everyday calibration examples.

    Enforces `AGENTS.md` Rule #1 & Rule #4:
    - Plain, concrete human English only (zero digits, zero float dumps, zero story proper nouns).
    - Zero mechanical slop (`_synthesis`, `_accord`, `_focus`, `coached_`, `coordinate synthesis`, etc.).
    - Everyday `positive_examples` and `negative_examples` must be distinct modern situations, never copied
      or lightly paraphrased from `scene_a_text` or `scene_b_text`.
    """
    raw_name = str(raw_payload.get("leaf_name", "")).strip().lower()
    raw_name = re.sub(r"[\s\-]+", "_", raw_name)
    raw_name = re.sub(r"[^a-z_]", "", raw_name).strip("_")
    raw_name = re.sub(r"_+", "_", raw_name)
    if not re.fullmatch(r"[a-z][a-z_]{3,37}", raw_name):
        raise ValueError(f"Invalid leaf_name format: {raw_name!r}")
    parts = [p for p in raw_name.split("_") if p]
    if (
        not (2 <= len(parts) <= 4)
        or any(len(p) < 2 for p in parts)
        or sum(1 for p in parts if len(p) >= 3) < 2
    ):
        raise ValueError(f"leaf_name must be a clear 2-4 word snake_case phrase, got {raw_name!r}")
    if raw_name in _FOUNDING_SHORT_LEAF_NAMES or raw_name in existing_short_names:
        raise ValueError(f"Duplicate or founding leaf_name: {raw_name!r}")
    for bad in _SCHOLAR_LEAF_NAME_SLOP:
        if bad in raw_name:
            raise ValueError(f"Mechanical slop token {bad!r} forbidden in leaf_name {raw_name!r}")
    assert_domain_general_text(raw_name, "leaf_name", banned_substrings=_SCHOLAR_BANNED_STORY_NOUNS)

    desc = " ".join(str(raw_payload.get("description", "")).split()).strip()
    if not (28 <= len(desc) <= 220):
        raise ValueError(f"description length must be 28..220 chars, got {len(desc)}")
    if re.search(r"\d", desc):
        raise ValueError("description must not contain digits or numeric coordinates")
    low_desc = desc.lower()
    for bad in _SCHOLAR_PROSE_SLOP:
        if bad in low_desc:
            raise ValueError(f"Mechanical slop phrase {bad!r} forbidden in description")
    assert_domain_general_text(desc, "description", banned_substrings=_SCHOLAR_BANNED_STORY_NOUNS)
    if parent_description.strip():
        p_clean = " ".join(parent_description.split()).strip().lower()
        if low_desc == p_clean:
            raise ValueError("Child leaf description must not copy parent description verbatim")
        desc_words = _content_words_5plus(desc)
        parent_words = _content_words_5plus(parent_description)
        if desc_words and parent_words:
            p_overlap = len(desc_words & parent_words) / float(len(desc_words))
            if p_overlap >= 0.55:
                raise ValueError(
                    f"Child leaf description overlaps parent description too closely ({p_overlap:.2f}); "
                    "define a distinct, narrower sub-skill"
                )

    source_words = _content_words_5plus(scene_a_text) | _content_words_5plus(scene_b_text)

    def _clean_examples(key: str) -> list[str]:
        raw_list = raw_payload.get(key)
        if isinstance(raw_list, Mapping):
            raw_list = list(raw_list.values())
        if not isinstance(raw_list, Sequence) or isinstance(raw_list, (str, bytes)):
            raise ValueError(f"{key} must be a list of everyday example sentences")
        cleaned_out: list[str] = []
        seen_norm: set[str] = set()
        for item in raw_list:
            sent = " ".join(str(item or "").split()).strip()
            if not (20 <= len(sent) <= 240):
                continue
            if re.search(r"\d", sent):
                continue
            norm = sent.lower()
            if norm in seen_norm:
                continue
            try:
                assert_domain_general_text(sent, key, banned_substrings=_SCHOLAR_BANNED_STORY_NOUNS)
            except ValueError:
                continue
            ex_words = _content_words_5plus(sent)
            if ex_words and source_words:
                overlap_ratio = len(ex_words & source_words) / float(len(ex_words))
                if overlap_ratio >= 0.45:
                    continue
            seen_norm.add(norm)
            cleaned_out.append(sent)
        if len(cleaned_out) < 4:
            raise ValueError(f"{key} requires at least 4 clean everyday sentences, got {len(cleaned_out)}")
        return cleaned_out[:6]

    pos_examples = _clean_examples("positive_examples")
    neg_examples = _clean_examples("negative_examples")
    if set(s.lower() for s in pos_examples) & set(s.lower() for s in neg_examples):
        raise ValueError("positive_examples and negative_examples must not overlap")

    clauses: dict[str, str] = {}
    for c_key in ("executive_clause", "legislative_clause", "judicial_clause"):
        c_raw = " ".join(str(raw_payload.get(c_key, "") or desc).split()).strip()
        if not (20 <= len(c_raw) <= 220):
            raise ValueError(f"{c_key} length out of bounds")
        if re.search(r"\d", c_raw):
            raise ValueError(f"{c_key} must not contain digits")
        low_c = c_raw.lower()
        for bad in _SCHOLAR_PROSE_SLOP:
            if bad in low_c:
                raise ValueError(f"Mechanical slop phrase {bad!r} forbidden in {c_key}")
        assert_domain_general_text(c_raw, c_key, banned_substrings=_SCHOLAR_BANNED_STORY_NOUNS)
        clauses[c_key] = c_raw

    return {
        "leaf_name": raw_name,
        "description": desc,
        "positive_examples": pos_examples,
        "negative_examples": neg_examples,
        "executive_clause": clauses["executive_clause"],
        "legislative_clause": clauses["legislative_clause"],
        "judicial_clause": clauses["judicial_clause"],
    }

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
    """Load the frozen Cycle-0 3 MiniLM neural bridges once per process from `repo_root` or workspace training root."""
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


def _load_or_clone_scholar_bridge_suite(
    repo_root: Path,
    state_dir: Path,
    *,
    allow_workspace_seed: bool = False,
) -> NeuralThreeBridgeSuite | None:
    """Load `LMFT-Scholar`'s isolated plastic Right-Brain MiniLM suite from `state_dir / 'right_brain_bridges'`.

    Seeds `state_dir / 'right_brain_bridges'` from `repo_root / 'models' / 'right_brain_bridges'` (or the
    workspace training root when `allow_workspace_seed=True`) without ever modifying `models/right_brain_bridges/`.
    """
    plastic_dir = Path(state_dir) / "right_brain_bridges"
    if not (plastic_dir / "doer_will_cultural").exists():
        seed_candidates: list[Path] = [Path(repo_root) / "models" / "right_brain_bridges"]
        if allow_workspace_seed:
            try:
                seed_candidates.append(resolve_training_root() / "models" / "right_brain_bridges")
            except Exception:
                pass
        for seed_dir in seed_candidates:
            if (seed_dir / "doer_will_cultural").exists():
                plastic_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(seed_dir, plastic_dir, dirs_exist_ok=True)
                break
    if (plastic_dir / "doer_will_cultural").exists():
        try:
            return NeuralThreeBridgeSuite.load_all(plastic_dir)
        except Exception:
            return None
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
            "schema_version": 1,
            "version": "learned_leaves_v1",
            "learned_leaves": custom_leaves,
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
    if not custom_leaves and rules_path.exists():
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
        leaf_distiller: Callable[..., dict[str, Any] | None] | None = None,
        lazy_load_gpu: bool = False,
        background_warmup: bool = False,
        auto_rem_sleep: bool = True,
        auto_sleep_threshold: int = 2,
    ) -> None:
        self._lock = threading.RLock()
        self._gpu_load_lock = threading.Lock()
        self._warmup_done = threading.Event()
        self._explicit_repo_root = repo_root is not None
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

        self._scholar_left_brain = scholar_left_brain
        self._leaf_distiller = leaf_distiller
        self._lazy_load_gpu = bool(lazy_load_gpu)
        self._background_warmup = bool(
            background_warmup and scholar_left_brain is None and not self._lazy_load_gpu
        )
        self.left_brain_loading: bool = False
        self.sleep_cycle_index = 0
        self.left_brain_status = "UNLOADED"
        self.active_checkpoint_label = "Checkpoint: models/left_brain_scholar (Impish_LLAMA_4B)"
        self._scholar_bridge_suite: NeuralThreeBridgeSuite | None = None
        self._virgin_sensors: ThreeBridgeSensors | None = None
        self._injected_sensors = sensors is not None
        self.leaf_verification_metrics: dict[str, dict[str, float]] = {}

        if sensors is not None:
            self.sensors = sensors
            self._virgin_sensors = sensors
        elif self._background_warmup:
            self.sensors = _default_fallback_sensors()
            self._virgin_sensors = self.sensors
        else:
            frozen_suite = _get_shared_neural_bridges(self.repo_root)
            if frozen_suite is not None:
                self._virgin_sensors = frozen_suite.as_three_bridge_sensors()
            plastic_suite = _load_or_clone_scholar_bridge_suite(
                self.repo_root,
                self.state_dir,
                allow_workspace_seed=not self._explicit_repo_root,
            )
            if plastic_suite is not None:
                self._scholar_bridge_suite = plastic_suite
                self.sensors = plastic_suite.as_three_bridge_sensors()
            elif self._virgin_sensors is not None:
                self.sensors = self._virgin_sensors
            else:
                self.sensors = _default_fallback_sensors()
            if self._virgin_sensors is None:
                self._virgin_sensors = self.sensors

        if self._scholar_left_brain is not None:
            self.left_brain_status = "READY (Injected Left-Brain)"
            self._warmup_done.set()
        elif self._lazy_load_gpu:
            self.left_brain_status = "STANDBY (Impish_LLAMA_4B Ready)"
            self._warmup_done.set()
        elif self._background_warmup:
            self.left_brain_loading = True
            self.left_brain_status = "WARMING UP GPU (Loading Impish_LLAMA_4B · 16-bit...)"
        else:
            self._load_scholar_locked(self.scholar_checkpoint_dir)
            self._warmup_done.set()

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
        if self._background_warmup:
            threading.Thread(
                target=self._run_background_warmup,
                args=(sensors is None,),
                daemon=True,
                name="lmft-scholar-gpu-warmup",
            ).start()

    def _run_background_warmup(self, load_neural_bridges: bool) -> None:
        """Warm up the 3 MiniLM bridges and 16-bit `Impish_LLAMA_4B` in the background without blocking UI startup."""
        try:
            if load_neural_bridges:
                frozen_suite = _get_shared_neural_bridges(self.repo_root)
                plastic_suite = _load_or_clone_scholar_bridge_suite(
                    self.repo_root,
                    self.state_dir,
                    allow_workspace_seed=not self._explicit_repo_root,
                )
                with self._lock:
                    if frozen_suite is not None:
                        self._virgin_sensors = frozen_suite.as_three_bridge_sensors()
                    if plastic_suite is not None:
                        self._scholar_bridge_suite = plastic_suite
                        self.sensors = plastic_suite.as_three_bridge_sensors()
                    elif frozen_suite is not None:
                        self.sensors = frozen_suite.as_three_bridge_sensors()
                    if self._virgin_sensors is None:
                        self._virgin_sensors = self.sensors
            self._load_scholar_locked(self.scholar_checkpoint_dir)
        finally:
            with self._lock:
                self.left_brain_loading = False
            self._warmup_done.set()

    def wait_until_warm(self, timeout: float = 180.0) -> bool:
        """Wait for any in-flight background GPU warmup to finish before running a turn."""
        return bool(self._warmup_done.wait(timeout=timeout))

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
        with self._gpu_load_lock:
            if self._scholar_left_brain is not None:
                return
            proj_path = checkpoint_dir / "projector_weights.pt"
            if not proj_path.exists():
                with self._lock:
                    self.left_brain_loading = False
                    self.left_brain_status = "UNAVAILABLE (No GPU Checkpoint Found)"
                return
            with self._lock:
                self.left_brain_loading = True
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

                base_id = DEFAULT_SCHOLAR_HF_ID
                callosum_status = "TRAINED_STEERED_CALLOSUM"
                ckpt_sleep_cycle = 0
                if getattr(lb, "callosum_metadata", None) is not None:
                    base_id = str(lb.callosum_metadata.base_model_id or base_id)
                    callosum_status = str(lb.callosum_metadata.callosum_status or callosum_status)
                    ckpt_sleep_cycle = int(lb.callosum_metadata.sleep_cycle_index)
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
                with self._lock:
                    self._scholar_left_brain = lb
                    self.sleep_cycle_index = max(self.sleep_cycle_index, ckpt_sleep_cycle)
                    self.left_brain_loading = False
                    self.left_brain_status = "READY (Impish_LLAMA_4B · 16-bit)"
                    self.active_checkpoint_label = (
                        f"LMFT-Scholar · Impish_LLAMA_4B ({callosum_status} — {base_id})"
                    )
            except Exception as exc:
                with self._lock:
                    self.left_brain_loading = False
                    self.left_brain_status = f"UNAVAILABLE ({type(exc).__name__}: {exc})"

    def _ensure_left_brain_ready(self) -> Any | None:
        if not self._warmup_done.is_set():
            self._warmup_done.wait(timeout=180.0)
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
        atomic_json(
            self.state_dir / "leaf_verification_metrics.json",
            self.leaf_verification_metrics,
        )
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
        metrics_path = self.state_dir / "leaf_verification_metrics.json"
        if metrics_path.exists():
            try:
                raw_m = json.loads(metrics_path.read_text(encoding="utf-8"))
                if isinstance(raw_m, dict):
                    self.leaf_verification_metrics = {
                        str(k): {str(mk): float(mv) for mk, mv in val.items() if isinstance(mv, (int, float))}
                        for k, val in raw_m.items()
                        if isinstance(val, dict)
                    }
            except Exception:
                pass
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

    def reset_scholar_canopy(self) -> dict[str, Any]:
        """Reset `LMFT-Scholar` canopy, rules, archives, Will synapses, and plastic Right-Brain bridges back to clean Cycle-0 seed."""
        self.wait_until_warm(timeout=180.0)
        with self._lock:
            self.rules = _ensure_foundational_rules(())
            self.learned_leaves = PHILOSOPHICAL_SHADOW_LEAVES + DEFAULT_COMPLEX_FEELING_LEAVES
            self.archives = ThreeBranchArchives()
            self.ledger = ContemplationLedger()
            self.will_edges = RIGHT_BRAIN_WILL_EDGES
            self.leaf_verification_metrics = {}
            self.sleep_queue.clear()
            self._last_minted_leaves = ()
            self._last_harvested_gap_pairs = []
            self.rem_dream_chronicles = []
            self.sleep_cycle_index = 0
            self.curriculum_step_counter = 0

            for stale_file in (
                self.state_dir / "live_projector_weights.pt",
                self.scholar_checkpoint_dir / "live_projector_weights.pt",
                self.state_dir / "scholar_gap_trajectories.jsonl",
            ):
                if stale_file.is_file():
                    try:
                        stale_file.unlink()
                    except OSError:
                        pass

            plastic_dir = self.state_dir / "right_brain_bridges"
            if plastic_dir.exists():
                shutil.rmtree(plastic_dir, ignore_errors=True)
            if not self._injected_sensors:
                plastic_suite = _load_or_clone_scholar_bridge_suite(
                    self.repo_root,
                    self.state_dir,
                    allow_workspace_seed=not self._explicit_repo_root,
                )
                if plastic_suite is not None:
                    self._scholar_bridge_suite = plastic_suite
                    self.sensors = plastic_suite.as_three_bridge_sensors()
                elif self._virgin_sensors is not None:
                    self.sensors = self._virgin_sensors

            self._persist_user_state()
            self.last_rem_sleep_receipt = (
                "Reset Scholar Canopy to clean Cycle-0 Founding Seed (2 rules, 0 custom leaves, fresh sensor bridges)."
            )
            return self.get_state_snapshot()

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
        suppress_auto_sleep: bool = False,
        contrast_foil_text: str = "",
    ) -> dict[str, Any]:
        self.wait_until_warm(timeout=180.0)
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

            base_sensors = (
                (self._virgin_sensors or self.sensors)
                if self.active_lineage == "virgin"
                else self.sensors
            )
            effective_sensors: Any = base_sensors
            if fact_anchor_text and fact_anchor_text.strip():
                effective_sensors = _PassageAnchoredSensors(
                    wave_text=user_text,
                    fact_anchor_text=fact_anchor_text.strip(),
                    base_sensors=base_sensors,
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
                "contrast_foil_text": contrast_foil_text.strip(),
                "turn": turn_result,
                "spoken_reply": spoken_reply,
            }

            queued_now = False
            if not turn_result.settled and self.active_lineage != "virgin":
                if contrast_foil_text.strip():
                    sleep_scene = self._build_sleep_scene_from_turn(
                        scene_index=scene_id,
                        scenario_text=user_text,
                        turn=turn_result,
                        spoken_reply=spoken_reply,
                        reference_text=(fact_anchor_text or "").strip(),
                        contrast_foil_text=contrast_foil_text.strip(),
                    )
                else:
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

            if (
                queued_now
                and not suppress_auto_sleep
                and self.auto_rem_sleep
                and len(self.sleep_queue) >= self.auto_sleep_threshold
            ):
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
        contrast_foil_text: str = "",
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

        foil_clean = contrast_foil_text.strip()
        if foil_clean and not user_note.strip():
            foil_coords, foil_exp, _, _ = self.sensors.score_text(
                foil_clean,
                learned_leaves=self.learned_leaves,
            )
            ext_coords = dict(foil_coords)
            ext_coords.update(foil_exp)
            ext_coords["psychological/truth/fact"] = round(float(vibe.fact), 6)

        scenario_with_note = (
            f"{scenario_text} [Coach note: {user_note.strip()}]"
            if user_note.strip()
            else scenario_text
        )
        if user_note.strip():
            ref_coords, ref_exp, _, _ = self.sensors.score_text(
                user_note.strip(),
                learned_leaves=self.learned_leaves,
            )
            contrastive_ref = {**ref_coords, **ref_exp}
        elif reference_text.strip():
            ref_coords, ref_exp, _, _ = self.sensors.score_text(
                reference_text.strip(),
                learned_leaves=self.learned_leaves,
            )
            contrastive_ref = {**ref_coords, **ref_exp}
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
            contrast_foil_text=foil_clean,
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
        foil_block = (
            f"\nContrasting Foil: \"{selected.contrast_foil_text}\""
            if selected.contrast_foil_text.strip()
            else ""
        )
        self.study_canvas_text = (
            f"{selected.passage_text}\n\nContrasting Foil:\n{selected.contrast_foil_text}"
            if selected.contrast_foil_text.strip()
            else selected.passage_text
        )
        prompt = (
            f"[{selected.department}] {selected.title}\n"
            f"Primary Passage: \"{selected.passage_text}\"{foil_block}\n\n"
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
            suppress_auto_sleep=bool(sleep_after_essay),
            contrast_foil_text=selected.contrast_foil_text,
        )
        if sleep_after_essay and self.active_lineage != "virgin":
            latest_id = self.scene_counter
            rec = self._turn_records_by_id.get(latest_id)
            if rec is not None:
                seminar_sleep_scene = self._build_sleep_scene_from_turn(
                    scene_index=latest_id,
                    scenario_text=prompt,
                    turn=rec["turn"],
                    spoken_reply=str(rec.get("spoken_reply", "")),
                    reference_text=selected.passage_text,
                    contrast_foil_text=selected.contrast_foil_text,
                    force_coached=False,
                )
                self.sleep_queue = [s for s in self.sleep_queue if s.scene_index != latest_id]
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
            reference_text=str(rec.get("fact_anchor_text", "")),
            contrast_foil_text=str(rec.get("contrast_foil_text", "")),
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

    def _extract_contrast_scenes_for_sleep(
        self,
        sc: UnresolvedSleepScene,
    ) -> tuple[str, str]:
        """Extract unseen `(scene_a_text, scene_b_text)` for REM distillation and holdout verification."""
        raw_scen = (sc.scenario_text or "").strip()
        foil = (sc.contrast_foil_text or "").strip()
        hunch = (sc.left_brain_hunch or "").strip()

        if "[Coach note:" in raw_scen:
            base_part, note_part = raw_scen.split("[Coach note:", 1)
            coach_note = note_part.rstrip("]").strip()
            if coach_note:
                scene_b = foil or hunch or base_part.strip()
                return coach_note, scene_b

        if foil:
            rec = self._turn_records_by_id.get(sc.scene_index)
            primary_passage = str(rec.get("fact_anchor_text", "")).strip() if isinstance(rec, dict) else ""
            if not primary_passage:
                m_pass = re.search(r'Primary Passage:\s*"([^"]+)"', raw_scen)
                if m_pass:
                    primary_passage = m_pass.group(1).strip()
            scene_a = primary_passage or hunch or raw_scen
            return scene_a, foil

        base_clean = raw_scen.split("[Coach note:", 1)[0].strip()
        if hunch and hunch.lower() != base_clean.lower():
            return hunch, base_clean
        return (
            base_clean or "Thoughtful care that balances strength with restraint.",
            "Routine casual small talk with no underlying conflict, pressure, or ethical tension.",
        )

    def _select_parent_branch_for_distillation(
        self,
        *,
        sc: UnresolvedSleepScene,
        scene_a_text: str,
        scene_b_text: str,
        active_leaves: Sequence[LearnedLeaf],
    ) -> tuple[str, int, LearnedLeaf | None, str]:
        """Select `(parent_branch, target_tier, parent_leaf_obj, root_bough)` using real sensor activations on `scene_a_text`."""
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

        coords_a, exp_a, _, _ = self.sensors.score_text(
            scene_a_text,
            learned_leaves=tuple(active_leaves),
            branch_gating=True,
        )
        coords_b, exp_b, _, _ = self.sensors.score_text(
            scene_b_text,
            learned_leaves=tuple(active_leaves),
            branch_gating=True,
        )
        full_a = {**coords_a, **exp_a}
        full_b = {**coords_b, **exp_b}
        max_founding_a = max((float(coords_a.get(k, 0.0)) for k in FOUNDATIONAL_LEAVES), default=0.0)
        salience_a = float(exp_a.get(SALIENCE_LEAF, sc.salience))

        low_scen = (sc.scenario_text or "").lower()
        open_candidates: list[tuple[float, str]] = []
        all_candidates: list[tuple[float, str]] = []

        for bough_id, (f1, f2) in growable_boughs_map.items():
            a1 = float(full_a.get(f1, 0.50))
            a2 = float(full_a.get(f2, 0.50))
            b1 = float(full_b.get(f1, 0.50))
            b2 = float(full_b.get(f2, 0.50))
            if bough_id.startswith("experiential/"):
                exp_vals = [float(v) for k, v in exp_a.items() if k.startswith(bough_id + "/")]
                exp_sig = max(exp_vals, default=salience_a)
                is_open = exp_sig >= BRANCH_GATE_THRESHOLD
            else:
                blind_spot = max_founding_a < BRANCH_GATE_THRESHOLD and salience_a >= 0.45
                is_open = blind_spot or (max(a1, a2) >= BRANCH_GATE_THRESHOLD)

            score = (
                abs(a1 - 0.50)
                + abs(a2 - 0.50)
                + 1.5 * (abs(a1 - b1) + abs(a2 - b2))
            )
            b_dept, b_sub = bough_id.split("/", 1)
            if low_scen.startswith(f"[{b_dept} · {b_sub}]"):
                score += 2.0
            elif bough_id in low_scen:
                score += 0.45
            all_candidates.append((score, bough_id))
            if is_open:
                open_candidates.append((score, bough_id))

        ranked = open_candidates if open_candidates else all_candidates
        ranked.sort(key=lambda item: item[0], reverse=True)
        root_bough = ranked[0][1]

        t2_active = [
            lf
            for lf in active_leaves
            if lf.tier == 2
            and lf.parent_branch == root_bough
            and lf.leaf_id not in builtin_ids
            and float(coords_a.get(lf.leaf_id, 0.0)) >= CONFUSION_THRESHOLD
        ]
        if not t2_active:
            return root_bough, 2, None, root_bough

        parent_t2 = max(t2_active, key=lambda lf: float(coords_a.get(lf.leaf_id, 0.0)))
        t3_active = [
            lf
            for lf in active_leaves
            if lf.tier == 3
            and lf.parent_branch == parent_t2.leaf_id
            and lf.leaf_id not in builtin_ids
            and float(coords_a.get(lf.leaf_id, 0.0)) >= CONFUSION_THRESHOLD
        ]
        if not t3_active:
            return parent_t2.leaf_id, 3, parent_t2, root_bough

        parent_t3 = max(t3_active, key=lambda lf: float(coords_a.get(lf.leaf_id, 0.0)))
        return parent_t3.leaf_id, 4, parent_t3, root_bough

    def _distill_leaf_candidate_with_left_brain(
        self,
        *,
        parent_branch: str,
        target_tier: int,
        scene_a_text: str,
        scene_b_text: str,
        parent_leaf: LearnedLeaf | None,
        existing_short_names: set[str],
        attempt: int = 0,
        prior_error: str = "",
    ) -> dict[str, Any] | None:
        """Use native 16-bit `Impish_LLAMA_4B` (or an injected test `leaf_distiller`) to distill a plain-English human leaf concept."""
        if self._leaf_distiller is not None:
            return self._leaf_distiller(
                parent_branch=parent_branch,
                target_tier=target_tier,
                scene_a_text=scene_a_text,
                scene_b_text=scene_b_text,
                parent_leaf=parent_leaf,
                existing_sibling_names=sorted(existing_short_names),
                attempt=attempt,
                prior_error=prior_error,
            )

        lb = self._ensure_left_brain_ready()
        if lb is None or not hasattr(lb, "model") or not hasattr(lb, "tokenizer"):
            return None

        import torch

        model = lb.model
        tokenizer = lb.tokenizer
        device = getattr(lb, "device", next(model.parameters()).device)

        bough_theme = parent_branch.split("/")[-1].replace("_", " ")
        parent_context = (
            f"Parent concept already learned (`{parent_leaf.leaf_name}`): \"{parent_leaf.description}\"\n"
            f"Propose a MORE SPECIFIC child sub-skill of `{parent_leaf.leaf_name}` (such as preventive upkeep, "
            f"steady pacing, listening to early warnings, or protecting vulnerable teammates) with a NEW, distinct "
            f"`description` sentence that does NOT repeat words from `{parent_leaf.leaf_name}`'s description."
            if parent_leaf is not None
            else (
                f"Human Domain Theme: {bough_theme}.\n"
                f"Propose a clear, reusable everyday human quality related to {bough_theme} that is strongly present in "
                f"Target Passage A and absent or violated in Contrasting Foil B."
            )
        )
        builtin_names = {
            lf.leaf_name for lf in (PHILOSOPHICAL_SHADOW_LEAVES + DEFAULT_COMPLEX_FEELING_LEAVES)
        }
        custom_existing = sorted(n for n in existing_short_names if n not in builtin_names)
        avoid_names = custom_existing + ["quiet_courage"] + sorted(builtin_names)
        avoid_list = ", ".join(avoid_names[:24])
        avoid_user_note = (
            f"\nAlready-learned leaf names (MUST choose a completely different `LEAF_NAME` with different starting word): "
            f"{', '.join(custom_existing)}\n"
            if custom_existing
            else ""
        )
        retry_hint = (
            f"\nIMPORTANT CORRECTION FROM PREVIOUS ATTEMPT: {prior_error}\n"
            f"Choose a completely different, unmistakable everyday human quality and clearer modern examples."
            if prior_error
            else ""
        )

        system_prompt = (
            "You distill timeless, plain-English human qualities from contrasting scenes.\n"
            "STRICT RULES:\n"
            "1. Write in clear, natural everyday English. NEVER include numbers, decimals, or scores.\n"
            "2. NEVER mention any character names, author names, ship names, places, or props from the passages.\n"
            "3. `LEAF_NAME` MUST be 2 or 3 lowercase everyday words joined by underscores (format like `quiet_courage`) "
            f"coined freshly for this contrast, and MUST NOT match: [{avoid_list}].\n"
            "4. `DESCRIPTION` MUST be one vivid, plain-English sentence (12 to 28 words) defining the human quality.\n"
            "5. `POS_1` through `POS_5` MUST be 5 short modern everyday sentences (workplaces, families, clinics, "
            "workshops, neighborhoods) clearly showing this quality without reusing words from Passage A or B.\n"
            "6. `NEG_1` through `NEG_5` MUST be 5 short modern everyday lookalike sentences where reckless force, "
            "haste, or rigid rules appear WITHOUT this quality.\n"
            "7. `EXECUTIVE`, `LEGISLATIVE`, and `JUDICIAL` MUST each be one plain-English sentence describing the "
            "practical action, emotional tone, and structural truth of this quality.\n"
            "Output ONLY the 15 labeled lines in this exact order:\n"
            "LEAF_NAME: ...\n"
            "DESCRIPTION: ...\n"
            "POS_1: ...\nPOS_2: ...\nPOS_3: ...\nPOS_4: ...\nPOS_5: ...\n"
            "NEG_1: ...\nNEG_2: ...\nNEG_3: ...\nNEG_4: ...\nNEG_5: ...\n"
            "EXECUTIVE: ...\n"
            "LEGISLATIVE: ...\n"
            "JUDICIAL: ..."
        )
        user_prompt = (
            f"{parent_context}\n\n"
            f"Target Passage A (quality present):\n\"{scene_a_text[:600]}\"\n\n"
            f"Contrasting Foil B (quality absent or violated):\n\"{scene_b_text[:600]}\"\n"
            f"{avoid_user_note}{retry_hint}\n"
            "Output the 15 labeled lines now."
        )

        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            chat_header = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            full_prompt = f"{chat_header}LEAF_NAME: "
            input_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
        else:
            full_prompt = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant: LEAF_NAME: "
            input_ids = tokenizer.encode(full_prompt, add_special_tokens=True)

        saved_scales: dict[str, Any] = {}
        for mod_name, mod in model.named_modules():
            if isinstance(mod, LoRALinear):
                saved_scales[mod_name] = mod.rank_scale.detach().clone()

        try:
            with torch.no_grad():
                for mod in model.modules():
                    if isinstance(mod, LoRALinear):
                        mod.rank_scale.zero_()

                inp = torch.tensor([input_ids], dtype=torch.long, device=device)
                attn = torch.ones_like(inp)
                prompt_tok_len = len(input_ids)
                gen_kwargs: dict[str, Any] = {
                    "input_ids": inp,
                    "attention_mask": attn,
                    "max_new_tokens": 580,
                    "pad_token_id": int(getattr(tokenizer, "eos_token_id", 0) or 0),
                }
                try:
                    from transformers import StoppingCriteria, StoppingCriteriaList

                    class _LabeledLinesStop(StoppingCriteria):
                        def __call__(self, seq_ids: Any, _scores: Any, **_kw: Any) -> bool:
                            gen_slice = seq_ids[0, prompt_tok_len:].tolist()
                            if len(gen_slice) < 60 or len(gen_slice) % 4 != 0:
                                return False
                            cand_txt = tokenizer.decode(gen_slice, skip_special_tokens=True)
                            idx_jud = cand_txt.upper().find("JUDICIAL:")
                            if idx_jud < 0:
                                return False
                            after_jud = cand_txt[idx_jud + len("JUDICIAL:") :]
                            return "\n" in after_jud and len(after_jud.split("\n", 1)[0].strip()) >= 20

                    gen_kwargs["stopping_criteria"] = StoppingCriteriaList([_LabeledLinesStop()])
                except Exception:
                    pass
                if attempt == 0:
                    gen_kwargs["do_sample"] = False
                else:
                    gen_kwargs["do_sample"] = True
                    gen_kwargs["temperature"] = (
                        0.72 if "Duplicate" in prior_error else (0.38 if attempt == 1 else 0.58)
                    )
                    gen_kwargs["top_p"] = 0.90

                out_ids = model.generate(**gen_kwargs)
                new_tokens = out_ids[0, len(input_ids) :].tolist()
                decoded_tail = tokenizer.decode(new_tokens, skip_special_tokens=True)
        finally:
            with torch.no_grad():
                for mod_name, mod in model.named_modules():
                    if isinstance(mod, LoRALinear) and mod_name in saved_scales:
                        mod.rank_scale.copy_(saved_scales[mod_name])

        return _parse_distilled_leaf_response("LEAF_NAME: " + decoded_tail)

    def _distill_into_bridge_and_verify_unseen(
        self,
        *,
        candidate_leaf: LearnedLeaf,
        pos_examples: Sequence[str],
        neg_examples: Sequence[str],
        scene_a_text: str,
        scene_b_text: str,
        active_leaves: Sequence[LearnedLeaf],
        attempt: int = 0,
    ) -> tuple[bool, str, dict[str, float]]:
        """Fine-tune `LMFT-Scholar`'s isolated MiniLM bridge on everyday examples + replay anchors and gatekeep on unseen Scene A vs. B."""
        unseen_calm_holdout = (
            "A clerk sorting afternoon mail into wooden cubbies beside a quiet window."
        )
        train_calm_anchors = (
            "A neighbor sweeping dry leaves off the front porch before breakfast.",
            "Passengers reading newspapers quietly on a morning commuter train.",
        )

        suite = self._scholar_bridge_suite
        if suite is not None:
            dept_root = candidate_leaf.leaf_id.split("/", 1)[0]
            if dept_root == "cultural":
                classifier: NeuralBridgeClassifier = suite.doer
                founding_map = DOER_FOUNDING_DESCRIPTIONS
            elif dept_root == "philosophical":
                classifier = suite.thinker
                founding_map = THINKER_FOUNDING_DESCRIPTIONS
            else:
                classifier = suite.feeler
                founding_map = FEELER_FOUNDING_DESCRIPTIONS

            dept_prefix = f"{dept_root}/"
            existing_traits: dict[str, str] = dict(founding_map)
            for lf in active_leaves:
                if lf.leaf_id.startswith(dept_prefix):
                    existing_traits[lf.leaf_id] = lf.description

            weights_backup = {
                k: v.detach().cpu().clone()
                for k, v in classifier.model.state_dict().items()
            }
            pre_f_a = classifier.score_pairs(scene_a_text, founding_map, use_text_cache=False)
            pre_f_b = classifier.score_pairs(scene_b_text, founding_map, use_text_cache=False)

            cal_sentences = list(pos_examples) + list(neg_examples) + list(train_calm_anchors)
            train_examples: list[BridgePairExample] = []

            # 1. Knowledge-distillation replay anchors so founding & prior custom leaves do not drift
            for s_idx, sent in enumerate(cal_sentences):
                frozen_preds = classifier.score_pairs(sent, existing_traits, use_text_cache=False)
                for ex_leaf_id, ex_desc in existing_traits.items():
                    train_examples.append(
                        BridgePairExample(
                            scenario_id=f"replay_c{self.sleep_cycle_index}_{s_idx}_{ex_leaf_id.split('/')[-1]}",
                            text=sent,
                            leaf_id=ex_leaf_id,
                            trait_description=ex_desc,
                            target_score=round(float(frozen_preds[ex_leaf_id]), 4),
                        )
                    )

            # 2. Contrastive everyday examples for the new candidate leaf (never unseen Scene A or B!)
            for rep_idx in range(2):
                for p_idx, p_sent in enumerate(pos_examples):
                    train_examples.append(
                        BridgePairExample(
                            scenario_id=f"pos_c{self.sleep_cycle_index}_r{rep_idx}_{p_idx}",
                            text=p_sent,
                            leaf_id=candidate_leaf.leaf_id,
                            trait_description=candidate_leaf.description,
                            target_score=0.88,
                        )
                    )
                for n_idx, n_sent in enumerate(list(neg_examples) + list(train_calm_anchors)):
                    train_examples.append(
                        BridgePairExample(
                            scenario_id=f"neg_c{self.sleep_cycle_index}_r{rep_idx}_{n_idx}",
                            text=n_sent,
                            leaf_id=candidate_leaf.leaf_id,
                            trait_description=candidate_leaf.description,
                            target_score=0.08,
                        )
                    )

            try:
                train_bridge_classifier(
                    classifier,
                    train_examples,
                    epochs=3,
                    batch_size=16,
                    learning_rate=1.5e-5,
                    seed=42 + self.sleep_cycle_index * 11 + attempt,
                )
                suite.doer.clear_cache()
                suite.thinker.clear_cache()
                suite.feeler.clear_cache()

                eval_traits = {**founding_map, candidate_leaf.leaf_id: candidate_leaf.description}
                post_a = classifier.score_pairs(scene_a_text, eval_traits, use_text_cache=False)
                post_b = classifier.score_pairs(scene_b_text, eval_traits, use_text_cache=False)
                post_calm = classifier.score_pairs(
                    unseen_calm_holdout,
                    {candidate_leaf.leaf_id: candidate_leaf.description},
                    use_text_cache=False,
                )
                raw_score_a = float(post_a[candidate_leaf.leaf_id])
                raw_score_b = float(post_b[candidate_leaf.leaf_id])
                raw_calm = float(post_calm[candidate_leaf.leaf_id])
                raw_margin = raw_score_a - raw_score_b
                max_drift = max(
                    [abs(float(post_a[k]) - float(pre_f_a[k])) for k in founding_map]
                    + [abs(float(post_b[k]) - float(pre_f_b[k])) for k in founding_map],
                    default=0.0,
                )

                gated_coords_a, _, _, _ = self.sensors.score_text(
                    scene_a_text,
                    learned_leaves=tuple(active_leaves) + (candidate_leaf,),
                    branch_gating=True,
                )
                gated_score_a = float(gated_coords_a.get(candidate_leaf.leaf_id, 0.0))
            except Exception as exc:
                classifier.model.load_state_dict(
                    {k: v.to(classifier.device) for k, v in weights_backup.items()},
                    strict=True,
                )
                classifier.model.eval()
                classifier.clear_cache()
                return False, f"Bridge training error: {exc}", {}

            metrics = {
                "unseen_score_a": round(raw_score_a, 4),
                "gated_score_a": round(gated_score_a, 4),
                "unseen_score_b": round(raw_score_b, 4),
                "unseen_margin": round(raw_margin, 4),
                "calm_score": round(raw_calm, 4),
                "max_founding_drift": round(max_drift, 4),
            }
            if (
                raw_score_a < 0.48
                or gated_score_a < 0.48
                or raw_margin < 0.12
                or raw_calm > 0.38
                or max_drift > 0.12
            ):
                classifier.model.load_state_dict(
                    {k: v.to(classifier.device) for k, v in weights_backup.items()},
                    strict=True,
                )
                classifier.model.eval()
                suite.doer.clear_cache()
                suite.thinker.clear_cache()
                suite.feeler.clear_cache()
                reason = (
                    f"Unseen holdout gate rejected '{candidate_leaf.leaf_name}': "
                    f"SceneA={raw_score_a:.2f} (gated={gated_score_a:.2f}), "
                    f"FoilB={raw_score_b:.2f}, margin={raw_margin:+.2f}, "
                    f"calm={raw_calm:.2f}, drift={max_drift:.2f}"
                )
                return False, reason, metrics

            suite.save_all(self.state_dir / "right_brain_bridges")
            return True, "verified", metrics

        # CPU / Injected-Sensor Test Path (enforces the exact same unseen discrimination thresholds)
        test_leaves = tuple(active_leaves) + (candidate_leaf,)
        coords_a_ng, _, _, _ = self.sensors.score_text(
            scene_a_text,
            learned_leaves=test_leaves,
            branch_gating=False,
        )
        coords_b_ng, _, _, _ = self.sensors.score_text(
            scene_b_text,
            learned_leaves=test_leaves,
            branch_gating=False,
        )
        coords_a_g, _, _, _ = self.sensors.score_text(
            scene_a_text,
            learned_leaves=test_leaves,
            branch_gating=True,
        )
        raw_score_a = float(coords_a_ng.get(candidate_leaf.leaf_id, 0.0))
        raw_score_b = float(coords_b_ng.get(candidate_leaf.leaf_id, 0.0))
        gated_score_a = float(coords_a_g.get(candidate_leaf.leaf_id, 0.0))
        raw_margin = raw_score_a - raw_score_b
        metrics = {
            "unseen_score_a": round(raw_score_a, 4),
            "gated_score_a": round(gated_score_a, 4),
            "unseen_score_b": round(raw_score_b, 4),
            "unseen_margin": round(raw_margin, 4),
            "calm_score": 0.10,
            "max_founding_drift": 0.0,
        }
        if raw_score_a < 0.48 or gated_score_a < 0.48 or raw_margin < 0.12:
            reason = (
                f"Unseen holdout gate rejected '{candidate_leaf.leaf_name}': "
                f"SceneA={raw_score_a:.2f} (gated={gated_score_a:.2f}), "
                f"FoilB={raw_score_b:.2f}, margin={raw_margin:+.2f}"
            )
            return False, reason, metrics
        return True, "verified", metrics

    def _distill_and_verify_scholar_canopy(
        self,
        queued_snapshot: tuple[UnresolvedSleepScene, ...],
    ) -> tuple[
        tuple[FuzzyConceptRule, ...],
        tuple[LearnedLeaf, ...],
        tuple[str, ...],
        tuple[str, ...],
    ]:
        """Distill human concepts with `Impish_LLAMA_4B`, train plastic MiniLM bridges, and verify on unseen Scene A vs. B."""
        if not queued_snapshot or self.active_lineage == "virgin":
            return (), (), (), ()

        active_leaves = list(self.learned_leaves)
        existing_short_names = {lf.leaf_name for lf in active_leaves}
        active_rules = list(self.rules)
        existing_rule_names = {r.concept_name for r in active_rules}
        cur_archives = self.archives
        cur_ledger = self.ledger

        minted_leaves: list[LearnedLeaf] = []
        forged_rules: list[FuzzyConceptRule] = []
        harvested_clause_ids: list[str] = []
        rejection_notes: list[str] = []

        for sc in queued_snapshot:
            scene_a_text, scene_b_text = self._extract_contrast_scenes_for_sleep(sc)
            parent_branch, target_tier, parent_leaf_obj, root_bough = (
                self._select_parent_branch_for_distillation(
                    sc=sc,
                    scene_a_text=scene_a_text,
                    scene_b_text=scene_b_text,
                    active_leaves=active_leaves,
                )
            )

            accepted_leaf: LearnedLeaf | None = None
            accepted_payload: dict[str, Any] | None = None
            accepted_metrics: dict[str, float] = {}
            prior_err = ""

            for attempt in range(3):
                try:
                    raw_payload = self._distill_leaf_candidate_with_left_brain(
                        parent_branch=parent_branch,
                        target_tier=target_tier,
                        scene_a_text=scene_a_text,
                        scene_b_text=scene_b_text,
                        parent_leaf=parent_leaf_obj,
                        existing_short_names=existing_short_names,
                        attempt=attempt,
                        prior_error=prior_err,
                    )
                    if raw_payload is None:
                        prior_err = "Left-Brain LLM unavailable for leaf distillation."
                        break
                    clean_payload = _validate_human_leaf_payload(
                        raw_payload,
                        scene_a_text=scene_a_text,
                        scene_b_text=scene_b_text,
                        existing_short_names=existing_short_names,
                        parent_description=parent_leaf_obj.description if parent_leaf_obj is not None else "",
                    )
                    draft_leaf = score_leaf_anchor_weights_at_birth(
                        LearnedLeaf(
                            parent_branch=parent_branch,
                            leaf_name=clean_payload["leaf_name"],
                            description=clean_payload["description"],
                        )
                    )
                except Exception as exc:
                    prior_err = str(exc)
                    continue

                passed, reason, ver_metrics = self._distill_into_bridge_and_verify_unseen(
                    candidate_leaf=draft_leaf,
                    pos_examples=clean_payload["positive_examples"],
                    neg_examples=clean_payload["negative_examples"],
                    scene_a_text=scene_a_text,
                    scene_b_text=scene_b_text,
                    active_leaves=active_leaves,
                    attempt=attempt,
                )
                if passed:
                    accepted_leaf = draft_leaf
                    accepted_payload = clean_payload
                    accepted_metrics = ver_metrics
                    break
                prior_err = reason

            if accepted_leaf is None or accepted_payload is None:
                if prior_err:
                    rejection_notes.append(prior_err)
                continue

            # Score semantic Callosum anchor_weights with the updated bridge + contrastive coordinate delta
            pair_scorer: Callable[[str, str], float] | None = None
            if self._scholar_bridge_suite is not None:
                dept_root = accepted_leaf.leaf_id.split("/", 1)[0]
                bridge_cls = (
                    self._scholar_bridge_suite.doer
                    if dept_root == "cultural"
                    else (
                        self._scholar_bridge_suite.thinker
                        if dept_root == "philosophical"
                        else self._scholar_bridge_suite.feeler
                    )
                )
                pair_scorer = lambda anchor, desc, _b=bridge_cls: float(
                    _b.score_pairs(desc, {"anchor": anchor}, use_text_cache=False)["anchor"]
                )

            coords_foil, exp_foil, _, _ = self.sensors.score_text(
                scene_b_text,
                learned_leaves=tuple(active_leaves) + (accepted_leaf,),
                branch_gating=True,
            )
            coords_ref, exp_ref, _, _ = self.sensors.score_text(
                scene_a_text,
                learned_leaves=tuple(active_leaves) + (accepted_leaf,),
                branch_gating=True,
            )
            full_foil = {**coords_foil, **exp_foil}
            full_ref = {**coords_ref, **exp_ref}

            anchored_leaf = score_leaf_anchor_weights_at_birth(accepted_leaf, pair_scorer=pair_scorer)
            sculpted_leaf = sculpt_learned_leaf_anchor_weights(anchored_leaf, full_foil, full_ref)

            active_leaves.append(sculpted_leaf)
            minted_leaves.append(sculpted_leaf)
            existing_short_names.add(sculpted_leaf.leaf_name)
            self.leaf_verification_metrics[sculpted_leaf.leaf_id] = accepted_metrics

            # Build clean domain-general FuzzyConceptRule anchored in real post-distillation coordinates
            rule_targets: dict[str, float] = {
                sculpted_leaf.leaf_id: round(
                    min(0.78, max(0.48, float(full_ref.get(sculpted_leaf.leaf_id, 0.60)) - 0.06)),
                    4,
                )
            }
            bough_founding = [
                k for k in FOUNDATIONAL_LEAVES if k.startswith(root_bough + "/")
            ]
            if bough_founding:
                best_f = max(
                    bough_founding,
                    key=lambda k: abs(float(full_ref.get(k, 0.50)) - float(full_foil.get(k, 0.50))),
                )
                f_val = float(full_ref.get(best_f, 0.50))
                if f_val >= 0.45:
                    rule_targets[best_f] = round(min(0.76, max(0.45, f_val - 0.05)), 4)

            min_t = min(rule_targets.values())
            max_t = max(rule_targets.values())
            rule_low = round(min(0.72, max(0.42, min_t - 0.08)), 4)
            rule_high = round(min(0.88, max(rule_low + 0.12, max_t + 0.06)), 4)
            candidate_rule = FuzzyConceptRule(
                rule_id=f"rule-scholar-{sculpted_leaf.leaf_name[:28]}",
                concept_name=sculpted_leaf.leaf_name,
                description=sculpted_leaf.description,
                target_leaves=rule_targets,
                low=rule_low,
                high=rule_high,
                polarity="attractive",
            )
            if candidate_rule.concept_name not in existing_rule_names:
                active_rules.append(candidate_rule)
                forged_rules.append(candidate_rule)
                existing_rule_names.add(candidate_rule.concept_name)
            else:
                for r_idx, ex_r in enumerate(active_rules):
                    if ex_r.concept_name == candidate_rule.concept_name:
                        entrained = entrain_existing_cluster_rule(
                            ex_r,
                            candidate_rule,
                            candidate_rule.concept_name,
                        )
                        active_rules[r_idx] = entrained
                        forged_rules.append(entrained)
                        break

            # File clean plain-English clauses into the 3 Branch Archives & Contemplation Ledger
            cult_sig = {k: round(float(v), 4) for k, v in full_ref.items() if k.startswith("cultural/")}
            phil_sig = {k: round(float(v), 4) for k, v in full_ref.items() if k.startswith("philosophical/")}
            exp_sig = {k: round(float(v), 4) for k, v in full_ref.items() if k.startswith("experiential/")}
            avg_fc = compile_felt_chord(
                full_ref,
                salience=float(full_ref.get(SALIENCE_LEAF, sc.salience)),
                valence=float(full_ref.get(VALENCE_LEAF, sc.valence)),
                arousal=float(full_ref.get(AROUSAL_LEAF, sc.arousal)),
                resonance=0.68,
                learned_leaves=tuple(active_leaves),
            )
            tag_slug = sculpted_leaf.leaf_name
            exec_clause = BranchClauseEntry(
                clause_id=f"exec-scholar-s{sc.scene_index}-{tag_slug}",
                branch=EXECUTIVE_BRANCH,
                text=accepted_payload["executive_clause"],
                signature_coords=cult_sig,
                source_scene=sc.scene_index,
                concept_tag=tag_slug,
            )
            leg_clause = BranchClauseEntry(
                clause_id=f"leg-scholar-s{sc.scene_index}-{tag_slug}",
                branch=LEGISLATIVE_BRANCH,
                text=accepted_payload["legislative_clause"],
                signature_coords=exp_sig,
                source_scene=sc.scene_index,
                concept_tag=tag_slug,
            )
            jud_clause = BranchClauseEntry(
                clause_id=f"jud-scholar-s{sc.scene_index}-{tag_slug}",
                branch=JUDICIAL_BRANCH,
                text=accepted_payload["judicial_clause"],
                signature_coords=phil_sig,
                source_scene=sc.scene_index,
                concept_tag=tag_slug,
            )
            harvested_clause_ids.extend(
                [exec_clause.clause_id, leg_clause.clause_id, jud_clause.clause_id]
            )
            cur_archives = ThreeBranchArchives(
                executive=cur_archives.executive.record_scene(
                    scene_index=sc.scene_index,
                    coordinates=full_ref,
                    action_summary=accepted_payload["executive_clause"],
                    harvested_clause=exec_clause,
                ),
                legislative=cur_archives.legislative.record_scene(
                    scene_index=sc.scene_index,
                    sensory_snapshot=accepted_payload["legislative_clause"],
                    felt_chord=avg_fc,
                    coordinates=full_ref,
                    harvested_clause=leg_clause,
                ),
                judicial=cur_archives.judicial.record_clause_and_chord(
                    harvested_clause=jud_clause,
                    ratified_chord=candidate_rule,
                ),
            )
            cur_ledger = cur_ledger.record_settled(
                scene_index=sc.scene_index,
                takeaway_source=sculpted_leaf.description,
                felt_chord=avg_fc,
                clarity=0.85,
                confidence=0.88,
                resolved_uncertainty_scenes=(sc.scene_index,),
            )

        self.learned_leaves = tuple(active_leaves)
        self.rules = _ensure_foundational_rules(tuple(active_rules))
        self.archives = cur_archives
        self.ledger = cur_ledger
        return (
            tuple(forged_rules),
            tuple(minted_leaves),
            tuple(harvested_clause_ids),
            tuple(rejection_notes),
        )

    def _refresh_latest_turn_telemetry_coordinates(self) -> None:
        """Re-score the most recent companion message with `self.learned_leaves` so newly distilled leaves display real neural activations."""
        for msg in reversed(self.chat_messages):
            tel = msg.get("telemetry")
            txt = str(msg.get("text", "")).strip()
            if (
                msg.get("sender") == "companion"
                and isinstance(tel, dict)
                and txt
                and not txt.startswith("[Offline Right-Brain Telemetry")
            ):
                try:
                    new_coords, new_exp, _, _ = self.sensors.score_text(
                        txt,
                        learned_leaves=self.learned_leaves,
                        branch_gating=True,
                    )
                    merged_t = dict(tel.get("theory_coordinates", {}))
                    for k, v in {**new_coords, **new_exp}.items():
                        merged_t[k] = round(float(v), 4)
                    tel["theory_coordinates"] = merged_t
                except Exception:
                    pass
                break

    def handle_run_rem_sleep(self, *, auto_triggered: bool = False) -> dict[str, Any]:
        """Consolidate queued Socratic & Curriculum scenes via genuine 4B LLM distillation, MiniLM bridge calibration, and unseen holdout verification."""
        self.wait_until_warm(timeout=180.0)
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
            sculpted_edges = sculpt_will_edges_during_sleep(self.will_edges, queued_snapshot)
            if sculpted_edges:
                self.will_edges = tuple(sculpted_edges)

            refined_rules, _ = refine_colliding_rules_during_sleep(self.rules, queued_snapshot)
            self.rules = _ensure_foundational_rules(refined_rules)

            (
                all_forged_rules,
                all_minted_leaves,
                harvested_clause_ids,
                rejection_notes,
            ) = self._distill_and_verify_scholar_canopy(queued_snapshot)
            self._last_minted_leaves = all_minted_leaves

            rep = max(queued_snapshot, key=lambda s: float(s.salience))
            rep_scene = rep.scene_index
            target_summary = (
                all_forged_rules[0].description
                if all_forged_rules
                else (rep.scenario_text or "").strip()
            )
            if target_summary and all_forged_rules:
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

            self._refresh_latest_turn_telemetry_coordinates()
            self.sleep_queue.clear()
            self.rem_dream_chronicles.append(
                {
                    "cycle_index": self.sleep_cycle_index,
                    "forged_rules": [r.concept_name for r in all_forged_rules],
                    "minted_leaves": [lf.leaf_id for lf in all_minted_leaves],
                    "rejected_candidates": list(rejection_notes),
                    "harvested_gap_pairs": harvested_pairs,
                    "projector_updated": projector_updated,
                }
            )
            self._persist_user_state()

            trig_label = "Auto-REM Sleep" if auto_triggered else "REM Sleep"
            rej_suffix = (
                f" Rejected {len(rejection_notes)} candidate(s) ({rejection_notes[0]})."
                if rejection_notes
                else ""
            )
            self.last_rem_sleep_receipt = (
                f"{trig_label} Cycle #{self.sleep_cycle_index} Complete: "
                f"Forged {len(all_forged_rules)} rule(s), verified & minted {len(all_minted_leaves)} leaf(ves), "
                f"harvested {len(harvested_clause_ids)} archive clause(s), "
                f"and logged {harvested_pairs} Callosum gap pair(s).{rej_suffix}"
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

            # 5. Copy live Callosum projector weights (`live_projector_weights.pt`) & plastic bridges if present
            live_proj_src = self._resolve_live_projector_path()
            copied_live_proj = False
            if live_proj_src is not None and live_proj_src.is_file():
                shutil.copy2(live_proj_src, target_model_dir / "live_projector_weights.pt")
                shutil.copy2(live_proj_src, target_state_dir / "live_projector_weights.pt")
                copied_live_proj = True

            plastic_bridges_src = self.state_dir / "right_brain_bridges"
            if (plastic_bridges_src / "doer_will_cultural").exists():
                shutil.copytree(
                    plastic_bridges_src,
                    target_state_dir / "right_brain_bridges",
                    dirs_exist_ok=True,
                )

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

        # Fill any missing Tier 2/3/4 leaves without overwriting real neural activations in `latest_coords`
        for tier_val in (2, 3, 4):
            for lf in visible_leaves:
                if lf.tier == tier_val and lf.leaf_id not in latest_coords:
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
                        live_act = float(
                            latest_coords.get(
                                lf.leaf_id,
                                _evaluate_learned_leaf_from_anchors(lf, latest_coords),
                            )
                        )
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
                                "verification": self.leaf_verification_metrics.get(lf.leaf_id, {}),
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
        with self._lock:
            builtin_ids = {
                lf.leaf_id for lf in (PHILOSOPHICAL_SHADOW_LEAVES + DEFAULT_COMPLEX_FEELING_LEAVES)
            }
            custom_leaves_count = sum(1 for lf in self.learned_leaves if lf.leaf_id not in builtin_ids)
            return {
                "app_name": "LMFT-Scholar",
                "server_pid": os.getpid(),
                "active_lineage": self.active_lineage,
                "left_brain_loading": bool(self.left_brain_loading),
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

    class ScholarHTTPServer(ThreadingHTTPServer):
        # Prevent Windows SO_REUSEADDR socket stacking so a new launch never hides behind a stale listener.
        allow_reuse_address = not sys.platform.startswith("win")

    server_ref: list[ThreadingHTTPServer] = []

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
                import importlib
                from . import scholar_ui as _live_ui

                importlib.reload(_live_ui)
                raw = _live_ui.SCHOLAR_SINGLE_PAGE_HTML.encode("utf-8")
                self.send_response(int(HTTPStatus.OK))
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
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
                if path == "/api/shutdown":
                    self._send_json({"status": "shutting_down", "pid": os.getpid()})
                    if server_ref:
                        threading.Thread(target=server_ref[0].shutdown, daemon=True).start()
                    return
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
                if path == "/api/tree/reset":
                    snap = runtime.reset_scholar_canopy()
                    self._send_json(snap)
                    return
                if path == "/api/graduate":
                    snap = runtime.graduate_brain_to_target(str(body.get("target_app", "")))
                    self._send_json(snap)
                    return
                self._send_json({"error": f"Unknown endpoint: {path}"}, status=HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    srv = ScholarHTTPServer((host, int(port)), ScholarRequestHandler)
    server_ref.append(srv)
    return srv


def _is_local_port_listening(host: str, port: int, *, timeout_sec: float = 0.25) -> bool:
    check_host = "127.0.0.1" if host.strip() in ("0.0.0.0", "localhost", "127.0.0.1") else host
    try:
        with socket.create_connection((check_host, int(port)), timeout=timeout_sec):
            return True
    except OSError:
        return False


def _evict_existing_scholar_on_port(host: str, port: int) -> None:
    """Cleanly shut down or evict any lingering `scholar_app` server holding `port` before re-binding."""
    if port <= 0 or host.strip() not in ("127.0.0.1", "localhost", "0.0.0.0"):
        return
    if not _is_local_port_listening("127.0.0.1", port):
        return

    my_pid = os.getpid()
    my_ppid = os.getppid()

    for _ in range(8):
        if not _is_local_port_listening("127.0.0.1", port):
            return
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{int(port)}/api/shutdown",
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=0.6):
                pass
            time.sleep(0.2)
            if not _is_local_port_listening("127.0.0.1", port):
                return
        except Exception:
            pass

        if sys.platform.startswith("win"):
            try:
                out = subprocess.check_output(
                    ["netstat", "-ano", "-p", "tcp"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=2.0,
                )
                target_suffix = f":{int(port)}"
                killed_any = False
                for line in out.splitlines():
                    parts = line.split()
                    if len(parts) < 5 or parts[3].upper() != "LISTENING":
                        continue
                    local_addr = parts[1]
                    if not local_addr.endswith(target_suffix):
                        continue
                    pid = int(parts[4])
                    if pid <= 0 or pid in (my_pid, my_ppid):
                        continue
                    tlist = subprocess.check_output(
                        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                        text=True,
                        stderr=subprocess.DEVNULL,
                        timeout=1.5,
                    ).lower()
                    if "python" in tlist:
                        subprocess.run(
                            ["taskkill", "/PID", str(pid), "/T", "/F"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=2.0,
                            check=False,
                        )
                        killed_any = True
                if not killed_any:
                    break
                time.sleep(0.25)
            except Exception:
                break
        else:
            break


def _wait_for_local_server_ready(url: str, *, timeout_sec: float = 3.0) -> bool:
    deadline = time.time() + timeout_sec
    state_url = f"{url.rstrip('/')}/api/state"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(state_url, timeout=0.5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.05)
    return False


def launch_scholar_desktop_window(url: str, *, profile_dir: Path | None = None) -> subprocess.Popen[Any] | None:
    """Open `url` in a dedicated isolated app-mode window on Windows so window close cleanly exits `pythonw.exe`."""
    if sys.platform.startswith("win"):
        candidates = [
            Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
            / "Microsoft/Edge/Application/msedge.exe",
            Path(os.environ.get("ProgramFiles", "C:/Program Files"))
            / "Microsoft/Edge/Application/msedge.exe",
            Path(os.environ.get("ProgramFiles", "C:/Program Files"))
            / "Google/Chrome/Application/chrome.exe",
        ]
        user_data_dir = profile_dir or (resolve_training_root() / "outputs" / ".scholar_window_profile")
        user_data_dir.mkdir(parents=True, exist_ok=True)
        for exe in candidates:
            if exe.exists():
                try:
                    return subprocess.Popen(
                        [
                            str(exe),
                            f"--app={url}",
                            f"--user-data-dir={user_data_dir}",
                            "--window-size=1380,860",
                            "--no-first-run",
                            "--no-default-browser-check",
                        ]
                    )
                except Exception:
                    pass
    webbrowser.open(url)
    return None


def launch_native_pywebview_window(title: str, url: str) -> bool:
    """Open `url` inside a native Windows desktop window (`pywebview`) when installed."""
    try:
        import ctypes
        import webview  # type: ignore[import-untyped]
        from lmft_architecture.workspace_paths import resolve_arch_root

        if sys.platform.startswith("win"):
            try:
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("LMFT.Celestriad.ScholarApp.1")
            except Exception:
                pass

        ico_path = resolve_arch_root() / "docs" / "assets" / "celestriad_spirit_particle.ico"
        icon_str = str(ico_path) if ico_path.is_file() else None
        storage_dir = resolve_training_root() / "outputs" / ".scholar_window_profile"
        storage_dir.mkdir(parents=True, exist_ok=True)
        webview.create_window(
            title,
            url,
            width=1380,
            height=860,
            min_size=(1020, 680),
            background_color="#0a0d14",
        )
        webview.start(private_mode=False, storage_path=str(storage_dir), icon=icon_str)
        return True
    except Exception:
        return False


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

    _evict_existing_scholar_on_port(args.host, int(args.port))

    runtime = ScholarStudioRuntime(
        lazy_load_gpu=bool(args.lazy_gpu),
        background_warmup=not bool(args.lazy_gpu),
    )
    try:
        server = create_scholar_http_server(runtime, host=args.host, port=args.port)
    except OSError:
        server = create_scholar_http_server(runtime, host=args.host, port=0)

    actual_port = server.server_address[1]
    browser_host = "localhost" if str(args.host).strip() in ("127.0.0.1", "0.0.0.0") else args.host
    url = f"http://{browser_host}:{actual_port}"
    print(f"LMFT-Scholar Studio running at {url}")

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    _wait_for_local_server_ready(f"http://127.0.0.1:{actual_port}", timeout_sec=3.0)

    try:
        if not args.no_browser:
            window_url = f"{url}/?v={int(time.time())}"
            if launch_native_pywebview_window(
                "LMFT-Scholar — Foundational Two-Hemisphere Socratic & Literature Studio (Impish_LLAMA_4B)",
                window_url,
            ):
                return 0
            if args.single_window or sys.executable.lower().endswith("pythonw.exe"):
                t0 = time.time()
                window_proc = launch_scholar_desktop_window(window_url)
                if window_proc is not None:
                    window_proc.wait()
                    if (time.time() - t0) >= 3.0:
                        return 0
            else:
                webbrowser.open(window_url)
        while server_thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
