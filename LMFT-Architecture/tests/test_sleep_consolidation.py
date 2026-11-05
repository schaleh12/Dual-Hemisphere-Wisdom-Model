"""Unit tests for Offline 9B Sleep-Mode Consolidation (`sleep_consolidation.py`)."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from lmft_architecture.branch_archives import ThreeBranchArchives
from lmft_architecture.contemplation_ledger import ContemplationLedger, SubliminalMarginTag
from lmft_architecture.model_assets import ModelError
from lmft_architecture.reflection_classifier import FuzzyConceptRule, load_ruleset
from lmft_architecture.sleep_consolidation import (
    UnresolvedSleepScene,
    assert_domain_general_text,
    refine_colliding_rules_during_sleep,
    run_sleep_consolidation,
)


def test_assert_domain_general_text_rejects_banned_substrings():
    banned = ("mason", "bakehouse", "lintel", "rye", "shutter")
    assert (
        assert_domain_general_text(
            "An enclosed space restricts shared access behind rigid rules.",
            banned_substrings=banned,
        )
        == "An enclosed space restricts shared access behind rigid rules."
    )
    with pytest.raises(ValueError, match="story-specific term"):
        assert_domain_general_text(
            "The mason carved rules into the lintel.",
            banned_substrings=banned,
        )


def test_refine_colliding_rules_during_sleep_tightens_any_colliding_rule():
    rules = (
        FuzzyConceptRule(
            rule_id="rule-custom-steady-routine",
            concept_name="custom_steady_routine",
            description="Steady self-contained routine without external dependencies.",
            target_leaves={
                "cultural/freedom/needlessness": 0.45,
                "cultural/freedom/self-sufficiency": 0.65,
                "philosophical/being/agency": 0.70,
                "philosophical/persistence/continuity": 0.65,
            },
            low=0.60,
            high=0.85,
        ),
    )
    unresolved = (
        UnresolvedSleepScene(
            scene_index=5,
            scenario_text="High-order enclosed scene where custom_steady_routine misfired.",
            external_coordinates={"philosophical/structure/chaos": 0.14},
            felt_chord_notes={},
            valence=0.40,
            salience=0.80,
            arousal=0.50,
            left_brain_hunch="Rigid barrier blocks communal sharing.",
            active_concept_collision="custom_steady_routine",
        ),
    )
    refined, refined_ids = refine_colliding_rules_during_sleep(rules, unresolved)
    assert refined_ids == ("rule-custom-steady-routine",)
    r = refined[0]
    assert r.target_leaves["philosophical/structure/chaos"] == 0.10
    assert r.target_leaves["cultural/freedom/needlessness"] == 0.75
    assert r.target_leaves["cultural/freedom/self-sufficiency"] == 0.75
    assert r.target_leaves["philosophical/persistence/continuity"] == 0.75
    assert r.low == 0.72
    assert r.high == 0.88


class _FakeSleepTeacher:
    def __init__(self, payload_by_cluster: dict[str, dict[str, str]]):
        self.payload_by_cluster = payload_by_cluster
        self.calls = 0

    def generate(self, role, request, prompt_version, validator):
        self.calls += 1
        user_msg = json.loads(request["messages"][1]["content"])
        ckey = user_msg["pattern_id"]
        raw = json.dumps(self.payload_by_cluster[ckey])
        validated = validator(raw)
        return validated, {"role": role, "prompt_version": prompt_version}


def test_run_sleep_consolidation_forges_general_rules_and_rejects_story_nouns(tmp_path: Path):
    base_rules = (
        FuzzyConceptRule(
            rule_id="rule-fair-cooperation",
            concept_name="fair_structured_cooperation",
            description="Orderly, fair cooperation where resources are shared equitably.",
            target_leaves={
                "cultural/justice/fairness": 0.90,
                "philosophical/structure/order": 0.85,
                "philosophical/structure/chaos": 0.10,
            },
        ),
    )
    unresolved = (
        UnresolvedSleepScene(
            scene_index=4,
            scenario_text="A rigid gatekeeper enforces arbitrary entry rules that exclude outsiders.",
            external_coordinates={
                "cultural/efficiency/power": 0.67,
                "cultural/efficiency/wastelessness": 0.61,
                "cultural/justice/honor": 0.64,
                "cultural/justice/fairness": 0.16,
                "cultural/freedom/self-sufficiency": 0.35,
                "cultural/freedom/needlessness": 0.25,
                "cultural/justice/exclusionary_gatekeeping": 0.56,
                "philosophical/being/agency": 0.70,
                "philosophical/being/existence": 0.94,
                "philosophical/structure/order": 0.84,
                "philosophical/structure/chaos": 0.08,
                "philosophical/persistence/continuity": 0.48,
                "philosophical/persistence/discreteness": 0.78,
            },
            felt_chord_notes={
                "experiential/phase/austere_sanctuary": 0.10,
                "experiential/phase/hollow_stillness": 0.48,
                "experiential/phase/bittersweet_release": 0.08,
                "experiential/synchronization/earned_return": 0.15,
            },
            valence=0.35,
            salience=0.80,
            arousal=0.45,
            left_brain_hunch="Rigid authority creates an exclusionary boundary that shuts out shared access.",
            social_drift_detected=True,
        ),
    )

    margin = SubliminalMarginTag(
        clarity=0.25,
        confidence=0.30,
        felt_reaction="unsettled chill",
        red_pen_prompt="What is missing?",
    )
    ledger = ContemplationLedger().record_uncertainty(
        scene_index=4,
        raw_hunch="Rigid authority creates an exclusionary boundary.",
        open_question="What is missing?",
        margin_tag=margin,
    )
    archives = ThreeBranchArchives()
    banned = ("mason", "bakehouse", "lintel")

    # 1. Verify that story-specific nouns from a Teacher are rejected
    bad_teacher = _FakeSleepTeacher(
        {
            "low_fairness_order_cluster": {
                "concept_name": "rigid_exclusionary_enclosure",
                "description": "The mason blocks access behind a carved lintel.",
                "executive_clause_text": "Access is denied through strict gatekeeping.",
                "legislative_clause_text": "Cold isolation replaces shared warmth.",
                "judicial_clause_text": "A discrete physical boundary seals the interior.",
            }
        }
    )
    with pytest.raises(ModelError, match="story-specific term"):
        run_sleep_consolidation(
            unresolved_scenes=unresolved,
            rules=base_rules,
            learned_leaves=(),
            archives=archives,
            ledger=ledger,
            teacher=bad_teacher,
            banned_substrings=banned,
        )

    # 2. Verify that domain-general Teacher definitions succeed and persist to disk
    good_teacher = _FakeSleepTeacher(
        {
            "low_fairness_order_cluster": {
                "concept_name": "rigid_exclusionary_enclosure",
                "description": "Ordered authority erects a strict boundary that denies equitable access to shared resources.",
                "executive_clause_text": "Strict access conditions exclude outsiders from a previously shared resource.",
                "legislative_clause_text": "Cold, isolated stillness replaces warm communal relief.",
                "judicial_clause_text": "High structural order and discrete boundaries seal the space from external flow.",
            }
        }
    )
    ruleset_out = tmp_path / "sleep_rules.json"
    archives_dir = tmp_path / "archives"
    ledger_out = tmp_path / "ledger.json"

    report = run_sleep_consolidation(
        unresolved_scenes=unresolved,
        rules=base_rules,
        learned_leaves=(),
        archives=archives,
        ledger=ledger,
        teacher=good_teacher,
        ruleset_out_path=ruleset_out,
        archives_out_dir=archives_dir,
        ledger_out_path=ledger_out,
        banned_substrings=banned,
    )
    assert len(report.forged_rules) == 1
    forged = report.forged_rules[0]
    assert forged.concept_name == "rigid_exclusionary_enclosure"
    assert forged.target_leaves["cultural/justice/fairness"] == 0.20
    assert forged.target_leaves["philosophical/structure/order"] == 0.75
    assert forged.target_leaves["philosophical/persistence/discreteness"] == 0.70
    assert report.resolved_scene_indices == (4,)
    assert len(report.updated_ledger.active_uncertainties) == 0
    assert len(report.updated_archives.judicial.ratified_chords) == 1

    reloaded = load_ruleset(ruleset_out)
    assert len(reloaded) == 2


def test_sleep_consolidation_clusters_non_story_coordinate_profiles():
    """Verify Sleep Mode clusters any unresolved coordinate profile (not just two hardcoded story beats)."""
    unresolved = (
        UnresolvedSleepScene(
            scene_index=2,
            scenario_text="An aggressive tactical gambit sacrifices steady material for tempo.",
            external_coordinates={
                "cultural/efficiency/power": 0.82,
                "cultural/efficiency/wastelessness": 0.25,
                "cultural/justice/honor": 0.55,
                "cultural/justice/fairness": 0.52,
                "cultural/freedom/self-sufficiency": 0.70,
                "cultural/freedom/needlessness": 0.30,
                "philosophical/being/agency": 0.88,
                "philosophical/being/existence": 0.75,
                "philosophical/structure/order": 0.45,
                "philosophical/structure/chaos": 0.30,
                "philosophical/persistence/continuity": 0.40,
                "philosophical/persistence/discreteness": 0.50,
            },
            felt_chord_notes={},
            valence=0.52,
            salience=0.78,
            arousal=0.80,
            left_brain_hunch="Aggressive tempo sacrifice trades stability for initiative.",
        ),
    )
    teacher = _FakeSleepTeacher(
        {
            "high_power_agency_cluster": {
                "concept_name": "bold_initiative_gambit",
                "description": "Deliberate high-agency force trades material conservation for decisive momentum.",
                "executive_clause_text": "Commanding force prioritizes active momentum over cautious conservation.",
                "legislative_clause_text": "Sharp urgency and focused tension drive the decisive commitment.",
                "judicial_clause_text": "High deliberate agency reshapes the structural balance.",
            }
        }
    )
    report = run_sleep_consolidation(
        unresolved_scenes=unresolved,
        rules=(),
        learned_leaves=(),
        archives=ThreeBranchArchives(),
        ledger=ContemplationLedger(),
        teacher=teacher,
    )
    assert len(report.forged_rules) == 1
    assert report.forged_rules[0].concept_name == "bold_initiative_gambit"
    assert report.forged_rules[0].target_leaves["cultural/efficiency/power"] == 0.78
    assert report.forged_rules[0].target_leaves["philosophical/being/agency"] == 0.75


def test_sleep_consolidation_skips_social_drift_vetoed_moderate_fairness_interior():
    """Verify a scene held back by social_drift with static fairness > 0.40 is not forged into a static rule."""
    unresolved = (
        UnresolvedSleepScene(
            scene_index=6,
            scenario_text="Inside the sealed room, stillness hangs while hooves scrape outside.",
            external_coordinates={
                "cultural/efficiency/power": 0.14,
                "cultural/justice/fairness": 0.51,
                "philosophical/structure/order": 0.75,
                "philosophical/structure/chaos": 0.25,
                "philosophical/persistence/discreteness": 0.70,
            },
            felt_chord_notes={},
            valence=0.35,
            salience=0.70,
            arousal=0.40,
            left_brain_hunch="Historical exclusion lingers despite the calm interior.",
            social_drift_detected=True,
        ),
    )
    teacher = _FakeSleepTeacher({})
    report = run_sleep_consolidation(
        unresolved_scenes=unresolved,
        rules=(),
        learned_leaves=(),
        archives=ThreeBranchArchives(),
        ledger=ContemplationLedger(),
        teacher=teacher,
    )
    assert len(report.forged_rules) == 0
    assert teacher.calls == 0

