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



def test_apply_dream_reflection_to_archives_harvests_four_sentence_poem():
    """Verify a 4-sentence REM Dream contemplation writes felt, structural, and action sentences into the 3-Branch Archives."""
    from lmft_architecture.sleep_consolidation import (
        apply_dream_reflection_to_archives,
        consolidate_sleep_clusters_locally,
    )

    unresolved = (
        UnresolvedSleepScene(
            scene_index=9,
            scenario_text="Move 12 (Black): played Bxf2+ instead of O-O.",
            external_coordinates={
                "cultural/efficiency/power": 0.78,
                "cultural/efficiency/wastelessness": 0.22,
                "cultural/justice/honor": 0.45,
                "cultural/justice/fairness": 0.48,
                "philosophical/being/agency": 0.76,
                "philosophical/structure/order": 0.28,
                "philosophical/structure/chaos": 0.74,
            },
            felt_chord_notes={},
            valence=0.28,
            salience=0.82,
            arousal=0.79,
            left_brain_hunch="Reckless bishop sacrifice exposed the king.",
        ),
    )
    report = consolidate_sleep_clusters_locally(
        queued_scenes=unresolved,
        rules=(),
        learned_leaves=(),
        archives=ThreeBranchArchives(),
        ledger=ContemplationLedger(),
    )
    dream_poem = (
        "A sudden flash of greed drew the bishop into the fire before the fortress was sealed. "
        "The resulting counter-shock left the uncastled monarch shivering on an open file. "
        "True structural order requires securing the shelter before igniting a tactical storm. "
        "Next time we castle first and let calm continuity anchor the attack."
    )
    updated = apply_dream_reflection_to_archives(
        report.updated_archives,
        scene_index=9,
        dream_text=dream_poem,
    )
    assert any(
        "sudden flash of greed" in c.text and "shivering on an open file" in c.text
        for c in updated.legislative.feeling_clauses
    )
    assert any(
        "True structural order requires securing the shelter" in c.text
        for c in updated.judicial.structural_clauses
    )
    assert any(
        "Next time we castle first" in c.text
        for c in updated.executive.action_clauses
    )


def test_local_rem_sleep_clean_english_archives_and_discriminative_tier2_leaf_birth() -> None:
    """Verify Fix #4b (zero snake_case rule IDs in archives) and Fix #5 (authentic Tier-2 LearnedLeaf birth with 8-float anchor_weights)."""
    from lmft_architecture.reflection_classifier import FOUNDATIONAL_LEAVES
    from lmft_architecture.sleep_consolidation import consolidate_sleep_clusters_locally
    from lmft_architecture.subliminal_gap_splice import pool_departmental_child_chord

    def _base_coords(**overrides: float) -> dict[str, float]:
        base = {k: 0.50 for k in FOUNDATIONAL_LEAVES}
        base.update(overrides)
        return base

    scenes = (
        UnresolvedSleepScene(
            scene_index=14,
            scenario_text="Move 7: Played Bxf7+ sacrificing the bishop prematurely.",
            external_coordinates=_base_coords(
                **{
                    "cultural/efficiency/wastelessness": 0.20,
                    "philosophical/structure/order": 0.32,
                    "philosophical/structure/chaos": 0.74,
                }
            ),
            felt_chord_notes={},
            valence=0.22,
            salience=0.84,
            arousal=0.72,
            left_brain_hunch="I lunged at f7 before castling our King to safety.",
            is_hazard=True,
            contrastive_reference_coordinates=_base_coords(
                **{
                    "cultural/efficiency/wastelessness": 0.78,
                    "philosophical/structure/order": 0.82,
                    "philosophical/structure/chaos": 0.24,
                }
            ),
            game_review_annotation="Played Bxf7+ instead of best move O-O.",
        ),
        UnresolvedSleepScene(
            scene_index=18,
            scenario_text="Move 11: Played Ng5+ overextending the knight.",
            external_coordinates=_base_coords(
                **{
                    "cultural/efficiency/wastelessness": 0.22,
                    "philosophical/structure/order": 0.35,
                    "philosophical/structure/chaos": 0.70,
                }
            ),
            felt_chord_notes={},
            valence=0.25,
            salience=0.80,
            arousal=0.68,
            left_brain_hunch="We chased a premature attack with the knight alone.",
            is_hazard=True,
            contrastive_reference_coordinates=_base_coords(
                **{
                    "cultural/efficiency/wastelessness": 0.76,
                    "philosophical/structure/order": 0.80,
                    "philosophical/structure/chaos": 0.26,
                }
            ),
            game_review_annotation="Played Ng5+ instead of developing the queenside.",
        ),
    )

    report = consolidate_sleep_clusters_locally(
        queued_scenes=scenes,
        rules=(),
        learned_leaves=(),
        archives=ThreeBranchArchives(),
        ledger=ContemplationLedger(),
    )
    # 1. Fix #4b: Zero snake_case `hazard_..._cluster` rule names inside any archive clause text
    for c in (
        report.updated_archives.executive.action_clauses
        + report.updated_archives.legislative.feeling_clauses
        + report.updated_archives.judicial.structural_clauses
    ):
        assert "_cluster" not in c.text
        assert "hazard_" not in c.text

    # 2. Fix #5: Authentic discriminative Tier-2 `LearnedLeaf` minted with 8-float `anchor_weights`
    assert len(report.minted_leaves) >= 1
    assert len(report.updated_learned_leaves) >= 1
    minted = report.minted_leaves[0]
    assert minted.tier == 2
    assert minted.anchor_weights is not None
    assert len(minted.anchor_weights) == 8
    assert max(minted.anchor_weights) == pytest.approx(1.0)
    # Verify `pool_departmental_child_chord` lights up the 8 departmental child chord slots!
    dept_prefix = minted.leaf_id.split("/")[0] + "/"
    child_vec = pool_departmental_child_chord(
        {minted.leaf_id: 0.82},
        department_prefix=dept_prefix,
        learned_leaves=report.updated_learned_leaves,
    )
    assert len(child_vec) == 8
    assert max(child_vec) > 0.50


def test_core_weight_space_sculpting_upgrades_a_b_c() -> None:
    """Verify D066 Core Weight-Space Sculpting (#C Rule Entrainment, #A LearnedLeaf Anchor Weights, #B Right-Brain Will Edges)."""
    from lmft_architecture.birth import create_birth_state
    from lmft_architecture.neural_bridge_classifiers import DEFAULT_COMPLEX_FEELING_LEAVES
    from lmft_architecture.reflection_classifier import FOUNDATIONAL_LEAVES, FuzzyConceptRule, LearnedLeaf
    from lmft_architecture.right_brain_equilibrium import (
        RIGHT_BRAIN_WILL_EDGES,
        compile_right_brain_vibe,
        sculpt_will_edges_during_sleep,
    )
    from lmft_architecture.sleep_consolidation import (
        consolidate_sleep_clusters_locally,
        entrain_existing_cluster_rule,
    )
    from lmft_architecture.subliminal_gap_splice import (
        _DEFAULT_LEAF_ANCHOR_WEIGHTS,
        sculpt_learned_leaf_anchor_weights,
    )

    # 1. Upgrade #C: Contrastive Rule Entrainment & Homeostatic Threshold Unsticking
    existing_rule = FuzzyConceptRule(
        rule_id="rule-sleep-hazard_high_power_chaos_cluster",
        concept_name="hazard_high_power_chaos_cluster",
        description="Old hazard rule stuck near ceiling.",
        target_leaves={
            "cultural/efficiency/wastelessness": 0.25,
            "cultural/efficiency/power": 0.72,
            "philosophical/structure/chaos": 0.65,
            "cultural/justice/fairness": 0.52,
        },
        low=0.78,
        high=0.93,
        polarity="repulsive",
    )
    incoming_rule = FuzzyConceptRule(
        rule_id="rule-sleep-hazard_high_power_chaos_cluster",
        concept_name="hazard_high_power_chaos_cluster",
        description="Updated hazard rule.",
        target_leaves={
            "cultural/efficiency/wastelessness": 0.20,
            "philosophical/structure/chaos": 0.75,
            "cultural/freedom/needlessness": 0.53,
        },
        low=0.74,
        high=0.91,
        polarity="repulsive",
    )
    entrained = entrain_existing_cluster_rule(
        existing_rule,
        incoming_rule,
        "hazard_high_power_chaos_cluster",
        max_target_leaves=3,
    )
    assert entrained.polarity == "repulsive"
    assert len(entrained.target_leaves) == 3
    # Shared keys (`wastelessness` and `chaos`) are prioritized over non-shared keys, and `power` (0.72, extremity 0.22) beats `fairness` (0.52) / `needlessness` (0.53)
    assert "cultural/efficiency/wastelessness" in entrained.target_leaves
    assert "philosophical/structure/chaos" in entrained.target_leaves
    assert "cultural/efficiency/power" in entrained.target_leaves
    assert entrained.low <= 0.75
    assert entrained.high <= 0.90

    # 2. Upgrade #A: Frozen Stage 1-5 + Shadow baseline leaves remain untouched; dynamic Tier-2 leaf is sculpted
    frozen_leaf = DEFAULT_COMPLEX_FEELING_LEAVES[0]
    assert frozen_leaf.leaf_id in _DEFAULT_LEAF_ANCHOR_WEIGHTS
    untouched = sculpt_learned_leaf_anchor_weights(
        frozen_leaf,
        {"cultural/efficiency/power": 0.95, "cultural/efficiency/wastelessness": 0.05},
        {"cultural/efficiency/power": 0.10, "cultural/efficiency/wastelessness": 0.90},
    )
    assert untouched == frozen_leaf

    dyn_leaf = LearnedLeaf(
        parent_branch="cultural/efficiency",
        leaf_name="hazard_high_power_chaos",
        description="Dynamic tactical overextension leaf.",
        anchor_weights=(1.0, 0.50, 0.40, 0.40, 0.40, 0.40, 0.50, 0.50),
    )
    sculpted_leaf = sculpt_learned_leaf_anchor_weights(
        dyn_leaf,
        {"cultural/efficiency/wastelessness": 0.15, "cultural/efficiency/power": 0.80},
        {"cultural/efficiency/wastelessness": 0.90, "cultural/efficiency/power": 0.50},
    )
    assert sculpted_leaf.anchor_weights != dyn_leaf.anchor_weights
    assert max(sculpted_leaf.anchor_weights) == pytest.approx(1.0)
    # `wastelessness` is index 1 in cultural department keys; its weight should increase under large contrastive delta
    assert sculpted_leaf.anchor_weights[1] > dyn_leaf.anchor_weights[1]

    # 3. Upgrade #B: Synaptic Weight-Space Sculpting for `RIGHT_BRAIN_WILL_EDGES`
    base_coords = {k: 0.50 for k in FOUNDATIONAL_LEAVES}
    haz_scene = UnresolvedSleepScene(
        scene_index=21,
        scenario_text="Impulsive tactical lunge.",
        external_coordinates={
            **base_coords,
            "cultural/efficiency/power": 0.84,
            "cultural/efficiency/wastelessness": 0.20,
            "cultural/freedom/needlessness": 0.22,
            "philosophical/structure/chaos": 0.72,
            "psychological/truth/fact": 0.25,
            dyn_leaf.leaf_id: 0.82,
        },
        felt_chord_notes={},
        valence=0.25,
        salience=0.82,
        arousal=0.85,
        left_brain_hunch="Lunged too early.",
        is_hazard=True,
        contrastive_reference_coordinates={
            **base_coords,
            "cultural/efficiency/power": 0.52,
            "cultural/efficiency/wastelessness": 0.82,
            "cultural/freedom/needlessness": 0.80,
            "philosophical/structure/chaos": 0.22,
        },
    )
    sculpted_edges = sculpt_will_edges_during_sleep(RIGHT_BRAIN_WILL_EDGES, (haz_scene,))
    orig_by_id = {e.id: e.weight for e in RIGHT_BRAIN_WILL_EDGES}
    new_by_id = {e.id: e.weight for e in sculpted_edges}
    assert new_by_id["rb-will-1"] < orig_by_id["rb-will-1"]
    assert new_by_id["rb-will-2"] < orig_by_id["rb-will-2"]
    assert new_by_id["rb-will-8"] < orig_by_id["rb-will-8"]  # Stronger inhibitory Needlessness -> Execution

    # Verify symmetric opposite-direction plasticity (no one-way ratchet)
    decisive_scene = UnresolvedSleepScene(
        scene_index=22,
        scenario_text="Missed decisive grounded conversion.",
        external_coordinates={
            **base_coords,
            "cultural/efficiency/power": 0.22,
            "cultural/freedom/needlessness": 0.82,
            "psychological/truth/fact": 0.88,
        },
        felt_chord_notes={},
        valence=0.65,
        salience=0.75,
        arousal=0.30,
        left_brain_hunch="Hesitated instead of converting.",
        is_hazard=False,
        contrastive_reference_coordinates={
            **base_coords,
            "cultural/efficiency/power": 0.86,
            "cultural/freedom/needlessness": 0.24,
        },
    )
    re_sculpted = {e.id: e.weight for e in sculpt_will_edges_during_sleep(sculpted_edges, (decisive_scene,))}
    assert re_sculpted["rb-will-1"] > new_by_id["rb-will-1"]
    assert re_sculpted["rb-will-2"] > new_by_id["rb-will-2"]
    assert re_sculpted["rb-will-8"] > new_by_id["rb-will-8"]

    birth = create_birth_state("test-d066", birth_event_id="birth-d066", random_seed=42)
    exp_map = {
        "experiential/reflection/novelty": 0.45,
        "experiential/reflection/salience": 0.75,
        "experiential/phase/valence": 0.55,
        "experiential/phase/arousal": 0.80,
        "experiential/synchronization/resonance": 0.60,
        "experiential/synchronization/entrainment": 0.55,
    }
    ext_for_vibe = {
        k: v
        for k, v in haz_scene.external_coordinates.items()
        if not k.startswith("psychological/")
    }
    vibe_default, _ = compile_right_brain_vibe(
        birth,
        ext_for_vibe,
        exp_map,
        (),
        pass_index=1,
        event_id="will-default",
        will_edges=RIGHT_BRAIN_WILL_EDGES,
    )
    vibe_sculpted, _ = compile_right_brain_vibe(
        birth,
        ext_for_vibe,
        exp_map,
        (),
        pass_index=1,
        event_id="will-sculpted",
        will_edges=sculpted_edges,
    )
    assert vibe_sculpted.intention < vibe_default.intention

    report = consolidate_sleep_clusters_locally(
        queued_scenes=(haz_scene,),
        rules=(),
        learned_leaves=(frozen_leaf, dyn_leaf),
        archives=ThreeBranchArchives(),
        ledger=ContemplationLedger(),
        will_edges=RIGHT_BRAIN_WILL_EDGES,
    )
    assert len(report.updated_will_edges) == len(RIGHT_BRAIN_WILL_EDGES)
    updated_leaves_by_id = {lf.leaf_id: lf for lf in report.updated_learned_leaves}
    assert updated_leaves_by_id[frozen_leaf.leaf_id] == frozen_leaf
    assert updated_leaves_by_id[dyn_leaf.leaf_id].anchor_weights != dyn_leaf.anchor_weights

    # 4. Verify Same-Family Cluster Archetype Winner-Take-All vs. Cross-Family Genuine Collision
    from lmft_architecture.reflection_classifier import (
        _evaluate_rules_with_inhibition,
        compute_contemplation,
    )
    from lmft_architecture.three_legged_runtime import extract_dream_vow_sentence

    haz_1 = FuzzyConceptRule(
        rule_id="rule-sleep-hazard_low_wastelessness_agency",
        concept_name="hazard_low_wastelessness_agency_cluster",
        description="Hazard sub-facet 1",
        target_leaves={
            "cultural/efficiency/wastelessness": 0.25,
            "cultural/freedom/needlessness": 0.53,
            "philosophical/being/agency": 0.72,
        },
        low=0.58,
        high=0.88,
        polarity="repulsive",
    )
    haz_2 = FuzzyConceptRule(
        rule_id="rule-sleep-hazard_high_power_agency",
        concept_name="hazard_high_power_agency_cluster",
        description="Hazard sub-facet 2",
        target_leaves={
            "cultural/efficiency/wastelessness": 0.25,
            "cultural/efficiency/power": 0.68,
            "philosophical/being/agency": 0.72,
        },
        low=0.58,
        high=0.88,
        polarity="repulsive",
    )
    coached_opp = FuzzyConceptRule(
        rule_id="rule-sleep-coached_high_power_agency",
        concept_name="coached_high_power_agency_cluster",
        description="Opposing coached rule",
        target_leaves={
            "cultural/efficiency/wastelessness": 0.75,
            "cultural/efficiency/power": 0.68,
            "philosophical/being/agency": 0.72,
        },
        low=0.58,
        high=0.88,
        polarity="attractive",
    )
    blunder_coords = {
        **base_coords,
        "cultural/efficiency/wastelessness": 0.20,
        "cultural/freedom/needlessness": 0.55,
        "cultural/efficiency/power": 0.85,
        "philosophical/being/agency": 0.73,
    }
    same_fam_scores = _evaluate_rules_with_inhibition((haz_1, haz_2, coached_opp), blunder_coords)
    sf_clarity, sf_conf, sf_matched, sf_confused, _ = compute_contemplation(same_fam_scores)
    assert not sf_confused
    assert sf_clarity >= 0.50
    assert sf_conf == pytest.approx(1.0)
    assert sf_matched == "hazard_low_wastelessness_agency_cluster"

    # Cross-family tie (`hazard_` at 1.00 vs `coached_` at 1.00) must still collide (`Clarity == 0.00, is_confused == True`)
    coached_bad = FuzzyConceptRule(
        rule_id="rule-sleep-coached_unanchored",
        concept_name="coached_high_power_agency_cluster",
        description="Synthetic cross-family tie probe",
        target_leaves={
            "cultural/efficiency/power": 0.68,
            "philosophical/being/agency": 0.72,
            "philosophical/structure/order": 0.50,
        },
        low=0.58,
        high=0.88,
        polarity="attractive",
    )
    cross_scores = _evaluate_rules_with_inhibition((haz_1, coached_bad), blunder_coords)
    cr_clarity, cr_conf, cr_matched, cr_confused, _ = compute_contemplation(cross_scores)
    assert cr_confused is True
    assert cr_clarity == pytest.approx(0.0)
    assert cr_conf == pytest.approx(1.0)
    assert cr_matched is None

    # Verify `extract_dream_vow_sentence` prefers explicit vow sentences over trailing meta-remarks
    dream_with_meta_tail = (
        "Looking back at my move Qe3 on the board, I rushed the queen sortie before castling. "
        "My vow for the next game is to castle early and coordinate my rooks before attacking. "
        "That's not quite right."
    )
    assert extract_dream_vow_sentence(dream_with_meta_tail) == (
        "My vow for the next game is to castle early and coordinate my rooks before attacking."
    )


def test_round2_no_coordinate_stamping_and_genuine_tier2_gate() -> None:
    """Verify Round 2 fixes: no 0.25/0.53 stamping, genuine calm_deviation gate, and unblended birth anchor_weights."""
    from lmft_architecture.reflection_classifier import FOUNDATIONAL_LEAVES
    from lmft_architecture.sleep_consolidation import consolidate_sleep_clusters_locally
    from lmft_architecture.subliminal_gap_splice import score_leaf_anchor_weights_at_birth

    base_coords = {k: 0.50 for k in FOUNDATIONAL_LEAVES}
    # 1. Hazard scene with genuine non-0.25 wastelessness (0.34) and non-0.53 needlessness (0.41)
    genuine_hazard = UnresolvedSleepScene(
        scene_index=31,
        scenario_text="Premature flank push on move 11.",
        external_coordinates={
            **base_coords,
            "cultural/efficiency/wastelessness": 0.34,
            "cultural/freedom/needlessness": 0.41,
            "philosophical/structure/chaos": 0.76,
            "cultural/efficiency/power": 0.74,
        },
        felt_chord_notes={},
        valence=0.32,
        salience=0.80,
        arousal=0.78,
        left_brain_hunch="Pushed too early on the flank.",
        is_hazard=True,
        contrastive_reference_coordinates={
            **base_coords,
            "cultural/efficiency/wastelessness": 0.78,
            "cultural/freedom/needlessness": 0.18,
            "philosophical/structure/chaos": 0.28,
            "cultural/efficiency/power": 0.54,
        },
    )
    report = consolidate_sleep_clusters_locally(
        queued_scenes=(genuine_hazard,),
        rules=(),
        learned_leaves=(),
        archives=ThreeBranchArchives(),
        ledger=ContemplationLedger(),
    )
    # Raw scene external_coordinates must not be mutated/stamped to 0.25 or 0.53
    assert genuine_hazard.external_coordinates["cultural/efficiency/wastelessness"] == pytest.approx(0.34)
    assert genuine_hazard.external_coordinates["cultural/freedom/needlessness"] == pytest.approx(0.41)
    assert len(report.forged_rules) == 1
    forged = report.forged_rules[0]
    assert forged.target_leaves.get("cultural/efficiency/wastelessness") == pytest.approx(0.34)
    assert forged.target_leaves.get("cultural/efficiency/wastelessness") != pytest.approx(0.25)

    # Verify minted Tier-2 leaf uses score_leaf_anchor_weights_at_birth directly without 45% raw-coordinate blend
    assert len(report.minted_leaves) == 1
    minted = report.minted_leaves[0]
    expected_birth_leaf = score_leaf_anchor_weights_at_birth(minted)
    assert minted.anchor_weights == expected_birth_leaf.anchor_weights

    # 2. Calm scene whose coordinates barely deviate from calm baseline (< 0.12) and have tiny contrastive delta (< 0.20)
    near_calm_scene = UnresolvedSleepScene(
        scene_index=32,
        scenario_text="Quiet positional maneuvering.",
        external_coordinates={
            **base_coords,
            "cultural/efficiency/power": 0.45,  # calm baseline is 0.42 -> diff 0.03 < 0.12
            "cultural/efficiency/wastelessness": 0.68,  # calm baseline is 0.72 -> diff 0.04 < 0.12
        },
        felt_chord_notes={},
        valence=0.52,
        salience=0.55,
        arousal=0.35,
        left_brain_hunch="Quiet developing move.",
        is_hazard=False,
        contrastive_reference_coordinates={
            **base_coords,
            "cultural/efficiency/power": 0.52,  # delta 0.07 < 0.20
            "cultural/efficiency/wastelessness": 0.72,  # delta 0.04 < 0.20
        },
    )
    calm_report = consolidate_sleep_clusters_locally(
        queued_scenes=(near_calm_scene,),
        rules=(),
        learned_leaves=(),
        archives=ThreeBranchArchives(),
        ledger=ContemplationLedger(),
    )
    assert len(calm_report.minted_leaves) == 0

    # 3. Deeper positional mistake (`is_hazard=True`) where 1-3 ply sensors still report high virtues
    #    (`wastelessness=0.73, power=0.91, agency=0.93, order=0.92, needlessness=0.08`) must NEVER
    #    forge a repulsive `hazard_*` rule anchored to healthy virtue coordinates or low needlessness
    healthy_sensor_mistake = UnresolvedSleepScene(
        scene_index=33,
        scenario_text="Played Rxa5 instead of best move Kh6.",
        external_coordinates={
            **base_coords,
            "cultural/efficiency/power": 0.913,
            "cultural/efficiency/wastelessness": 0.73,
            "cultural/freedom/needlessness": 0.08,
            "philosophical/being/agency": 0.929,
            "philosophical/structure/order": 0.92,
            "philosophical/persistence/continuity": 0.95,
        },
        felt_chord_notes={},
        valence=0.45,
        salience=0.75,
        arousal=0.65,
        left_brain_hunch="Pulled the rook out early.",
        is_hazard=True,
        contrastive_reference_coordinates={
            **base_coords,
            "cultural/efficiency/power": 0.65,
            "cultural/efficiency/wastelessness": 0.72,
            "cultural/freedom/needlessness": 0.08,
            "philosophical/being/agency": 0.68,
            "philosophical/structure/order": 0.88,
            "philosophical/persistence/continuity": 0.88,
        },
        game_review_annotation="Game Review [? MISTAKE_?]: Played Rxa5 (-1.10 pawns) instead of best move Kh6.",
    )
    mistake_report = consolidate_sleep_clusters_locally(
        queued_scenes=(healthy_sensor_mistake,),
        rules=(),
        learned_leaves=(),
        archives=ThreeBranchArchives(),
        ledger=ContemplationLedger(),
    )
    assert len(mistake_report.forged_rules) == 0
    assert len(mistake_report.minted_leaves) == 0
    assert 33 in mistake_report.resolved_scene_indices





