"""Unit tests for Steps 1–4:
- Step 1: 12-Leaf Coordinate Mapper + Fuzzy Ruleset -> Experiential -> Reflection (Novelty / Salience)
- Step 2: Confusion Detection -> Psychological -> Contemplation (Clarity / Confidence)
- Step 3: Desperation Feedback Request -> Psychological -> Will (Intention / Execution) -> Teacher Rule
- Step 4: Persistent Ruleset Save / Load & Recognition After Growth
"""

import json
import pytest

from lmft_lab.birth import create_birth_state
from lmft_lab.model_assets import ModelError
from lmft_lab.reflection_classifier import (
    CLARITY_LEAF,
    CONFIDENCE_LEAF,
    EXECUTION_LEAF,
    FACT_LEAF,
    FOUNDATIONAL_LEAVES,
    INTENTION_LEAF,
    NOVELTY_LEAF,
    SALIENCE_LEAF,
    THEORY_LEAF,
    FuzzyConceptRule,
    LearnedLeaf,
    evaluate_reflection,
    generate_executive_response,
    load_learned_leaves,
    load_ruleset,
    request_teacher_growth,
    request_teacher_rule,
    save_ruleset,
    validate_slm_output,
)


class StubModels:
    """Deterministic stub for unit tests (no LLM server required during pytest)."""

    def __init__(self, raw_response: str):
        self.raw_response = raw_response
        self.config = {
            "generation": {
                "context_size": 8192,
                "max_tokens": 1024,
                "temperature": 0,
                "seed": 42,
                "timeout_seconds": 120,
            }
        }

    def generate(self, role, request, prompt_version, validator):
        validated = validator(self.raw_response)
        return validated, {"role": role, "prompt_version": prompt_version, "request_sha256": "stub-sha256"}


def base_coords(**overrides) -> dict[str, float]:
    coords = {leaf: 0.1 for leaf in FOUNDATIONAL_LEAVES}
    coords.update(overrides)
    return coords


def sample_rules():
    return (
        FuzzyConceptRule(
            rule_id="rule-fair-cooperation",
            concept_name="fair_structured_cooperation",
            description="Voluntary, fair, orderly cooperation where participants share resources equitably.",
            target_leaves={
                "cultural/justice/fairness": 0.90,
                "philosophical/structure/order": 0.85,
                "philosophical/structure/chaos": 0.10,
            },
            low=0.60,
            high=0.85,
        ),
    )


def test_fuzzy_concept_rule_coordinate_matching():
    rule = sample_rules()[0]
    matching = base_coords(
        **{
            "cultural/justice/fairness": 0.90,
            "philosophical/structure/order": 0.85,
            "philosophical/structure/chaos": 0.10,
        }
    )
    assert rule.evaluate(matching) == 1.0

    opposite = base_coords(
        **{
            "cultural/justice/fairness": 0.05,
            "philosophical/structure/order": 0.10,
            "philosophical/structure/chaos": 0.95,
        }
    )
    assert rule.evaluate(opposite) == 0.0


def test_familiar_pattern_inhibits_execution_and_needs_no_teacher():
    birth = create_birth_state("ind-1", birth_event_id="birth-1", random_seed=42)
    rules = sample_rules()
    coords = base_coords(
        **{
            "cultural/justice/fairness": 0.90,
            "philosophical/structure/order": 0.85,
            "philosophical/structure/chaos": 0.10,
        }
    )
    raw = json.dumps({
        "summary": "Neighbors organize an orderly, equal-share harvest schedule.",
        "salience_estimate": 0.80,
        "leaf_coordinates": coords,
    })
    res = evaluate_reflection(
        birth,
        "The neighbors agreed on an orderly schedule where every family receives an equal share of the harvest.",
        rules,
        StubModels(raw),
        event_id="ev-1",
    )
    assert res.novelty == 0.0
    assert res.clarity == 1.0
    assert res.confidence == 1.0
    assert res.execution == 0.0
    assert res.is_confused is False
    assert res.needs_teacher is False
    assert res.matched_concept == "fair_structured_cooperation"


def test_unfamiliar_pattern_fires_execution_and_learns_from_teacher(tmp_path):
    birth = create_birth_state("ind-1", birth_event_id="birth-1", random_seed=42)
    rules = sample_rules()
    coords = base_coords(
        **{
            "cultural/efficiency/power": 0.90,
            "cultural/justice/fairness": 0.10,
            "philosophical/structure/order": 0.15,
            "philosophical/structure/chaos": 0.85,
        }
    )
    slm_raw = json.dumps({
        "summary": "An armed gang seizes emergency generators during a blackout.",
        "salience_estimate": 0.90,
        "leaf_coordinates": coords,
    })
    sentence = "During the blackout, an armed gang smashed the depot gates and seized all the emergency generators."

    # 1. Before Teacher help: unfamiliar -> Execution > 0.20 -> needs_teacher is True
    before = evaluate_reflection(birth, sentence, rules, StubModels(slm_raw), event_id="ev-before")
    assert before.novelty == 1.0
    assert before.clarity == 0.0
    assert before.confidence == 0.0
    assert before.intention > 0.40
    assert before.execution > 0.35
    assert before.needs_teacher is True

    # 2. Step 3: Teacher LLM proposes a new FuzzyConceptRule
    teacher_raw = json.dumps({
        "concept_name": "coercive_force_seizure",
        "description": "Seizing resources by force during disorder without regard for fairness.",
        "defining_leaves": [
            {"leaf_id": "cultural/efficiency/power", "target_value": 0.90},
            {"leaf_id": "philosophical/structure/chaos", "target_value": 0.85},
            {"leaf_id": "cultural/justice/fairness", "target_value": 0.10},
        ],
    })
    new_rule, _ = request_teacher_rule(before, rules, StubModels(teacher_raw))
    assert new_rule.concept_name == "coercive_force_seizure"

    # 3. Step 4: Save updated ruleset to disk and reload
    ruleset_file = tmp_path / "ruleset.json"
    save_ruleset(ruleset_file, (*rules, new_rule))
    reloaded_rules = load_ruleset(ruleset_file)
    assert len(reloaded_rules) == 2

    # 4. Re-evaluate with the grown ruleset: now recognized without needing the Teacher!
    after = evaluate_reflection(birth, sentence, reloaded_rules, StubModels(slm_raw), event_id="ev-after")
    assert after.matched_concept == "coercive_force_seizure"
    assert after.novelty == 0.0
    assert after.clarity == 1.0
    assert after.confidence == 1.0
    assert after.execution == 0.0
    assert after.needs_teacher is False


def test_colliding_rules_keep_confidence_high_but_drop_clarity():
    birth = create_birth_state("ind-1", birth_event_id="birth-1", random_seed=42)
    rules = (
        FuzzyConceptRule(
            rule_id="rule-a",
            concept_name="concept_alpha",
            description="Orderly fair schedule.",
            target_leaves={"cultural/justice/fairness": 0.85, "philosophical/structure/order": 0.85},
            low=0.60,
            high=0.85,
        ),
        FuzzyConceptRule(
            rule_id="rule-b",
            concept_name="concept_beta",
            description="Another rule with nearly identical coordinates.",
            target_leaves={"cultural/justice/fairness": 0.80, "philosophical/structure/order": 0.85},
            low=0.60,
            high=0.85,
        ),
    )
    coords = base_coords(**{"cultural/justice/fairness": 0.85, "philosophical/structure/order": 0.85})
    raw = json.dumps({
        "summary": "Overlapping rule test.",
        "salience_estimate": 0.75,
        "leaf_coordinates": coords,
    })
    res = evaluate_reflection(birth, "Test collision.", rules, StubModels(raw), event_id="ev-3")
    assert res.confidence == 1.0
    assert res.clarity == 0.0
    assert res.is_confused is True
    assert "ambiguous_rule_collision" in res.confusion_reason


def test_validation_rejects_missing_leaf_or_out_of_range():
    coords = base_coords()
    coords.pop("cultural/justice/fairness")
    bad = json.dumps({
        "summary": "Incomplete coordinates.",
        "salience_estimate": 0.5,
        "leaf_coordinates": coords,
    })
    with pytest.raises(ModelError, match="leaf_coordinates"):
        validate_slm_output(bad)


def test_blind_spot_grows_learned_child_leaf_and_persists(tmp_path):
    birth = create_birth_state("ind-1", birth_event_id="birth-1", random_seed=42)
    rules = sample_rules()
    # All 12 founding leaves are low (0.20 <= 0.30) -> sensor blind spot!
    low_coords = base_coords(**{"philosophical/structure/chaos": 0.30, "cultural/efficiency/power": 0.30})
    slm_raw_before = json.dumps({
        "summary": "A sudden earthquake shatters a bridge.",
        "salience_estimate": 0.90,
        "leaf_coordinates": low_coords,
    })
    sentence = "A sudden magnitude-seven earthquake shattered the highway bridge."
    before = evaluate_reflection(birth, sentence, rules, StubModels(slm_raw_before), event_id="ev-quake-1")
    assert before.needs_teacher is True

    teacher_raw = json.dumps({
        "concept_name": "natural_environmental_catastrophe",
        "description": "Sudden destructive natural or environmental force.",
        "defining_leaves": [
            {"leaf_id": "philosophical/being/agency", "target_value": 0.10},
        ],
        "new_child_leaves": [
            {
                "parent_branch": "cultural/efficiency",
                "leaf_name": "environmental_force",
                "description": "Destructive or impactful natural, weather, or environmental force.",
            }
        ],
    })
    new_rule, learned_leaves, _ = request_teacher_growth(
        before,
        rules,
        (),
        StubModels(teacher_raw),
    )
    assert len(learned_leaves) == 1
    assert learned_leaves[0].leaf_id == "cultural/efficiency/environmental_force"
    assert "cultural/efficiency/environmental_force" in new_rule.target_leaves

    path = tmp_path / "grown_tree.json"
    save_ruleset(path, (*rules, new_rule), learned_leaves=learned_leaves)
    reloaded_rules = load_ruleset(path)
    reloaded_leaves = load_learned_leaves(path)
    assert len(reloaded_leaves) == 1

    # Now evaluate with the new learned child leaf present
    coords_after = {
        **low_coords,
        "cultural/efficiency/environmental_force": 0.85,
    }
    slm_raw_after = json.dumps({
        "summary": "A flash flood washes away a canyon road.",
        "salience_estimate": 0.85,
        "leaf_coordinates": coords_after,
    })
    after = evaluate_reflection(
        birth,
        "A flash flood tore through the canyon overnight.",
        reloaded_rules,
        StubModels(slm_raw_after),
        event_id="ev-flood-2",
        learned_leaves=reloaded_leaves,
    )
    assert after.matched_concept == "natural_environmental_catastrophe"
    assert after.is_confused is False
    assert after.needs_teacher is False


def test_tier3_recursive_sub_branch_growth_and_specificity_inhibition(tmp_path):
    birth = create_birth_state("ind-1", birth_event_id="birth-1", random_seed=42)
    tier2_leaf = LearnedLeaf(
        parent_branch="philosophical/structure",
        leaf_name="tectonic_collapse",
        description="Sudden structural or geological ground failure.",
    )
    assert tier2_leaf.tier == 2
    tier2_rule = FuzzyConceptRule(
        rule_id="rule-seismic-rupture",
        concept_name="seismic_fault_rupture",
        description="Sudden tectonic fault earthquake.",
        target_leaves={
            "philosophical/structure/tectonic_collapse": 0.80,
            "philosophical/being/agency": 0.05,
        },
        low=0.60,
        high=0.85,
    )
    tier2_competitor = FuzzyConceptRule(
        rule_id="rule-ground-failure",
        concept_name="general_ground_failure",
        description="Overlapping geological failure rule causing a Tier-2 collision.",
        target_leaves={
            "philosophical/structure/tectonic_collapse": 0.80,
            "philosophical/structure/order": 0.10,
        },
        low=0.60,
        high=0.85,
    )
    rules = (tier2_rule, tier2_competitor)
    coords = base_coords(
        **{
            "philosophical/being/agency": 0.05,
            "philosophical/structure/order": 0.10,
            "philosophical/structure/tectonic_collapse": 0.85,
        }
    )
    slm_raw = json.dumps({
        "summary": "Underground limestone caverns dissolved by groundwater collapse into a sinkhole.",
        "salience_estimate": 0.85,
        "leaf_coordinates": coords,
    })
    confused = evaluate_reflection(
        birth,
        "Centuries of acidic groundwater dissolved the limestone bedrock until a sinkhole swallowed the road.",
        rules,
        StubModels(slm_raw),
        event_id="ev-tier2-collision",
        learned_leaves=(tier2_leaf,),
    )
    assert confused.is_confused is True
    assert "ambiguous_rule_collision" in confused.confusion_reason

    teacher_raw = json.dumps({
        "concept_name": "karst_sinkhole_subsidence",
        "description": "Gradual groundwater dissolution of bedrock causing a localized sinkhole collapse.",
        "defining_leaves": [
            {"leaf_id": "philosophical/being/agency", "target_value": 0.05},
        ],
        "new_child_leaves": [
            {
                "parent_branch": "philosophical/structure/tectonic_collapse",
                "leaf_name": "sinkhole_subsidence",
                "description": "Sinkhole subsidence caused by underground water dissolving soluble bedrock.",
            }
        ],
    })
    tier3_rule, updated_leaves, _ = request_teacher_growth(
        confused,
        rules,
        (tier2_leaf,),
        StubModels(teacher_raw),
    )
    assert len(updated_leaves) == 2
    tier3_leaf = updated_leaves[1]
    assert tier3_leaf.leaf_id == "philosophical/structure/tectonic_collapse/sinkhole_subsidence"
    assert tier3_leaf.tier == 3
    assert tier3_rule.tier == 3

    # When the Tier-3 sub-leaf fires (0.85), it inhibits the Tier-2 rules (1.0 -> 0.50) and wins cleanly
    coords_with_tier3 = {
        **coords,
        "philosophical/structure/tectonic_collapse/sinkhole_subsidence": 0.85,
    }
    after = evaluate_reflection(
        birth,
        "Groundwater dissolved the limestone cavern until a deep sinkhole opened beneath the highway.",
        (tier2_rule, tier3_rule),
        StubModels(json.dumps({
            "summary": "Sinkhole collapse.",
            "salience_estimate": 0.85,
            "leaf_coordinates": coords_with_tier3,
        })),
        event_id="ev-tier3-win",
        learned_leaves=updated_leaves,
    )
    assert after.matched_concept == "karst_sinkhole_subsidence"
    assert after.is_confused is False
    assert after.rule_activations["karst_sinkhole_subsidence"] == 1.0
    assert after.rule_activations["seismic_fault_rupture"] == 0.5


def test_continuous_memory_state_carries_context_across_followup_and_resets_on_shift():
    birth = create_birth_state("ind-1", birth_event_id="birth-1", random_seed=42)
    rules = sample_rules()

    # Turn 1: Clear match on fair_structured_cooperation
    turn1_coords = base_coords(
        **{
            "cultural/justice/fairness": 0.90,
            "philosophical/structure/order": 0.85,
            "philosophical/persistence/continuity": 0.80,
            "philosophical/structure/chaos": 0.10,
        }
    )
    turn1 = evaluate_reflection(
        birth,
        "The neighbors agreed on an orderly schedule where every family receives an equal share of the harvest.",
        rules,
        StubModels(json.dumps({
            "summary": "Orderly equal harvest schedule.",
            "salience_estimate": 0.80,
            "leaf_coordinates": turn1_coords,
        })),
        event_id="ev-seq-1",
    )
    assert turn1.matched_concept == "fair_structured_cooperation"
    assert turn1.valence > 0.30
    assert turn1.arousal > 0.50
    assert turn1.retention == 0.0

    # Turn 2: Brief follow-up sentence with attenuated fairness/order (0.50) that fails statelessly
    turn2_coords = base_coords(
        **{
            "cultural/justice/fairness": 0.50,
            "philosophical/structure/order": 0.50,
            "philosophical/persistence/continuity": 0.85,
            "philosophical/structure/chaos": 0.10,
        }
    )
    turn2_raw = json.dumps({
        "summary": "Later that afternoon, they quietly stacked the remaining crates according to the list.",
        "salience_estimate": 0.55,
        "leaf_coordinates": turn2_coords,
    })
    stateless_turn2 = evaluate_reflection(
        birth,
        "Later that afternoon, they quietly stacked the remaining crates according to the list.",
        rules,
        StubModels(turn2_raw),
        event_id="ev-seq-2-stateless",
    )
    assert stateless_turn2.is_confused is True

    stateful_turn2 = evaluate_reflection(
        birth,
        "Later that afternoon, they quietly stacked the remaining crates according to the list.",
        rules,
        StubModels(turn2_raw),
        event_id="ev-seq-2-stateful",
        memory_state=turn1.next_memory_state,
    )
    assert stateful_turn2.retention > 0.50
    assert stateful_turn2.resonance >= 0.55
    assert stateful_turn2.is_confused is False
    assert stateful_turn2.matched_concept == "fair_structured_cooperation"

    # Turn 3: Sudden chaotic interruption drops resonance and does NOT carry over prior cooperation
    turn3_coords = base_coords(
        **{
            "cultural/efficiency/power": 0.95,
            "cultural/justice/fairness": 0.05,
            "philosophical/structure/order": 0.05,
            "philosophical/structure/chaos": 0.95,
            "philosophical/persistence/continuity": 0.05,
            "philosophical/persistence/discreteness": 0.95,
        }
    )
    stateful_turn3 = evaluate_reflection(
        birth,
        "Suddenly the main transformer exploded, plunging the district into chaos.",
        rules,
        StubModels(json.dumps({
            "summary": "Sudden transformer explosion.",
            "salience_estimate": 0.95,
            "leaf_coordinates": turn3_coords,
        })),
        event_id="ev-seq-3-interrupt",
        memory_state=stateful_turn2.next_memory_state,
    )
    assert stateful_turn3.resonance < 0.55
    assert stateful_turn3.matched_concept is None
    assert stateful_turn3.is_confused is True
    assert stateful_turn3.executive_decision.mode == "ESCALATE_TO_TEACHER"


def test_executive_decision_modes_and_truth_leaves_across_24_founding_leaves():
    birth = create_birth_state("ind-exec", birth_event_id="birth-exec", random_seed=42)
    rules = sample_rules()

    # 1. Direct Clear Match -> Fact >= 0.50, Mode == COMMIT_FAST
    turn1_coords = base_coords(
        **{
            "cultural/justice/fairness": 0.90,
            "philosophical/structure/order": 0.85,
            "philosophical/persistence/continuity": 0.80,
            "philosophical/structure/chaos": 0.10,
        }
    )
    turn1 = evaluate_reflection(
        birth,
        "The neighbors agreed on an orderly schedule where every family receives an equal share.",
        rules,
        StubModels(json.dumps({
            "summary": "Equal-share orderly schedule.",
            "salience_estimate": 0.80,
            "leaf_coordinates": turn1_coords,
        })),
        event_id="ev-exec-1",
    )
    assert turn1.fact == 1.0
    assert turn1.theory == 0.0
    assert turn1.executive_decision.mode == "COMMIT_FAST"
    assert turn1.executive_decision.active_concept == "fair_structured_cooperation"
    assert turn1.propagation_result.activations[FACT_LEAF] == 1.0
    assert turn1.propagation_result.activations[THEORY_LEAF] == 0.0

    # 2. Working-Memory Carryover Turn -> Fact == 0.0, Theory >= 0.50, Mode == VERIFY_HYPOTHESIS
    turn2_coords = base_coords(
        **{
            "cultural/justice/fairness": 0.50,
            "philosophical/structure/order": 0.50,
            "philosophical/persistence/continuity": 0.85,
            "philosophical/structure/chaos": 0.10,
        }
    )
    turn2 = evaluate_reflection(
        birth,
        "Later that afternoon, they quietly stacked the remaining crates.",
        rules,
        StubModels(json.dumps({
            "summary": "Stacking crates afterward.",
            "salience_estimate": 0.50,
            "leaf_coordinates": turn2_coords,
        })),
        event_id="ev-exec-2",
        memory_state=turn1.next_memory_state,
    )
    assert turn2.fact == 0.0
    assert turn2.theory >= 0.50
    assert turn2.executive_decision.mode == "VERIFY_HYPOTHESIS"
    assert turn2.executive_decision.active_concept == "fair_structured_cooperation"

    # Closed-loop executive response generation
    exec_reply, _ = generate_executive_response(
        turn2,
        StubModels(json.dumps({
            "executive_mode": "VERIFY_HYPOTHESIS",
            "active_concept": "fair_structured_cooperation",
            "spoken_response": "This appears to be a continuation of fair structured cooperation from earlier; please confirm before proceeding.",
        })),
    )
    assert exec_reply["executive_mode"] == "VERIFY_HYPOTHESIS"
    assert exec_reply["active_concept"] == "fair_structured_cooperation"



