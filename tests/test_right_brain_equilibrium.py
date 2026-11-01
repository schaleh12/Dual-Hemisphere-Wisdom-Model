"""Automated tests for the Right-Brain Dominant 3-Bridge Equilibrium Loop and Settled Diary."""

import pytest

from lmft_lab.birth import create_birth_state
from lmft_lab.model_assets import ModelError
from lmft_lab.reflection_classifier import (
    AROUSAL_LEAF,
    SALIENCE_LEAF,
    VALENCE_LEAF,
    FuzzyConceptRule,
    LearnedLeaf,
)
from lmft_lab.right_brain_equilibrium import (
    TRUTH_PHILOSOPHICAL_LEAVES,
    WILL_CULTURAL_LEAVES,
    SettledDiary,
    ThreeBridgeSensors,
    load_diary,
    save_diary,
    settle_scenario,
    validate_plain_english_left_brain_reply,
)


def _birth():
    return create_birth_state("rb-test", birth_event_id="birth-1", random_seed=7)


def _uniform(keys: tuple[str, ...], val: float = 0.10, **overrides: float) -> dict[str, float]:
    out = {k: val for k in keys}
    for short_or_full, v in overrides.items():
        matched = [k for k in keys if k == short_or_full or k.endswith("/" + short_or_full)]
        if not matched:
            raise KeyError(f"unknown key override: {short_or_full}")
        out[matched[0]] = v
    return out


def _sample_rules() -> tuple[FuzzyConceptRule, ...]:
    return (
        FuzzyConceptRule(
            rule_id="rule-fair-coop",
            concept_name="fair_structured_cooperation",
            description="Shared orderly cooperation with high fairness and low chaos.",
            target_leaves={
                "cultural/justice/fairness": 0.70,
                "philosophical/structure/order": 0.65,
                "philosophical/structure/chaos": 0.15,
            },
        ),
        FuzzyConceptRule(
            rule_id="rule-raid",
            concept_name="violent_resource_seizure",
            description="Forceful high-chaos seizure of shared resources.",
            target_leaves={
                "cultural/efficiency/power": 0.75,
                "cultural/justice/fairness": 0.15,
                "philosophical/structure/chaos": 0.70,
            },
        ),
    )


def test_untangled_will_at_ease_vs_wordless_reflex_without_waking_left_brain():
    birth = _birth()
    rules = _sample_rules()

    def left_brain_should_not_be_called(*_args, **_kwargs):
        raise AssertionError("Left Brain should stay asleep during AT_EASE and WORDLESS_REFLEX")

    class GuardLeftBrain:
        respond_to_tug = staticmethod(left_brain_should_not_be_called)

    # 1. Calm background moment -> AT_EASE in 1 pass, 0 Left-Brain calls
    calm_sensors = ThreeBridgeSensors(
        will_cultural=lambda _t, _c: _uniform(WILL_CULTURAL_LEAVES, 0.15, needlessness=0.75, fairness=0.40),
        truth_philosophical=lambda _t, _c: _uniform(TRUTH_PHILOSOPHICAL_LEAVES, 0.15, order=0.35, chaos=0.05),
        contemplation_experiential=lambda _t: {
            SALIENCE_LEAF: 0.22,
            VALENCE_LEAF: 0.65,
            AROUSAL_LEAF: 0.18,
        },
    )
    calm_res = settle_scenario(
        birth,
        "Morning mist clears slowly over the quiet stone overflow trough.",
        calm_sensors,
        rules,
        left_brain=GuardLeftBrain(),
    )
    assert calm_res.settled is True
    assert calm_res.outcome == "AT_EASE"
    assert calm_res.left_brain_calls == 0
    assert len(calm_res.passes) == 1
    assert calm_res.final_vibe.execution < 0.25

    # 2. Urgent known threat -> WORDLESS_REFLEX in 1 pass, Execution > 0.25, 0 Left-Brain calls
    raid_sensors = ThreeBridgeSensors(
        will_cultural=lambda _t, _c: _uniform(
            WILL_CULTURAL_LEAVES, 0.15, power=0.85, fairness=0.10, needlessness=0.05
        ),
        truth_philosophical=lambda _t, _c: _uniform(
            TRUTH_PHILOSOPHICAL_LEAVES, 0.20, agency=0.80, chaos=0.80, order=0.15
        ),
        contemplation_experiential=lambda _t: {
            SALIENCE_LEAF: 0.85,
            VALENCE_LEAF: 0.15,
            AROUSAL_LEAF: 0.90,
        },
    )
    raid_res = settle_scenario(
        birth,
        "Armed riders ram the depot latch and haul out the winter grain sacks.",
        raid_sensors,
        rules,
        left_brain=GuardLeftBrain(),
    )
    assert raid_res.settled is True
    assert raid_res.outcome == "WORDLESS_REFLEX"
    assert raid_res.final_vibe.active_concept == "violent_resource_seizure"
    assert raid_res.final_vibe.execution >= 0.25
    assert raid_res.left_brain_calls == 0


def test_branch_gated_child_drawers_across_separate_bridges():
    birth = _birth()
    cult_child = LearnedLeaf("cultural/justice", "transactional_fraud", "Bribery to falsify records.")
    phil_child = LearnedLeaf("philosophical/structure", "tectonic_collapse", "Structural fracture or collapse.")
    tier3_child = LearnedLeaf(
        "philosophical/structure/tectonic_collapse",
        "human_intervention_in_collapse",
        "Deliberate rescue bracing inside a structural collapse.",
    )
    learned = (cult_child, phil_child, tier3_child)

    def will_cult(_text: str, children: tuple[LearnedLeaf, ...]):
        # Cultural/justice is active (honor=0.40 >= 0.25), so transactional_fraud opens
        base = _uniform(WILL_CULTURAL_LEAVES, 0.10, honor=0.40)
        for c in children:
            base[c.leaf_id] = 0.85
        return base

    def truth_phil(_text: str, children: tuple[LearnedLeaf, ...]):
        # Philosophical/structure is completely quiet (0.10 < 0.25), so tectonic_collapse + tier3 skip
        base = _uniform(TRUTH_PHILOSOPHICAL_LEAVES, 0.10, agency=0.35)
        for c in children:
            base[c.leaf_id] = 0.90
        return base

    sensors = ThreeBridgeSensors(
        will_cultural=will_cult,
        truth_philosophical=truth_phil,
        contemplation_experiential=lambda _t: {
            SALIENCE_LEAF: 0.65,
            VALENCE_LEAF: 0.25,
            AROUSAL_LEAF: 0.60,
        },
    )
    res = settle_scenario(
        birth,
        "An envelope of bills slides across the ledger desk to erase a missing fuel pallet.",
        sensors,
        _sample_rules(),
        learned_leaves=learned,
    )
    assert "cultural/justice/transactional_fraud" in res.final_vibe.queried_child_leaves
    assert "philosophical/structure/tectonic_collapse" in res.final_vibe.skipped_child_leaves
    assert (
        "philosophical/structure/tectonic_collapse/human_intervention_in_collapse"
        in res.final_vibe.skipped_child_leaves
    )


def test_equilibrium_loop_and_diary_resolve_ambiguous_scene_and_persist(tmp_path):
    birth = _birth()
    rules = _sample_rules()

    # Scene 1: Hoarding wall goes up -> unsettled tension recorded into SettledDiary
    class HonestLeftBrain:
        def __init__(self):
            self.received_packets = []

        def respond_to_tug(self, scenario_text: str, tug_packet):
            self.received_packets.append(dict(tug_packet))
            if "yellow blocks" in scenario_text:
                return (
                    "The builder has sealed the shared public spring behind yellow blocks, "
                    "turning orderly care into exclusionary hoarding that shuts travelers out."
                )
            # Scene 2: sees Scene 1's diary note and clarifies that the cracked wall restores shared water
            assert any("sealed the shared public spring" in (d.get("note") or "") for d in tug_packet["settled_diary"])
            return (
                "The hoarding wall from the earlier enclosure gave way, restoring shared orderly "
                "water flow so thirsty travelers and animals can drink fairly together again."
            )

    lb = HonestLeftBrain()

    def will_cult(text: str, _c):
        if "yellow blocks" in text or "exclusionary hoarding" in text:
            return _uniform(WILL_CULTURAL_LEAVES, 0.15, fairness=0.10, wastelessness=0.50)
        if "restoring shared orderly" in text:
            return _uniform(WILL_CULTURAL_LEAVES, 0.25, fairness=0.95, power=0.65, needlessness=0.10)
        # Raw Scene 2 before diary context: genuinely ambiguous fairness=0.35
        return _uniform(WILL_CULTURAL_LEAVES, 0.20, fairness=0.35, power=0.55, needlessness=0.10)

    def truth_phil(text: str, _c):
        if "yellow blocks" in text or "exclusionary hoarding" in text:
            return _uniform(TRUTH_PHILOSOPHICAL_LEAVES, 0.20, order=0.75, chaos=0.10)
        if "restoring shared orderly" in text:
            return _uniform(TRUTH_PHILOSOPHICAL_LEAVES, 0.25, order=0.85, chaos=0.10, continuity=0.75)
        # Raw Scene 2 before diary context: ambiguous order=0.45, chaos=0.25
        return _uniform(TRUTH_PHILOSOPHICAL_LEAVES, 0.20, order=0.45, chaos=0.25, continuity=0.30)

    sensors = ThreeBridgeSensors(
        will_cultural=will_cult,
        truth_philosophical=truth_phil,
        contemplation_experiential=lambda _t: {
            SALIENCE_LEAF: 0.72,
            VALENCE_LEAF: 0.45,
            AROUSAL_LEAF: 0.70,
        },
    )

    scene1 = settle_scenario(
        birth,
        "He laid dressed yellow blocks across the doorway from within, sealing the basin.",
        sensors,
        rules,
        scene_index=1,
        left_brain=lb,
        diary=SettledDiary(),
    )
    assert scene1.settled is False
    assert scene1.outcome == "UNRESOLVED_TENSION"
    assert scene1.needs_new_instinct is True
    assert scene1.final_conscious_note is not None

    diary_file = tmp_path / "settled_diary.json"
    save_diary(diary_file, scene1.next_diary)
    reloaded_diary = load_diary(diary_file)
    assert len(reloaded_diary.entries) == 1

    # Scene 2: Pass 1 is confused/ambiguous -> tugs Left Brain with Diary -> Pass 2 resolves cleanly!
    scene2 = settle_scenario(
        birth,
        "At dawn the wall split open and meltwater braided freely out to three stray kids.",
        sensors,
        rules,
        scene_index=2,
        left_brain=lb,
        prior_memory=scene1.next_memory,
        diary=reloaded_diary,
    )
    assert scene2.passes[0].vibe.posture == "TUG_LEFT_BRAIN"
    assert scene2.settled is True
    assert scene2.outcome == "WORDLESS_REFLEX"
    assert scene2.final_vibe.active_concept == "fair_structured_cooperation"
    assert scene2.left_brain_calls == 1
    assert len(scene2.passes) == 2


def test_sensory_reality_anchor_blocks_left_brain_from_whitewashing_stark_injustice():
    birth = _birth()
    rules = _sample_rules()

    class RationalizingLeftBrain:
        def respond_to_tug(self, _scenario_text: str, _tug_packet):
            # Attempts to excuse stark unfairness with flowery rationalizations
            return "Everything here is completely fair, cooperative, and harmonious for everyone."

    def will_cult(text: str, _c):
        if "completely fair" in text:
            return _uniform(WILL_CULTURAL_LEAVES, 0.30, fairness=0.95)
        # Raw sensory scene has stark unfairness (0.10)
        return _uniform(WILL_CULTURAL_LEAVES, 0.20, fairness=0.10, power=0.70)

    def truth_phil(text: str, _c):
        if "completely fair" in text:
            return _uniform(TRUTH_PHILOSOPHICAL_LEAVES, 0.25, order=0.85, chaos=0.05)
        return _uniform(TRUTH_PHILOSOPHICAL_LEAVES, 0.20, order=0.78, chaos=0.10)

    sensors = ThreeBridgeSensors(
        will_cultural=will_cult,
        truth_philosophical=truth_phil,
        contemplation_experiential=lambda _t: {
            SALIENCE_LEAF: 0.75,
            VALENCE_LEAF: 0.30,
            AROUSAL_LEAF: 0.70,
        },
    )

    res = settle_scenario(
        birth,
        "Dry hooves scraped outside in the heat while the keeper locked the water behind stone.",
        sensors,
        rules,
        left_brain=RationalizingLeftBrain(),
    )
    # Cognitive-dissonance anchor (0.75 * 0.10 + 0.25 * 0.95 = 0.3125) blocks whitewashing
    assert res.final_vibe.coordinates["cultural/justice/fairness"] == pytest.approx(0.3125)
    assert any("cognitive_dissonance" in r for r in res.final_vibe.tug_reasons)
    assert res.settled is False
    assert res.outcome == "UNRESOLVED_TENSION"
    assert res.needs_new_instinct is True


def test_plain_english_validator_strips_think_tags_and_rejects_json_dumps():
    cleaned = validate_plain_english_left_brain_reply(
        "<think>internal scratch</think>  The keeper's rigid rules shut out thirsty travelers. "
    )
    assert cleaned == "The keeper's rigid rules shut out thirsty travelers."

    with pytest.raises(ModelError):
        validate_plain_english_left_brain_reply('{"fairness": 0.55, "order": 0.80}')
