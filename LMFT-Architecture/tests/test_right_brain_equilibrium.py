"""Automated tests for Right-Brain 3-Bridge Equilibrium, Self-Registry, and Will->Theory Circuit."""

import pytest

from lmft_architecture.birth import create_birth_state
from lmft_architecture.model_assets import ModelError
from lmft_architecture.reflection_classifier import (
    AROUSAL_LEAF,
    SALIENCE_LEAF,
    VALENCE_LEAF,
    FuzzyConceptRule,
    LearnedLeaf,
)
from lmft_architecture.right_brain_equilibrium import (
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
    assert calm_res.final_vibe.self_registry.internal_thought_tag == "NONE"
    assert calm_res.final_vibe.self_registry.epistemic_state == "QUIET_EQUILIBRIUM"
    assert calm_res.final_vibe.synthesis.feeler.stance == "CALM_ABSORPTION"
    assert calm_res.final_vibe.synthesis.doer.stance == "HOLD_STEADY"
    assert calm_res.final_vibe.execution < 0.25

    # 2. Urgent known threat -> WORDLESS_REFLEX in 1 pass, Execution >= 0.25, 0 Left-Brain calls
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
    assert raid_res.final_vibe.self_registry.epistemic_state == "GROUNDED_EXTERNAL_FACT"
    assert raid_res.final_vibe.synthesis.doer.stance == "ACT_DECISIVELY"
    assert raid_res.final_vibe.synthesis.thinker.stance == "VERIFIED_EXTERNAL_FACT"
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
        base = _uniform(WILL_CULTURAL_LEAVES, 0.10, honor=0.40)
        for c in children:
            base[c.leaf_id] = 0.85
        return base

    def truth_phil(_text: str, children: tuple[LearnedLeaf, ...]):
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
    assert res.final_vibe.self_registry.known_instinct_count == 2
    assert res.final_vibe.self_registry.learned_drawer_count == 3


def test_will_outbound_to_theory_inbound_resolves_ambiguous_scene_without_mutating_external_fact(tmp_path):
    birth = _birth()
    rules = _sample_rules()

    class HonestLeftBrain:
        def __init__(self):
            self.received_packets = []

        def respond_to_tug(self, scenario_text: str, tug_packet):
            self.received_packets.append(dict(tug_packet))
            assert tug_packet["sender_branch"] == "psychological/will"
            assert tug_packet["target_channel"] == "psychological/truth/theory"
            assert "will_opinion" in tug_packet
            assert "interjected_emotion" in tug_packet
            assert tug_packet["self_registry"]["known_instinct_count"] == 2
            if "yellow blocks" in scenario_text:
                return (
                    "The builder has sealed the shared public spring behind yellow blocks, "
                    "turning orderly care into exclusionary hoarding that shuts travelers out."
                )
            assert any("sealed the shared public spring" in (d.get("brief_gist") or "") for d in tug_packet["settled_diary"])
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
        return _uniform(WILL_CULTURAL_LEAVES, 0.20, fairness=0.35, power=0.55, needlessness=0.10)

    def truth_phil(text: str, _c):
        if "yellow blocks" in text or "exclusionary hoarding" in text:
            return _uniform(TRUTH_PHILOSOPHICAL_LEAVES, 0.20, order=0.75, chaos=0.10)
        if "restoring shared orderly" in text:
            return _uniform(TRUTH_PHILOSOPHICAL_LEAVES, 0.25, order=0.85, chaos=0.10, continuity=0.75)
        return _uniform(TRUTH_PHILOSOPHICAL_LEAVES, 0.20, order=0.45, chaos=0.25, continuity=0.30)

    sensors = ThreeBridgeSensors(
        will_cultural=will_cult,
        truth_philosophical=truth_phil,
        contemplation_experiential=lambda t: {
            SALIENCE_LEAF: 0.72,
            VALENCE_LEAF: 0.75 if "restoring shared orderly" in t else 0.30,
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
    assert scene1.final_vibe.self_registry.epistemic_state == "KNOWN_INSTINCT_BLIND_SPOT"
    assert scene1.final_vibe.synthesis.doer.stance == "CONTEST_EXCLUSION"
    assert scene1.final_vibe.synthesis.feeler.stance == "FELT_UNEASE"

    diary_file = tmp_path / "settled_diary.json"
    save_diary(diary_file, scene1.next_diary)
    reloaded_diary = load_diary(diary_file)
    assert len(reloaded_diary.entries) == 1
    assert reloaded_diary.entries[0].epistemic_state == "KNOWN_INSTINCT_BLIND_SPOT"

    # Scene 2: Pass 1 is ambiguous -> Will tugs Left Brain -> Theory resolves WITHOUT overwriting raw Fact!
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
    assert scene2.outcome == "THEORY_SYNTHESIS"
    assert scene2.final_vibe.active_concept == "fair_structured_cooperation"
    # External world coordinates & Fact remain 100% untouched by internal Left-Brain thoughts!
    assert scene2.final_vibe.coordinates["cultural/justice/fairness"] == pytest.approx(0.35)
    assert scene2.final_vibe.theory_coordinates["cultural/justice/fairness"] == pytest.approx(0.95)
    assert scene2.final_vibe.fact < 0.20
    assert scene2.final_vibe.theory >= 0.60
    assert scene2.final_vibe.self_registry.internal_thought_tag == "INTERNAL_LEFT_BRAIN_THEORY"
    assert scene2.final_vibe.self_registry.epistemic_state == "THEORY_GUIDED_SYNTHESIS"
    assert scene2.final_vibe.synthesis.thinker.stance == "CANDIDATE_THEORY_FIT"


def test_self_registry_vetoes_left_brain_rationalization_and_keeps_raw_fact_intact():
    birth = _birth()
    rules = _sample_rules()

    class RationalizingLeftBrain:
        def respond_to_tug(self, _scenario_text: str, _tug_packet):
            return "Everything here is completely fair, cooperative, and harmonious for everyone."

    def will_cult(text: str, _c):
        if "completely fair" in text:
            return _uniform(WILL_CULTURAL_LEAVES, 0.30, fairness=0.95)
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
    # External Fairness stays locked at 0.10 (never diluted by rationalizing words!), Theory is vetoed to 0.0
    assert res.final_vibe.coordinates["cultural/justice/fairness"] == pytest.approx(0.10)
    assert res.final_vibe.theory_coordinates["cultural/justice/fairness"] == pytest.approx(0.95)
    assert res.final_vibe.theory == pytest.approx(0.0)
    assert res.final_vibe.self_registry.epistemic_state == "SELF_RATIONALIZATION_DISSONANCE"
    assert any("cognitive_dissonance" in r for r in res.final_vibe.tug_reasons)
    assert res.settled is False
    assert res.outcome == "UNRESOLVED_TENSION"
    assert res.needs_new_instinct is True


def test_past_diary_mentions_in_left_brain_reply_never_pollute_current_external_facts():
    birth = _birth()
    admin_rot = LearnedLeaf("cultural/efficiency", "administrative_rot", "Bureaucratic neglect or red tape.")
    rules = _sample_rules() + (
        FuzzyConceptRule(
            rule_id="rule-rot",
            concept_name="bureaucratic_negligence",
            description="Administrative rot and low fairness.",
            target_leaves={
                "cultural/efficiency/administrative_rot": 0.70,
                "cultural/justice/fairness": 0.25,
            },
        ),
    )

    class PastReferencingLeftBrain:
        def respond_to_tug(self, _scenario_text: str, _tug_packet):
            # Mentions past administrative rot while explaining current thaw
            return "Unlike the earlier administrative rot that shut out the herder, the water now flows."

    def will_cult(text: str, children: tuple[LearnedLeaf, ...]):
        if "administrative rot" in text:
            base = _uniform(WILL_CULTURAL_LEAVES, 0.20, fairness=0.15, honor=0.30)
            for c in children:
                base[c.leaf_id] = 0.85
            return base
        # Current external scene has zero administrative rot and moderate fairness=0.45
        base = _uniform(WILL_CULTURAL_LEAVES, 0.20, fairness=0.45, honor=0.35)
        for c in children:
            base[c.leaf_id] = 0.0
        return base

    def truth_phil(_text: str, children: tuple[LearnedLeaf, ...]):
        base = _uniform(TRUTH_PHILOSOPHICAL_LEAVES, 0.20, order=0.40, chaos=0.25)
        for c in children:
            base[c.leaf_id] = 0.0
        return base

    sensors = ThreeBridgeSensors(
        will_cultural=will_cult,
        truth_philosophical=truth_phil,
        contemplation_experiential=lambda _t: {
            SALIENCE_LEAF: 0.68,
            VALENCE_LEAF: 0.45,
            AROUSAL_LEAF: 0.55,
        },
    )

    res = settle_scenario(
        birth,
        "Meltwater braids quietly through the gap into the open grass.",
        sensors,
        rules,
        learned_leaves=(admin_rot,),
        left_brain=PastReferencingLeftBrain(),
    )
    # External coordinates remain untouched: administrative_rot stays 0.0 and fairness stays 0.45!
    assert res.final_vibe.coordinates["cultural/efficiency/administrative_rot"] == pytest.approx(0.0)
    assert res.final_vibe.coordinates["cultural/justice/fairness"] == pytest.approx(0.45)
    assert res.final_vibe.active_concept != "bureaucratic_negligence"


def test_feeler_unease_vetoes_tone_deaf_theory_when_valence_stays_bleak():
    birth = _birth()
    rules = _sample_rules()

    class ToneDeafLeftBrain:
        def respond_to_tug(self, _scenario_text: str, _tug_packet):
            # Pitches a high-fairness/order theory, but emotional tone stays bleak (0.20)
            return "Strict rules maintain orderly fairness by turning away outsiders at the gate."

    def will_cult(text: str, _c):
        if "orderly fairness" in text:
            return _uniform(WILL_CULTURAL_LEAVES, 0.20, fairness=0.90, power=0.45)
        return _uniform(WILL_CULTURAL_LEAVES, 0.20, fairness=0.35, power=0.45)

    def truth_phil(text: str, _c):
        if "orderly fairness" in text:
            return _uniform(TRUTH_PHILOSOPHICAL_LEAVES, 0.20, order=0.80, chaos=0.10)
        return _uniform(TRUTH_PHILOSOPHICAL_LEAVES, 0.20, order=0.45, chaos=0.25)

    sensors = ThreeBridgeSensors(
        will_cultural=will_cult,
        truth_philosophical=truth_phil,
        contemplation_experiential=lambda _t: {
            SALIENCE_LEAF: 0.72,
            VALENCE_LEAF: 0.20,  # Feeler remains in FELT_UNEASE (<= 0.35)!
            AROUSAL_LEAF: 0.65,
        },
    )

    res = settle_scenario(
        birth,
        "Three carved rules on the lintel turn away a thirsty old herder at dusk.",
        sensors,
        rules,
        left_brain=ToneDeafLeftBrain(),
    )
    assert res.final_vibe.synthesis.feeler.stance == "FELT_UNEASE"
    assert "feeler_unease_vetoes_tone_deaf_theory" in res.final_vibe.tug_reasons
    assert res.settled is False
    assert res.outcome == "UNRESOLVED_TENSION"


def test_feeler_grows_complex_experiential_drawers_and_forms_poetic_harmony_chords():
    birth = _birth()
    bittersweet = LearnedLeaf(
        "experiential/phase",
        "bittersweet_release",
        "Simultaneous fracture of an enclosure and life-giving relief.",
    )
    earned_thaw = LearnedLeaf(
        "experiential/phase/bittersweet_release",
        "earned_thaw_after_enclosure",
        "Water returning to shared flow after stagnant isolation cracks open.",
    )
    learned = (bittersweet, earned_thaw)

    poetic_chord_rule = FuzzyConceptRule(
        rule_id="rule-earned-thaw",
        concept_name="restorative_fracture_into_flow",
        description="A stagnant barrier cracking open into bittersweet, life-giving shared release.",
        target_leaves={
            "cultural/justice/fairness": 0.70,
            "philosophical/persistence/continuity": 0.70,
            "experiential/phase/bittersweet_release/earned_thaw_after_enclosure": 0.80,
        },
    )
    rules = _sample_rules() + (poetic_chord_rule,)

    def feeler(_text: str, children: tuple[LearnedLeaf, ...] = ()):
        out = {
            SALIENCE_LEAF: 0.78,
            VALENCE_LEAF: 0.68,
            AROUSAL_LEAF: 0.72,
        }
        for c in children:
            out[c.leaf_id] = 0.88
        return out

    sensors = ThreeBridgeSensors(
        will_cultural=lambda _t, _c: _uniform(WILL_CULTURAL_LEAVES, 0.20, fairness=0.75, power=0.55, needlessness=0.15),
        truth_philosophical=lambda _t, _c: _uniform(TRUTH_PHILOSOPHICAL_LEAVES, 0.20, continuity=0.78, order=0.35, chaos=0.30),
        contemplation_experiential=feeler,
    )

    res = settle_scenario(
        birth,
        "At dawn the yellow wall split open and meltwater braided freely out to three stray kids.",
        sensors,
        rules,
        learned_leaves=learned,
    )
    assert "experiential/phase/bittersweet_release" in res.final_vibe.queried_child_leaves
    assert "experiential/phase/bittersweet_release/earned_thaw_after_enclosure" in res.final_vibe.queried_child_leaves
    assert res.settled is True
    assert res.final_vibe.active_concept == "restorative_fracture_into_flow"
    assert res.final_vibe.synthesis.conflicts == ()
    assert res.final_vibe.synthesis.harmony == pytest.approx(0.70)


def test_plain_english_validator_strips_think_tags_and_rejects_json_dumps():
    cleaned = validate_plain_english_left_brain_reply(
        "<think>internal scratch</think>  The keeper's rigid rules shut out thirsty travelers. "
    )
    assert cleaned == "The keeper's rigid rules shut out thirsty travelers."

    with pytest.raises(ModelError):
        validate_plain_english_left_brain_reply('{"fairness": 0.55, "order": 0.80}')


def test_six_core_evolution_upgrades_polarity_softmin_experiential_rollout_and_plasticity(tmp_path):
    """Verify Upgrades #1–#6: Rule Polarity, Soft-Min Tie-Breaker, 6-Leaf Experiential Parity, Pre-Gap Rollout, and Callosum Micro-Plasticity."""
    from lmft_architecture.gap_trajectory_schema import ContrastiveGapPair
    from lmft_architecture.reflection_classifier import (
        NOVELTY_LEAF,
        RESONANCE_LEAF,
        RETENTION_LEAF,
        _evaluate_rules_with_inhibition,
        compute_contemplation,
    )
    from lmft_architecture.right_brain_equilibrium import compile_right_brain_vibe
    from lmft_architecture.subliminal_gap_splice import SubliminalChordVector, SubliminalGapProjector
    from lmft_architecture.three_legged_runtime import select_pre_gap_branch_chord_rollout
    from lmft_architecture.train_left_brain_lora import update_projector_on_harvested_gap_pairs

    birth = _birth()

    # 1. Upgrade #1: Rule Polarity ('attractive' vs 'repulsive')
    att_rule = FuzzyConceptRule(
        rule_id="r-att",
        concept_name="steady_bastion",
        description="Attractive rule.",
        target_leaves={"philosophical/structure/order": 0.75, "cultural/justice/fairness": 0.70},
    )
    rep_rule = FuzzyConceptRule(
        rule_id="r-rep",
        concept_name="hazard_overextension_trap",
        description="Repulsive hazard rule.",
        target_leaves={"cultural/efficiency/power": 0.80, "philosophical/structure/chaos": 0.75},
    )
    assert att_rule.polarity == "attractive"
    assert "polarity" not in att_rule.to_dict()
    assert rep_rule.polarity == "repulsive"
    assert rep_rule.to_dict()["polarity"] == "repulsive"

    haz_coords = {
        **_uniform(WILL_CULTURAL_LEAVES, 0.30, power=0.85, fairness=0.45),
        **_uniform(TRUTH_PHILOSOPHICAL_LEAVES, 0.30, chaos=0.80, order=0.40),
    }
    haz_exp = {SALIENCE_LEAF: 0.75, VALENCE_LEAF: 0.45, AROUSAL_LEAF: 0.70}
    haz_vibe, _ = compile_right_brain_vibe(
        birth,
        haz_coords,
        haz_exp,
        (att_rule, rep_rule),
        pass_index=1,
        event_id="ev-hazard-polarity",
    )
    assert haz_vibe.posture == "TUG_LEFT_BRAIN"
    assert "hazard_rule_warning(hazard_overextension_trap)" in haz_vibe.tug_reasons

    # 2. Upgrade #2: Compensated Soft-Min breaks false clarity=0.0 ties among colliding rules
    rule_strong_support = FuzzyConceptRule(
        rule_id="r-strong",
        concept_name="strong_supporting_leaves",
        description="Has two 1.0 leaves and one 0.70 leaf.",
        target_leaves={
            "cultural/justice/fairness": 0.75,
            "philosophical/structure/order": 0.75,
            "philosophical/persistence/continuity": 0.75,
        },
        low=0.67,
        high=0.73,
    )
    rule_flat_support = FuzzyConceptRule(
        rule_id="r-flat",
        concept_name="flat_supporting_leaves",
        description="Has all three leaves at 0.70 similarity.",
        target_leaves={
            "cultural/efficiency/power": 0.75,
            "cultural/efficiency/wastelessness": 0.75,
            "cultural/justice/honor": 0.75,
        },
        low=0.67,
        high=0.73,
    )
    tie_coords = {
        **_uniform(WILL_CULTURAL_LEAVES, 0.20, fairness=0.80, power=0.45, wastelessness=0.45, honor=0.45),
        **_uniform(TRUTH_PHILOSOPHICAL_LEAVES, 0.20, order=0.80, continuity=0.45),
    }
    # Pure min_similarity is 0.70 (0.50 firing) for both rules, but soft-min breaks the tie cleanly
    acts = _evaluate_rules_with_inhibition((rule_strong_support, rule_flat_support), tie_coords)
    clarity, _, winner, _, _ = compute_contemplation(acts)
    assert acts["strong_supporting_leaves"] > acts["flat_supporting_leaves"]
    assert clarity >= 0.50
    assert winner == "strong_supporting_leaves"

    # 3. Upgrade #4: 6-Leaf Experiential Sensor Parity (optional novelty, resonance, entrainment)
    sync_drawer = LearnedLeaf(
        "experiential/synchronization",
        "cadence_lock",
        "Synchronized cadence lock.",
    )
    sensors_6leaf = ThreeBridgeSensors(
        will_cultural=lambda _t, _c: _uniform(WILL_CULTURAL_LEAVES, 0.30),
        truth_philosophical=lambda _t, _c: _uniform(TRUTH_PHILOSOPHICAL_LEAVES, 0.30),
        contemplation_experiential=lambda _t, children=(): {
            SALIENCE_LEAF: 0.25,
            VALENCE_LEAF: 0.60,
            AROUSAL_LEAF: 0.40,
            NOVELTY_LEAF: 0.72,
            RESONANCE_LEAF: 0.84,
            RETENTION_LEAF: 0.78,
            **{c.leaf_id: 0.82 for c in children},
        },
    )
    scored_coords, scored_exp, queried, _ = sensors_6leaf.score_text(
        "Quiet cadence.",
        learned_leaves=(sync_drawer,),
    )
    assert sync_drawer.leaf_id in queried
    assert scored_exp[RESONANCE_LEAF] == pytest.approx(0.84)
    vibe_6leaf, _ = compile_right_brain_vibe(
        birth,
        scored_coords,
        scored_exp,
        (att_rule,),
        pass_index=1,
        event_id="ev-6leaf",
    )
    assert vibe_6leaf.resonance == pytest.approx(0.84)
    assert vibe_6leaf.retention == pytest.approx(0.78)

    # 4. Upgrade #3: Pre-Gap 3-Branch Chord Rollout in pure coordinate space
    base_chord = SubliminalChordVector.from_right_brain_vibe(vibe_6leaf)
    rollout_coords = {
        **_uniform(WILL_CULTURAL_LEAVES, 0.30, fairness=0.42, wastelessness=0.42, **{"self-sufficiency": 0.42}),
        **_uniform(TRUTH_PHILOSOPHICAL_LEAVES, 0.30, order=0.75),
    }
    rolled_chord = select_pre_gap_branch_chord_rollout(
        base_chord,
        rollout_coords,
        (att_rule, rep_rule),
    )
    assert rolled_chord.doer_vector != base_chord.doer_vector

    # 5. Upgrade #5: L2-Anchored Live Micro-Callosum Plasticity
    proj = SubliminalGapProjector(hidden_size=32)
    chord_b = SubliminalChordVector.from_right_brain_vibe(haz_vibe)
    pair = ContrastiveGapPair(
        pair_id="test_micro_plasticity_1",
        domain="social_dilemma",
        split="train",
        user_prompt="Test prompt.",
        clause_1_prefix="Weighing the moment,",
        chord_a=base_chord,
        posture_label_a="SETTLED_EASE",
        continuation_a="we maintain balanced order and fair cooperation.",
        chord_b=chord_b,
        posture_label_b="CONCEPT_COLLISION",
        continuation_b="we rush blindly into chaos.",
    )
    meta = update_projector_on_harvested_gap_pairs(
        proj,
        [pair],
        state_dir=tmp_path / "plasticity_state",
    )
    assert meta["updated"] is True
    assert meta["pairs_used"] == 1
    assert (tmp_path / "plasticity_state" / "live_projector_weights.pt").exists()
