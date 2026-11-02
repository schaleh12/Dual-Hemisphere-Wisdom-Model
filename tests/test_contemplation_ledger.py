"""Automated tests for Tactical Hail, Felt Chords, Chord-Rhyming Memory, and Contemplation Ledger."""

import pytest

from lmft_lab.birth import create_birth_state
from lmft_lab.chord_memory import (
    ChordMemoryBank,
    compile_felt_chord,
    load_chord_memory_bank,
    retrieve_chord_echoes,
    save_chord_memory_bank,
)
from lmft_lab.contemplation_ledger import (
    CONTEMPLATION_BRANCH,
    ContemplationLedger,
    build_subliminal_margin_tag,
    load_contemplation_ledger,
    load_raw_thought_stream,
    save_contemplation_ledger,
)
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
    QwenLeftBrain,
    ThreeBridgeSensors,
    build_tactical_hail,
    settle_scenario,
)


def _birth():
    return create_birth_state("rb-soil-test", birth_event_id="birth-soil-1", random_seed=11)


def _uniform(keys: tuple[str, ...], val: float = 0.10, **overrides: float) -> dict[str, float]:
    out = {k: val for k in keys}
    for short_or_full, v in overrides.items():
        matched = [k for k in keys if k == short_or_full or k.endswith("/" + short_or_full)]
        if not matched:
            raise KeyError(f"unknown key override: {short_or_full}")
        out[matched[0]] = v
    return out


def test_felt_chord_and_chord_rhyming_retrieve_both_rhyme_and_inversion(tmp_path):
    sanctuary = LearnedLeaf("experiential/phase", "austere_sanctuary", "Spare, open shelter offered freely.")
    hollow = LearnedLeaf("experiential/phase", "hollow_stillness", "Cold, sealed quiet cut off from life.")
    release = LearnedLeaf("experiential/phase", "bittersweet_release", "Barrier giving way into freeing flow.")
    learned = (sanctuary, hollow, release)

    # Scene 1: Open, orderly shelter
    c1 = {
        **_uniform(WILL_CULTURAL_LEAVES, 0.15, fairness=0.85, wastelessness=0.65),
        **_uniform(TRUTH_PHILOSOPHICAL_LEAVES, 0.15, agency=0.75, order=0.72, continuity=0.78, discreteness=0.12),
        sanctuary.leaf_id: 0.84,
        hollow.leaf_id: 0.08,
        release.leaf_id: 0.10,
    }
    fc1 = compile_felt_chord(c1, salience=0.68, valence=0.80, arousal=0.25, learned_leaves=learned)
    assert "austere sanctuary" in fc1.render_for_left_brain()
    assert "experiential/" not in fc1.render_for_left_brain()

    bank = ChordMemoryBank().append(
        scene_index=1,
        scene_text_or_gist="A harbour keeper leaves the stone breakwater lantern lit and the landing steps open.",
        coordinates=c1,
        felt_chord=fc1,
    )

    # Scene 2: Unrelated high-chaos storm
    c2 = {
        **_uniform(WILL_CULTURAL_LEAVES, 0.10, power=0.20, fairness=0.40),
        **_uniform(TRUTH_PHILOSOPHICAL_LEAVES, 0.10, chaos=0.88, order=0.10, continuity=0.20),
    }
    fc2 = compile_felt_chord(c2, salience=0.75, valence=0.35, arousal=0.90, learned_leaves=learned)
    bank = bank.append(
        scene_index=2,
        scene_text_or_gist="Sudden gale-force squalls tear canvas tarps loose along the outer pier.",
        coordinates=c2,
        felt_chord=fc2,
    )

    # Scene 3: Same deliberate order/agency as Scene 1, but locked and exclusionary -> Chord Inversion with Scene 1!
    c3 = {
        **_uniform(WILL_CULTURAL_LEAVES, 0.15, fairness=0.12, power=0.70, wastelessness=0.60),
        **_uniform(TRUTH_PHILOSOPHICAL_LEAVES, 0.15, agency=0.76, order=0.82, continuity=0.22, discreteness=0.85),
        sanctuary.leaf_id: 0.05,
        hollow.leaf_id: 0.86,
        release.leaf_id: 0.05,
    }
    fc3 = compile_felt_chord(c3, salience=0.76, valence=0.20, arousal=0.28, learned_leaves=learned)
    echo_3 = retrieve_chord_echoes(c3, fc3, bank, current_scene_index=3)
    assert echo_3.chord_inversion is not None
    assert echo_3.chord_inversion.scene_index == 1
    assert "exclusion" in echo_3.chord_inversion.relationship_summary
    assert "hollow stillness" in echo_3.chord_inversion.relationship_summary

    bank = bank.append(
        scene_index=3,
        scene_text_or_gist="The keeper bolts iron shutters over the landing and turns drenched fishermen away.",
        coordinates=c3,
        felt_chord=fc3,
    )

    mem_file = tmp_path / "chord_memory.json"
    save_chord_memory_bank(mem_file, bank)
    bank = load_chord_memory_bank(mem_file)
    assert len(bank.entries) == 3

    # Scene 4: Shutters unbolted and open harbour flow restored -> Rhymes with Scene 1 AND Inverts Scene 3!
    c4 = {
        **_uniform(WILL_CULTURAL_LEAVES, 0.15, fairness=0.82, wastelessness=0.58),
        **_uniform(TRUTH_PHILOSOPHICAL_LEAVES, 0.15, agency=0.68, order=0.62, continuity=0.80, discreteness=0.15),
        sanctuary.leaf_id: 0.65,
        hollow.leaf_id: 0.10,
        release.leaf_id: 0.82,
    }
    fc4 = compile_felt_chord(c4, salience=0.74, valence=0.78, arousal=0.45, learned_leaves=learned)
    echo_4 = retrieve_chord_echoes(c4, fc4, bank, current_scene_index=4)
    assert echo_4.structural_rhyme is not None
    assert echo_4.structural_rhyme.scene_index == 1
    assert echo_4.chord_inversion is not None
    assert echo_4.chord_inversion.scene_index == 3


def test_two_layer_contemplation_ledger_compacts_certainties_and_bounds_uncertainties(tmp_path):
    fc_warm = compile_felt_chord({}, salience=0.65, valence=0.78, arousal=0.30)
    fc_cold = compile_felt_chord({}, salience=0.75, valence=0.22, arousal=0.60)
    tag = build_subliminal_margin_tag(
        clarity=0.18,
        confidence=0.34,
        felt_chord=fc_cold,
        has_exclusion_tension=True,
    )

    ledger = ContemplationLedger(max_certainties=3, max_uncertainties=2)
    assert ledger.branch_owner == CONTEMPLATION_BRANCH

    # Multi-sentence draft blather gets compacted to 1 crisp line on settle
    ledger = ledger.record_settled(
        scene_index=1,
        takeaway_source=(
            "Crew members share the watch rotation fairly across all three decks. "
            "Extra second sentence of raw anxiety blather that should be stripped once settled."
        ),
        felt_chord=fc_warm,
        clarity=0.82,
        confidence=0.88,
    )
    assert len(ledger.settled_certainties) == 1
    assert "Extra second sentence" not in ledger.settled_certainties[0].takeaway

    # Add 3 uncertainties -> bounded strictly to max_uncertainties=2
    for idx in (2, 3, 4):
        ledger = ledger.record_uncertainty(
            scene_index=idx,
            raw_hunch=f"Initial tentative hunch for scene {idx}.",
            open_question=f"Who is being shut out in scene {idx}?",
            margin_tag=tag,
        )
    assert [u.scene_index for u in ledger.active_uncertainties] == [3, 4]

    ledger_path = tmp_path / "contemplation_ledger.json"
    save_contemplation_ledger(ledger_path, ledger)
    reloaded = load_contemplation_ledger(ledger_path)
    assert [u.scene_index for u in reloaded.active_uncertainties] == [3, 4]
    rendered = reloaded.render_for_left_brain()
    assert "SETTLED CERTAINTIES" in rendered
    assert "ACTIVE UNCERTAINTIES" in rendered
    assert "Red-Pen Margin | Clarity: low (0.18)" in rendered


def test_multi_pass_self_talk_revision_and_pure_tactical_hail_prompt_hygiene(tmp_path):
    birth = _birth()
    rules = (
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
    )

    class DummyModels:
        config = {"generation": {"temperature": 0.0, "seed": 7, "max_tokens": 160}}

    qwen_lb = QwenLeftBrain(DummyModels())

    class SelfDebatingLeftBrain:
        def __init__(self):
            self.calls = 0
            self.rendered_prompts: list[str] = []

        def respond_to_tug(self, scenario_text: str, tug_packet):
            self.calls += 1
            msgs = qwen_lb.build_prompt_messages(scenario_text, tug_packet)
            combined_prompt = msgs[0]["content"] + "\n" + msgs[1]["content"]
            self.rendered_prompts.append(combined_prompt)

            # Verify zero dry bureaucratic/HR debug labels leak into Qwen3-4B's prompt!
            forbidden_tokens = (
                "DOER_WILL_CULTURAL",
                "THINKER_TRUTH_PHILOSOPHICAL",
                "FEELER_CONTEMPLATION_EXPERIENTIAL",
                "KNOWN_INSTINCT_BLIND_SPOT",
                "unfamiliar_pattern_no_matching_rule",
                "cultural/justice/fairness",
                "philosophical/structure/order",
            )
            for bad in forbidden_tokens:
                assert bad not in combined_prompt

            if self.calls == 1:
                # Pass 2: tentative weak first hunch that fails to resolve ambiguity
                return "Water is moving past broken stone, though it is unclear whether order is returning."
            # Pass 3: sees its own prior hunch + Red-Pen Margin Note and revises sharply!
            assert "YOUR PRIOR HUNCH ON THIS SCENE" in combined_prompt
            assert "Red-Pen Margin" in combined_prompt
            return (
                "Revising my first hunch: the barrier's collapse restores shared, orderly access "
                "so everyone nearby can drink fairly in peace."
            )

    def will_cult(text: str, _c):
        if "restores shared, orderly access" in text:
            return _uniform(WILL_CULTURAL_LEAVES, 0.20, fairness=0.92, needlessness=0.15)
        return _uniform(WILL_CULTURAL_LEAVES, 0.20, fairness=0.32, power=0.45, needlessness=0.10)

    def truth_phil(text: str, _c):
        if "restores shared, orderly access" in text:
            return _uniform(TRUTH_PHILOSOPHICAL_LEAVES, 0.20, order=0.84, chaos=0.10, continuity=0.78)
        return _uniform(TRUTH_PHILOSOPHICAL_LEAVES, 0.20, order=0.38, chaos=0.22, continuity=0.35)

    sensors = ThreeBridgeSensors(
        will_cultural=will_cult,
        truth_philosophical=truth_phil,
        contemplation_experiential=lambda t: {
            SALIENCE_LEAF: 0.72,
            VALENCE_LEAF: 0.76 if "restores shared, orderly access" in t else 0.48,
            AROUSAL_LEAF: 0.55,
        },
    )

    raw_jsonl = tmp_path / "raw_thoughts.jsonl"
    ledger_json = tmp_path / "live_ledger.json"
    debater = SelfDebatingLeftBrain()

    res = settle_scenario(
        birth,
        "Fresh runoff threads through the breached culvert toward the communal cistern.",
        sensors,
        rules,
        scene_index=1,
        left_brain=debater,
        raw_thought_stream_path=raw_jsonl,
        contemplation_ledger_path=ledger_json,
        max_passes=3,
    )

    assert debater.calls == 2
    assert res.settled is True
    assert res.outcome == "THEORY_SYNTHESIS"
    assert len(res.passes) == 3
    # External Fact stays strictly uncorrupted by internal Left-Brain self-talk!
    assert res.final_vibe.coordinates["cultural/justice/fairness"] == pytest.approx(0.32)
    assert res.final_vibe.fact < 0.20
    assert res.final_vibe.theory >= 0.60

    # File 1 (.jsonl): both self-talk passes archived with Pass 3 linking prior_hunch_debated
    stream = load_raw_thought_stream(raw_jsonl)
    assert len(stream) == 2
    assert stream[0].pass_index == 2
    assert stream[0].settled_on_this_pass is False
    assert stream[1].pass_index == 3
    assert stream[1].prior_hunch_debated == stream[0].raw_self_talk_draft
    assert stream[1].settled_on_this_pass is True

    # File 2 (.json): uncertainty cleared into a 1-line SettledCertainty once settled
    saved_ledger = load_contemplation_ledger(ledger_json)
    assert len(saved_ledger.settled_certainties) == 1
    assert len(saved_ledger.active_uncertainties) == 0

    # Also verify direct build_tactical_hail helper
    hail = build_tactical_hail(res.final_vibe)
    assert "CURRENT SCENE:" in hail.render_for_left_brain("Sample scene.")
