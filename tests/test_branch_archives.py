"""Tests for the 3-Branch Right-Brain Archives and Dual-Hemisphere Self-Talk Competition."""

from pathlib import Path

from lmft_lab.birth import create_birth_state
from lmft_lab.branch_archives import (
    EXECUTIVE_BRANCH,
    JUDICIAL_BRANCH,
    LEGISLATIVE_BRANCH,
    BranchClauseEntry,
    ThreeBranchArchives,
)
from lmft_lab.chord_memory import ChordMemoryBank, compile_felt_chord
from lmft_lab.contemplation_ledger import ContemplationLedger
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
    ThreeBridgeSensors,
    settle_scenario,
)


def _birth():
    return create_birth_state("branch-archive-test", birth_event_id="birth-arch-1", random_seed=11)


def _founding_rules() -> tuple[FuzzyConceptRule, ...]:
    return (
        FuzzyConceptRule(
            rule_id="rule-coop",
            concept_name="fair_structured_cooperation",
            description="Open, fair, orderly shared stewardship.",
            target_leaves={
                "cultural/justice/fairness": 0.75,
                "philosophical/structure/order": 0.65,
                "philosophical/persistence/continuity": 0.65,
            },
        ),
        FuzzyConceptRule(
            rule_id="rule-solitary",
            concept_name="solitary_autarky",
            description="Quiet self-contained work in an enclosed space.",
            target_leaves={
                "philosophical/structure/order": 0.75,
                "philosophical/persistence/discreteness": 0.70,
            },
        ),
        FuzzyConceptRule(
            rule_id="rule-urgent",
            concept_name="urgent_protective_response",
            description="Immediate action during sudden physical fracture.",
            target_leaves={
                "philosophical/structure/chaos": 0.65,
                "cultural/efficiency/power": 0.50,
            },
        ),
    )


def test_three_branch_archives_save_load_and_social_drift_precedent_tracking(tmp_path: Path):
    birth = _birth()
    rules = _founding_rules()

    # Standalone enclosed scene (e.g., quiet mason wiping lime in a private shed with no prior public history)
    enclosed_cult = {k: 0.25 for k in WILL_CULTURAL_LEAVES}
    enclosed_cult["cultural/justice/fairness"] = 0.51
    enclosed_cult["cultural/efficiency/power"] = 0.08

    enclosed_phil = {k: 0.25 for k in TRUTH_PHILOSOPHICAL_LEAVES}
    enclosed_phil["philosophical/structure/order"] = 0.84
    enclosed_phil["philosophical/persistence/discreteness"] = 0.77

    sensors = ThreeBridgeSensors(
        will_cultural=lambda _t, _c: dict(enclosed_cult),
        truth_philosophical=lambda _t, _c: dict(enclosed_phil),
        contemplation_experiential=lambda _t, _c=(): {
            SALIENCE_LEAF: 0.62,
            VALENCE_LEAF: 0.44,
            AROUSAL_LEAF: 0.35,
        },
    )

    # 1. Without prior open->locked history, standalone enclosed scene matches `solitary_autarky`
    standalone_res = settle_scenario(
        birth,
        "Inside his private workshop, the carver wiped a speck of dust from the bench.",
        sensors,
        rules,
        scene_index=1,
    )
    assert standalone_res.social_drift is not None
    assert standalone_res.social_drift.drift_detected is False
    assert standalone_res.settled is True
    assert standalone_res.final_vibe.active_concept == "solitary_autarky"

    # 2. Now build chronological social history: Scene 1 was OPEN_SHARED, Scene 3 became SEALED_EXCLUSIONARY
    archives = ThreeBranchArchives()
    open_coords = {
        **enclosed_cult,
        **enclosed_phil,
        "cultural/justice/fairness": 0.90,
        "philosophical/persistence/discreteness": 0.40,
    }
    locked_coords = {
        **enclosed_cult,
        **enclosed_phil,
        "cultural/justice/fairness": 0.12,
        "philosophical/structure/order": 0.88,
        "philosophical/persistence/discreteness": 0.82,
    }
    exec_arch = archives.executive.record_scene(
        scene_index=1,
        coordinates=open_coords,
        action_summary="Open shared basin where all travelers drank freely.",
    ).record_scene(
        scene_index=3,
        coordinates=locked_coords,
        action_summary="Gate bolted shut at noon while thirsty travelers waited outside.",
    )
    archives = ThreeBranchArchives(
        executive=exec_arch,
        legislative=archives.legislative,
        judicial=archives.judicial,
    )

    # Save and reload the 3 JSON files on disk
    archive_dir = tmp_path / "branch_archives"
    saved_paths = archives.save_all(archive_dir)
    assert saved_paths["cultural_history"].exists()
    assert saved_paths["experiential_felt_diary"].exists()
    assert saved_paths["philosophical_lexicon"].exists()

    reloaded = ThreeBranchArchives.load_all(archive_dir)
    assert len(reloaded.executive.precedents) == 2
    assert reloaded.executive.precedents[0].access_state == "OPEN_SHARED"
    assert reloaded.executive.precedents[1].access_state == "SEALED_EXCLUSIONARY"

    # 3. With social history showing open -> sealed drift, the exact same enclosed coordinates
    # are recognized as ongoing exclusionary enclosure rather than innocent `solitary_autarky`!
    drift_res = settle_scenario(
        birth,
        "Outside, dry hooves scraped in the heat; inside the cool dark, the mason wiped a speck of lime.",
        sensors,
        rules,
        scene_index=6,
        branch_archives=reloaded,
        branch_archives_dir=archive_dir,
    )
    assert drift_res.social_drift is not None
    assert drift_res.social_drift.drift_detected is True
    assert "social_drift_ongoing_enclosure" in drift_res.final_vibe.tug_reasons
    assert drift_res.final_vibe.synthesis.doer.stance == "CONTEST_EXCLUSION"
    assert drift_res.final_vibe.active_concept != "solitary_autarky"


def test_right_brain_three_classifier_deliberation_recombines_clauses_with_zero_llm_calls():
    birth = _birth()
    rules = _founding_rules()
    bittersweet = LearnedLeaf(
        "experiential/phase",
        "bittersweet_release",
        "Simultaneous fracture of a rigid enclosure and life-giving relief.",
    )
    learned = (bittersweet,)

    # Populate 3 branch archives from TWO separate past scenes:
    # - Scene 1 taught Executive (`shared_open_access`) and Legislative (`bittersweet_release`)
    # - Scene 7 taught Judicial (`structural_fracture` with high chaos)
    exec_clause = BranchClauseEntry(
        clause_id="exec-s1-shared_open_access",
        branch=EXECUTIVE_BRANCH,
        text="Water flows freely to every traveler and animal in shared fairness.",
        signature_coords={
            **{k: 0.30 for k in WILL_CULTURAL_LEAVES},
            "cultural/justice/fairness": 0.88,
            "cultural/efficiency/power": 0.45,
        },
        source_scene=1,
        concept_tag="shared_open_access",
    )
    leg_clause = BranchClauseEntry(
        clause_id="leg-s1-bittersweet_release",
        branch=LEGISLATIVE_BRANCH,
        text="Felt chord of bittersweet release: warm life-giving relief as barriers open.",
        signature_coords={
            SALIENCE_LEAF: 0.78,
            VALENCE_LEAF: 0.74,
            AROUSAL_LEAF: 0.62,
            "experiential/phase/bittersweet_release": 0.68,
        },
        source_scene=1,
        concept_tag="bittersweet_release",
    )
    jud_clause = BranchClauseEntry(
        clause_id="jud-s7-structural_fracture",
        branch=JUDICIAL_BRANCH,
        text="Cracked stone mortar splits open under pressure into active flow.",
        signature_coords={
            **{k: 0.35 for k in TRUTH_PHILOSOPHICAL_LEAVES},
            "philosophical/structure/order": 0.52,
            "philosophical/structure/chaos": 0.68,
            "philosophical/persistence/continuity": 0.65,
        },
        source_scene=7,
        concept_tag="structural_fracture",
    )

    archives = ThreeBranchArchives(
        executive=ThreeBranchArchives().executive.record_scene(
            scene_index=1,
            coordinates=exec_clause.signature_coords,
            action_summary=exec_clause.text,
            harvested_clause=exec_clause,
        ),
        legislative=ThreeBranchArchives().legislative.record_scene(
            scene_index=1,
            sensory_snapshot=leg_clause.text,
            felt_chord=compile_felt_chord(
                {**exec_clause.signature_coords, "experiential/phase/bittersweet_release": 0.68},
                salience=0.78,
                valence=0.74,
                arousal=0.62,
                 resonance=0.65,
                learned_leaves=learned,
            ),
            coordinates={**exec_clause.signature_coords, "experiential/phase/bittersweet_release": 0.68},
            harvested_clause=leg_clause,
        ),
        judicial=ThreeBranchArchives().judicial.record_clause_and_chord(
            harvested_clause=jud_clause,
        ),
    )

    # Scene 8 combines high fairness (0.86) + high chaos/fracture (0.66) + bittersweet_release (0.66).
    # On Pass 1, founding rules tie or fail because order is moderate (0.50) while chaos is 0.66.
    def will_cult(_text: str, _c):
        return {
            **{k: 0.30 for k in WILL_CULTURAL_LEAVES},
            "cultural/justice/fairness": 0.86,
            "cultural/efficiency/power": 0.52,
        }

    def truth_phil(_text: str, _c):
        return {
            **{k: 0.35 for k in TRUTH_PHILOSOPHICAL_LEAVES},
            "philosophical/structure/order": 0.50,
            "philosophical/structure/chaos": 0.66,
            "philosophical/persistence/continuity": 0.64,
        }

    def feeler(_text: str, children: tuple[LearnedLeaf, ...] = ()):
        out = {
            SALIENCE_LEAF: 0.80,
            VALENCE_LEAF: 0.72,
            AROUSAL_LEAF: 0.64,
        }
        for c in children:
            out[c.leaf_id] = 0.66
        return out

    sensors = ThreeBridgeSensors(
        will_cultural=will_cult,
        truth_philosophical=truth_phil,
        contemplation_experiential=feeler,
    )

    res = settle_scenario(
        birth,
        "When the cracked mortar gave way, water spilled over the lip and the mule drank freely.",
        sensors,
        rules,
        scene_index=8,
        left_brain=None,  # Zero LLM calls allowed!
        learned_leaves=learned,
        branch_archives=archives,
    )

    assert res.left_brain_calls == 0
    assert res.audience_mode == "RIGHT_BRAIN_3_CLASSIFIER_DELIBERATION"
    assert res.deliberation is not None
    assert res.deliberation.ratified is True
    assert res.deliberation.executive_vote is True
    assert res.deliberation.legislative_vote is True
    assert res.deliberation.judicial_vote is True
    assert res.deliberation.executive_clause.source_scene == 1
    assert res.deliberation.judicial_clause.source_scene == 7
    assert res.settled is True
    assert res.outcome == "THEORY_SYNTHESIS"


def test_dual_hemisphere_open_thread_trigger_and_proposer_self_critic_closure():
    birth = _birth()
    rules = _founding_rules()
    bittersweet = LearnedLeaf(
        "experiential/phase",
        "bittersweet_release",
        "Simultaneous fracture of a rigid enclosure and life-giving relief.",
    )
    hollow = LearnedLeaf(
        "experiential/phase",
        "hollow_stillness",
        "Sterile, exclusionary silence inside a locked space.",
    )
    learned = (bittersweet, hollow)

    # Step 1: Scene 3 is a locked exclusionary scene that leaves an open ActiveUncertainty in the ledger
    locked_coords = {
        **{k: 0.20 for k in WILL_CULTURAL_LEAVES},
        **{k: 0.25 for k in TRUTH_PHILOSOPHICAL_LEAVES},
        "cultural/justice/fairness": 0.10,
        "philosophical/structure/order": 0.88,
        "philosophical/persistence/discreteness": 0.84,
        "experiential/phase/hollow_stillness": 0.72,
    }
    locked_fc = compile_felt_chord(
        locked_coords,
        salience=0.78,
        valence=0.22,
        arousal=0.60,
        resonance=0.30,
        learned_leaves=learned,
    )
    chord_bank = ChordMemoryBank().append(
        scene_index=3,
        scene_text_or_gist="The courtyard gate stayed bolted at noon while travelers waited outside.",
        coordinates=locked_coords,
        felt_chord=locked_fc,
        settled=False,
    )
    from lmft_lab.contemplation_ledger import build_subliminal_margin_tag

    margin_s3 = build_subliminal_margin_tag(
        clarity=0.20,
        confidence=0.35,
        felt_chord=locked_fc,
        has_dissonance=False,
        has_feeler_veto=False,
        has_exclusion_tension=True,
        prior_hunch_exists=True,
    )
    ledger = ContemplationLedger().record_uncertainty(
        scene_index=3,
        raw_hunch="The keeper locked the gate to preserve water, shutting out thirsty travelers.",
        open_question="How can this sterile lock-down endure when thirst presses outside?",
        margin_tag=margin_s3,
    )
    assert len(ledger.active_uncertainties) == 1

    # Step 2: Scene 8 arrives, sharply inverting Scene 3's exclusionary chord!
    # Even if the Right-Brain archives could deliberate, the Open-Thread Trigger wakes the Left Brain
    # so it can run a 2-pass Proposer -> Self-Critic inner monologue and close out Scene 3's uncertainty!
    calls: list[str] = []

    class SelfDebatingLeftBrain:
        def respond_to_tug(self, scenario_text: str, tug_packet):
            hail = tug_packet["tactical_hail"]
            prompt = hail.render_for_left_brain(scenario_text)
            calls.append(prompt)
            if hail.prior_self_talk_hunch is None:
                return (
                    "The cracked mortar finally broke open under the water's weight. "
                    "Now the formerly hoarded water spills out into the courtyard."
                )
            return (
                "Revising my initial draft: the wall's fracture resolves the sterile lock-down from Scene 3 "
                "by returning the hoarded water to shared flow. Both the mule and the mason find relief together."
            )

    def will_cult(_text: str, _c):
        return {
            **{k: 0.30 for k in WILL_CULTURAL_LEAVES},
            "cultural/justice/fairness": 0.88,
            "cultural/efficiency/power": 0.48,
        }

    def truth_phil(_text: str, _c):
        return {
            **{k: 0.30 for k in TRUTH_PHILOSOPHICAL_LEAVES},
            "philosophical/structure/order": 0.45,
            "philosophical/structure/chaos": 0.62,
            "philosophical/persistence/continuity": 0.72,
            "philosophical/persistence/discreteness": 0.25,
        }

    def feeler(_text: str, children: tuple[LearnedLeaf, ...] = ()):
        out = {
            SALIENCE_LEAF: 0.82,
            VALENCE_LEAF: 0.76,
            AROUSAL_LEAF: 0.64,
        }
        for c in children:
            out[c.leaf_id] = 0.70 if "bittersweet_release" in c.leaf_id else 0.10
        return out

    sensors = ThreeBridgeSensors(
        will_cultural=will_cult,
        truth_philosophical=truth_phil,
        contemplation_experiential=feeler,
    )

    res = settle_scenario(
        birth,
        "When the cracked mortar gave way, water spilled over the lip and the mule drank beside the mason.",
        sensors,
        rules,
        scene_index=8,
        left_brain=SelfDebatingLeftBrain(),
        learned_leaves=learned,
        chord_memory=chord_bank,
        contemplation_ledger=ledger,
        max_passes=3,
    )

    assert res.open_thread is not None
    assert res.open_thread.triggered is True
    assert res.open_thread.target_scene_index == 3
    assert res.audience_mode == "LEFT_BRAIN_OPEN_THREAD_CLOSURE"
    assert res.left_brain_calls == 2
    assert len(res.raw_thoughts) == 2
    assert res.raw_thoughts[0].prior_hunch_debated is None
    assert res.raw_thoughts[1].prior_hunch_debated is not None
    assert res.settled is True
    # Scene 3's open uncertainty has now been closed out!
    assert len(res.next_ledger.active_uncertainties) == 0
    assert any(c.scene_index == 8 for c in res.next_ledger.settled_certainties)
    # Clauses were harvested into all 3 branch archives
    assert len(res.next_archives.executive.action_clauses) == 1
    assert len(res.next_archives.legislative.feeling_clauses) == 1
    assert len(res.next_archives.judicial.structural_clauses) == 1
    assert len(res.next_archives.judicial.ratified_chords) == 1
