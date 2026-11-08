"""Tests for the 3-Branch Right-Brain Archives and Dual-Hemisphere Self-Talk Competition."""

from pathlib import Path

from lmft_architecture.birth import create_birth_state
from lmft_architecture.branch_archives import (
    EXECUTIVE_BRANCH,
    JUDICIAL_BRANCH,
    LEGISLATIVE_BRANCH,
    BranchClauseEntry,
    ThreeBranchArchives,
)
from lmft_architecture.chord_memory import ChordMemoryBank, compile_felt_chord
from lmft_architecture.contemplation_ledger import ContemplationLedger
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

    # 3. With social history showing open -> sealed drift, the Executive branch flags
    # `social_drift_ongoing_enclosure` and refuses to settle as innocent `solitary_autarky`,
    # while keeping raw sensor coordinates (`fairness == 0.51`) 100% untouched and
    # refusing to harvest unsettled drafts into the clause library!
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
    assert drift_res.final_vibe.coordinates["cultural/justice/fairness"] == 0.51
    assert "social_drift_ongoing_enclosure" in drift_res.final_vibe.tug_reasons
    assert drift_res.final_vibe.synthesis.doer.stance == "CONTEST_EXCLUSION"
    assert drift_res.settled is False
    assert drift_res.outcome == "UNRESOLVED_TENSION"
    assert len(drift_res.next_archives.executive.action_clauses) == 0
    assert len(drift_res.next_archives.judicial.ratified_chords) == 0


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
    restorative_rule = FuzzyConceptRule(
        rule_id="rule-restorative-release",
        concept_name="restorative_fracture_into_flow",
        description="Fractured barrier releases hoarded water into shared relief.",
        target_leaves={
            "cultural/justice/fairness": 0.75,
            "philosophical/structure/chaos": 0.55,
            "experiential/phase/bittersweet_release": 0.60,
        },
    )
    rules = _founding_rules() + (restorative_rule,)
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
    from lmft_architecture.contemplation_ledger import build_subliminal_margin_tag

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


def test_sparse_branch_signatures_contrastive_role_separation_and_coordinate_voting():
    """Verify MIT sparse wire pruning (`_sparse_branch_signature`), contrastive Executive/Judicial/Legislative role separation during REM sleep, and `0.70 * archived + 0.30 * text` coordinate-space committee voting."""
    from lmft_architecture.branch_archives import (
        _sparse_branch_signature,
        deliberate_across_three_branches,
    )
    from lmft_architecture.right_brain_equilibrium import RIGHT_BRAIN_WILL_EDGES
    from lmft_architecture.sleep_consolidation import (
        UnresolvedSleepScene,
        apply_dream_reflection_to_archives,
        consolidate_sleep_clusters_locally,
    )

    # 1. `_sparse_branch_signature` keeps 6 Founding branch leaves + at most top 2 active Tier-2 leaves (> 0.35)
    bloated_cult = {
        **{k: 0.52 for k in WILL_CULTURAL_LEAVES},
        "cultural/efficiency/wastelessness": 0.68,
        "cultural/efficiency/administrative_rot": 0.35,  # default 0.35 -> pruned
        "cultural/efficiency/coached_high_wastelessness_chaos": 0.84,  # top 1 learned leaf
        "cultural/efficiency/coached_high_power_chaos": 0.79,  # top 2 learned leaf
        "cultural/efficiency/coached_low_needlessness_agency": 0.62,  # 3rd learned leaf -> pruned
        "cultural/freedom/coached_low_self_sufficiency_chaos": 0.58,  # 4th learned leaf -> pruned
        "cultural/efficiency/hazard_low_wastelessness_chaos": 0.25,  # <= 0.35 -> pruned
    }
    sparse_cult = _sparse_branch_signature(bloated_cult, "cultural/", max_learned_leaves=2)
    assert len(sparse_cult) == 8  # 6 Founding cultural leaves + 2 sharpest Tier-2 leaves
    assert "cultural/efficiency/coached_high_wastelessness_chaos" in sparse_cult
    assert "cultural/efficiency/coached_high_power_chaos" in sparse_cult
    assert "cultural/efficiency/administrative_rot" not in sparse_cult
    assert "cultural/efficiency/coached_low_needlessness_agency" not in sparse_cult

    # 2. Contrastive "Problem vs. Remedy" Role Separation in `consolidate_sleep_clusters_locally`
    hazard_scene = UnresolvedSleepScene(
        scene_index=101,
        scenario_text="Chess table — Move 6: White played Bxf7+.",
        external_coordinates={
            **{k: 0.50 for k in WILL_CULTURAL_LEAVES},
            **{k: 0.50 for k in TRUTH_PHILOSOPHICAL_LEAVES},
            "cultural/efficiency/wastelessness": 0.20,
            "cultural/freedom/needlessness": 0.62,
            "philosophical/structure/chaos": 0.74,
            "philosophical/being/existence": 0.38,
        },
        felt_chord_notes={},
        valence=0.22,
        salience=0.84,
        arousal=0.70,
        left_brain_hunch="Unresolved scene tension: Chess table — Move 6. Legal replies include Kxf7, Ke7.",
        is_hazard=True,
        contrastive_reference_coordinates={
            **{k: 0.50 for k in WILL_CULTURAL_LEAVES},
            **{k: 0.50 for k in TRUTH_PHILOSOPHICAL_LEAVES},
            "cultural/efficiency/wastelessness": 0.88,
            "cultural/freedom/needlessness": 0.08,
            "philosophical/structure/chaos": 0.24,
            "philosophical/being/existence": 0.85,
        },
        game_review_annotation="Game Review [?? BLUNDER_??]: Played Bxf7+ instead of best move O-O.",
    )
    report = consolidate_sleep_clusters_locally(
        queued_scenes=(hazard_scene,),
        rules=_founding_rules(),
        learned_leaves=(),
        archives=ThreeBranchArchives(),
        ledger=ContemplationLedger(),
        hazard_scene_indices={101},
    )
    exec_cl = report.updated_archives.executive.action_clauses[-1]
    jud_cl = report.updated_archives.judicial.structural_clauses[-1]
    leg_cl = report.updated_archives.legislative.feeling_clauses[-1]

    # Judicial stores the problem signature (high chaos = 0.74, low existence = 0.38)
    assert jud_cl.signature_coords["philosophical/structure/chaos"] == 0.74
    assert jud_cl.signature_coords["philosophical/being/existence"] == 0.38
    # Executive sculpts toward the contrastive remedy (0.70 * 0.88 + 0.30 * 0.25 = 0.691 wastelessness, 0.70 * 0.08 + 0.30 * 0.62 = 0.242 needlessness)
    assert exec_cl.signature_coords["cultural/efficiency/wastelessness"] > 0.65
    assert exec_cl.signature_coords["cultural/freedom/needlessness"] < 0.30
    assert "Legal replies include" not in exec_cl.text
    assert "Unresolved scene tension:" not in exec_cl.text
    # Legislative stores the felt chord signature
    assert leg_cl.signature_coords[VALENCE_LEAF] == 0.22

    # 3. `apply_dream_reflection_to_archives` blocks repetitive dream sentences from overwriting clean clauses
    before_exec_text = exec_cl.text
    after_rep_archives = apply_dream_reflection_to_archives(
        report.updated_archives,
        scene_index=101,
        dream_text="Leap forward with Rc1 to leap forward with Rc1 and leap forward with Rc1.",
    )
    assert after_rep_archives.executive.action_clauses[-1].text == before_exec_text

    # 4. Coordinate-Space Committee Voting (`0.70 * archived_clause_coords + 0.30 * text_sensor_coords`)
    # Even if `sensors.score_text(candidate_bid_text)` returns distorted text coordinates (e.g. fairness=0.10),
    # the stored archive clause coordinates (`fairness=0.88`) anchor `bid_coords` at `0.70 * 0.88 + 0.30 * 0.10 = 0.646`!
    birth = _birth()
    fair_exec = BranchClauseEntry(
        clause_id="exec-fair-1",
        branch=EXECUTIVE_BRANCH,
        text="Kb2 followed by Rc1 and Qd4.",
        signature_coords={**{k: 0.50 for k in WILL_CULTURAL_LEAVES}, "cultural/justice/fairness": 0.88},
        source_scene=1,
        concept_tag="high_fairness",
    )
    calm_jud = BranchClauseEntry(
        clause_id="jud-calm-1",
        branch=JUDICIAL_BRANCH,
        text="Kb2 followed by Rc1 and Qd4.",
        signature_coords={**{k: 0.50 for k in TRUTH_PHILOSOPHICAL_LEAVES}, "philosophical/structure/order": 0.82, "philosophical/structure/chaos": 0.18},
        source_scene=1,
        concept_tag="high_order",
    )
    warm_leg = BranchClauseEntry(
        clause_id="leg-warm-1",
        branch=LEGISLATIVE_BRANCH,
        text="Kb2 followed by Rc1 and Qd4.",
        signature_coords={SALIENCE_LEAF: 0.72, VALENCE_LEAF: 0.75, AROUSAL_LEAF: 0.45},
        source_scene=1,
        concept_tag="warm_relief",
    )
    test_arch = ThreeBranchArchives(
        executive=ThreeBranchArchives().executive.record_scene(
            scene_index=1,
            coordinates=fair_exec.signature_coords,
            action_summary=fair_exec.text,
            harvested_clause=fair_exec,
        ),
        legislative=ThreeBranchArchives().legislative.record_scene(
            scene_index=1,
            sensory_snapshot=warm_leg.text,
            felt_chord=compile_felt_chord(fair_exec.signature_coords, salience=0.72, valence=0.75, arousal=0.45, resonance=0.70),
            coordinates=fair_exec.signature_coords,
            harvested_clause=warm_leg,
        ),
        judicial=ThreeBranchArchives().judicial.record_clause_and_chord(harvested_clause=calm_jud),
    )
    distorted_text_sensors = ThreeBridgeSensors(
        will_cultural=lambda _t, _c: {**{k: 0.20 for k in WILL_CULTURAL_LEAVES}, "cultural/justice/fairness": 0.10},
        truth_philosophical=lambda _t, _c: {**{k: 0.20 for k in TRUTH_PHILOSOPHICAL_LEAVES}, "philosophical/structure/order": 0.20},
        contemplation_experiential=lambda _t, _c=(): {SALIENCE_LEAF: 0.30, VALENCE_LEAF: 0.20, AROUSAL_LEAF: 0.80},
    )
    ext_c = {**fair_exec.signature_coords, **calm_jud.signature_coords}
    ext_e = {SALIENCE_LEAF: 0.72, VALENCE_LEAF: 0.75, AROUSAL_LEAF: 0.45}
    fc = compile_felt_chord(ext_c, salience=0.72, valence=0.75, arousal=0.45, resonance=0.70)
    delib = deliberate_across_three_branches(
        birth,
        ext_c,
        ext_e,
        fc,
        distorted_text_sensors,
        test_arch,
        RIGHT_BRAIN_WILL_EDGES,
        "test-v1",
        scene_index=2,
    )
    # 0.70 * 0.88 + 0.30 * 0.10 = 0.646
    assert abs(delib.bid_coordinates["cultural/justice/fairness"] - 0.646) < 1e-3
    # 0.70 * 0.75 + 0.30 * 0.20 = 0.585
    assert abs(delib.bid_experiential[VALENCE_LEAF] - 0.585) < 1e-3

    # 5. Verify on-disk `LMFT-Training/outputs/branch_archives/*.json` are sparse and free of overfit LoRA rants
    repo_arch_dir = Path(__file__).resolve().parents[2] / "LMFT-Training" / "outputs" / "branch_archives"
    if repo_arch_dir.exists():
        disk_arch = ThreeBranchArchives.load_all(repo_arch_dir)
        for cl in disk_arch.executive.action_clauses:
            assert len(cl.signature_coords) <= 8
            assert "sudden diagonal leap" not in cl.text.lower()
            assert "leap forward with rc1" not in cl.text.lower()
        for cl in disk_arch.judicial.structural_clauses:
            assert len(cl.signature_coords) <= 8
            assert "sudden diagonal leap" not in cl.text.lower()
        for cl in disk_arch.legislative.feeling_clauses:
            assert len(cl.signature_coords) <= 8
            assert "curfew" not in cl.text.lower()


def test_foundational_clause_pinning_concept_tag_entrainment_and_three_directory_seeding():
    """Verify `source_scene <= 8` pinning, MIT `concept_tag` entrainment (`<= 2` non-foundational clones), and balanced 3-Faculty seeding across Master Seed, Chess, and Bloodmoon."""
    from collections import Counter

    exec_arch = ThreeBranchArchives().executive
    # 1. Record 6 foundational scenes (`source_scene` 1..6)
    for s_idx in range(1, 7):
        f_clause = BranchClauseEntry(
            clause_id=f"exec-s{s_idx}-foundational",
            branch=EXECUTIVE_BRANCH,
            text=f"Foundational Sanctuary clause {s_idx}.",
            signature_coords={**{k: 0.50 for k in WILL_CULTURAL_LEAVES}, "cultural/justice/fairness": 0.90},
            source_scene=s_idx,
            concept_tag="shared_open_access",
        )
        exec_arch = exec_arch.record_scene(
            scene_index=s_idx,
            coordinates=f_clause.signature_coords,
            action_summary=f_clause.text,
            harvested_clause=f_clause,
        )

    # 2. Record 3 non-foundational clauses sharing `concept_tag="low_wastelessness"` (Scenes 101, 102, 103)
    w_values = [0.20, 0.40, 0.80]
    for offset, w_val in enumerate(w_values, start=101):
        nf_clause = BranchClauseEntry(
            clause_id=f"exec-sleep-s{offset}-low_wastelessness",
            branch=EXECUTIVE_BRANCH,
            text=f"Chess wastelessness remedy at scene {offset}.",
            signature_coords={**{k: 0.50 for k in WILL_CULTURAL_LEAVES}, "cultural/efficiency/wastelessness": w_val},
            source_scene=offset,
            concept_tag="low_wastelessness",
        )
        exec_arch = exec_arch.record_scene(
            scene_index=offset,
            coordinates=nf_clause.signature_coords,
            action_summary=nf_clause.text,
            harvested_clause=nf_clause,
        )

    non_found_waste = [
        c for c in exec_arch.action_clauses if c.source_scene > 8 and c.concept_tag == "low_wastelessness"
    ]
    # Capped at 2 non-foundational slots (`Scene 102` and entrained `Scene 101 -> Scene 103`)
    assert len(non_found_waste) == 2
    assert [c.source_scene for c in non_found_waste] == [102, 103]
    # Scene 103 blended `0.30 * 0.20 + 0.70 * 0.80 = 0.62`
    assert abs(non_found_waste[-1].signature_coords["cultural/efficiency/wastelessness"] - 0.62) < 1e-4

    # 3. Record 30 more non-foundational scenes with distinct tags -> Foundational Scenes 1..6 remain pinned!
    for extra_idx in range(200, 230):
        extra_cl = BranchClauseEntry(
            clause_id=f"exec-sleep-s{extra_idx}-tag_{extra_idx}",
            branch=EXECUTIVE_BRANCH,
            text=f"Later clause at scene {extra_idx}.",
            signature_coords={**{k: 0.50 for k in WILL_CULTURAL_LEAVES}},
            source_scene=extra_idx,
            concept_tag=f"tag_{extra_idx}",
        )
        exec_arch = exec_arch.record_scene(
            scene_index=extra_idx,
            coordinates=extra_cl.signature_coords,
            action_summary=extra_cl.text,
            harvested_clause=extra_cl,
        )

    assert len(exec_arch.action_clauses) == 24
    assert len(exec_arch.precedents) == 24
    assert [c.source_scene for c in exec_arch.action_clauses if c.source_scene <= 8] == [1, 2, 3, 4, 5, 6]
    assert [p.scene_index for p in exec_arch.precedents if p.scene_index <= 8] == [1, 2, 3, 4, 5, 6]

    # 4. Verify all 3 Schooled `branch_archives/` directories on disk have Foundational + Literary + Chess clauses
    repo_root = Path(__file__).resolve().parents[2]
    schooled_dirs = [
        repo_root / "LMFT-Training" / "outputs" / "branch_archives",
        repo_root / "LMFT-Training" / "outputs" / "chess_kibitzer_state" / "branch_archives",
        repo_root / "LMFT-Training" / "outputs" / "impish_bloodmoon_state" / "branch_archives",
    ]
    for arch_dir in schooled_dirs:
        assert arch_dir.exists(), f"Missing schooled branch_archives directory: {arch_dir}"
        loaded = ThreeBranchArchives.load_all(arch_dir)
        assert len([p for p in loaded.executive.precedents if p.scene_index <= 8]) == 8
        assert len([s for s in loaded.legislative.diary_snapshots if s.scene_index <= 8]) == 8
        for clause_list in (
            loaded.executive.action_clauses,
            loaded.legislative.feeling_clauses,
            loaded.judicial.structural_clauses,
        ):
            assert len([c for c in clause_list if c.source_scene <= 8]) == 6
            assert len([c for c in clause_list if 9 <= c.source_scene <= 12]) == 4
            assert len([c for c in clause_list if c.source_scene > 12]) >= 10
            nf_counts = Counter(c.concept_tag for c in clause_list if c.source_scene > 8)
            assert max(nf_counts.values()) <= 2

    virgin_arch_dir = repo_root / "LMFT-Architecture" / "baselines" / "virgin_dual_hemisphere_v1" / "branch_archives"
    if virgin_arch_dir.exists():
        virgin_loaded = ThreeBranchArchives.load_all(virgin_arch_dir)
        assert len(virgin_loaded.executive.action_clauses) == 0
        assert len(virgin_loaded.legislative.feeling_clauses) == 0
        assert len(virgin_loaded.judicial.structural_clauses) == 0


