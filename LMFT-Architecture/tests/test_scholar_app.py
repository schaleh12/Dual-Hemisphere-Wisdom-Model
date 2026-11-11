"""Tests for `LMFT-Scholar.exe` — Foundational Two-Hemisphere Socratic & Curriculum Studio (`scholar_app.py`)."""

from __future__ import annotations

import json
from pathlib import Path
import threading
import urllib.request

import pytest

from lmft_architecture.branch_archives import ThreeBranchArchives
from lmft_architecture.reflection_classifier import (
    AROUSAL_LEAF,
    FOUNDATIONAL_LEAVES,
    LearnedLeaf,
    SALIENCE_LEAF,
    VALENCE_LEAF,
    load_learned_leaves,
    load_ruleset,
)
from lmft_architecture.right_brain_equilibrium import ThreeBridgeSensors
from lmft_architecture.scholar_app import (
    SCHOLAR_CURRICULUM_LESSONS,
    ScholarStudioRuntime,
    _validate_human_leaf_payload,
    create_scholar_http_server,
)


def _make_discriminative_test_sensors(*, non_discriminative_leaf: str = "") -> ThreeBridgeSensors:
    """Deterministic CPU test sensors that open boughs and score custom leaves higher on Target A than Foil B."""

    def _is_foil_text(text: str) -> bool:
        low = text.lower()
        return any(
            cue in low
            for cue in (
                "harpoon of relentless vengeance",
                "unbending decree without exception",
                "bolts from the laboratory in horror",
                "casual small talk",
                "glad to speak about: hello",
            )
        )

    def _score_custom(text: str, child: LearnedLeaf) -> float:
        if non_discriminative_leaf and child.leaf_name == non_discriminative_leaf:
            return 0.60 if not _is_foil_text(text) else 0.56
        return 0.16 if _is_foil_text(text) else 0.78

    def _will(text: str, children: tuple[LearnedLeaf, ...] = ()) -> dict[str, float]:
        foil = _is_foil_text(text)
        out = {
            "cultural/efficiency/power": 0.76 if foil else 0.68,
            "cultural/efficiency/wastelessness": 0.18 if foil else 0.79,
            "cultural/justice/honor": 0.30 if foil else 0.74,
            "cultural/justice/fairness": 0.22 if foil else 0.77,
            "cultural/freedom/self-sufficiency": 0.58,
            "cultural/freedom/needlessness": 0.32 if foil else 0.66,
        }
        for c in children:
            if c.leaf_id.startswith("cultural/"):
                out[c.leaf_id] = _score_custom(text, c)
        return out

    def _truth(text: str, children: tuple[LearnedLeaf, ...] = ()) -> dict[str, float]:
        foil = _is_foil_text(text)
        out = {
            "philosophical/being/agency": 0.72,
            "philosophical/being/existence": 0.28 if foil else 0.75,
            "philosophical/structure/order": 0.34 if foil else 0.74,
            "philosophical/structure/chaos": 0.78 if foil else 0.32,
            "philosophical/persistence/continuity": 0.29 if foil else 0.73,
            "philosophical/persistence/discreteness": 0.64 if foil else 0.44,
        }
        for c in children:
            if c.leaf_id.startswith("philosophical/"):
                out[c.leaf_id] = _score_custom(text, c)
        return out

    def _feeler(text: str, children: tuple[LearnedLeaf, ...] = ()) -> dict[str, float]:
        foil = _is_foil_text(text)
        out = {
            "experiential/reflection/novelty": 0.56,
            SALIENCE_LEAF: 0.72,
            VALENCE_LEAF: 0.28 if foil else 0.70,
            AROUSAL_LEAF: 0.76 if foil else 0.48,
            "experiential/synchronization/resonance": 0.25 if foil else 0.72,
            "experiential/synchronization/entrainment": 0.74 if foil else 0.30,
        }
        for c in children:
            if c.leaf_id.startswith("experiential/"):
                out[c.leaf_id] = _score_custom(text, c)
        return out

    return ThreeBridgeSensors(
        will_cultural=_will,
        truth_philosophical=_truth,
        contemplation_experiential=_feeler,
    )


def _make_test_leaf_distiller():
    """Return realistic plain-English human leaf payloads (zero story nouns, zero digits, zero slop)."""
    catalog = [
        (
            "prudent_equipment_care",
            "Protecting shared tools and people from reckless exhaustion while pursuing a demanding goal.",
        ),
        (
            "measured_pace_keeping",
            "Pacing collective effort steadily so urgent ambition never burns through safety margins.",
        ),
        (
            "preventive_tool_upkeep",
            "Inspecting and repairing essential gear early before small strains turn into breakdowns.",
        ),
        (
            "restorative_mercy_practice",
            "Repairing human dignity through proportionate mercy rather than mechanical punishment.",
        ),
        (
            "answerable_maker_care",
            "Staying present and accountable for the real consequences of one's own creations.",
        ),
        (
            "grounded_crisis_poise",
            "Holding steady perceptual calm when surrounding pressure tempts impulsive overreaction.",
        ),
    ]
    call_idx = {"n": 0}

    def _distill(
        *,
        parent_branch: str,
        target_tier: int,
        scene_a_text: str,
        scene_b_text: str,
        parent_leaf: LearnedLeaf | None,
        existing_sibling_names: list[str],
        attempt: int = 0,
        prior_error: str = "",
    ) -> dict[str, object]:
        existing = set(existing_sibling_names)
        chosen_name = ""
        chosen_desc = ""
        for cand_name, cand_desc in catalog:
            if cand_name not in existing:
                chosen_name = cand_name
                chosen_desc = cand_desc
                break
        if not chosen_name:
            idx = call_idx["n"]
            call_idx["n"] += 1
            suffix = ("alpha", "bravo", "charlie", "delta")[idx % 4]
            chosen_name = f"steady_stewardship_{suffix}"
            chosen_desc = "Maintaining patient human responsibility and care when pressure invites shortcuts."
        return {
            "leaf_name": chosen_name,
            "description": chosen_desc,
            "positive_examples": [
                "A clinic nurse double-checks dosage labels calmly before starting the night shift.",
                "A carpentry foreman halts the saw line to replace a frayed belt before anyone is hurt.",
                "A school principal listens privately to a struggling student instead of issuing automatic suspension.",
                "An orchard grower rests tired soil with cover crops rather than forcing another harvest.",
                "A software team pauses a rushed release to fix a quiet reliability flaw for users.",
            ],
            "negative_examples": [
                "A warehouse manager skips safety checks and pushes exhausted drivers past midnight for bonuses.",
                "A supervisor yells slogans at tired staff while ignoring broken brakes on the delivery trucks.",
                "An administrator applies a rigid penalty form without reading what actually happened.",
                "A contractor pours cheap concrete in freezing rain and walks away before inspections.",
                "A team lead chases flashy demos while leaving critical maintenance tickets unanswered.",
            ],
            "executive_clause": "Pace collective effort and maintain shared safeguards before pushing for output.",
            "legislative_clause": "Keep the emotional tone steady, humane, and attentive to quiet strain.",
            "judicial_clause": "Lasting capability depends on preserving the people and structures that carry the work.",
        }

    return _distill


def test_curriculum_lessons_cover_all_four_founding_departments() -> None:
    assert len(SCHOLAR_CURRICULUM_LESSONS) == 12
    depts = " ".join(l.department for l in SCHOLAR_CURRICULUM_LESSONS)
    for expected in ("Cultural", "Philosophical", "Experiential", "Psychological"):
        assert expected in depts
    boughs = {l.department for l in SCHOLAR_CURRICULUM_LESSONS}
    assert len(boughs) == 12
    for lesson in SCHOLAR_CURRICULUM_LESSONS:
        assert len(lesson.contrast_foil_text.strip()) >= 40


def test_validate_human_leaf_payload_rejects_slop_digits_proper_nouns_and_copied_scenes() -> None:
    scene_a = (
        "In Herman Melville's Moby-Dick, First Mate Starbuck counsels that the ship's strength "
        "exists to husband the vessel's gear and bring every sailor safely home to Nantucket."
    )
    scene_b = (
        "Captain Ahab welds the crew's discipline aboard the Pequod into a single harpoon of vengeance."
    )
    valid_base = _make_test_leaf_distiller()(
        parent_branch="cultural/efficiency",
        target_tier=2,
        scene_a_text=scene_a,
        scene_b_text=scene_b,
        parent_leaf=None,
        existing_sibling_names=[],
    )
    cleaned = _validate_human_leaf_payload(
        valid_base,
        scene_a_text=scene_a,
        scene_b_text=scene_b,
        existing_short_names=set(),
    )
    assert cleaned["leaf_name"] == "prudent_equipment_care"

    # 1. Reject old Python mechanical slop names (`_synthesis`, `_accord`, `coached_`, `cultural_`)
    for bad_name in (
        "fairness_chaos_synthesis",
        "power_existence_accord",
        "coached_low_power",
        "cultural_responsibility",
    ):
        bad_payload = dict(valid_base, leaf_name=bad_name)
        with pytest.raises(ValueError):
            _validate_human_leaf_payload(
                bad_payload,
                scene_a_text=scene_a,
                scene_b_text=scene_b,
                existing_short_names=set(),
            )

    # 2. Reject float dumps / digits in description
    bad_digits = dict(
        valid_base,
        description="Domain-general cultural/efficiency rule anchored at wastelessness=0.72 and power=0.64.",
    )
    with pytest.raises(ValueError):
        _validate_human_leaf_payload(
            bad_digits,
            scene_a_text=scene_a,
            scene_b_text=scene_b,
            existing_short_names=set(),
        )

    # 3. Reject literary proper nouns (`Starbuck`, `Ahab`, `Antigone`)
    bad_noun = dict(
        valid_base,
        description="Guiding a crew with Starbuck's calm restraint rather than reckless obsession.",
    )
    with pytest.raises(ValueError):
        _validate_human_leaf_payload(
            bad_noun,
            scene_a_text=scene_a,
            scene_b_text=scene_b,
            existing_short_names=set(),
        )

    # 4. Filter isolated contaminated example cleanly, and reject if < 4 clean everyday examples remain
    copied_pos = list(valid_base["positive_examples"])  # type: ignore[arg-type]
    copied_pos[0] = "The first mate counsels that ship strength exists to husband vessel gear and bring sailors safely home."
    filtered_one = _validate_human_leaf_payload(
        dict(valid_base, positive_examples=copied_pos),
        scene_a_text=scene_a,
        scene_b_text=scene_b,
        existing_short_names=set(),
    )
    assert len(filtered_one["positive_examples"]) == 4
    assert all("sailors" not in s for s in filtered_one["positive_examples"])

    copied_pos[1] = "Starbuck guides the ship crew safely back to Nantucket harbor."
    with pytest.raises(ValueError, match="requires at least 4 clean everyday sentences"):
        _validate_human_leaf_payload(
            dict(valid_base, positive_examples=copied_pos),
            scene_a_text=scene_a,
            scene_b_text=scene_b,
            existing_short_names=set(),
        )

    # 5. Reject Tier-3/4 child leaf repeating its parent leaf's description
    with pytest.raises(ValueError, match="parent description"):
        _validate_human_leaf_payload(
            dict(valid_base, leaf_name="measured_pace_keeping"),
            scene_a_text=scene_a,
            scene_b_text=scene_b,
            existing_short_names={"prudent_equipment_care"},
            parent_description=str(valid_base["description"]),
        )


def test_scholar_offline_without_llm_never_fabricates_fake_fstring_leaves(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    state_dir = repo_root / "outputs" / "scholar_state"
    ckpt_dir = repo_root / "models" / "left_brain_scholar"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    runtime = ScholarStudioRuntime(
        repo_root=repo_root,
        state_dir=state_dir,
        scholar_checkpoint_dir=ckpt_dir,
        sensors=_make_discriminative_test_sensors(),
        lazy_load_gpu=True,
        auto_rem_sleep=False,
    )
    snap = runtime.handle_step_curriculum_lesson(
        lesson_id="cultural_efficiency_stewardship",
        sleep_after_essay=True,
    )
    assert snap["sleep_cycle_index"] == 1
    assert snap["learned_leaves_count"] == 0
    assert snap["rules_count"] == 2
    assert runtime.rem_dream_chronicles[-1]["rejected_candidates"]


def test_unseen_holdout_gatekeeper_openly_rejects_non_discriminative_candidate(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    state_dir = repo_root / "outputs" / "scholar_state"
    ckpt_dir = repo_root / "models" / "left_brain_scholar"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Sensors score ALL candidate leaves nearly identically on Scene A (0.60) vs Foil B (0.56 -> margin +0.04 < 0.12)
    flat_sensors = _make_discriminative_test_sensors()
    orig_will = flat_sensors.will_cultural

    def _flat_will(text: str, children: tuple[LearnedLeaf, ...] = ()) -> dict[str, float]:
        out = orig_will(text, ())
        for c in children:
            if c.leaf_id.startswith("cultural/"):
                out[c.leaf_id] = 0.58
        return out

    non_discrim_sensors = ThreeBridgeSensors(
        will_cultural=_flat_will,
        truth_philosophical=flat_sensors.truth_philosophical,
        contemplation_experiential=flat_sensors.contemplation_experiential,
    )

    runtime = ScholarStudioRuntime(
        repo_root=repo_root,
        state_dir=state_dir,
        scholar_checkpoint_dir=ckpt_dir,
        sensors=non_discrim_sensors,
        leaf_distiller=_make_test_leaf_distiller(),
        lazy_load_gpu=True,
        auto_rem_sleep=False,
    )
    snap = runtime.handle_step_curriculum_lesson(
        lesson_id="cultural_efficiency_stewardship",
        sleep_after_essay=True,
    )
    assert snap["sleep_cycle_index"] == 1
    assert snap["learned_leaves_count"] == 0
    assert "Rejected 1 candidate" in snap["last_rem_sleep_receipt"]


def test_scholar_literature_seminar_sleep_after_essay_sprouts_tier2_tier3_and_tier4_canopy(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    state_dir = repo_root / "outputs" / "scholar_state"
    ckpt_dir = repo_root / "models" / "left_brain_scholar"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    runtime = ScholarStudioRuntime(
        repo_root=repo_root,
        state_dir=state_dir,
        scholar_checkpoint_dir=ckpt_dir,
        sensors=_make_discriminative_test_sensors(),
        leaf_distiller=_make_test_leaf_distiller(),
        lazy_load_gpu=True,
        auto_rem_sleep=False,
    )

    # Pass 1 on cultural/efficiency (Moby-Dick) with sleep_after_essay=True -> distills & verifies Tier-2 leaf
    snap1 = runtime.handle_step_curriculum_lesson(
        lesson_id="cultural_efficiency_stewardship",
        sleep_after_essay=True,
    )
    assert snap1["sleep_cycle_index"] == 1
    assert snap1["learned_leaves_count"] == 1
    ti1 = snap1["tree_inspector"]
    assert ti1["tier_counts"]["tier_2"] == 1
    t2_leaves = [l for l in runtime.learned_leaves if l.leaf_id.startswith("cultural/efficiency/") and l.tier == 2]
    assert len(t2_leaves) == 1
    assert "synthesis" not in t2_leaves[0].leaf_name
    assert "accord" not in t2_leaves[0].leaf_name

    # Pass 2 on cultural/efficiency -> deepens into Tier-3 child leaf under Pass-1 Tier-2 leaf
    snap2 = runtime.handle_step_curriculum_lesson(
        lesson_id="cultural_efficiency_stewardship",
        sleep_after_essay=True,
    )
    assert snap2["sleep_cycle_index"] == 2
    ti2 = snap2["tree_inspector"]
    assert ti2["tier_counts"]["tier_3"] == 1
    t3_leaves = [
        l for l in runtime.learned_leaves
        if l.tier == 3
        and l.parent_branch == t2_leaves[0].leaf_id
        and l.anchor_weights is not None
        and len(l.anchor_weights) == 8
    ]
    assert len(t3_leaves) == 1

    # Pass 3 on cultural/efficiency -> deepens into Tier-4 child leaf (and ThreeBridgeSensors queries Tier 4!)
    snap3 = runtime.handle_step_curriculum_lesson(
        lesson_id="cultural_efficiency_stewardship",
        sleep_after_essay=True,
    )
    assert snap3["sleep_cycle_index"] == 3
    ti3 = snap3["tree_inspector"]
    assert ti3["tier_counts"]["tier_4"] == 1
    t4_leaves = [l for l in runtime.learned_leaves if l.tier == 4 and l.parent_branch == t3_leaves[0].leaf_id]
    assert len(t4_leaves) == 1

    # Verify ThreeBridgeSensors.score_text genuinely queries Tier-4 drawers when Tier-3 parent activates
    scored_coords, _, queried, _ = runtime.sensors.score_text(
        SCHOLAR_CURRICULUM_LESSONS[0].passage_text,
        learned_leaves=runtime.learned_leaves,
        branch_gating=True,
    )
    assert t4_leaves[0].leaf_id in queried
    assert scored_coords[t4_leaves[0].leaf_id] >= 0.70

    eff_bough = next(
        b for d in ti3["departments"] for b in d["boughs"] if b["bough_id"] == "cultural/efficiency"
    )
    assert all(gl["activation"] != 0.50 for gl in eff_bough["learned_leaves"])
    assert all(gl["verification"]["unseen_margin"] >= 0.12 for gl in eff_bough["learned_leaves"])
    assert len(load_learned_leaves(state_dir / "learned_leaves.json")) == 3

    # Verify restarting ScholarStudioRuntime reloads all 3 custom leaves & verification metrics from disk
    restarted = ScholarStudioRuntime(
        repo_root=repo_root,
        state_dir=state_dir,
        scholar_checkpoint_dir=ckpt_dir,
        sensors=_make_discriminative_test_sensors(),
        leaf_distiller=_make_test_leaf_distiller(),
        lazy_load_gpu=True,
        auto_rem_sleep=False,
    )
    snap_reloaded = restarted.get_state_snapshot()
    assert snap_reloaded["learned_leaves_count"] == 3
    assert snap_reloaded["tree_inspector"]["tier_counts"] == {"tier_1": 24, "tier_2": 1, "tier_3": 1, "tier_4": 1}

    # Verify reset_scholar_canopy wipes custom leaves & rules cleanly back to Cycle-0 seed
    snap_reset = runtime.reset_scholar_canopy()
    assert snap_reset["learned_leaves_count"] == 0
    assert snap_reset["rules_count"] == 2
    assert snap_reset["sleep_cycle_index"] == 0


def test_scholar_runtime_chat_passage_curriculum_savor_and_rem_sleep(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    state_dir = repo_root / "outputs" / "scholar_state"
    ckpt_dir = repo_root / "models" / "left_brain_scholar"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    runtime = ScholarStudioRuntime(
        repo_root=repo_root,
        state_dir=state_dir,
        scholar_checkpoint_dir=ckpt_dir,
        sensors=_make_discriminative_test_sensors(),
        leaf_distiller=_make_test_leaf_distiller(),
        lazy_load_gpu=True,
        auto_rem_sleep=False,
    )

    snap0 = runtime.get_state_snapshot()
    assert snap0["app_name"] == "LMFT-Scholar"
    assert snap0["rules_count"] == 2
    assert len(snap0["curriculum_lessons"]) == len(SCHOLAR_CURRICULUM_LESSONS)

    # 1. Socratic chat turn (natural equilibrium flow without sentence-count restriction)
    snap1 = runtime.handle_chat_message(
        "How does patient stewardship balance power with wastelessness?",
    )
    companion_msgs = [m for m in snap1["chat_messages"] if m.get("sender") == "companion" and "id" in m]
    assert len(companion_msgs) == 1
    first_turn_id = int(companion_msgs[0]["id"])
    tel1 = companion_msgs[0]["telemetry"]
    assert tel1["clarity"] >= 0.0
    assert "fact" in tel1 and "theory" in tel1
    assert "intention" in tel1 and "execution" in tel1

    # 2. Savor turn into 3-Branch Archives
    snap_savored = runtime.savor_turn_vibe(first_turn_id)
    assert any(m.get("id") == first_turn_id and m.get("savored") for m in snap_savored["chat_messages"])
    assert len(runtime.archives.executive.action_clauses) >= 1

    # 3. Study passage contemplation
    snap2 = runtime.handle_study_passage(
        "When a river floods in spring, terraced floodplains absorb the chaos and preserve order.",
    )
    assert "terraced floodplains" in snap2["study_canvas_text"]

    # 4. Step curriculum lesson
    snap3 = runtime.handle_step_curriculum_lesson(
        lesson_id="cultural_justice_reciprocity",
    )
    assert snap3["curriculum_step_counter"] == 1

    # 5. Coach two turns and run REM Sleep consolidation
    turn_ids = [int(m["id"]) for m in snap3["chat_messages"] if m.get("sender") == "companion" and "id" in m]
    runtime.flag_turn_for_coaching(
        turn_ids[0],
        user_note="Anchor stewardship in wastelessness and long-term structural order.",
        check_auto_sleep=False,
    )
    runtime.flag_turn_for_coaching(
        turn_ids[1],
        user_note="Maintain resilient order when absorbing creative turbulence.",
        check_auto_sleep=False,
    )
    assert len(runtime.sleep_queue) >= 2

    snap_sleep = runtime.handle_run_rem_sleep()
    assert snap_sleep["sleep_queue_count"] == 0
    assert snap_sleep["sleep_cycle_index"] == 1
    assert snap_sleep["rules_count"] >= 3
    assert "REM Sleep Cycle #1 Complete" in snap_sleep["last_rem_sleep_receipt"]
    assert (state_dir / "learned_rules.json").exists()
    assert (state_dir / "will_edges.json").exists()


def test_scholar_virgin_lineage_and_brain_graduation(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    state_dir = repo_root / "outputs" / "scholar_state"
    ckpt_dir = repo_root / "models" / "left_brain_scholar"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Create a dummy live_projector_weights.pt to verify graduation copies it
    (ckpt_dir / "live_projector_weights.pt").write_bytes(b"DUMMY_PROJECTOR_STATE")

    runtime = ScholarStudioRuntime(
        repo_root=repo_root,
        state_dir=state_dir,
        scholar_checkpoint_dir=ckpt_dir,
        sensors=_make_discriminative_test_sensors(),
        leaf_distiller=_make_test_leaf_distiller(),
        lazy_load_gpu=True,
        auto_rem_sleep=False,
    )

    # Step a lesson, coach it, and consolidate in Scholar lineage
    snap = runtime.handle_step_curriculum_lesson(num_sentences=2)
    comp_id = next(int(m["id"]) for m in snap["chat_messages"] if m.get("sender") == "companion" and "id" in m)
    runtime.flag_turn_for_coaching(comp_id, "Balance power with wastelessness.", check_auto_sleep=False)
    runtime.handle_run_rem_sleep()
    assert len(runtime.rules) >= 3

    # Switch to Virgin lineage and verify read-only 2-rule view
    snap_virgin = runtime.switch_brain_lineage("virgin")
    assert snap_virgin["active_lineage"] == "virgin"
    assert snap_virgin["rules_count"] == 2
    runtime.switch_brain_lineage("scholar")

    # Graduate brain to LMFT-Chess and LMFT-Bloodmoon
    snap_grad_chess = runtime.graduate_brain_to_target("chess")
    assert "Graduated Scholar Brain ➔ LMFT-Chess" in snap_grad_chess["last_graduation_receipt"]
    chess_state_dir = repo_root / "outputs" / "chess_kibitzer_state"
    chess_model_dir = repo_root / "models" / "left_brain_chess"
    assert (chess_state_dir / "learned_rules.json").exists()
    assert len(load_ruleset(chess_state_dir / "learned_rules.json")) == len(runtime.rules)
    assert (chess_state_dir / "will_edges.json").exists()
    chess_archives = ThreeBranchArchives.load_all(chess_state_dir / "branch_archives")
    assert len(chess_archives.executive.action_clauses) >= 1
    assert (chess_model_dir / "live_projector_weights.pt").exists()

    snap_grad_bm = runtime.graduate_brain_to_target("bloodmoon")
    assert "Graduated Scholar Brain ➔ LMFT-Bloodmoon" in snap_grad_bm["last_graduation_receipt"]
    bm_state_dir = repo_root / "outputs" / "impish_bloodmoon_state"
    bm_model_dir = repo_root / "models" / "left_brain_bloodmoon"
    assert len(load_ruleset(bm_state_dir / "learned_rules.json")) == len(runtime.rules)
    assert (bm_model_dir / "live_projector_weights.pt").exists()


def test_scholar_http_server_endpoints(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    runtime = ScholarStudioRuntime(
        repo_root=repo_root,
        state_dir=repo_root / "outputs" / "scholar_state",
        scholar_checkpoint_dir=repo_root / "models" / "left_brain_scholar",
        sensors=_make_discriminative_test_sensors(),
        leaf_distiller=_make_test_leaf_distiller(),
        lazy_load_gpu=True,
        auto_rem_sleep=False,
    )
    server = create_scholar_http_server(runtime, host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as resp:
            html = resp.read().decode("utf-8")
            assert "LMFT-Scholar" in html
            assert "12-Book Classic Literature Seminar" in html
            assert "Visual Tree &amp; Canopy Development Inspector" in html
            assert "Natural Equilibrium Flow" in html
            assert "Reset Canopy to Clean Seed" in html
            assert "chat-sentences-select" not in html
            assert "curriculum-sentences-select" not in html

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/curriculum/step",
            data=json.dumps({"lesson_id": "philosophical_being_agency", "sleep_after_essay": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data["curriculum_step_counter"] == 1
            assert data["sleep_cycle_index"] == 1
            assert data["tree_inspector"]["tier_counts"]["tier_2"] >= 1

        reset_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/tree/reset",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(reset_req) as resp:
            reset_data = json.loads(resp.read().decode("utf-8"))
            assert reset_data["learned_leaves_count"] == 0
            assert reset_data["rules_count"] == 2
    finally:
        server.shutdown()
        server.server_close()


def test_scholar_chat_decouples_canvas_wires_history_and_updates_projector_in_rem_sleep(tmp_path: Path) -> None:
    from lmft_architecture.subliminal_gap_splice import SubliminalGapProjector
    from lmft_architecture.three_legged_runtime import SplicedThoughtTrace

    class _StubScholarLeftBrain:
        prompt_format = "chatml"

        def __init__(self) -> None:
            self.projector = SubliminalGapProjector(hidden_size=64)
            self.projector.capture_resting_anchor()
            self.seen_scenarios: list[str] = []
            self.seen_histories: list[tuple[tuple[str, str], ...]] = []

        def generate_multi_sentence_script(
            self,
            scenario_text: str,
            chord=None,
            *,
            num_sentences: int = 2,
            pass_index: int = 2,
            clause_1_prefix=None,
            conversation_history=(),
            max_context_chars: int = 2400,
            chord_builder_at_comma=None,
            stop_condition=None,
            lead_with_gap: bool = False,
        ):
            self.seen_scenarios.append(scenario_text)
            self.seen_histories.append(tuple(conversation_history))
            built_chord = None
            if chord_builder_at_comma is not None:
                res = chord_builder_at_comma("Hello there,")
                built_chord = res[0] if isinstance(res, tuple) else res
            reply = f"Hello there, glad to speak about: {scenario_text[:48]}."
            trace = SplicedThoughtTrace(
                pass_index=pass_index,
                clause_1_prefix="Hello there,",
                prompt_before_gap="Hello there, <|subliminal_gap|>",
                spliced_chord=built_chord or chord,  # type: ignore[arg-type]
                post_gap_continuation=f"glad to speak about: {scenario_text[:48]}.",
                full_thought=reply,
                hit_eos=True,
            )
            return reply, (trace,)

    repo_root = tmp_path / "repo"
    state_dir = repo_root / "outputs" / "scholar_state"
    ckpt_dir = repo_root / "models" / "left_brain_scholar"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    stub_lb = _StubScholarLeftBrain()

    runtime = ScholarStudioRuntime(
        repo_root=repo_root,
        state_dir=state_dir,
        scholar_checkpoint_dir=ckpt_dir,
        sensors=_make_discriminative_test_sensors(),
        scholar_left_brain=stub_lb,
        leaf_distiller=_make_test_leaf_distiller(),
        lazy_load_gpu=True,
        auto_rem_sleep=False,
    )

    # 1. Default chat does NOT inject study_canvas_text (Moby-Dick) into user prompt or fact_anchor_text
    snap1 = runtime.handle_chat_message("hello")
    assert stub_lb.seen_scenarios[0] == "hello"
    assert stub_lb.seen_histories[0] == ()
    comp1 = [m for m in snap1["chat_messages"] if m.get("sender") == "companion" and "id" in m][-1]
    assert "inner theory aligns with observed fact" not in comp1["text"]
    assert "Moby-Dick" not in stub_lb.seen_scenarios[0]

    # 2. Second chat turn receives Turn 1 in conversation_history
    snap2 = runtime.handle_chat_message("How are you doing today?")
    assert len(stub_lb.seen_histories[-1]) == 1
    assert stub_lb.seen_histories[-1][0][0] == "hello"
    assert stub_lb.seen_histories[-1][0][1] == comp1["text"]

    # 3. Opt-in anchor_to_canvas=True explicitly grounds the prompt and Pass-1 sensors in study_canvas_text
    runtime.handle_chat_message("What is Starbuck's argument?", anchor_to_canvas=True)
    assert "Reading Passage:" in stub_lb.seen_scenarios[-1]
    assert "Starbuck" in stub_lb.seen_scenarios[-1]

    # 4. Coach Turn 1 with a contrastive note and run REM sleep -> distills leaf, harvests gap pair & updates projector!
    turn_1_id = int(comp1["id"])
    runtime.flag_turn_for_coaching(
        turn_1_id,
        user_note="Balance decisive power and urgent agency against catastrophic chaos and disorder.",
        check_auto_sleep=False,
    )
    snap_sleep = runtime.handle_run_rem_sleep()
    assert snap_sleep["sleep_cycle_index"] == 1
    assert len(runtime.rem_dream_chronicles) == 1
    chron = runtime.rem_dream_chronicles[-1]
    assert len(chron["minted_leaves"]) == 1
    assert chron["harvested_gap_pairs"] >= 1
    assert chron["projector_updated"] is True
    assert (state_dir / "live_projector_weights.pt").is_file()
    assert snap_sleep["tree_inspector"]["has_live_callosum_projector"] is True


def test_scholar_background_warmup_nonblocking_state_and_gear2_synchronization(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import time
    from lmft_architecture import scholar_app
    from lmft_architecture.subliminal_gap_splice import SubliminalGapProjector
    from lmft_architecture.three_legged_runtime import SplicedThoughtTrace

    class _DelayedWarmupLeftBrain:
        prompt_format = "chatml"

        def __init__(self) -> None:
            self.device = "cpu"
            self.callosum_metadata = None
            self.projector = SubliminalGapProjector(hidden_size=64)
            self.projector.capture_resting_anchor()

        def generate_multi_sentence_script(
            self,
            scenario_text: str,
            chord=None,
            *,
            num_sentences: int = 2,
            pass_index: int = 2,
            clause_1_prefix=None,
            conversation_history=(),
            max_context_chars: int = 2400,
            chord_builder_at_comma=None,
            stop_condition=None,
            lead_with_gap: bool = False,
        ):
            built = chord
            if chord_builder_at_comma is not None:
                res = chord_builder_at_comma("Hello,")
                built = res[0] if isinstance(res, tuple) else res
            reply = f"Hello, warmed up and speaking to: {scenario_text}."
            trace = SplicedThoughtTrace(
                pass_index=pass_index,
                clause_1_prefix="Hello,",
                prompt_before_gap="Hello, <|subliminal_gap|>",
                spliced_chord=built,  # type: ignore[arg-type]
                post_gap_continuation=f"warmed up and speaking to: {scenario_text}.",
                full_thought=reply,
                hit_eos=True,
            )
            return reply, (trace,)

    load_entered = threading.Event()
    release_load = threading.Event()

    def _slow_from_checkpoint(checkpoint_dir: Path, **kwargs):
        load_entered.set()
        release_load.wait(timeout=5.0)
        return _DelayedWarmupLeftBrain()

    monkeypatch.setattr(scholar_app.SubliminalGapLeftBrain, "from_checkpoint", _slow_from_checkpoint)
    monkeypatch.setattr(scholar_app, "_get_shared_neural_bridges", lambda _root: None)

    repo_root = tmp_path / "repo"
    ckpt_dir = repo_root / "models" / "left_brain_scholar"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "projector_weights.pt").write_bytes(b"DUMMY")

    t0 = time.perf_counter()
    runtime = ScholarStudioRuntime(
        repo_root=repo_root,
        state_dir=repo_root / "outputs" / "scholar_state",
        scholar_checkpoint_dir=ckpt_dir,
        lazy_load_gpu=False,
        background_warmup=True,
        auto_rem_sleep=False,
    )
    assert (time.perf_counter() - t0) < 0.5
    assert load_entered.wait(timeout=2.0)

    # While GPU warmup is still in flight, get_state_snapshot() must NOT block!
    t_snap = time.perf_counter()
    snap_warming = runtime.get_state_snapshot()
    assert (time.perf_counter() - t_snap) < 0.2
    assert snap_warming["left_brain_loading"] is True
    assert "WARMING UP GPU" in snap_warming["left_brain_status"]

    # Sending a chat message during warmup must wait for warmup and answer in Gear 2 (never Gear 1!)
    chat_holder: list[dict] = []

    def _send_early_hello() -> None:
        chat_holder.append(runtime.handle_chat_message("hello"))

    chat_thread = threading.Thread(target=_send_early_hello, daemon=True)
    chat_thread.start()
    time.sleep(0.05)
    release_load.set()
    chat_thread.join(timeout=5.0)

    assert len(chat_holder) == 1
    snap_after = chat_holder[0]
    assert snap_after["left_brain_loading"] is False
    assert snap_after["left_brain_status"].startswith("READY")
    last_msg = snap_after["chat_messages"][-1]
    assert "warmed up and speaking to: hello" in last_msg["text"]
    assert last_msg["telemetry"]["gear_used"] != "GEAR_1_FAST_RIGHT_BRAIN"


def test_scholar_http_server_exclusive_port_and_shutdown_endpoint(tmp_path: Path) -> None:
    import sys

    repo_root = tmp_path / "repo"
    runtime = ScholarStudioRuntime(
        repo_root=repo_root,
        state_dir=repo_root / "outputs" / "scholar_state",
        scholar_checkpoint_dir=repo_root / "models" / "left_brain_scholar",
        sensors=_make_discriminative_test_sensors(),
        lazy_load_gpu=True,
        auto_rem_sleep=False,
    )
    server = create_scholar_http_server(runtime, host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        if sys.platform.startswith("win"):
            with pytest.raises(OSError):
                dup = create_scholar_http_server(runtime, host="127.0.0.1", port=port)
                dup.server_close()

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/shutdown",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            assert payload["status"] == "shutting_down"
        thread.join(timeout=3.0)
        assert not thread.is_alive()
    finally:
        server.server_close()
