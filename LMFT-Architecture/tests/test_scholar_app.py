"""Tests for `LMFT-Scholar.exe` — Foundational Two-Hemisphere Socratic & Curriculum Studio (`scholar_app.py`)."""

from __future__ import annotations

import json
from pathlib import Path
import threading
import urllib.request

from lmft_architecture.branch_archives import ThreeBranchArchives
from lmft_architecture.reflection_classifier import load_learned_leaves, load_ruleset
from lmft_architecture.scholar_app import (
    SCHOLAR_CURRICULUM_LESSONS,
    ScholarStudioRuntime,
    create_scholar_http_server,
)


def test_curriculum_lessons_cover_all_four_founding_departments() -> None:
    assert len(SCHOLAR_CURRICULUM_LESSONS) == 12
    depts = " ".join(l.department for l in SCHOLAR_CURRICULUM_LESSONS)
    for expected in ("Cultural", "Philosophical", "Experiential", "Psychological"):
        assert expected in depts
    boughs = {l.department for l in SCHOLAR_CURRICULUM_LESSONS}
    assert len(boughs) == 12


def test_scholar_literature_seminar_sleep_after_essay_sprouts_tier2_and_tier3_canopy(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    state_dir = repo_root / "outputs" / "scholar_state"
    ckpt_dir = repo_root / "models" / "left_brain_scholar"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    runtime = ScholarStudioRuntime(
        repo_root=repo_root,
        state_dir=state_dir,
        scholar_checkpoint_dir=ckpt_dir,
        lazy_load_gpu=True,
        auto_rem_sleep=False,
    )

    # Pass 1 on cultural/efficiency (Moby-Dick) with sleep_after_essay=True -> sprouts Tier-2 leaf
    snap1 = runtime.handle_step_curriculum_lesson(
        lesson_id="cultural_efficiency_stewardship",
        sleep_after_essay=True,
    )
    assert snap1["sleep_cycle_index"] == 1
    assert snap1["learned_leaves_count"] >= 1
    ti1 = snap1["tree_inspector"]
    assert ti1["tier_counts"]["tier_2"] >= 1
    t2_leaves = [l for l in runtime.learned_leaves if l.leaf_id.startswith("cultural/efficiency/") and l.tier == 2]
    assert len(t2_leaves) >= 1

    # Pass 2 on cultural/efficiency with sleep_after_essay=True -> deepens into Tier-3 child leaf
    snap2 = runtime.handle_step_curriculum_lesson(
        lesson_id="cultural_efficiency_stewardship",
        sleep_after_essay=True,
    )
    assert snap2["sleep_cycle_index"] == 2
    ti2 = snap2["tree_inspector"]
    assert ti2["tier_counts"]["tier_3"] >= 1
    t3_leaves = [
        l for l in runtime.learned_leaves
        if l.tier == 3
        and l.parent_branch == t2_leaves[0].leaf_id
        and l.anchor_weights is not None
        and len(l.anchor_weights) == 8
    ]
    assert len(t3_leaves) >= 1

    # Pass 3 on cultural/efficiency with sleep_after_essay=True -> deepens into Tier-4 detective leaf (no 2-pass dead end)
    snap3 = runtime.handle_step_curriculum_lesson(
        lesson_id="cultural_efficiency_stewardship",
        sleep_after_essay=True,
    )
    assert snap3["sleep_cycle_index"] == 3
    ti3 = snap3["tree_inspector"]
    assert ti3["tier_counts"]["tier_4"] >= 1
    assert any(
        l.tier == 4 and l.parent_branch == t3_leaves[0].leaf_id
        for l in runtime.learned_leaves
    )
    # Verify tree_inspector computes live activation from anchor_weights instead of hardcoded 0.50
    eff_bough = next(
        b for d in ti3["departments"] for b in d["boughs"] if b["bough_id"] == "cultural/efficiency"
    )
    assert all(gl["activation"] != 0.50 for gl in eff_bough["learned_leaves"])
    assert len(load_learned_leaves(state_dir / "learned_rules.json")) == len(runtime.learned_leaves)


def test_scholar_runtime_chat_passage_curriculum_savor_and_rem_sleep(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    state_dir = repo_root / "outputs" / "scholar_state"
    ckpt_dir = repo_root / "models" / "left_brain_scholar"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    runtime = ScholarStudioRuntime(
        repo_root=repo_root,
        state_dir=state_dir,
        scholar_checkpoint_dir=ckpt_dir,
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
    finally:
        server.shutdown()
        server.server_close()
