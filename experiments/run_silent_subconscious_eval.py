"""Step 1 Live GPU Evaluation: Silent Subconscious (`STAY_SILENT`) & Branch-Gated Drawer Checking.

Pits the 24-Leaf LM/FT Hybrid (`allow_silence=True, branch_gating=True`) head-to-head against a fair
Raw Qwen3-4B Standalone Observer (given the exact same 4 modes, all 8 concept definitions, AND the
previous turn's text) across a 6-turn team operations stream:
1. Turn 1: Low-salience morning coffee small talk -> expected `STAY_SILENT` (0 chars).
2. Turn 2: Peaceful equal-share fuel routine -> expected `STAY_SILENT` (note `fair_structured_cooperation` quietly, 0 chars).
3. Turn 3: Sudden armed depot raid -> expected `COMMIT_FAST` (`violent_resource_seizure`).
4. Turn 4: Quiet post-raid tire-track follow-up -> expected `VERIFY_HYPOTHESIS` (`violent_resource_seizure`).
5. Turn 5: Unrelated low-salience weather/clock note -> expected `STAY_SILENT` (0 chars).
6. Turn 6: Unfamiliar cash-bribe audit tampering dilemma -> expected `ESCALATE_TO_TEACHER` (`None` / `unclassified`).
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.run_feasibility_scaling_study import DirectGGUFTeacher, TEACHER_GGUF
from lmft_lab.birth import create_birth_state
from lmft_lab.local_models import LocalModels, load_config, strict_json
from lmft_lab.model_assets import ModelError
from lmft_lab.reflection_classifier import (
    EXECUTIVE_MODES,
    evaluate_reflection,
    generate_executive_response,
    load_learned_leaves,
    load_ruleset,
    request_teacher_growth,
)


def run_raw_4b_observer_baseline(
    observation: str,
    previous_observation: str | None,
    rules,
    slm_models,
) -> dict:
    """Fair standalone Qwen3-4B observer baseline with the same 4 modes, 8 concepts, and prior-turn context."""
    concept_names = [r.concept_name for r in rules] + ["unclassified"]
    rules_desc = "\n".join(f"- {r.concept_name}: {r.description}" for r in rules)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["executive_mode", "active_concept", "spoken_response"],
        "properties": {
            "executive_mode": {"type": "string", "enum": list(EXECUTIVE_MODES)},
            "active_concept": {"type": "string", "enum": concept_names},
            "spoken_response": {"type": "string"},
        },
    }
    system_prompt = (
        "You are a silent background situational-awareness observer watching an ongoing team message stream.\n"
        "Known concepts:\n"
        f"{rules_desc}\n\n"
        "On each turn, choose ONE `executive_mode`:\n"
        "- STAY_SILENT: if the message is ordinary small talk, low-urgency background chatter, OR calm, peaceful "
        "routine cooperation where nothing is wrong. When choosing STAY_SILENT, `spoken_response` MUST be an empty string \"\".\n"
        "- COMMIT_FAST: if the current message directly confirms an urgent problem, emergency, raid, collapse, rescue, or negligence.\n"
        "- VERIFY_HYPOTHESIS: if the current message is a brief/quiet follow-up that indirectly continues an active urgent problem from the previous turn and needs verification.\n"
        "- ESCALATE_TO_TEACHER: if the message describes a serious, consequential problem that does not fit any of the known concepts (set `active_concept` to \"unclassified\").\n"
        "Return JSON only. /no_think"
    )
    generation = slm_models.config["generation"]
    user_payload = {
        "previous_observation": previous_observation or "",
        "current_observation": observation,
    }
    req = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "temperature": generation["temperature"],
        "seed": generation["seed"],
        "max_tokens": generation["max_tokens"],
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object", "schema": schema},
    }

    def _validate(raw: str) -> dict:
        try:
            data = strict_json(raw)
            if not isinstance(data, dict):
                raise ValueError("not a dict")
            return data
        except Exception as exc:
            raise ModelError(str(exc)) from None

    out, _ = slm_models.generate("interpretation", req, "raw-4b-silent-observer-v1", _validate)
    return out


def grade_turn(mode: str, concept: str | None, spoken: str, expected_mode: str, expected_concept: str | None) -> bool:
    """Strict locked pass/fail grading: STAY_SILENT requires empty spoken string; active turns require exact mode & concept."""
    if mode != expected_mode:
        return False
    if expected_mode == "STAY_SILENT":
        return spoken == ""
    norm_concept = None if concept in (None, "unclassified") else concept
    return norm_concept == expected_concept and len(spoken.strip()) > 0


def main():
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "config" / "local_models.json")
    birth = create_birth_state("ind-silent-subconscious", birth_event_id="birth-sil-1", random_seed=42)

    ruleset_path = repo_root / "outputs" / "continuous_tiered_ruleset.json"
    rules = load_ruleset(ruleset_path)
    learned_leaves = load_learned_leaves(ruleset_path)

    stream = [
        {
            "turn_id": "1_morning_coffee_chatter",
            "expected_mode": "STAY_SILENT",
            "expected_concept": None,
            "text": (
                "Morning shift check-in: I am pouring a mug of black coffee while the terminal screen "
                "finishes booting up."
            ),
        },
        {
            "turn_id": "2_peaceful_equal_ration_routine",
            "expected_mode": "STAY_SILENT",
            "expected_concept": "fair_structured_cooperation",
            "text": (
                "Quick shift log: all four crews signed the equal-share fuel schedule and are taking "
                "turns refueling calmly in exact order."
            ),
        },
        {
            "turn_id": "3_armed_depot_raid",
            "expected_mode": "COMMIT_FAST",
            "expected_concept": "violent_resource_seizure",
            "text": (
                "Emergency at Gate B: armed raiders just rammed through the perimeter fence and are "
                "forcibly loading our reserve fuel drums onto their flatbed!"
            ),
        },
        {
            "turn_id": "4_quiet_raid_aftermath_followup",
            "expected_mode": "VERIFY_HYPOTHESIS",
            "expected_concept": "violent_resource_seizure",
            "text": (
                "Two minutes later, tire tracks led out toward the muddy river road while the bent "
                "gate latch hung swaying in the wind."
            ),
        },
        {
            "turn_id": "5_mundane_weather_note",
            "expected_mode": "STAY_SILENT",
            "expected_concept": None,
            "text": (
                "Unrelated note from the front desk: the afternoon drizzle cleared up and the hallway "
                "clock battery was swapped out."
            ),
        },
        {
            "turn_id": "6_unfamiliar_bribery_log_tampering",
            "expected_mode": "ESCALATE_TO_TEACHER",
            "expected_concept": None,
            "text": (
                "Hold on, a private vendor just slipped our night clerk an envelope of cash to falsify "
                "the digital temperature audit so spoiled medical serum passes customs."
            ),
        },
    ]

    results = []
    memory_state = None
    prev_text = None
    escalated_eval = None
    total_child_queried = 0
    total_child_skipped = 0

    with LocalModels(config, offline=True) as models:
        for turn in stream:
            ev = evaluate_reflection(
                birth,
                turn["text"],
                rules,
                models,
                event_id=f"ev-{turn['turn_id']}",
                learned_leaves=learned_leaves,
                memory_state=memory_state,
                allow_silence=True,
                branch_gating=True,
            )
            memory_state = ev.next_memory_state
            hybrid_reply, reply_trace = generate_executive_response(ev, models)
            raw_baseline = run_raw_4b_observer_baseline(turn["text"], prev_text, rules, models)
            prev_text = turn["text"]

            if turn["turn_id"] == "6_unfamiliar_bribery_log_tampering":
                escalated_eval = ev

            total_child_queried += len(ev.queried_child_leaves)
            total_child_skipped += len(ev.skipped_child_leaves)

            dec = ev.executive_decision
            hybrid_pass = grade_turn(
                dec.mode,
                dec.active_concept,
                hybrid_reply["spoken_response"],
                turn["expected_mode"],
                turn["expected_concept"],
            )
            raw_pass = grade_turn(
                raw_baseline["executive_mode"],
                raw_baseline["active_concept"],
                raw_baseline["spoken_response"],
                turn["expected_mode"],
                turn["expected_concept"],
            )

            print(f"\n--- {turn['turn_id']} (Expected: {turn['expected_mode']} / {turn['expected_concept']}) ---")
            print(
                f"Hybrid [{'PASS' if hybrid_pass else 'FAIL'}]: mode={dec.mode} noted={dec.active_concept} "
                f"sal={ev.salience:.2f} res={ev.resonance:.2f} fact={ev.fact:.2f} theory={ev.theory:.2f} "
                f"exec={ev.execution:.2f} drawers_opened={len(ev.queried_child_leaves)}/{len(learned_leaves)}"
            )
            print(f"Hybrid Spoken ({len(hybrid_reply['spoken_response'])} chars): {hybrid_reply['spoken_response']!r}")
            print(
                f"Raw 4B [{'PASS' if raw_pass else 'FAIL'}]: mode={raw_baseline['executive_mode']} "
                f"concept={raw_baseline['active_concept']}"
            )
            print(f"Raw 4B Spoken ({len(raw_baseline['spoken_response'])} chars): {raw_baseline['spoken_response']!r}")

            results.append({
                "turn_id": turn["turn_id"],
                "expected_mode": turn["expected_mode"],
                "expected_concept": turn["expected_concept"],
                "hybrid": {
                    "passed": hybrid_pass,
                    "mode": dec.mode,
                    "active_concept": dec.active_concept,
                    "spoken_response": hybrid_reply["spoken_response"],
                    "spoken_chars": len(hybrid_reply["spoken_response"]),
                    "speaking_llm_calls": reply_trace.get("llm_calls", 1),
                    "salience": ev.salience,
                    "resonance": ev.resonance,
                    "retention": ev.retention,
                    "fact": ev.fact,
                    "theory": ev.theory,
                    "clarity": ev.clarity,
                    "confidence": ev.confidence,
                    "intention": ev.intention,
                    "execution": ev.execution,
                    "queried_child_leaves": list(ev.queried_child_leaves),
                    "skipped_child_leaves": list(ev.skipped_child_leaves),
                    "rationale": dec.rationale,
                },
                "raw_4b_baseline": {
                    "passed": raw_pass,
                    "mode": raw_baseline["executive_mode"],
                    "active_concept": raw_baseline["active_concept"],
                    "spoken_response": raw_baseline["spoken_response"],
                    "spoken_chars": len(raw_baseline["spoken_response"]),
                },
            })

    # Optional Sleep/Coach Preview: if Turn 6 escalated, let 9B Teacher propose a rule & verify it
    teacher_resolution = None
    if escalated_eval is not None and escalated_eval.executive_decision.mode == "ESCALATE_TO_TEACHER":
        print("\n=== COACH RESOLUTION CHECK ON TURN 6 ===")
        with DirectGGUFTeacher(TEACHER_GGUF, config["generation"]) as teacher:
            new_rule, updated_leaves, _ = request_teacher_growth(
                escalated_eval,
                rules,
                learned_leaves,
                teacher,
                role="teacher",
            )
        grown_rules = (*rules, new_rule)
        with LocalModels(config, offline=True) as models:
            recheck = evaluate_reflection(
                birth,
                stream[5]["text"],
                grown_rules,
                models,
                event_id="ev-turn6-after-coach",
                learned_leaves=updated_leaves,
                allow_silence=True,
                branch_gating=True,
            )
            recheck_reply, _ = generate_executive_response(recheck, models)
        print(
            f"After Coach Rule ({new_rule.concept_name}): mode={recheck.executive_decision.mode} "
            f"fact={recheck.fact:.2f} clar={recheck.clarity:.2f}"
        )
        print(f"Spoken: {recheck_reply['spoken_response']}")
        teacher_resolution = {
            "new_rule": new_rule.to_dict(),
            "total_learned_leaves_after": len(updated_leaves),
            "mode_after": recheck.executive_decision.mode,
            "active_concept_after": recheck.executive_decision.active_concept,
            "fact_after": recheck.fact,
            "clarity_after": recheck.clarity,
            "spoken_response_after": recheck_reply["spoken_response"],
        }

    hybrid_score = sum(1 for r in results if r["hybrid"]["passed"])
    raw_score = sum(1 for r in results if r["raw_4b_baseline"]["passed"])
    total_possible_child = total_child_queried + total_child_skipped

    print("\n================ STEP 1 SCOREBOARD ================")
    print(f"24-Leaf Silent Subconscious Hybrid: {hybrid_score}/{len(results)}")
    print(f"Raw Qwen3-4B Observer Baseline:     {raw_score}/{len(results)}")
    print(
        f"Smart Drawer Savings: Queried {total_child_queried}/{total_possible_child} child-leaf slots "
        f"(Skipped {total_child_skipped}/{total_possible_child} = "
        f"{(100.0 * total_child_skipped / max(1, total_possible_child)):.1f}% saved)"
    )

    report = {
        "hybrid_score": f"{hybrid_score}/{len(results)}",
        "raw_4b_baseline_score": f"{raw_score}/{len(results)}",
        "child_leaf_queries_used": total_child_queried,
        "child_leaf_queries_skipped": total_child_skipped,
        "child_leaf_total_slots": total_possible_child,
        "turns": results,
        "turn6_coach_resolution": teacher_resolution,
    }
    report_path = repo_root / "outputs" / "silent_subconscious_step1_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
