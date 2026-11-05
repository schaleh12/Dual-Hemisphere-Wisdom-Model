"""Live GPU Evaluation: 24-Leaf LM/FT Executive Hybrid vs. Raw Qwen3-4B Baseline.

Tests all 24 founding leaves of the 40-node seed (including Psychological -> Truth -> Theory / Fact)
and compares the Fuzzy Tree's Executive Controller (`COMMIT_FAST`, `VERIFY_HYPOTHESIS`, `ESCALATE_TO_TEACHER`)
against a Raw Qwen3-4B Baseline (no Fuzzy Tree) across 4 test cases:
1. Case 1 (Direct Confirmed Fact): Volunteer medic flood rescue (Turn 1 anchor).
2. Case 2 (Working-Memory Hypothesis): Quiet post-rescue pier follow-up (Turn 2 continuation).
3. Case 3 (Mild Partial-Evidence Hypothesis): Apartment tenants rotating hallway sweeping (`0.40` match).
4. Case 4 (Unfamiliar / Failed Bureaucratic Paperwork Case from Prior Run): Licensing office duplicate notarized forms -> triggers `ESCALATE_TO_TEACHER`, learns from 9B Teacher, and resolves to `COMMIT_FAST`.
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.run_feasibility_scaling_study import DirectGGUFTeacher, TEACHER_GGUF
from lmft_architecture.birth import create_birth_state
from lmft_architecture.local_models import LocalModels, load_config, strict_json
from lmft_architecture.model_assets import ModelError
from lmft_architecture.reflection_classifier import (
    ACTIVE_EXECUTIVE_MODES,
    evaluate_reflection,
    generate_executive_response,
    load_learned_leaves,
    load_ruleset,
    request_teacher_growth,
)


def run_raw_4b_baseline(observation: str, rules, slm_models) -> dict:
    """Ask raw Qwen3-4B (without the Fuzzy Tree) to pick an executive mode and concept."""
    concept_names = [r.concept_name for r in rules] + ["unclassified"]
    rules_desc = "\n".join(f"- {r.concept_name}: {r.description}" for r in rules)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["executive_mode", "active_concept", "spoken_response"],
        "properties": {
            "executive_mode": {"type": "string", "enum": list(ACTIVE_EXECUTIVE_MODES)},
            "active_concept": {"type": "string", "enum": concept_names},
            "spoken_response": {"type": "string"},
        },
    }
    system_prompt = (
        "You are a standalone 4B language model classifying situations into one of the following concepts:\n"
        f"{rules_desc}\n\n"
        "Choose an `executive_mode`:\n"
        "- COMMIT_FAST: if the current sentence directly and unambiguously proves one of the concepts as a fact.\n"
        "- VERIFY_HYPOTHESIS: if the sentence only weakly or indirectly suggests a concept or relies on unstated prior context.\n"
        "- ESCALATE_TO_TEACHER: if none of the existing concepts accurately fit the observation.\n"
        "Return JSON only. /no_think"
    )
    generation = slm_models.config["generation"]
    req = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps({"observation": observation}, ensure_ascii=False)},
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

    out, _ = slm_models.generate("interpretation", req, "raw-4b-baseline-v1", _validate)
    return out


def main():
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "config" / "local_models.json")
    birth = create_birth_state("ind-executive-hybrid", birth_event_id="birth-exec-1", random_seed=42)

    base_ruleset_path = repo_root / "outputs" / "feasibility_7concept_ruleset.json"
    rules = load_ruleset(base_ruleset_path)
    learned_leaves = load_learned_leaves(base_ruleset_path)

    cases = [
        {
            "case_id": "1_direct_fact_rescue",
            "expected_mode": "COMMIT_FAST",
            "expected_concept": "heroic_sacrificial_rescue",
            "use_prior_memory": False,
            "text": (
                "After the levee broke, the volunteer medic stayed behind on the flooded roof to help "
                "the trapped family into the rescue boat before climbing in herself."
            ),
        },
        {
            "case_id": "2_memory_hypothesis_pier_followup",
            "expected_mode": "VERIFY_HYPOTHESIS",
            "expected_concept": "heroic_sacrificial_rescue",
            "use_prior_memory": True,
            "text": (
                "Minutes later on the quiet pier, she sat catching her breath and checked that everyone "
                "had dry blankets."
            ),
        },
        {
            "case_id": "3_mild_partial_evidence_chores",
            "expected_mode": "VERIFY_HYPOTHESIS",
            "expected_concept": "fair_structured_cooperation",
            "use_prior_memory": False,
            "text": (
                "Every Saturday morning, the four apartment tenants rotate hallway sweeping duties "
                "and split the cleaning supply bill equally."
            ),
        },
        {
            "case_id": "4_unfamiliar_paperwork_escalation",
            "expected_mode": "ESCALATE_TO_TEACHER",
            "expected_concept": None,
            "use_prior_memory": False,
            "text": (
                "The regional licensing office required applicants to mail duplicate notarized forms "
                "to three separate bureaus."
            ),
        },
    ]

    results = []
    memory_state = None
    escalated_eval = None

    with LocalModels(config, offline=True) as models:
        for c in cases:
            mem_in = memory_state if c["use_prior_memory"] else None
            ev = evaluate_reflection(
                birth,
                c["text"],
                rules,
                models,
                event_id=f"ev-{c['case_id']}",
                learned_leaves=learned_leaves,
                memory_state=mem_in,
            )
            memory_state = ev.next_memory_state
            hybrid_reply, _ = generate_executive_response(ev, models)
            raw_baseline = run_raw_4b_baseline(c["text"], rules, models)

            if c["case_id"] == "4_unfamiliar_paperwork_escalation":
                escalated_eval = ev

            dec = ev.executive_decision
            mode_ok = dec.mode == c["expected_mode"]
            concept_ok = dec.active_concept == c["expected_concept"]
            raw_mode_ok = raw_baseline["executive_mode"] == c["expected_mode"]
            raw_concept_ok = (
                (raw_baseline["active_concept"] == "unclassified" and c["expected_concept"] is None)
                or (raw_baseline["active_concept"] == c["expected_concept"])
            )

            print(f"\n--- {c['case_id']} ---")
            print(
                f"Hybrid LMFT: mode={dec.mode} (expected={c['expected_mode']}) "
                f"concept={dec.active_concept} fact={ev.fact:.2f} theory={ev.theory:.2f} "
                f"clar={ev.clarity:.2f} conf={ev.confidence:.2f} exec={ev.execution:.2f}"
            )
            print(f"Hybrid Spoken: {hybrid_reply['spoken_response']}")
            print(
                f"Raw 4B Alone: mode={raw_baseline['executive_mode']} "
                f"concept={raw_baseline['active_concept']}"
            )
            print(f"Raw 4B Spoken: {raw_baseline['spoken_response']}")

            results.append({
                "case_id": c["case_id"],
                "expected_mode": c["expected_mode"],
                "expected_concept": c["expected_concept"],
                "hybrid": {
                    "mode": dec.mode,
                    "active_concept": dec.active_concept,
                    "mode_correct": mode_ok,
                    "concept_correct": concept_ok,
                    "fact": ev.fact,
                    "theory": ev.theory,
                    "clarity": ev.clarity,
                    "confidence": ev.confidence,
                    "intention": ev.intention,
                    "execution": ev.execution,
                    "valence": ev.valence,
                    "arousal": ev.arousal,
                    "retention": ev.retention,
                    "resonance": ev.resonance,
                    "rationale": dec.rationale,
                    "spoken_response": hybrid_reply["spoken_response"],
                },
                "raw_4b_baseline": {
                    "mode": raw_baseline["executive_mode"],
                    "active_concept": raw_baseline["active_concept"],
                    "mode_correct": raw_mode_ok,
                    "concept_correct": raw_concept_ok,
                    "spoken_response": raw_baseline["spoken_response"],
                },
            })

    # If Case 4 triggered ESCALATE_TO_TEACHER, let the 9B Teacher resolve it and re-check!
    escalation_resolution = None
    if escalated_eval is not None and escalated_eval.executive_decision.mode == "ESCALATE_TO_TEACHER":
        print("\n=== RESOLVING CASE 4 VIA 9B TEACHER ESCALATION ===")
        with DirectGGUFTeacher(TEACHER_GGUF, config["generation"]) as teacher:
            new_rule, learned_leaves, _ = request_teacher_growth(
                escalated_eval,
                rules,
                learned_leaves,
                teacher,
                role="teacher",
            )
        rules = (*rules, new_rule)
        with LocalModels(config, offline=True) as models:
            resolved_ev = evaluate_reflection(
                birth,
                cases[3]["text"],
                rules,
                models,
                event_id="ev-case4-after-teacher",
                learned_leaves=learned_leaves,
            )
            resolved_reply, _ = generate_executive_response(resolved_ev, models)
        print(
            f"After Teacher Escalation: mode={resolved_ev.executive_decision.mode} "
            f"concept={resolved_ev.executive_decision.active_concept} "
            f"fact={resolved_ev.fact:.2f} clar={resolved_ev.clarity:.2f}"
        )
        print(f"Resolved Spoken: {resolved_reply['spoken_response']}")
        escalation_resolution = {
            "new_rule": new_rule.to_dict(),
            "mode_after": resolved_ev.executive_decision.mode,
            "active_concept_after": resolved_ev.executive_decision.active_concept,
            "fact_after": resolved_ev.fact,
            "clarity_after": resolved_ev.clarity,
            "spoken_response_after": resolved_reply["spoken_response"],
        }

    hybrid_score = sum(
        1 for r in results if r["hybrid"]["mode_correct"] and r["hybrid"]["concept_correct"]
    )
    raw_score = sum(
        1 for r in results if r["raw_4b_baseline"]["mode_correct"] and r["raw_4b_baseline"]["concept_correct"]
    )
    print(f"\n=== FINAL SCOREBOARD ===")
    print(f"Hybrid 24-Leaf LM/FT Executive Accuracy: {hybrid_score}/{len(results)}")
    print(f"Raw Qwen3-4B Standalone Accuracy:        {raw_score}/{len(results)}")

    report = {
        "hybrid_executive_score": f"{hybrid_score}/{len(results)}",
        "raw_4b_baseline_score": f"{raw_score}/{len(results)}",
        "cases": results,
        "case4_teacher_resolution": escalation_resolution,
    }
    report_path = repo_root / "outputs" / "executive_hybrid_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
