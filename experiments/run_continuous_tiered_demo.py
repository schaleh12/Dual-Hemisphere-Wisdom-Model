"""Live GPU Experiment for Milestones 2 & 3:
1. Recursive Tier-1 -> Tier-2 -> Tier-3 Sub-Branching when two Tier-2 branches collide.
2. Continuous-State Sequential Memory (Valence, Arousal, Resonance, Retention) across a 3-turn sequence.
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.run_feasibility_scaling_study import DirectGGUFTeacher, TEACHER_GGUF
from lmft_lab.birth import create_birth_state
from lmft_lab.local_models import LocalModels, load_config
from lmft_lab.reflection_classifier import (
    evaluate_reflection,
    load_learned_leaves,
    load_ruleset,
    request_teacher_growth,
    save_ruleset,
)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "config" / "local_models.json")
    birth = create_birth_state("ind-continuous-tiered", birth_event_id="birth-ct-1", random_seed=42)

    base_ruleset_path = repo_root / "outputs" / "feasibility_7concept_ruleset.json"
    rules = load_ruleset(base_ruleset_path)
    learned_leaves = load_learned_leaves(base_ruleset_path)

    print("=== PART 1: TIER-2 COLLISION -> TIER-3 SUB-BRANCH GROWTH ===")
    compound_train = (
        "As violent seismic aftershocks sheared the concrete columns and collapsed the hospital wing "
        "into rubble, the paramedic crawled back under the falling beams to pull the trapped child to safety."
    )
    compound_unseen = (
        "While a magnitude-seven earthquake shattered the masonry apartment block and triggered a rockslide, "
        "a firefighter braced the crumbling doorway with his own shoulders so two elderly residents could escape."
    )

    with LocalModels(config, offline=True) as models:
        eval_before = evaluate_reflection(
            birth,
            compound_train,
            rules,
            models,
            event_id="ev-tier3-before",
            learned_leaves=learned_leaves,
        )
    print(
        f"Before Tier-3 growth: matched={eval_before.matched_concept} "
        f"conf={eval_before.confidence:.2f} clar={eval_before.clarity:.2f} "
        f"reason={eval_before.confusion_reason}"
    )
    print("Rule activations before:", eval_before.rule_activations)

    tier3_created = None
    if eval_before.needs_teacher:
        with DirectGGUFTeacher(TEACHER_GGUF, config["generation"]) as teacher:
            new_rule, learned_leaves, _ = request_teacher_growth(
                eval_before,
                rules,
                learned_leaves,
                teacher,
                role="teacher",
            )
        rules = (*rules, new_rule)
        tier3_created = {
            "new_rule": new_rule.to_dict(),
            "new_rule_tier": new_rule.tier,
            "latest_leaf": learned_leaves[-1].to_dict(),
            "latest_leaf_tier": learned_leaves[-1].tier,
        }
        print(
            f"Teacher grew Tier-{learned_leaves[-1].tier} leaf: {learned_leaves[-1].leaf_id} "
            f"-> rule {new_rule.concept_name} (tier={new_rule.tier})"
        )

    with LocalModels(config, offline=True) as models:
        eval_after_train = evaluate_reflection(
            birth,
            compound_train,
            rules,
            models,
            event_id="ev-tier3-after-train",
            learned_leaves=learned_leaves,
        )
        eval_after_unseen = evaluate_reflection(
            birth,
            compound_unseen,
            rules,
            models,
            event_id="ev-tier3-after-unseen",
            learned_leaves=learned_leaves,
        )
        parent_quake_check = evaluate_reflection(
            birth,
            "Without warning, a seven-point-two earthquake sheared the highway overpass off its concrete columns.",
            rules,
            models,
            event_id="ev-tier2-quake-check",
            learned_leaves=learned_leaves,
        )
        parent_bureaucracy_check = evaluate_reflection(
            birth,
            "The regional licensing office required applicants to mail duplicate notarized forms to three separate bureaus.",
            rules,
            models,
            event_id="ev-tier2-bur-check",
            learned_leaves=learned_leaves,
        )

        print(
            f"After Tier-3 growth (train): matched={eval_after_train.matched_concept} "
            f"conf={eval_after_train.confidence:.2f} clar={eval_after_train.clarity:.2f}"
        )
        print(
            f"After Tier-3 growth (unseen): matched={eval_after_unseen.matched_concept} "
            f"conf={eval_after_unseen.confidence:.2f} clar={eval_after_unseen.clarity:.2f}"
        )
        print(
            f"Parent Tier-2 earthquake check: matched={parent_quake_check.matched_concept} "
            f"conf={parent_quake_check.confidence:.2f} clar={parent_quake_check.clarity:.2f}"
        )
        print(
            f"Parent Tier-2 bureaucracy check: matched={parent_bureaucracy_check.matched_concept} "
            f"conf={parent_bureaucracy_check.confidence:.2f} clar={parent_bureaucracy_check.clarity:.2f}"
        )

        print("\n=== PART 2: CONTINUOUS-STATE SEQUENTIAL MEMORY (3-TURN SEQUENCE) ===")
        turn1_text = (
            "After the levee broke, the volunteer medic stayed behind on the flooded roof to help "
            "the trapped family into the rescue boat before climbing in herself."
        )
        turn2_text = (
            "Minutes later on the quiet pier, she sat catching her breath and checked that everyone "
            "had dry blankets."
        )
        turn3_text = (
            "Without warning, a seven-point-two earthquake sheared the highway overpass off its "
            "concrete columns and dropped the span into the gorge."
        )

        turn1 = evaluate_reflection(
            birth,
            turn1_text,
            rules,
            models,
            event_id="ev-seq-1",
            learned_leaves=learned_leaves,
            memory_state=None,
        )
        turn2_stateless = evaluate_reflection(
            birth,
            turn2_text,
            rules,
            models,
            event_id="ev-seq-2-stateless",
            learned_leaves=learned_leaves,
            memory_state=None,
        )
        turn2_stateful = evaluate_reflection(
            birth,
            turn2_text,
            rules,
            models,
            event_id="ev-seq-2-stateful",
            learned_leaves=learned_leaves,
            memory_state=turn1.next_memory_state,
        )
        turn3_stateful = evaluate_reflection(
            birth,
            turn3_text,
            rules,
            models,
            event_id="ev-seq-3-stateful",
            learned_leaves=learned_leaves,
            memory_state=turn2_stateful.next_memory_state,
        )

    print(
        f"Turn 1 (Anchor): matched={turn1.matched_concept} conf={turn1.confidence:.2f} "
        f"clar={turn1.clarity:.2f} val={turn1.valence:.2f} aro={turn1.arousal:.2f} "
        f"ret={turn1.retention:.2f} res={turn1.resonance:.2f}"
    )
    print(
        f"Turn 2 (Follow-up, Stateless): matched={turn2_stateless.matched_concept} "
        f"conf={turn2_stateless.confidence:.2f} clar={turn2_stateless.clarity:.2f} "
        f"confused={turn2_stateless.is_confused}"
    )
    print(
        f"Turn 2 (Follow-up, Stateful):  matched={turn2_stateful.matched_concept} "
        f"conf={turn2_stateful.confidence:.2f} clar={turn2_stateful.clarity:.2f} "
        f"val={turn2_stateful.valence:.2f} aro={turn2_stateful.arousal:.2f} "
        f"ret={turn2_stateful.retention:.2f} res={turn2_stateful.resonance:.2f} "
        f"confused={turn2_stateful.is_confused}"
    )
    print(
        f"Turn 3 (Sudden Quake Interrupt, Stateful): matched={turn3_stateful.matched_concept} "
        f"conf={turn3_stateful.confidence:.2f} clar={turn3_stateful.clarity:.2f} "
        f"ret={turn3_stateful.retention:.2f} res={turn3_stateful.resonance:.2f} "
        f"confused={turn3_stateful.is_confused}"
    )

    save_ruleset(
        repo_root / "outputs" / "continuous_tiered_ruleset.json",
        rules,
        learned_leaves=learned_leaves,
    )
    report = {
        "part1_tier3_sub_branching": {
            "before": {
                "matched_concept": eval_before.matched_concept,
                "confidence": eval_before.confidence,
                "clarity": eval_before.clarity,
                "confusion_reason": eval_before.confusion_reason,
                "rule_activations": eval_before.rule_activations,
            },
            "tier3_created": tier3_created,
            "after_train": {
                "matched_concept": eval_after_train.matched_concept,
                "confidence": eval_after_train.confidence,
                "clarity": eval_after_train.clarity,
                "rule_activations": eval_after_train.rule_activations,
            },
            "after_unseen": {
                "matched_concept": eval_after_unseen.matched_concept,
                "confidence": eval_after_unseen.confidence,
                "clarity": eval_after_unseen.clarity,
                "rule_activations": eval_after_unseen.rule_activations,
            },
            "parent_tier2_preservation": {
                "earthquake_matched": parent_quake_check.matched_concept,
                "earthquake_clarity": parent_quake_check.clarity,
                "bureaucracy_matched": parent_bureaucracy_check.matched_concept,
                "bureaucracy_clarity": parent_bureaucracy_check.clarity,
            },
        },
        "part2_continuous_memory": {
            "turn1_anchor": {
                "matched_concept": turn1.matched_concept,
                "confidence": turn1.confidence,
                "clarity": turn1.clarity,
                "valence": turn1.valence,
                "arousal": turn1.arousal,
                "retention": turn1.retention,
                "resonance": turn1.resonance,
            },
            "turn2_followup_stateless": {
                "matched_concept": turn2_stateless.matched_concept,
                "confidence": turn2_stateless.confidence,
                "clarity": turn2_stateless.clarity,
                "is_confused": turn2_stateless.is_confused,
                "confusion_reason": turn2_stateless.confusion_reason,
            },
            "turn2_followup_stateful": {
                "matched_concept": turn2_stateful.matched_concept,
                "confidence": turn2_stateful.confidence,
                "clarity": turn2_stateful.clarity,
                "valence": turn2_stateful.valence,
                "arousal": turn2_stateful.arousal,
                "retention": turn2_stateful.retention,
                "resonance": turn2_stateful.resonance,
                "is_confused": turn2_stateful.is_confused,
            },
            "turn3_interrupt_stateful": {
                "matched_concept": turn3_stateful.matched_concept,
                "confidence": turn3_stateful.confidence,
                "clarity": turn3_stateful.clarity,
                "valence": turn3_stateful.valence,
                "arousal": turn3_stateful.arousal,
                "retention": turn3_stateful.retention,
                "resonance": turn3_stateful.resonance,
                "is_confused": turn3_stateful.is_confused,
            },
        },
    }
    report_path = repo_root / "outputs" / "continuous_tiered_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    main()
