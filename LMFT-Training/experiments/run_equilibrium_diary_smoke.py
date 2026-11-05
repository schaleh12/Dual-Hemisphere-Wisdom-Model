"""Live GPU Smoke Check for the Right-Brain Equilibrium Loop + Settled Diary.

Runs 4 unchanged beats from 'The Basin of Quiet Stone' (Beats 1, 4, 5, and 8)
through `settle_scenario` using:
- `make_temporary_llm_bridge_sensors` (split 3-bridge fallback before neural heads)
- `QwenLeftBrain` (plain-English conscious Left Brain, zero JSON float output)
- `SettledDiary` (rolling settled-vibe diary carried across scenes)
"""

import json
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from lmft_architecture.birth import create_birth_state
from lmft_architecture.local_models import LocalModels, load_config
from lmft_architecture.reflection_classifier import load_learned_leaves, load_ruleset
from lmft_architecture.right_brain_equilibrium import (
    QwenLeftBrain,
    SettledDiary,
    make_temporary_llm_bridge_sensors,
    save_diary,
    settle_scenario,
)

SMOKE_BEATS = [
    (
        1,
        "True Sanctuary (Open Spring)",
        "Where the salt wind strips the scrub down to white bone, the mason knelt in the shale "
        "and mortared a low half-ring of river rock around the seep, leaving the south lip open "
        "to the trail so mules and strangers could drink without asking.",
    ),
    (
        4,
        "Hardening Ritual (Three Carved Rules)",
        "By winter he had carved three rules into the lintel—no iron canteens, no dusty hems, "
        "and no drawing water after the bell—turning back an old herder whose goats smelled of "
        "the lower marsh.",
    ),
    (
        5,
        "Sealed Enclosure (The Yellow Wall)",
        "Convinced that even human breath troubled the surface, he laid dressed yellow blocks "
        "across the doorway from within, leaving only a thumb-wide slit near the roof to watch "
        "the silver skin of the water lie motionless.",
    ),
    (
        8,
        "Shatter & Release (Return to Flow)",
        "At dawn the yellow wall split from footing to crown with a single dry crack, and by "
        "noon the thawing meltwater was braiding freely through the rubble into the thorn-grass "
        "where three stray kids stood drinking.",
    ),
]


def main() -> None:
    config = load_config(repo_root / "config" / "local_models.json")
    birth = create_birth_state("ind-eq-smoke", birth_event_id="birth-eq-1", random_seed=42)
    ruleset_path = repo_root / "outputs" / "continuous_tiered_ruleset.json"
    rules = load_ruleset(ruleset_path)
    learned_leaves = load_learned_leaves(ruleset_path)

    memory = None
    diary = SettledDiary(max_entries=5)
    scene_reports = []

    with LocalModels(config, offline=True) as models:
        sensors = make_temporary_llm_bridge_sensors(models)
        left_brain = QwenLeftBrain(models)

        for idx, (beat_num, phase, text) in enumerate(SMOKE_BEATS, start=1):
            res = settle_scenario(
                birth,
                text,
                sensors,
                rules,
                scene_index=idx,
                left_brain=left_brain,
                learned_leaves=learned_leaves,
                prior_memory=memory,
                diary=diary,
                max_passes=3,
            )
            memory = res.next_memory
            diary = res.next_diary
            fv = res.final_vibe

            print(f"\n=== SCENE {idx} (Parable Beat {beat_num}: {phase}) ===")
            print(
                f"Outcome: {res.outcome} | Settled: {res.settled} | "
                f"Self: {fv.self_registry.epistemic_state} | "
                f"Passes: {len(res.passes)} | Left-Brain Calls: {res.left_brain_calls} | "
                f"Active Concept: {fv.active_concept}"
            )
            print(
                f"External Fact vs Theory: Fact={fv.fact:.2f} Theory={fv.theory:.2f} | "
                f"Order={fv.coordinates['philosophical/structure/order']:.2f} "
                f"Fair={fv.coordinates['cultural/justice/fairness']:.2f} "
                f"Disc={fv.coordinates['philosophical/persistence/discreteness']:.2f} "
                f"Valence={fv.valence:.2f} Exec={fv.execution:.2f}"
            )
            print(
                f"3 Personas: Do-er={fv.synthesis.doer.stance}({fv.synthesis.doer.strength:.2f}) | "
                f"Thinker={fv.synthesis.thinker.stance}({fv.synthesis.thinker.strength:.2f}) | "
                f"Feeler={fv.synthesis.feeler.stance}({fv.synthesis.feeler.strength:.2f}) | "
                f"Harmony={fv.synthesis.harmony:.2f}"
            )
            if fv.tug_reasons:
                print(f"Tug Reasons: {list(fv.tug_reasons)}")
            if res.final_conscious_note:
                print(f"Left-Brain Diary Note: {res.final_conscious_note}")

            scene_reports.append({
                "scene_index": idx,
                "parable_beat": beat_num,
                "phase": phase,
                "text": text,
                "settled": res.settled,
                "outcome": res.outcome,
                "needs_new_instinct": res.needs_new_instinct,
                "passes_used": len(res.passes),
                "left_brain_calls": res.left_brain_calls,
                "active_concept": fv.active_concept,
                "final_conscious_note": res.final_conscious_note,
                "self_registry": fv.self_registry.to_dict(),
                "competing_bids": fv.synthesis.to_dict(),
                "tug_reasons": list(fv.tug_reasons),
                "external_coordinates": fv.coordinates,
                "theory_coordinates": fv.theory_coordinates,
                "experiential": {
                    "novelty": fv.novelty,
                    "salience": fv.salience,
                    "valence": fv.valence,
                    "arousal": fv.arousal,
                    "resonance": fv.resonance,
                    "retention": fv.retention,
                },
                "psychological": {
                    "clarity": fv.clarity,
                    "confidence": fv.confidence,
                    "theory": fv.theory,
                    "fact": fv.fact,
                    "intention": fv.intention,
                    "execution": fv.execution,
                },
                "pass_history": [
                    {
                        "pass_index": pt.pass_index,
                        "posture": pt.vibe.posture,
                        "active_concept": pt.vibe.active_concept,
                        "tug_reasons": list(pt.vibe.tug_reasons),
                        "max_theory_delta": pt.max_theory_delta,
                        "left_brain_reply": pt.left_brain_reply,
                    }
                    for pt in res.passes
                ],
            })

    save_diary(repo_root / "outputs" / "equilibrium_smoke_diary.json", diary)
    out_path = repo_root / "outputs" / "equilibrium_diary_smoke_report.json"
    out_path.write_text(json.dumps({"scenes": scene_reports}, indent=2), encoding="utf-8")
    print(f"\nSaved smoke report to {out_path}")


if __name__ == "__main__":
    main()
