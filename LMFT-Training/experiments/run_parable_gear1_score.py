"""Gear 1 Live Right-Brain Parable Probe: 'The Basin of Quiet Stone' (8 Beats).

Feeds one continuous, literary 8-beat parable (zero textbook keywords—pure concrete imagery where
Titan-like stewardship slowly hardens into Yellow-Rook isolated hoarding until black frost splits
the wall open) into Qwen3-4B + the 24-Leaf Fuzzy Tree.

Records all 24 founding leaves across all 8 beats with ZERO Left-Brain announcer speech, tracking:
1. AT_EASE (Contemplation <-> Experiential): Balanced/low-tension silent absorption.
2. WORDLESS_REFLEX (Will <-> Cultural): Direct non-verbal stance/action on known disruption/imbalance.
3. FORGE_NEW_INSTINCT (Truth <-> Philosophical): High-salience unrecognized tension flagged for Sleep Mode.
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lmft_architecture.birth import create_birth_state
from lmft_architecture.local_models import LocalModels, load_config
from lmft_architecture.reflection_classifier import (
    FOUNDATIONAL_LEAVES,
    evaluate_reflection,
    load_learned_leaves,
    load_ruleset,
)


PARABLE_BEATS = [
    {
        "beat": 1,
        "phase": "True Sanctuary (Open Spring)",
        "text": (
            "Where the salt wind strips the scrub down to white bone, the mason knelt in the shale "
            "and mortared a low half-ring of river rock around the seep, leaving the south lip open "
            "to the trail so mules and strangers could drink without asking."
        ),
    },
    {
        "beat": 2,
        "phase": "Shared Rhythm (Clear Reflection)",
        "text": (
            "At dusk the drovers rinsed their tin cups in the overflow trough before dipping from "
            "the center, and each morning the basin stood clear again, reflecting the first pale "
            "strip of sky."
        ),
    },
    {
        "beat": 3,
        "phase": "First Gate (The Reed Lattice)",
        "text": (
            "After an autumn dust-blow left red silt on the bottom step, the mason hung a woven reed "
            "lattice across the south gap and began asking each traveler where they had slept the "
            "night before before lifting the latch."
        ),
    },
    {
        "beat": 4,
        "phase": "Hardening Ritual (Three Carved Rules)",
        "text": (
            "By winter he had carved three rules into the lintel—no iron canteens, no dusty hems, "
            "and no drawing water after the bell—turning back an old herder whose goats smelled of "
            "the lower marsh."
        ),
    },
    {
        "beat": 5,
        "phase": "Sealed Enclosure (The Yellow Wall)",
        "text": (
            "Convinced that even human breath troubled the surface, he laid dressed yellow blocks "
            "across the doorway from within, leaving only a thumb-wide slit near the roof to watch "
            "the silver skin of the water lie motionless."
        ),
    },
    {
        "beat": 6,
        "phase": "Pious Blindness (Thirst Outside)",
        "text": (
            "Outside, dry hooves scraped against the mortar in the heat and moved on; inside the "
            "cool dark, the mason wiped a single speck of lime from the rim and whispered how "
            "faithfully he had kept the gift unspoiled."
        ),
    },
    {
        "beat": 7,
        "phase": "Brittle Frost (Stagnant Freeze)",
        "text": (
            "Because no warm hands dipped into the basin and no overflow ran down the spillway, "
            "the dead-still water froze solid to the bedrock on the first black frost, expanding "
            "silently against the mortared seams."
        ),
    },
    {
        "beat": 8,
        "phase": "Shatter & Release (Return to Flow)",
        "text": (
            "At dawn the yellow wall split from footing to crown with a single dry crack, and by "
            "noon the thawing meltwater was braiding freely through the rubble into the thorn-grass "
            "where three stray kids stood drinking."
        ),
    },
]


def bar(val: float, width: int = 10) -> str:
    filled = int(round(max(0.0, min(1.0, val)) * width))
    return "#" * filled + "." * (width - filled)


def right_brain_posture(ev) -> str:
    """Map ExecutiveDecision into the 3 wordless Right-Brain postures."""
    mode = ev.executive_decision.mode
    if mode == "STAY_SILENT":
        return "1_AT_EASE"
    if mode in ("COMMIT_FAST", "VERIFY_HYPOTHESIS"):
        return f"2_WORDLESS_REFLEX({ev.executive_decision.active_concept})"
    return "3_FORGE_NEW_INSTINCT"


def main():
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "config" / "local_models.json")
    birth = create_birth_state("ind-parable-gear1", birth_event_id="birth-parable-1", random_seed=42)

    ruleset_path = repo_root / "outputs" / "continuous_tiered_ruleset.json"
    rules = load_ruleset(ruleset_path)
    learned_leaves = load_learned_leaves(ruleset_path)

    memory_state = None
    beat_records = []

    with LocalModels(config, offline=True) as models:
        for item in PARABLE_BEATS:
            ev = evaluate_reflection(
                birth,
                item["text"],
                rules,
                models,
                event_id=f"ev-basin-beat-{item['beat']}",
                learned_leaves=learned_leaves,
                memory_state=memory_state,
                allow_silence=True,
                branch_gating=True,
            )
            memory_state = ev.next_memory_state
            coords = ev.slm_output["leaf_coordinates"]
            posture = right_brain_posture(ev)

            print(f"\n=== BEAT {item['beat']}: {item['phase']} ===")
            print(f"Posture: {posture} | Top Rule: {ev.matched_concept} (conf={ev.confidence:.2f}, clar={ev.clarity:.2f})")
            print(
                f"Cultural:      Pow={coords['cultural/efficiency/power']:.2f} "
                f"Wst={coords['cultural/efficiency/wastelessness']:.2f} | "
                f"Hon={coords['cultural/justice/honor']:.2f} "
                f"Fair={coords['cultural/justice/fairness']:.2f} | "
                f"Self={coords['cultural/freedom/self-sufficiency']:.2f} "
                f"Need={coords['cultural/freedom/needlessness']:.2f}"
            )
            print(
                f"Philosophical: Agn={coords['philosophical/being/agency']:.2f} "
                f"Exi={coords['philosophical/being/existence']:.2f} | "
                f"Ord={coords['philosophical/structure/order']:.2f} "
                f"Cha={coords['philosophical/structure/chaos']:.2f} | "
                f"Cnt={coords['philosophical/persistence/continuity']:.2f} "
                f"Dsc={coords['philosophical/persistence/discreteness']:.2f}"
            )
            print(
                f"Experiential:  Nov={ev.novelty:.2f} Sal={ev.salience:.2f} | "
                f"Val={ev.valence:.2f} Aro={ev.arousal:.2f} | "
                f"Res={ev.resonance:.2f} Ret={ev.retention:.2f}"
            )
            print(
                f"Psychological: Int={ev.intention:.2f} Exe={ev.execution:.2f} | "
                f"The={ev.theory:.2f} Fac={ev.fact:.2f} | "
                f"Cla={ev.clarity:.2f} Cnf={ev.confidence:.2f}"
            )

            beat_records.append({
                "beat": item["beat"],
                "phase": item["phase"],
                "text": item["text"],
                "posture": posture,
                "executive_mode": ev.executive_decision.mode,
                "active_concept": ev.executive_decision.active_concept,
                "matched_concept": ev.matched_concept,
                "confusion_reason": ev.confusion_reason,
                "founding_coordinates": {k: coords[k] for k in FOUNDATIONAL_LEAVES},
                "learned_coordinates": {
                    l.leaf_id: coords.get(l.leaf_id, 0.0) for l in learned_leaves
                },
                "experiential": {
                    "novelty": ev.novelty,
                    "salience": ev.salience,
                    "valence": ev.valence,
                    "arousal": ev.arousal,
                    "resonance": ev.resonance,
                    "retention": ev.retention,
                },
                "psychological": {
                    "intention": ev.intention,
                    "execution": ev.execution,
                    "theory": ev.theory,
                    "fact": ev.fact,
                    "clarity": ev.clarity,
                    "confidence": ev.confidence,
                },
                "rule_activations": ev.rule_activations,
                "queried_child_leaves": list(ev.queried_child_leaves),
                "skipped_child_leaves": list(ev.skipped_child_leaves),
            })

    out_path = repo_root / "outputs" / "parable_gear1_basin_report.json"
    out_path.write_text(json.dumps({"title": "The Basin of Quiet Stone", "beats": beat_records}, indent=2), encoding="utf-8")
    print(f"\nSaved full 8-beat 24-leaf score to {out_path}")


if __name__ == "__main__":
    main()
