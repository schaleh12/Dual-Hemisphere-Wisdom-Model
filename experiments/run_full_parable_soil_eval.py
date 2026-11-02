"""Full 8-Beat Live Benchmark on 'The Basin of Quiet Stone' — Complete Right-Brain Soil.

Runs all 8 unchanged beats of 'The Basin of Quiet Stone' end-to-end with:
1. The 3 trained, genuinely separate Right-Brain neural bridge classifiers
   (`models/right_brain_bridges/*` -> `ThreeBridgeSensors`, zero classification on Qwen3-4B).
2. Explicit Tactical Hail + Pure Felt Chords + Pre-Verbal Chord-Rhyming & Chord-Inversion
   across memory (`ChordMemoryBank`).
3. Two-Layer `Contemplation (Clarity / Confidence)` Context Ledger:
   - Raw Thought Stream (`outputs/parable_8beat_raw_thoughts.jsonl`)
   - Live Context Ledger (`outputs/parable_8beat_contemplation_ledger.json`)
4. `Qwen3-4B` acting strictly as the plain-English conscious Left Brain feeding `Truth / Theory`,
   with external `Fact` strictly protected by `InternalSelfRegistry`.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from experiments.build_and_train_bridge_classifiers import EXTRA_CALIBRATION_CULTURAL_LEAVES
from experiments.run_parable_gear1_score import PARABLE_BEATS
from lmft_lab.birth import create_birth_state
from lmft_lab.chord_memory import ChordMemoryBank, save_chord_memory_bank
from lmft_lab.contemplation_ledger import ContemplationLedger, load_raw_thought_stream
from lmft_lab.local_models import LocalModels, load_config
from lmft_lab.neural_bridge_classifiers import (
    DEFAULT_COMPLEX_FEELING_LEAVES,
    NeuralThreeBridgeSuite,
)
from lmft_lab.reflection_classifier import load_learned_leaves, load_ruleset
from lmft_lab.right_brain_equilibrium import (
    QwenLeftBrain,
    SettledDiary,
    ThreeBridgeSensors,
    save_diary,
    settle_scenario,
)
from lmft_lab.runtime_store import atomic_json


def main() -> None:
    config = load_config(repo_root / "config" / "local_models.json")
    birth = create_birth_state("ind-parable-soil-8beat", birth_event_id="birth-soil-8", random_seed=42)

    ruleset_path = repo_root / "outputs" / "continuous_tiered_ruleset.json"
    rules = load_ruleset(ruleset_path)
    base_learned = load_learned_leaves(ruleset_path)
    learned_leaves = base_learned + EXTRA_CALIBRATION_CULTURAL_LEAVES + DEFAULT_COMPLEX_FEELING_LEAVES

    bridges_dir = repo_root / "models" / "right_brain_bridges"
    suite = NeuralThreeBridgeSuite.load_all(bridges_dir)
    base_neural_sensors = suite.as_three_bridge_sensors()

    # Wrap sensors with millisecond timing instrumentation so every sensor call is timed honestly
    sensor_call_latencies_ms: list[float] = []

    def _timed(fn):
        def _wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            sensor_call_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            return out

        return _wrapped

    timed_sensors = ThreeBridgeSensors(
        will_cultural=_timed(base_neural_sensors.will_cultural),
        truth_philosophical=_timed(base_neural_sensors.truth_philosophical),
        contemplation_experiential=_timed(base_neural_sensors.contemplation_experiential),
    )

    # Warm up CUDA kernels on a non-parable sentence before beat 1 timing
    suite.benchmark_latency_ms(
        "Crew members share tools openly on the pier.",
        learned_leaves=learned_leaves,
        warmup_runs=3,
        timed_runs=3,
    )
    sensor_call_latencies_ms.clear()

    raw_stream_path = repo_root / "outputs" / "parable_8beat_raw_thoughts.jsonl"
    if raw_stream_path.exists():
        raw_stream_path.unlink()
    ledger_path = repo_root / "outputs" / "parable_8beat_contemplation_ledger.json"
    chord_bank_path = repo_root / "outputs" / "parable_8beat_chord_memory.json"
    diary_path = repo_root / "outputs" / "parable_8beat_settled_diary.json"

    memory = None
    diary = SettledDiary(max_entries=5)
    chord_memory = ChordMemoryBank(max_entries=16)
    ledger = ContemplationLedger(max_certainties=5, max_uncertainties=2)

    beat_reports: list[dict] = []

    with LocalModels(config, offline=True) as models:
        left_brain = QwenLeftBrain(models)

        for item in PARABLE_BEATS:
            beat_num = int(item["beat"])
            phase = str(item["phase"])
            text = str(item["text"])

            sensor_call_latencies_ms.clear()
            t_beat_start = time.perf_counter()

            res = settle_scenario(
                birth,
                text,
                timed_sensors,
                rules,
                scene_index=beat_num,
                left_brain=left_brain,
                learned_leaves=learned_leaves,
                prior_memory=memory,
                diary=diary,
                chord_memory=chord_memory,
                contemplation_ledger=ledger,
                raw_thought_stream_path=raw_stream_path,
                contemplation_ledger_path=ledger_path,
                max_passes=3,
            )

            beat_total_ms = round((time.perf_counter() - t_beat_start) * 1000.0, 2)
            pass1_sensor_ms = round(sum(sensor_call_latencies_ms[:6]), 2) if res.left_brain_calls > 0 else round(sum(sensor_call_latencies_ms), 2)
            all_sensor_ms = round(sum(sensor_call_latencies_ms), 2)

            memory = res.next_memory
            diary = res.next_diary
            chord_memory = res.next_chord_memory
            ledger = res.next_ledger
            fv = res.final_vibe
            fc = res.felt_chord

            complex_feelings = {
                l.leaf_name: fv.coordinates.get(l.leaf_id, 0.0)
                for l in DEFAULT_COMPLEX_FEELING_LEAVES
            }

            print(f"\n=== BEAT {beat_num}: {phase} ===")
            print(
                f"Outcome: {res.outcome} | Settled: {res.settled} | "
                f"Self: {fv.self_registry.epistemic_state} | "
                f"Passes: {len(res.passes)} | Left-Brain Hails: {res.left_brain_calls} | "
                f"Active Concept: {fv.active_concept}"
            )
            print(
                f"Timing: Neural Sensors={all_sensor_ms:.1f} ms (Pass1~{pass1_sensor_ms:.1f} ms) | "
                f"Total Beat={beat_total_ms:.1f} ms"
            )
            print(
                f"Do-er/Thinker Coordinates: "
                f"Fair={fv.coordinates['cultural/justice/fairness']:.2f} "
                f"Hon={fv.coordinates['cultural/justice/honor']:.2f} "
                f"Pow={fv.coordinates['cultural/efficiency/power']:.2f} "
                f"Gate={fv.coordinates.get('cultural/justice/exclusionary_gatekeeping', 0.0):.2f} | "
                f"Ord={fv.coordinates['philosophical/structure/order']:.2f} "
                f"Cnt={fv.coordinates['philosophical/persistence/continuity']:.2f} "
                f"Dsc={fv.coordinates['philosophical/persistence/discreteness']:.2f} "
                f"Cha={fv.coordinates['philosophical/structure/chaos']:.2f} "
                f"Collapse={fv.coordinates.get('philosophical/structure/tectonic_collapse', 0.0):.2f}"
            )
            print(
                f"Feeler Chord: Sal={fv.salience:.2f} Val={fv.valence:.2f} Aro={fv.arousal:.2f} | "
                f"Sanctuary={complex_feelings['austere_sanctuary']:.2f} "
                f"Hollow={complex_feelings['hollow_stillness']:.2f} "
                f"Release={complex_feelings['bittersweet_release']:.2f} "
                f"Return={complex_feelings['earned_return']:.2f}"
            )
            if fc is not None:
                print(f"Felt Atmosphere: {fc.dominant_mood}")
            if res.chord_echoes and res.chord_echoes.has_echoes():
                print(f"Pre-Verbal Memory Echoes:\n{res.chord_echoes.render_for_left_brain()}")
            if res.final_conscious_note:
                print(f"Left-Brain Final Self-Talk: {res.final_conscious_note}")

            beat_reports.append({
                "beat": beat_num,
                "phase": phase,
                "text": text,
                "settled": res.settled,
                "outcome": res.outcome,
                "needs_new_instinct": res.needs_new_instinct,
                "passes_used": len(res.passes),
                "left_brain_hails": res.left_brain_calls,
                "active_concept": fv.active_concept,
                "neural_sensor_total_ms": all_sensor_ms,
                "beat_wall_ms": beat_total_ms,
                "felt_chord": fc.to_dict() if fc else None,
                "complex_feelings": complex_feelings,
                "chord_echoes": res.chord_echoes.to_dict() if res.chord_echoes else None,
                "last_tactical_hail": res.last_tactical_hail.to_dict() if res.last_tactical_hail else None,
                "final_conscious_note": res.final_conscious_note,
                "self_registry": fv.self_registry.to_dict(),
                "competing_bids": fv.synthesis.to_dict(),
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
                        "clarity": pt.vibe.clarity,
                        "confidence": pt.vibe.confidence,
                        "fact": pt.vibe.fact,
                        "theory": pt.vibe.theory,
                        "max_theory_delta": pt.max_theory_delta,
                        "left_brain_reply": pt.left_brain_reply,
                    }
                    for pt in res.passes
                ],
            })

    save_diary(diary_path, diary)
    save_chord_memory_bank(chord_bank_path, chord_memory)
    archived_raw_thoughts = load_raw_thought_stream(raw_stream_path)

    valences = [b["experiential"]["valence"] for b in beat_reports]
    summary_stats = {
        "total_beats": len(beat_reports),
        "settled_beats": [b["beat"] for b in beat_reports if b["settled"]],
        "unresolved_beats_flagged_for_sleep_mode": [b["beat"] for b in beat_reports if b["needs_new_instinct"]],
        "total_left_brain_hails": sum(b["left_brain_hails"] for b in beat_reports),
        "archived_raw_thought_records": len(archived_raw_thoughts),
        "final_ledger_settled_certainties": len(ledger.settled_certainties),
        "final_ledger_active_uncertainties": len(ledger.active_uncertainties),
        "valence_curve_across_8_beats": valences,
        "valence_min": round(min(valences), 4),
        "valence_max": round(max(valences), 4),
        "valence_spread": round(max(valences) - min(valences), 4),
        "false_earthquake_alarm_on_beat_8": beat_reports[-1]["active_concept"] == "catastrophic_geophysical_disruption",
    }

    report_path = repo_root / "outputs" / "parable_soil_8beat_report.json"
    atomic_json(
        report_path,
        {
            "title": "The Basin of Quiet Stone — Full 8-Beat Right-Brain Soil Benchmark",
            "summary": summary_stats,
            "final_contemplation_ledger": ledger.to_dict(),
            "beats": beat_reports,
        },
    )
    print("\n=== 8-BEAT SUMMARY ===")
    print(json.dumps(summary_stats, indent=2))
    print(f"Saved full 8-beat report to {report_path}")


if __name__ == "__main__":
    main()
