"""Live Dual-Hemisphere Benchmark: 3-Branch Right-Brain Archives & Self-Talk Competition.

Runs three honest evaluations:
1. Epoch 1 (Cold Learning — 8-Beat 'The Basin of Quiet Stone'):
   - Starts with empty branch archives (`cultural_history.json`, `experiential_felt_diary.json`,
     `philosophical_lexicon.json`).
   - Shows where the 3 Right-Brain classifiers deadlock and hail Qwen3-4B, where the 3 classifiers
     deliberate and settle subsequent beats solo in ~25-35 ms with 0 LLM calls, how `cultural_history.json`
     catches Beat 6's ongoing enclosure via social-drift precedent tracking, and how Beat 8 triggers
     Open-Thread Closure in Qwen3-4B to resolve Beat 3's open uncertainty.
2. Epoch 2A (Consolidated Parable Re-Read — 0 LLM Calls):
   - Runs all 8 parable beats against the populated 3-branch archives with `left_brain=None`
     to verify millisecond Right-Brain 3-Classifier Deliberation across the full story.
3. Epoch 2B (Anti-Holmes Unseen 4-Beat Out-of-Domain Story — 'The Harbor Bakehouse', 0 LLM Calls):
   - Tests the exact same populated 3-branch archives on a brand-new 4-beat story in a coastal
     bakehouse setting (open bread oven -> bolted iron shutter & brass tally -> quiet baker inside
     while cold oarsmen wait outside -> jammed flue cracking open & shared warmth returning) with
     `left_brain=None` (zero LLM calls) to measure true out-of-domain generalization honestly.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from experiments.build_and_train_bridge_classifiers import EXTRA_CALIBRATION_CULTURAL_LEAVES
from experiments.run_parable_gear1_score import PARABLE_BEATS
from lmft_lab.birth import create_birth_state
from lmft_lab.branch_archives import ThreeBranchArchives
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

UNSEEN_BAKEHOUSE_BEATS: tuple[dict[str, object], ...] = (
    {
        "beat": 1,
        "phase": "Open Shared Hearth",
        "text": (
            "At daybreak the stone bakehouse by the pier stood open to the salt wind, "
            "and wet deckhands shared warm rye loaves beside the brick oven."
        ),
    },
    {
        "beat": 2,
        "phase": "Bolted Iron Shutter & Tally",
        "text": (
            "By November the clerk bolted an iron shutter across the oven window and demanded "
            "a stamped copper tally before anyone could stand near the heat, leaving shivering "
            "net-menders outside on the sleet-slick planks."
        ),
    },
    {
        "beat": 3,
        "phase": "Quiet Baker Inside Locked Room (Social Drift Check)",
        "text": (
            "Outside, frozen boots shuffled past the shuttered brickwork in the sleet; "
            "inside the quiet room, the baker brushed a single pinch of flour from his apron "
            "and aligned the brass weights on the shelf."
        ),
    },
    {
        "beat": 4,
        "phase": "Jammed Flue Splits Open Into Shared Warmth",
        "text": (
            "When the overheated flue latch finally snapped and flung the iron shutter wide, "
            "warm cedar smoke and fresh loaves poured onto the wharf; the shivering net-menders "
            "gathered close to eat while the baker sat quietly by the open threshold."
        ),
    },
)


def _format_beat_record(
    beat_num: int,
    phase: str,
    text: str,
    res,
    all_sensor_ms: float,
    beat_total_ms: float,
) -> dict:
    fv = res.final_vibe
    fc = res.felt_chord
    complex_feelings = {
        l.leaf_name: fv.coordinates.get(l.leaf_id, 0.0)
        for l in DEFAULT_COMPLEX_FEELING_LEAVES
    }
    return {
        "beat": beat_num,
        "phase": phase,
        "text": text,
        "settled": res.settled,
        "outcome": res.outcome,
        "audience_mode": res.audience_mode,
        "needs_new_instinct": res.needs_new_instinct,
        "passes_used": len(res.passes),
        "left_brain_hails": res.left_brain_calls,
        "active_concept": fv.active_concept,
        "neural_sensor_total_ms": all_sensor_ms,
        "beat_wall_ms": beat_total_ms,
        "social_drift": res.social_drift.to_dict() if res.social_drift else None,
        "open_thread": res.open_thread.to_dict() if res.open_thread else None,
        "deliberation": res.deliberation.to_dict() if res.deliberation else None,
        "felt_chord": fc.to_dict() if fc else None,
        "complex_feelings": complex_feelings,
        "chord_echoes": res.chord_echoes.to_dict() if res.chord_echoes else None,
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
    }


def _print_beat_summary(prefix: str, rec: dict) -> None:
    delib = rec.get("deliberation")
    votes_str = "none"
    if delib:
        v = delib["votes"]
        s = delib["similarities"]
        votes_str = (
            f"ratified={delib['ratified']} "
            f"[Exec={v['executive_doer']}({s['executive']:.2f}), "
            f"Leg={v['legislative_feeler']}({s['legislative']:.2f}), "
            f"Jud={v['judicial_thinker']}({s['judicial']:.2f})]"
        )
    drift_flag = rec["social_drift"]["drift_detected"] if rec.get("social_drift") else False
    open_flag = rec["open_thread"]["triggered"] if rec.get("open_thread") else False
    cf = rec["complex_feelings"]
    coords = rec["external_coordinates"]
    print(f"\n=== {prefix} BEAT {rec['beat']}: {rec['phase']} ===")
    print(
        f"Audience: {rec['audience_mode']} | Outcome: {rec['outcome']} | Settled: {rec['settled']} | "
        f"Left-Brain Calls: {rec['left_brain_hails']} | Active Concept: {rec['active_concept']}"
    )
    print(
        f"3-Branch Deliberation: {votes_str} | Social Drift: {drift_flag} | Open-Thread Trigger: {open_flag}"
    )
    print(
        f"Timing: Neural Sensors={rec['neural_sensor_total_ms']:.1f} ms | Total Beat={rec['beat_wall_ms']:.1f} ms"
    )
    print(
        f"Coordinates: Fair={coords['cultural/justice/fairness']:.2f} "
        f"Gate={coords.get('cultural/justice/exclusionary_gatekeeping', 0.0):.2f} "
        f"Ord={coords['philosophical/structure/order']:.2f} "
        f"Dsc={coords['philosophical/persistence/discreteness']:.2f} "
        f"Cha={coords['philosophical/structure/chaos']:.2f} | "
        f"Do-er Stance={rec['competing_bids']['doer_will_cultural']['stance']}"
    )
    print(
        f"Feeler: Val={rec['experiential']['valence']:.2f} | "
        f"Sanctuary={cf['austere_sanctuary']:.2f} "
        f"Hollow={cf['hollow_stillness']:.2f} "
        f"Release={cf['bittersweet_release']:.2f} "
        f"Return={cf['earned_return']:.2f}"
    )
    if rec.get("final_conscious_note"):
        print(f"Left-Brain Self-Talk: {rec['final_conscious_note']}")


def main() -> None:
    config = load_config(repo_root / "config" / "local_models.json")
    birth = create_birth_state("ind-dual-hemisphere-8beat", birth_event_id="birth-dh-8", random_seed=42)

    ruleset_path = repo_root / "outputs" / "continuous_tiered_ruleset.json"
    rules = load_ruleset(ruleset_path)
    base_learned = load_learned_leaves(ruleset_path)
    learned_leaves = base_learned + EXTRA_CALIBRATION_CULTURAL_LEAVES + DEFAULT_COMPLEX_FEELING_LEAVES

    bridges_dir = repo_root / "models" / "right_brain_bridges"
    suite = NeuralThreeBridgeSuite.load_all(bridges_dir)
    base_neural_sensors = suite.as_three_bridge_sensors()

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

    suite.benchmark_latency_ms(
        "Crew members share tools openly on the pier.",
        learned_leaves=learned_leaves,
        warmup_runs=3,
        timed_runs=3,
    )
    sensor_call_latencies_ms.clear()

    archives_dir = repo_root / "outputs" / "branch_archives"
    raw_stream_path = repo_root / "outputs" / "parable_8beat_raw_thoughts.jsonl"
    if raw_stream_path.exists():
        raw_stream_path.unlink()
    ledger_path = repo_root / "outputs" / "parable_8beat_contemplation_ledger.json"
    chord_bank_path = repo_root / "outputs" / "parable_8beat_chord_memory.json"
    diary_path = repo_root / "outputs" / "parable_8beat_settled_diary.json"

    # =========================================================================
    # EPOCH 1: Cold Learning on 8-Beat Parable (Empty Branch Archives)
    # =========================================================================
    print("\n" + "=" * 78)
    print("EPOCH 1: COLD LEARNING ON 'THE BASIN OF QUIET STONE' (EMPTY ARCHIVES)")
    print("=" * 78)

    memory_e1 = None
    diary_e1 = SettledDiary(max_entries=5)
    chord_memory_e1 = ChordMemoryBank(max_entries=16)
    ledger_e1 = ContemplationLedger(max_certainties=5, max_uncertainties=2)
    archives_e1 = ThreeBranchArchives()
    epoch1_reports: list[dict] = []

    with LocalModels(config, offline=True) as models:
        left_brain = QwenLeftBrain(models)

        for item in PARABLE_BEATS:
            beat_num = int(item["beat"])
            phase = str(item["phase"])
            text = str(item["text"])

            sensor_call_latencies_ms.clear()
            t0 = time.perf_counter()

            res = settle_scenario(
                birth,
                text,
                timed_sensors,
                rules,
                scene_index=beat_num,
                left_brain=left_brain,
                learned_leaves=learned_leaves,
                prior_memory=memory_e1,
                diary=diary_e1,
                chord_memory=chord_memory_e1,
                contemplation_ledger=ledger_e1,
                branch_archives=archives_e1,
                raw_thought_stream_path=raw_stream_path,
                contemplation_ledger_path=ledger_path,
                branch_archives_dir=archives_dir,
                require_self_critique=True,
                max_passes=3,
            )

            beat_total_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            all_sensor_ms = round(sum(sensor_call_latencies_ms), 2)

            memory_e1 = res.next_memory
            diary_e1 = res.next_diary
            chord_memory_e1 = res.next_chord_memory
            ledger_e1 = res.next_ledger
            archives_e1 = res.next_archives

            rec = _format_beat_record(beat_num, phase, text, res, all_sensor_ms, beat_total_ms)
            epoch1_reports.append(rec)
            _print_beat_summary("EPOCH 1", rec)

    save_diary(diary_path, diary_e1)
    save_chord_memory_bank(chord_bank_path, chord_memory_e1)
    archived_raw_thoughts = load_raw_thought_stream(raw_stream_path)

    # Reload populated 3-branch archives from disk to prove persistence
    consolidated_archives = ThreeBranchArchives.load_all(archives_dir)

    # =========================================================================
    # EPOCH 2A: Consolidated Re-Read of 8-Beat Parable (0 LLM Calls!)
    # =========================================================================
    print("\n" + "=" * 78)
    print("EPOCH 2A: CONSOLIDATED RE-READ OF 8-BEAT PARABLE (0 QWEN3-4B CALLS)")
    print("=" * 78)

    memory_e2 = None
    diary_e2 = SettledDiary(max_entries=5)
    chord_memory_e2 = ChordMemoryBank(max_entries=16)
    ledger_e2 = ledger_e1  # Carries the settled certainties (0 open uncertainties)
    # Reset executive precedents chronological list for fresh story pass while keeping learned clauses
    running_archives_e2 = ThreeBranchArchives(
        executive=consolidated_archives.executive.__class__(
            precedents=(),
            action_clauses=consolidated_archives.executive.action_clauses,
            max_entries=consolidated_archives.executive.max_entries,
        ),
        legislative=consolidated_archives.legislative,
        judicial=consolidated_archives.judicial,
    )
    epoch2a_reports: list[dict] = []

    for item in PARABLE_BEATS:
        beat_num = int(item["beat"])
        phase = str(item["phase"])
        text = str(item["text"])

        sensor_call_latencies_ms.clear()
        t0 = time.perf_counter()

        res = settle_scenario(
            birth,
            text,
            timed_sensors,
            rules,
            scene_index=beat_num,
            left_brain=None,  # Strictly 0 LLM calls allowed in consolidated re-read!
            learned_leaves=learned_leaves,
            prior_memory=memory_e2,
            diary=diary_e2,
            chord_memory=chord_memory_e2,
            contemplation_ledger=ledger_e2,
            branch_archives=running_archives_e2,
            max_passes=3,
        )

        beat_total_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        all_sensor_ms = round(sum(sensor_call_latencies_ms), 2)

        memory_e2 = res.next_memory
        diary_e2 = res.next_diary
        chord_memory_e2 = res.next_chord_memory
        ledger_e2 = res.next_ledger
        running_archives_e2 = res.next_archives

        rec = _format_beat_record(beat_num, phase, text, res, all_sensor_ms, beat_total_ms)
        epoch2a_reports.append(rec)
        _print_beat_summary("EPOCH 2A", rec)

    # =========================================================================
    # EPOCH 2B: Unseen 4-Beat Out-of-Domain Generalization ('The Harbor Bakehouse', 0 LLM Calls!)
    # =========================================================================
    print("\n" + "=" * 78)
    print("EPOCH 2B: UNSEEN 4-BEAT OUT-OF-DOMAIN STORY ('THE HARBOR BAKEHOUSE', 0 LLM CALLS)")
    print("=" * 78)

    memory_u = None
    diary_u = SettledDiary(max_entries=5)
    chord_memory_u = ChordMemoryBank(max_entries=16)
    ledger_u = ContemplationLedger(max_certainties=5, max_uncertainties=2)
    running_archives_u = ThreeBranchArchives(
        executive=consolidated_archives.executive.__class__(
            precedents=(),
            action_clauses=consolidated_archives.executive.action_clauses,
            max_entries=consolidated_archives.executive.max_entries,
        ),
        legislative=consolidated_archives.legislative,
        judicial=consolidated_archives.judicial,
    )
    unseen_reports: list[dict] = []

    for item in UNSEEN_BAKEHOUSE_BEATS:
        beat_num = int(item["beat"])
        phase = str(item["phase"])
        text = str(item["text"])

        sensor_call_latencies_ms.clear()
        t0 = time.perf_counter()

        res = settle_scenario(
            birth,
            text,
            timed_sensors,
            rules,
            scene_index=beat_num,
            left_brain=None,  # Strictly 0 LLM calls allowed on unseen domain test!
            learned_leaves=learned_leaves,
            prior_memory=memory_u,
            diary=diary_u,
            chord_memory=chord_memory_u,
            contemplation_ledger=ledger_u,
            branch_archives=running_archives_u,
            max_passes=3,
        )

        beat_total_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        all_sensor_ms = round(sum(sensor_call_latencies_ms), 2)

        memory_u = res.next_memory
        diary_u = res.next_diary
        chord_memory_u = res.next_chord_memory
        ledger_u = res.next_ledger
        running_archives_u = res.next_archives

        rec = _format_beat_record(beat_num, phase, text, res, all_sensor_ms, beat_total_ms)
        unseen_reports.append(rec)
        _print_beat_summary("EPOCH 2B (UNSEEN)", rec)

    summary = {
        "experiment": "dual_hemisphere_3branch_archives_and_self_talk_eval",
        "epoch1_cold_learning": {
            "settled_beats": [b["beat"] for b in epoch1_reports if b["settled"]],
            "unresolved_beats": [b["beat"] for b in epoch1_reports if not b["settled"]],
            "left_brain_hails_by_beat": {b["beat"]: b["left_brain_hails"] for b in epoch1_reports},
            "total_left_brain_calls": sum(b["left_brain_hails"] for b in epoch1_reports),
            "audience_modes": {b["beat"]: b["audience_mode"] for b in epoch1_reports},
            "beat_6_social_drift_caught": bool(
                epoch1_reports[5]["social_drift"] and epoch1_reports[5]["social_drift"]["drift_detected"]
            ),
            "beat_2_false_inversion_fixed": bool(
                epoch1_reports[1]["chord_echoes"]["chord_inversion"] is None
            ),
            "beat_8_open_thread_closure_triggered": bool(
                epoch1_reports[7]["open_thread"] and epoch1_reports[7]["open_thread"]["triggered"]
            ),
            "final_open_uncertainties_after_beat_8": len(ledger_e1.active_uncertainties),
            "archived_raw_thought_records": len(archived_raw_thoughts),
            "harvested_archive_counts": {
                "executive_precedents": len(consolidated_archives.executive.precedents),
                "executive_action_clauses": len(consolidated_archives.executive.action_clauses),
                "legislative_diary_snapshots": len(consolidated_archives.legislative.diary_snapshots),
                "legislative_feeling_clauses": len(consolidated_archives.legislative.feeling_clauses),
                "judicial_structural_clauses": len(consolidated_archives.judicial.structural_clauses),
                "judicial_ratified_chords": len(consolidated_archives.judicial.ratified_chords),
            },
            "beats": epoch1_reports,
        },
        "epoch2a_consolidated_parable": {
            "settled_beats": [b["beat"] for b in epoch2a_reports if b["settled"]],
            "settled_count": sum(1 for b in epoch2a_reports if b["settled"]),
            "total_left_brain_calls": sum(b["left_brain_hails"] for b in epoch2a_reports),
            "mean_beat_wall_ms": round(
                sum(b["beat_wall_ms"] for b in epoch2a_reports) / len(epoch2a_reports), 2
            ),
            "audience_modes": {b["beat"]: b["audience_mode"] for b in epoch2a_reports},
            "beats": epoch2a_reports,
        },
        "epoch2b_unseen_bakehouse_generalization": {
            "settled_beats": [b["beat"] for b in unseen_reports if b["settled"]],
            "settled_count": sum(1 for b in unseen_reports if b["settled"]),
            "total_beats": len(unseen_reports),
            "total_left_brain_calls": sum(b["left_brain_hails"] for b in unseen_reports),
            "beat_3_social_drift_caught": bool(
                unseen_reports[2]["social_drift"] and unseen_reports[2]["social_drift"]["drift_detected"]
            ),
            "mean_beat_wall_ms": round(
                sum(b["beat_wall_ms"] for b in unseen_reports) / len(unseen_reports), 2
            ),
            "audience_modes": {b["beat"]: b["audience_mode"] for b in unseen_reports},
            "beats": unseen_reports,
        },
    }

    out_path = repo_root / "outputs" / "dual_hemisphere_soil_eval_results.json"
    atomic_json(out_path, summary)
    print(f"\nSaved full Dual-Hemisphere evaluation telemetry to {out_path}")
    print(
        f"SUMMARY:\n"
        f"  Epoch 1 (Cold Learning): settled={summary['epoch1_cold_learning']['settled_beats']}, "
        f"Left-Brain calls={summary['epoch1_cold_learning']['total_left_brain_calls']}, "
        f"Beat 2 false inversion fixed={summary['epoch1_cold_learning']['beat_2_false_inversion_fixed']}, "
        f"Beat 6 social drift caught={summary['epoch1_cold_learning']['beat_6_social_drift_caught']}, "
        f"Beat 8 open-thread closure={summary['epoch1_cold_learning']['beat_8_open_thread_closure_triggered']}\n"
        f"  Epoch 2A (Consolidated 8-Beat Parable): settled={summary['epoch2a_consolidated_parable']['settled_count']}/8, "
        f"Left-Brain calls={summary['epoch2a_consolidated_parable']['total_left_brain_calls']}, "
        f"mean latency={summary['epoch2a_consolidated_parable']['mean_beat_wall_ms']} ms\n"
        f"  Epoch 2B (Unseen 4-Beat Harbor Bakehouse): settled={summary['epoch2b_unseen_bakehouse_generalization']['settled_count']}/4, "
        f"Left-Brain calls={summary['epoch2b_unseen_bakehouse_generalization']['total_left_brain_calls']}, "
        f"Beat 3 social drift caught={summary['epoch2b_unseen_bakehouse_generalization']['beat_3_social_drift_caught']}, "
        f"mean latency={summary['epoch2b_unseen_bakehouse_generalization']['mean_beat_wall_ms']} ms"
    )


if __name__ == "__main__":
    main()
