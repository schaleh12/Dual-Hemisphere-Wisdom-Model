"""Stage 5 Runner: Nighttime REM Sleep Growth (`sleep_cycle_index = 1 -> 2`).

Executes the full 4-Phase Daytime -> REM Sleep -> Next-Day Wake loop on the RTX 4080
with strict sequential VRAM discipline and all 3 Auditor guardrails:
1. Phase 1 (`Day 1 Pre-REM Wake`, `sleep_cycle_index = 1`): Runs the 8-Beat Parable and
   4-Beat Unseen Bakehouse through `run_wake_stream` (`run_three_legged_turn`), logging
   `settled`, `gear_used`, and `left_brain_calls` per scene, then unloads `Qwen/Qwen3-4B`.
2. Phase 2 (`REM Phase A — Right-Brain Consolidation`, `Qwen3.5-9B`): Refines colliding
   Tier-1 rules and forges domain-general sleep rules and 3-branch clauses.
3. Phase 3 (`REM Phase B — Contrastive Dream Synthesis`, `Qwen3.5-9B` + 3 MiniLM Bridges):
   Synthesizes domain-general `GapScenarioSeed` dream prompts from the newly forged sleep
   rules, generates sensor-verified contrastive `<|subliminal_gap|>` pairs, saves
   `outputs/gap_trajectories_rem_dreams.jsonl`, and unloads `Qwen3.5-9B`.
4. Phase 4 (`REM Phase C — Gentle Incremental LoRA Update`, `sleep_cycle_index: 1 -> 2`):
   Reloads `Qwen/Qwen3-4B` (`sleep_cycle_index = 1`), trains gently (`epochs=2`,
   `lr_projector=1.8e-4`, `lr_lora=6e-5`) on the 17 replay pairs + new REM dream pairs,
   saves `models/left_brain_lora_cycle2/` (`sleep_cycle_index = 2`), evaluates Cycle 1 vs.
   Cycle 2 side-by-side on the 6 unseen holdout pairs, and runs `Day 2 Post-REM Wake`.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import torch

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from experiments.build_and_train_bridge_classifiers import (
    BANNED_PARABLE_SUBSTRINGS,
    EXTRA_CALIBRATION_CULTURAL_LEAVES,
)
from experiments.run_dual_hemisphere_soil_eval import (
    BANNED_BAKEHOUSE_SUBSTRINGS,
    UNSEEN_BAKEHOUSE_BEATS,
)
from experiments.run_feasibility_scaling_study import TEACHER_GGUF, DirectGGUFTeacher
from experiments.run_parable_gear1_score import PARABLE_BEATS
from lmft_lab.birth import create_birth_state
from lmft_lab.gap_trajectory_generator import release_gpu_vram
from lmft_lab.gap_trajectory_schema import (
    load_contrastive_gap_pairs_jsonl,
    save_contrastive_gap_pairs_jsonl,
)
from lmft_lab.local_models import load_config
from lmft_lab.neural_bridge_classifiers import (
    DEFAULT_COMPLEX_FEELING_LEAVES,
    NeuralThreeBridgeSuite,
)
from lmft_lab.reflection_classifier import load_learned_leaves, load_ruleset
from lmft_lab.rem_sleep_cycle import (
    REMSleepMaturationReport,
    generate_rem_dream_contrastive_pairs,
    run_incremental_lora_cycle_2_pass,
    run_wake_stream,
    synthesize_rem_dream_seeds_from_forged_rules,
)
from lmft_lab.runtime_store import atomic_json
from lmft_lab.sleep_consolidation import run_sleep_consolidation
from lmft_lab.three_legged_runtime import SubliminalGapLeftBrain
from lmft_lab.train_left_brain_lora import evaluate_holdout_pairs


def _cuda_allocated_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.memory_allocated(0) / (1024 ** 2), 2)


def main() -> None:
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    birth = create_birth_state("ind-stage5-rem-cycle", birth_event_id="birth-rem-1", random_seed=42)
    pre_sleep_ruleset_path = repo_root / "outputs" / "continuous_tiered_ruleset.json"
    initial_rules = load_ruleset(pre_sleep_ruleset_path)
    base_learned = load_learned_leaves(pre_sleep_ruleset_path)
    learned_leaves = base_learned + EXTRA_CALIBRATION_CULTURAL_LEAVES + DEFAULT_COMPLEX_FEELING_LEAVES

    replay_train_pairs = load_contrastive_gap_pairs_jsonl(repo_root / "outputs" / "gap_trajectories_train.jsonl")
    holdout_pairs = load_contrastive_gap_pairs_jsonl(repo_root / "outputs" / "gap_trajectories_holdout.jsonl")

    parable_texts = tuple(str(b["text"]) for b in PARABLE_BEATS)
    bakehouse_texts = tuple(str(b["text"]) for b in UNSEEN_BAKEHOUSE_BEATS)
    all_daytime_texts = parable_texts + bakehouse_texts
    banned_substrings = tuple(BANNED_PARABLE_SUBSTRINGS) + tuple(BANNED_BAKEHOUSE_SUBSTRINGS)

    bridges_dir = repo_root / "models" / "right_brain_bridges"
    print(f"Loading 3 Right-Brain MiniLM Bridge Classifiers from {bridges_dir} ...")
    suite = NeuralThreeBridgeSuite.load_all(bridges_dir)
    sensors = suite.as_three_bridge_sensors()

    vram_audit: dict[str, float] = {
        "initial_minilm_bridges_mb": _cuda_allocated_mb(),
    }

    # =========================================================================
    # PHASE 1: Daytime Wake Stream (Cycle 1, Pre-REM)
    # =========================================================================
    ckpt_cycle1_dir = repo_root / "models" / "left_brain_lora"
    print(f"\n[Phase 1] Loading Cycle 1 SubliminalGapLeftBrain from {ckpt_cycle1_dir} ...")
    left_brain_c1 = SubliminalGapLeftBrain.from_checkpoint(
        ckpt_cycle1_dir,
        device=device,
        max_clause_1_tokens=14,
        max_post_gap_tokens=32,
    )
    vram_audit["phase1_qwen4b_loaded_mb"] = _cuda_allocated_mb()

    print("[Phase 1] Evaluating Cycle 1 holdout baseline on the 6 unseen holdout pairs ...")
    cycle1_holdout = evaluate_holdout_pairs(
        left_brain_c1.model,
        left_brain_c1.tokenizer,
        left_brain_c1.projector,
        holdout_pairs,
        gap_token_id=left_brain_c1.gap_token_id,
        device=device,
        sensors=sensors,
        rules=initial_rules,
        learned_leaves=learned_leaves,
        max_new_tokens=32,
    )

    print("[Phase 1] Running Day 1 (Pre-REM) Wake Stream on 8-Beat Parable ...")
    day1_parable_summary, day1_archives, day1_ledger = run_wake_stream(
        birth,
        parable_texts,
        sensors,
        initial_rules,
        cycle_label="day1_pre_rem_parable",
        sleep_cycle_index=1,
        subliminal_left_brain=left_brain_c1,
        learned_leaves=learned_leaves,
        max_passes=2,
    )
    print(
        f"  Day 1 Parable: settled={day1_parable_summary.settled_count}/{day1_parable_summary.total_scenes} "
        f"(Gear1={day1_parable_summary.gear_1_count}, Gear2={day1_parable_summary.gear_2_count}, "
        f"Gear3_Unresolved={day1_parable_summary.gear_3_unresolved_count}, "
        f"left_brain_calls={day1_parable_summary.total_left_brain_calls})"
    )

    print("[Phase 1] Running Day 1 (Pre-REM) Unseen Harbor Bakehouse check ...")
    day1_bakehouse_summary, _, _ = run_wake_stream(
        birth,
        bakehouse_texts,
        sensors,
        initial_rules,
        cycle_label="day1_pre_rem_unseen_bakehouse",
        sleep_cycle_index=1,
        subliminal_left_brain=None,
        learned_leaves=learned_leaves,
        initial_archives=day1_archives,
        initial_ledger=day1_ledger,
        max_passes=2,
    )
    print(
        f"  Day 1 Unseen Bakehouse (0 LLM calls): settled={day1_bakehouse_summary.settled_count}/{day1_bakehouse_summary.total_scenes} "
        f"(Gear1={day1_bakehouse_summary.gear_1_count}, Gear3_Unresolved={day1_bakehouse_summary.gear_3_unresolved_count})"
    )

    # Explicitly unload Qwen3-4B before loading Qwen3.5-9B!
    del left_brain_c1
    release_gpu_vram()
    vram_audit["after_phase1_unload_qwen4b_mb"] = _cuda_allocated_mb()
    print(f"[Phase 1] Unloaded Qwen3-4B. Allocated PyTorch VRAM: {vram_audit['after_phase1_unload_qwen4b_mb']} MB")

    # =========================================================================
    # PHASE 2 & PHASE 3: Offline 9B REM Consolidation + Contrastive Dream Synthesis
    # =========================================================================
    config = load_config(repo_root / "config" / "local_models.json")
    generation_cfg = dict(config["generation"])
    generation_cfg["max_tokens"] = 320

    gap_json_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["clause_1_prefix", "continuation_a", "continuation_b"],
        "properties": {
            "clause_1_prefix": {"type": "string", "minLength": 12, "maxLength": 140},
            "continuation_a": {"type": "string", "minLength": 20, "maxLength": 240},
            "continuation_b": {"type": "string", "minLength": 20, "maxLength": 240},
        },
    }

    print(f"\n[Phase 2 & 3] Loading Offline 9B Teacher from {TEACHER_GGUF} ...")
    with DirectGGUFTeacher(TEACHER_GGUF, generation_cfg) as teacher:
        print(
            f"[Phase 2] Running REM Phase A (Right-Brain Rule & Archive Consolidation) on "
            f"{len(day1_parable_summary.unresolved_sleep_scenes)} unresolved daytime scenes ..."
        )
        sleep_report = run_sleep_consolidation(
            unresolved_scenes=day1_parable_summary.unresolved_sleep_scenes,
            rules=initial_rules,
            learned_leaves=learned_leaves,
            archives=day1_archives,
            ledger=day1_ledger,
            teacher=teacher,
            generation=generation_cfg,
            banned_substrings=banned_substrings,
        )
        print(
            f"  REM Phase A: refined_rules={list(sleep_report.refined_rule_ids)}, "
            f"forged_rules={[r.concept_name for r in sleep_report.forged_rules]}, "
            f"harvested_clauses={len(sleep_report.harvested_clause_ids)}"
        )

        print("[Phase 3] Synthesizing domain-general REM dream seeds from newly forged rules ...")
        dream_seeds = synthesize_rem_dream_seeds_from_forged_rules(
            sleep_report.forged_rules,
            teacher=teacher,
            generation=generation_cfg,
            banned_substrings=banned_substrings,
            daytime_scenarios=all_daytime_texts,
        )
        print(f"  Synthesized {len(dream_seeds)} disjoint REM dream seeds.")

        call_counter = 0

        def _teacher_dream_gen(prompt: str) -> str:
            nonlocal call_counter
            call_counter += 1
            req = {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a precise Dual-Hemisphere REM dream author. "
                            "Return only valid JSON matching the requested schema."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2 if call_counter > 1 else 0.0,
                "seed": int(generation_cfg.get("seed", 42)) + 100 + call_counter,
                "max_tokens": 320,
                "stream": False,
                "cache_prompt": False,
                "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {"type": "json_object", "schema": gap_json_schema},
            }
            raw_reply, _trace = teacher.generate(
                "teacher",
                req,
                "rem-dream-pair-teacher-v1",
                lambda raw: raw,
            )
            return raw_reply

        dream_pairs, dream_logs = generate_rem_dream_contrastive_pairs(
            dream_seeds,
            teacher_generate_fn=_teacher_dream_gen,
            sensors=sensors,
            rules=sleep_report.updated_rules,
            learned_leaves=learned_leaves,
            max_attempts=3,
        )
        rem_dreams_path = repo_root / "outputs" / "gap_trajectories_rem_dreams.jsonl"
        save_contrastive_gap_pairs_jsonl(rem_dreams_path, dream_pairs)
        mean_dream_l2 = (
            round(sum(p.chord_distance for p in dream_pairs) / len(dream_pairs), 4)
            if dream_pairs
            else 0.0
        )
        print(
            f"  REM Phase B: accepted {len(dream_pairs)}/{len(dream_seeds)} dream pairs "
            f"(mean_L2={mean_dream_l2:.4f}), saved to {rem_dreams_path}."
        )

    release_gpu_vram()
    vram_audit["after_phase3_unload_qwen9b_mb"] = _cuda_allocated_mb()
    print(f"[Phase 3] Unloaded Qwen3.5-9B. Allocated PyTorch VRAM: {vram_audit['after_phase3_unload_qwen9b_mb']} MB")

    # =========================================================================
    # PHASE 4: Gentle Incremental LoRA Update (sleep_cycle_index: 1 -> 2) & Day 2 Wake
    # =========================================================================
    print(f"\n[Phase 4] Reloading Cycle 1 SubliminalGapLeftBrain for gentle REM Cycle 2 update ...")
    left_brain_c2 = SubliminalGapLeftBrain.from_checkpoint(
        ckpt_cycle1_dir,
        device=device,
        max_clause_1_tokens=14,
        max_post_gap_tokens=32,
    )
    if hasattr(left_brain_c2.model, "gradient_checkpointing_enable"):
        left_brain_c2.model.gradient_checkpointing_enable()
    left_brain_c2.model.config.use_cache = False

    ckpt_cycle2_dir = repo_root / "models" / "left_brain_lora_cycle2"
    rem_epochs = 1
    rem_lr_proj = 1.0e-4
    rem_lr_lora = 2e-5
    rem_warmup_steps = 3
    rem_replay_ratio = 2

    print(
        f"[Phase 4] Running gentle incremental REM LoRA update ({rem_epochs} epoch, "
        f"lr_proj={rem_lr_proj}, lr_lora={rem_lr_lora}, warmup={rem_warmup_steps}, replay_ratio={rem_replay_ratio}:1) "
        f"on {len(replay_train_pairs)} replay + {len(dream_pairs)} dream pairs ..."
    )
    lora_history, callosum_meta_c2, combined_pairs = run_incremental_lora_cycle_2_pass(
        left_brain_c2.model,
        left_brain_c2.tokenizer,
        left_brain_c2.projector,
        replay_train_pairs=replay_train_pairs,
        dream_pairs=dream_pairs,
        holdout_pairs=holdout_pairs,
        gap_token_id=left_brain_c2.gap_token_id,
        device=device,
        lora_config=left_brain_c2.lora_config,
        sleep_cycle_before=1,
        checkpoint_out_dir=ckpt_cycle2_dir,
        daytime_scenarios=all_daytime_texts,
        banned_substrings=banned_substrings,
        epochs=rem_epochs,
        lr_projector=rem_lr_proj,
        lr_lora=rem_lr_lora,
        warmup_steps=rem_warmup_steps,
        replay_ratio=rem_replay_ratio,
    )
    left_brain_c2.callosum_metadata = callosum_meta_c2
    vram_audit["phase4_peak_vram_mb"] = (
        round(torch.cuda.max_memory_allocated(0) / (1024 ** 2), 2)
        if torch.cuda.is_available()
        else 0.0
    )

    print("[Phase 4] Evaluating matured Cycle 2 checkpoint on the 6 unseen holdout pairs ...")
    cycle2_holdout = evaluate_holdout_pairs(
        left_brain_c2.model,
        left_brain_c2.tokenizer,
        left_brain_c2.projector,
        holdout_pairs,
        gap_token_id=left_brain_c2.gap_token_id,
        device=device,
        sensors=sensors,
        rules=sleep_report.updated_rules,
        learned_leaves=learned_leaves,
        max_new_tokens=32,
    )

    print("[Phase 4] Running Day 2 (Post-REM) Wake Stream on 8-Beat Parable ...")
    day2_parable_summary, _, _ = run_wake_stream(
        birth,
        parable_texts,
        sensors,
        sleep_report.updated_rules,
        cycle_label="day2_post_rem_parable",
        sleep_cycle_index=2,
        subliminal_left_brain=left_brain_c2,
        learned_leaves=learned_leaves,
        initial_archives=sleep_report.updated_archives,
        initial_ledger=sleep_report.updated_ledger,
        max_passes=2,
    )
    print(
        f"  Day 2 Parable: settled={day2_parable_summary.settled_count}/{day2_parable_summary.total_scenes} "
        f"(Gear1={day2_parable_summary.gear_1_count}, Gear2={day2_parable_summary.gear_2_count}, "
        f"Gear3_Unresolved={day2_parable_summary.gear_3_unresolved_count}, "
        f"left_brain_calls={day2_parable_summary.total_left_brain_calls})"
    )

    print("[Phase 4] Running Day 2 (Post-REM) Unseen Harbor Bakehouse check ...")
    day2_bakehouse_summary, _, _ = run_wake_stream(
        birth,
        bakehouse_texts,
        sensors,
        sleep_report.updated_rules,
        cycle_label="day2_post_rem_unseen_bakehouse",
        sleep_cycle_index=2,
        subliminal_left_brain=left_brain_c2,
        learned_leaves=learned_leaves,
        initial_archives=sleep_report.updated_archives,
        initial_ledger=sleep_report.updated_ledger,
        max_passes=2,
    )
    print(
        f"  Day 2 Unseen Bakehouse: settled={day2_bakehouse_summary.settled_count}/{day2_bakehouse_summary.total_scenes} "
        f"(Gear1={day2_bakehouse_summary.gear_1_count}, Gear2={day2_bakehouse_summary.gear_2_count}, "
        f"Gear3_Unresolved={day2_bakehouse_summary.gear_3_unresolved_count}, "
        f"left_brain_calls={day2_bakehouse_summary.total_left_brain_calls})"
    )

    rem_report = REMSleepMaturationReport(
        sleep_cycle_before=1,
        sleep_cycle_after=callosum_meta_c2.sleep_cycle_index,
        day1_pre_rem_wake=day1_parable_summary,
        rem_phase_a_consolidation={
            "refined_rule_ids": list(sleep_report.refined_rule_ids),
            "forged_rule_names": [r.concept_name for r in sleep_report.forged_rules],
            "forged_rules": [r.to_dict() for r in sleep_report.forged_rules],
            "harvested_clause_ids": list(sleep_report.harvested_clause_ids),
            "resolved_scene_indices": list(sleep_report.resolved_scene_indices),
            "day1_unseen_bakehouse_pre_rem": day1_bakehouse_summary.to_dict(),
        },
        rem_phase_b_dream_synthesis={
            "dream_seeds_count": len(dream_seeds),
            "dream_pairs_accepted_count": len(dream_pairs),
            "mean_dream_l2": mean_dream_l2,
            "attempt_logs": [log.to_dict() for log in dream_logs],
        },
        rem_phase_c_lora_maturation={
            "replay_train_pairs_count": len(replay_train_pairs),
            "new_dream_pairs_count": len(dream_pairs),
            "combined_train_pairs_count": len(combined_pairs),
            "epochs": rem_epochs,
            "lr_projector": rem_lr_proj,
            "lr_lora": rem_lr_lora,
            "epoch_history": lora_history,
            "corpus_callosum_metadata": callosum_meta_c2.to_dict(),
            "day2_unseen_bakehouse_post_rem": day2_bakehouse_summary.to_dict(),
            "total_wall_time_s": round(time.perf_counter() - t0, 2),
        },
        cycle1_holdout_evaluation=cycle1_holdout,
        cycle2_holdout_evaluation=cycle2_holdout,
        day2_post_rem_wake=day2_parable_summary,
        vram_phase_audit_mb=vram_audit,
    )

    report_out_path = repo_root / "outputs" / "rem_sleep_cycle_report.json"
    atomic_json(report_out_path, rem_report.to_dict())

    del left_brain_c2
    del suite
    release_gpu_vram()

    print("\n" + "=" * 78)
    print("STAGE 5 COMPLETE: NIGHTTIME REM SLEEP GROWTH (CYCLE 1 -> CYCLE 2)")
    print("=" * 78)
    print(f"Day 1 Parable Settled (Pre-REM)     : {day1_parable_summary.settled_count}/8 (Gear1={day1_parable_summary.gear_1_count}, Gear2={day1_parable_summary.gear_2_count}, Gear3={day1_parable_summary.gear_3_unresolved_count})")
    print(f"Day 2 Parable Settled (Post-REM)    : {day2_parable_summary.settled_count}/8 (Gear1={day2_parable_summary.gear_1_count}, Gear2={day2_parable_summary.gear_2_count}, Gear3={day2_parable_summary.gear_3_unresolved_count})")
    print(f"Day 1 Unseen Bakehouse (Pre-REM)    : {day1_bakehouse_summary.settled_count}/4 (Gear1={day1_bakehouse_summary.gear_1_count}, Gear3={day1_bakehouse_summary.gear_3_unresolved_count})")
    print(f"Day 2 Unseen Bakehouse (Post-REM)   : {day2_bakehouse_summary.settled_count}/4 (Gear1={day2_bakehouse_summary.gear_1_count}, Gear2={day2_bakehouse_summary.gear_2_count}, Gear3={day2_bakehouse_summary.gear_3_unresolved_count})")
    print(f"Holdout Positive Shift (C1 -> C2)   : {cycle1_holdout.positive_chord_swap_shift_count}/6 -> {cycle2_holdout.positive_chord_swap_shift_count}/6")
    print(f"Holdout Zero-Cal Flip (C1 -> C2)    : {cycle1_holdout.zero_calibrated_both_flipped_count}/6 -> {cycle2_holdout.zero_calibrated_both_flipped_count}/6")
    print(f"Holdout No-Gap Flip (C1 -> C2)      : {cycle1_holdout.no_gap_calibrated_both_flipped_count}/6 -> {cycle2_holdout.no_gap_calibrated_both_flipped_count}/6")
    print(f"Holdout Mean Swap Margin (C1 -> C2) : {cycle1_holdout.mean_net_chord_swap_margin:+.4f} -> {cycle2_holdout.mean_net_chord_swap_margin:+.4f}")
    print(f"Saved Cycle 2 checkpoint            : {ckpt_cycle2_dir}")
    print(f"Saved Stage 5 report                : {report_out_path}")


if __name__ == "__main__":
    main()
