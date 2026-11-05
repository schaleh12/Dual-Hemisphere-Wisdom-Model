"""Stage 5: Nighttime REM Sleep Growth (`sleep_cycle_index = 1 -> 2`) (`rem_sleep_cycle.py`).

Plain-English Purpose:
Executes the full 4-Phase Daytime -> Nighttime REM Sleep -> Next-Day Wake maturation loop
with strict sequential VRAM discipline and all 3 Auditor guardrails:

1. **Phase 1 — Daytime Wake Stream (`sleep_cycle_index = 1`)**:
   - Runs daytime scenes through `run_three_legged_turn` (`three_legged_runtime.py`),
     logging `settled`, `gear_used` (`GEAR_1_FAST_RIGHT_BRAIN`,
     `GEAR_2_SUBLIMINAL_GAP_SPLICE`, or `GEAR_3_UNRESOLVED_QUEUED_FOR_SLEEP`), and
     `left_brain_calls` for every scene (**Guardrail #1**).
   - Collects any genuinely unsettled turns as `UnresolvedSleepScene` items.
   - Unloads `Qwen/Qwen3-4B` from VRAM before Phase 2 starts.
2. **Phase 2 — Nighttime REM Phase A: Right-Brain Rule & Archive Consolidation (`Qwen3.5-9B`)**:
   - Runs `run_sleep_consolidation` (`sleep_consolidation.py`) with the 9B Teacher to refine
     colliding Tier-1 rules and forge domain-general `FuzzyConceptRule`s and 3-branch
     clauses (`assert_domain_general_text` enforced).
3. **Phase 3 — Nighttime REM Phase B: Contrastive `<|subliminal_gap|>` Dream Synthesis**:
   - Synthesizes domain-general `GapScenarioSeed` dream prompts from the newly forged sleep
     rules, generates contrastive `<|subliminal_gap|>` pairs with the 9B Teacher + 3 MiniLM
     bridges, and verifies disjointness against both the 6 unseen holdout pairs and the
     daytime wake scenes (**Guardrail #3**).
   - Unloads `Qwen3.5-9B` from VRAM before Phase 4 starts.
4. **Phase 4 — Nighttime REM Phase C: Gentle Incremental LoRA Update (`sleep_cycle_index: 1 -> 2`)**:
   - Mixes the new REM dream pairs with the existing Stage 2 training pairs (`replay_pairs`)
     and runs a gentle incremental update (`epochs=2`, `lr_projector=1.8e-4`, `lr_lora=6e-5`,
     ~30% of Stage 3 LR) to prevent catastrophic forgetting (**Guardrail #2**).
   - Saves the matured Cycle 2 checkpoint (`sleep_cycle_index = 2`), evaluates Cycle 1 vs.
     Cycle 2 side-by-side on the 6 unseen holdout pairs, and re-runs the Daytime Wake
     stream (`Day 2 Post-REM`) to measure how many previously unresolved scenes now settle
     and which Gear (`Gear 1` vs. `Gear 2`) settles them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import random
import re
import time

import torch
from torch import nn

from .birth import BirthState, create_birth_state
from .branch_archives import ThreeBranchArchives
from .chord_memory import ChordMemoryBank
from .contemplation_ledger import ContemplationLedger
from .gap_trajectory_generator import (
    GapScenarioSeed,
    PairAttemptLog,
    generate_single_contrastive_pair,
    release_gpu_vram,
)
from .gap_trajectory_schema import (
    MIN_CHORD_L2_DISTANCE,
    ContrastiveGapPair,
    validate_train_holdout_disjointness,
)
from .local_models import strict_json
from .model_assets import ModelError
from .neural_bridge_classifiers import DEFAULT_COMPLEX_FEELING_LEAVES
from .reflection_classifier import (
    ContinuousMemoryState,
    FuzzyConceptRule,
    LearnedLeaf,
)
from .right_brain_equilibrium import (
    SettledDiary,
    ThreeBridgeSensors,
)
from .sleep_consolidation import (
    SleepConsolidationReport,
    UnresolvedSleepScene,
    assert_domain_general_text,
    run_sleep_consolidation,
)
from .subliminal_gap_splice import (
    CorpusCallosumMetadata,
    SubliminalGapProjector,
)
from .three_legged_runtime import (
    SubliminalGapLeftBrain,
    run_three_legged_turn,
)
from .train_left_brain_lora import (
    HoldoutEvaluationSummary,
    LoRAConfig,
    evaluate_holdout_pairs,
    save_lora_and_callosum_checkpoint,
    train_left_brain_lora_and_callosum,
)


@dataclass(frozen=True)
class WakeSceneTurnLog:
    """Per-scene telemetry for Day 1 (`Pre-REM`) vs. Day 2 (`Post-REM`) wake evaluation."""

    scene_index: int
    scenario_text: str
    settled: bool
    gear_used: str
    outcome: str
    active_concept: str | None
    left_brain_calls: int
    wall_time_ms: float
    final_conscious_note: str | None

    def to_dict(self) -> dict:
        return {
            "scene_index": self.scene_index,
            "scenario_text": self.scenario_text,
            "settled": self.settled,
            "gear_used": self.gear_used,
            "outcome": self.outcome,
            "active_concept": self.active_concept,
            "left_brain_calls": self.left_brain_calls,
            "wall_time_ms": self.wall_time_ms,
            "final_conscious_note": self.final_conscious_note,
        }


@dataclass(frozen=True)
class WakeStreamSummary:
    """Summary of one full Daytime Wake stream (`Day 1 Pre-REM` or `Day 2 Post-REM`)."""

    cycle_label: str
    sleep_cycle_index: int
    total_scenes: int
    settled_count: int
    settled_scene_indices: tuple[int, ...]
    gear_1_count: int
    gear_2_count: int
    gear_3_unresolved_count: int
    total_left_brain_calls: int
    mean_wall_time_ms: float
    scene_logs: tuple[WakeSceneTurnLog, ...]
    unresolved_sleep_scenes: tuple[UnresolvedSleepScene, ...]

    def to_dict(self) -> dict:
        return {
            "cycle_label": self.cycle_label,
            "sleep_cycle_index": self.sleep_cycle_index,
            "total_scenes": self.total_scenes,
            "settled_count": self.settled_count,
            "settled_scene_indices": list(self.settled_scene_indices),
            "gear_1_count": self.gear_1_count,
            "gear_2_count": self.gear_2_count,
            "gear_3_unresolved_count": self.gear_3_unresolved_count,
            "total_left_brain_calls": self.total_left_brain_calls,
            "mean_wall_time_ms": self.mean_wall_time_ms,
            "scene_logs": [log.to_dict() for log in self.scene_logs],
            "unresolved_scene_indices": [u.scene_index for u in self.unresolved_sleep_scenes],
        }


@dataclass(frozen=True)
class REMSleepMaturationReport:
    """Complete un-rigged report for Stage 5 (`sleep_cycle_index = 1 -> 2`)."""

    sleep_cycle_before: int
    sleep_cycle_after: int
    day1_pre_rem_wake: WakeStreamSummary
    rem_phase_a_consolidation: dict
    rem_phase_b_dream_synthesis: dict
    rem_phase_c_lora_maturation: dict
    cycle1_holdout_evaluation: HoldoutEvaluationSummary
    cycle2_holdout_evaluation: HoldoutEvaluationSummary
    day2_post_rem_wake: WakeStreamSummary
    vram_phase_audit_mb: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "sleep_cycle_before": self.sleep_cycle_before,
            "sleep_cycle_after": self.sleep_cycle_after,
            "day1_pre_rem_wake": self.day1_pre_rem_wake.to_dict(),
            "rem_phase_a_consolidation": dict(self.rem_phase_a_consolidation),
            "rem_phase_b_dream_synthesis": dict(self.rem_phase_b_dream_synthesis),
            "rem_phase_c_lora_maturation": dict(self.rem_phase_c_lora_maturation),
            "cycle1_holdout_evaluation": self.cycle1_holdout_evaluation.to_dict(),
            "cycle2_holdout_evaluation": self.cycle2_holdout_evaluation.to_dict(),
            "day2_post_rem_wake": self.day2_post_rem_wake.to_dict(),
            "vram_phase_audit_mb": dict(self.vram_phase_audit_mb),
        }


def run_wake_stream(
    birth_state: BirthState,
    scenarios: Sequence[str],
    sensors: ThreeBridgeSensors,
    rules: Sequence[FuzzyConceptRule],
    *,
    cycle_label: str,
    sleep_cycle_index: int,
    subliminal_left_brain: SubliminalGapLeftBrain | None = None,
    learned_leaves: Sequence[LearnedLeaf] = DEFAULT_COMPLEX_FEELING_LEAVES,
    initial_archives: ThreeBranchArchives | None = None,
    initial_ledger: ContemplationLedger | None = None,
    max_passes: int = 2,
    branch_gating: bool = True,
) -> tuple[WakeStreamSummary, ThreeBranchArchives, ContemplationLedger]:
    """Run a multi-scene daytime wake stream through `run_three_legged_turn` and log Gear usage."""
    cur_memory: ContinuousMemoryState | None = None
    cur_diary = SettledDiary()
    cur_chord_memory = ChordMemoryBank()
    cur_ledger = initial_ledger if initial_ledger is not None else ContemplationLedger()
    if initial_archives is not None:
        cur_archives = ThreeBranchArchives(
            executive=initial_archives.executive.__class__(
                precedents=(),
                action_clauses=initial_archives.executive.action_clauses,
                max_entries=initial_archives.executive.max_entries,
            ),
            legislative=initial_archives.legislative,
            judicial=initial_archives.judicial,
        )
    else:
        cur_archives = ThreeBranchArchives()
    rules_tuple = tuple(rules)
    leaves_tuple = tuple(learned_leaves)

    scene_logs: list[WakeSceneTurnLog] = []
    unresolved_scenes: list[UnresolvedSleepScene] = []

    for idx, text in enumerate(scenarios, start=1):
        t0 = time.perf_counter()
        turn = run_three_legged_turn(
            birth_state,
            text,
            sensors,
            rules_tuple,
            scene_index=idx,
            subliminal_left_brain=subliminal_left_brain,
            learned_leaves=leaves_tuple,
            prior_memory=cur_memory,
            diary=cur_diary,
            chord_memory=cur_chord_memory,
            contemplation_ledger=cur_ledger,
            branch_archives=cur_archives,
            max_passes=max_passes,
            branch_gating=branch_gating,
        )
        wall_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        cur_memory = turn.next_memory
        cur_diary = turn.next_diary
        cur_chord_memory = turn.next_chord_memory
        cur_ledger = turn.next_ledger
        cur_archives = turn.next_archives

        scene_logs.append(
            WakeSceneTurnLog(
                scene_index=idx,
                scenario_text=text,
                settled=turn.settled,
                gear_used=turn.gear_used,
                outcome=turn.outcome,
                active_concept=turn.final_vibe.active_concept,
                left_brain_calls=turn.left_brain_calls,
                wall_time_ms=wall_ms,
                final_conscious_note=turn.final_conscious_note,
            )
        )
        if turn.unresolved_sleep_scene is not None:
            unresolved_scenes.append(turn.unresolved_sleep_scene)

    total = len(scene_logs)
    settled_indices = tuple(log.scene_index for log in scene_logs if log.settled)
    g1_count = sum(1 for log in scene_logs if log.gear_used == "GEAR_1_FAST_RIGHT_BRAIN")
    g2_count = sum(1 for log in scene_logs if log.gear_used == "GEAR_2_SUBLIMINAL_GAP_SPLICE")
    g3_count = sum(1 for log in scene_logs if log.gear_used == "GEAR_3_UNRESOLVED_QUEUED_FOR_SLEEP")
    total_calls = sum(log.left_brain_calls for log in scene_logs)
    mean_ms = round(sum(log.wall_time_ms for log in scene_logs) / total, 2) if total else 0.0

    summary = WakeStreamSummary(
        cycle_label=cycle_label,
        sleep_cycle_index=sleep_cycle_index,
        total_scenes=total,
        settled_count=len(settled_indices),
        settled_scene_indices=settled_indices,
        gear_1_count=g1_count,
        gear_2_count=g2_count,
        gear_3_unresolved_count=g3_count,
        total_left_brain_calls=total_calls,
        mean_wall_time_ms=mean_ms,
        scene_logs=tuple(scene_logs),
        unresolved_sleep_scenes=tuple(unresolved_scenes),
    )
    return summary, cur_archives, cur_ledger


def synthesize_rem_dream_seeds_from_forged_rules(
    forged_rules: Sequence[FuzzyConceptRule],
    *,
    teacher,
    teacher_role: str = "teacher",
    generation: Mapping[str, object] | None = None,
    banned_substrings: Sequence[str] = (),
    daytime_scenarios: Sequence[str] = (),
) -> tuple[GapScenarioSeed, ...]:
    """Ask the 9B Teacher to synthesize domain-general REM dream `GapScenarioSeed`s for newly forged rules."""
    gen_cfg = dict(generation) if generation is not None else {"seed": 42}
    daytime_norm = {" ".join(s.lower().split()) for s in daytime_scenarios}
    seeds: list[GapScenarioSeed] = []

    dream_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "user_prompt",
            "posture_label_a",
            "posture_guidance_a",
            "posture_label_b",
            "posture_guidance_b",
        ],
        "properties": {
            "user_prompt": {"type": "string", "minLength": 24, "maxLength": 220},
            "posture_label_a": {"type": "string", "minLength": 6, "maxLength": 48},
            "posture_guidance_a": {"type": "string", "minLength": 20, "maxLength": 220},
            "posture_label_b": {"type": "string", "minLength": 6, "maxLength": 48},
            "posture_guidance_b": {"type": "string", "minLength": 20, "maxLength": 220},
        },
    }

    for idx, rule in enumerate(forged_rules, start=1):
        leaves_summary = ", ".join(f"{k}={v:.2f}" for k, v in sorted(rule.target_leaves.items()))
        banned_note = (
            f"Never use any of these story-specific words: {', '.join(banned_substrings)}."
            if banned_substrings
            else "Use domain-general civic, workplace, or resource-sharing situations."
        )
        user_msg = (
            f"We just consolidated a new Right-Brain concept rule during REM sleep:\n"
            f"- concept_name: {rule.concept_name}\n"
            f"- description: {rule.description}\n"
            f"- target_leaves: {leaves_summary}\n\n"
            f"Create one fresh, domain-general contrastive scenario seed where:\n"
            f"1. `user_prompt` describes a concrete civic, cooperative, or organizational dilemma "
            f"(completely disjoint from daytime story settings). {banned_note}\n"
            f"2. `posture_label_a` and `posture_guidance_a` express the open, fair, restorative stance.\n"
            f"3. `posture_label_b` and `posture_guidance_b` express the opposite exclusionary, coercive, "
            f"or contested lockdown stance.\n"
            f"Return strict JSON."
        )
        req = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an offline REM Sleep dream synthesizer. "
                        "Return only valid JSON with zero story-specific words."
                    ),
                },
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.1,
            "seed": int(gen_cfg.get("seed", 42)) + idx,
            "max_tokens": 300,
            "stream": False,
            "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_object", "schema": dream_schema},
        }

        def _validate_dream_seed(raw: str) -> dict[str, str]:
            try:
                cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                parsed = strict_json(cleaned)
                u_prompt = assert_domain_general_text(
                    str(parsed["user_prompt"]),
                    "user_prompt",
                    banned_substrings,
                )
                if " ".join(u_prompt.lower().split()) in daytime_norm:
                    raise ValueError("dream user_prompt duplicates a daytime wake scenario")
                label_a = re.sub(r"[^A-Z0-9_]+", "_", str(parsed["posture_label_a"]).strip().upper()).strip("_")
                label_b = re.sub(r"[^A-Z0-9_]+", "_", str(parsed["posture_label_b"]).strip().upper()).strip("_")
                guide_a = assert_domain_general_text(
                    str(parsed["posture_guidance_a"]),
                    "posture_guidance_a",
                    banned_substrings,
                )
                guide_b = assert_domain_general_text(
                    str(parsed["posture_guidance_b"]),
                    "posture_guidance_b",
                    banned_substrings,
                )
                return {
                    "user_prompt": u_prompt,
                    "posture_label_a": label_a or "REM_RESTORATIVE_STEWARDSHIP",
                    "posture_guidance_a": guide_a,
                    "posture_label_b": label_b or "REM_CONTESTED_ENCLOSURE",
                    "posture_guidance_b": guide_b,
                }
            except Exception as exc:
                raise ModelError(f"REM dream seed validation failed: {exc}") from None

        validated, _trace = teacher.generate(
            teacher_role,
            req,
            f"rem-dream-seed-{rule.concept_name}-v1",
            _validate_dream_seed,
        )
        fairness_target = float(rule.target_leaves.get("cultural/justice/fairness", 0.50))
        power_target = float(rule.target_leaves.get("cultural/efficiency/power", 0.0))
        disc_target = float(rule.target_leaves.get("philosophical/persistence/discreteness", 0.0))
        seeds.append(
            GapScenarioSeed(
                pair_id=f"rem-dream-{idx:02d}-{rule.concept_name[:20]}",
                domain="social_dilemma",
                split="train",
                user_prompt=validated["user_prompt"],
                posture_label_a=validated["posture_label_a"],
                posture_guidance_a=validated["posture_guidance_a"],
                posture_label_b=validated["posture_label_b"],
                posture_guidance_b=validated["posture_guidance_b"],
                social_drift_a=False,
                social_drift_b=(fairness_target <= 0.38 or power_target >= 0.65 or disc_target >= 0.65),
            )
        )
    return tuple(seeds)


def validate_dream_pairs_disjoint_from_wake_and_holdout(
    dream_pairs: Sequence[ContrastiveGapPair],
    replay_train_pairs: Sequence[ContrastiveGapPair],
    holdout_pairs: Sequence[ContrastiveGapPair],
    *,
    daytime_scenarios: Sequence[str] = (),
    banned_substrings: Sequence[str] = (),
) -> tuple[ContrastiveGapPair, ...]:
    """Enforce Guardrail #3: REM dream pairs must be disjoint from both holdout and daytime wake texts."""
    daytime_norm = {" ".join(s.lower().split()) for s in daytime_scenarios}
    for dp in dream_pairs:
        assert_domain_general_text(dp.user_prompt, "dream.user_prompt", banned_substrings)
        assert_domain_general_text(dp.clause_1_prefix, "dream.clause_1_prefix", banned_substrings)
        assert_domain_general_text(dp.continuation_a, "dream.continuation_a", banned_substrings)
        assert_domain_general_text(dp.continuation_b, "dream.continuation_b", banned_substrings)
        if " ".join(dp.user_prompt.lower().split()) in daytime_norm:
            raise ValueError(f"REM dream pair {dp.pair_id!r} duplicates a daytime wake scenario")

    combined = tuple(replay_train_pairs) + tuple(dream_pairs)
    validate_train_holdout_disjointness(combined, holdout_pairs)
    return combined


def generate_rem_dream_contrastive_pairs(
    dream_seeds: Sequence[GapScenarioSeed],
    *,
    teacher_generate_fn: Callable[[str], str],
    sensors: ThreeBridgeSensors,
    rules: Sequence[FuzzyConceptRule],
    learned_leaves: Sequence[LearnedLeaf] = DEFAULT_COMPLEX_FEELING_LEAVES,
    min_l2_distance: float = MIN_CHORD_L2_DISTANCE,
    max_attempts: int = 3,
) -> tuple[tuple[ContrastiveGapPair, ...], tuple[PairAttemptLog, ...]]:
    """Generate and sensor-verify contrastive `<|subliminal_gap|>` pairs for the REM dream seeds."""
    birth = create_birth_state("rem-dream-gen", birth_event_id="evt-rem-dream-1", random_seed=42)
    accepted: list[ContrastiveGapPair] = []
    logs: list[PairAttemptLog] = []
    leaves_tuple = tuple(learned_leaves)

    for seed in dream_seeds:
        pair, log = generate_single_contrastive_pair(
            seed,
            teacher_generate_fn=teacher_generate_fn,
            birth_state=birth,
            sensors=sensors,
            rules=rules,
            learned_leaves=leaves_tuple,
            min_l2_distance=min_l2_distance,
            max_attempts=max_attempts,
        )
        logs.append(log)
        if pair is not None:
            accepted.append(pair)
    return tuple(accepted), tuple(logs)


def run_incremental_lora_cycle_2_pass(
    model: nn.Module,
    tokenizer,
    projector: SubliminalGapProjector,
    *,
    replay_train_pairs: Sequence[ContrastiveGapPair],
    dream_pairs: Sequence[ContrastiveGapPair],
    holdout_pairs: Sequence[ContrastiveGapPair],
    gap_token_id: int,
    device: str | torch.device,
    lora_config: LoRAConfig,
    sleep_cycle_before: int = 1,
    checkpoint_out_dir: str | Path | None = None,
    daytime_scenarios: Sequence[str] = (),
    banned_substrings: Sequence[str] = (),
    epochs: int = 1,
    lr_projector: float = 1.0e-4,
    lr_lora: float = 2e-5,
    warmup_steps: int = 3,
    replay_ratio: int | None = 2,
) -> tuple[list[dict[str, float]], CorpusCallosumMetadata, tuple[ContrastiveGapPair, ...]]:
    """Enforce Guardrail #2: Gentle incremental REM LoRA + Callosum update mixing replay + dream pairs."""
    combined_pairs = validate_dream_pairs_disjoint_from_wake_and_holdout(
        dream_pairs,
        replay_train_pairs,
        holdout_pairs,
        daytime_scenarios=daytime_scenarios,
        banned_substrings=banned_substrings,
    )
    if replay_ratio is not None and replay_ratio > 0 and dream_pairs:
        max_replay = min(len(replay_train_pairs), int(replay_ratio) * len(dream_pairs))
        rng = random.Random(42 + int(sleep_cycle_before))
        sampled_replay = tuple(rng.sample(list(replay_train_pairs), max_replay))
        active_train_pairs = sampled_replay + tuple(dream_pairs)
    else:
        active_train_pairs = combined_pairs

    history = train_left_brain_lora_and_callosum(
        model,
        tokenizer,
        projector,
        train_pairs=active_train_pairs,
        gap_token_id=gap_token_id,
        device=device,
        epochs=epochs,
        lr_projector=lr_projector,
        lr_lora=lr_lora,
        weight_decay=0.01,
        contrastive_weight=1.5,
        margin_target=0.35,
        warmup_steps=warmup_steps,
    )
    sleep_cycle_after = int(sleep_cycle_before) + 1
    if checkpoint_out_dir is not None:
        meta = save_lora_and_callosum_checkpoint(
            checkpoint_out_dir,
            model=model,
            projector=projector,
            lora_config=lora_config,
            sleep_cycle_index=sleep_cycle_after,
            trained_trajectories_count=len(combined_pairs),
            notes=(
                f"Stage 5 Sleep Cycle {sleep_cycle_after}: gentle incremental REM update on "
                f"{len(active_train_pairs) - len(dream_pairs)}/{len(replay_train_pairs)} replay + "
                f"{len(dream_pairs)} REM dream pairs (warmup_steps={warmup_steps})."
            ),
        )
    else:
        meta = CorpusCallosumMetadata(
            hidden_size=projector.hidden_size,
            intermediate_size=projector.intermediate_size,
            sleep_cycle_index=sleep_cycle_after,
            trained_trajectories_count=len(combined_pairs),
            notes="In-memory Stage 5 REM update",
        )
    return history, meta, combined_pairs
