"""Sub-Step 2B Runner: Generate `outputs/gap_trajectories_train.jsonl` and `outputs/gap_trajectories_holdout.jsonl`.

Uses:
1. Real local `Qwen3.5-9B-Q4_K_M.gguf` Teacher (`DirectGGUFTeacher`) to generate the shared
   `clause_1_prefix` and contrasting `continuation_a` / `continuation_b` clauses.
2. Real 3 PyTorch MiniLM-L12 Bridge Classifiers (`models/right_brain_bridges/*`) to score
   and verify `chord_a` and `chord_b`.
3. Explicitly unloads `Qwen3.5-9B` and clears CUDA cache at completion so 100% of the
   16 GB RTX 4080 VRAM is free before Sub-Step 3.
"""

from __future__ import annotations

from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from experiments.build_and_train_bridge_classifiers import EXTRA_CALIBRATION_CULTURAL_LEAVES
from experiments.run_feasibility_scaling_study import TEACHER_GGUF, DirectGGUFTeacher
from lmft_architecture.gap_trajectory_generator import (
    HOLDOUT_GAP_SCENARIO_SEEDS,
    TRAIN_GAP_SCENARIO_SEEDS,
    generate_contrastive_gap_dataset,
    release_gpu_vram,
)
from lmft_architecture.local_models import load_config
from lmft_architecture.neural_bridge_classifiers import (
    DEFAULT_COMPLEX_FEELING_LEAVES,
    NeuralThreeBridgeSuite,
)
from lmft_architecture.reflection_classifier import load_learned_leaves, load_ruleset


def main() -> None:
    config = load_config(repo_root / "config" / "local_models.json")
    generation_cfg = dict(config["generation"])
    generation_cfg["max_tokens"] = 320

    sleep_ruleset_path = repo_root / "outputs" / "sleep_consolidated_ruleset.json"
    fallback_ruleset_path = repo_root / "outputs" / "continuous_tiered_ruleset.json"
    active_ruleset_path = sleep_ruleset_path if sleep_ruleset_path.exists() else fallback_ruleset_path

    rules = load_ruleset(active_ruleset_path)
    base_learned_path = fallback_ruleset_path if fallback_ruleset_path.exists() else active_ruleset_path
    base_learned = load_learned_leaves(base_learned_path)
    learned_leaves = base_learned + EXTRA_CALIBRATION_CULTURAL_LEAVES + DEFAULT_COMPLEX_FEELING_LEAVES

    bridges_dir = repo_root / "models" / "right_brain_bridges"
    print(f"Loading 3 Right-Brain MiniLM Bridge Classifiers from {bridges_dir} ...")
    suite = NeuralThreeBridgeSuite.load_all(bridges_dir)
    sensors = suite.as_three_bridge_sensors()

    train_out = repo_root / "outputs" / "gap_trajectories_train.jsonl"
    holdout_out = repo_root / "outputs" / "gap_trajectories_holdout.jsonl"
    report_out = repo_root / "outputs" / "gap_trajectory_generation_report.json"

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

    print(f"Loading Offline 9B Teacher from {TEACHER_GGUF} ...")
    with DirectGGUFTeacher(TEACHER_GGUF, generation_cfg) as teacher:
        call_counter = 0

        def _teacher_gen(prompt: str) -> str:
            nonlocal call_counter
            call_counter += 1
            req = {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a precise Dual-Hemisphere training data author. "
                            "Return only valid JSON matching the requested schema."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2 if call_counter > 1 else 0.0,
                "seed": int(generation_cfg.get("seed", 42)) + call_counter,
                "max_tokens": 320,
                "stream": False,
                "cache_prompt": False,
                "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {"type": "json_object", "schema": gap_json_schema},
            }
            raw_reply, _trace = teacher.generate(
                "teacher",
                req,
                "gap-trajectory-teacher-v1",
                lambda raw: raw,
            )
            return raw_reply

        train_pairs, holdout_pairs, report = generate_contrastive_gap_dataset(
            train_seeds=TRAIN_GAP_SCENARIO_SEEDS,
            holdout_seeds=HOLDOUT_GAP_SCENARIO_SEEDS,
            teacher_generate_fn=_teacher_gen,
            sensors=sensors,
            rules=rules,
            learned_leaves=learned_leaves,
            train_out_path=train_out,
            holdout_out_path=holdout_out,
            report_out_path=report_out,
            max_attempts=3,
        )

    # Explicitly unload MiniLM suite and clear CUDA cache so 100% of VRAM is free
    del suite
    release_gpu_vram()

    print("\n" + "=" * 76)
    print("SUB-STEP 2B COMPLETE: CONTRASTIVE GAP TRAJECTORY GENERATION REPORT")
    print("=" * 76)
    print(f"Train pairs accepted   : {report.train_accepted} / {report.train_seeded} (mean L2 = {report.mean_train_l2:.4f})")
    print(f"Holdout pairs accepted : {report.holdout_accepted} / {report.holdout_seeded} (mean L2 = {report.mean_holdout_l2:.4f})")
    print(f"First-try passes       : {report.first_try_passes}")
    print(f"Retries needed         : {report.retries_needed}")
    print(f"Rejected seeds         : {report.rejected_count}")
    print(f"Wall time              : {report.wall_time_s:.2f} s")
    print(f"Saved train JSONL      : {train_out}")
    print(f"Saved holdout JSONL    : {holdout_out}")
    print(f"Saved report JSON      : {report_out}")


if __name__ == "__main__":
    main()
