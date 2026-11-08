"""CLI Script: Retrain Qwen/Qwen3-8B and SicariusSicariiStuff/Impish_Bloodmoon_12B on the 72-pair polarity-fixed dataset with Semantic Birth Anchoring."""

from __future__ import annotations

import gc
from pathlib import Path
import sys
import time

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCH_SRC = REPO_ROOT / "LMFT-Architecture" / "src"
if str(ARCH_SRC) not in sys.path:
    sys.path.insert(0, str(ARCH_SRC))

from lmft_architecture.gap_trajectory_schema import (  # noqa: E402
    load_contrastive_gap_pairs_jsonl,
    validate_train_holdout_disjointness,
)
from lmft_architecture.github_auto_sync import build_release_manifest  # noqa: E402
from lmft_architecture.impish_bloodmoon_adapter import (  # noqa: E402
    train_impish_bloodmoon_callosum_and_lora,
)


def _print_model_summary(label: str, res: dict) -> None:
    pre = res.get("pre_training_holdout_baseline") or {}
    post = res.get("holdout_report") or {}
    print(f"\n==================== {label} ====================")
    if pre:
        print(
            f"  [Cycle 0 Pre-Train Baseline] raw_flipped={pre.get('both_branches_flipped_count')}/{pre.get('total_holdout_pairs')}, "
            f"zero_cal={pre.get('zero_calibrated_both_flipped_count')}/{pre.get('total_holdout_pairs')}, "
            f"nogap_cal={pre.get('no_gap_calibrated_both_flipped_count')}/{pre.get('total_holdout_pairs')}, "
            f"pos_swap={pre.get('positive_chord_swap_shift_count')}/{pre.get('total_holdout_pairs')}, "
            f"mean_net_swap={pre.get('mean_net_chord_swap_margin', 0.0):+.4f}, "
            f"greedy_changed={pre.get('greedy_text_changed_count')}/{pre.get('total_holdout_pairs')}, "
            f"sensor_l2={pre.get('mean_generated_sensor_l2', 0.0):.4f}"
        )
    print(
        f"  [Post-Train Holdout Eval]    raw_flipped={post.get('both_branches_flipped_count')}/{post.get('total_holdout_pairs')}, "
        f"zero_cal={post.get('zero_calibrated_both_flipped_count')}/{post.get('total_holdout_pairs')}, "
        f"nogap_cal={post.get('no_gap_calibrated_both_flipped_count')}/{post.get('total_holdout_pairs')}, "
        f"pos_swap={post.get('positive_chord_swap_shift_count')}/{post.get('total_holdout_pairs')}, "
        f"mean_net_swap={post.get('mean_net_chord_swap_margin', 0.0):+.4f}, "
        f"unplugged_shift={post.get('max_unplugged_chord_swap_shift', 0.0):.6f}, "
        f"greedy_changed={post.get('greedy_text_changed_count')}/{post.get('total_holdout_pairs')}, "
        f"sensor_l2={post.get('mean_generated_sensor_l2', 0.0):.4f}"
    )
    for pr in post.get("pair_results", []):
        print(
            f"    - {pr['pair_id']}: ma={pr['margin_under_chord_a']:+.4f}, "
            f"mb={pr['margin_under_chord_b']:+.4f}, swap={pr['net_chord_swap_margin']:+.4f}, "
            f"raw_flip={pr['both_branches_flipped']}, zero_cal_flip={pr['zero_calibrated_both_flipped']}"
        )
        print(f"      Chord A Greedy: {pr['generated_continuation_a']}")
        print(f"      Chord B Greedy: {pr['generated_continuation_b']}")


def main() -> None:
    t0 = time.perf_counter()
    train_root = REPO_ROOT / "LMFT-Training"
    train_pairs = load_contrastive_gap_pairs_jsonl(train_root / "outputs" / "gap_trajectories_train.jsonl")
    holdout_pairs = load_contrastive_gap_pairs_jsonl(train_root / "outputs" / "gap_trajectories_holdout.jsonl")
    validate_train_holdout_disjointness(train_pairs, holdout_pairs)
    print(f"Loaded {len(train_pairs)} train pairs and {len(holdout_pairs)} holdout pairs (disjointness verified).")

    # 1. Retrain Checkpoint #2: Qwen/Qwen3-8B (4-bit NF4, 5 epochs, intermediate_size=256)
    import json

    qwen8b_dir = train_root / "models" / "left_brain_qwen3_8b"
    qwen8b_meta_path = qwen8b_dir / "corpus_callosum_meta.json"
    qwen8b_report_path = qwen8b_dir / "impish_bloodmoon_holdout_report.json"
    already_8b = (
        qwen8b_meta_path.exists()
        and qwen8b_report_path.exists()
        and json.loads(qwen8b_meta_path.read_text(encoding="utf-8")).get("trained_trajectories_count") == len(train_pairs)
    )
    if already_8b:
        print(f"\n[1/2] Qwen/Qwen3-8B already trained on {len(train_pairs)} pairs in {qwen8b_dir}; reusing saved checkpoint.")
        saved_8b_rep = json.loads(qwen8b_report_path.read_text(encoding="utf-8"))
        _print_model_summary(
            "Qwen/Qwen3-8B (Saved)",
            {
                "pre_training_holdout_baseline": saved_8b_rep.get("pre_training_holdout_baseline"),
                "holdout_report": saved_8b_rep,
            },
        )
    else:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
        print(f"\n[1/2] Training Qwen/Qwen3-8B (4-bit NF4) -> {qwen8b_dir} ...")
        t_8b = time.perf_counter()
        res_8b = train_impish_bloodmoon_callosum_and_lora(
            qwen8b_dir,
            hf_model_id="Qwen/Qwen3-8B",
            train_pairs=train_pairs,
            holdout_pairs=holdout_pairs,
            epochs=5,
            lr_projector=6e-4,
            lr_lora=1.5e-4,
            lora_rank=16,
            lora_alpha=32.0,
            lora_dropout=0.05,
            num_top_layers=18,
            intermediate_size=256,
            sleep_cycle_index=2,
            warmup_steps=12,
            load_in_4bit=True,
            repo_root=REPO_ROOT,
        )
        print(f"Finished Qwen/Qwen3-8B in {time.perf_counter() - t_8b:.1f}s")
        _print_model_summary("Qwen/Qwen3-8B", res_8b)
        del res_8b

    # Free GPU memory before loading 12B model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Retrain Checkpoint #3: SicariusSicariiStuff/Impish_Bloodmoon_12B (4-bit NF4, 5 epochs, intermediate_size=256)
    torch.manual_seed(42)
    impish_dir = train_root / "models" / "left_brain_impish_bloodmoon"
    print(f"\n[2/2] Training SicariusSicariiStuff/Impish_Bloodmoon_12B (4-bit NF4) -> {impish_dir} ...")
    t_12b = time.perf_counter()
    res_12b = train_impish_bloodmoon_callosum_and_lora(
        impish_dir,
        hf_model_id="SicariusSicariiStuff/Impish_Bloodmoon_12B",
        train_pairs=train_pairs,
        holdout_pairs=holdout_pairs,
        epochs=5,
        lr_projector=6e-4,
        lr_lora=1.5e-4,
        lora_rank=16,
        lora_alpha=32.0,
        lora_dropout=0.05,
        num_top_layers=8,
        intermediate_size=256,
        sleep_cycle_index=1,
        warmup_steps=12,
        load_in_4bit=True,
        repo_root=REPO_ROOT,
    )
    print(f"Finished SicariusSicariiStuff/Impish_Bloodmoon_12B in {time.perf_counter() - t_12b:.1f}s")
    _print_model_summary("SicariusSicariiStuff/Impish_Bloodmoon_12B", res_12b)

    build_release_manifest(REPO_ROOT)
    print(f"\nAll done in {time.perf_counter() - t0:.1f}s.")


if __name__ == "__main__":
    main()
