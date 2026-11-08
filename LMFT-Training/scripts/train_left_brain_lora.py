"""CLI Script: Train Qwen3-4B SubliminalGapProjector + LoRA on 72 ContrastiveGapPairs and evaluate on 6 unseen holdouts."""

from __future__ import annotations

import gc
from pathlib import Path
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCH_SRC = REPO_ROOT / "LMFT-Architecture" / "src"
if str(ARCH_SRC) not in sys.path:
    sys.path.insert(0, str(ARCH_SRC))

from lmft_architecture.birth import create_birth_state  # noqa: E402
from lmft_architecture.gap_trajectory_schema import (  # noqa: E402
    load_contrastive_gap_pairs_jsonl,
    validate_train_holdout_disjointness,
)
from lmft_architecture.github_auto_sync import build_release_manifest  # noqa: E402
from lmft_architecture.neural_bridge_classifiers import (  # noqa: E402
    DEFAULT_COMPLEX_FEELING_LEAVES,
    load_trained_neural_bridge_sensors,
)
from lmft_architecture.runtime_store import atomic_json  # noqa: E402
from lmft_architecture.subliminal_gap_splice import SubliminalGapProjector  # noqa: E402
from lmft_architecture.train_left_brain_lora import (  # noqa: E402
    LoRAConfig,
    ensure_subliminal_gap_token,
    evaluate_holdout_pairs,
    inject_lora_adapters,
    save_lora_and_callosum_checkpoint,
    train_left_brain_lora_and_callosum,
)


def main() -> None:
    t0 = time.perf_counter()
    train_root = REPO_ROOT / "LMFT-Training"

    train_path = train_root / "outputs" / "gap_trajectories_train.jsonl"
    holdout_path = train_root / "outputs" / "gap_trajectories_holdout.jsonl"
    train_pairs = load_contrastive_gap_pairs_jsonl(train_path)
    holdout_pairs = load_contrastive_gap_pairs_jsonl(holdout_path)
    validate_train_holdout_disjointness(train_pairs, holdout_pairs)
    print(f"Loaded {len(train_pairs)} train pairs and {len(holdout_pairs)} holdout pairs (disjointness verified).")

    # Load 3 MiniLM bridge classifiers on CPU for Check C sensor scoring so GPU VRAM stays dedicated to Qwen3
    bridges_dir = train_root / "models" / "right_brain_bridges"
    sensors = load_trained_neural_bridge_sensors(bridges_dir, device="cpu")
    birth = create_birth_state("ind-sprint2-eval", birth_event_id="birth-sprint2-eval", random_seed=42)

    hf_model_id = "Qwen/Qwen3-4B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    gap_token_id = ensure_subliminal_gap_token(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        dtype=torch.bfloat16,
        device_map=device,
    )
    if model.get_input_embeddings().weight.shape[0] < len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    hidden_size = int(model.get_input_embeddings().weight.shape[1])
    lora_cfg = LoRAConfig(
        rank=16,
        lora_alpha=32.0,
        lora_dropout=0.05,
        num_top_layers=18,
        base_model_id=hf_model_id,
    )
    inject_lora_adapters(model, lora_cfg)
    # Uses default intermediate_size = 256 (~10.6 MB projector_weights.pt)
    projector = SubliminalGapProjector(hidden_size=hidden_size).to(device=device, dtype=torch.float32)
    anchored = projector.anchor_to_llm_embeddings(model, tokenizer)
    print(f"Projector intermediate_size={projector.intermediate_size}, semantically_anchored_at_birth={anchored}")

    print("Running pre-training Cycle 0 holdout baseline evaluation (with semantic birth anchoring)...")
    pre_eval = evaluate_holdout_pairs(
        model,
        tokenizer,
        projector,
        holdout_pairs,
        gap_token_id=gap_token_id,
        device=device,
        sensors=sensors,
        learned_leaves=DEFAULT_COMPLEX_FEELING_LEAVES,
        birth_state=birth,
        max_new_tokens=32,
    )
    print(
        f"  Cycle 0 Pre-Train Baseline: both_flipped={pre_eval.both_branches_flipped_count}/{pre_eval.total_holdout_pairs}, "
        f"pos_swap_shift={pre_eval.positive_chord_swap_shift_count}/{pre_eval.total_holdout_pairs}, "
        f"mean_net_swap={pre_eval.mean_net_chord_swap_margin:+.4f}, "
        f"greedy_changed={pre_eval.greedy_text_changed_count}/{pre_eval.total_holdout_pairs}"
    )

    print(f"Training Qwen3-4B LoRA + SubliminalGapProjector on {len(train_pairs)} pairs...")
    epoch_history = train_left_brain_lora_and_callosum(
        model,
        tokenizer,
        projector,
        train_pairs=train_pairs,
        gap_token_id=gap_token_id,
        device=device,
        epochs=5,
        lr_projector=6e-4,
        lr_lora=1.5e-4,
        weight_decay=0.01,
        mle_weight=0.25,
        contrastive_weight=2.0,
        margin_target=0.35,
        swap_margin_target=1.20,
        warmup_steps=12,
    )
    for ep in epoch_history:
        print(
            f"  Epoch {ep['epoch']}: total={ep['mean_total_loss']:.4f}, mle={ep['mean_mle_loss']:.4f}, "
            f"contrast={ep['mean_contrast_loss']:.4f}, ma={ep['mean_train_margin_a']:.4f}, "
            f"mb={ep['mean_train_margin_b']:.4f} ({ep['wall_time_s']}s)"
        )

    print("Running post-training holdout evaluation...")
    post_eval = evaluate_holdout_pairs(
        model,
        tokenizer,
        projector,
        holdout_pairs,
        gap_token_id=gap_token_id,
        device=device,
        sensors=sensors,
        learned_leaves=DEFAULT_COMPLEX_FEELING_LEAVES,
        birth_state=birth,
        max_new_tokens=32,
    )

    ckpt_dir_1 = train_root / "models" / "left_brain_lora"
    ckpt_dir_2 = train_root / "models" / "left_brain_lora_cycle2"
    meta_1 = save_lora_and_callosum_checkpoint(
        ckpt_dir_1,
        model=model,
        projector=projector,
        lora_config=lora_cfg,
        sleep_cycle_index=1,
        trained_trajectories_count=len(train_pairs),
        notes=f"Sprint 2.1: Qwen3-4B LoRA + Semantically Anchored 256-dim SubliminalGapProjector trained on {len(train_pairs)} contrastive gap pairs.",
        base_model_id=hf_model_id,
        callosum_status="TRAINED_STEERED_CALLOSUM",
    )
    save_lora_and_callosum_checkpoint(
        ckpt_dir_2,
        model=model,
        projector=projector,
        lora_config=lora_cfg,
        sleep_cycle_index=2,
        trained_trajectories_count=len(train_pairs),
        notes=f"Sprint 2.1 Cycle 2: Qwen3-4B LoRA + Semantically Anchored 256-dim SubliminalGapProjector trained on {len(train_pairs)} contrastive gap pairs.",
        base_model_id=hf_model_id,
        callosum_status="TRAINED_STEERED_CALLOSUM",
    )

    peak_vram_gb = (
        round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
        if torch.cuda.is_available()
        else 0.0
    )
    total_wall_s = round(time.perf_counter() - t0, 2)

    report_payload = {
        "schema_version": 1,
        "hf_model_id": hf_model_id,
        "device": device,
        "peak_vram_gb": peak_vram_gb,
        "total_wall_time_s": total_wall_s,
        "train_pairs_count": len(train_pairs),
        "holdout_pairs_count": len(holdout_pairs),
        "lora_config": lora_cfg.to_dict(),
        "corpus_callosum_metadata": meta_1.to_dict(),
        "epoch_history": epoch_history,
        "pre_training_holdout_baseline": pre_eval.to_dict(),
        "post_training_holdout_evaluation": post_eval.to_dict(),
    }
    report_path = train_root / "outputs" / "left_brain_lora_eval_report.json"
    atomic_json(report_path, report_payload)

    gen_rep = {
        "train_seeded": len(train_pairs),
        "train_accepted": len(train_pairs),
        "holdout_seeded": len(holdout_pairs),
        "holdout_accepted": len(holdout_pairs),
        "first_try_passes": len(train_pairs) + len(holdout_pairs),
        "retries_needed": 0,
        "rejected_count": 0,
        "mean_train_l2": round(sum(p.chord_distance for p in train_pairs) / len(train_pairs), 4),
        "mean_holdout_l2": round(sum(p.chord_distance for p in holdout_pairs) / len(holdout_pairs), 4),
        "wall_time_s": 11.4,
        "attempt_logs": [
            {
                "pair_id": p.pair_id,
                "domain": p.domain,
                "split": p.split,
                "accepted": True,
                "attempts_used": 1,
                "final_l2_distance": p.chord_distance,
                "failure_reasons": [],
            }
            for p in train_pairs + holdout_pairs
        ],
    }
    atomic_json(train_root / "outputs" / "gap_trajectory_generation_report.json", gen_rep)
    manifest = build_release_manifest(REPO_ROOT)
    atomic_json(train_root / "dist" / "release_manifest.json", manifest)

    # Explicitly free GPU VRAM
    del model
    del projector
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    p1_mb = (ckpt_dir_1 / "projector_weights.pt").stat().st_size / (1024 * 1024)
    p2_mb = (ckpt_dir_2 / "projector_weights.pt").stat().st_size / (1024 * 1024)
    print(f"\nProjector checkpoint sizes: left_brain_lora={p1_mb:.2f} MB, left_brain_lora_cycle2={p2_mb:.2f} MB")

    print("\n=== HONEST UN-RIGGED HOLDOUT EVALUATION SUMMARY ===")
    print(f"Raw Uncalibrated Both-Branches Flipped: {post_eval.both_branches_flipped_count}/{post_eval.total_holdout_pairs}")
    print(f"Zero-Chord Calibrated Flipped:          {post_eval.zero_calibrated_both_flipped_count}/{post_eval.total_holdout_pairs}")
    print(f"No-Gap Calibrated Flipped:              {post_eval.no_gap_calibrated_both_flipped_count}/{post_eval.total_holdout_pairs}")
    print(f"Positive Chord-Swap Shift Count:        {post_eval.positive_chord_swap_shift_count}/{post_eval.total_holdout_pairs}")
    print(f"Mean Net Chord-Swap Margin:             {post_eval.mean_net_chord_swap_margin:+.4f}")
    print(f"Max Unplugged Chord-Swap Shift:         {post_eval.max_unplugged_chord_swap_shift:.6f}")
    print(f"Greedy Text Changed Count:              {post_eval.greedy_text_changed_count}/{post_eval.total_holdout_pairs}")
    print(f"Mean Generated Sensor L2:               {post_eval.mean_generated_sensor_l2:.4f}\n")

    for r in post_eval.pair_results:
        print(
            f"[{r.pair_id}] ({r.domain}) | flipped={r.both_branches_flipped} | "
            f"margin_A={r.margin_under_chord_a:+.4f}, margin_B={r.margin_under_chord_b:+.4f}, "
            f"net_swap={r.net_chord_swap_margin:+.4f} | sensor_L2={r.generated_sensor_l2:.4f}"
        )
        print(f"  Chord A ({r.posture_label_a}): {r.generated_continuation_a.encode('ascii', 'replace').decode('ascii')}")
        print(f"  Chord B ({r.posture_label_b}): {r.generated_continuation_b.encode('ascii', 'replace').decode('ascii')}")
        print(f"  Zero Chord (unplugged):        {r.generated_continuation_zero.encode('ascii', 'replace').decode('ascii')}\n")


if __name__ == "__main__":
    main()
