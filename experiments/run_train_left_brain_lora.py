"""Sub-Step 3 Runner: Train `Qwen/Qwen3-4B` Left-Brain LoRA + `SubliminalGapProjector` and Run Unseen Holdout Evaluation.

Executes all 4 Auditor requirements for Sub-Step 3:
1. Loads real HuggingFace PyTorch `safetensors` weights for `Qwen/Qwen3-4B` in `torch.bfloat16`
   with all base parameters frozen (`requires_grad_(False)`), `gradient_checkpointing_enable()`,
   `use_cache=False`, and `batch_size=1`.
2. Wraps `q_proj, k_proj, v_proj, o_proj` across the top transformer layers with `LoRALinear`
   and trains alongside `SubliminalGapProjector`, saving the checkpoint to `models/left_brain_lora/`
   with `sleep_cycle_index = 1`.
3. Uses `build_post_gap_only_labels` + `splice_subliminal_gaps_into_embeds` so all pre-gap tokens
   and all 4 virtual gap tokens have `label = -100`.
4. Runs Check A (Holdout Log-Probability Flip Rate), Check B (Unplug-the-Cord Check), and
   Check C (Greedy Post-Gap Generation + Real 3-Bridge MiniLM Sensor Verification) on the
   6 unseen holdout pairs (`outputs/gap_trajectories_holdout.jsonl`) and writes the un-rigged
   report to `outputs/left_brain_lora_eval_report.json`.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from experiments.build_and_train_bridge_classifiers import EXTRA_CALIBRATION_CULTURAL_LEAVES
from lmft_lab.gap_trajectory_generator import release_gpu_vram
from lmft_lab.gap_trajectory_schema import (
    load_contrastive_gap_pairs_jsonl,
    validate_train_holdout_disjointness,
)
from lmft_lab.neural_bridge_classifiers import (
    DEFAULT_COMPLEX_FEELING_LEAVES,
    NeuralThreeBridgeSuite,
)
from lmft_lab.reflection_classifier import load_learned_leaves, load_ruleset
from lmft_lab.runtime_store import atomic_json
from lmft_lab.subliminal_gap_splice import SubliminalGapProjector
from lmft_lab.train_left_brain_lora import (
    LoRAConfig,
    ensure_subliminal_gap_token,
    evaluate_holdout_pairs,
    inject_lora_adapters,
    save_lora_and_callosum_checkpoint,
    train_left_brain_lora_and_callosum,
)

HF_MODEL_ID = "Qwen/Qwen3-4B"


def main() -> None:
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    train_path = repo_root / "outputs" / "gap_trajectories_train.jsonl"
    holdout_path = repo_root / "outputs" / "gap_trajectories_holdout.jsonl"
    ckpt_dir = repo_root / "models" / "left_brain_lora"
    report_path = repo_root / "outputs" / "left_brain_lora_eval_report.json"

    train_pairs = load_contrastive_gap_pairs_jsonl(train_path)
    holdout_pairs = load_contrastive_gap_pairs_jsonl(holdout_path)
    validate_train_holdout_disjointness(train_pairs, holdout_pairs)
    print(f"Loaded {len(train_pairs)} train pairs and {len(holdout_pairs)} unseen holdout pairs.")

    print(f"Loading tokenizer and PyTorch bfloat16 weights for {HF_MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    gap_token_id = ensure_subliminal_gap_token(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID,
        dtype=torch.bfloat16,
    )
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    model.to(device)

    lora_cfg = LoRAConfig(
        rank=16,
        lora_alpha=32.0,
        lora_dropout=0.05,
        num_top_layers=18,
    )
    wrapped_modules = inject_lora_adapters(model, lora_cfg)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    hidden_size = int(model.config.hidden_size)
    projector = SubliminalGapProjector(
        hidden_size=hidden_size,
        intermediate_size=256,
    ).to(device=device, dtype=torch.float32)

    trainable_lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_proj_params = sum(p.numel() for p in projector.parameters() if p.requires_grad)
    print(
        f"Wrapped {len(wrapped_modules)} Linear layers with LoRALinear "
        f"(trainable LoRA params = {trainable_lora_params:,}, "
        f"trainable Projector params = {trainable_proj_params:,}, "
        f"hidden_size = {hidden_size})."
    )

    # 1. Pre-Training Holdout Baseline (Epoch 0, before training)
    print("Evaluating pre-training (Epoch 0) baseline on the 6 unseen holdout pairs ...")
    pre_train_summary = evaluate_holdout_pairs(
        model,
        tokenizer,
        projector,
        holdout_pairs,
        gap_token_id=gap_token_id,
        device=device,
        sensors=None,
        max_new_tokens=16,
    )
    print(
        f"Epoch 0 Holdout Baseline: both_flipped={pre_train_summary.both_branches_flipped_count}/{pre_train_summary.total_holdout_pairs}, "
        f"pos_swap_shift={pre_train_summary.positive_chord_swap_shift_count}/{pre_train_summary.total_holdout_pairs}, "
        f"mean_swap_margin={pre_train_summary.mean_net_chord_swap_margin:.4f}"
    )

    # 2. Train SubliminalGapProjector + LoRA on the 17 training pairs
    epochs = 6
    print(f"Training SubliminalGapProjector + LoRA for {epochs} epochs on {len(train_pairs)} pairs ...")
    epoch_history = train_left_brain_lora_and_callosum(
        model,
        tokenizer,
        projector,
        train_pairs=train_pairs,
        gap_token_id=gap_token_id,
        device=device,
        epochs=epochs,
        lr_projector=6e-4,
        lr_lora=2e-4,
        weight_decay=0.01,
        contrastive_weight=1.5,
        margin_target=0.35,
    )
    for ep in epoch_history:
        print(
            f"  Epoch {ep['epoch']}: total_loss={ep['mean_total_loss']:.4f} "
            f"(mle={ep['mean_mle_loss']:.4f}, contrast={ep['mean_contrast_loss']:.4f}, "
            f"margin_a={ep['mean_train_margin_a']:.4f}, margin_b={ep['mean_train_margin_b']:.4f}, "
            f"{ep['wall_time_s']}s)"
        )

    # 3. Save trained LoRA + Corpus Callosum checkpoint (sleep_cycle_index = 1)
    callosum_meta = save_lora_and_callosum_checkpoint(
        ckpt_dir,
        model=model,
        projector=projector,
        lora_config=lora_cfg,
        sleep_cycle_index=1,
        trained_trajectories_count=len(train_pairs),
        notes="Stage 3 Sleep Cycle 1: Qwen3-4B LoRA + SubliminalGapProjector trained on 17 contrastive gap pairs.",
    )
    print(f"Saved checkpoint to {ckpt_dir} (sleep_cycle_index={callosum_meta.sleep_cycle_index}).")

    # 4. Load 3 Right-Brain MiniLM Bridge Classifiers for Check C sensor verification
    sleep_ruleset_path = repo_root / "outputs" / "sleep_consolidated_ruleset.json"
    fallback_ruleset_path = repo_root / "outputs" / "continuous_tiered_ruleset.json"
    active_ruleset_path = sleep_ruleset_path if sleep_ruleset_path.exists() else fallback_ruleset_path
    rules = load_ruleset(active_ruleset_path)
    base_learned = load_learned_leaves(fallback_ruleset_path if fallback_ruleset_path.exists() else active_ruleset_path)
    learned_leaves = base_learned + EXTRA_CALIBRATION_CULTURAL_LEAVES + DEFAULT_COMPLEX_FEELING_LEAVES

    bridges_dir = repo_root / "models" / "right_brain_bridges"
    print(f"Loading 3 Right-Brain MiniLM Bridge Classifiers from {bridges_dir} for Check C ...")
    suite = NeuralThreeBridgeSuite.load_all(bridges_dir)
    sensors = suite.as_three_bridge_sensors()

    # 5. Post-Training Unseen Holdout Evaluation (Check A + Check B + Check C)
    print("Running Post-Training Holdout Evaluation (Check A + Check B + Check C) on 6 unseen pairs ...")
    post_train_summary = evaluate_holdout_pairs(
        model,
        tokenizer,
        projector,
        holdout_pairs,
        gap_token_id=gap_token_id,
        device=device,
        sensors=sensors,
        rules=rules,
        learned_leaves=learned_leaves,
        max_new_tokens=32,
    )

    peak_vram_gb = (
        round(torch.cuda.max_memory_allocated(0) / (1024 ** 3), 3)
        if torch.cuda.is_available()
        else 0.0
    )
    total_wall_s = round(time.perf_counter() - t0, 2)

    report_payload = {
        "schema_version": 1,
        "hf_model_id": HF_MODEL_ID,
        "device": str(device),
        "peak_vram_gb": peak_vram_gb,
        "total_wall_time_s": total_wall_s,
        "train_pairs_count": len(train_pairs),
        "holdout_pairs_count": len(holdout_pairs),
        "lora_config": lora_cfg.to_dict(),
        "corpus_callosum_metadata": callosum_meta.to_dict(),
        "epoch_history": epoch_history,
        "pre_training_holdout_baseline": pre_train_summary.to_dict(),
        "post_training_holdout_evaluation": post_train_summary.to_dict(),
    }
    atomic_json(report_path, report_payload)

    del model
    del projector
    del suite
    release_gpu_vram()

    print("\n" + "=" * 78)
    print("SUB-STEP 3 COMPLETE: LEFT-BRAIN LORA + CORPUS CALLOSUM HOLDOUT REPORT")
    print("=" * 78)
    print(f"Peak VRAM allocated                 : {peak_vram_gb:.3f} GB")
    print(f"Check A (Both Branches Flipped)     : {post_train_summary.both_branches_flipped_count} / {post_train_summary.total_holdout_pairs}")
    print(f"Check A (Positive Chord-Swap Shift) : {post_train_summary.positive_chord_swap_shift_count} / {post_train_summary.total_holdout_pairs}")
    print(f"Check A (Mean Net Chord-Swap Margin): {post_train_summary.mean_net_chord_swap_margin:+.4f} (pre-train: {pre_train_summary.mean_net_chord_swap_margin:+.4f})")
    print(f"Check B (Max Unplugged Shift)       : {post_train_summary.max_unplugged_chord_swap_shift:.6f}")
    print(f"Check C (Greedy Text Changed)       : {post_train_summary.greedy_text_changed_count} / {post_train_summary.total_holdout_pairs}")
    print(f"Check C (Mean Generated Sensor L2)  : {post_train_summary.mean_generated_sensor_l2:.4f}")
    print(f"Saved checkpoint directory          : {ckpt_dir}")
    print(f"Saved evaluation report             : {report_path}")


if __name__ == "__main__":
    main()
