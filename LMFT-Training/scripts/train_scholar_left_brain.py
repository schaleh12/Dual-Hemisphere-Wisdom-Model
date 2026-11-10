"""CLI Script: Train dedicated `models/left_brain_scholar/` (`Impish_LLAMA_4B` ChatML + Norm-Locked Callosum + QK-Only LoRA + Self-Distilled Base-Voice Preservation)."""

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
from lmft_architecture.branch_archives import (  # noqa: E402
    ThreeBranchArchives,
    deliberate_across_three_branches,
)
from lmft_architecture.chord_memory import compile_felt_chord  # noqa: E402
from lmft_architecture.gap_trajectory_schema import (  # noqa: E402
    build_chatml_user_header,
    load_contrastive_gap_pairs_jsonl,
    validate_train_holdout_disjointness,
)
from lmft_architecture.neural_bridge_classifiers import (  # noqa: E402
    DEFAULT_COMPLEX_FEELING_LEAVES,
    load_trained_neural_bridge_sensors,
)
from lmft_architecture.philosophical_abstraction_matrix import (  # noqa: E402
    PHILOSOPHICAL_SHADOW_LEAVES,
)
from lmft_architecture.right_brain_equilibrium import (  # noqa: E402
    RIGHT_BRAIN_WILL_EDGES,
    RIGHT_BRAIN_WILL_VERSION,
    compile_right_brain_vibe,
)
from lmft_architecture.runtime_store import atomic_json  # noqa: E402
from lmft_architecture.subliminal_gap_splice import (  # noqa: E402
    SubliminalChordVector,
    SubliminalGapProjector,
)
from lmft_architecture.three_legged_runtime import (  # noqa: E402
    _compile_mid_sentence_comma_vibe,
)
from lmft_architecture.train_left_brain_lora import (  # noqa: E402
    LoRAConfig,
    SelfDistilledPreservationAnchor,
    build_preservation_anchor_from_base_model,
    ensure_subliminal_gap_token,
    evaluate_holdout_pairs,
    inject_lora_adapters,
    save_lora_and_callosum_checkpoint,
    train_left_brain_lora_and_callosum,
)
from lmft_architecture.virgin_baseline import VIRGIN_FOUNDATIONAL_RULES  # noqa: E402


# 24 everyday conversational, study, science/history, and world-literature prompts
# strictly disjoint from live test prompts ("hello", "How are you doing today?", Moby-Dick/Starbuck/Ahab),
# all 8 unseen holdout pairs, and all 12 SCHOLAR_CURRICULUM_LESSONS books.
EVERYDAY_SELF_DISTILLATION_PROMPTS: tuple[str, ...] = (
    # Casual greetings & check-ins
    "hi there",
    "Hey!",
    "Good afternoon, hope your week is going smoothly.",
    "Thanks for helping me think that through earlier.",
    "What kinds of topics do you enjoy talking about?",
    "Good evening! Just stopping by for a quick chat.",
    # Practical study & reflection
    "What is a good way to stay focused while reading a dense chapter?",
    "Why does taking a short walk often clear up a mental block?",
    "How do you usually organize notes when comparing two different arguments?",
    "What makes a conversation feel genuinely thoughtful rather than rushed?",
    "Why is it easier to spot flaws in an essay after sleeping on it overnight?",
    "How can someone ask sharper questions during a seminar discussion?",
    # General science & history
    "Why does the moon show the same face to Earth throughout its orbit?",
    "How do tree rings record droughts and wet seasons across decades?",
    "Why did printing presses spread ideas so much faster than hand-copied manuscripts?",
    "How does ocean evaporation drive rainfall over distant mountain ranges?",
    "Why do migratory birds fly in V-shaped formations over long distances?",
    "How did Roman aqueducts maintain a steady downhill flow across valleys?",
    # World literature & philosophy (outside curriculum books)
    "Why does Don Quixote insist on seeing windmills as giants in Cervantes's novel?",
    "What role does Virgil play as Dante's guide through the Inferno?",
    "Why does Enkidu's friendship transform Gilgamesh in the Epic of Gilgamesh?",
    "How does Montaigne use personal essays to test his own assumptions?",
    "Why does Socrates compare himself to a gadfly in Plato's Apology?",
    "How does Scheherazade use unfinished stories to survive in One Thousand and One Nights?",
)


@torch.no_grad()
def _generate_frozen_base_chatml_reply(
    model,
    tokenizer,
    user_prompt: str,
    *,
    gap_token_id: int,
    device: str,
    max_new_tokens: int = 40,
) -> str:
    """Greedily generate pure native `Impish_LLAMA_4B`'s ChatML response before LoRA injection."""
    model.eval()
    header = build_chatml_user_header(tokenizer, user_prompt)
    input_ids = torch.tensor(
        [tokenizer.encode(header, add_special_tokens=False)],
        dtype=torch.long,
        device=device,
    )
    attn_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    stop_ids = {int(x) for x in (eos_id, im_end_id) if x is not None and int(x) >= 0}

    generated: list[int] = []
    past_kv = None
    step_ids = input_ids
    for _ in range(max_new_tokens):
        out = model(
            input_ids=step_ids,
            attention_mask=attn_mask,
            past_key_values=past_kv,
            use_cache=True,
        )
        past_kv = out.past_key_values
        logits = out.logits[0, -1, :].clone()
        if 0 <= gap_token_id < logits.shape[-1]:
            logits[gap_token_id] = -1e9
        next_id = int(torch.argmax(logits).item())
        if next_id in stop_ids:
            break
        generated.append(next_id)
        step_ids = torch.tensor([[next_id]], dtype=torch.long, device=device)
        attn_mask = torch.cat([attn_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=1)
        decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
        if len(generated) >= 4 and decoded.endswith((".", "!", "?")):
            break
    return " ".join(tokenizer.decode(generated, skip_special_tokens=True).replace("\n", " ").split()).strip()


def _collect_everyday_preservation_anchors(
    model,
    tokenizer,
    *,
    sensors,
    birth_state,
    gap_token_id: int,
    device: str,
) -> list[SelfDistilledPreservationAnchor]:
    """Collect frozen base-model KL preservation anchors across everyday prompts at both gap positions."""
    archives = ThreeBranchArchives()
    learned_leaves = PHILOSOPHICAL_SHADOW_LEAVES + DEFAULT_COMPLEX_FEELING_LEAVES
    anchors: list[SelfDistilledPreservationAnchor] = []

    for idx, prompt_text in enumerate(EVERYDAY_SELF_DISTILLATION_PROMPTS):
        base_reply = _generate_frozen_base_chatml_reply(
            model,
            tokenizer,
            prompt_text,
            gap_token_id=gap_token_id,
            device=device,
            max_new_tokens=40,
        )
        if not base_reply:
            continue

        ext_coords, ext_exp, queried, skipped = sensors.score_text(
            prompt_text,
            learned_leaves=learned_leaves,
            branch_gating=True,
        )
        social_drift = archives.executive.detect_social_drift(ext_coords)
        vibe_1, _ = compile_right_brain_vibe(
            birth_state,
            ext_coords,
            ext_exp,
            VIRGIN_FOUNDATIONAL_RULES,
            pass_index=1,
            event_id=f"pres-lead-{idx}",
            learned_leaves=learned_leaves,
            queried_child_leaves=queried,
            skipped_child_leaves=skipped,
            social_drift=social_drift,
            will_edges=RIGHT_BRAIN_WILL_EDGES,
        )
        felt_1 = vibe_1.felt_chord or compile_felt_chord(
            ext_coords,
            salience=vibe_1.salience,
            valence=vibe_1.valence,
            arousal=vibe_1.arousal,
            resonance=vibe_1.resonance,
            learned_leaves=learned_leaves,
        )
        delib_1 = deliberate_across_three_branches(
            birth_state,
            ext_coords,
            ext_exp,
            felt_1,
            sensors,
            archives,
            RIGHT_BRAIN_WILL_EDGES,
            RIGHT_BRAIN_WILL_VERSION,
            scene_index=idx + 1,
            learned_leaves=learned_leaves,
            social_drift=social_drift,
            branch_gating=True,
        )
        chord_lead = SubliminalChordVector.from_right_brain_vibe(
            vibe_1,
            deliberation=delib_1,
            social_drift=social_drift,
            learned_leaves=learned_leaves,
        )
        anchors.append(
            build_preservation_anchor_from_base_model(
                model,
                tokenizer,
                user_prompt=prompt_text,
                clause_1_prefix="",
                continuation_text=base_reply,
                chord=chord_lead,
                gap_token_id=gap_token_id,
                device=device,
                lead_with_gap=True,
                prompt_format="chatml",
                top_k=64,
            )
        )

        comma_pos = base_reply.find(",")
        if comma_pos > 1:
            c1_part = base_reply[: comma_pos + 1].strip()
            cont_part = base_reply[comma_pos + 1 :].strip()
            if len(cont_part.split()) >= 2:
                comma_coords, comma_exp, _, _ = sensors.score_text(
                    c1_part,
                    learned_leaves=learned_leaves,
                    branch_gating=True,
                )
                raw_comma_vibe, _ = compile_right_brain_vibe(
                    birth_state,
                    ext_coords,
                    ext_exp,
                    VIRGIN_FOUNDATIONAL_RULES,
                    pass_index=2,
                    event_id=f"pres-comma-{idx}",
                    learned_leaves=learned_leaves,
                    internal_theory_coords=comma_coords,
                    internal_theory_experiential=comma_exp,
                    queried_child_leaves=queried,
                    skipped_child_leaves=skipped,
                    social_drift=social_drift,
                    will_edges=RIGHT_BRAIN_WILL_EDGES,
                )
                mid_comma_vibe = _compile_mid_sentence_comma_vibe(vibe_1, raw_comma_vibe)
                comma_chord = SubliminalChordVector.from_right_brain_vibe(
                    mid_comma_vibe,
                    deliberation=delib_1,
                    social_drift=social_drift,
                    learned_leaves=learned_leaves,
                )
                anchors.append(
                    build_preservation_anchor_from_base_model(
                        model,
                        tokenizer,
                        user_prompt=prompt_text,
                        clause_1_prefix=c1_part,
                        continuation_text=cont_part,
                        chord=comma_chord,
                        gap_token_id=gap_token_id,
                        device=device,
                        lead_with_gap=False,
                        prompt_format="chatml",
                        top_k=64,
                    )
                )
    return anchors


def main() -> None:
    t0 = time.perf_counter()
    train_root = REPO_ROOT / "LMFT-Training"

    train_path = train_root / "outputs" / "scholar_gap_trajectories_train.jsonl"
    holdout_path = train_root / "outputs" / "scholar_gap_trajectories_holdout.jsonl"
    train_pairs = load_contrastive_gap_pairs_jsonl(train_path)
    holdout_pairs = load_contrastive_gap_pairs_jsonl(holdout_path)
    validate_train_holdout_disjointness(train_pairs, holdout_pairs)
    print(f"Loaded {len(train_pairs)} Scholar train pairs and {len(holdout_pairs)} unseen holdout pairs.")

    bridges_dir = train_root / "models" / "right_brain_bridges"
    sensors = load_trained_neural_bridge_sensors(bridges_dir, device="cpu")
    birth = create_birth_state("ind-scholar-eval", birth_event_id="birth-scholar-eval", random_seed=42)

    hf_model_id = "SicariusSicariiStuff/Impish_LLAMA_4B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    torch.manual_seed(42)
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id, fix_mistral_regex=True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id)

    model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
    )

    hidden_size = int(model.get_input_embeddings().weight.shape[1])
    projector = SubliminalGapProjector(hidden_size=hidden_size).to(device=device, dtype=torch.float32)
    anchored = projector.anchor_to_llm_embeddings(model, tokenizer)
    # Lock output_norm to the LLM's measured word-embedding RMS so virtual tokens
    # steer purely by semantic direction on the native word sphere and never inflate off-manifold.
    projector.output_norm.requires_grad_(False)

    gap_token_id = ensure_subliminal_gap_token(tokenizer)
    if model.get_input_embeddings().weight.shape[0] < len(tokenizer):
        try:
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        except TypeError:
            model.resize_token_embeddings(len(tokenizer))

    with torch.no_grad():
        emb_w = model.get_input_embeddings().weight[:128256].detach().to(dtype=torch.float32)
        word_norm = float(torch.norm(emb_w, dim=-1).mean().item())
        probe_chord = train_pairs[0].chord_a.to_tensor(device=device, dtype=torch.float32).unsqueeze(0)
        v_tokens = projector(probe_chord)
        virt_norm_init = float(torch.norm(v_tokens, dim=-1).mean().item())
    print(
        f"Projector hidden_size={hidden_size}, intermediate_size={projector.intermediate_size}, "
        f"anchored={anchored}, word_l2_norm={word_norm:.4f}, virtual_gap_l2_norm_init={virt_norm_init:.4f}"
    )

    print("Collecting self-distilled everyday preservation anchors from frozen native Impish_LLAMA_4B...")
    preservation_anchors = _collect_everyday_preservation_anchors(
        model,
        tokenizer,
        sensors=sensors,
        birth_state=birth,
        gap_token_id=gap_token_id,
        device=device,
    )
    lead_cnt = sum(1 for a in preservation_anchors if a.lead_with_gap)
    comma_cnt = len(preservation_anchors) - lead_cnt
    print(f"  Collected {len(preservation_anchors)} self-distilled anchors ({lead_cnt} lead_with_gap, {comma_cnt} comma-spliced).")

    # QK-Only LoRA: routes attention across virtual `<|subliminal_gap|>` tokens without writing
    # training-set phrases into the residual stream via `v_proj` / `o_proj`.
    lora_cfg = LoRAConfig(
        rank=16,
        lora_alpha=16.0,
        lora_dropout=0.05,
        target_modules=("q_proj", "k_proj"),
        num_top_layers=16,
        base_model_id=hf_model_id,
    )
    inject_lora_adapters(model, lora_cfg)

    print("Running Cycle 0 pre-training holdout evaluation (ChatML)...")
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
        max_new_tokens=36,
        prompt_format="chatml",
    )
    print(
        f"  Pre-Train (comma gap): both_flipped={pre_eval.both_branches_flipped_count}/{pre_eval.total_holdout_pairs}, "
        f"pos_swap={pre_eval.positive_chord_swap_shift_count}/{pre_eval.total_holdout_pairs}, "
        f"net_swap={pre_eval.mean_net_chord_swap_margin:+.4f}, "
        f"greedy_changed={pre_eval.greedy_text_changed_count}/{pre_eval.total_holdout_pairs}"
    )

    print(
        f"Training Scholar Left-Brain ({hf_model_id}) with QK-Only LoRA + {len(preservation_anchors)} "
        f"self-distilled KL preservation anchors (prompt_format='chatml')..."
    )
    epoch_history = train_left_brain_lora_and_callosum(
        model,
        tokenizer,
        projector,
        train_pairs=train_pairs,
        gap_token_id=gap_token_id,
        device=device,
        epochs=4,
        lr_projector=6e-4,
        lr_lora=1.2e-4,
        weight_decay=0.01,
        mle_weight=0.25,
        contrastive_weight=1.35,
        margin_target=0.25,
        swap_margin_target=0.75,
        warmup_steps=8,
        include_full_turn_mle=False,
        train_both_gap_positions=True,
        prompt_format="chatml",
        preservation_anchors=preservation_anchors,
        preservation_kl_weight=1.0,
    )
    for ep in epoch_history:
        print(
            f"  Epoch {ep['epoch']}: total={ep['mean_total_loss']:.4f}, mle={ep['mean_mle_loss']:.4f}, "
            f"contrast={ep['mean_contrast_loss']:.4f}, pres_kl={ep['mean_preservation_kl_loss']:.4f}, "
            f"ma={ep['mean_train_margin_a']:.4f}, mb={ep['mean_train_margin_b']:.4f} ({ep['wall_time_s']}s)"
        )

    with torch.no_grad():
        v_tokens_post = projector(probe_chord)
        virt_norm_post = float(torch.norm(v_tokens_post, dim=-1).mean().item())
    print(f"Post-training virtual_gap_l2_norm={virt_norm_post:.4f} (word_l2_norm={word_norm:.4f})")

    scholar_ckpt_dir = train_root / "models" / "left_brain_scholar"
    for stale_name in ("live_projector_weights.pt", "live_lora_rank_scales.pt", "live_projector_meta.json"):
        stale_file = scholar_ckpt_dir / stale_name
        if stale_file.exists():
            stale_file.unlink()

    meta = save_lora_and_callosum_checkpoint(
        scholar_ckpt_dir,
        model=model,
        projector=projector,
        lora_config=lora_cfg,
        sleep_cycle_index=1,
        trained_trajectories_count=len(train_pairs),
        notes=(
            f"LMFT-Scholar ChatML checkpoint ({hf_model_id}): norm-locked SubliminalGapProjector "
            f"(norm {virt_norm_init:.4f} -> {virt_norm_post:.4f}) + QK-only LoRA + "
            f"{len(preservation_anchors)} self-distilled base-voice KL anchors."
        ),
        base_model_id=hf_model_id,
        callosum_status="TRAINED_STEERED_CALLOSUM",
        prompt_format="chatml",
    )
    print(f"Saved Scholar checkpoint to {scholar_ckpt_dir}.")

    print("Evaluating unseen holdout pairs (comma gap, lead_with_gap=False)...")
    post_eval_comma = evaluate_holdout_pairs(
        model,
        tokenizer,
        projector,
        holdout_pairs,
        gap_token_id=gap_token_id,
        device=device,
        sensors=sensors,
        learned_leaves=DEFAULT_COMPLEX_FEELING_LEAVES,
        birth_state=birth,
        max_new_tokens=36,
        lead_with_gap=False,
        prompt_format="chatml",
    )
    print(
        f"  Post-Train (comma gap): both_flipped={post_eval_comma.both_branches_flipped_count}/{post_eval_comma.total_holdout_pairs}, "
        f"pos_swap={post_eval_comma.positive_chord_swap_shift_count}/{post_eval_comma.total_holdout_pairs}, "
        f"net_swap={post_eval_comma.mean_net_chord_swap_margin:+.4f}, "
        f"greedy_changed={post_eval_comma.greedy_text_changed_count}/{post_eval_comma.total_holdout_pairs}, "
        f"mean_sensor_l2={post_eval_comma.mean_generated_sensor_l2:.4f}"
    )

    print("Evaluating unseen holdout pairs (turn-start gap, lead_with_gap=True)...")
    post_eval_lead = evaluate_holdout_pairs(
        model,
        tokenizer,
        projector,
        holdout_pairs,
        gap_token_id=gap_token_id,
        device=device,
        sensors=sensors,
        learned_leaves=DEFAULT_COMPLEX_FEELING_LEAVES,
        birth_state=birth,
        max_new_tokens=36,
        lead_with_gap=True,
        prompt_format="chatml",
    )
    print(
        f"  Post-Train (lead gap):  both_flipped={post_eval_lead.both_branches_flipped_count}/{post_eval_lead.total_holdout_pairs}, "
        f"pos_swap={post_eval_lead.positive_chord_swap_shift_count}/{post_eval_lead.total_holdout_pairs}, "
        f"net_swap={post_eval_lead.mean_net_chord_swap_margin:+.4f}, "
        f"greedy_changed={post_eval_lead.greedy_text_changed_count}/{post_eval_lead.total_holdout_pairs}, "
        f"mean_sensor_l2={post_eval_lead.mean_generated_sensor_l2:.4f}"
    )

    elapsed = round(time.perf_counter() - t0, 2)
    report_payload = {
        "base_model_id": hf_model_id,
        "prompt_format": "chatml",
        "word_embedding_mean_l2_norm": round(word_norm, 6),
        "virtual_gap_l2_norm_init": round(virt_norm_init, 6),
        "virtual_gap_l2_norm_post_train": round(virt_norm_post, 6),
        "self_distilled_preservation_anchor_count": len(preservation_anchors),
        "include_full_turn_mle": False,
        "train_both_gap_positions": True,
        "lora_config": lora_cfg.to_dict(),
        "callosum_metadata": meta.to_dict(),
        "epoch_history": epoch_history,
        "pre_training_holdout_comma": pre_eval.to_dict(),
        "post_training_holdout_comma": post_eval_comma.to_dict(),
        "post_training_holdout_lead": post_eval_lead.to_dict(),
        "elapsed_wall_time_s": elapsed,
    }
    report_path = scholar_ckpt_dir / "scholar_holdout_report.json"
    atomic_json(report_path, report_payload)
    print(f"\nSaved holdout report to {report_path} in {elapsed}s.")
    for item in post_eval_comma.pair_results:
        print(
            f"  [{item.pair_id}] flipped={item.both_branches_flipped} | "
            f"margin_a={item.margin_under_chord_a:+.3f}, margin_b={item.margin_under_chord_b:+.3f}, "
            f"net_swap={item.net_chord_swap_margin:+.3f}"
        )
        print(f"    greedy_a: {item.generated_continuation_a.encode('ascii', 'replace').decode('ascii')}")
        print(f"    greedy_b: {item.generated_continuation_b.encode('ascii', 'replace').decode('ascii')}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
