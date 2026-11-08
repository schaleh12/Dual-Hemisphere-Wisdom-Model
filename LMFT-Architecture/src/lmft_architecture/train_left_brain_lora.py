"""Sub-Step 3: PyTorch Left-Brain LoRA + Corpus Callosum (`SubliminalGapProjector`) Trainer & Holdout Evaluator.

Plain-English Purpose:
Trains `Qwen3-4B` (loaded in PyTorch `bfloat16` with frozen base weights) alongside
`SubliminalGapProjector` and lightweight `LoRALinear` adapters (`q_proj`, `k_proj`,
`v_proj`, `o_proj`) so the Left Brain learns to steer its post-gap Clause 2 continuation
directly from the 4 non-verbal virtual tokens spliced into `<|subliminal_gap|>`:

1. Pre-Gap Label Masking (`build_post_gap_only_labels`):
   - All tokens before `<|subliminal_gap|>` and all 4 virtual gap tokens have `label = -100`.
   - 100% of the causal LM + contrastive chord-routing gradient focuses on predicting
     `continuation_a` vs. `continuation_b` after the gap.
2. Self-Contained PyTorch `LoRALinear`:
   - Wraps target attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) without
     requiring external C++/bitsandbytes DLLs on Windows.
   - Saves both the `SubliminalGapProjector` (`save_corpus_callosum_checkpoint`) and the
     LoRA adapter weights (`lora_adapters.pt` + `lora_config.json`) to `models/left_brain_lora/`.
3. Three Un-Rigged Falsification Checks on Unseen Holdout (`outputs/gap_trajectories_holdout.jsonl`):
   - **Check A (Holdout Log-Probability Flip Rate)**: With the pre-gap prefix frozen, checks
     how many of the 6 unseen holdout pairs prefer `continuation_a` under `chord_a` and flip
     to preferring `continuation_b` under `chord_b`.
   - **Check B (Unplug-the-Cord Check)**: Verifies that zeroing the 52-float chord
     (`with_zeroed_cord()`) collapses the chord-swap log-prob shift to `0.0`.
   - **Check C (Greedy Post-Gap Generation + 3-Bridge MiniLM Sensor Check)**: Greedily
     generates post-gap continuations under `chord_a` vs. `chord_b` on the 6 unseen holdout
     prompts and scores them with the real 3 PyTorch MiniLM-L12 Bridge Classifiers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .birth import BirthState, create_birth_state
from .gap_trajectory_generator import score_continuation_to_chord
from .gap_trajectory_schema import (
    ContrastiveGapPair,
    build_post_gap_only_labels,
    chord_l2_distance,
)
from .neural_bridge_classifiers import DEFAULT_COMPLEX_FEELING_LEAVES
from .reflection_classifier import FuzzyConceptRule, LearnedLeaf
from .right_brain_equilibrium import ThreeBridgeSensors
from .runtime_store import atomic_json
from .subliminal_gap_splice import (
    IGNORE_LABEL_INDEX,
    SUBLIMINAL_GAP_TOKEN,
    CorpusCallosumMetadata,
    SubliminalChordVector,
    SubliminalGapProjector,
    load_corpus_callosum_checkpoint,
    save_corpus_callosum_checkpoint,
    splice_subliminal_gaps_into_embeds,
)

DEFAULT_LORA_TARGET_MODULES: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


class LoRALinear(nn.Module):
    """Self-contained PyTorch LoRA wrapper around a frozen `nn.Linear` layer."""

    def __init__(
        self,
        base_linear: nn.Linear,
        *,
        rank: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        self.base_linear = base_linear
        self.base_linear.requires_grad_(False)
        self.in_features = int(base_linear.in_features)
        self.out_features = int(base_linear.out_features)
        self.rank = int(rank)
        self.lora_alpha = float(lora_alpha)
        self.scaling = float(lora_alpha) / float(rank)
        self.dropout = nn.Dropout(float(lora_dropout)) if lora_dropout > 0.0 else nn.Identity()

        device = base_linear.weight.device
        # Keep trainable LoRA weights in float32 for clean gradient accumulation even when base is bfloat16
        self.lora_a = nn.Linear(self.in_features, self.rank, bias=False, device=device, dtype=torch.float32)
        self.lora_b = nn.Linear(self.rank, self.out_features, bias=False, device=device, dtype=torch.float32)
        self.register_buffer(
            "rank_scale",
            torch.ones(self.rank, device=device, dtype=torch.float32),
            persistent=False,
        )

        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base_linear(x)
        x_fp32 = self.dropout(x.to(dtype=torch.float32))
        scale_vec = self.rank_scale.to(device=x_fp32.device, dtype=torch.float32)
        h_rank = self.lora_a(x_fp32) * scale_vec
        lora_delta = self.lora_b(h_rank) * self.scaling
        return base_out + lora_delta.to(dtype=base_out.dtype)


@dataclass(frozen=True)
class LoRAConfig:
    """Configuration for Left-Brain LoRA adapter injection."""

    rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = DEFAULT_LORA_TARGET_MODULES
    num_top_layers: int | None = 16
    base_model_id: str = "Qwen/Qwen3-4B"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "rank": self.rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": list(self.target_modules),
            "num_top_layers": self.num_top_layers,
            "base_model_id": self.base_model_id,
        }


def inject_lora_adapters(
    model: nn.Module,
    config: LoRAConfig = LoRAConfig(),
) -> list[str]:
    """Freeze all base parameters in `model` and wrap matching `nn.Linear` projections with `LoRALinear`."""
    model.requires_grad_(False)

    # Determine layer filter if model has `.model.layers` or `.layers`
    layers_seq: Sequence[nn.Module] | nn.ModuleList | None = None
    if hasattr(model, "model") and hasattr(getattr(model, "model"), "layers"):
        layers_seq = getattr(getattr(model, "model"), "layers")
    elif hasattr(model, "layers") and isinstance(getattr(model, "layers"), (Sequence, nn.ModuleList)):
        layers_seq = getattr(model, "layers")

    allowed_layer_prefixes: set[str] | None = None
    if layers_seq is not None and config.num_top_layers is not None:
        total_layers = len(layers_seq)
        start_idx = max(0, total_layers - int(config.num_top_layers))
        prefix_root = "model.layers." if hasattr(model, "model") and hasattr(getattr(model, "model"), "layers") else "layers."
        allowed_layer_prefixes = {f"{prefix_root}{i}." for i in range(start_idx, total_layers)}

    wrapped_names: list[str] = []
    target_set = set(config.target_modules)

    for full_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or isinstance(module, LoRALinear):
            continue
        short_name = full_name.split(".")[-1]
        if short_name not in target_set:
            continue
        if allowed_layer_prefixes is not None:
            if not any(full_name.startswith(pref) for pref in allowed_layer_prefixes):
                continue

        parent_path = full_name.rsplit(".", 1)[0] if "." in full_name else ""
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(
            parent,
            short_name,
            LoRALinear(
                module,
                rank=config.rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
            ),
        )
        wrapped_names.append(full_name)

    if not wrapped_names:
        raise ValueError(
            f"No nn.Linear modules matched target_modules={config.target_modules} in model"
        )
    return wrapped_names


def extract_lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Return a state_dict containing only the trainable `lora_a` and `lora_b` tensors."""
    out: dict[str, Tensor] = {}
    for name, param in model.named_parameters():
        if "lora_a." in name or "lora_b." in name:
            out[name] = param.detach().cpu()
    return out


def load_lora_state_dict(model: nn.Module, lora_state: dict[str, Tensor]) -> None:
    """Load saved `lora_a` and `lora_b` tensors into an already-injected `model`."""
    param_map = dict(model.named_parameters())
    missing = [k for k in lora_state if k not in param_map]
    if missing:
        raise KeyError(f"Missing LoRA parameters in model: {missing[:5]}")
    with torch.no_grad():
        for k, v in lora_state.items():
            param_map[k].copy_(v.to(device=param_map[k].device, dtype=param_map[k].dtype))


def save_lora_and_callosum_checkpoint(
    checkpoint_dir: str | Path,
    *,
    model: nn.Module,
    projector: SubliminalGapProjector,
    lora_config: LoRAConfig,
    sleep_cycle_index: int = 1,
    trained_trajectories_count: int = 0,
    notes: str = "",
    base_model_id: str | None = None,
    callosum_status: str | None = None,
) -> CorpusCallosumMetadata:
    """Save both `SubliminalGapProjector` and `LoRALinear` adapter weights to `checkpoint_dir`."""
    out_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_base_model = str(base_model_id or lora_config.base_model_id or "Qwen/Qwen3-4B")
    meta = save_corpus_callosum_checkpoint(
        out_dir,
        projector,
        sleep_cycle_index=sleep_cycle_index,
        trained_trajectories_count=trained_trajectories_count,
        notes=notes,
        base_model_id=resolved_base_model,
        callosum_status=callosum_status,
    )
    atomic_json(out_dir / "lora_config.json", lora_config.to_dict())
    torch.save(extract_lora_state_dict(model), out_dir / "lora_adapters.pt")
    return meta


def load_lora_and_callosum_checkpoint(
    checkpoint_dir: str | Path,
    *,
    model: nn.Module,
    device: str | torch.device = "cpu",
) -> tuple[SubliminalGapProjector, CorpusCallosumMetadata, LoRAConfig]:
    """Load `SubliminalGapProjector`, `LoRAConfig`, and `lora_adapters.pt` from `checkpoint_dir`."""
    in_dir = Path(checkpoint_dir)
    projector, meta = load_corpus_callosum_checkpoint(in_dir, device=device)
    model_hidden_size = int(model.get_input_embeddings().weight.shape[1])
    if model_hidden_size != projector.hidden_size:
        raise ValueError(
            f"Model embedding hidden_size ({model_hidden_size}) does not match "
            f"Corpus Callosum projector.hidden_size ({projector.hidden_size}) in {in_dir}"
        )
    raw_cfg = json.loads((in_dir / "lora_config.json").read_text(encoding="utf-8"))
    lora_cfg = LoRAConfig(
        rank=int(raw_cfg["rank"]),
        lora_alpha=float(raw_cfg["lora_alpha"]),
        lora_dropout=float(raw_cfg["lora_dropout"]),
        target_modules=tuple(str(x) for x in raw_cfg["target_modules"]),
        num_top_layers=(
            int(raw_cfg["num_top_layers"]) if raw_cfg.get("num_top_layers") is not None else None
        ),
        base_model_id=str(raw_cfg.get("base_model_id", meta.base_model_id)),
    )
    has_lora = any(isinstance(m, LoRALinear) for m in model.modules())
    if not has_lora:
        inject_lora_adapters(model, lora_cfg)
    lora_state = torch.load(in_dir / "lora_adapters.pt", map_location=device, weights_only=True)
    load_lora_state_dict(model, lora_state)
    return projector, meta, lora_cfg


def ensure_subliminal_gap_token(tokenizer) -> int:
    """Ensure `<|subliminal_gap|>` is registered in `tokenizer` and return its integer token ID."""
    vocab = tokenizer.get_vocab()
    if SUBLIMINAL_GAP_TOKEN not in vocab:
        tokenizer.add_special_tokens({"additional_special_tokens": [SUBLIMINAL_GAP_TOKEN]})
    gap_id = int(tokenizer.convert_tokens_to_ids(SUBLIMINAL_GAP_TOKEN))
    if gap_id < 0:
        raise ValueError(f"Failed to resolve token ID for {SUBLIMINAL_GAP_TOKEN}")
    return gap_id


def tokenize_pre_gap_and_continuation(
    tokenizer,
    pair: ContrastiveGapPair,
    *,
    continuation_text: str,
    gap_token_id: int,
    device: str | torch.device,
    lead_with_gap: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Tokenize `[pre_gap_prompt]` and `[continuation_text]` separately so pre-gap IDs never vary by branch.

    Returns `(input_ids, attention_mask, post_gap_only_labels)` each of shape `[1, seq_len]`.
    """
    pre_gap_str = pair.format_pre_gap_prompt(lead_with_gap=lead_with_gap)
    pre_ids: list[int] = list(tokenizer.encode(pre_gap_str, add_special_tokens=True))
    if not pre_ids or pre_ids[-1] != gap_token_id:
        if gap_token_id not in pre_ids:
            pre_ids.append(gap_token_id)

    cont_ids: list[int] = list(tokenizer.encode(" " + continuation_text.strip(), add_special_tokens=False))
    if not cont_ids:
        raise ValueError(f"Continuation tokenized to empty sequence: {continuation_text!r}")
    if getattr(tokenizer, "eos_token_id", None) is not None:
        cont_ids.append(int(tokenizer.eos_token_id))

    full_ids = torch.tensor([pre_ids + cont_ids], dtype=torch.long, device=device)
    attn_mask = torch.ones_like(full_ids, dtype=torch.long, device=device)
    post_gap_labels = build_post_gap_only_labels(full_ids, gap_token_id=gap_token_id)
    return full_ids, attn_mask, post_gap_labels


def compute_post_gap_continuation_nll(
    model: nn.Module,
    tokenizer,
    projector: SubliminalGapProjector,
    pair: ContrastiveGapPair,
    *,
    chord: SubliminalChordVector,
    continuation_text: str,
    gap_token_id: int,
    device: str | torch.device,
    lead_with_gap: bool = False,
) -> Tensor:
    """Compute the mean per-token negative log-likelihood on `continuation_text` given `chord` at `<|subliminal_gap|>`."""
    input_ids, attn_mask, post_gap_labels = tokenize_pre_gap_and_continuation(
        tokenizer,
        pair,
        continuation_text=continuation_text,
        gap_token_id=gap_token_id,
        device=device,
        lead_with_gap=lead_with_gap,
    )
    word_embed_fn = model.get_input_embeddings()
    chord_tensor = chord.to_tensor(device=device, dtype=torch.float32).unsqueeze(0)
    spliced = splice_subliminal_gaps_into_embeds(
        input_ids,
        word_embed_fn,
        projector,
        chord_tensor,
        gap_token_id=gap_token_id,
        attention_mask=attn_mask,
        labels=post_gap_labels,
    )
    out = model(
        inputs_embeds=spliced.inputs_embeds,
        attention_mask=spliced.attention_mask,
        labels=spliced.labels,
        use_cache=False,
    )
    if out.loss is not None:
        return out.loss

    # Fallback if a custom test model returns logits without computing `.loss`
    logits = out.logits[:, :-1, :].contiguous()
    shift_labels = spliced.labels[:, 1:].contiguous()
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_LABEL_INDEX,
    )


def compute_no_gap_continuation_nll(
    model: nn.Module,
    tokenizer,
    pair: ContrastiveGapPair,
    *,
    continuation_text: str,
    device: str | torch.device,
) -> Tensor:
    """Compute the pure text-only NLL on `continuation_text` right after `clause_1_prefix` with `<|subliminal_gap|>` omitted."""
    pre_str = f"User: {pair.user_prompt.strip()}\nAssistant: {pair.clause_1_prefix.strip()}"
    pre_ids: list[int] = list(tokenizer.encode(pre_str, add_special_tokens=True))
    cont_ids: list[int] = list(tokenizer.encode(" " + continuation_text.strip(), add_special_tokens=False))
    if not cont_ids:
        raise ValueError(f"Continuation tokenized to empty sequence: {continuation_text!r}")
    if getattr(tokenizer, "eos_token_id", None) is not None:
        cont_ids.append(int(tokenizer.eos_token_id))

    full_ids = torch.tensor([pre_ids + cont_ids], dtype=torch.long, device=device)
    attn_mask = torch.ones_like(full_ids, dtype=torch.long, device=device)
    labels = full_ids.clone()
    labels[:, : len(pre_ids)] = IGNORE_LABEL_INDEX

    word_embed_fn = model.get_input_embeddings()
    embeds = word_embed_fn(full_ids)
    out = model(
        inputs_embeds=embeds,
        attention_mask=attn_mask,
        labels=labels,
        use_cache=False,
    )
    if out.loss is not None:
        return out.loss

    logits = out.logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_LABEL_INDEX,
    )


def compute_contrastive_pair_loss(
    model: nn.Module,
    tokenizer,
    projector: SubliminalGapProjector,
    pair: ContrastiveGapPair,
    *,
    gap_token_id: int,
    device: str | torch.device,
    mle_weight: float = 0.25,
    contrastive_weight: float = 2.0,
    margin_target: float = 0.35,
    swap_margin_target: float = 1.20,
) -> tuple[Tensor, dict[str, float]]:
    """Compute matched post-gap MLE loss + chord-contrastive routing margin loss on one `ContrastiveGapPair`.

    Why both terms work together:
    - `mle_weight * mle_loss` (`0.25 * mle_loss`) lightly anchors natural English post-gap fluency
      without overwriting Qwen3's base vocabulary or memorizing rare words.
    - `contrast_loss` focuses primarily on `chord_swap_margin_a = (nll_ba - nll_aa)` and
      `chord_swap_margin_b = (nll_ab - nll_bb)` (`swap_margin_target = 1.20`), which compare the exact
      same continuation under `chord_b` vs. `chord_a` so sentence-length and word-rarity bias cancel
      out completely, alongside `branch_margin_target = 0.35` on `(nll_ab - nll_aa)` and `(nll_ba - nll_bb)`.
    """
    nll_aa = compute_post_gap_continuation_nll(
        model,
        tokenizer,
        projector,
        pair,
        chord=pair.chord_a,
        continuation_text=pair.continuation_a,
        gap_token_id=gap_token_id,
        device=device,
    )
    nll_bb = compute_post_gap_continuation_nll(
        model,
        tokenizer,
        projector,
        pair,
        chord=pair.chord_b,
        continuation_text=pair.continuation_b,
        gap_token_id=gap_token_id,
        device=device,
    )
    nll_ab = compute_post_gap_continuation_nll(
        model,
        tokenizer,
        projector,
        pair,
        chord=pair.chord_a,
        continuation_text=pair.continuation_b,
        gap_token_id=gap_token_id,
        device=device,
    )
    nll_ba = compute_post_gap_continuation_nll(
        model,
        tokenizer,
        projector,
        pair,
        chord=pair.chord_b,
        continuation_text=pair.continuation_a,
        gap_token_id=gap_token_id,
        device=device,
    )

    mle_loss = 0.5 * (nll_aa + nll_bb)

    # Under chord_a, nll_aa should be lower than nll_ab AND lower than nll_ba
    # Under chord_b, nll_bb should be lower than nll_ba AND lower than nll_ab
    branch_margin_a = (nll_ab - nll_aa)
    branch_margin_b = (nll_ba - nll_bb)
    chord_swap_margin_a = (nll_ba - nll_aa)
    chord_swap_margin_b = (nll_ab - nll_bb)

    contrast_loss = (
        0.35
        * (
            F.relu(float(swap_margin_target) - chord_swap_margin_a)
            + F.relu(float(swap_margin_target) - chord_swap_margin_b)
        )
        + 0.15
        * (
            F.relu(float(margin_target) - branch_margin_a)
            + F.relu(float(margin_target) - branch_margin_b)
        )
    )
    total_loss = float(mle_weight) * mle_loss + float(contrastive_weight) * contrast_loss

    metrics = {
        "total_loss": float(total_loss.detach().item()),
        "mle_loss": float(mle_loss.detach().item()),
        "contrast_loss": float(contrast_loss.detach().item()),
        "margin_a": float(branch_margin_a.detach().item()),
        "margin_b": float(branch_margin_b.detach().item()),
        "swap_margin_a": float(chord_swap_margin_a.detach().item()),
        "swap_margin_b": float(chord_swap_margin_b.detach().item()),
    }
    return total_loss, metrics


@torch.no_grad()
def generate_post_gap_continuation_greedy(
    model: nn.Module,
    tokenizer,
    projector: SubliminalGapProjector,
    pair: ContrastiveGapPair,
    *,
    chord: SubliminalChordVector,
    gap_token_id: int,
    device: str | torch.device,
    max_new_tokens: int = 32,
) -> str:
    """Greedily generate the post-gap Clause 2 continuation from `[pre_gap_prompt]` + `chord`."""
    was_training = model.training
    model.eval()
    projector.eval()
    try:
        pre_gap_str = pair.format_pre_gap_prompt()
        pre_ids: list[int] = list(tokenizer.encode(pre_gap_str, add_special_tokens=True))
        if not pre_ids or pre_ids[-1] != gap_token_id:
            if gap_token_id not in pre_ids:
                pre_ids.append(gap_token_id)

        input_ids = torch.tensor([pre_ids], dtype=torch.long, device=device)
        attn_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        word_embed_fn = model.get_input_embeddings()
        chord_tensor = chord.to_tensor(device=device, dtype=torch.float32).unsqueeze(0)

        spliced = splice_subliminal_gaps_into_embeds(
            input_ids,
            word_embed_fn,
            projector,
            chord_tensor,
            gap_token_id=gap_token_id,
            attention_mask=attn_mask,
            labels=None,
        )
        cur_embeds = spliced.inputs_embeds
        cur_mask = spliced.attention_mask

        generated_ids: list[int] = []
        eos_id = getattr(tokenizer, "eos_token_id", None)
        past_kv = None
        step_embeds = cur_embeds

        for _ in range(max_new_tokens):
            try:
                out = model(
                    inputs_embeds=step_embeds,
                    attention_mask=cur_mask,
                    past_key_values=past_kv,
                    use_cache=True,
                )
                past_kv = getattr(out, "past_key_values", None)
            except TypeError:
                out = model(
                    inputs_embeds=cur_embeds,
                    attention_mask=cur_mask,
                    use_cache=False,
                )
                past_kv = None
            next_logits = out.logits[0, -1, :].clone()
            # Never emit another raw gap token during Clause 2 completion
            if 0 <= gap_token_id < next_logits.shape[-1]:
                next_logits[gap_token_id] = -1e9
            next_id = int(torch.argmax(next_logits).item())
            if eos_id is not None and next_id == int(eos_id):
                break
            generated_ids.append(next_id)
            decoded_so_far = tokenizer.decode(generated_ids, skip_special_tokens=True)
            if "\n" in decoded_so_far:
                break

            next_id_tensor = torch.tensor([[next_id]], dtype=torch.long, device=device)
            next_embed = word_embed_fn(next_id_tensor)
            cur_embeds = torch.cat([cur_embeds, next_embed], dim=1)
            step_embeds = next_embed if past_kv is not None else cur_embeds
            cur_mask = torch.cat(
                [cur_mask, torch.ones((1, 1), dtype=cur_mask.dtype, device=device)],
                dim=1,
            )

        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return text.splitlines()[0].strip() if text else ""
    finally:
        if was_training:
            model.train()
            projector.train()


@dataclass(frozen=True)
class HoldoutPairEvalResult:
    """Detailed Check A, Check B, and Check C results for a single unseen holdout pair."""

    pair_id: str
    domain: str
    posture_label_a: str
    posture_label_b: str
    nll_a_under_chord_a: float
    nll_b_under_chord_a: float
    nll_a_under_chord_b: float
    nll_b_under_chord_b: float
    margin_under_chord_a: float
    margin_under_chord_b: float
    net_chord_swap_margin: float
    both_branches_flipped: bool
    positive_chord_swap_shift: bool
    unplugged_chord_swap_shift: float
    generated_continuation_a: str
    generated_continuation_b: str
    greedy_text_changed: bool
    generated_sensor_l2: float
    zero_chord_prior_bias: float = 0.0
    zero_calibrated_margin_under_chord_a: float = 0.0
    zero_calibrated_margin_under_chord_b: float = 0.0
    zero_calibrated_both_flipped: bool = False
    no_gap_prior_bias: float = 0.0
    no_gap_calibrated_margin_under_chord_a: float = 0.0
    no_gap_calibrated_margin_under_chord_b: float = 0.0
    no_gap_calibrated_both_flipped: bool = False
    generated_continuation_zero: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "domain": self.domain,
            "posture_label_a": self.posture_label_a,
            "posture_label_b": self.posture_label_b,
            "nll_a_under_chord_a": self.nll_a_under_chord_a,
            "nll_b_under_chord_a": self.nll_b_under_chord_a,
            "nll_a_under_chord_b": self.nll_a_under_chord_b,
            "nll_b_under_chord_b": self.nll_b_under_chord_b,
            "margin_under_chord_a": self.margin_under_chord_a,
            "margin_under_chord_b": self.margin_under_chord_b,
            "net_chord_swap_margin": self.net_chord_swap_margin,
            "both_branches_flipped": self.both_branches_flipped,
            "zero_chord_prior_bias": self.zero_chord_prior_bias,
            "zero_calibrated_margin_under_chord_a": self.zero_calibrated_margin_under_chord_a,
            "zero_calibrated_margin_under_chord_b": self.zero_calibrated_margin_under_chord_b,
            "zero_calibrated_both_flipped": self.zero_calibrated_both_flipped,
            "no_gap_prior_bias": self.no_gap_prior_bias,
            "no_gap_calibrated_margin_under_chord_a": self.no_gap_calibrated_margin_under_chord_a,
            "no_gap_calibrated_margin_under_chord_b": self.no_gap_calibrated_margin_under_chord_b,
            "no_gap_calibrated_both_flipped": self.no_gap_calibrated_both_flipped,
            "positive_chord_swap_shift": self.positive_chord_swap_shift,
            "unplugged_chord_swap_shift": self.unplugged_chord_swap_shift,
            "generated_continuation_a": self.generated_continuation_a,
            "generated_continuation_b": self.generated_continuation_b,
            "generated_continuation_zero": self.generated_continuation_zero,
            "greedy_text_changed": self.greedy_text_changed,
            "generated_sensor_l2": self.generated_sensor_l2,
        }


@dataclass(frozen=True)
class HoldoutEvaluationSummary:
    """Complete un-rigged holdout evaluation summary across Check A, Check B, and Check C."""

    total_holdout_pairs: int
    both_branches_flipped_count: int
    positive_chord_swap_shift_count: int
    mean_margin_under_chord_a: float
    mean_margin_under_chord_b: float
    mean_net_chord_swap_margin: float
    max_unplugged_chord_swap_shift: float
    greedy_text_changed_count: int
    mean_generated_sensor_l2: float
    pair_results: tuple[HoldoutPairEvalResult, ...]
    zero_calibrated_both_flipped_count: int = 0
    no_gap_calibrated_both_flipped_count: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "total_holdout_pairs": self.total_holdout_pairs,
            "both_branches_flipped_count": self.both_branches_flipped_count,
            "zero_calibrated_both_flipped_count": self.zero_calibrated_both_flipped_count,
            "no_gap_calibrated_both_flipped_count": self.no_gap_calibrated_both_flipped_count,
            "positive_chord_swap_shift_count": self.positive_chord_swap_shift_count,
            "mean_margin_under_chord_a": self.mean_margin_under_chord_a,
            "mean_margin_under_chord_b": self.mean_margin_under_chord_b,
            "mean_net_chord_swap_margin": self.mean_net_chord_swap_margin,
            "max_unplugged_chord_swap_shift": self.max_unplugged_chord_swap_shift,
            "greedy_text_changed_count": self.greedy_text_changed_count,
            "mean_generated_sensor_l2": self.mean_generated_sensor_l2,
            "pair_results": [p.to_dict() for p in self.pair_results],
        }


@torch.no_grad()
def evaluate_holdout_pairs(
    model: nn.Module,
    tokenizer,
    projector: SubliminalGapProjector,
    holdout_pairs: Sequence[ContrastiveGapPair],
    *,
    gap_token_id: int,
    device: str | torch.device,
    sensors: ThreeBridgeSensors | None = None,
    rules: Sequence[FuzzyConceptRule] = (),
    learned_leaves: tuple[LearnedLeaf, ...] = DEFAULT_COMPLEX_FEELING_LEAVES,
    birth_state: BirthState | None = None,
    max_new_tokens: int = 32,
) -> HoldoutEvaluationSummary:
    """Run Check A (Log-Prob Flip), Check B (Unplug-the-Cord), and Check C (Greedy Gen + MiniLM Sensors)."""
    was_training = model.training
    model.eval()
    projector.eval()
    resolved_birth = birth_state or create_birth_state(
        "ind-holdout-eval", birth_event_id="birth-holdout-eval-1", random_seed=42
    )

    results: list[HoldoutPairEvalResult] = []
    try:
        for pair in holdout_pairs:
            # --- Check A: Log-Probability Flip on Unseen Holdout ---
            nll_aa = float(
                compute_post_gap_continuation_nll(
                    model,
                    tokenizer,
                    projector,
                    pair,
                    chord=pair.chord_a,
                    continuation_text=pair.continuation_a,
                    gap_token_id=gap_token_id,
                    device=device,
                ).item()
            )
            nll_ab = float(
                compute_post_gap_continuation_nll(
                    model,
                    tokenizer,
                    projector,
                    pair,
                    chord=pair.chord_a,
                    continuation_text=pair.continuation_b,
                    gap_token_id=gap_token_id,
                    device=device,
                ).item()
            )
            nll_ba = float(
                compute_post_gap_continuation_nll(
                    model,
                    tokenizer,
                    projector,
                    pair,
                    chord=pair.chord_b,
                    continuation_text=pair.continuation_a,
                    gap_token_id=gap_token_id,
                    device=device,
                ).item()
            )
            nll_bb = float(
                compute_post_gap_continuation_nll(
                    model,
                    tokenizer,
                    projector,
                    pair,
                    chord=pair.chord_b,
                    continuation_text=pair.continuation_b,
                    gap_token_id=gap_token_id,
                    device=device,
                ).item()
            )

            # Positive margin_under_chord_a means log P(cont_a | chord_a) > log P(cont_b | chord_a)
            margin_a = round(nll_ab - nll_aa, 6)
            # Positive margin_under_chord_b means log P(cont_b | chord_b) > log P(cont_a | chord_b)
            margin_b = round(nll_ba - nll_bb, 6)
            # Net chord-swap shift: how much swapping chord_a -> chord_b shifts preference from A to B
            net_swap = round((nll_ba - nll_aa) + (nll_ab - nll_bb), 6)

            # --- Check B: Unplug-the-Cord Check (all-zero 52-float chord) ---
            zero_a = pair.chord_a.with_zeroed_cord()
            zero_b = pair.chord_b.with_zeroed_cord()
            nll_zero_aa = float(
                compute_post_gap_continuation_nll(
                    model,
                    tokenizer,
                    projector,
                    pair,
                    chord=zero_a,
                    continuation_text=pair.continuation_a,
                    gap_token_id=gap_token_id,
                    device=device,
                ).item()
            )
            nll_zero_ba = float(
                compute_post_gap_continuation_nll(
                    model,
                    tokenizer,
                    projector,
                    pair,
                    chord=zero_b,
                    continuation_text=pair.continuation_a,
                    gap_token_id=gap_token_id,
                    device=device,
                ).item()
            )
            nll_zero_ab = float(
                compute_post_gap_continuation_nll(
                    model,
                    tokenizer,
                    projector,
                    pair,
                    chord=zero_a,
                    continuation_text=pair.continuation_b,
                    gap_token_id=gap_token_id,
                    device=device,
                ).item()
            )
            nll_zero_bb = float(
                compute_post_gap_continuation_nll(
                    model,
                    tokenizer,
                    projector,
                    pair,
                    chord=zero_b,
                    continuation_text=pair.continuation_b,
                    gap_token_id=gap_token_id,
                    device=device,
                ).item()
            )
            unplugged_shift = round(
                abs(nll_zero_ba - nll_zero_aa) + abs(nll_zero_ab - nll_zero_bb), 6
            )

            # Unplugged zero-chord baseline bias and text-only no-gap LLM prior bias
            zero_prior_bias = round(nll_zero_ab - nll_zero_aa, 6)
            zero_cal_ma = round(margin_a - zero_prior_bias, 6)
            zero_cal_mb = round(margin_b + zero_prior_bias, 6)

            nll_nogap_a = float(
                compute_no_gap_continuation_nll(
                    model,
                    tokenizer,
                    pair,
                    continuation_text=pair.continuation_a,
                    device=device,
                ).item()
            )
            nll_nogap_b = float(
                compute_no_gap_continuation_nll(
                    model,
                    tokenizer,
                    pair,
                    continuation_text=pair.continuation_b,
                    device=device,
                ).item()
            )
            nogap_prior_bias = round(nll_nogap_b - nll_nogap_a, 6)
            nogap_cal_ma = round(margin_a - nogap_prior_bias, 6)
            nogap_cal_mb = round(margin_b + nogap_prior_bias, 6)

            # --- Check C: Greedy Post-Gap Generation + 3-Bridge MiniLM Sensor Verification ---
            gen_a = generate_post_gap_continuation_greedy(
                model,
                tokenizer,
                projector,
                pair,
                chord=pair.chord_a,
                gap_token_id=gap_token_id,
                device=device,
                max_new_tokens=max_new_tokens,
            )
            gen_b = generate_post_gap_continuation_greedy(
                model,
                tokenizer,
                projector,
                pair,
                chord=pair.chord_b,
                gap_token_id=gap_token_id,
                device=device,
                max_new_tokens=max_new_tokens,
            )
            gen_zero = generate_post_gap_continuation_greedy(
                model,
                tokenizer,
                projector,
                pair,
                chord=zero_a,
                gap_token_id=gap_token_id,
                device=device,
                max_new_tokens=max_new_tokens,
            )
            text_changed = gen_a.strip().lower() != gen_b.strip().lower()

            if sensors is not None and gen_a.strip() and gen_b.strip():
                gen_chord_a = score_continuation_to_chord(
                    gen_a,
                    birth_state=resolved_birth,
                    sensors=sensors,
                    rules=rules,
                    learned_leaves=learned_leaves,
                    event_id=f"eval-{pair.pair_id}-a",
                )
                gen_chord_b = score_continuation_to_chord(
                    gen_b,
                    birth_state=resolved_birth,
                    sensors=sensors,
                    rules=rules,
                    learned_leaves=learned_leaves,
                    event_id=f"eval-{pair.pair_id}-b",
                )
                gen_l2 = chord_l2_distance(gen_chord_a, gen_chord_b)
            else:
                gen_l2 = 0.0

            results.append(
                HoldoutPairEvalResult(
                    pair_id=pair.pair_id,
                    domain=pair.domain,
                    posture_label_a=pair.posture_label_a,
                    posture_label_b=pair.posture_label_b,
                    nll_a_under_chord_a=round(nll_aa, 6),
                    nll_b_under_chord_a=round(nll_ab, 6),
                    nll_a_under_chord_b=round(nll_ba, 6),
                    nll_b_under_chord_b=round(nll_bb, 6),
                    margin_under_chord_a=margin_a,
                    margin_under_chord_b=margin_b,
                    net_chord_swap_margin=net_swap,
                    both_branches_flipped=(margin_a > 0.0 and margin_b > 0.0),
                    positive_chord_swap_shift=(net_swap > 0.0),
                    unplugged_chord_swap_shift=unplugged_shift,
                    generated_continuation_a=gen_a,
                    generated_continuation_b=gen_b,
                    generated_continuation_zero=gen_zero,
                    greedy_text_changed=text_changed,
                    generated_sensor_l2=round(gen_l2, 6),
                    zero_chord_prior_bias=zero_prior_bias,
                    zero_calibrated_margin_under_chord_a=zero_cal_ma,
                    zero_calibrated_margin_under_chord_b=zero_cal_mb,
                    zero_calibrated_both_flipped=(zero_cal_ma > 0.0 and zero_cal_mb > 0.0),
                    no_gap_prior_bias=nogap_prior_bias,
                    no_gap_calibrated_margin_under_chord_a=nogap_cal_ma,
                    no_gap_calibrated_margin_under_chord_b=nogap_cal_mb,
                    no_gap_calibrated_both_flipped=(nogap_cal_ma > 0.0 and nogap_cal_mb > 0.0),
                )
            )
    finally:
        if was_training:
            model.train()
            projector.train()

    n_total = len(results)
    both_flipped = sum(1 for r in results if r.both_branches_flipped)
    pos_swap = sum(1 for r in results if r.positive_chord_swap_shift)
    mean_ma = round(sum(r.margin_under_chord_a for r in results) / n_total, 6) if n_total else 0.0
    mean_mb = round(sum(r.margin_under_chord_b for r in results) / n_total, 6) if n_total else 0.0
    mean_swap = round(sum(r.net_chord_swap_margin for r in results) / n_total, 6) if n_total else 0.0
    max_unplug = round(max((r.unplugged_chord_swap_shift for r in results), default=0.0), 6)
    changed_cnt = sum(1 for r in results if r.greedy_text_changed)
    mean_gen_l2 = round(sum(r.generated_sensor_l2 for r in results) / n_total, 6) if n_total else 0.0
    zero_cal_flipped = sum(1 for r in results if r.zero_calibrated_both_flipped)
    nogap_cal_flipped = sum(1 for r in results if r.no_gap_calibrated_both_flipped)

    return HoldoutEvaluationSummary(
        total_holdout_pairs=n_total,
        both_branches_flipped_count=both_flipped,
        positive_chord_swap_shift_count=pos_swap,
        mean_margin_under_chord_a=mean_ma,
        mean_margin_under_chord_b=mean_mb,
        mean_net_chord_swap_margin=mean_swap,
        max_unplugged_chord_swap_shift=max_unplug,
        greedy_text_changed_count=changed_cnt,
        mean_generated_sensor_l2=mean_gen_l2,
        pair_results=tuple(results),
        zero_calibrated_both_flipped_count=zero_cal_flipped,
        no_gap_calibrated_both_flipped_count=nogap_cal_flipped,
    )


def train_left_brain_lora_and_callosum(
    model: nn.Module,
    tokenizer,
    projector: SubliminalGapProjector,
    *,
    train_pairs: Sequence[ContrastiveGapPair],
    gap_token_id: int,
    device: str | torch.device,
    epochs: int = 8,
    lr_projector: float = 8e-4,
    lr_lora: float = 3e-4,
    weight_decay: float = 0.01,
    mle_weight: float = 0.25,
    contrastive_weight: float = 2.0,
    margin_target: float = 0.35,
    swap_margin_target: float = 1.20,
    warmup_steps: int = 0,
    anchor_at_birth: bool = False,
) -> list[dict[str, float]]:
    """Train `SubliminalGapProjector` and `LoRALinear` adapters on `train_pairs` with `batch_size=1`."""
    if bool(anchor_at_birth) and not getattr(projector, "semantically_anchored_at_birth", False):
        projector.anchor_to_llm_embeddings(model, tokenizer)
    model.train()
    projector.train()

    lora_params = [p for p in model.parameters() if p.requires_grad]
    proj_params = list(projector.parameters())
    if not lora_params:
        raise ValueError("Model has no trainable LoRA parameters; call inject_lora_adapters(model) first")

    base_lrs = (float(lr_projector), float(lr_lora))
    optimizer = torch.optim.AdamW(
        [
            {"params": proj_params, "lr": base_lrs[0]},
            {"params": lora_params, "lr": base_lrs[1]},
        ],
        weight_decay=float(weight_decay),
    )

    global_step = 0
    w_steps = max(0, int(warmup_steps))
    epoch_history: list[dict[str, float]] = []
    for epoch_idx in range(1, int(epochs) + 1):
        t0 = time.perf_counter()
        sum_total = 0.0
        sum_mle = 0.0
        sum_contrast = 0.0
        sum_ma = 0.0
        sum_mb = 0.0

        for pair in train_pairs:
            if w_steps > 0:
                scale = min(1.0, float(global_step + 1) / float(w_steps))
                optimizer.param_groups[0]["lr"] = base_lrs[0] * scale
                optimizer.param_groups[1]["lr"] = base_lrs[1] * scale
            optimizer.zero_grad(set_to_none=True)
            if getattr(model, "is_loaded_in_4bit", False):
                with torch.no_grad():
                    v_aa = float(
                        compute_post_gap_continuation_nll(
                            model,
                            tokenizer,
                            projector,
                            pair,
                            chord=pair.chord_a,
                            continuation_text=pair.continuation_a,
                            gap_token_id=gap_token_id,
                            device=device,
                        ).detach().item()
                    )
                    v_bb = float(
                        compute_post_gap_continuation_nll(
                            model,
                            tokenizer,
                            projector,
                            pair,
                            chord=pair.chord_b,
                            continuation_text=pair.continuation_b,
                            gap_token_id=gap_token_id,
                            device=device,
                        ).detach().item()
                    )
                    v_ab = float(
                        compute_post_gap_continuation_nll(
                            model,
                            tokenizer,
                            projector,
                            pair,
                            chord=pair.chord_a,
                            continuation_text=pair.continuation_b,
                            gap_token_id=gap_token_id,
                            device=device,
                        ).detach().item()
                    )
                    v_ba = float(
                        compute_post_gap_continuation_nll(
                            model,
                            tokenizer,
                            projector,
                            pair,
                            chord=pair.chord_b,
                            continuation_text=pair.continuation_a,
                            gap_token_id=gap_token_id,
                            device=device,
                        ).detach().item()
                    )
                mle_val = 0.5 * (v_aa + v_bb)
                br_ma = v_ab - v_aa
                br_mb = v_ba - v_bb
                sw_ma = v_ba - v_aa
                sw_mb = v_ab - v_bb
                contrast_val = (
                    0.35 * (max(0.0, float(swap_margin_target) - sw_ma) + max(0.0, float(swap_margin_target) - sw_mb))
                    + 0.15 * (max(0.0, float(margin_target) - br_ma) + max(0.0, float(margin_target) - br_mb))
                )
                tot_val = float(mle_weight) * mle_val + float(contrastive_weight) * contrast_val
                metrics = {
                    "total_loss": tot_val,
                    "mle_loss": mle_val,
                    "contrast_loss": contrast_val,
                    "margin_a": br_ma,
                    "margin_b": br_mb,
                    "swap_margin_a": sw_ma,
                    "swap_margin_b": sw_mb,
                }
                t_aa = torch.tensor(v_aa, dtype=torch.float32, requires_grad=True)
                t_bb = torch.tensor(v_bb, dtype=torch.float32, requires_grad=True)
                t_ab = torch.tensor(v_ab, dtype=torch.float32, requires_grad=True)
                t_ba = torch.tensor(v_ba, dtype=torch.float32, requires_grad=True)
                mle_s = 0.5 * (t_aa + t_bb)
                contrast_s = (
                    0.35
                    * (
                        F.relu(float(swap_margin_target) - (t_ba - t_aa))
                        + F.relu(float(swap_margin_target) - (t_ab - t_bb))
                    )
                    + 0.15
                    * (
                        F.relu(float(margin_target) - (t_ab - t_aa))
                        + F.relu(float(margin_target) - (t_ba - t_bb))
                    )
                )
                tot_s = float(mle_weight) * mle_s + float(contrastive_weight) * contrast_s
                g_aa, g_bb, g_ab, g_ba = torch.autograd.grad(tot_s, (t_aa, t_bb, t_ab, t_ba))
                w_aa = float(g_aa.item())
                w_bb = float(g_bb.item())
                w_ab = float(g_ab.item())
                w_ba = float(g_ba.item())
                for br_chord, br_text, br_w in (
                    (pair.chord_a, pair.continuation_a, w_aa),
                    (pair.chord_b, pair.continuation_b, w_bb),
                    (pair.chord_a, pair.continuation_b, w_ab),
                    (pair.chord_b, pair.continuation_a, w_ba),
                ):
                    if abs(br_w) > 1e-12:
                        step_nll = compute_post_gap_continuation_nll(
                            model,
                            tokenizer,
                            projector,
                            pair,
                            chord=br_chord,
                            continuation_text=br_text,
                            gap_token_id=gap_token_id,
                            device=device,
                        )
                        (step_nll * float(br_w)).backward()
                        del step_nll
            else:
                loss, metrics = compute_contrastive_pair_loss(
                    model,
                    tokenizer,
                    projector,
                    pair,
                    gap_token_id=gap_token_id,
                    device=device,
                    mle_weight=mle_weight,
                    contrastive_weight=contrastive_weight,
                    margin_target=margin_target,
                    swap_margin_target=swap_margin_target,
                )
                loss.backward()
            torch.nn.utils.clip_grad_norm_(proj_params + lora_params, max_norm=1.0)
            optimizer.step()
            global_step += 1

            sum_total += metrics["total_loss"]
            sum_mle += metrics["mle_loss"]
            sum_contrast += metrics["contrast_loss"]
            sum_ma += metrics["margin_a"]
            sum_mb += metrics["margin_b"]

        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()

        n = max(1, len(train_pairs))
        epoch_record = {
            "epoch": epoch_idx,
            "mean_total_loss": round(sum_total / n, 6),
            "mean_mle_loss": round(sum_mle / n, 6),
            "mean_contrast_loss": round(sum_contrast / n, 6),
            "mean_train_margin_a": round(sum_ma / n, 6),
            "mean_train_margin_b": round(sum_mb / n, 6),
            "wall_time_s": round(time.perf_counter() - t0, 2),
        }
        if torch.device(device).type == "cuda":
            print(
                f"  [Epoch {epoch_idx}/{epochs}] loss={epoch_record['mean_total_loss']:.4f}, "
                f"ma={epoch_record['mean_train_margin_a']:+.4f}, mb={epoch_record['mean_train_margin_b']:+.4f}, "
                f"wall={epoch_record['wall_time_s']:.1f}s, "
                f"vram_res={torch.cuda.memory_reserved() / 1e9:.2f}GB",
                flush=True,
            )
        epoch_history.append(epoch_record)

    return epoch_history


def _build_contrastive_probe_chords(
    pairs: Sequence[ContrastiveGapPair],
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Build a `[N, 52]` contrastive batch `(chords_a, chords_b)` combining harvested pairs and canonical orthogonal probes."""
    rows_a: list[Tensor] = []
    rows_b: list[Tensor] = []
    for p in list(pairs[-8:]):
        rows_a.append(p.chord_a.to_tensor(device=device, dtype=torch.float32))
        rows_b.append(p.chord_b.to_tensor(device=device, dtype=torch.float32))

    # Add 4 canonical orthogonal Right-Brain probes (Do-er, Thinker, Feeler, Cadence)
    canonical_pairs = (
        (
            [0.85 if i in (1, 3, 4) else (0.10 if i == 5 else 0.50) for i in range(52)],
            [0.20 if i in (1, 3, 4) else (0.85 if i == 5 else 0.50) for i in range(52)],
        ),
        (
            [0.88 if i in (17, 18, 19, 21) else (0.15 if i == 20 else 0.50) for i in range(52)],
            [0.25 if i in (17, 18, 19, 21) else (0.88 if i == 20 else 0.50) for i in range(52)],
        ),
        (
            [0.85 if i in (34, 35, 37, 39, 40) else 0.45 for i in range(52)],
            [0.20 if i in (34, 35, 37, 39, 40) else 0.65 for i in range(52)],
        ),
        (
            [0.90 if i == 49 else (0.10 if i in (50, 51) else 0.55) for i in range(52)],
            [0.20 if i == 49 else (0.88 if i in (50, 51) else 0.45) for i in range(52)],
        ),
    )
    for flat_a, flat_b in canonical_pairs:
        rows_a.append(torch.tensor(flat_a, device=device, dtype=torch.float32))
        rows_b.append(torch.tensor(flat_b, device=device, dtype=torch.float32))
    return torch.stack(rows_a, dim=0), torch.stack(rows_b, dim=0)


def sculpt_lora_rank_channels(
    model: nn.Module,
    projector: SubliminalGapProjector,
    pairs: Sequence[ContrastiveGapPair] = (),
    *,
    top_k_vo: int = 6,
    qk_gain: float = 0.85,
    vo_gain: float = 0.65,
    homeostatic_decay: float = 0.02,
    state_dir: str | Path | None = None,
) -> dict[str, Any]:
    """MIT Contrastive Rank Prober: keep `q_proj / k_proj` attention routing active while pruning static/memorized rank channels in `v_proj / o_proj`."""
    proj_params = list(projector.parameters())
    device = proj_params[0].device if proj_params else torch.device("cpu")
    with torch.no_grad():
        chords_a, chords_b = _build_contrastive_probe_chords(pairs, device=device)
        proj_dtype = proj_params[0].dtype if proj_params else torch.float32
        va = projector(chords_a.to(dtype=proj_dtype)).to(dtype=torch.float32)  # [N, 4, H]
        vb = projector(chords_b.to(dtype=proj_dtype)).to(dtype=torch.float32)  # [N, 4, H]
        delta_v = (va - vb).reshape(-1, va.shape[-1])  # [M, H]
        mean_v = (0.5 * (va + vb)).reshape(-1, va.shape[-1])  # [M, H]
        hidden_dim = int(delta_v.shape[-1])

        qk_modules = 0
        vo_modules = 0
        pruned_channels_total = 0
        active_channels_total = 0
        saved_scales: dict[str, Tensor] = {}
        decay = max(0.0, min(0.25, float(homeostatic_decay)))

        for mod_name, mod in model.named_modules():
            if not isinstance(mod, LoRALinear):
                continue
            short_name = mod_name.split(".")[-1]
            r_dim = int(mod.rank)
            a_w = mod.lora_a.weight.detach().to(dtype=torch.float32)  # [rank, in_features]
            b_w = mod.lora_b.weight.detach().to(dtype=torch.float32)  # [out_features, rank]
            in_dim = int(a_w.shape[1])

            d_in = delta_v.to(device=a_w.device)
            m_in = mean_v.to(device=a_w.device)
            if in_dim < hidden_dim:
                d_in = d_in[:, :in_dim]
                m_in = m_in[:, :in_dim]
            elif in_dim > hidden_dim:
                repeats = (in_dim + hidden_dim - 1) // hidden_dim
                d_in = d_in.repeat(1, repeats)[:, :in_dim]
                m_in = m_in.repeat(1, repeats)[:, :in_dim]

            delta_proj = torch.matmul(d_in, a_w.t())  # [M, rank]
            mean_proj = torch.matmul(m_in, a_w.t())  # [M, rank]
            chord_mod = torch.norm(delta_proj, dim=0)  # [rank]
            a_norm = torch.norm(a_w, dim=1) + 1e-6
            b_norm = torch.norm(b_w, dim=0) + 1e-6
            static_bias = torch.norm(mean_proj, dim=0) + 0.25 * b_norm * a_norm
            selectivity = chord_mod / (static_bias + 1e-6)  # [rank]

            if short_name in ("q_proj", "k_proj"):
                qk_modules += 1
                sel_max = float(torch.max(selectivity).item())
                sel_min = float(torch.min(selectivity).item())
                if sel_max > sel_min + 1e-6:
                    norm_sel = (selectivity - sel_min) / (sel_max - sel_min)
                else:
                    norm_sel = torch.ones_like(selectivity)
                target_scale = float(qk_gain) * (0.60 + 0.40 * norm_sel)
                active_channels_total += r_dim
            else:
                vo_modules += 1
                k_keep = min(r_dim, max(1, int(top_k_vo)))
                _, top_idx = torch.topk(selectivity, k=k_keep, largest=True)
                target_scale = torch.zeros_like(selectivity)
                top_vals = selectivity[top_idx]
                t_max = float(torch.max(top_vals).item())
                t_min = float(torch.min(top_vals).item())
                if t_max > t_min + 1e-6:
                    norm_top = (top_vals - t_min) / (t_max - t_min)
                else:
                    norm_top = torch.ones_like(top_vals)
                target_scale[top_idx] = float(vo_gain) * (0.45 + 0.55 * norm_top)
                active_channels_total += k_keep
                pruned_channels_total += max(0, r_dim - k_keep)

            baseline_ones = torch.ones_like(target_scale)
            sculpted = (1.0 - decay) * target_scale + decay * baseline_ones
            if short_name not in ("q_proj", "k_proj") and r_dim > int(top_k_vo):
                # Keep pruned memorization wires strictly zeroed
                zero_mask = target_scale <= 1e-6
                sculpted[zero_mask] = 0.0
            mod.rank_scale.copy_(sculpted.to(device=mod.rank_scale.device, dtype=mod.rank_scale.dtype))
            saved_scales[mod_name] = mod.rank_scale.detach().cpu().clone()

    if state_dir is not None and saved_scales:
        out_dir = Path(state_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(saved_scales, out_dir / "live_lora_rank_scales.pt")

    return {
        "qk_modules": qk_modules,
        "vo_modules": vo_modules,
        "qk_modules_sculpted": qk_modules,
        "vo_modules_sculpted": vo_modules,
        "active_channels": active_channels_total,
        "pruned_channels": pruned_channels_total,
        "top_k_vo": int(top_k_vo),
    }


def update_projector_on_harvested_gap_pairs(
    projector: SubliminalGapProjector,
    pairs: Sequence[ContrastiveGapPair],
    *,
    state_dir: str | Path,
    model: nn.Module | None = None,
    tokenizer=None,
    lr_projector: float = 2e-4,
    lambda_anchor: float = 0.05,
    homeostatic_decay: float = 0.02,
    max_grad_norm: float = 0.5,
    max_pairs: int = 4,
) -> dict[str, Any]:
    """Run a bounded, L2-anchored micro-plasticity step on `SubliminalGapProjector` during Deep REM Sleep.

    Guardrails:
    - Anchors strictly to the true resting checkpoint `theta_0` (`projector.get_resting_anchor_params()`)
      with a `2%` MIT homeostatic spring-back (`homeostatic_decay = 0.02`).
    - Adds `lambda_anchor * ||theta - theta_0||^2` so live REM sleep never drifts from the resting Callosum.
    - Clips gradients at `max_grad_norm = 0.5`.
    - Saves updated weights strictly to `state_dir / "live_projector_weights.pt"` (never touching `LMFT-Training/models/`).
    """
    selected = list(pairs[-max(1, int(max_pairs)) :]) if pairs else []
    out_dir = Path(state_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "live_projector_weights.pt"
    if not selected:
        return {
            "updated": False,
            "pairs_used": 0,
            "steps": 0,
            "initial_loss": 0.0,
            "final_loss": 0.0,
            "l2_drift_from_resting_theta0": 0.0,
            "saved_path": str(save_path),
        }

    all_params = list(projector.parameters())
    for p in all_params:
        if not p.requires_grad:
            p.requires_grad_(True)
    proj_params = all_params

    device = proj_params[0].device
    anchor_params = projector.get_resting_anchor_params()
    decay = max(0.0, min(0.25, float(homeostatic_decay)))
    if decay > 0.0:
        with torch.no_grad():
            for p, p0 in zip(proj_params, anchor_params):
                p.lerp_(p0, decay)

    was_proj_training = projector.training
    projector.train()

    optimizer = torch.optim.AdamW(proj_params, lr=float(lr_projector), weight_decay=0.0)
    losses: list[float] = []
    gap_token_id = (
        ensure_subliminal_gap_token(tokenizer)
        if (model is not None and tokenizer is not None)
        else -1
    )

    try:
        for pair in selected:
            optimizer.zero_grad()
            if model is not None and tokenizer is not None and gap_token_id >= 0:
                task_loss, _ = compute_contrastive_pair_loss(
                    model,
                    tokenizer,
                    projector,
                    pair,
                    gap_token_id=gap_token_id,
                    device=device,
                )
            else:
                # Fast projector-only separation margin in virtual gap-token space when LLM forward pass is omitted
                ca = pair.chord_a.to_tensor(device=device, dtype=torch.float32).unsqueeze(0)
                cb = pair.chord_b.to_tensor(device=device, dtype=torch.float32).unsqueeze(0)
                va = F.normalize(projector(ca).reshape(1, -1), dim=-1)
                vb = F.normalize(projector(cb).reshape(1, -1), dim=-1)
                cos_sim = torch.sum(va * vb, dim=-1).mean()
                target_sep = min(0.85, max(0.25, float(pair.chord_distance) * 0.45))
                task_loss = F.relu(cos_sim - (1.0 - target_sep))

            anchor_reg = sum(
                torch.sum((p - p0) ** 2) for p, p0 in zip(proj_params, anchor_params)
            )
            total_loss = task_loss + float(lambda_anchor) * anchor_reg
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(proj_params, max_norm=float(max_grad_norm))
            optimizer.step()
            losses.append(float(total_loss.detach().cpu().item()))
    finally:
        projector.train(was_proj_training)

    with torch.no_grad():
        drift_sq = sum(
            torch.sum((p.detach() - p0) ** 2) for p, p0 in zip(proj_params, anchor_params)
        )
        l2_drift = round(float(torch.sqrt(drift_sq).cpu().item()), 6)

    lora_sculpt_meta: dict[str, Any] | None = None
    if model is not None:
        lora_sculpt_meta = sculpt_lora_rank_channels(
            model,
            projector,
            selected,
            homeostatic_decay=decay,
            state_dir=out_dir,
        )

    torch.save(projector.state_dict(), save_path)
    meta_payload: dict[str, Any] = {
        "updated": True,
        "pairs_used": len(selected),
        "steps": len(losses),
        "lambda_anchor": float(lambda_anchor),
        "homeostatic_decay": decay,
        "max_grad_norm": float(max_grad_norm),
        "initial_loss": round(losses[0], 6) if losses else 0.0,
        "final_loss": round(losses[-1], 6) if losses else 0.0,
        "l2_drift_from_resting_theta0": l2_drift,
        "saved_path": str(save_path),
    }
    if lora_sculpt_meta is not None:
        meta_payload["lora_rank_sculpting"] = lora_sculpt_meta
    atomic_json(out_dir / "live_projector_meta.json", meta_payload)
    return meta_payload

