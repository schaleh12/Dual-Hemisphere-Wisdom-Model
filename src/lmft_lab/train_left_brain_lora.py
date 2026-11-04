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

        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base_linear(x)
        x_fp32 = self.dropout(x.to(dtype=torch.float32))
        lora_delta = self.lora_b(self.lora_a(x_fp32)) * self.scaling
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
) -> tuple[Tensor, Tensor, Tensor]:
    """Tokenize `[pre_gap_prompt]` and `[continuation_text]` separately so pre-gap IDs never vary by branch.

    Returns `(input_ids, attention_mask, post_gap_only_labels)` each of shape `[1, seq_len]`.
    """
    pre_gap_str = pair.format_pre_gap_prompt()
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
) -> Tensor:
    """Compute the mean per-token negative log-likelihood on `continuation_text` given `chord` at `<|subliminal_gap|>`."""
    input_ids, attn_mask, post_gap_labels = tokenize_pre_gap_and_continuation(
        tokenizer,
        pair,
        continuation_text=continuation_text,
        gap_token_id=gap_token_id,
        device=device,
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
    contrastive_weight: float = 1.5,
    margin_target: float = 0.35,
) -> tuple[Tensor, dict[str, float]]:
    """Compute matched post-gap MLE loss + chord-contrastive routing margin loss on one `ContrastiveGapPair`.

    Why both terms work together:
    - `mle_loss` teaches the Left Brain to generate smooth post-gap continuations (`cont_a` under `chord_a`,
      `cont_b` under `chord_b`) with `-100` masking on all pre-gap tokens.
    - `contrast_loss` compares `(chord_a -> cont_a)` vs. `(chord_b -> cont_a)` and `(chord_b -> cont_b)` vs.
      `(chord_a -> cont_b)`. Since all tokens except the 4 virtual `<|subliminal_gap|>` embeddings are 100%
      identical in those comparisons, 100% of `contrast_loss` must flow through `SubliminalGapProjector`
      and the LoRA attention weights attending to the 4 virtual gap tokens.
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

    contrast_loss = 0.25 * (
        F.relu(margin_target - branch_margin_a)
        + F.relu(margin_target - branch_margin_b)
        + F.relu(margin_target - chord_swap_margin_a)
        + F.relu(margin_target - chord_swap_margin_b)
    )
    total_loss = mle_loss + float(contrastive_weight) * contrast_loss

    metrics = {
        "total_loss": float(total_loss.detach().item()),
        "mle_loss": float(mle_loss.detach().item()),
        "contrast_loss": float(contrast_loss.detach().item()),
        "margin_a": float(branch_margin_a.detach().item()),
        "margin_b": float(branch_margin_b.detach().item()),
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

        for _ in range(max_new_tokens):
            out = model(
                inputs_embeds=cur_embeds,
                attention_mask=cur_mask,
                use_cache=False,
            )
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
    contrastive_weight: float = 1.5,
    margin_target: float = 0.35,
    warmup_steps: int = 0,
) -> list[dict[str, float]]:
    """Train `SubliminalGapProjector` and `LoRALinear` adapters on `train_pairs` with `batch_size=1`."""
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
            optimizer.zero_grad()
            loss, metrics = compute_contrastive_pair_loss(
                model,
                tokenizer,
                projector,
                pair,
                gap_token_id=gap_token_id,
                device=device,
                contrastive_weight=contrastive_weight,
                margin_target=margin_target,
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
        epoch_history.append(epoch_record)

    return epoch_history
