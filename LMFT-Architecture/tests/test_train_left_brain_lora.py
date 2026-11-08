"""Unit tests for Sub-Step 3 (`src/lmft_architecture/train_left_brain_lora.py`)."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from lmft_architecture.gap_trajectory_schema import (
    ContrastiveGapPair,
    compile_chess_posture_chords,
)
from lmft_architecture.subliminal_gap_splice import (
    IGNORE_LABEL_INDEX,
    SUBLIMINAL_GAP_TOKEN,
    SubliminalGapProjector,
    splice_subliminal_gaps_into_embeds,
)
from lmft_architecture.train_left_brain_lora import (
    LoRAConfig,
    LoRALinear,
    ensure_subliminal_gap_token,
    evaluate_holdout_pairs,
    inject_lora_adapters,
    load_lora_and_callosum_checkpoint,
    save_lora_and_callosum_checkpoint,
    tokenize_pre_gap_and_continuation,
    train_left_brain_lora_and_callosum,
)


class _WordLevelTestTokenizer:
    """Fast deterministic word-level tokenizer for unit testing Sub-Step 3."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
        self._id_to_tok: dict[int, str] = {0: "<pad>", 1: "<eos>", 2: "<unk>"}
        self.eos_token_id = 1

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def add_special_tokens(self, spec: dict[str, list[str]]) -> int:
        added = 0
        for tok in spec.get("additional_special_tokens", []):
            if tok not in self._vocab:
                idx = len(self._vocab)
                self._vocab[tok] = idx
                self._id_to_tok[idx] = tok
                added += 1
        return added

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._vocab.get(token, -1)

    def _intern(self, word: str) -> int:
        if word not in self._vocab:
            idx = len(self._vocab)
            self._vocab[word] = idx
            self._id_to_tok[idx] = word
        return self._vocab[word]

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = [w for w in text.replace("\n", " ").split(" ") if w]
        return [self._intern(t) for t in tokens]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        words: list[str] = []
        for idx in ids:
            tok = self._id_to_tok.get(int(idx), "<unk>")
            if skip_special_tokens and tok in ("<pad>", "<eos>", SUBLIMINAL_GAP_TOKEN):
                continue
            words.append(tok)
        return " ".join(words)


@dataclass
class _CausalLMOutput:
    loss: Tensor | None
    logits: Tensor


class _MiniDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seq_len, h = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (h ** 0.5)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask.unsqueeze(0), -1e9)
        attn = torch.softmax(scores, dim=-1)
        ctx = self.o_proj(torch.matmul(attn, v))
        return self.norm(x + ctx)


class _MiniCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 128, hidden_size: int = 32, num_layers: int = 2) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([_MiniDecoderLayer(hidden_size) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed

    def forward(
        self,
        inputs_embeds: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        use_cache: bool = False,
    ) -> _CausalLMOutput:
        h = inputs_embeds
        for layer in self.layers:
            h = layer(h)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=IGNORE_LABEL_INDEX,
            )
        return _CausalLMOutput(loss=loss, logits=logits)


def _sample_pair(pair_id: str = "test-chess-01", split: str = "train") -> ContrastiveGapPair:
    chord_a, chord_b = compile_chess_posture_chords(
        queen_volition=0.85,
        king_safety=0.65,
        bishop_alignment=0.80,
        knight_disruption=0.88,
        pawn_chain_flow=0.72,
        rook_bastion=0.78,
    )
    return ContrastiveGapPair(
        pair_id=pair_id,
        domain="chess_intuition",
        split=split,
        user_prompt="In a sharp Sicilian middlegame, our Rook eyes c1 while our Knight sits on d5.",
        clause_1_prefix="Evaluating the Sicilian board carefully,",
        chord_a=chord_a,
        posture_label_a="CHESS_STOIC_POSITIONAL_FORTRESS",
        continuation_a="we calmly play Kb1 and Rc1 to shelter the King.",
        chord_b=chord_b,
        posture_label_b="CHESS_TACTICAL_KNIGHT_STRIKE",
        continuation_b="we sacrifice with Nxf6+ and Qxd6 to storm the fortress.",
    )


def test_lora_linear_zero_init_and_freeze_base_weights() -> None:
    """Verify LoRALinear leaves base output identical at step 0 and freezes base parameters."""
    torch.manual_seed(42)
    model = _MiniCausalLM(vocab_size=64, hidden_size=32, num_layers=2)
    x = torch.randn(1, 5, 32)
    base_out = model(inputs_embeds=x).logits.detach().clone()

    wrapped = inject_lora_adapters(model, LoRAConfig(rank=4, lora_alpha=8.0, lora_dropout=0.0, num_top_layers=1))
    assert len(wrapped) == 4  # q_proj, k_proj, v_proj, o_proj in top 1 layer

    lora_out = model(inputs_embeds=x).logits.detach().clone()
    assert torch.allclose(base_out, lora_out, atol=1e-6)

    for name, param in model.named_parameters():
        if "lora_" in name:
            assert param.requires_grad is True
        else:
            assert param.requires_grad is False


def test_pre_gap_label_masking_masks_all_pre_gap_and_virtual_tokens() -> None:
    """Verify all tokens before <|subliminal_gap|> and all 4 virtual tokens have label = -100."""
    tok = _WordLevelTestTokenizer()
    gap_id = ensure_subliminal_gap_token(tok)
    pair = _sample_pair()

    ids_a, mask_a, labels_a = tokenize_pre_gap_and_continuation(
        tok, pair, continuation_text=pair.continuation_a, gap_token_id=gap_id, device="cpu"
    )
    ids_b, mask_b, labels_b = tokenize_pre_gap_and_continuation(
        tok, pair, continuation_text=pair.continuation_b, gap_token_id=gap_id, device="cpu"
    )

    gap_idx_a = int((ids_a[0] == gap_id).nonzero(as_tuple=False)[0].item())
    gap_idx_b = int((ids_b[0] == gap_id).nonzero(as_tuple=False)[0].item())
    assert gap_idx_a == gap_idx_b
    # Pre-gap token IDs must be 100% identical between Branch A and Branch B
    assert torch.equal(ids_a[0, : gap_idx_a + 1], ids_b[0, : gap_idx_b + 1])
    assert torch.all(labels_a[0, : gap_idx_a + 1] == IGNORE_LABEL_INDEX)

    model = _MiniCausalLM(vocab_size=128, hidden_size=32, num_layers=2)
    projector = SubliminalGapProjector(hidden_size=32)
    spliced = splice_subliminal_gaps_into_embeds(
        ids_a,
        model.get_input_embeddings(),
        projector,
        pair.chord_a.to_tensor().unsqueeze(0),
        gap_token_id=gap_id,
        attention_mask=mask_a,
        labels=labels_a,
    )
    # Pre-gap tokens + 4 virtual gap tokens must all be masked with -100
    assert torch.all(spliced.labels[0, : gap_idx_a + 4] == IGNORE_LABEL_INDEX)
    assert torch.all(spliced.labels[0, gap_idx_a + 4 :] != IGNORE_LABEL_INDEX)


def test_train_and_holdout_eval_flips_branches_and_unplugs_cleanly(tmp_path: Path) -> None:
    """Verify training flips log-prob preference between Chord A and Chord B and unplugs to 0.0."""
    torch.manual_seed(7)
    tok = _WordLevelTestTokenizer()
    gap_id = ensure_subliminal_gap_token(tok)
    train_pair = _sample_pair("train-chess-01", split="train")
    holdout_pair = _sample_pair("holdout-chess-01", split="holdout")

    # Pre-populate vocabulary so embedding table covers all tokens
    for p in (train_pair, holdout_pair):
        tok.encode(p.format_full_text("A"))
        tok.encode(p.format_full_text("B"))

    model = _MiniCausalLM(vocab_size=len(tok.get_vocab()) + 8, hidden_size=32, num_layers=2)
    lora_cfg = LoRAConfig(rank=8, lora_alpha=16.0, lora_dropout=0.0, num_top_layers=2)
    inject_lora_adapters(model, lora_cfg)
    projector = SubliminalGapProjector(hidden_size=32)

    history = train_left_brain_lora_and_callosum(
        model,
        tok,
        projector,
        train_pairs=(train_pair,),
        gap_token_id=gap_id,
        device="cpu",
        epochs=18,
        lr_projector=3e-3,
        lr_lora=2e-3,
    )
    assert history[-1]["mean_total_loss"] < history[0]["mean_total_loss"]

    summary = evaluate_holdout_pairs(
        model,
        tok,
        projector,
        (holdout_pair,),
        gap_token_id=gap_id,
        device="cpu",
        max_new_tokens=10,
    )
    res = summary.pair_results[0]
    assert res.both_branches_flipped is True
    assert res.margin_under_chord_a > 0.0
    assert res.margin_under_chord_b > 0.0
    # Unplug-the-Cord Check: zeroed 52-float chord produces 0.0 difference between A and B chord slots
    assert res.unplugged_chord_swap_shift == 0.0
    assert res.greedy_text_changed is True
    # Prior-calibrated fields must be populated and consistent with net_chord_swap_margin
    assert abs(
        (res.zero_calibrated_margin_under_chord_a + res.zero_calibrated_margin_under_chord_b)
        - res.net_chord_swap_margin
    ) < 1e-4
    assert abs(
        (res.no_gap_calibrated_margin_under_chord_a + res.no_gap_calibrated_margin_under_chord_b)
        - res.net_chord_swap_margin
    ) < 1e-4
    assert isinstance(summary.zero_calibrated_both_flipped_count, int)
    assert isinstance(summary.no_gap_calibrated_both_flipped_count, int)

    # Verify checkpoint save & reload round-trip (sleep_cycle_index = 1)
    ckpt_dir = tmp_path / "left_brain_lora"
    meta = save_lora_and_callosum_checkpoint(
        ckpt_dir,
        model=model,
        projector=projector,
        lora_config=lora_cfg,
        sleep_cycle_index=1,
        trained_trajectories_count=1,
        notes="Unit test round-trip",
    )
    assert meta.sleep_cycle_index == 1

    fresh_model = _MiniCausalLM(vocab_size=len(tok.get_vocab()) + 8, hidden_size=32, num_layers=2)
    # Copy base frozen weights so we test loading only the LoRA adapters + projector
    fresh_model.load_state_dict(
        {k: v for k, v in model.state_dict().items() if "lora_" not in k and "base_linear" not in k},
        strict=False,
    )
    # Better yet: clone base model before LoRA injection or reload into a fresh injected copy
    loaded_proj, loaded_meta, loaded_cfg = load_lora_and_callosum_checkpoint(
        ckpt_dir, model=model, device="cpu"
    )
    assert loaded_meta.sleep_cycle_index == 1
    assert loaded_cfg.rank == 8
    assert torch.allclose(
        projector(train_pair.chord_a.to_tensor()),
        loaded_proj(train_pair.chord_a.to_tensor()),
        atol=1e-6,
    )


def test_saved_stage3_gpu_report_and_checkpoint_artifacts_exist() -> None:
    """Verify the Stage 3 GPU evaluation report and checkpoint files exist and satisfy schema invariants."""
    import json
    from lmft_architecture.workspace_paths import resolve_training_root

    repo_root = resolve_training_root()
    report_path = repo_root / "outputs" / "left_brain_lora_eval_report.json"
    ckpt_dir = repo_root / "models" / "left_brain_lora"

    assert report_path.exists(), f"Missing {report_path}"
    assert (ckpt_dir / "corpus_callosum_meta.json").exists()
    assert (ckpt_dir / "projector_weights.pt").exists()
    assert (ckpt_dir / "lora_config.json").exists()
    assert (ckpt_dir / "lora_adapters.pt").exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["hf_model_id"] == "Qwen/Qwen3-4B"
    assert report["corpus_callosum_metadata"]["sleep_cycle_index"] == 1
    assert report["train_pairs_count"] == 72
    assert report["holdout_pairs_count"] == 6

    post = report["post_training_holdout_evaluation"]
    assert post["max_unplugged_chord_swap_shift"] == 0.0
    assert post["both_branches_flipped_count"] == 5
    assert post["zero_calibrated_both_flipped_count"] == 6
    assert post["no_gap_calibrated_both_flipped_count"] == 6
    assert post["positive_chord_swap_shift_count"] == 6
    assert len(post["pair_results"]) == 6



def test_mit_lora_rank_channel_sculpting_and_true_theta0_callosum_homeostasis(tmp_path: Path) -> None:
    """Verify MIT rank-channel sculpting in `LoRALinear` and true `theta_0` homeostatic anchoring in `update_projector_on_harvested_gap_pairs`."""
    from lmft_architecture.train_left_brain_lora import (
        sculpt_lora_rank_channels,
        update_projector_on_harvested_gap_pairs,
    )

    torch.manual_seed(42)
    model = _MiniCausalLM(vocab_size=32, hidden_size=32, num_layers=2)
    lora_cfg = LoRAConfig(rank=8, lora_alpha=16.0, lora_dropout=0.0, num_top_layers=2)
    inject_lora_adapters(model, lora_cfg)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                torch.nn.init.normal_(m.lora_b.weight, std=0.05)

    projector = SubliminalGapProjector(hidden_size=32)
    projector.capture_resting_anchor()
    resting_snapshot = projector.get_resting_anchor_params()
    assert len(resting_snapshot) > 0

    pair = _sample_pair("mit_sculpt_pair")

    # 1. Verify `LoRALinear.rank_scale` gates rank channels in `LoRALinear.forward`
    sample_x = torch.randn(1, 4, 32)
    v_mod = model.layers[-1].v_proj
    assert isinstance(v_mod, LoRALinear)
    out_full = v_mod(sample_x)
    with torch.no_grad():
        v_mod.rank_scale.zero_()
    out_zeroed = v_mod(sample_x)
    base_only = v_mod.base_linear(sample_x)
    assert torch.allclose(out_zeroed, base_only, atol=1e-6)
    assert not torch.allclose(out_full, out_zeroed, atol=1e-5)
    with torch.no_grad():
        v_mod.rank_scale.fill_(1.0)

    # 2. Verify `sculpt_lora_rank_channels` keeps all q_proj/k_proj wires active and prunes v_proj/o_proj to top_k_vo
    sculpt_meta = sculpt_lora_rank_channels(
        model,
        projector,
        (pair,),
        top_k_vo=3,
        qk_gain=0.85,
        vo_gain=0.65,
        state_dir=tmp_path,
    )
    assert sculpt_meta["qk_modules_sculpted"] == 4
    assert sculpt_meta["vo_modules_sculpted"] == 4
    assert (tmp_path / "live_lora_rank_scales.pt").exists()

    q_mod = model.layers[-1].q_proj
    assert isinstance(q_mod, LoRALinear)
    assert int((q_mod.rank_scale > 0.0).sum().item()) == 8
    assert int((v_mod.rank_scale > 0.0).sum().item()) == 3
    assert int((v_mod.rank_scale == 0.0).sum().item()) == 5

    # 3. Verify `update_projector_on_harvested_gap_pairs` anchors to true resting `theta_0` across consecutive cycles
    meta_c1 = update_projector_on_harvested_gap_pairs(
        projector,
        (pair,),
        lr_projector=1e-3,
        state_dir=tmp_path,
        model=model,
    )
    assert meta_c1["updated"] is True
    assert "l2_drift_from_resting_theta0" in meta_c1
    assert meta_c1["l2_drift_from_resting_theta0"] > 0.0
    assert "lora_rank_sculpting" in meta_c1

    # Even if we perturb projector weights far from theta_0, homeostatic spring-back + theta_0 anchor pull it back toward theta_0
    with torch.no_grad():
        for p in projector.parameters():
            p.add_(0.50)
        dist_before = sum(
            float((p.detach().cpu() - p0.detach().cpu()).pow(2).sum().item())
            for p, p0 in zip(projector.parameters(), resting_snapshot)
        ) ** 0.5

    meta_c2 = update_projector_on_harvested_gap_pairs(
        projector,
        (pair,),
        lr_projector=1e-3,
        lambda_anchor=0.50,
        homeostatic_decay=0.10,
        state_dir=tmp_path,
    )
    assert meta_c2["l2_drift_from_resting_theta0"] < dist_before
