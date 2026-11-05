"""Tests for the Impish Bloodmoon Left-Brain Adapter & Virgin Baseline Clone (`tests/test_impish_bloodmoon_adapter.py`).

Verifies all 4 Auditor guardrails:
1. Tokenizer- & dimension-agnostic `SubliminalGapLeftBrain` (`<think>` only for tokenizers with `<think>` in vocab, `hidden_size` mismatch check, and `.gguf` rejection).
2. Isolated Virgin Baseline clone (`outputs/impish_bloodmoon_state/`) without contaminating `baselines/virgin_dual_hemisphere_v1/` or `outputs/club_user_state/`.
3. Honest `UNSTEERED_BOOTSTRAP_CYCLE_0` vs. `TRAINED_STEERED_CALLOSUM` status + Holdout Checks A, B, and C.
4. Desktop App `switch_brain_lineage("impish_bloodmoon")` selector and Patreon passphrase encryption/decryption.
"""

from __future__ import annotations

from pathlib import Path
import pytest
import torch

from lmft_architecture.gap_trajectory_schema import compile_chess_posture_chords
from lmft_architecture.impish_bloodmoon_adapter import (
    DEFAULT_IMPISH_BLOODMOON_HIDDEN_SIZE,
    DEFAULT_IMPISH_BLOODMOON_HF_ID,
    TRAINED_STEERED_STATUS,
    UNSTEERED_BOOTSTRAP_STATUS,
    bootstrap_impish_bloodmoon_virgin_checkpoint,
    clone_virgin_baseline_for_impish_bloodmoon,
    decrypt_patreon_checkpoint_bundle,
    detect_lora_target_modules,
    detect_model_hidden_size,
    encrypt_patreon_checkpoint_bundle,
    train_impish_bloodmoon_callosum_and_lora,
)
from lmft_architecture.sanctuary_club_app import SINGLE_PAGE_HTML, SanctuaryClubRuntime
from lmft_architecture.subliminal_gap_splice import SubliminalGapProjector, load_corpus_callosum_checkpoint
from lmft_architecture.three_legged_runtime import (
    SubliminalGapLeftBrain,
    _tokenizer_has_single_think_token,
    validate_pytorch_hf_model_source,
)
from lmft_architecture.train_left_brain_lora import ensure_subliminal_gap_token
from lmft_architecture.virgin_baseline import verify_virgin_baseline_snapshot
from tests.test_train_left_brain_lora import (
    _MiniCausalLM,
    _sample_pair,
    _WordLevelTestTokenizer,
)


def test_guardrail_1_gguf_rejection_think_token_detection_and_hidden_size_check(tmp_path: Path) -> None:
    """Verify `.gguf` rejection, `<think>` vocab detection, and `hidden_size` mismatch validation."""
    with pytest.raises(ValueError, match="not a quantized .gguf file"):
        validate_pytorch_hf_model_source("C:/models/Impish_Bloodmoon_4B-Q4_K_M.gguf")

    gguf_dir = tmp_path / "gguf_only_dir"
    gguf_dir.mkdir()
    (gguf_dir / "impish_bloodmoon.gguf").write_bytes(b"GGUF")
    with pytest.raises(ValueError, match="contains only .gguf files"):
        validate_pytorch_hf_model_source(gguf_dir)

    assert DEFAULT_IMPISH_BLOODMOON_HF_ID == "SicariusSicariiStuff/Impish_Bloodmoon_12B"
    assert DEFAULT_IMPISH_BLOODMOON_HIDDEN_SIZE == 5120
    assert validate_pytorch_hf_model_source(DEFAULT_IMPISH_BLOODMOON_HF_ID) == DEFAULT_IMPISH_BLOODMOON_HF_ID

    # Llama/Impish Bloodmoon tokenizer without `<think>` vs. Qwen3 tokenizer with `<think>`
    llama_tok = _WordLevelTestTokenizer()
    assert _tokenizer_has_single_think_token(llama_tok) is False

    qwen_tok = _WordLevelTestTokenizer()
    qwen_tok.add_special_tokens({"additional_special_tokens": ["<think>", "</think>"]})
    assert _tokenizer_has_single_think_token(qwen_tok) is True

    # Dimension mismatch between model (hidden_size=48) and projector (hidden_size=32) raises ValueError
    gap_id = ensure_subliminal_gap_token(llama_tok)
    model_48 = _MiniCausalLM(vocab_size=64, hidden_size=48, num_layers=2)
    proj_32 = SubliminalGapProjector(hidden_size=32)
    with pytest.raises(ValueError, match="does not match SubliminalGapProjector.hidden_size"):
        SubliminalGapLeftBrain(model_48, llama_tok, proj_32, gap_token_id=gap_id)


def test_guardrail_2_isolated_virgin_clone_never_contaminates_baseline(tmp_path: Path) -> None:
    """Verify `clone_virgin_baseline_for_impish_bloodmoon` creates an isolated clone and leaves `Cycle 0` untouched."""
    impish_state_dir = tmp_path / "impish_bloodmoon_state"
    cloned = clone_virgin_baseline_for_impish_bloodmoon(impish_state_dir)

    assert cloned.snapshot_dir == impish_state_dir
    assert len(cloned.rules) == 2
    assert len(cloned.learned_leaves) == 10
    assert len(cloned.archives.executive.precedents) == 0
    assert (impish_state_dir / "club_ruleset.json").exists()
    assert (impish_state_dir / "impish_bloodmoon_lineage_meta.json").exists()

    # Canonical `baselines/virgin_dual_hemisphere_v1/` remains 100% intact
    verify_virgin_baseline_snapshot()


def test_guardrail_3_bootstrap_vs_trained_callosum_and_holdout_checks(tmp_path: Path) -> None:
    """Verify `UNSTEERED_BOOTSTRAP_CYCLE_0` zero-delta initialization and `TRAINED_STEERED_CALLOSUM` training."""
    torch.manual_seed(19)
    tok = _WordLevelTestTokenizer()
    pair = _sample_pair("impish-train-01", split="train")
    holdout_pair = _sample_pair("impish-holdout-01", split="holdout")
    tok.encode(pair.format_full_text("A"))
    tok.encode(pair.format_full_text("B"))

    # Use non-Qwen hidden_size=48 to verify dimension-agnostic projector + LoRA
    model = _MiniCausalLM(vocab_size=len(tok.get_vocab()) + 16, hidden_size=48, num_layers=2)
    assert detect_model_hidden_size(model) == 48
    assert detect_lora_target_modules(model) == ("q_proj", "k_proj", "v_proj", "o_proj")

    boot_dir = tmp_path / "impish_ckpt_boot"
    boot_meta = bootstrap_impish_bloodmoon_virgin_checkpoint(
        boot_dir,
        model=model,
        hf_model_id=DEFAULT_IMPISH_BLOODMOON_HF_ID,
        intermediate_size=32,
        lora_rank=8,
        num_top_layers=2,
    )
    assert boot_meta.callosum_status == UNSTEERED_BOOTSTRAP_STATUS
    assert boot_meta.sleep_cycle_index == 0
    assert boot_meta.trained_trajectories_count == 0
    assert boot_meta.base_model_id == DEFAULT_IMPISH_BLOODMOON_HF_ID
    assert boot_meta.hidden_size == 48

    # Verify zero-initialized bootstrap projector emits 0.0 virtual tokens before training
    loaded_proj, loaded_meta = load_corpus_callosum_checkpoint(boot_dir)
    assert loaded_meta.callosum_status == UNSTEERED_BOOTSTRAP_STATUS
    virt_tokens = loaded_proj(pair.chord_a.to_tensor().unsqueeze(0))
    assert float(virt_tokens.abs().max().item()) == 0.0

    # Now train the Corpus Callosum + LoRA on `pair` and run Holdout Checks A, B, and C
    train_dir = tmp_path / "impish_ckpt_trained"
    model_train = _MiniCausalLM(vocab_size=len(tok.get_vocab()) + 16, hidden_size=48, num_layers=2)
    report = train_impish_bloodmoon_callosum_and_lora(
        train_dir,
        model=model_train,
        tokenizer=tok,
        hf_model_id=DEFAULT_IMPISH_BLOODMOON_HF_ID,
        train_pairs=(pair,),
        holdout_pairs=(holdout_pair,),
        epochs=12,
        lr_projector=3e-3,
        lr_lora=2e-3,
        lora_rank=8,
        num_top_layers=2,
        intermediate_size=32,
        device="cpu",
    )
    assert report["callosum_status"] == TRAINED_STEERED_STATUS
    assert report["sleep_cycle_index"] == 1
    assert report["hidden_size"] == 48
    assert report["holdout_report"] is not None
    assert report["holdout_report"]["max_unplugged_chord_swap_shift"] == 0.0


def test_guardrail_4_app_lineage_selector_and_patreon_encryption(tmp_path: Path) -> None:
    """Verify `switch_brain_lineage('impish_bloodmoon')` never secretly uses Qwen3-4B and tests Patreon encryption."""
    assert 'value="impish_bloodmoon"' in SINGLE_PAGE_HTML
    assert 'id="patreon-unlock-bar"' in SINGLE_PAGE_HTML

    # Initialize runtime with a Qwen Left Brain (`qwen_lb`) and no Impish checkpoint installed
    qwen_tok = _WordLevelTestTokenizer()
    qwen_gap_id = ensure_subliminal_gap_token(qwen_tok)
    qwen_model = _MiniCausalLM(vocab_size=64, hidden_size=32, num_layers=2)
    qwen_proj = SubliminalGapProjector(hidden_size=32)
    qwen_lb = SubliminalGapLeftBrain(
        qwen_model,
        qwen_tok,
        qwen_proj,
        gap_token_id=qwen_gap_id,
        max_clause_1_tokens=4,
        max_post_gap_tokens=5,
    )

    schooled_dir = tmp_path / "club_schooled_state"
    impish_dir = tmp_path / "impish_bloodmoon_state"
    impish_ckpt_dir = tmp_path / "test_impish_ckpt"
    runtime = SanctuaryClubRuntime(
        state_dir=schooled_dir,
        impish_state_dir=impish_dir,
        impish_checkpoint_dir=impish_ckpt_dir,
        left_brain=qwen_lb,
        lazy_load_gpu=False,
    )
    assert runtime.left_brain is qwen_lb
    schooled_rules_count = len(runtime.rules)

    # Switch to `impish_bloodmoon` -> must unload/detach `qwen_lb` and report IMPISH_CHECKPOINT_NOT_INSTALLED!
    snap_impish = runtime.switch_brain_lineage("impish_bloodmoon")
    assert snap_impish["active_lineage"] == "impish_bloodmoon"
    assert snap_impish["rules_count"] == 2
    assert "Impish Bloodmoon" in snap_impish["active_checkpoint_label"]
    assert runtime.left_brain is None
    assert snap_impish["left_brain_status"].startswith("IMPISH_CHECKPOINT_NOT_INSTALLED")

    # Verify a chat turn in `impish_bloodmoon` mode NEVER calls `qwen_lb`!
    qwen_lb.last_traces.clear()
    snap_chat = runtime.handle_chat_message("Testing Impish lineage isolation.", room="sanctuary")
    assert len(qwen_lb.last_traces) == 0
    chat_turn_id = int(snap_chat["chat_messages"][-1]["id"])
    assert runtime._turn_records_by_id[chat_turn_id]["turn"].left_brain_calls == 0
    assert snap_chat["chat_messages"][-1]["telemetry"]["spliced_trace"] == {}

    # Forge a rule during REM Sleep while in `impish_bloodmoon` mode -> saves ONLY to `impish_dir`!
    from lmft_architecture.dilemma_crucible import CRUCIBLE_STAGES

    stage = CRUCIBLE_STAGES["poetry_story"]
    runtime.handle_crucible_dilemma(
        "poetry_story",
        stage["default_wave"],
        stage["default_structure"],
        harvest_for_sleep=True,
    )
    snap_rem = runtime.handle_run_rem_sleep()
    assert snap_rem["rules_count"] > 2
    assert (impish_dir / "club_ruleset.json").exists()
    assert not (schooled_dir / "club_ruleset.json").exists()
    verify_virgin_baseline_snapshot()

    # Switch back to `schooled` -> restores `qwen_lb` and original schooled rule count
    snap_back = runtime.switch_brain_lineage("schooled")
    assert snap_back["active_lineage"] == "schooled"
    assert snap_back["rules_count"] == schooled_rules_count
    assert runtime.left_brain is qwen_lb
    assert snap_back["left_brain_status"].startswith("READY")

    # Verify Patreon encrypted checkpoint bundle round-trip and wrong-passphrase rejection
    raw_ckpt = tmp_path / "raw_impish_ckpt"
    bootstrap_impish_bloodmoon_virgin_checkpoint(raw_ckpt, hidden_size=64, intermediate_size=32)
    enc_file = tmp_path / "impish_bloodmoon_patreon.enc"
    encrypt_patreon_checkpoint_bundle(raw_ckpt, enc_file, "bloodmoon-secret-2026", pbkdf2_iterations=1000)
    assert enc_file.exists()

    with pytest.raises(ValueError, match="Invalid Patreon unlock key"):
        decrypt_patreon_checkpoint_bundle(enc_file, tmp_path / "bad_out", "wrong-passphrase")

    unlocked_dir = tmp_path / "unlocked_impish_ckpt"
    decrypt_patreon_checkpoint_bundle(enc_file, unlocked_dir, "bloodmoon-secret-2026")
    _, unlocked_meta = load_corpus_callosum_checkpoint(unlocked_dir)
    assert unlocked_meta.hidden_size == 64
    assert unlocked_meta.callosum_status == UNSTEERED_BOOTSTRAP_STATUS

    snap_unlocked = runtime.unlock_patreon_bundle("bloodmoon-secret-2026", enc_path=enc_file)
    assert snap_unlocked["active_lineage"] == "impish_bloodmoon"
    assert (impish_ckpt_dir / "corpus_callosum_meta.json").exists()
    assert snap_unlocked["chat_messages"][-1]["role_label"] == "Patreon Bundle Unlocked"
