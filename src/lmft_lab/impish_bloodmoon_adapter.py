"""Impish Bloodmoon Left-Brain Adapter & Virgin Baseline Clone (`src/lmft_lab/impish_bloodmoon_adapter.py`).

Plain-English Purpose:
Pairs an isolated clone of the `Cycle 0` Virgin Dual-Hemisphere Baseline (`baselines/virgin_dual_hemisphere_v1/`)
with an alternate HuggingFace CausalLM Left Brain (defaulting to `SicariusSicariiStuff/Impish_Bloodmoon_4B`,
`hidden_size = 3072`), implementing all 4 Auditor guardrails:
1. **Tokenizer- & Dimension-Agnostic `<|subliminal_gap|>` Splicing**:
   - Rejects quantized `.gguf` files with a clear error explaining that `<|subliminal_gap|>` embedding
     splicing and LoRA training require HuggingFace PyTorch `safetensors` weights.
   - Auto-detects `hidden_size` (`d_model`) from `model.get_input_embeddings()` and auto-detects
     attention projection module names (`q_proj`, `k_proj`, `v_proj`, `o_proj`, etc.) for `LoRALinear`.
2. **Isolated Virgin Baseline Clone (`outputs/impish_bloodmoon_state/`)**:
   - Clones the verified `Cycle 0` Virgin Baseline into `outputs/impish_bloodmoon_state/` without
     ever modifying `baselines/virgin_dual_hemisphere_v1/` or `outputs/club_user_state/`.
3. **Honest Bootstrap (`UNSTEERED_BOOTSTRAP_CYCLE_0`) vs. Trained Callosum (`TRAINED_STEERED_CALLOSUM`)**:
   - `bootstrap_impish_bloodmoon_virgin_checkpoint` initializes a zero-delta `LoRALinear` + zero-output
     `SubliminalGapProjector` so base prose is uncorrupted prior to training, tagged honestly as
     `UNSTEERED_BOOTSTRAP_CYCLE_0`.
   - `train_impish_bloodmoon_callosum_and_lora` trains `SubliminalGapProjector` + `LoRALinear` on
     `outputs/gap_trajectories_train.jsonl`, saves `TRAINED_STEERED_CALLOSUM` weights, and runs the
     3 holdout falsification checks (`Check A`, `Check B`, `Check C`).
4. **Optional Patreon Passphrase Encryption (`encrypt_patreon_checkpoint_bundle` / `decrypt_patreon_checkpoint_bundle`)**:
   - Encrypts the checkpoint directory with PBKDF2-HMAC-SHA256 + authenticated keystream so the
     bundle bytes cannot be loaded without the passphrase even if someone edits Python `if` checks.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
import hashlib
import hmac
import json
from pathlib import Path
import secrets
from typing import Any

import torch
from torch import nn

from .birth import create_birth_state
from .branch_archives import ThreeBranchArchives
from .contemplation_ledger import load_contemplation_ledger, save_contemplation_ledger
from .gap_trajectory_schema import ContrastiveGapPair, load_contrastive_gap_pairs_jsonl
from .reflection_classifier import load_learned_leaves, load_ruleset, save_ruleset
from .right_brain_equilibrium import ThreeBridgeSensors
from .runtime_store import atomic_json
from .subliminal_gap_splice import (
    CorpusCallosumMetadata,
    SubliminalGapProjector,
    save_corpus_callosum_checkpoint,
)
from .three_legged_runtime import validate_pytorch_hf_model_source
from .train_left_brain_lora import (
    DEFAULT_LORA_TARGET_MODULES,
    LoRAConfig,
    LoRALinear,
    ensure_subliminal_gap_token,
    evaluate_holdout_pairs,
    inject_lora_adapters,
    save_lora_and_callosum_checkpoint,
    train_left_brain_lora_and_callosum,
)
from .virgin_baseline import (
    VIRGIN_BASELINE_VERSION,
    VirginBaselineState,
    load_virgin_baseline_state,
    verify_virgin_baseline_snapshot,
)

DEFAULT_IMPISH_BLOODMOON_HF_ID = "SicariusSicariiStuff/Impish_Bloodmoon_4B"
DEFAULT_IMPISH_BLOODMOON_HIDDEN_SIZE = 3072
IMPISH_BLOODMOON_STATE_REL_DIR = Path("outputs") / "impish_bloodmoon_state"
IMPISH_BLOODMOON_CHECKPOINT_REL_DIR = Path("models") / "left_brain_impish_bloodmoon"

UNSTEERED_BOOTSTRAP_STATUS = "UNSTEERED_BOOTSTRAP_CYCLE_0"
TRAINED_STEERED_STATUS = "TRAINED_STEERED_CALLOSUM"


def resolve_impish_bloodmoon_state_dir(repo_root: str | Path | None = None) -> Path:
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    return root / IMPISH_BLOODMOON_STATE_REL_DIR


def resolve_impish_bloodmoon_checkpoint_dir(repo_root: str | Path | None = None) -> Path:
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    return root / IMPISH_BLOODMOON_CHECKPOINT_REL_DIR


def detect_model_hidden_size(model: nn.Module) -> int:
    """Return the token embedding hidden size (`d_model`) of `model`."""
    embed = model.get_input_embeddings()
    if embed is None or not hasattr(embed, "weight"):
        raise ValueError("Model does not expose get_input_embeddings().weight")
    return int(embed.weight.shape[1])


def detect_lora_target_modules(model: nn.Module) -> tuple[str, ...]:
    """Inspect `nn.Linear` submodules in `model` and return compatible attention projection names."""
    linear_short_names: set[str] = set()
    for full_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            short = full_name.rsplit(".", 1)[-1]
            linear_short_names.add(short)
        elif isinstance(module, LoRALinear):
            short = full_name.rsplit(".", 1)[-1]
            linear_short_names.add(short)

    candidate_sets: tuple[tuple[str, ...], ...] = (
        DEFAULT_LORA_TARGET_MODULES,
        ("qkv_proj", "o_proj"),
        ("c_attn", "c_proj"),
        ("query_key_value", "dense"),
        ("Wqkv", "out_proj"),
    )
    for cand in candidate_sets:
        matched = tuple(name for name in cand if name in linear_short_names)
        if matched:
            return matched

    fallback = tuple(
        sorted(
            name
            for name in linear_short_names
            if name not in ("lm_head", "head", "embed_out", "score")
        )
    )
    if fallback:
        return fallback
    raise ValueError("Could not detect any trainable nn.Linear target modules for LoRA injection.")


def clone_virgin_baseline_for_impish_bloodmoon(
    target_state_dir: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
    hf_model_id: str = DEFAULT_IMPISH_BLOODMOON_HF_ID,
    force_reset: bool = False,
) -> VirginBaselineState:
    """Clone the verified `Cycle 0` Virgin Baseline into an isolated `outputs/impish_bloodmoon_state/` directory."""
    validate_pytorch_hf_model_source(hf_model_id)
    verify_virgin_baseline_snapshot(repo_root=repo_root)
    virgin = load_virgin_baseline_state(
        repo_root=repo_root,
        instance_id="lmft-impish-bloodmoon-virgin-clone",
    )

    state_dir = (
        Path(target_state_dir).resolve()
        if target_state_dir is not None
        else resolve_impish_bloodmoon_state_dir(repo_root)
    )
    state_dir.mkdir(parents=True, exist_ok=True)
    ruleset_path = state_dir / "club_ruleset.json"
    archives_dir = state_dir / "branch_archives"
    ledger_path = state_dir / "contemplation_ledger.json"
    meta_path = state_dir / "impish_bloodmoon_lineage_meta.json"

    if force_reset or not ruleset_path.exists():
        save_ruleset(ruleset_path, virgin.rules, virgin.learned_leaves)
        virgin.archives.save_all(archives_dir)
        save_contemplation_ledger(ledger_path, virgin.ledger)
        atomic_json(
            state_dir / "club_meta.json",
            {"savored_turn_ids": [], "scene_counter": 0},
        )
        atomic_json(
            meta_path,
            {
                "cloned_from_virgin_version": VIRGIN_BASELINE_VERSION,
                "base_model_id": str(hf_model_id),
                "sleep_cycle_index": 0,
                "isolated_state_dir": str(state_dir),
            },
        )

    loaded_rules = load_ruleset(ruleset_path)
    loaded_leaves = load_learned_leaves(ruleset_path)
    loaded_archives = (
        ThreeBranchArchives.load_all(archives_dir)
        if (archives_dir / "philosophical_lexicon.json").exists()
        else ThreeBranchArchives()
    )
    loaded_ledger = (
        load_contemplation_ledger(ledger_path)
        if ledger_path.exists()
        else virgin.ledger
    )
    manifest = (
        json.loads(meta_path.read_text(encoding="utf-8"))
        if meta_path.exists()
        else {"cloned_from_virgin_version": VIRGIN_BASELINE_VERSION, "base_model_id": str(hf_model_id)}
    )
    return VirginBaselineState(
        version=VIRGIN_BASELINE_VERSION,
        snapshot_dir=state_dir,
        birth_state=create_birth_state("lmft-impish-bloodmoon", birth_event_id="e0", random_seed=42),
        rules=loaded_rules,
        learned_leaves=loaded_leaves,
        archives=loaded_archives,
        ledger=loaded_ledger,
        manifest=manifest,
    )


def zero_initialize_projector_output_heads(projector: SubliminalGapProjector) -> None:
    """Zero-initialize the final linear layer of each department head so an untrained bootstrap projector emits 0.0."""
    with torch.no_grad():
        for head in projector.department_heads:
            final_linear = head[-1]
            if isinstance(final_linear, nn.Linear):
                nn.init.zeros_(final_linear.weight)
                if final_linear.bias is not None:
                    nn.init.zeros_(final_linear.bias)


def bootstrap_impish_bloodmoon_virgin_checkpoint(
    checkpoint_dir: str | Path | None = None,
    *,
    model: nn.Module | None = None,
    hf_model_id: str = DEFAULT_IMPISH_BLOODMOON_HF_ID,
    hidden_size: int = DEFAULT_IMPISH_BLOODMOON_HIDDEN_SIZE,
    intermediate_size: int = 256,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.05,
    num_top_layers: int | None = 16,
    repo_root: str | Path | None = None,
) -> CorpusCallosumMetadata:
    """Create a `Cycle 0` zero-delta checkpoint in `models/left_brain_impish_bloodmoon/` tagged `UNSTEERED_BOOTSTRAP_CYCLE_0`."""
    resolved_hf_id = validate_pytorch_hf_model_source(hf_model_id)
    out_dir = (
        Path(checkpoint_dir).resolve()
        if checkpoint_dir is not None
        else resolve_impish_bloodmoon_checkpoint_dir(repo_root)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_hidden = detect_model_hidden_size(model) if model is not None else int(hidden_size)
    projector = SubliminalGapProjector(
        hidden_size=resolved_hidden,
        intermediate_size=int(intermediate_size),
    )
    zero_initialize_projector_output_heads(projector)

    notes = (
        f"Cycle 0 Unsteered Bootstrap for {resolved_hf_id} (hidden_size={resolved_hidden}). "
        "LoRA B-matrices and Projector output heads are zero-initialized so base prose is uncorrupted; "
        "run train_impish_bloodmoon_callosum_and_lora to enable 52-float chord steering."
    )

    if model is not None:
        target_mods = detect_lora_target_modules(model)
        lora_cfg = LoRAConfig(
            rank=int(lora_rank),
            lora_alpha=float(lora_alpha),
            lora_dropout=float(lora_dropout),
            target_modules=target_mods,
            num_top_layers=num_top_layers,
            base_model_id=resolved_hf_id,
        )
        has_lora = any(isinstance(m, LoRALinear) for m in model.modules())
        if not has_lora:
            inject_lora_adapters(model, lora_cfg)
        return save_lora_and_callosum_checkpoint(
            out_dir,
            model=model,
            projector=projector,
            lora_config=lora_cfg,
            sleep_cycle_index=0,
            trained_trajectories_count=0,
            notes=notes,
            base_model_id=resolved_hf_id,
            callosum_status=UNSTEERED_BOOTSTRAP_STATUS,
        )

    lora_cfg = LoRAConfig(
        rank=int(lora_rank),
        lora_alpha=float(lora_alpha),
        lora_dropout=float(lora_dropout),
        target_modules=DEFAULT_LORA_TARGET_MODULES,
        num_top_layers=num_top_layers,
        base_model_id=resolved_hf_id,
    )
    meta = save_corpus_callosum_checkpoint(
        out_dir,
        projector,
        sleep_cycle_index=0,
        trained_trajectories_count=0,
        notes=notes,
        base_model_id=resolved_hf_id,
        callosum_status=UNSTEERED_BOOTSTRAP_STATUS,
    )
    atomic_json(out_dir / "lora_config.json", lora_cfg.to_dict())
    torch.save({}, out_dir / "lora_adapters.pt")
    return meta


def train_impish_bloodmoon_callosum_and_lora(
    checkpoint_dir: str | Path | None = None,
    *,
    model: nn.Module | None = None,
    tokenizer: Any | None = None,
    hf_model_id: str = DEFAULT_IMPISH_BLOODMOON_HF_ID,
    train_pairs: Sequence[ContrastiveGapPair] | None = None,
    holdout_pairs: Sequence[ContrastiveGapPair] | None = None,
    sensors: ThreeBridgeSensors | None = None,
    epochs: int = 6,
    lr_projector: float = 8e-4,
    lr_lora: float = 3e-4,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.05,
    num_top_layers: int | None = 16,
    intermediate_size: int = 256,
    device: str | torch.device | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Train `SubliminalGapProjector` + `LoRALinear` for Impish Bloodmoon and run Holdout Checks A, B, and C."""
    resolved_hf_id = validate_pytorch_hf_model_source(hf_model_id)
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    out_dir = (
        Path(checkpoint_dir).resolve()
        if checkpoint_dir is not None
        else resolve_impish_bloodmoon_checkpoint_dir(root)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    if model is None or tokenizer is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = torch.bfloat16 if resolved_device.type == "cuda" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(resolved_hf_id)
        model = AutoModelForCausalLM.from_pretrained(resolved_hf_id, dtype=dtype)

    gap_token_id = ensure_subliminal_gap_token(tokenizer)
    vocab_len = len(tokenizer) if hasattr(tokenizer, "__len__") else len(tokenizer.get_vocab())
    if hasattr(model, "resize_token_embeddings") and vocab_len > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(vocab_len)
    model.to(resolved_device)

    resolved_hidden = detect_model_hidden_size(model)
    target_mods = detect_lora_target_modules(model)
    lora_cfg = LoRAConfig(
        rank=int(lora_rank),
        lora_alpha=float(lora_alpha),
        lora_dropout=float(lora_dropout),
        target_modules=target_mods,
        num_top_layers=num_top_layers,
        base_model_id=resolved_hf_id,
    )
    has_lora = any(isinstance(m, LoRALinear) for m in model.modules())
    if not has_lora:
        inject_lora_adapters(model, lora_cfg)

    projector = SubliminalGapProjector(
        hidden_size=resolved_hidden,
        intermediate_size=int(intermediate_size),
    ).to(resolved_device)

    if train_pairs is None:
        train_jsonl = root / "outputs" / "gap_trajectories_train.jsonl"
        train_pairs = load_contrastive_gap_pairs_jsonl(train_jsonl)
    if holdout_pairs is None:
        holdout_jsonl = root / "outputs" / "gap_trajectories_holdout.jsonl"
        if holdout_jsonl.exists():
            holdout_pairs = load_contrastive_gap_pairs_jsonl(holdout_jsonl)

    epoch_metrics = train_left_brain_lora_and_callosum(
        model,
        tokenizer,
        projector,
        train_pairs=train_pairs,
        gap_token_id=gap_token_id,
        device=resolved_device,
        epochs=int(epochs),
        lr_projector=float(lr_projector),
        lr_lora=float(lr_lora),
    )

    meta = save_lora_and_callosum_checkpoint(
        out_dir,
        model=model,
        projector=projector,
        lora_config=lora_cfg,
        sleep_cycle_index=1,
        trained_trajectories_count=len(train_pairs) * 2,
        notes=f"Trained Corpus Callosum + LoRA for {resolved_hf_id} ({len(train_pairs)} contrastive pairs, {epochs} epochs).",
        base_model_id=resolved_hf_id,
        callosum_status=TRAINED_STEERED_STATUS,
    )

    holdout_report: dict[str, Any] | None = None
    if holdout_pairs:
        eval_res = evaluate_holdout_pairs(
            model,
            tokenizer,
            projector,
            holdout_pairs=holdout_pairs,
            gap_token_id=gap_token_id,
            device=resolved_device,
            sensors=sensors,
        )
        holdout_report = eval_res.to_dict()
        atomic_json(out_dir / "impish_bloodmoon_holdout_report.json", holdout_report)

    return {
        "checkpoint_dir": str(out_dir),
        "base_model_id": resolved_hf_id,
        "hidden_size": resolved_hidden,
        "callosum_status": meta.callosum_status,
        "sleep_cycle_index": meta.sleep_cycle_index,
        "trained_trajectories_count": meta.trained_trajectories_count,
        "epochs": len(epoch_metrics),
        "final_epoch": (
            epoch_metrics[-1].to_dict()
            if (epoch_metrics and hasattr(epoch_metrics[-1], "to_dict"))
            else (dict(epoch_metrics[-1]) if epoch_metrics else None)
        ),
        "holdout_report": holdout_report,
    }


def _derive_keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    """Generate a deterministic cryptographic keystream from `key` and `nonce` using SHA-256 counter blocks."""
    blocks: list[bytes] = []
    produced = 0
    counter = 0
    while produced < length:
        blk = hmac.new(key, nonce + counter.to_bytes(8, "big"), hashlib.sha256).digest()
        blocks.append(blk)
        produced += len(blk)
        counter += 1
    return b"".join(blocks)[:length]


def _xor_bytes(data: bytes, stream: bytes) -> bytes:
    return bytes(a ^ b for a, b in zip(data, stream))


def encrypt_patreon_checkpoint_bundle(
    checkpoint_dir: str | Path,
    output_enc_path: str | Path,
    passphrase: str,
    *,
    pbkdf2_iterations: int = 200_000,
) -> Path:
    """Encrypt all files in `checkpoint_dir` into an authenticated `.enc` bundle locked by `passphrase`."""
    clean_pass = passphrase.strip()
    if len(clean_pass) < 6:
        raise ValueError("Patreon passphrase must be at least 6 characters.")
    in_dir = Path(checkpoint_dir).resolve()
    if not in_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {in_dir}")

    payload_files: dict[str, str] = {}
    for fpath in sorted(in_dir.rglob("*")):
        if fpath.is_file():
            rel = fpath.relative_to(in_dir).as_posix()
            payload_files[rel] = base64.b64encode(fpath.read_bytes()).decode("ascii")

    plaintext = json.dumps({"files": payload_files}, sort_keys=True).encode("utf-8")
    salt = secrets.token_bytes(16)
    nonce = secrets.token_bytes(16)
    master_key = hashlib.pbkdf2_hmac("sha256", clean_pass.encode("utf-8"), salt, int(pbkdf2_iterations), dklen=64)
    enc_key = master_key[:32]
    mac_key = master_key[32:]

    keystream = _derive_keystream(enc_key, nonce, len(plaintext))
    ciphertext = _xor_bytes(plaintext, keystream)
    mac = hmac.new(mac_key, salt + nonce + ciphertext, hashlib.sha256).hexdigest()

    envelope = {
        "schema_version": 1,
        "kdf": "PBKDF2-HMAC-SHA256",
        "iterations": int(pbkdf2_iterations),
        "salt_b64": base64.b64encode(salt).decode("ascii"),
        "nonce_b64": base64.b64encode(nonce).decode("ascii"),
        "ciphertext_b64": base64.b64encode(ciphertext).decode("ascii"),
        "hmac_sha256": mac,
    }
    out_path = Path(output_enc_path).resolve()
    atomic_json(out_path, envelope)
    return out_path


def decrypt_patreon_checkpoint_bundle(
    enc_path: str | Path,
    target_checkpoint_dir: str | Path,
    passphrase: str,
) -> Path:
    """Verify and decrypt an `.enc` Patreon checkpoint bundle into `target_checkpoint_dir`."""
    clean_pass = passphrase.strip()
    if not clean_pass:
        raise ValueError("Passphrase is required to unlock the encrypted Patreon bundle.")
    src = Path(enc_path).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Encrypted Patreon bundle not found: {src}")

    envelope = json.loads(src.read_text(encoding="utf-8"))
    iterations = int(envelope["iterations"])
    salt = base64.b64decode(envelope["salt_b64"])
    nonce = base64.b64decode(envelope["nonce_b64"])
    ciphertext = base64.b64decode(envelope["ciphertext_b64"])
    expected_mac = str(envelope["hmac_sha256"])

    master_key = hashlib.pbkdf2_hmac("sha256", clean_pass.encode("utf-8"), salt, iterations, dklen=64)
    enc_key = master_key[:32]
    mac_key = master_key[32:]

    actual_mac = hmac.new(mac_key, salt + nonce + ciphertext, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(actual_mac, expected_mac):
        raise ValueError("Invalid Patreon unlock key: cryptographic HMAC verification failed.")

    keystream = _derive_keystream(enc_key, nonce, len(ciphertext))
    plaintext = _xor_bytes(ciphertext, keystream)
    decoded = json.loads(plaintext.decode("utf-8"))

    out_dir = Path(target_checkpoint_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for rel_path, b64_bytes in decoded.get("files", {}).items():
        dest = out_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(base64.b64decode(b64_bytes))
    return out_dir
