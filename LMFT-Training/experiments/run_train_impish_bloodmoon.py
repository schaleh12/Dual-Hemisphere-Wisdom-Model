"""CLI Runner: Bootstrap or Train `Impish Bloodmoon` Corpus Callosum + LoRA (`experiments/run_train_impish_bloodmoon.py`).

Plain-English Purpose:
1. Clones the verified `Cycle 0` Virgin Baseline into `outputs/impish_bloodmoon_state/`.
2. Either initializes a `Cycle 0` zero-delta bootstrap checkpoint (`--bootstrap-only`,
   tagged `UNSTEERED_BOOTSTRAP_CYCLE_0`) or trains `SubliminalGapProjector` + `LoRALinear`
   on `outputs/gap_trajectories_train.jsonl` and evaluates Holdout Checks A, B, and C on
   `outputs/gap_trajectories_holdout.jsonl` (`TRAINED_STEERED_CALLOSUM`).
3. Optionally packages `models/left_brain_impish_bloodmoon/` into an encrypted Patreon
   bundle (`--pack-patreon-passphrase`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lmft_architecture.impish_bloodmoon_adapter import (  # noqa: E402
    DEFAULT_IMPISH_BLOODMOON_HIDDEN_SIZE,
    DEFAULT_IMPISH_BLOODMOON_HF_ID,
    bootstrap_impish_bloodmoon_virgin_checkpoint,
    clone_virgin_baseline_for_impish_bloodmoon,
    decrypt_patreon_checkpoint_bundle,
    encrypt_patreon_checkpoint_bundle,
    resolve_impish_bloodmoon_checkpoint_dir,
    resolve_impish_bloodmoon_state_dir,
    train_impish_bloodmoon_callosum_and_lora,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap, train, or encrypt/decrypt the Impish Bloodmoon 12B Left-Brain LoRA + Corpus Callosum checkpoint."
    )
    parser.add_argument(
        "--hf-model-id",
        default=DEFAULT_IMPISH_BLOODMOON_HF_ID,
        help=f"HuggingFace model ID or local PyTorch directory (default: {DEFAULT_IMPISH_BLOODMOON_HF_ID})",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(resolve_impish_bloodmoon_checkpoint_dir(REPO_ROOT)),
        help="Output checkpoint directory (default: models/left_brain_impish_bloodmoon)",
    )
    parser.add_argument(
        "--state-dir",
        default=str(resolve_impish_bloodmoon_state_dir(REPO_ROOT)),
        help="Isolated Right-Brain state directory (default: outputs/impish_bloodmoon_state)",
    )
    parser.add_argument(
        "--bootstrap-only",
        action="store_true",
        help="Create a Cycle 0 UNSTEERED_BOOTSTRAP_CYCLE_0 checkpoint without loading GPU weights or training.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=DEFAULT_IMPISH_BLOODMOON_HIDDEN_SIZE,
        help=f"Embedding hidden size when using --bootstrap-only (default: {DEFAULT_IMPISH_BLOODMOON_HIDDEN_SIZE})",
    )
    parser.add_argument("--epochs", type=int, default=6, help="Training epochs (default: 6)")
    parser.add_argument("--lr-projector", type=float, default=8e-4, help="Projector learning rate (default: 8e-4)")
    parser.add_argument("--lr-lora", type=float, default=3e-4, help="LoRA learning rate (default: 3e-4)")
    parser.add_argument(
        "--encrypt-bundle",
        default="",
        help="Encrypt --checkpoint-dir into the specified .enc file using --passphrase (without re-training).",
    )
    parser.add_argument(
        "--decrypt-bundle",
        default="",
        help="Decrypt the specified .enc file into --checkpoint-dir using --passphrase.",
    )
    parser.add_argument(
        "--passphrase",
        default="",
        help="Passphrase for --encrypt-bundle or --decrypt-bundle.",
    )
    parser.add_argument(
        "--pack-patreon-passphrase",
        default="",
        help="Optional passphrase to also write an encrypted dist/impish_bloodmoon_patreon.enc bundle after training.",
    )
    args = parser.parse_args(argv)

    if args.decrypt_bundle.strip():
        secret = args.passphrase.strip() or args.pack_patreon_passphrase.strip()
        out_dir = decrypt_patreon_checkpoint_bundle(
            args.decrypt_bundle.strip(),
            args.checkpoint_dir,
            secret,
        )
        print(f"Decrypted Patreon bundle into: {out_dir}")
        return 0

    if args.encrypt_bundle.strip():
        secret = args.passphrase.strip() or args.pack_patreon_passphrase.strip()
        enc_out = encrypt_patreon_checkpoint_bundle(
            args.checkpoint_dir,
            args.encrypt_bundle.strip(),
            secret,
        )
        print(f"Encrypted Patreon bundle written to: {enc_out}")
        return 0

    clone = clone_virgin_baseline_for_impish_bloodmoon(
        args.state_dir,
        repo_root=REPO_ROOT,
        hf_model_id=args.hf_model_id,
    )
    print(f"Verified & cloned Virgin Baseline into isolated state dir: {clone.snapshot_dir}")

    if args.bootstrap_only:
        meta = bootstrap_impish_bloodmoon_virgin_checkpoint(
            args.checkpoint_dir,
            hf_model_id=args.hf_model_id,
            hidden_size=args.hidden_size,
            repo_root=REPO_ROOT,
        )
        print(json.dumps(meta.to_dict(), indent=2))
    else:
        report = train_impish_bloodmoon_callosum_and_lora(
            args.checkpoint_dir,
            hf_model_id=args.hf_model_id,
            epochs=args.epochs,
            lr_projector=args.lr_projector,
            lr_lora=args.lr_lora,
            repo_root=REPO_ROOT,
        )
        print(json.dumps(report, indent=2))

    pack_secret = args.pack_patreon_passphrase.strip() or args.passphrase.strip()
    if pack_secret:
        enc_out = REPO_ROOT / "dist" / "impish_bloodmoon_patreon.enc"
        encrypt_patreon_checkpoint_bundle(
            args.checkpoint_dir,
            enc_out,
            pack_secret,
        )
        print(f"Encrypted Patreon bundle written to: {enc_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
