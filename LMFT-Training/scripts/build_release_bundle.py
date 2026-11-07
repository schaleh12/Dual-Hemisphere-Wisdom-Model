"""Build a portable GitHub Release zip bundle for the LMFT-Chess and LMFT-Bloodmoon desktop apps (`build_release_bundle.py`).

Creates `dist/lmft-sanctuary-club-v0.1.0-win64.zip` containing:
- `LMFT-Chess.exe` & `LMFT-Bloodmoon.exe` (standalone Windows launchers)
- `src/lmft_architecture/*.py` (all core runtime, right-brain, comma-splice, chess & bloodmoon modules)
- `outputs/rem_sleep_consolidated_ruleset.json` & `outputs/continuous_tiered_ruleset.json`
- `models/right_brain_bridges/` & `models/left_brain_lora_cycle2/` (when present on disk)
"""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

RELEASE_VERSION = "v0.1.0"
BUNDLE_NAME = f"lmft-sanctuary-club-{RELEASE_VERSION}-win64.zip"


def build_release_zip(
    repo_root: str | Path | None = None,
    *,
    output_dir: str | Path | None = None,
    include_model_weights: bool = False,
) -> Path:
    from lmft_architecture.github_auto_sync import build_release_manifest
    from lmft_architecture.workspace_paths import (
        resolve_arch_root,
        resolve_training_root,
        resolve_workspace_root,
    )

    arch_root = resolve_arch_root(repo_root)
    train_root = resolve_training_root(repo_root)
    ws_root = resolve_workspace_root(repo_root)

    out_dir = Path(output_dir).resolve() if output_dir is not None else (train_root / "dist")
    out_dir.mkdir(parents=True, exist_ok=True)
    is_default_dist = out_dir == (train_root / "dist").resolve()
    manifest = build_release_manifest(arch_root, write_to_disk=is_default_dist)
    manifest_path = out_dir / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    zip_path = out_dir / BUNDLE_NAME

    include_pairs: list[tuple[str, Path]] = [
        ("dist/release_manifest.json", manifest_path),
    ]
    for root_file in (
        "LMFT-Installer-Wizard.exe",
        "LMFT-Chess.exe",
        "LMFT-Bloodmoon.exe",
        "README.md",
        "RELEASE_NOTES.md",
    ):
        for base in (ws_root, arch_root):
            p = base / root_file
            if p.exists():
                include_pairs.append((root_file, p))
                break

    for asset_rel in (
        "docs/assets/celestial_seal_poem.jpg",
        "docs/assets/blue_screen_of_life_poem.jpg",
        "docs/assets/arecibo_desktop_icon.jpg",
        "docs/assets/arecibo_desktop_icon.ico",
        "docs/assets/celestriad_spirit_particle.jpg",
        "docs/assets/celestriad_spirit_particle.ico",
        "docs/assets/icy_sea_board.png",
    ):
        asset_path = arch_root / asset_rel
        if asset_path.exists():
            include_pairs.append((asset_rel, asset_path))

    for py_file in sorted((arch_root / "src" / "lmft_architecture").rglob("*")):
        if py_file.is_file() and py_file.suffix in (".py", ".enc"):
            include_pairs.append((str(py_file.relative_to(arch_root)), py_file))

    virgin_dir = arch_root / "baselines" / "virgin_dual_hemisphere_v1"
    if virgin_dir.exists():
        for vf in sorted(virgin_dir.rglob("*")):
            if vf.is_file():
                include_pairs.append((str(vf.relative_to(arch_root)), vf))

    for ruleset_rel in (
        "outputs/rem_sleep_consolidated_ruleset.json",
        "outputs/sleep_consolidated_ruleset.json",
        "outputs/continuous_tiered_ruleset.json",
        "outputs/bitboard_sleeve_weights.json",
        "outputs/branch_archives/cultural_history.json",
        "outputs/branch_archives/experiential_felt_diary.json",
        "outputs/branch_archives/philosophical_lexicon.json",
    ):
        p = train_root / ruleset_rel
        if p.exists():
            include_pairs.append((ruleset_rel, p))

    if include_model_weights:
        for model_rel in (
            "models/right_brain_bridges",
            "models/left_brain_qwen3_8b",
            "models/left_brain_lora_cycle2",
        ):
            m_dir = train_root / model_rel
            if m_dir.exists():
                for mf in sorted(m_dir.rglob("*")):
                    if mf.is_file():
                        include_pairs.append((str(mf.relative_to(train_root)), mf))

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arcname, file_path in include_pairs:
            zf.write(file_path, arcname=arcname)

    return zip_path


if __name__ == "__main__":
    created = build_release_zip(include_model_weights=False)
    print(f"Created GitHub Release bundle: {created} ({created.stat().st_size / 1024:.1f} KB)")
