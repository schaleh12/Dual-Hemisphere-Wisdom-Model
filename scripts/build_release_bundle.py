"""Build a portable GitHub Release zip bundle for the LMFT Sanctuary & Chess Club app (`build_release_bundle.py`).

Creates `dist/lmft-sanctuary-club-v0.1.0-win64.zip` containing:
- `Start-LMFT-Club.bat` (one-click Windows launcher)
- `src/lmft_lab/*.py` (all core runtime, right-brain, comma-splice, chess & sanctuary modules)
- `outputs/rem_sleep_consolidated_ruleset.json` & `outputs/continuous_tiered_ruleset.json`
- `models/right_brain_bridges/` & `models/left_brain_lora_cycle2/` (when present on disk)
"""

from __future__ import annotations

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
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[1]
    out_dir = Path(output_dir).resolve() if output_dir is not None else (root / "dist")
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / BUNDLE_NAME

    include_files: list[Path] = []
    launcher = root / "Start-LMFT-Club.bat"
    if launcher.exists():
        include_files.append(launcher)

    for py_file in sorted((root / "src" / "lmft_lab").rglob("*.py")):
        include_files.append(py_file)

    virgin_dir = root / "baselines" / "virgin_dual_hemisphere_v1"
    if virgin_dir.exists():
        for vf in sorted(virgin_dir.rglob("*")):
            if vf.is_file():
                include_files.append(vf)

    for ruleset_rel in (
        "outputs/rem_sleep_consolidated_ruleset.json",
        "outputs/sleep_consolidated_ruleset.json",
        "outputs/continuous_tiered_ruleset.json",
    ):
        p = root / ruleset_rel
        if p.exists():
            include_files.append(p)

    if include_model_weights:
        for model_rel in ("models/right_brain_bridges", "models/left_brain_lora_cycle2"):
            m_dir = root / model_rel
            if m_dir.exists():
                for mf in sorted(m_dir.rglob("*")):
                    if mf.is_file():
                        include_files.append(mf)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in include_files:
            rel_arc = file_path.relative_to(root)
            zf.write(file_path, arcname=str(rel_arc))

    return zip_path


if __name__ == "__main__":
    created = build_release_zip(include_model_weights=False)
    print(f"Created GitHub Release bundle: {created} ({created.stat().st_size / 1024:.1f} KB)")
