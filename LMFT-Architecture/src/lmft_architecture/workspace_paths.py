"""Workspace & Folder Split Path Resolution (`workspace_paths.py`).

Plain-English Purpose:
Allows the top-level `Language Model Fuzzy Tree Research` workspace to cleanly
separate:
- `LMFT-Architecture/` — Pure core architecture (`src/`, `baselines/`, `config/`, `docs/`, `tests/`)
- `LMFT-Training/`     — Training weights, runtime outputs, experiments, and release bundles (`models/`, `outputs/`, `experiments/`, `dist/`, `scripts/`)
- Top-Level Root       — `.git`, `.gitignore`, `README.md`, `PROJECT_STATE.md`, `DECISIONS.md`, `AGENTS.md`, `LMFT-Architecture.exe`

While preserving 100% isolated `tmp_path` behavior whenever a unit test passes a temporary `repo_root`.
"""

from __future__ import annotations

from pathlib import Path

ARCH_ROOT: Path = Path(__file__).resolve().parents[2]


def _is_real_workspace_path(candidate: Path) -> bool:
    """Return True if `candidate` is the real workspace root, `LMFT-Architecture`, or `LMFT-Training`."""
    resolved = candidate.resolve()
    workspace_parent = ARCH_ROOT.parent.resolve()
    return resolved in (
        ARCH_ROOT,
        workspace_parent,
        (workspace_parent / "LMFT-Training").resolve(),
    )


def resolve_arch_root(repo_root: str | Path | None = None) -> Path:
    """Resolve the core architecture root (`LMFT-Architecture/`) or return a custom test `tmp_path`."""
    if repo_root is None:
        return ARCH_ROOT
    r = Path(repo_root).resolve()
    if not _is_real_workspace_path(r):
        return r
    return ARCH_ROOT


def resolve_training_root(repo_root: str | Path | None = None) -> Path:
    """Resolve the training & runtime state root (`LMFT-Training/`) or return a custom test `tmp_path`."""
    if repo_root is not None:
        r = Path(repo_root).resolve()
        if not _is_real_workspace_path(r):
            return r
    sibling = ARCH_ROOT.parent / "LMFT-Training"
    if sibling.exists():
        return sibling.resolve()
    return ARCH_ROOT


def resolve_workspace_root(repo_root: str | Path | None = None) -> Path:
    """Resolve the top-level Git workspace root (`Language Model Fuzzy Tree Research/`) or return a custom test `tmp_path`."""
    if repo_root is not None:
        r = Path(repo_root).resolve()
        if not _is_real_workspace_path(r):
            return r
    parent = ARCH_ROOT.parent
    if (parent / "PROJECT_STATE.md").exists() or (parent / ".git").exists():
        return parent.resolve()
    return ARCH_ROOT


def resolve_split_path(rel_path: str | Path, repo_root: str | Path | None = None) -> Path:
    """Resolve a relative project path (`src/...`, `experiments/...`, `outputs/...`, `config/...`, `AGENTS.md`) across the split."""
    p = Path(rel_path)
    if p.is_absolute():
        return p
    norm = p.as_posix()
    while norm.startswith("./"):
        norm = norm[2:]
    if repo_root is not None and not _is_real_workspace_path(Path(repo_root)):
        return Path(repo_root).resolve() / norm
    first = norm.split("/", 1)[0]
    if first in ("experiments", "outputs", "models", "dist", "scripts"):
        return resolve_training_root(repo_root) / norm
    if first in ("AGENTS.md", "DECISIONS.md", "PROJECT_STATE.md", "README.md", ".gitignore"):
        ws = resolve_workspace_root(repo_root) / norm
        if ws.exists():
            return ws
    return resolve_arch_root(repo_root) / norm
