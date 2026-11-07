"""GitHub SHA-256 Release Manifest & Safe Auto-Sync (`github_auto_sync.py`).

Plain-English Purpose:
Allows the standalone LMFT-Chess and LMFT-Bloodmoon desktop applications to:
1. Build a deterministic `dist/release_manifest.json` containing the current version,
   a single top-level build hash (`build_sha256`), and per-file SHA-256 digests for
   all tracked code and encrypted release bundles (`dist/impish_bloodmoon_patreon.enc`).
2. Compare the local installation's SHA-256 hashes against the remote GitHub manifest
   (`schaleh12/Dual-Hemisphere-Wisdom-Model`) and sync updated code/bundle files with
   strict SHA-256 verification.
3. Enforce Auditor Guardrail #4: Never overwrite, modify, or delete user runtime state
   in `outputs/chess_kibitzer_state/`, `outputs/club_state/`, `outputs/impish_bloodmoon_state/`,
   `threads.json`, or `outputs/impish_sparring_trajectories.jsonl`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
from pathlib import Path
from typing import Any
import urllib.request

APP_RELEASE_VERSION = "0.2.0"
GITHUB_REPO_SLUG = "schaleh12/Dual-Hemisphere-Wisdom-Model"
DEFAULT_REMOTE_MANIFEST_URL = (
    f"https://raw.githubusercontent.com/{GITHUB_REPO_SLUG}/main/LMFT-Training/dist/release_manifest.json"
)
LEGACY_REMOTE_MANIFEST_URL = (
    f"https://raw.githubusercontent.com/{GITHUB_REPO_SLUG}/main/dist/release_manifest.json"
)

SHARED_RELEASE_SEED_PATHS: tuple[str, ...] = (
    "outputs/rem_sleep_consolidated_ruleset.json",
    "outputs/sleep_consolidated_ruleset.json",
    "outputs/continuous_tiered_ruleset.json",
    "outputs/bitboard_sleeve_weights.json",
    "outputs/branch_archives/cultural_history.json",
    "outputs/branch_archives/experiential_felt_diary.json",
    "outputs/branch_archives/philosophical_lexicon.json",
)

PROTECTED_USER_STATE_PREFIXES: tuple[str, ...] = (
    "outputs/",
    "outputs/club_state",
    "outputs/club_user_state",
    "outputs/chess_kibitzer_state",
    "outputs/impish_bloodmoon_state",
    "outputs/impish_sparring_trajectories.jsonl",
)


def _normalize_rel_path(rel_path: str | Path) -> str:
    return str(rel_path).replace("\\", "/").lstrip("./")


def is_protected_user_state_path(rel_path: str | Path) -> bool:
    """Return True if `rel_path` belongs to user runtime state that must never be overwritten."""
    norm = _normalize_rel_path(rel_path)
    if not norm or ".." in norm.split("/"):
        return True
    if norm in SHARED_RELEASE_SEED_PATHS:
        return False
    if norm.endswith("threads.json") or norm.endswith("impish_sparring_trajectories.jsonl"):
        return True
    if (
        "club_state" in norm
        or "chess_kibitzer_state" in norm
        or "impish_bloodmoon_state" in norm
    ):
        return True
    return any(norm == p.rstrip("/") or norm.startswith(p) for p in PROTECTED_USER_STATE_PREFIXES)


def compute_file_sha256(path: str | Path) -> str:
    """Compute the hex SHA-256 digest of a file on disk."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _resolve_manifest_base_dir(repo_root: str | Path, rel: str) -> Path:
    from lmft_architecture.workspace_paths import resolve_arch_root, resolve_training_root

    norm = _normalize_rel_path(rel)
    if norm.startswith(("dist/", "models/", "outputs/", "experiments/", "scripts/")):
        return resolve_training_root(repo_root)
    return resolve_arch_root(repo_root)


def build_release_manifest(
    repo_root: str | Path,
    *,
    version: str = APP_RELEASE_VERSION,
    write_to_disk: bool = True,
) -> dict[str, Any]:
    """Build `dist/release_manifest.json` over tracked code and release bundles."""
    from lmft_architecture.workspace_paths import resolve_arch_root, resolve_training_root

    arch_root = resolve_arch_root(repo_root)
    train_root = resolve_training_root(repo_root)
    files_map: dict[str, dict[str, Any]] = {}

    candidate_pairs: list[tuple[str, Path]] = []
    src_dir = arch_root / "src" / "lmft_architecture"
    if src_dir.exists():
        for py_path in sorted(src_dir.glob("*.py")):
            rel_str = _normalize_rel_path(py_path.resolve().relative_to(arch_root))
            candidate_pairs.append((rel_str, py_path))
    for asset_rel in (
        "docs/assets/celestial_seal_poem.jpg",
        "docs/assets/icy_sea_board.png",
    ):
        asset_path = arch_root / asset_rel
        if asset_path.exists():
            candidate_pairs.append((asset_rel, asset_path))
    for rel_extra in SHARED_RELEASE_SEED_PATHS:
        extra = train_root / rel_extra
        if extra.exists():
            candidate_pairs.append((rel_extra, extra))

    for rel, file_path in candidate_pairs:
        if is_protected_user_state_path(rel):
            continue
        files_map[rel] = {
            "sha256": compute_file_sha256(file_path),
            "size_bytes": file_path.stat().st_size,
        }

    summary_hasher = hashlib.sha256()
    for rel_key in sorted(files_map):
        summary_hasher.update(f"{rel_key}:{files_map[rel_key]['sha256']}\n".encode("utf-8"))
    build_sha256 = summary_hasher.hexdigest()

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "version": version,
        "github_repo": GITHUB_REPO_SLUG,
        "build_sha256": build_sha256,
        "build_short_hash": build_sha256[:12],
        "files": files_map,
    }
    if write_to_disk:
        out_path = train_root / "dist" / "release_manifest.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_or_sync_against_manifest(
    local_root: str | Path,
    remote_manifest: Mapping[str, Any],
    *,
    file_fetcher: Callable[[str], bytes] | None = None,
    apply_sync: bool = False,
) -> dict[str, Any]:
    """Compare local installation files against `remote_manifest` and optionally auto-sync updates.

    Enforces Guardrail #4: Never touches or overwrites any path matching `is_protected_user_state_path`.
    """
    remote_files = remote_manifest.get("files", {})
    if not isinstance(remote_files, Mapping):
        raise ValueError("remote_manifest['files'] must be a mapping")

    changed_files: list[str] = []
    synced_files: list[str] = []
    skipped_protected: list[str] = []
    active_fetcher = file_fetcher if file_fetcher is not None else fetch_remote_github_file

    for rel_raw, meta in sorted(remote_files.items()):
        rel = _normalize_rel_path(rel_raw)
        if is_protected_user_state_path(rel):
            skipped_protected.append(rel)
            continue
        expected_sha = str(meta.get("sha256", "")).strip().lower()
        if not expected_sha:
            continue
        base_dir = _resolve_manifest_base_dir(local_root, rel)
        local_file = base_dir / rel
        local_sha = compute_file_sha256(local_file).lower() if local_file.exists() else ""
        if local_sha != expected_sha:
            changed_files.append(rel)
            if apply_sync and active_fetcher is not None:
                payload = active_fetcher(rel)
                actual_sha = hashlib.sha256(payload).hexdigest().lower()
                if actual_sha != expected_sha:
                    raise ValueError(
                        f"SHA-256 mismatch for {rel!r}: expected {expected_sha}, got {actual_sha}"
                    )
                local_file.parent.mkdir(parents=True, exist_ok=True)
                local_file.write_bytes(payload)
                synced_files.append(rel)

    if not changed_files:
        status = "up_to_date"
    elif apply_sync and len(synced_files) == len(changed_files):
        status = "synced"
    else:
        status = "update_available"

    return {
        "status": status,
        "remote_version": str(remote_manifest.get("version", APP_RELEASE_VERSION)),
        "remote_build_short_hash": str(remote_manifest.get("build_short_hash", "")),
        "changed_files": changed_files,
        "synced_files": synced_files,
        "skipped_protected": skipped_protected,
    }


def fetch_remote_github_file(
    rel_path: str,
    *,
    repo_slug: str = GITHUB_REPO_SLUG,
    branch: str = "main",
    timeout_sec: float = 15.0,
) -> bytes:
    """Fetch a single tracked file from GitHub raw content (`LMFT-Training/` or `LMFT-Architecture/`)."""
    norm = _normalize_rel_path(rel_path)
    if norm.startswith(("dist/", "models/", "outputs/", "experiments/", "scripts/")):
        sub_prefix = f"LMFT-Training/{norm}"
    else:
        sub_prefix = f"LMFT-Architecture/{norm}"
    candidates = (
        f"https://raw.githubusercontent.com/{repo_slug}/{branch}/{sub_prefix}",
        f"https://raw.githubusercontent.com/{repo_slug}/{branch}/{norm}",
    )
    last_err: Exception | None = None
    for url in candidates:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": f"LMFT-Sanctuary-Club/{APP_RELEASE_VERSION}"},
            )
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                return resp.read()
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Failed to fetch {rel_path!r} from GitHub: {last_err}")


def fetch_remote_github_manifest(
    manifest_url: str = DEFAULT_REMOTE_MANIFEST_URL,
    *,
    timeout_sec: float = 5.0,
) -> dict[str, Any]:
    """Fetch `release_manifest.json` from GitHub over HTTPS."""
    urls = (
        (manifest_url, LEGACY_REMOTE_MANIFEST_URL)
        if manifest_url == DEFAULT_REMOTE_MANIFEST_URL
        else (manifest_url,)
    )
    last_err: Exception | None = None
    for url in urls:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": f"LMFT-Sanctuary-Club/{APP_RELEASE_VERSION}"},
            )
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Failed to fetch remote manifest: {last_err}")
