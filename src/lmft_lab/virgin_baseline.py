"""Virgin Dual-Hemisphere Baseline Snapshot (`src/lmft_lab/virgin_baseline.py`).

Plain-English Purpose:
Provides a clean, integrity-checked separation between:
1. **The Core Architecture ("Lab" / Virgin Baseline — `Cycle 0`)**:
   - Frozen 40-node Founding Seed (`24` founding leaves in `src/lmft_lab/birth.py`)
   - Calibrated 5-Tier Philosophical Abstraction Matrix (`Wellspring of Being` on
     Cyan/Water/King, `Ground` on Yellow/Rook)
   - 10 Built-in Structural Tier-2 Leaves (`4` Complex Feeling leaves + `6` Shadow
     Pathology leaves) with zero story-specific or REM-forged leaves
   - 2 Foundational Equilibrium Rules (`grounded_coordination`, `sanctuary_equilibrium`)
     with zero parable/story/coaching rules
   - Empty 3-Branch Archives (`0` precedents, `0` clauses, `0` ratified chords) and
     an empty `ContemplationLedger`
2. **Downstream Schooling / Experience ("School" — `Cycle 1+` & User Coaching)**:
   - Rules and 3-branch clauses forged during offline REM Sleep (`outputs/rem_sleep_consolidated_ruleset.json`)
   - Left-Brain `Qwen3-4B` LoRA + `SubliminalGapProjector` checkpoints (`models/left_brain_lora_cycle2/`)
   - User-coached turns and savored vibes (`outputs/club_user_state/`)

This separation makes debugging and accountability effortless: if a quirk appears in
the Virgin Baseline (`Cycle 0`), it is a Core Architecture issue; if the Virgin Baseline
is clean and the quirk only appears in `Cycle 1+`, it is a Schooling / Training issue.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .birth import BirthState, create_birth_state
from .branch_archives import ThreeBranchArchives
from .contemplation_ledger import (
    ContemplationLedger,
    load_contemplation_ledger,
    save_contemplation_ledger,
)
from .neural_bridge_classifiers import DEFAULT_COMPLEX_FEELING_LEAVES
from .philosophical_abstraction_matrix import (
    MATRIX_BY_SEED_NAME,
    PHILOSOPHICAL_ABSTRACTION_MATRIX,
    PHILOSOPHICAL_SHADOW_LEAVES,
)
from .reflection_classifier import (
    FOUNDATIONAL_LEAVES,
    FuzzyConceptRule,
    LearnedLeaf,
    load_learned_leaves,
    load_ruleset,
    save_ruleset,
)
from .runtime_store import atomic_json

VIRGIN_BASELINE_VERSION = "virgin-dual-hemisphere-v1"
VIRGIN_BASELINE_REL_DIR = Path("baselines") / "virgin_dual_hemisphere_v1"

VIRGIN_FOUNDATIONAL_RULES: tuple[FuzzyConceptRule, ...] = (
    FuzzyConceptRule(
        rule_id="rule_grounded_coordination",
        concept_name="grounded_coordination",
        description="Balanced structure, sheltered King existence (Wellspring of Being), and fair cooperation.",
        target_leaves={
            "philosophical/being/existence": 0.68,
            "philosophical/structure/order": 0.65,
        },
    ),
    FuzzyConceptRule(
        rule_id="rule_sanctuary_equilibrium",
        concept_name="sanctuary_equilibrium",
        description="Calm co-regulation with steady order and low chaos.",
        target_leaves={
            "cultural/freedom/needlessness": 0.78,
            "philosophical/being/existence": 0.68,
        },
        low=0.70,
        high=0.86,
    ),
)

VIRGIN_STRUCTURAL_LEAVES: tuple[LearnedLeaf, ...] = (
    DEFAULT_COMPLEX_FEELING_LEAVES + PHILOSOPHICAL_SHADOW_LEAVES
)


@dataclass(frozen=True)
class VirginBaselineState:
    """Loaded `Cycle 0` Virgin Dual-Hemisphere state with zero downstream schooling."""

    version: str
    snapshot_dir: Path
    birth_state: BirthState
    rules: tuple[FuzzyConceptRule, ...]
    learned_leaves: tuple[LearnedLeaf, ...]
    archives: ThreeBranchArchives
    ledger: ContemplationLedger
    manifest: dict[str, Any]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_virgin_baseline_dir(repo_root: str | Path | None = None) -> Path:
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    return root / VIRGIN_BASELINE_REL_DIR


def materialize_virgin_baseline_snapshot(
    target_dir: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
) -> Path:
    """Write the canonical `Cycle 0` Virgin Dual-Hemisphere snapshot files and SHA-256 manifest."""
    snap_dir = (
        Path(target_dir).resolve()
        if target_dir is not None
        else resolve_virgin_baseline_dir(repo_root)
    )
    snap_dir.mkdir(parents=True, exist_ok=True)

    ruleset_path = snap_dir / "virgin_ruleset.json"
    archives_dir = snap_dir / "branch_archives"
    ledger_path = snap_dir / "contemplation_ledger.json"

    save_ruleset(ruleset_path, VIRGIN_FOUNDATIONAL_RULES, VIRGIN_STRUCTURAL_LEAVES)
    empty_archives = ThreeBranchArchives()
    archive_paths = empty_archives.save_all(archives_dir)
    empty_ledger = ContemplationLedger()
    save_contemplation_ledger(ledger_path, empty_ledger)

    tracked_files = {
        "virgin_ruleset.json": _sha256_file(ruleset_path),
        "branch_archives/cultural_history.json": _sha256_file(archive_paths["cultural_history"]),
        "branch_archives/experiential_felt_diary.json": _sha256_file(archive_paths["experiential_felt_diary"]),
        "branch_archives/philosophical_lexicon.json": _sha256_file(archive_paths["philosophical_lexicon"]),
        "contemplation_ledger.json": _sha256_file(ledger_path),
    }

    birth_probe = create_birth_state("virgin-probe", birth_event_id="e0", random_seed=42)
    all_founding_leaves = [n for n in birth_probe.nodes if n.depth == 2]
    manifest = {
        "version": VIRGIN_BASELINE_VERSION,
        "sleep_cycle_index": 0,
        "lineage": "VIRGIN_CORE_ARCHITECTURE_CYCLE_0",
        "founding_nodes_count": len(birth_probe.nodes),
        "foundational_leaves_count": len(all_founding_leaves),
        "philosophical_matrix_leaves_count": len(PHILOSOPHICAL_ABSTRACTION_MATRIX),
        "existence_king_subtitle": MATRIX_BY_SEED_NAME["existence"].chess_subtitle,
        "discreteness_rook_element": MATRIX_BY_SEED_NAME["discreteness"].element,
        "structural_tier2_leaves_count": len(VIRGIN_STRUCTURAL_LEAVES),
        "structural_tier2_leaf_ids": [leaf.leaf_id for leaf in VIRGIN_STRUCTURAL_LEAVES],
        "foundational_rules_count": len(VIRGIN_FOUNDATIONAL_RULES),
        "foundational_rule_ids": [rule.rule_id for rule in VIRGIN_FOUNDATIONAL_RULES],
        "story_or_rem_rules_count": 0,
        "archive_precedents_count": 0,
        "ratified_chords_count": 0,
        "files_sha256": tracked_files,
    }
    atomic_json(snap_dir / "virgin_manifest.json", manifest)
    return snap_dir


def verify_virgin_baseline_snapshot(
    snapshot_dir: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify that the `Cycle 0` Virgin Baseline snapshot is intact and uncontaminated by schooling."""
    snap_dir = (
        Path(snapshot_dir).resolve()
        if snapshot_dir is not None
        else resolve_virgin_baseline_dir(repo_root)
    )
    manifest_path = snap_dir / "virgin_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Virgin baseline manifest not found at {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != VIRGIN_BASELINE_VERSION:
        raise ValueError(f"Unexpected virgin baseline version: {manifest.get('version')!r}")
    if int(manifest.get("sleep_cycle_index", -1)) != 0:
        raise ValueError("Virgin baseline must have sleep_cycle_index == 0")
    if manifest.get("existence_king_subtitle") != "Wellspring of Being":
        raise ValueError(
            "Virgin baseline requires existence/King chess_subtitle == 'Wellspring of Being' "
            "(never 'Ground of Being', to avoid collision with the Rook 'Ground' element)."
        )

    files_sha256 = manifest.get("files_sha256", {})
    if not isinstance(files_sha256, dict) or not files_sha256:
        raise ValueError("Virgin baseline manifest missing files_sha256 map")

    for rel_path, expected_sha in files_sha256.items():
        fpath = snap_dir / rel_path
        if not fpath.exists():
            raise FileNotFoundError(f"Missing virgin baseline file: {fpath}")
        actual_sha = _sha256_file(fpath)
        if actual_sha != expected_sha:
            raise ValueError(
                f"Virgin baseline file modified/contaminated: {rel_path} "
                f"(expected {expected_sha}, got {actual_sha})"
            )

    rules = load_ruleset(snap_dir / "virgin_ruleset.json")
    leaves = load_learned_leaves(snap_dir / "virgin_ruleset.json")
    archives = ThreeBranchArchives.load_all(snap_dir / "branch_archives")
    ledger = load_contemplation_ledger(snap_dir / "contemplation_ledger.json")

    expected_rule_ids = {r.rule_id for r in VIRGIN_FOUNDATIONAL_RULES}
    actual_rule_ids = {r.rule_id for r in rules}
    if actual_rule_ids != expected_rule_ids:
        raise ValueError(
            f"Virgin ruleset contaminated by non-foundational rules: {sorted(actual_rule_ids)}"
        )

    expected_leaf_ids = {l.leaf_id for l in VIRGIN_STRUCTURAL_LEAVES}
    actual_leaf_ids = {l.leaf_id for l in leaves}
    if actual_leaf_ids != expected_leaf_ids:
        raise ValueError(
            f"Virgin learned_leaves contaminated by story leaves: {sorted(actual_leaf_ids ^ expected_leaf_ids)}"
        )

    if (
        len(archives.executive.precedents) != 0
        or len(archives.executive.action_clauses) != 0
        or len(archives.legislative.diary_snapshots) != 0
        or len(archives.legislative.feeling_clauses) != 0
        or len(archives.judicial.structural_clauses) != 0
        or len(archives.judicial.ratified_chords) != 0
        or len(ledger.settled_certainties) != 0
        or len(ledger.active_uncertainties) != 0
    ):
        raise ValueError("Virgin baseline archives or contemplation ledger must be empty (0 scenes/clauses).")

    return manifest


def load_virgin_baseline_state(
    snapshot_dir: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
    instance_id: str = "lmft-virgin-baseline-cycle0",
) -> VirginBaselineState:
    """Verify and load a clean `Cycle 0` Virgin Dual-Hemisphere state from disk."""
    snap_dir = (
        Path(snapshot_dir).resolve()
        if snapshot_dir is not None
        else resolve_virgin_baseline_dir(repo_root)
    )
    if not (snap_dir / "virgin_manifest.json").exists():
        materialize_virgin_baseline_snapshot(snap_dir, repo_root=repo_root)

    manifest = verify_virgin_baseline_snapshot(snap_dir, repo_root=repo_root)
    birth = create_birth_state(
        instance_id,
        birth_event_id="virgin-birth-cycle0",
        random_seed=42,
    )
    rules = load_ruleset(snap_dir / "virgin_ruleset.json")
    leaves = load_learned_leaves(snap_dir / "virgin_ruleset.json")
    archives = ThreeBranchArchives.load_all(snap_dir / "branch_archives")
    ledger = load_contemplation_ledger(snap_dir / "contemplation_ledger.json")

    return VirginBaselineState(
        version=VIRGIN_BASELINE_VERSION,
        snapshot_dir=snap_dir,
        birth_state=birth,
        rules=rules,
        learned_leaves=leaves,
        archives=archives,
        ledger=ledger,
        manifest=manifest,
    )
