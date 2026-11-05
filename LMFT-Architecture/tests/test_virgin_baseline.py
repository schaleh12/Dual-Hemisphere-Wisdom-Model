"""Unit tests for the `Cycle 0` Virgin Dual-Hemisphere Baseline Snapshot & `Wellspring of Being` ontology."""

from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from lmft_architecture.philosophical_abstraction_matrix import MATRIX_BY_SEED_NAME
from lmft_architecture.reflection_classifier import FuzzyConceptRule, save_ruleset
from lmft_architecture.sanctuary_club_app import SanctuaryClubRuntime
from lmft_architecture.virgin_baseline import (
    VIRGIN_FOUNDATIONAL_RULES,
    VIRGIN_STRUCTURAL_LEAVES,
    load_virgin_baseline_state,
    materialize_virgin_baseline_snapshot,
    resolve_virgin_baseline_dir,
    verify_virgin_baseline_snapshot,
)


def test_wellspring_of_being_eliminates_ground_element_collision() -> None:
    """Verify King (`existence` / Water) uses 'Wellspring of Being' while Rook (`discreteness`) owns the 'Ground' element."""
    existence_entry = MATRIX_BY_SEED_NAME["existence"]
    agency_entry = MATRIX_BY_SEED_NAME["agency"]
    discreteness_entry = MATRIX_BY_SEED_NAME["discreteness"]

    assert existence_entry.element == "Water"
    assert existence_entry.chess_piece == "King"
    assert existence_entry.chess_subtitle == "Wellspring of Being"
    assert "ground" not in existence_entry.chess_subtitle.lower()

    assert agency_entry.chess_piece == "Queen"
    assert agency_entry.shadow_subtitle == "Unanchored volition"

    assert discreteness_entry.chess_piece == "Rook"
    assert discreteness_entry.element == "Ground"


def test_virgin_baseline_snapshot_integrity_and_tamper_detection(tmp_path: Path) -> None:
    """Verify `baselines/virgin_dual_hemisphere_v1/` is intact and rejects schooling contamination."""
    canonical_dir = resolve_virgin_baseline_dir()
    manifest = verify_virgin_baseline_snapshot(canonical_dir)
    assert manifest["sleep_cycle_index"] == 0
    assert manifest["founding_nodes_count"] == 40
    assert manifest["foundational_leaves_count"] == 24
    assert manifest["structural_tier2_leaves_count"] == 10
    assert manifest["foundational_rules_count"] == 2
    assert manifest["story_or_rem_rules_count"] == 0
    assert manifest["existence_king_subtitle"] == "Wellspring of Being"

    state = load_virgin_baseline_state(canonical_dir)
    assert len(state.birth_state.nodes) == 40
    assert len(state.rules) == 2
    assert len(state.learned_leaves) == 10
    assert len(state.archives.judicial.ratified_chords) == 0

    # Copy snapshot to tmp_path and verify tamper detection catches an injected story rule
    copy_dir = tmp_path / "tampered_virgin"
    shutil.copytree(canonical_dir, copy_dir)
    contaminated_rules = VIRGIN_FOUNDATIONAL_RULES + (
        FuzzyConceptRule(
            rule_id="rule-story-contamination",
            concept_name="story_contamination",
            description="Downstream story rule that does not belong in Cycle 0.",
            target_leaves={"philosophical/structure/order": 0.75},
        ),
    )
    save_ruleset(copy_dir / "virgin_ruleset.json", contaminated_rules, VIRGIN_STRUCTURAL_LEAVES)
    with pytest.raises(ValueError, match="modified/contaminated"):
        verify_virgin_baseline_snapshot(copy_dir)


def test_sanctuary_club_lineage_switch_virgin_vs_schooled(tmp_path: Path) -> None:
    """Verify `switch_brain_lineage` toggles cleanly between `schooled` (Cycle 2) and `virgin` (Cycle 0) without amnesia."""
    state_dir = tmp_path / "user_state"
    runtime = SanctuaryClubRuntime(state_dir=state_dir, lazy_load_gpu=False)
    schooled_rules_count = len(runtime.rules)
    assert runtime.active_lineage == "schooled"
    assert runtime.sleep_cycle_index == 2

    # Switch to Cycle 0 Virgin Baseline
    snap_virgin = runtime.switch_brain_lineage("virgin")
    assert snap_virgin["active_lineage"] == "virgin"
    assert snap_virgin["sleep_cycle_index"] == 0
    assert snap_virgin["rules_count"] == 2
    assert snap_virgin["ratified_chords_count"] == 0

    # Switch back to Schooled (Cycle 2)
    snap_schooled = runtime.switch_brain_lineage("schooled")
    assert snap_schooled["active_lineage"] == "schooled"
    assert snap_schooled["sleep_cycle_index"] == 2
    assert snap_schooled["rules_count"] == schooled_rules_count
