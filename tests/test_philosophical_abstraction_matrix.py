"""Unit tests for the Calibrated 5-Tier Philosophical Abstraction Matrix (`src/lmft_lab/philosophical_abstraction_matrix.py`)."""

from __future__ import annotations

import pytest

from lmft_lab.philosophical_abstraction_matrix import (
    MATRIX_BY_CHESS_PIECE,
    MATRIX_BY_LEAF_ID,
    MATRIX_BY_SEED_NAME,
    PHILOSOPHICAL_ABSTRACTION_MATRIX,
    PHILOSOPHICAL_SIBLING_PAIRS,
    compute_philosophical_color_harmony,
    detect_shadow_pathologies,
)


def test_five_tier_matrix_entries_match_calibrated_wind_ice_chart() -> None:
    """Verify all 6 rows of the 5-Tier Matrix match the calibrated Wind <-> Ice chart."""
    assert len(PHILOSOPHICAL_ABSTRACTION_MATRIX) == 6
    assert len(MATRIX_BY_LEAF_ID) == 6
    assert len(MATRIX_BY_CHESS_PIECE) == 6

    expected = (
        ("agency", "Extra-dimensional", "philosophical/being/agency", "Red", "Red Summoning", "Lightning", "Queen", "Volition", "Psychosis", "Mammal", "Light"),
        ("existence", "Extra-dimensional", "philosophical/being/existence", "Cyan", "Cyan Sun Magic", "Water", "King", "Ground of Being", "Paranoia", "Reptile", "Sun"),
        ("order", "Inter-dimensional", "philosophical/structure/order", "Magenta", "Magenta Medium", "Wind", "Bishop", "Prevailing currents", "Prejudice", "Fish", "Ocean"),
        ("chaos", "Inter-dimensional", "philosophical/structure/chaos", "Green", "Green Instrument", "Ice", "Knight", "Brittle shatter / frost", "Corruption", "Bird", "Sky"),
        ("continuity", "Hyper-dimensional", "philosophical/persistence/continuity", "Blue", "Blue Soil Magic", "Fire", "Pawn", "Swarm continuum", "Mania / Greed", "Insect", "Soil"),
        ("discreteness", "Hyper-dimensional", "philosophical/persistence/discreteness", "Yellow", "Yellow Command", "Ground", "Rook", "Quantized bastion", "Narcissism", "Amphibian", "Shadow"),
    )
    for (
        seed_coord,
        dim,
        leaf_id,
        color,
        materia,
        element,
        piece,
        piece_sub,
        pathology,
        kingdom,
        env,
    ) in expected:
        entry = MATRIX_BY_SEED_NAME[seed_coord]
        assert entry.dimension == dim
        assert entry.leaf_id == leaf_id
        assert entry.color_name == color
        assert entry.materia_title == materia
        assert entry.element == element
        assert entry.chess_piece == piece
        assert entry.chess_subtitle == piece_sub
        assert entry.shadow_pathology == pathology
        assert entry.kingdom == kingdom
        assert entry.environment == env


def test_complementary_sibling_color_wheel_symmetry() -> None:
    """Verify Red+Cyan, Magenta+Green, and Blue+Yellow sum to (1, 1, 1) white/silver light."""
    for entry_a, entry_b in PHILOSOPHICAL_SIBLING_PAIRS:
        summed = tuple(entry_a.unit_rgb[i] + entry_b.unit_rgb[i] for i in range(3))
        assert summed == (1.0, 1.0, 1.0)

    balanced_coords = {entry.leaf_id: 0.70 for entry in PHILOSOPHICAL_ABSTRACTION_MATRIX}
    balances = compute_philosophical_color_harmony(balanced_coords)
    assert len(balances) == 3
    for bal in balances:
        assert bal.imbalance == 0.0
        assert bal.harmony == 1.0
        assert bal.blended_rgb == (0.7, 0.7, 0.7)
        assert bal.dominant_color is None


def test_detect_all_six_tier4_shadow_pathologies() -> None:
    """Verify each of the 6 Shadow Pathologies triggers under its exact ungrounded spike condition."""
    # Balanced calm coordinates -> 0 warnings
    calm_coords = {entry.leaf_id: 0.55 for entry in PHILOSOPHICAL_ABSTRACTION_MATRIX}
    calm_coords["cultural/justice/fairness"] = 0.75
    calm_coords["cultural/efficiency/wastelessness"] = 0.75
    assert detect_shadow_pathologies(calm_coords, fact=0.80, arousal=0.30, resonance=0.75, novelty=0.50) == ()

    # 1. Red -> Queen -> Psychosis
    w_psychosis = detect_shadow_pathologies(
        {"philosophical/being/agency": 0.88, "philosophical/being/existence": 0.20},
        fact=0.22,
    )
    assert len(w_psychosis) == 1
    assert w_psychosis[0].shadow_pathology == "Psychosis"
    assert w_psychosis[0].chess_piece == "Queen"
    assert w_psychosis[0].color_name == "Red"

    # 2. Cyan -> King -> Paranoia
    w_paranoia = detect_shadow_pathologies(
        {"philosophical/being/existence": 0.86, "philosophical/being/agency": 0.18},
        arousal=0.72,
    )
    assert len(w_paranoia) == 1
    assert w_paranoia[0].shadow_pathology == "Paranoia"
    assert w_paranoia[0].chess_piece == "King"
    assert w_paranoia[0].color_name == "Cyan"

    # 3. Magenta -> Bishop -> Prejudice
    w_prejudice = detect_shadow_pathologies(
        {
            "philosophical/structure/order": 0.85,
            "philosophical/structure/chaos": 0.15,
            "cultural/justice/fairness": 0.18,
        },
        novelty=0.20,
    )
    assert len(w_prejudice) == 1
    assert w_prejudice[0].shadow_pathology == "Prejudice"
    assert w_prejudice[0].chess_piece == "Bishop"
    assert w_prejudice[0].element == "Wind"

    # 4. Green -> Knight -> Corruption
    w_corruption = detect_shadow_pathologies(
        {
            "philosophical/structure/chaos": 0.88,
            "philosophical/structure/order": 0.16,
            "philosophical/persistence/continuity": 0.20,
        }
    )
    assert len(w_corruption) == 1
    assert w_corruption[0].shadow_pathology == "Corruption"
    assert w_corruption[0].chess_piece == "Knight"
    assert w_corruption[0].element == "Ice"

    # 5. Blue -> Pawn -> Mania / Greed
    w_mania = detect_shadow_pathologies(
        {
            "philosophical/persistence/continuity": 0.85,
            "philosophical/persistence/discreteness": 0.18,
            "cultural/efficiency/wastelessness": 0.20,
            "cultural/efficiency/power": 0.78,
        }
    )
    assert len(w_mania) == 1
    assert w_mania[0].shadow_pathology == "Mania / Greed"
    assert w_mania[0].chess_piece == "Pawn"
    assert w_mania[0].color_name == "Blue"

    # 6. Yellow -> Rook -> Narcissism (Beat 5 yellow stone enclosure!)
    w_narcissism = detect_shadow_pathologies(
        {
            "philosophical/persistence/discreteness": 0.86,
            "philosophical/persistence/continuity": 0.22,
            "cultural/justice/fairness": 0.15,
            "cultural/freedom/self-sufficiency": 0.84,
        },
        resonance=0.18,
    )
    assert len(w_narcissism) == 1
    assert w_narcissism[0].shadow_pathology == "Narcissism"
    assert w_narcissism[0].chess_piece == "Rook"
    assert w_narcissism[0].color_name == "Yellow"
