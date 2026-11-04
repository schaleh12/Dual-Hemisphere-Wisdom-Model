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
        ("existence", "Extra-dimensional", "philosophical/being/existence", "Cyan", "Cyan Sun Magic", "Water", "King", "Wellspring of Being", "Paranoia", "Reptile", "Sun"),
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
            "cultural/justice/fairness": 0.25,
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


def test_philosophical_matrix_woven_into_vibe_subliminal_chord_chess_and_turn_result():
    """Verify the 4 integration points across RightBrainVibe, SubliminalChordVector, Chess, and ThreeLeggedTurnResult."""
    from lmft_lab.birth import create_birth_state
    from lmft_lab.gap_trajectory_schema import compile_chess_posture_chords
    from lmft_lab.philosophical_abstraction_matrix import PHILOSOPHICAL_SHADOW_LEAVES
    from lmft_lab.right_brain_equilibrium import (
        EXPERIENTIAL_SENSOR_LEAVES,
        TRUTH_PHILOSOPHICAL_LEAVES,
        WILL_CULTURAL_LEAVES,
        ThreeBridgeSensors,
        compile_right_brain_vibe,
    )
    from lmft_lab.subliminal_gap_splice import SubliminalChordVector
    from lmft_lab.three_legged_runtime import run_three_legged_turn

    # 1. Verify PHILOSOPHICAL_SHADOW_LEAVES are real Tier-2 child drawers with 8-float anchor_weights
    assert len(PHILOSOPHICAL_SHADOW_LEAVES) == 6
    shadow_ids = tuple(l.leaf_id for l in PHILOSOPHICAL_SHADOW_LEAVES)
    assert shadow_ids == (
        "philosophical/being/psychosis",
        "philosophical/being/paranoia",
        "philosophical/structure/prejudice",
        "philosophical/structure/corruption",
        "philosophical/persistence/mania_greed",
        "philosophical/persistence/narcissism",
    )

    birth = create_birth_state("ind-matrix-weave", birth_event_id="evt-mw-1", random_seed=42)
    calm_coords = {
        **{k: 0.50 for k in WILL_CULTURAL_LEAVES},
        **{k: 0.50 for k in TRUTH_PHILOSOPHICAL_LEAVES},
    }
    calm_exp = {
        "experiential/reflection/salience": 0.65,
        "experiential/phase/valence": 0.50,
        "experiential/phase/arousal": 0.45,
    }
    vibe_calm, _ = compile_right_brain_vibe(
        birth, calm_coords, calm_exp, (), pass_index=1, event_id="ev-calm"
    )
    assert len(vibe_calm.sibling_color_balances) == 3
    assert len(vibe_calm.shadow_pathology_warnings) == 0
    chord_calm = SubliminalChordVector.from_right_brain_vibe(vibe_calm)
    assert chord_calm.thinker_vector[8:16] == (0.0,) * 8

    # When an external scene triggers Narcissism (Rook yellow enclosure), the shadow leaf activates
    # in vibe.coordinates and flows directly into thinker_vector[8:16]!
    narcissism_coords = {
        **calm_coords,
        "philosophical/persistence/discreteness": 0.88,
        "philosophical/persistence/continuity": 0.18,
        "cultural/justice/fairness": 0.16,
    }
    vibe_narc, _ = compile_right_brain_vibe(
        birth, narcissism_coords, calm_exp, (), pass_index=1, event_id="ev-narc"
    )
    assert len(vibe_narc.shadow_pathology_warnings) == 1
    assert vibe_narc.shadow_pathology_warnings[0].shadow_pathology == "Narcissism"
    assert vibe_narc.coordinates["philosophical/persistence/narcissism"] > 0.50
    chord_narc = SubliminalChordVector.from_right_brain_vibe(vibe_narc)
    assert any(v > 0.0 for v in chord_narc.thinker_vector[8:16])

    # 2. Verify ungrounded Left-Brain Theory shadow pathology triggers dissonance_notes & cadence_rupture_drive
    psychosis_theory_coords = {
        **calm_coords,
        "philosophical/being/agency": 0.92,
        "philosophical/being/existence": 0.16,
    }
    vibe_dissonant, _ = compile_right_brain_vibe(
        birth,
        calm_coords,
        {**calm_exp, "experiential/phase/arousal": 0.72, "experiential/phase/valence": 0.32},
        (),
        pass_index=2,
        event_id="ev-dissonant",
        internal_theory_coords=psychosis_theory_coords,
    )
    assert any(
        "shadow_pathology_dissonance(psychosis:" in note
        for note in vibe_dissonant.self_registry.dissonance_notes
    )
    chord_dissonant = SubliminalChordVector.from_right_brain_vibe(vibe_dissonant)
    assert chord_dissonant.dissonance_pressure > chord_calm.dissonance_pressure

    # 3. Verify compile_chess_posture_chords routes ungrounded piece spikes into thinker_vector[8:16]
    _, chord_b_grounded = compile_chess_posture_chords(
        queen_volition=0.88,
        king_safety=0.65,
        bishop_alignment=0.70,
        knight_disruption=0.80,
        pawn_chain_flow=0.65,
        rook_bastion=0.70,
    )
    _, chord_b_psychosis = compile_chess_posture_chords(
        queen_volition=0.88,
        king_safety=0.18,
        bishop_alignment=0.70,
        knight_disruption=0.80,
        pawn_chain_flow=0.65,
        rook_bastion=0.70,
    )
    assert chord_b_grounded.thinker_vector[8:16] != chord_b_psychosis.thinker_vector[8:16]

    # 4. Verify ThreeLeggedTurnResult exposes sibling_color_balances, shadow_pathology_warnings, and abstraction_profile
    sensors = ThreeBridgeSensors(
        will_cultural=lambda _t, _c=(): {k: narcissism_coords[k] for k in WILL_CULTURAL_LEAVES},
        truth_philosophical=lambda _t, _c=(): {k: narcissism_coords[k] for k in TRUTH_PHILOSOPHICAL_LEAVES},
        contemplation_experiential=lambda _t, _c=(): {k: calm_exp[k] for k in EXPERIENTIAL_SENSOR_LEAVES},
    )
    turn = run_three_legged_turn(
        birth,
        "A sealed yellow bastion shuts out shared continuity.",
        sensors,
        (),
        scene_index=1,
        subliminal_left_brain=None,
    )
    assert len(turn.sibling_color_balances) == 3
    assert len(turn.shadow_pathology_warnings) == 1
    assert turn.shadow_pathology_warnings[0].shadow_pathology == "Narcissism"
    assert turn.abstraction_profile["shadow_pathology_warnings"][0]["shadow_leaf_id"] == "philosophical/persistence/narcissism"

