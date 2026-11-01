"""Unit tests for the FTL Bridge Simulator (`src/lmft_lab/ftl_bridge_sim.py`)."""

import pytest

from lmft_lab.ftl_bridge_sim import (
    FTLShipState,
    allocate_ship_power_fuzzy,
    decay_memory_by_seconds,
    resolve_beacon_outcome,
)
from lmft_lab.reflection_classifier import ContinuousMemoryState


def test_clock_time_half_life_retains_memory_across_60hz_frames():
    initial = ContinuousMemoryState(
        step_count=1,
        retained_coordinates={"cultural/justice/fairness": 1.0, "philosophical/structure/order": 0.8},
        last_matched_concept="fair_structured_cooperation",
        retention_decay=1.0,
    )
    # Simulate 1,800 frames at 60 Hz (= 30.0 real seconds, which is 1 half-life of 30s)
    state = initial
    for _ in range(1800):
        state = decay_memory_by_seconds(state, dt_seconds=1.0 / 60.0, half_life_seconds=30.0)

    # After 1,800 frames (30 seconds), 1.0 should decay to ~0.50 (NOT 0.0!)
    assert state is not None
    assert pytest.approx(state.retained_coordinates["cultural/justice/fairness"], abs=1e-2) == 0.50
    assert pytest.approx(state.retained_coordinates["philosophical/structure/order"], abs=1e-2) == 0.40


def test_allocate_ship_power_fuzzy_conserves_8_reactor_bars():
    ship = FTLShipState(reactor_total=8)

    # 1. Hostile combat: high power + chaos -> Shields and Weapons get priority
    combat_ship, ms = allocate_ship_power_fuzzy(
        ship,
        {"cultural/efficiency/power": 0.95, "philosophical/structure/chaos": 0.90},
        "RAISE_SHIELDS_AND_RETURN_FIRE",
    )
    assert (
        combat_ship.power_shields
        + combat_ship.power_engines
        + combat_ship.power_weapons
        + combat_ship.power_medbay
        == 8
    )
    assert combat_ship.power_weapons >= 3
    assert combat_ship.power_shields >= 2
    assert ms < 5.0

    # 2. Environmental collapse (solar flare / asteroid field): Shields + Engines get priority
    evade_ship, _ = allocate_ship_power_fuzzy(
        ship,
        {"philosophical/structure/tectonic_collapse": 0.95, "philosophical/structure/chaos": 0.85},
        "MAX_ENGINES_EMERGENCY_JUMP",
    )
    assert (
        evade_ship.power_shields
        + evade_ship.power_engines
        + evade_ship.power_weapons
        + evade_ship.power_medbay
        == 8
    )
    assert evade_ship.power_engines >= 3


def test_resolve_beacon_outcome_penalizes_mistakes():
    ship = FTLShipState(hull=30, scrap=30, fuel=10)
    after_mistake, outcome = resolve_beacon_outcome(
        ship,
        chosen_action="OPEN_HAIL_AND_TRADE",
        optimal_action="RAISE_SHIELDS_AND_RETURN_FIRE",
    )
    assert outcome["status"] == "TACTICAL_MISTAKE"
    assert after_mistake.hull == 24
    assert after_mistake.scrap == 25
    assert after_mistake.fuel == 9
