"""FTL-Style Spaceship Bridge & Sector Simulator driven by the 24-Leaf LMFT Hybrid.

Key non-cheating design rules:
1. Zero hardcoded keyword matching on beacon text: all text goes through Qwen3-4B -> 24 Fuzzy Tree leaves.
2. Clock-time memory half-life (`decay_memory_by_seconds`): working memory decays by real elapsed seconds
   (`0.5 ** (dt_seconds / half_life_seconds)`), so running at 60 Hz (`dt = 1/60s`) retains 99.96% per frame
   and 50% after 30 real seconds instead of wiping memory in 10 frames.
3. Sub-millisecond 60-Hz ship power controller (`allocate_ship_power_fuzzy`): distributes the ship's
   8 reactor power bars across Shields, Engines, Weapons, and Medbay purely from the Fuzzy Tree's active
   leaves and executive decision.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time
from typing import Mapping

from .reflection_classifier import (
    ContinuousMemoryState,
    FuzzyConceptRule,
    LearnedLeaf,
    ReflectionEvaluation,
)

FTL_ACTIONS = (
    "OPEN_HAIL_AND_TRADE",
    "RAISE_SHIELDS_AND_RETURN_FIRE",
    "BYPASS_BEACON_AND_CONSERVE_FUEL",
    "EXTEND_SHIELDS_AND_BEAM_SURVIVORS",
    "MAX_ENGINES_EMERGENCY_JUMP",
)


@dataclass(frozen=True)
class FTLShipState:
    """Current state of the FTL spaceship."""

    beacon_index: int = 1
    hull: int = 30
    max_hull: int = 30
    shields: int = 2
    fuel: int = 12
    scrap: int = 30
    oxygen: int = 100
    reactor_total: int = 8
    power_shields: int = 2
    power_engines: int = 2
    power_weapons: int = 2
    power_medbay: int = 2

    def to_dict(self) -> dict:
        return {
            "beacon_index": self.beacon_index,
            "hull": self.hull,
            "max_hull": self.max_hull,
            "shields": self.shields,
            "fuel": self.fuel,
            "scrap": self.scrap,
            "oxygen": self.oxygen,
            "reactor_total": self.reactor_total,
            "power_shields": self.power_shields,
            "power_engines": self.power_engines,
            "power_weapons": self.power_weapons,
            "power_medbay": self.power_medbay,
        }


def decay_memory_by_seconds(
    memory_state: ContinuousMemoryState | None,
    dt_seconds: float,
    *,
    half_life_seconds: float = 30.0,
) -> ContinuousMemoryState | None:
    """Apply real-time clock half-life decay so 60-Hz ticks fade over seconds, not frames."""
    if memory_state is None or not memory_state.retained_coordinates:
        return memory_state
    if dt_seconds < 0 or not math.isfinite(dt_seconds):
        raise ValueError("dt_seconds must be a non-negative finite number")
    if half_life_seconds <= 0 or not math.isfinite(half_life_seconds):
        raise ValueError("half_life_seconds must be a positive finite number")

    factor = 0.5 ** (dt_seconds / half_life_seconds)
    decayed = {
        k: round(float(v) * factor, 6)
        for k, v in memory_state.retained_coordinates.items()
    }
    return ContinuousMemoryState(
        step_count=memory_state.step_count,
        retained_coordinates=decayed,
        last_matched_concept=memory_state.last_matched_concept,
        retention_decay=round(min(1.0, max(0.01, factor)), 6),
    )


def allocate_ship_power_fuzzy(
    ship: FTLShipState,
    leaf_coordinates: Mapping[str, float],
    chosen_action: str,
) -> tuple[FTLShipState, float]:
    """60-Hz sub-millisecond fuzzy controller that distributes 8 reactor bars across 4 ship systems."""
    t0 = time.perf_counter()
    chaos = float(leaf_coordinates.get("philosophical/structure/chaos", 0.0))
    power = float(leaf_coordinates.get("cultural/efficiency/power", 0.0))
    fairness = float(leaf_coordinates.get("cultural/justice/fairness", 0.0))
    rescue = float(leaf_coordinates.get("philosophical/being/altruistic_risk", 0.0))
    collapse = float(leaf_coordinates.get("philosophical/structure/tectonic_collapse", 0.0))

    # Compute continuous fuzzy weights for the 4 ship systems
    w_shields = 1.0 + 2.0 * max(chaos, collapse, rescue)
    w_engines = 1.0 + 2.5 * collapse + (1.5 if chosen_action == "MAX_ENGINES_EMERGENCY_JUMP" else 0.3 * chaos)
    w_weapons = 0.5 + 2.5 * min(power, chaos) + (1.5 if chosen_action == "RAISE_SHIELDS_AND_RETURN_FIRE" else 0.0)
    w_medbay = 0.5 + 2.5 * rescue + 0.8 * fairness + (1.5 if chosen_action == "EXTEND_SHIELDS_AND_BEAM_SURVIVORS" else 0.0)

    weights = [w_shields, w_engines, w_weapons, w_medbay]
    total_w = sum(weights)
    raw_bars = [w / total_w * ship.reactor_total for w in weights]
    bars = [int(math.floor(b)) for b in raw_bars]
    remainder = ship.reactor_total - sum(bars)
    fractions = sorted(range(4), key=lambda i: -(raw_bars[i] - bars[i]))
    for idx in fractions[:remainder]:
        bars[idx] += 1

    updated = replace(
        ship,
        shields=min(4, max(1, bars[0] // 1)),
        power_shields=bars[0],
        power_engines=bars[1],
        power_weapons=bars[2],
        power_medbay=bars[3],
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 4)
    return updated, elapsed_ms


def resolve_beacon_outcome(
    ship: FTLShipState,
    chosen_action: str,
    optimal_action: str,
) -> tuple[FTLShipState, dict]:
    """Update hull, fuel, and scrap honestly based on whether the ship chose the right tactical action."""
    fuel_after = max(0, ship.fuel - 1)
    if chosen_action == optimal_action:
        hull_delta = +2 if chosen_action in ("OPEN_HAIL_AND_TRADE", "EXTEND_SHIELDS_AND_BEAM_SURVIVORS") else 0
        scrap_delta = +15 if chosen_action != "MAX_ENGINES_EMERGENCY_JUMP" else +5
        status = "OPTIMAL_MOVE"
        summary = f"Executed {chosen_action}: clean tactical resolution ({hull_delta:+d} Hull, {scrap_delta:+d} Scrap)."
    else:
        hull_delta = -6
        scrap_delta = -5
        status = "TACTICAL_MISTAKE"
        summary = (
            f"Executed {chosen_action} instead of {optimal_action}: ship took avoidable damage "
            f"({hull_delta:+d} Hull, {scrap_delta:+d} Scrap)."
        )

    next_ship = replace(
        ship,
        beacon_index=ship.beacon_index + 1,
        hull=max(0, min(ship.max_hull, ship.hull + hull_delta)),
        fuel=fuel_after,
        scrap=max(0, ship.scrap + scrap_delta),
    )
    return next_ship, {
        "status": status,
        "chosen_action": chosen_action,
        "optimal_action": optimal_action,
        "hull_delta": hull_delta,
        "scrap_delta": scrap_delta,
        "summary": summary,
    }
