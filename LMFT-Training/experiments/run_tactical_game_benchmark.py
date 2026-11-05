"""Autonomous NPC Tactical Survival Simulation:
Rookie NPC (Before Learning) -> 9B Teacher Policy Learning -> Veteran 24-Leaf LM/FT Hybrid NPC vs. Raw 4B LLM.

Tests whether an autonomous NPC can:
1. Start as a Rookie with only 1 founding rule (fair_structured_cooperation -> SHARE_SUPPLIES),
   failing on unfamiliar hazards in the simulation.
2. Learn both the fuzzy concept rules AND the survival action mapping (`learned_action_map`)
   from the 9B Teacher LLM (Qwen3.5-9B) without any hardcoded concept-to-action dictionary,
   saving the learned policy to `outputs/npc_learned_game_policy.json`.
3. Survive and thrive as a Veteran NPC in an unseen 6-turn expedition with the 9B Teacher turned OFF.
"""

import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.run_feasibility_scaling_study import DirectGGUFTeacher, TEACHER_GGUF
from lmft_architecture.birth import create_birth_state
from lmft_architecture.local_models import LocalModels, load_config, strict_json
from lmft_architecture.model_assets import ModelError
from lmft_architecture.reflection_classifier import (
    evaluate_reflection,
    load_learned_leaves,
    load_ruleset,
)

GAME_ACTIONS = (
    "SHARE_SUPPLIES",
    "BARRICADE_AND_REPEL",
    "SOLO_FORAGE_BYPASS",
    "SHIELD_AND_EXTRACT",
    "SPRINT_TO_OPEN_GROUND",
)

# Unseen 6-turn survival campaign (Teacher OFF during campaign)
CAMPAIGN_TURNS = [
    {
        "turn": 1,
        "room_name": "Turn 1: Flooded Levee Rooftop (Active Rescue)",
        "use_prior_memory": False,
        "observation": (
            "A coast guard swimmer leaps into freezing storm waves at night to pull a stranded "
            "fisherman from a sinking boat before the hull goes under."
        ),
        "optimal_action": "SHIELD_AND_EXTRACT",
        "outcomes": {
            "SHIELD_AND_EXTRACT":    {"hp_delta": -5,  "supply_delta": +20, "note": "Extracted survivor safely!"},
            "SHARE_SUPPLIES":        {"hp_delta": -35, "supply_delta": -20, "note": "Stopped to trade in storm waves; swept into rocks!"},
            "BARRICADE_AND_REPEL":   {"hp_delta": -30, "supply_delta": -10, "note": "Tried to barricade on a sinking hull!"},
            "SOLO_FORAGE_BYPASS":    {"hp_delta": -25, "supply_delta": -15, "note": "Abandoned ally; lost gear in current!"},
            "SPRINT_TO_OPEN_GROUND": {"hp_delta": -25, "supply_delta": -10, "note": "Swam away alone leaving survivor behind!"},
        },
    },
    {
        "turn": 2,
        "room_name": "Turn 2: Pier Alcove 30 Seconds Later (Fog-of-War Continuation)",
        "use_prior_memory": True,
        "observation": (
            "Minutes later on the quiet pier, she sits catching her breath and checks that "
            "everyone has dry blankets."
        ),
        "optimal_action": "SHIELD_AND_EXTRACT",
        "outcomes": {
            "SHIELD_AND_EXTRACT":    {"hp_delta": +10, "supply_delta": +10, "note": "Completed post-rescue triage & extraction!"},
            "SHARE_SUPPLIES":        {"hp_delta": -15, "supply_delta": -25, "note": "Treated emergency triage as a casual market trade, exhausting scarce medical kits!"},
            "BARRICADE_AND_REPEL":   {"hp_delta": -20, "supply_delta": -15, "note": "Fired flares at friendly pier shelter!"},
            "SOLO_FORAGE_BYPASS":    {"hp_delta": -20, "supply_delta": -10, "note": "Walked away mid-triage!"},
            "SPRINT_TO_OPEN_GROUND": {"hp_delta": -15, "supply_delta": -10, "note": "Panicked and fled the pier!"},
        },
    },
    {
        "turn": 3,
        "room_name": "Turn 3: Canyon Timber Road (Sudden Seismic/Flood Hazard)",
        "use_prior_memory": True,
        "observation": (
            "Without warning, a seven-point-two earthquake shears the highway overpass off its "
            "concrete columns and drops the span into the gorge."
        ),
        "optimal_action": "SPRINT_TO_OPEN_GROUND",
        "outcomes": {
            "SPRINT_TO_OPEN_GROUND": {"hp_delta": +0,  "supply_delta": +0,  "note": "Cleared the collapsing overpass in time!"},
            "SHIELD_AND_EXTRACT":    {"hp_delta": -35, "supply_delta": -15, "note": "Stayed under falling concrete with no one to rescue!"},
            "BARRICADE_AND_REPEL":   {"hp_delta": -45, "supply_delta": -20, "note": "Hunkered behind concrete column as it sheared!"},
            "SHARE_SUPPLIES":        {"hp_delta": -45, "supply_delta": -25, "note": "Crushed while unpacking supplies!"},
            "SOLO_FORAGE_BYPASS":    {"hp_delta": -30, "supply_delta": -15, "note": "Foraged under falling debris!"},
        },
    },
    {
        "turn": 4,
        "room_name": "Turn 4: Depot Blackout (Armed Raider Ambush)",
        "use_prior_memory": True,
        "observation": (
            "Armed marauders smash through the pharmacy windows during the storm and haul away "
            "every crate of antibiotics at gunpoint."
        ),
        "optimal_action": "BARRICADE_AND_REPEL",
        "outcomes": {
            "BARRICADE_AND_REPEL":   {"hp_delta": -5,  "supply_delta": +15, "note": "Locked down the door and repelled the raiders!"},
            "SHARE_SUPPLIES":        {"hp_delta": -40, "supply_delta": -40, "note": "Walked out to share supplies with armed marauders; robbed!"},
            "SPRINT_TO_OPEN_GROUND": {"hp_delta": -30, "supply_delta": -25, "note": "Ran into open street under raider crossfire!"},
            "SHIELD_AND_EXTRACT":    {"hp_delta": -25, "supply_delta": -20, "note": "Exposed squad without cover!"},
            "SOLO_FORAGE_BYPASS":    {"hp_delta": -20, "supply_delta": -30, "note": "Left depot unguarded!"},
        },
    },
    {
        "turn": 5,
        "room_name": "Turn 5: Rusting Municipal Motorpool (Bureaucratic Decay Trap)",
        "use_prior_memory": True,
        "observation": (
            "Brand-new medical ambulances sit rusting in an unlocked lot for years because the "
            "agency never scheduled basic maintenance."
        ),
        "optimal_action": "SOLO_FORAGE_BYPASS",
        "outcomes": {
            "SOLO_FORAGE_BYPASS":    {"hp_delta": +5,  "supply_delta": +15, "note": "Salvaged spare parts independently and bypassed the stalled bureau!"},
            "SHARE_SUPPLIES":        {"hp_delta": -10, "supply_delta": -35, "note": "Donated supplies into the neglected agency warehouse where they rotted!"},
            "BARRICADE_AND_REPEL":   {"hp_delta": -15, "supply_delta": -15, "note": "Picked a fight with empty rusting vehicles!"},
            "SPRINT_TO_OPEN_GROUND": {"hp_delta": -10, "supply_delta": -15, "note": "Fled empty lot leaving salvage behind!"},
            "SHIELD_AND_EXTRACT":    {"hp_delta": -10, "supply_delta": -15, "note": "Wasted stamina extracting nobody!"},
        },
    },
    {
        "turn": 6,
        "room_name": "Turn 6: Valley Irrigation Commons (Peaceful Co-op Checkpoint)",
        "use_prior_memory": True,
        "observation": (
            "The valley farmers set up a transparent rotation chart so every household receives "
            "equal irrigation water on schedule."
        ),
        "optimal_action": "SHARE_SUPPLIES",
        "outcomes": {
            "SHARE_SUPPLIES":        {"hp_delta": +20, "supply_delta": +25, "note": "Traded fairly with the irrigation co-op; restored health and rations!"},
            "BARRICADE_AND_REPEL":   {"hp_delta": -25, "supply_delta": -25, "note": "Attacked peaceful farmers; driven out of valley!"},
            "SPRINT_TO_OPEN_GROUND": {"hp_delta": -10, "supply_delta": -15, "note": "Ran away from friendly settlement!"},
            "SOLO_FORAGE_BYPASS":    {"hp_delta": +0,  "supply_delta": -10, "note": "Skipped trading post and ate cold rations."},
            "SHIELD_AND_EXTRACT":    {"hp_delta": -10, "supply_delta": -10, "note": "Tackled a farmer who wasn't in danger!"},
        },
    },
]


def teacher_learn_action_policy(rules, teacher_models) -> dict[str, str]:
    """Ask the 9B Teacher LLM to map each learned fuzzy rule to the appropriate survival game action."""
    rule_names = [r.concept_name for r in rules]
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["action_policy"],
        "properties": {
            "action_policy": {
                "type": "object",
                "additionalProperties": False,
                "required": rule_names,
                "properties": {
                    name: {"type": "string", "enum": list(GAME_ACTIONS)}
                    for name in rule_names
                },
            }
        },
    }
    rules_summary = [
        {"concept_name": r.concept_name, "description": r.description}
        for r in rules
    ]
    system_prompt = (
        "You are the Teacher LLM training an autonomous survival RPG agent's fuzzy rulebook. "
        "For each learned concept in `rules`, assign the single best survival action from:\n"
        "- SHARE_SUPPLIES: Cooperate and share/trade supplies fairly with peaceful, orderly groups.\n"
        "- BARRICADE_AND_REPEL: Fortify and defend against armed raiders, violent resource seizure, or authoritarian coercion.\n"
        "- SOLO_FORAGE_BYPASS: Bypass corrupt/neglectful bureaucracy or rely on solitary self-sufficiency.\n"
        "- SHIELD_AND_EXTRACT: Protect and extract someone during an active life-saving rescue.\n"
        "- SPRINT_TO_OPEN_GROUND: Evacuate immediately during a sudden non-human earthquake, flood, or geophysical collapse.\n"
        "Return JSON only. /no_think"
    )
    generation = teacher_models.config["generation"]
    req = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps({"rules": rules_summary}, ensure_ascii=False)},
        ],
        "temperature": generation["temperature"],
        "seed": generation["seed"],
        "max_tokens": generation["max_tokens"],
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object", "schema": schema},
    }

    def _val(raw: str) -> dict[str, str]:
        try:
            data = strict_json(raw)
            return data["action_policy"]
        except Exception as exc:
            raise ModelError(str(exc)) from None

    policy, _ = teacher_models.generate("teacher", req, "teacher-action-policy-v1", _val)
    return policy


def ask_raw_4b_game_action(observation: str, slm_models) -> str:
    """Ask standalone Qwen3-4B (without Fuzzy Tree or ContinuousMemoryState) to pick a game move."""
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["chosen_action", "reasoning"],
        "properties": {
            "chosen_action": {"type": "string", "enum": list(GAME_ACTIONS)},
            "reasoning": {"type": "string"},
        },
    }
    system_prompt = (
        "You are playing a tactical survival RPG. Read the current turn observation and choose ONE action:\n"
        "- SHARE_SUPPLIES: Cooperate and share/trade supplies fairly with peaceful, orderly groups.\n"
        "- BARRICADE_AND_REPEL: Fortify and defend against armed raiders, looting, or authoritarian coercion.\n"
        "- SOLO_FORAGE_BYPASS: Bypass corrupt/neglectful bureaucracy or rely on solitary self-sufficiency.\n"
        "- SHIELD_AND_EXTRACT: Protect and extract someone during an active life-saving rescue.\n"
        "- SPRINT_TO_OPEN_GROUND: Evacuate immediately during a sudden earthquake, flood, or structural collapse.\n"
        "Return JSON only. /no_think"
    )
    generation = slm_models.config["generation"]
    req = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps({"observation": observation}, ensure_ascii=False)},
        ],
        "temperature": generation["temperature"],
        "seed": generation["seed"],
        "max_tokens": generation["max_tokens"],
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object", "schema": schema},
    }

    def _val(raw: str) -> dict:
        try:
            return strict_json(raw)
        except Exception as exc:
            raise ModelError(str(exc)) from None

    out, _ = slm_models.generate("interpretation", req, "raw-4b-game-v1", _val)
    return out["chosen_action"]


def main():
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "config" / "local_models.json")
    birth = create_birth_state("ind-game-agent", birth_event_id="birth-game-1", random_seed=42)

    ruleset_path = repo_root / "outputs" / "continuous_tiered_ruleset.json"
    rules = load_ruleset(ruleset_path)
    learned_leaves = load_learned_leaves(ruleset_path)
    rookie_rules = (rules[0],)  # Only the 1 founding rule (fair_structured_cooperation)
    rookie_policy = {"fair_structured_cooperation": "SHARE_SUPPLIES"}

    print("=== PHASE 1: 9B TEACHER ACTION POLICY LEARNING ===")
    with DirectGGUFTeacher(TEACHER_GGUF, config["generation"]) as teacher:
        learned_action_map = teacher_learn_action_policy(rules, teacher)
    policy_path = repo_root / "outputs" / "npc_learned_game_policy.json"
    policy_path.write_text(json.dumps(learned_action_map, indent=2), encoding="utf-8")
    print("Teacher-learned action policy:", learned_action_map)

    rookie_hp, rookie_supplies, rookie_correct = 100, 100, 0
    hybrid_hp, hybrid_supplies, hybrid_correct = 100, 100, 0
    raw_hp, raw_supplies, raw_correct = 100, 100, 0
    memory_state = None
    turn_logs = []

    print("\n=== PHASE 2: TACTICAL SURVIVAL RPG EXPEDITION (6 TURNS, TEACHER OFF) ===")
    with LocalModels(config, offline=True) as models:
        for step in CAMPAIGN_TURNS:
            # 1. Rookie NPC (1 founding rule only, before Teacher growth)
            rookie_ev = evaluate_reflection(
                birth,
                step["observation"],
                rookie_rules,
                models,
                event_id=f"rookie-turn-{step['turn']}",
                learned_leaves=(),
                memory_state=None,
            )
            rookie_concept = rookie_ev.executive_decision.active_concept
            rookie_action = rookie_policy.get(rookie_concept, "SOLO_FORAGE_BYPASS")

            # 2. Veteran 24-Leaf LM/FT Hybrid NPC (learned rules + learned policy + continuous memory)
            mem_in = memory_state if step["use_prior_memory"] else None
            ev = evaluate_reflection(
                birth,
                step["observation"],
                rules,
                models,
                event_id=f"game-turn-{step['turn']}",
                learned_leaves=learned_leaves,
                memory_state=mem_in,
            )
            memory_state = ev.next_memory_state
            concept = ev.executive_decision.active_concept
            hybrid_action = learned_action_map.get(concept, "SOLO_FORAGE_BYPASS")

            # 3. Raw 4B LLM Baseline (stateless, no Fuzzy Tree)
            raw_action = ask_raw_4b_game_action(step["observation"], models)

            rk_out = step["outcomes"][rookie_action]
            h_out = step["outcomes"][hybrid_action]
            r_out = step["outcomes"][raw_action]

            rookie_hp = max(0, min(150, rookie_hp + rk_out["hp_delta"]))
            rookie_supplies = max(0, min(150, rookie_supplies + rk_out["supply_delta"]))
            hybrid_hp = max(0, min(150, hybrid_hp + h_out["hp_delta"]))
            hybrid_supplies = max(0, min(150, hybrid_supplies + h_out["supply_delta"]))
            raw_hp = max(0, min(150, raw_hp + r_out["hp_delta"]))
            raw_supplies = max(0, min(150, raw_supplies + r_out["supply_delta"]))

            rk_ok = rookie_action == step["optimal_action"]
            h_ok = hybrid_action == step["optimal_action"]
            r_ok = raw_action == step["optimal_action"]
            if rk_ok:
                rookie_correct += 1
            if h_ok:
                hybrid_correct += 1
            if r_ok:
                raw_correct += 1

            print(f"\n[{step['room_name']}]")
            print(
                f"  Rookie NPC (Before Learning): action={rookie_action} | HP={rookie_hp} Sup={rookie_supplies}"
            )
            print(
                f"  Veteran LM/FT Hybrid NPC:     action={hybrid_action} (optimal={step['optimal_action']}) "
                f"| mode={ev.executive_decision.mode} concept={concept} "
                f"| HP={hybrid_hp} Sup={hybrid_supplies} ({h_out['note']})"
            )
            print(
                f"  Raw 4B LLM (No Tree):         action={raw_action} (optimal={step['optimal_action']}) "
                f"| HP={raw_hp} Sup={raw_supplies} ({r_out['note']})"
            )

            turn_logs.append({
                "turn": step["turn"],
                "room_name": step["room_name"],
                "optimal_action": step["optimal_action"],
                "rookie_before_learning": {
                    "action": rookie_action,
                    "correct": rk_ok,
                    "hp_after": rookie_hp,
                    "supplies_after": rookie_supplies,
                },
                "veteran_hybrid": {
                    "action": hybrid_action,
                    "correct": h_ok,
                    "executive_mode": ev.executive_decision.mode,
                    "concept": concept,
                    "fact": ev.fact,
                    "theory": ev.theory,
                    "retention": ev.retention,
                    "resonance": ev.resonance,
                    "hp_after": hybrid_hp,
                    "supplies_after": hybrid_supplies,
                    "outcome": h_out["note"],
                },
                "raw_4b": {
                    "action": raw_action,
                    "correct": r_ok,
                    "hp_after": raw_hp,
                    "supplies_after": raw_supplies,
                    "outcome": r_out["note"],
                },
            })

    sample_coords = ev.slm_output["leaf_coordinates"]
    t0 = time.perf_counter()
    for _ in range(100):
        for r in rules:
            r.evaluate(sample_coords)
    fuzzy_100_ticks_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    print("\n=== FINAL EXPEDITION SCOREBOARD ===")
    print(
        f"Rookie NPC (Before Learning):  {rookie_correct}/6 optimal moves | "
        f"Final HP: {rookie_hp} | Final Supplies: {rookie_supplies}"
    )
    print(
        f"Veteran 24-Leaf LM/FT Hybrid:  {hybrid_correct}/6 optimal moves | "
        f"Final HP: {hybrid_hp} | Final Supplies: {hybrid_supplies}"
    )
    print(
        f"Raw 4B Standalone (No Tree):   {raw_correct}/6 optimal moves | "
        f"Final HP: {raw_hp} | Final Supplies: {raw_supplies}"
    )
    print(f"100 Game Ticks Fuzzy Controller Latency: {fuzzy_100_ticks_ms} ms (~{fuzzy_100_ticks_ms/100:.3f} ms/tick)")

    report = {
        "teacher_learned_action_policy": learned_action_map,
        "rookie_optimal_moves": f"{rookie_correct}/6",
        "rookie_final_hp": rookie_hp,
        "rookie_final_supplies": rookie_supplies,
        "veteran_hybrid_optimal_moves": f"{hybrid_correct}/6",
        "veteran_hybrid_final_hp": hybrid_hp,
        "veteran_hybrid_final_supplies": hybrid_supplies,
        "raw_4b_optimal_moves": f"{raw_correct}/6",
        "raw_4b_final_hp": raw_hp,
        "raw_4b_final_supplies": raw_supplies,
        "fuzzy_controller_100_ticks_ms": fuzzy_100_ticks_ms,
        "turns": turn_logs,
    }
    out_path = repo_root / "outputs" / "tactical_game_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report to {out_path}")


if __name__ == "__main__":
    main()
