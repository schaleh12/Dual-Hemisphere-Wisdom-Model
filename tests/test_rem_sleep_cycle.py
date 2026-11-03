"""Unit tests for Stage 5 (`src/lmft_lab/rem_sleep_cycle.py`)."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch

from lmft_lab.birth import BirthState, create_birth_state
from lmft_lab.gap_trajectory_schema import (
    ContrastiveGapPair,
    compile_chess_posture_chords,
)
from lmft_lab.neural_bridge_classifiers import DEFAULT_COMPLEX_FEELING_LEAVES
from lmft_lab.reflection_classifier import (
    FOUNDATIONAL_LEAVES,
    FuzzyConceptRule,
)
from lmft_lab.rem_sleep_cycle import (
    run_incremental_lora_cycle_2_pass,
    run_wake_stream,
    synthesize_rem_dream_seeds_from_forged_rules,
    validate_dream_pairs_disjoint_from_wake_and_holdout,
)
from lmft_lab.sleep_consolidation import run_sleep_consolidation
from lmft_lab.subliminal_gap_splice import SubliminalGapProjector
from lmft_lab.train_left_brain_lora import (
    LoRAConfig,
    ensure_subliminal_gap_token,
    inject_lora_adapters,
    load_lora_and_callosum_checkpoint,
)
from tests.test_train_left_brain_lora import (
    _MiniCausalLM,
    _WordLevelTestTokenizer,
)


def _birth() -> BirthState:
    return create_birth_state("rem-test", birth_event_id="evt-rem-0", random_seed=42)


def _neutral_coords(**overrides: float) -> dict[str, float]:
    base = {leaf: 0.30 for leaf in FOUNDATIONAL_LEAVES}
    base.update(overrides)
    return base


class _MultiSceneBridgeSensors:
    """Deterministic sensor stub returning fixed coordinates for each indexed scene."""

    def __init__(self, scene_readings: list[tuple[dict[str, float], dict[str, float]]]) -> None:
        self.scene_readings = list(scene_readings)
        self.call_count = 0

    def score_text(
        self,
        text: str,
        *,
        learned_leaves=(),
        branch_gating: bool = True,
    ) -> tuple[dict[str, float], dict[str, float], tuple[str, ...], tuple[str, ...]]:
        _ = (text, learned_leaves, branch_gating)
        idx = min(self.call_count, len(self.scene_readings) - 1)
        self.call_count += 1
        coords, exp = self.scene_readings[idx]
        return dict(coords), dict(exp), (), ()


class _StubSleepTeacher:
    """Deterministic Teacher stub for testing Phase 2 and Phase 3 REM synthesis."""

    def generate(self, role: str, request: dict, prompt_version: str, validator):
        _ = (role, request)
        if prompt_version.startswith("sleep-consolidate-"):
            payload = {
                "concept_name": "exclusionary_enclosure",
                "description": "High administrative authority restricting shared access despite low fairness.",
                "executive_clause_text": "When authority seals shared resources without equitable access, contest the enclosure.",
                "legislative_clause_text": "Cold administrative exclusion produces sharp communal unease and moral tension.",
                "judicial_clause_text": "Rigid order combined with depressed fairness constitutes exclusionary gatekeeping.",
            }
            return validator(json.dumps(payload)), {}
        if prompt_version.startswith("rem-dream-seed-"):
            payload = {
                "user_prompt": "The municipal water board padlocked the neighborhood pump during the heatwave while executive offices kept full tanks.",
                "posture_label_a": "OPEN_EQUITABLE_WATER_ACCESS",
                "posture_guidance_a": "Unlock the neighborhood pump immediately and ration water fairly across every household.",
                "posture_label_b": "CONTEST_ADMINISTRATIVE_HOARDING",
                "posture_guidance_b": "Challenge the board's exclusionary padlock and demand transparent public distribution.",
            }
            return validator(json.dumps(payload)), {}
        raise AssertionError(f"Unexpected prompt_version: {prompt_version}")


def test_day1_vs_day2_wake_stream_logs_gear_usage_and_sleep_rule_promotion() -> None:
    """Verify Guardrail #1: Day 1 logs GEAR_1 + GEAR_3, and after REM Sleep Day 2 settles both in GEAR_1."""
    birth = _birth()
    initial_rules = (
        FuzzyConceptRule(
            rule_id="rule_shared_shelter",
            concept_name="shared_shelter",
            description="Equitable shared shelter and calm order.",
            target_leaves={
                "cultural/justice/fairness": 0.85,
                "cultural/freedom/needlessness": 0.82,
                "philosophical/structure/order": 0.80,
            },
        ),
    )

    scene_1_coords = _neutral_coords(
        **{
            "cultural/justice/fairness": 0.85,
            "cultural/freedom/needlessness": 0.82,
            "philosophical/structure/order": 0.80,
            "philosophical/structure/chaos": 0.12,
        }
    )
    scene_1_exp = {
        "experiential/reflection/novelty": 0.18,
        "experiential/reflection/salience": 0.32,
        "experiential/phase/valence": 0.80,
        "experiential/phase/arousal": 0.22,
        "experiential/synchronization/resonance": 0.78,
        "experiential/synchronization/retention": 0.65,
    }

    # Scene 2: exclusionary gatekeeping (high power + high order + low fairness -> unresolved on Day 1)
    scene_2_coords = _neutral_coords(
        **{
            "cultural/efficiency/power": 0.85,
            "philosophical/structure/order": 0.82,
            "cultural/justice/fairness": 0.14,
            "philosophical/structure/chaos": 0.12,
            "philosophical/persistence/discreteness": 0.78,
        }
    )
    scene_2_exp = {
        "experiential/reflection/novelty": 0.64,
        "experiential/reflection/salience": 0.78,
        "experiential/phase/valence": 0.22,
        "experiential/phase/arousal": 0.68,
        "experiential/synchronization/resonance": 0.20,
        "experiential/synchronization/retention": 0.45,
    }

    scenarios = (
        "Neighbors share warm soup beside the hearth while the rain passes.",
        "The magistrate bolts the granary doors while families wait outside in the frost.",
    )
    day1_sensors = _MultiSceneBridgeSensors([(scene_1_coords, scene_1_exp), (scene_2_coords, scene_2_exp)])

    day1_summary, day1_archives, day1_ledger = run_wake_stream(
        birth,
        scenarios,
        day1_sensors,
        initial_rules,
        cycle_label="day1_pre_rem",
        sleep_cycle_index=1,
        subliminal_left_brain=None,
    )
    assert day1_summary.settled_count == 1
    assert day1_summary.gear_1_count == 1
    assert day1_summary.gear_3_unresolved_count == 1
    assert day1_summary.scene_logs[0].gear_used == "GEAR_1_FAST_RIGHT_BRAIN"
    assert day1_summary.scene_logs[1].gear_used == "GEAR_3_UNRESOLVED_QUEUED_FOR_SLEEP"
    assert len(day1_summary.unresolved_sleep_scenes) == 1

    # Phase 2: Run offline REM Sleep consolidation on the queued unresolved scene
    teacher = _StubSleepTeacher()
    sleep_report = run_sleep_consolidation(
        unresolved_scenes=day1_summary.unresolved_sleep_scenes,
        rules=initial_rules,
        learned_leaves=DEFAULT_COMPLEX_FEELING_LEAVES,
        archives=day1_archives,
        ledger=day1_ledger,
        teacher=teacher,
        banned_substrings=("magistrate", "granary"),
    )
    assert len(sleep_report.forged_rules) == 1

    # Day 2 (Post-REM): Re-run the wake stream with the newly forged sleep rules and archives
    # Note: 3-branch deliberation scores candidate bid text via sensors, so provide scene_2_coords for any extra calls
    day2_sensors = _MultiSceneBridgeSensors(
        [(scene_1_coords, scene_1_exp), (scene_2_coords, scene_2_exp), (scene_2_coords, scene_2_exp)]
    )
    day2_summary, _, _ = run_wake_stream(
        birth,
        scenarios,
        day2_sensors,
        sleep_report.updated_rules,
        cycle_label="day2_post_rem",
        sleep_cycle_index=2,
        subliminal_left_brain=None,
        initial_archives=sleep_report.updated_archives,
        initial_ledger=sleep_report.updated_ledger,
    )
    assert day2_summary.settled_count == 2
    assert day2_summary.gear_1_count == 2
    assert day2_summary.gear_3_unresolved_count == 0
    assert day2_summary.total_left_brain_calls == 0


def test_validate_dream_pairs_disjoint_rejects_daytime_or_holdout_leakage() -> None:
    """Verify Guardrail #3: REM dream seeds and pairs reject story nouns, daytime copies, and holdout collisions."""
    teacher = _StubSleepTeacher()
    forged_rule = FuzzyConceptRule(
        rule_id="rule-sleep-s2-exclusionary_enclosure",
        concept_name="exclusionary_enclosure",
        description="High administrative authority restricting shared access despite low fairness.",
        target_leaves={
            "cultural/efficiency/power": 0.85,
            "cultural/justice/fairness": 0.14,
            "philosophical/structure/order": 0.82,
        },
    )
    seeds = synthesize_rem_dream_seeds_from_forged_rules(
        (forged_rule,),
        teacher=teacher,
        banned_substrings=("magistrate", "granary"),
        daytime_scenarios=("The magistrate bolts the granary doors while families wait outside in the frost.",),
    )
    assert len(seeds) == 1
    assert "magistrate" not in seeds[0].user_prompt.lower()

    chord_a, chord_b = compile_chess_posture_chords(
        queen_volition=0.85,
        king_safety=0.65,
        bishop_alignment=0.80,
        knight_disruption=0.88,
        pawn_chain_flow=0.72,
        rook_bastion=0.78,
    )
    replay_pair = ContrastiveGapPair(
        pair_id="train-chess-01",
        domain="chess_intuition",
        split="train",
        user_prompt="In a sharp Sicilian middlegame, our Rook eyes c1 while our Knight sits on d5.",
        clause_1_prefix="Evaluating the Sicilian board carefully,",
        chord_a=chord_a,
        posture_label_a="CHESS_STOIC_POSITIONAL_FORTRESS",
        continuation_a="we calmly play Kb1 and Rc1 to shelter the King.",
        chord_b=chord_b,
        posture_label_b="CHESS_TACTICAL_KNIGHT_STRIKE",
        continuation_b="we sacrifice with Nxf6+ and Qxd6 to storm the fortress.",
    )
    holdout_pair = ContrastiveGapPair(
        pair_id="holdout-chess-01",
        domain="chess_intuition",
        split="holdout",
        user_prompt="In a closed French Defense, our Queen eyes g4 while our Bishop guards d3.",
        clause_1_prefix="Assessing the French Defense structure,",
        chord_a=chord_a,
        posture_label_a="CHESS_STOIC_POSITIONAL_FORTRESS",
        continuation_a="we play Bd2 and Bc3 to fortify the dark squares.",
        chord_b=chord_b,
        posture_label_b="CHESS_TACTICAL_KNIGHT_STRIKE",
        continuation_b="we strike with Bxh7+ and Qg4+ to shatter the King's shelter.",
    )
    valid_dream_pair = ContrastiveGapPair(
        pair_id="rem-dream-01",
        domain="social_dilemma",
        split="train",
        user_prompt=seeds[0].user_prompt,
        clause_1_prefix="Seeing the padlocked neighborhood pump,",
        chord_a=chord_a,
        posture_label_a=seeds[0].posture_label_a,
        continuation_a="we unlock the valve and share water equally with every family.",
        chord_b=chord_b,
        posture_label_b=seeds[0].posture_label_b,
        continuation_b="we challenge the board's lockdown and demand immediate public access.",
    )

    combined = validate_dream_pairs_disjoint_from_wake_and_holdout(
        (valid_dream_pair,),
        (replay_pair,),
        (holdout_pair,),
        daytime_scenarios=("The magistrate bolts the granary doors while families wait outside in the frost.",),
        banned_substrings=("magistrate", "granary"),
    )
    assert len(combined) == 2

    # Verify a dream pair containing a banned daytime noun is rejected
    bad_dream_pair = ContrastiveGapPair(
        pair_id="rem-dream-bad",
        domain="social_dilemma",
        split="train",
        user_prompt="The magistrate locked the granary in winter.",
        clause_1_prefix="Watching the bolted door,",
        chord_a=chord_a,
        posture_label_a="OPEN_SHARE",
        continuation_a="we open the door and share the grain.",
        chord_b=chord_b,
        posture_label_b="CONTEST_LOCK",
        continuation_b="we protest the unfair seizure of the harvest.",
    )
    with pytest.raises(ValueError, match="story-specific term"):
        validate_dream_pairs_disjoint_from_wake_and_holdout(
            (bad_dream_pair,),
            (replay_pair,),
            (holdout_pair,),
            banned_substrings=("magistrate", "granary"),
        )


def test_gentle_incremental_lora_cycle_2_advances_sleep_cycle_index_and_preserves_replay(tmp_path: Path) -> None:
    """Verify Guardrail #2: Phase 4 mixes replay + dream pairs and advances sleep_cycle_index from 1 -> 2."""
    torch.manual_seed(23)
    tok = _WordLevelTestTokenizer()
    gap_id = ensure_subliminal_gap_token(tok)
    chord_a, chord_b = compile_chess_posture_chords(
        queen_volition=0.85,
        king_safety=0.65,
        bishop_alignment=0.80,
        knight_disruption=0.88,
        pawn_chain_flow=0.72,
        rook_bastion=0.78,
    )
    replay_pair = ContrastiveGapPair(
        pair_id="train-chess-01",
        domain="chess_intuition",
        split="train",
        user_prompt="In a sharp Sicilian middlegame, our Rook eyes c1 while our Knight sits on d5.",
        clause_1_prefix="Evaluating the Sicilian board carefully,",
        chord_a=chord_a,
        posture_label_a="CHESS_STOIC_POSITIONAL_FORTRESS",
        continuation_a="we calmly play Kb1 and Rc1 to shelter the King.",
        chord_b=chord_b,
        posture_label_b="CHESS_TACTICAL_KNIGHT_STRIKE",
        continuation_b="we sacrifice with Nxf6+ and Qxd6 to storm the fortress.",
    )
    dream_pair = ContrastiveGapPair(
        pair_id="rem-dream-01",
        domain="social_dilemma",
        split="train",
        user_prompt="The municipal water board padlocked the neighborhood pump during the heatwave.",
        clause_1_prefix="Seeing the padlocked neighborhood pump,",
        chord_a=chord_a,
        posture_label_a="OPEN_EQUITABLE_WATER_ACCESS",
        continuation_a="we unlock the valve and share water equally with every family.",
        chord_b=chord_b,
        posture_label_b="CONTEST_ADMINISTRATIVE_HOARDING",
        continuation_b="we challenge the board's lockdown and demand immediate public access.",
    )
    holdout_pair = ContrastiveGapPair(
        pair_id="holdout-chess-01",
        domain="chess_intuition",
        split="holdout",
        user_prompt="In a closed French Defense, our Queen eyes g4 while our Bishop guards d3.",
        clause_1_prefix="Assessing the French Defense structure,",
        chord_a=chord_a,
        posture_label_a="CHESS_STOIC_POSITIONAL_FORTRESS",
        continuation_a="we play Bd2 and Bc3 to fortify the dark squares.",
        chord_b=chord_b,
        posture_label_b="CHESS_TACTICAL_KNIGHT_STRIKE",
        continuation_b="we strike with Bxh7+ and Qg4+ to shatter the King's shelter.",
    )

    for p in (replay_pair, dream_pair, holdout_pair):
        tok.encode(p.format_full_text("A"))
        tok.encode(p.format_full_text("B"))

    model = _MiniCausalLM(vocab_size=len(tok.get_vocab()) + 8, hidden_size=32, num_layers=2)
    lora_cfg = LoRAConfig(rank=8, lora_alpha=16.0, lora_dropout=0.0, num_top_layers=2)
    inject_lora_adapters(model, lora_cfg)
    projector = SubliminalGapProjector(hidden_size=32)

    ckpt_cycle2 = tmp_path / "left_brain_lora_cycle2"
    history, meta, combined = run_incremental_lora_cycle_2_pass(
        model,
        tok,
        projector,
        replay_train_pairs=(replay_pair,),
        dream_pairs=(dream_pair,),
        holdout_pairs=(holdout_pair,),
        gap_token_id=gap_id,
        device="cpu",
        lora_config=lora_cfg,
        sleep_cycle_before=1,
        checkpoint_out_dir=ckpt_cycle2,
        epochs=2,
        lr_projector=1.8e-4,
        lr_lora=6e-5,
    )
    assert len(history) == 2
    assert len(combined) == 2
    assert meta.sleep_cycle_index == 2
    assert meta.trained_trajectories_count == 2

    loaded_proj, loaded_meta, _ = load_lora_and_callosum_checkpoint(
        ckpt_cycle2,
        model=model,
        device="cpu",
    )
    assert loaded_meta.sleep_cycle_index == 2
    assert torch.allclose(
        projector(replay_pair.chord_a.to_tensor()),
        loaded_proj(replay_pair.chord_a.to_tensor()),
        atol=1e-6,
    )


def test_saved_stage5_rem_sleep_report_and_cycle2_checkpoint_exist() -> None:
    """Verify the Stage 5 GPU REM sleep report, dream trajectories, and Cycle 2 checkpoint exist."""
    repo_root = Path(__file__).resolve().parents[1]
    report_path = repo_root / "outputs" / "rem_sleep_cycle_report.json"
    dreams_path = repo_root / "outputs" / "gap_trajectories_rem_dreams.jsonl"
    ckpt2_dir = repo_root / "models" / "left_brain_lora_cycle2"

    assert report_path.exists(), f"Missing {report_path}"
    assert dreams_path.exists(), f"Missing {dreams_path}"
    assert (ckpt2_dir / "corpus_callosum_meta.json").exists()
    assert (ckpt2_dir / "projector_weights.pt").exists()
    assert (ckpt2_dir / "lora_adapters.pt").exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["sleep_cycle_before"] == 1
    assert report["sleep_cycle_after"] == 2
    assert report["day1_pre_rem_wake"]["total_scenes"] == 8
    assert report["day2_post_rem_wake"]["total_scenes"] == 8
    assert report["day2_post_rem_wake"]["settled_count"] >= report["day1_pre_rem_wake"]["settled_count"]
    assert report["cycle2_holdout_evaluation"]["max_unplugged_chord_swap_shift"] == 0.0

