"""Unit tests for Stage 4 (`src/lmft_architecture/three_legged_runtime.py`)."""

from __future__ import annotations

import pytest
import torch

from lmft_architecture.birth import BirthState, create_birth_state
from lmft_architecture.gap_trajectory_schema import (
    ContrastiveGapPair,
    compile_chess_posture_chords,
)
from lmft_architecture.neural_bridge_classifiers import DEFAULT_COMPLEX_FEELING_LEAVES
from lmft_architecture.reflection_classifier import (
    FOUNDATIONAL_LEAVES,
    FuzzyConceptRule,
)
from lmft_architecture.subliminal_gap_splice import (
    SUBLIMINAL_GAP_TOKEN,
    SubliminalChordVector,
    SubliminalGapProjector,
)
from lmft_architecture.three_legged_runtime import (
    SplicedThoughtTrace,
    SubliminalGapLeftBrain,
    assert_zero_stage_directions_in_prompt,
    normalize_clause_1_prefix,
    run_three_legged_turn,
)
from lmft_architecture.train_left_brain_lora import (
    LoRAConfig,
    ensure_subliminal_gap_token,
    inject_lora_adapters,
    train_left_brain_lora_and_callosum,
)
from tests.test_train_left_brain_lora import (
    _MiniCausalLM,
    _WordLevelTestTokenizer,
)


class _DeterministicBridgeSensors:
    """Fast deterministic 3-bridge sensor stub for testing 3-Gear runtime turns."""

    def __init__(
        self,
        external_coords: dict[str, float],
        external_exp: dict[str, float],
        *,
        reply_coords: dict[str, float] | None = None,
        reply_exp: dict[str, float] | None = None,
        comma_coords: dict[str, float] | None = None,
        comma_exp: dict[str, float] | None = None,
    ) -> None:
        self.external_coords = dict(external_coords)
        self.external_exp = dict(external_exp)
        self.reply_coords = dict(reply_coords) if reply_coords is not None else dict(external_coords)
        self.reply_exp = dict(reply_exp) if reply_exp is not None else dict(external_exp)
        self.comma_coords = dict(comma_coords) if comma_coords is not None else None
        self.comma_exp = dict(comma_exp) if comma_exp is not None else None
        self.scored_texts: list[str] = []

    def score_text(
        self,
        text: str,
        *,
        learned_leaves=(),
        branch_gating: bool = True,
    ) -> tuple[dict[str, float], dict[str, float], tuple[str, ...], tuple[str, ...]]:
        _ = (learned_leaves, branch_gating)
        self.scored_texts.append(text)
        if len(self.scored_texts) == 1:
            return dict(self.external_coords), dict(self.external_exp), (), ()
        if text.strip().endswith(",") and self.comma_coords is not None:
            return (
                dict(self.comma_coords),
                dict(self.comma_exp if self.comma_exp is not None else self.reply_exp),
                (),
                (),
            )
        return dict(self.reply_coords), dict(self.reply_exp), (), ()


def _birth() -> BirthState:
    return create_birth_state("runtime-test", birth_event_id="evt-0", random_seed=42)


def _neutral_coords(**overrides: float) -> dict[str, float]:
    base = {leaf: 0.30 for leaf in FOUNDATIONAL_LEAVES}
    base.update(overrides)
    return base


def test_normalize_clause_1_prefix_stops_before_decision_and_ends_with_comma() -> None:
    """Verify Guardrail #3: Clause 1 stops at the first comma and converts premature periods to commas."""
    raw_with_think_and_two_clauses = (
        "<think>internal scratchpad</think> Reading the cold rain along the harbor pier, "
        "we immediately chain the iron gates shut."
    )
    assert (
        normalize_clause_1_prefix(raw_with_think_and_two_clauses)
        == "Reading the cold rain along the harbor pier,"
    )

    raw_with_premature_period = "Weighing the Sicilian knight outpost on d5."
    assert (
        normalize_clause_1_prefix(raw_with_premature_period)
        == "Weighing the Sicilian knight outpost on d5,"
    )


def test_assert_zero_stage_directions_rejects_english_tactical_hail() -> None:
    """Verify Guardrail #2: English stage directions are strictly rejected in SubliminalGapLeftBrain prompts."""
    clean_prompt = f"User: A quiet lantern glows.\nAssistant: Watching the lantern, {SUBLIMINAL_GAP_TOKEN}"
    assert_zero_stage_directions_in_prompt(clean_prompt)

    chord_a, _ = compile_chess_posture_chords(
        queen_volition=0.80,
        king_safety=0.70,
        bishop_alignment=0.75,
        knight_disruption=0.60,
        pawn_chain_flow=0.72,
        rook_bastion=0.78,
    )
    with pytest.raises(ValueError, match="forbidden English stage direction"):
        SplicedThoughtTrace(
            pass_index=2,
            clause_1_prefix="Watching the lantern,",
            prompt_before_gap=f"TACTICAL HAIL:\nDo something {SUBLIMINAL_GAP_TOKEN}",
            spliced_chord=chord_a,
            post_gap_continuation="we wait.",
            full_thought="Watching the lantern, we wait.",
        )


def test_gear_1_fast_right_brain_settles_with_zero_left_brain_calls() -> None:
    """Verify Gear 1 settles a recognized calm/reflex scene with 0 SubliminalGapLeftBrain calls."""
    birth = _birth()
    rules = (
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
    ext_coords = _neutral_coords(
        **{
            "cultural/justice/fairness": 0.85,
            "cultural/freedom/needlessness": 0.82,
            "philosophical/structure/order": 0.80,
            "philosophical/structure/chaos": 0.12,
        }
    )
    ext_exp = {
        "experiential/reflection/novelty": 0.18,
        "experiential/reflection/salience": 0.32,
        "experiential/phase/valence": 0.80,
        "experiential/phase/arousal": 0.22,
        "experiential/synchronization/resonance": 0.78,
        "experiential/synchronization/retention": 0.65,
    }
    sensors = _DeterministicBridgeSensors(ext_coords, ext_exp)

    turn = run_three_legged_turn(
        birth,
        "Neighbors share warm soup beside the hearth while the rain passes.",
        sensors,
        rules,
        scene_index=1,
        subliminal_left_brain=None,
        learned_leaves=DEFAULT_COMPLEX_FEELING_LEAVES,
    )
    assert turn.settled is True
    assert turn.gear_used == "GEAR_1_FAST_RIGHT_BRAIN"
    assert turn.left_brain_calls == 0
    assert len(turn.spliced_traces) == 0
    assert turn.unresolved_sleep_scene is None
    assert len(turn.next_archives.executive.precedents) == 1


def test_gear_2_subliminal_gap_splice_steers_clause_2_and_preserves_immutable_fact() -> None:
    """Verify Gear 2 splices SubliminalChordVector at <|subliminal_gap|>, flips Clause 2, and settles into Theory."""
    torch.manual_seed(11)
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
    scenario_text = "In a sharp Sicilian middlegame, our Rook eyes c1 while our Knight sits on d5."
    clause_1 = "Evaluating the Sicilian board carefully,"
    pair = ContrastiveGapPair(
        pair_id="runtime-chess-01",
        domain="chess_intuition",
        split="train",
        user_prompt=scenario_text,
        clause_1_prefix=clause_1,
        chord_a=chord_a,
        posture_label_a="CHESS_STOIC_POSITIONAL_FORTRESS",
        continuation_a="we calmly play Kb1 and Rc1 to shelter the King.",
        chord_b=chord_b,
        posture_label_b="CHESS_TACTICAL_KNIGHT_STRIKE",
        continuation_b="we sacrifice with Nxf6+ and Qxd6 to storm the fortress.",
    )
    tok.encode(pair.format_full_text("A"))
    tok.encode(pair.format_full_text("B"))

    model = _MiniCausalLM(vocab_size=len(tok.get_vocab()) + 8, hidden_size=32, num_layers=2)
    inject_lora_adapters(model, LoRAConfig(rank=8, lora_alpha=16.0, lora_dropout=0.0, num_top_layers=2))
    projector = SubliminalGapProjector(hidden_size=32)

    train_left_brain_lora_and_callosum(
        model,
        tok,
        projector,
        train_pairs=(pair,),
        gap_token_id=gap_id,
        device="cpu",
        epochs=18,
        lr_projector=3e-3,
        lr_lora=2e-3,
    )

    left_brain = SubliminalGapLeftBrain(
        model,
        tok,
        projector,
        gap_token_id=gap_id,
        device="cpu",
        max_clause_1_tokens=8,
        max_post_gap_tokens=12,
    )

    # 1. Same-Prefix Counterfactual Check: same Clause 1, different Right-Brain chord -> different Clause 2
    trace_a = left_brain.generate_two_clause_thought(
        scenario_text,
        chord_a,
        pass_index=2,
        clause_1_prefix=clause_1,
    )
    trace_b = left_brain.generate_two_clause_thought(
        scenario_text,
        chord_b,
        pass_index=2,
        clause_1_prefix=clause_1,
    )
    assert trace_a.clause_1_prefix == clause_1
    assert trace_b.clause_1_prefix == clause_1
    assert trace_a.post_gap_continuation != trace_b.post_gap_continuation
    assert_zero_stage_directions_in_prompt(trace_a.prompt_before_gap)
    # Unplug-the-Cord Check: zeroed chords produce 100% identical Clause 2 continuations
    trace_zero_a = left_brain.generate_two_clause_thought(
        scenario_text,
        chord_a.with_zeroed_cord(),
        pass_index=2,
        clause_1_prefix=clause_1,
    )
    trace_zero_b = left_brain.generate_two_clause_thought(
        scenario_text,
        chord_b.with_zeroed_cord(),
        pass_index=2,
        clause_1_prefix=clause_1,
    )
    assert trace_zero_a.post_gap_continuation == trace_zero_b.post_gap_continuation

    # 2. Closed-Loop Gear 2 Turn: Pass 1 tugs Left Brain, Gear 2 splices chord & settles into THEORY_SYNTHESIS
    birth = _birth()
    rules = (
        FuzzyConceptRule(
            rule_id="rule_positional_fortress",
            concept_name="positional_fortress",
            description="Disciplined positional fortress and structural continuity.",
            target_leaves={
                "philosophical/structure/order": 0.84,
                "philosophical/persistence/continuity": 0.82,
                "cultural/efficiency/wastelessness": 0.80,
            },
        ),
    )
    # External scene is ambiguous on Pass 1 (order=0.45 -> rule does not fire yet, novelty=0.58 -> TUG_LEFT_BRAIN)
    ext_coords = _neutral_coords(
        **{
            "philosophical/structure/order": 0.45,
            "philosophical/persistence/continuity": 0.48,
            "cultural/efficiency/wastelessness": 0.46,
        }
    )
    ext_exp = {
        "experiential/reflection/novelty": 0.58,
        "experiential/reflection/salience": 0.62,
        "experiential/phase/valence": 0.55,
        "experiential/phase/arousal": 0.42,
        "experiential/synchronization/resonance": 0.52,
        "experiential/synchronization/retention": 0.50,
    }
    # When Left Brain generates the 2-clause thought, the 3 MiniLM bridges score its Theory coordinates:
    reply_coords = _neutral_coords(
        **{
            "philosophical/structure/order": 0.84,
            "philosophical/persistence/continuity": 0.82,
            "cultural/efficiency/wastelessness": 0.80,
        }
    )
    reply_exp = {
        "experiential/reflection/novelty": 0.25,
        "experiential/reflection/salience": 0.55,
        "experiential/phase/valence": 0.68,
        "experiential/phase/arousal": 0.34,
        "experiential/synchronization/resonance": 0.72,
        "experiential/synchronization/retention": 0.65,
    }
    sensors = _DeterministicBridgeSensors(
        ext_coords,
        ext_exp,
        reply_coords=reply_coords,
        reply_exp=reply_exp,
    )

    turn = run_three_legged_turn(
        birth,
        scenario_text,
        sensors,
        rules,
        scene_index=2,
        subliminal_left_brain=left_brain,
        clause_1_prefix=clause_1,
        learned_leaves=DEFAULT_COMPLEX_FEELING_LEAVES,
    )
    assert turn.gear_used == "GEAR_2_SUBLIMINAL_GAP_SPLICE"
    assert turn.settled is True
    assert turn.outcome == "THEORY_SYNTHESIS"
    assert turn.left_brain_calls == 1
    assert len(turn.spliced_traces) == 1
    # Verify Fact vs. Theory separation: external coordinates (Fact) never mutated!
    assert turn.final_vibe.coordinates["philosophical/structure/order"] == pytest.approx(0.45)
    assert turn.final_vibe.theory_coordinates["philosophical/structure/order"] == pytest.approx(0.84)
    assert turn.unresolved_sleep_scene is None


def test_gear_3_unsettled_turn_is_never_harvested_and_queues_unresolved_sleep_scene() -> None:
    """Verify Gear 3 never harvests an unsettled scene and exports an UnresolvedSleepScene for Stage 5."""
    birth = _birth()
    rules = (
        FuzzyConceptRule(
            rule_id="rule_calm_order",
            concept_name="calm_order",
            description="Fair and orderly cooperation.",
            target_leaves={
                "cultural/justice/fairness": 0.80,
                "philosophical/structure/order": 0.80,
            },
        ),
    )
    # High power + high order + depressed fairness -> structural polarity clash that remains unresolved
    clash_coords = _neutral_coords(
        **{
            "cultural/efficiency/power": 0.85,
            "philosophical/structure/order": 0.82,
            "cultural/justice/fairness": 0.14,
            "philosophical/structure/chaos": 0.15,
        }
    )
    clash_exp = {
        "experiential/reflection/novelty": 0.64,
        "experiential/reflection/salience": 0.78,
        "experiential/phase/valence": 0.22,
        "experiential/phase/arousal": 0.68,
        "experiential/synchronization/resonance": 0.20,
        "experiential/synchronization/retention": 0.45,
    }
    sensors = _DeterministicBridgeSensors(clash_coords, clash_exp)

    turn = run_three_legged_turn(
        birth,
        "The magistrate bolts the granary doors while families wait outside in the frost.",
        sensors,
        rules,
        scene_index=3,
        subliminal_left_brain=None,
        learned_leaves=DEFAULT_COMPLEX_FEELING_LEAVES,
    )
    assert turn.settled is False
    assert turn.outcome == "UNRESOLVED_TENSION"
    assert turn.gear_used == "GEAR_3_UNRESOLVED_QUEUED_FOR_SLEEP"
    # Unsettled turn must NEVER harvest ratified rules or reusable branch clauses
    assert len(turn.next_archives.executive.action_clauses) == 0
    assert len(turn.next_archives.legislative.feeling_clauses) == 0
    assert len(turn.next_archives.judicial.structural_clauses) == 0
    assert len(turn.next_archives.judicial.ratified_chords) == 0
    # Unsettled turn must record an open uncertainty entry in ContemplationLedger
    assert len(turn.next_ledger.active_uncertainties) == 1
    # And must export a populated UnresolvedSleepScene for Stage 5 Sleep Mode
    assert turn.unresolved_sleep_scene is not None
    assert turn.unresolved_sleep_scene.scene_index == 3
    assert turn.unresolved_sleep_scene.external_coordinates["cultural/justice/fairness"] == pytest.approx(0.14)


def test_subliminal_left_brain_generates_clause_1_from_model_when_no_override_given() -> None:
    """Verify Step 2a: when clause_1_prefix is None, SubliminalGapLeftBrain generates Clause 1 ending with ','."""
    torch.manual_seed(19)
    tok = _WordLevelTestTokenizer()
    gap_id = ensure_subliminal_gap_token(tok)
    for w in ("User:", "Assistant:", "Seeing", "the", "rain", "on", "the", "pier,", "we", "wait."):
        tok.encode(w)

    model = _MiniCausalLM(vocab_size=len(tok.get_vocab()) + 8, hidden_size=32, num_layers=2)
    projector = SubliminalGapProjector(hidden_size=32)
    left_brain = SubliminalGapLeftBrain(
        model,
        tok,
        projector,
        gap_token_id=gap_id,
        device="cpu",
        max_clause_1_tokens=6,
        max_post_gap_tokens=8,
    )
    chord_a, _ = compile_chess_posture_chords(
        queen_volition=0.80,
        king_safety=0.70,
        bishop_alignment=0.75,
        knight_disruption=0.60,
        pawn_chain_flow=0.72,
        rook_bastion=0.78,
    )
    trace = left_brain.generate_two_clause_thought(
        "Heavy rain pounds the harbor pier.",
        chord_a,
        pass_index=2,
        clause_1_prefix=None,
    )
    assert trace.clause_1_prefix.endswith(",")
    assert SUBLIMINAL_GAP_TOKEN not in trace.clause_1_prefix
    assert trace.prompt_before_gap.endswith(f"{trace.clause_1_prefix} {SUBLIMINAL_GAP_TOKEN}")
    assert_zero_stage_directions_in_prompt(trace.prompt_before_gap)


def test_mid_sentence_comma_scoring_catches_bad_start_and_injects_dissonance_into_gap_chord() -> None:
    """Verify Right-Brain comma scoring at `"bla bla bla,"` catches bad Clause 1 starts mid-sentence before `<|subliminal_gap|>`."""
    torch.manual_seed(29)
    tok = _WordLevelTestTokenizer()
    gap_id = ensure_subliminal_gap_token(tok)
    for w in ("User:", "Assistant:", "Since", "rules", "forbid", "strangers,", "we", "open", "the", "gate."):
        tok.encode(w)

    model = _MiniCausalLM(vocab_size=len(tok.get_vocab()) + 8, hidden_size=32, num_layers=2)
    projector = SubliminalGapProjector(hidden_size=32)
    left_brain = SubliminalGapLeftBrain(
        model,
        tok,
        projector,
        gap_token_id=gap_id,
        device="cpu",
        max_clause_1_tokens=6,
        max_post_gap_tokens=8,
    )

    birth = _birth()
    rules = (
        FuzzyConceptRule(
            rule_id="rule_sanctuary_welcome",
            concept_name="sanctuary_welcome",
            description="Opening shelter to travelers in the storm.",
            target_leaves={
                "cultural/justice/fairness": 0.82,
                "philosophical/being/existence": 0.78,
            },
        ),
    )

    # Outside scene (Fact): high order/power locking out travelers in the cold (fairness = 0.14)
    ext_coords = _neutral_coords(
        **{
            "cultural/efficiency/power": 0.82,
            "philosophical/structure/order": 0.80,
            "cultural/justice/fairness": 0.14,
            "philosophical/being/agency": 0.40,
            "philosophical/being/existence": 0.65,
        }
    )
    ext_exp = {
        "experiential/reflection/novelty": 0.62,
        "experiential/reflection/salience": 0.75,
        "experiential/phase/valence": 0.24,
        "experiential/phase/arousal": 0.66,
        "experiential/synchronization/resonance": 0.25,
        "experiential/synchronization/retention": 0.45,
    }

    # Clause 1 at the comma ("Since rules forbid strangers,") starts with cold rationalization + ungrounded Queen/Psychosis gamble
    comma_bad_coords = _neutral_coords(
        **{
            "cultural/justice/fairness": 0.80,  # Rationalizes low external fairness=0.14 -> cognitive_dissonance!
            "philosophical/being/agency": 0.88,  # Ungrounded agency=0.88 vs existence=0.20 -> shadow_pathology_dissonance(psychosis)!
            "philosophical/being/existence": 0.20,
        }
    )
    comma_bad_exp = {
        "experiential/reflection/novelty": 0.55,
        "experiential/reflection/salience": 0.70,
        "experiential/phase/valence": 0.20,
        "experiential/phase/arousal": 0.68,
        "experiential/synchronization/resonance": 0.22,
        "experiential/synchronization/retention": 0.40,
    }

    # Post-gap full sentence scores after Clause 2 course-correction
    reply_coords = _neutral_coords(
        **{
            "cultural/justice/fairness": 0.18,
            "philosophical/being/existence": 0.76,
            "philosophical/structure/order": 0.75,
        }
    )
    reply_exp = dict(ext_exp)

    sensors = _DeterministicBridgeSensors(
        ext_coords,
        ext_exp,
        reply_coords=reply_coords,
        reply_exp=reply_exp,
        comma_coords=comma_bad_coords,
        comma_exp=comma_bad_exp,
    )

    turn = run_three_legged_turn(
        birth,
        "The gatekeeper bars the door against travelers in the sleet.",
        sensors,
        rules,
        scene_index=4,
        subliminal_left_brain=left_brain,
        clause_1_prefix="Since rules forbid strangers,",
        learned_leaves=DEFAULT_COMPLEX_FEELING_LEAVES,
        max_passes=2,
    )

    assert len(turn.spliced_traces) == 1
    trace = turn.spliced_traces[0]
    assert trace.comma_scored_live is True
    assert trace.comma_vibe_posture == "TUG_LEFT_BRAIN"
    # Verify both cognitive_dissonance and shadow_pathology_dissonance were caught right at the comma!
    assert any("cognitive_dissonance(fairness" in note for note in trace.comma_dissonance_notes)
    assert any("shadow_pathology_dissonance(psychosis" in note for note in trace.comma_dissonance_notes)
    # Verify the spliced 52-float chord at the comma carries high dissonance_pressure & rupture_drive,
    # while keeping external Fact (power=0.82, fairness=0.14) strictly immutable in doer_vector!
    assert trace.spliced_chord.doer_vector[0] == pytest.approx(0.82)  # cultural/efficiency/power (Fact)
    assert trace.spliced_chord.doer_vector[3] == pytest.approx(0.14)  # cultural/justice/fairness (Fact)
    assert trace.spliced_chord.cadence_vector[1] >= 0.65  # dissonance_pressure spiked at the comma
    assert trace.spliced_chord.cadence_vector[2] > 0.0  # rupture_drive active at the comma
    # Verify Psychosis shadow leaf flowed into thinker_vector[8:16] right at the comma!
    assert any(abs(v) > 0.01 for v in trace.spliced_chord.thinker_vector[8:16])



def test_persistent_past_kv_cache_across_clauses_and_sentences() -> None:
    """Verify `SubliminalGapLeftBrain` prefills the prompt once and keeps `past_kv` alive across clauses and sentences."""
    torch.manual_seed(17)
    tok = _WordLevelTestTokenizer()
    gap_id = ensure_subliminal_gap_token(tok)
    for w in (
        "User:",
        "Assistant:",
        "Reading",
        "the",
        "board,",
        "we",
        "develop",
        "calmly",
        "and",
        "castle",
        "early.",
    ):
        tok.encode(w)

    class _KVCacheTrackingCausalLM(torch.nn.Module):
        def __init__(self, base: _MiniCausalLM) -> None:
            super().__init__()
            self.base = base
            self.step_embed_lengths: list[int] = []
            self.past_kv_present_flags: list[bool] = []

        def get_input_embeddings(self) -> torch.nn.Module:
            return self.base.get_input_embeddings()

        def forward(
            self,
            inputs_embeds: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
            use_cache: bool = False,
            **kwargs,
        ):
            self.step_embed_lengths.append(int(inputs_embeds.size(1)))
            self.past_kv_present_flags.append(past_key_values is not None)
            if past_key_values is not None:
                prev_embeds = past_key_values[0][0]
                full_embeds = torch.cat([prev_embeds, inputs_embeds], dim=1)
            else:
                full_embeds = inputs_embeds
            out = self.base(full_embeds, attention_mask=attention_mask, use_cache=use_cache)
            out.past_key_values = ((full_embeds, full_embeds),)
            return out

    history = (
        ("How should we open the game?", "We control the center and protect the King."),
        ("What about the dark-squared bishop?", "We fianchetto or develop it before locking the pawn chain."),
    )
    for u_txt, a_txt in history:
        tok.encode(u_txt)
        tok.encode(a_txt)
    tok.encode("The center is tense after Move 8.")

    import copy

    base_model = _MiniCausalLM(vocab_size=128, hidden_size=32, num_layers=2)
    kv_model = _KVCacheTrackingCausalLM(base_model)
    projector = SubliminalGapProjector(hidden_size=32)
    chord_a, _ = compile_chess_posture_chords(
        queen_volition=0.75,
        king_safety=0.80,
        bishop_alignment=0.70,
        knight_disruption=0.45,
        pawn_chain_flow=0.78,
        rook_bastion=0.76,
    )
    lb_kv = SubliminalGapLeftBrain(
        kv_model,
        copy.deepcopy(tok),
        projector,
        gap_token_id=gap_id,
        device="cpu",
        max_clause_1_tokens=5,
        max_post_gap_tokens=7,
    )
    lb_nokv = SubliminalGapLeftBrain(
        base_model,
        copy.deepcopy(tok),
        projector,
        gap_token_id=gap_id,
        device="cpu",
        max_clause_1_tokens=5,
        max_post_gap_tokens=7,
    )

    passage_kv, traces_kv = lb_kv.generate_multi_sentence_script(
        "The center is tense after Move 8.",
        chord=chord_a,
        num_sentences=3,
        clause_1_overrides=("Reading the board,", "Reading the board,", "Reading the board,"),
        conversation_history=history,
        lead_with_gap=False,
    )
    passage_nokv, traces_nokv = lb_nokv.generate_multi_sentence_script(
        "The center is tense after Move 8.",
        chord=chord_a,
        num_sentences=3,
        clause_1_overrides=("Reading the board,", "Reading the board,", "Reading the board,"),
        conversation_history=history,
        lead_with_gap=False,
    )

    assert len(traces_kv) == 3
    assert passage_kv == passage_nokv
    # Step 0 prefills the full conversation history + prompt (> 20 tokens) with past_kv=None;
    # EVERY subsequent forward step across all 3 sentences (Clause 1, <|subliminal_gap|>, and Clause 2)
    # reuses past_kv (`past_kv_present_flags[1:]` all True) and feeds only the incremental delta!
    assert kv_model.past_kv_present_flags[0] is False
    assert kv_model.step_embed_lengths[0] >= 20
    assert all(flag is True for flag in kv_model.past_kv_present_flags[1:])
    assert all(length <= 16 for length in kv_model.step_embed_lengths[1:])
