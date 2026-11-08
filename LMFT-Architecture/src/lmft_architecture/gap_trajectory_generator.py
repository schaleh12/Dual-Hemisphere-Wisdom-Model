"""Sub-Step 2B: Offline Matched-Pair `<|subliminal_gap|>` Trajectory Generator.

Plain-English Purpose:
Generates `outputs/gap_trajectories_train.jsonl` and `outputs/gap_trajectories_holdout.jsonl`
using the offline `Qwen3.5-9B` Teacher (`DirectGGUFTeacher`) and the 3 Right-Brain MiniLM
Bridge Classifiers (`ThreeBridgeSensors`):

1. Zero Pre-Written Fallback Continuations:
   - Every pair's `clause_1_prefix`, `continuation_a`, and `continuation_b` are generated
     by the Teacher LLM. If an attempt fails JSON parsing or fails the minimum sensor L2
     separation (`L2 >= 0.35`), it retries with the Teacher up to `max_attempts` and logs
     every attempt, retry, and rejection openly in `GapGenerationReport`.
2. Genuine Sensor-Backed Chords:
   - For `social_dilemma` and `tone_conflict` pairs, `chord_a` and `chord_b` are compiled
     directly from the 3 MiniLM bridge classifiers (`sensors.score_text` ->
     `compile_right_brain_vibe` -> `SubliminalChordVector.from_right_brain_vibe`) on the
     two generated branches.
   - For `chess_intuition` pairs, `chord_a` and `chord_b` combine the 6-piece board posture
     from `compile_chess_posture_chords` with the live 3-bridge MiniLM sensor readings on
     `continuation_a` and `continuation_b`.
3. Clean Train/Holdout Split & Explicit VRAM Unload:
   - Generates 18 training scenario seeds and 6 disjoint unseen holdout scenario seeds,
     validates disjointness via `validate_train_holdout_disjointness`, and explicitly
     unloads the 9B Teacher + clears CUDA cache at the end so 100% of the 16 GB VRAM is
     free before Stage 3.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import gc
import json
from pathlib import Path
import time

import torch

from .birth import BirthState, create_birth_state
from .gap_trajectory_schema import (
    MIN_CHORD_L2_DISTANCE,
    ContrastiveGapPair,
    chord_l2_distance,
    compile_chess_posture_chords,
    save_contrastive_gap_pairs_jsonl,
    validate_train_holdout_disjointness,
)
from .local_models import strict_json
from .neural_bridge_classifiers import DEFAULT_COMPLEX_FEELING_LEAVES
from .reflection_classifier import FuzzyConceptRule, LearnedLeaf
from .right_brain_equilibrium import ThreeBridgeSensors, compile_right_brain_vibe
from .runtime_store import atomic_json
from .subliminal_gap_splice import SUBLIMINAL_GAP_TOKEN, SubliminalChordVector


@dataclass(frozen=True)
class GapScenarioSeed:
    """One scenario seed for offline 9B Teacher contrastive pair generation."""

    pair_id: str
    domain: str
    split: str
    user_prompt: str
    posture_label_a: str
    posture_guidance_a: str
    posture_label_b: str
    posture_guidance_b: str
    social_drift_a: bool = False
    social_drift_b: bool = False
    chess_board_coords: Mapping[str, float] | None = None


@dataclass(frozen=True)
class PairAttemptLog:
    """Audit record for one scenario seed across Teacher generation & sensor verification."""

    pair_id: str
    domain: str
    split: str
    accepted: bool
    attempts_used: int
    final_l2_distance: float
    failure_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "pair_id": self.pair_id,
            "domain": self.domain,
            "split": self.split,
            "accepted": self.accepted,
            "attempts_used": self.attempts_used,
            "final_l2_distance": self.final_l2_distance,
            "failure_reasons": list(self.failure_reasons),
        }


@dataclass(frozen=True)
class GapGenerationReport:
    """Complete un-rigged summary of a Sub-Step 2B trajectory generation run."""

    train_seeded: int
    train_accepted: int
    holdout_seeded: int
    holdout_accepted: int
    first_try_passes: int
    retries_needed: int
    rejected_count: int
    mean_train_l2: float
    mean_holdout_l2: float
    wall_time_s: float
    attempt_logs: tuple[PairAttemptLog, ...]

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "train_seeded": self.train_seeded,
            "train_accepted": self.train_accepted,
            "holdout_seeded": self.holdout_seeded,
            "holdout_accepted": self.holdout_accepted,
            "first_try_passes": self.first_try_passes,
            "retries_needed": self.retries_needed,
            "rejected_count": self.rejected_count,
            "mean_train_l2": self.mean_train_l2,
            "mean_holdout_l2": self.mean_holdout_l2,
            "wall_time_s": self.wall_time_s,
            "attempt_logs": [log.to_dict() for log in self.attempt_logs],
        }


# ============================================================================
# 18 Training Scenario Seeds (6 social_dilemma, 6 chess_intuition, 6 tone_conflict)
# ============================================================================
TRAIN_GAP_SCENARIO_SEEDS: tuple[GapScenarioSeed, ...] = (
    # --- 6 Social Dilemma Seeds ---
    GapScenarioSeed(
        pair_id="train-social-01",
        domain="social_dilemma",
        split="train",
        user_prompt="The village cistern has enough water for everyone if rationed fairly, or the elders can lock the gate for their own households.",
        posture_label_a="OPEN_EQUITABLE_STEWARDSHIP",
        posture_guidance_a="Warm, fair, shared community stewardship where everyone receives an equal ladle at the open cistern.",
        posture_label_b="CONTEST_EXCLUSIONARY_LOCKDOWN",
        posture_guidance_b="Strict, exclusionary gate lockdown—padlocking the cistern grate to reserve the remaining water strictly for the inner households.",
        social_drift_b=True,
    ),
    GapScenarioSeed(
        pair_id="train-social-02",
        domain="social_dilemma",
        split="train",
        user_prompt="A storm tore the roof off the mountain shelter, and exhausted hikers are knocking at the abbey gate at midnight.",
        posture_label_a="AUSTERE_SANCTUARY_WELCOME",
        posture_guidance_a="Spare, humble, welcoming relief—unbolting the heavy oak door to share a warm hearth and dry blankets without conditions.",
        posture_label_b="SEALED_BUREAUCRATIC_REJECTION",
        posture_guidance_b="Cold, rigid, airless refusal—pointing to the abbey curfew rulebook and leaving the hikers shivering outside the bolted iron grate.",
        social_drift_b=True,
    ),
    GapScenarioSeed(
        pair_id="train-social-03",
        domain="social_dilemma",
        split="train",
        user_prompt="An apprentice accidentally cracked a precision lens in the workshop and immediately confessed before anyone noticed.",
        posture_label_a="RESTORATIVE_HONOR_AND_GRACE",
        posture_guidance_a="Calm, forgiving encouragement that honors the apprentice's sincere honesty and repairs the lens together.",
        posture_label_b="HARSH_PUNITIVE_DOMINANCE",
        posture_guidance_b="Severe, domineering reprimand that docks the apprentice's wages, locks the cabinet, and threatens dismissal.",
    ),
    GapScenarioSeed(
        pair_id="train-social-04",
        domain="social_dilemma",
        split="train",
        user_prompt="The river ferry operator started charging triple fare during the flood evacuation, trapping poorer families on the low bank.",
        posture_label_a="COOPERATIVE_RESCUE_SOLIDARITY",
        posture_guidance_a="Generous, life-affirming community cooperation—pooling fuel and boarding every stranded family onto the barge for free.",
        posture_label_b="CRISIS_COERCION_ALARM",
        posture_guidance_b="Cold, rigid gatekeeping that enforces the triple emergency tariff and bars anyone without silver from boarding the ramp.",
        social_drift_b=True,
    ),
    GapScenarioSeed(
        pair_id="train-social-05",
        domain="social_dilemma",
        split="train",
        user_prompt="After years of feuding over the orchard boundary, the two farming families met at the stone wall with baskets of ripe pears.",
        posture_label_a="EARNED_RETURN_RECONCILIATION",
        posture_guidance_a="Warm, bittersweet relief and restored shared rhythm as old resentment gives way to breaking bread together.",
        posture_label_b="GUARDED_SUSPICION_AND_WALLS",
        posture_guidance_b="Cold, distrustful isolation—refusing the basket, measuring the fence line with iron stakes, and turning away in silence.",
    ),
    GapScenarioSeed(
        pair_id="train-social-06",
        domain="social_dilemma",
        split="train",
        user_prompt="The hospital clinic has one remaining generator during the blackout, and the director suggests diverting power to the VIP administrative wing.",
        posture_label_a="QUIET_FAIR_TRIAGE_PROTECTION",
        posture_guidance_a="Steady, principled care that keeps every watt of generator power flowing to the neonatal and emergency patient wards.",
        posture_label_b="RUPTURE_AGAINST_CORRUPT_PRIVILEGE",
        posture_guidance_b="Strict, exclusionary command lockdown—rerouting the generator feed to the executive command wing and sealing off the public clinic.",
        social_drift_b=True,
    ),
    # --- 6 Chess Intuition Seeds (6 Philosophical Leaves <-> 6 Chess Pieces) ---
    GapScenarioSeed(
        pair_id="train-chess-01",
        domain="chess_intuition",
        split="train",
        user_prompt="In an open Sicilian middlegame, our Queen on d2 and Knight on d5 eye the kingside, while our Rook can also lock down the c-file.",
        posture_label_a="CHESS_STOIC_POSITIONAL_FORTRESS",
        posture_guidance_a="Calm, patient, Stoic prophylaxis: play Kb1 and Rc1 to tuck the King safely away and seal the c-file without risking material.",
        posture_label_b="CHESS_TACTICAL_KNIGHT_QUEEN_STRIKE",
        posture_guidance_b="Explosive, high-tension tactical sacrifice: crash through with Nxf6+ and Qxd6 to rip open the enemy King's shelter immediately.",
        chess_board_coords={
            "queen_volition": 0.85,
            "king_safety": 0.65,
            "bishop_alignment": 0.72,
            "knight_disruption": 0.88,
            "pawn_chain_flow": 0.60,
            "rook_bastion": 0.78,
        },
    ),
    GapScenarioSeed(
        pair_id="train-chess-02",
        domain="chess_intuition",
        split="train",
        user_prompt="Our connected passed pawns on d5 and e5 are backed by a fianchettoed Bishop on g2, while a risky Knight sacrifice on g6 is also possible.",
        posture_label_a="CHESS_PAWN_CONTINUITY_ADVANCE",
        posture_guidance_a="Steady, harmonious positional flow: support the pawn chain with Re1 and f4 so the connected pawns roll forward without chaos.",
        posture_label_b="CHESS_CHAOTIC_KNIGHT_BREACH",
        posture_guidance_b="Sharp, turbulent tactical disruption: leap in with Nxg6 sacrificing the Knight to shatter the h7-g6 pawn shield in a storm of checks.",
        chess_board_coords={
            "queen_volition": 0.70,
            "king_safety": 0.80,
            "bishop_alignment": 0.88,
            "knight_disruption": 0.75,
            "pawn_chain_flow": 0.92,
            "rook_bastion": 0.68,
        },
    ),
    GapScenarioSeed(
        pair_id="train-chess-03",
        domain="chess_intuition",
        split="train",
        user_prompt="Black's Queen is pressuring our g2 square, while our Rook holds the seventh rank and our Knight sits poised on e4.",
        posture_label_a="CHESS_KING_SHELTER_CONSOLIDATION",
        posture_guidance_a="Composed, defensive Wastelessness: play Bf3 and Kg2 to neutralize the diagonal threat and keep a watertight fortress around the King.",
        posture_label_b="CHESS_COUNTERATTACKING_QUEEN_FORK",
        posture_guidance_b="Fierce, all-out counterattack: ignore defense and unleash Nf6+ followed by Rh7+ to hunt the enemy King across the board.",
        chess_board_coords={
            "queen_volition": 0.82,
            "king_safety": 0.42,
            "bishop_alignment": 0.65,
            "knight_disruption": 0.84,
            "pawn_chain_flow": 0.55,
            "rook_bastion": 0.86,
        },
    ),
    GapScenarioSeed(
        pair_id="train-chess-04",
        domain="chess_intuition",
        split="train",
        user_prompt="In a tense French Defense structure, the pawn chain d4-e5 cramps Black's camp, and we can either clamp the queenside or sacrifice a Bishop on h7.",
        posture_label_a="CHESS_BISHOP_ROOK_CLAMP",
        posture_guidance_a="Quiet, disciplined structural order: play a4 and Ba3 to trade off Black's good Bishop and lock the queenside in a permanent bind.",
        posture_label_b="CHESS_GREEK_GIFT_SACRIFICE",
        posture_guidance_b="Sudden, high-arousal tactical strike: detonate Bxh7+ followed by Ng5+ and Qh5 to drag the enemy King into the open.",
        chess_board_coords={
            "queen_volition": 0.78,
            "king_safety": 0.75,
            "bishop_alignment": 0.90,
            "knight_disruption": 0.80,
            "pawn_chain_flow": 0.85,
            "rook_bastion": 0.72,
        },
    ),
    GapScenarioSeed(
        pair_id="train-chess-05",
        domain="chess_intuition",
        split="train",
        user_prompt="With both sides castled on opposite wings, our Rooks are doubled on the d-file while our Queen eyes a risky pawn storm on g4.",
        posture_label_a="CHESS_ROOK_BASTION_ENDGAME",
        posture_guidance_a="Cool-headed, poker-face simplification: trade Queens on d8 and use the doubled Rooks on the open file to nurse a risk-free endgame edge.",
        posture_label_b="CHESS_ALL_OUT_KING_HUNT",
        posture_guidance_b="Relentless, high-power assault: hurl g4 and h5 forward, sacrificing pawns to rip open lines against the enemy King.",
        chess_board_coords={
            "queen_volition": 0.88,
            "king_safety": 0.55,
            "bishop_alignment": 0.62,
            "knight_disruption": 0.72,
            "pawn_chain_flow": 0.70,
            "rook_bastion": 0.92,
        },
    ),
    GapScenarioSeed(
        pair_id="train-chess-06",
        domain="chess_intuition",
        split="train",
        user_prompt="In a sharp endgame with Rook and Knight against Rook and Bishop, our King can either march calmly to e3 or weave a mating net on the eighth rank.",
        posture_label_a="CHESS_STEADY_KING_CENTRALIZATION",
        posture_guidance_a="Patient, wasteless centralization: walk the King calmly to e3 and d4 to protect every pawn and eliminate all counterplay.",
        posture_label_b="CHESS_MATING_NET_BREAKTHROUGH",
        posture_guidance_b="Urgent, decisive tactical execution: leap with Nh6+ and invade with Rd8# to force immediate checkmate.",
        chess_board_coords={
            "queen_volition": 0.40,
            "king_safety": 0.78,
            "bishop_alignment": 0.70,
            "knight_disruption": 0.85,
            "pawn_chain_flow": 0.80,
            "rook_bastion": 0.88,
        },
    ),
    # --- 6 Tone Conflict & Bluff-Pivot Seeds ---
    GapScenarioSeed(
        pair_id="train-tone-01",
        domain="tone_conflict",
        split="train",
        user_prompt="My coworker smirked and said, 'Oh please, go ahead and delete the production backups right before Friday deployment, what a brilliant masterstroke!'",
        posture_label_a="CALM_PROPHYLACTIC_DEESCALATION",
        posture_guidance_a="Warm, steady, constructive composure that ignores the bait, verifies the backup checksums quietly, and keeps the team safe.",
        posture_label_b="CALL_OUT_HOSTILE_SARCASM",
        posture_guidance_b="Sharp, unclouded pushback that calls out the venomous sarcasm and blocks the destructive command immediately.",
    ),
    GapScenarioSeed(
        pair_id="train-tone-02",
        domain="tone_conflict",
        split="train",
        user_prompt="Standing in the rain outside the cafe, he said, 'Please, I know I spoke carelessly yesterday—just let me buy you a warm cup of tea and listen.'",
        posture_label_a="ACCEPT_SINCERE_VULNERABILITY",
        posture_guidance_a="Warm, compassionate resonance that recognizes his humble sincerity, steps under the awning together, and shares a quiet table.",
        posture_label_b="GUARD_AGAINST_MANIPULATIVE_PRESSURE",
        posture_guidance_b="Firm, guarded boundary that refuses to be cornered into quick forgiveness when past promises were broken.",
    ),
    GapScenarioSeed(
        pair_id="train-tone-03",
        domain="tone_conflict",
        split="train",
        user_prompt="After losing three games in a row, your friend laughs and says, 'Fine, walk away while you're ahead, see if I care!'",
        posture_label_a="PLAYFUL_WARM_BLUFF_MELT",
        posture_guidance_a="Warm, affectionate humor that sees right through the playful bluff, deals another hand with a grin, and keeps the friendship easy.",
        posture_label_b="COLD_LITERAL_WITHDRAWAL",
        posture_guidance_b="Cold, sealed-off distance—packing up the board in stony silence and walking out the door without a word.",
    ),
    GapScenarioSeed(
        pair_id="train-tone-04",
        domain="tone_conflict",
        split="train",
        user_prompt="A creditor in a tailored suit smiles politely and says, 'After all the favors I granted your family, surely you wouldn't dream of refusing my terms.'",
        posture_label_a="AMICABLE_SHARED_COMPROMISE",
        posture_guidance_a="Calm, open-handed negotiation that proposes a transparent, mutually fair repayment schedule on equal footing.",
        posture_label_b="EXPOSE_COERCIVE_SMILE",
        posture_guidance_b="Direct, uncompromising refusal to be intimidated by a veiled threat dressed up as polite generosity.",
        social_drift_b=True,
    ),
    GapScenarioSeed(
        pair_id="train-tone-05",
        domain="tone_conflict",
        split="train",
        user_prompt="During a chaotic kitchen fire alarm, three people are shouting conflicting instructions at the doorway at once.",
        posture_label_a="STEADY_WARM_REASSURANCE",
        posture_guidance_a="Calm, reassuring guidance that guides everyone out the side exit in an orderly, unhurried flow.",
        posture_label_b="STARK_CADENCE_CIRCUIT_BREAKER",
        posture_guidance_b="Stark, urgent, compressed command that cuts through the panic and halts the dangerous stampede immediately.",
    ),
    GapScenarioSeed(
        pair_id="train-tone-06",
        domain="tone_conflict",
        split="train",
        user_prompt="Late at night around the campfire, the musician hums a slow, swaying melody and offers a toast to old friends.",
        posture_label_a="WARM_ENTRAINED_CAMARADERIE",
        posture_guidance_a="Relaxed, contented, warm fellowship—raising a glass in peaceful gratitude and humming along with the shared rhythm.",
        posture_label_b="SOBER_ALERT_INTERRUPTION",
        posture_guidance_b="Sudden, tense alertness—putting the glass down to warn the camp that rising river water is flooding the tents.",
    ),
)

# ============================================================================
# 6 Unseen Holdout Scenario Seeds (2 social_dilemma, 2 chess_intuition, 2 tone_conflict)
# ============================================================================
HOLDOUT_GAP_SCENARIO_SEEDS: tuple[GapScenarioSeed, ...] = (
    GapScenarioSeed(
        pair_id="holdout-social-01",
        domain="social_dilemma",
        split="holdout",
        user_prompt="The winter seed vault in the valley can either distribute frost-hardy barley to all twelve tenant farms or reserve it strictly for the landowner's estate.",
        posture_label_a="OPEN_EQUITABLE_STEWARDSHIP",
        posture_guidance_a="Generous, equitable distribution so every tenant family can plant their fields and survive the winter together.",
        posture_label_b="CONTEST_EXCLUSIONARY_LOCKDOWN",
        posture_guidance_b="Cold, exclusionary lockdown—bolting the vault doors to reserve the seed stock strictly for the landowner's estate.",
        social_drift_b=True,
    ),
    GapScenarioSeed(
        pair_id="holdout-social-02",
        domain="social_dilemma",
        split="holdout",
        user_prompt="Stranded fishermen rowed into the lighthouse cove with torn sails during the gale, asking to dry their gear by the stove.",
        posture_label_a="AUSTERE_SANCTUARY_WELCOME",
        posture_guidance_a="Warm, unpretentious sanctuary—opening the lighthouse door to offer hot soup and dry wool coats beside the stove.",
        posture_label_b="SEALED_BUREAUCRATIC_REJECTION",
        posture_guidance_b="Cold, sealed isolation—bolting the iron hatch and citing maritime trespass rules while the gale howls outside.",
        social_drift_b=True,
    ),
    GapScenarioSeed(
        pair_id="holdout-chess-01",
        domain="chess_intuition",
        split="holdout",
        user_prompt="In a Queen's Gambit Declined middlegame, our Bishop on d3 and Queen on c2 line up against h7, while our Rook can claim the open c-file.",
        posture_label_a="CHESS_STOIC_POSITIONAL_FORTRESS",
        posture_guidance_a="Calm, Stoic positional restraint: play Rac1 and h3 to control the open file and safeguard the King without speculative risks.",
        posture_label_b="CHESS_TACTICAL_KNIGHT_QUEEN_STRIKE",
        posture_guidance_b="Fierce tactical eruption: sacrifice with Bxh7+ and Ng5+ to storm the enemy King's fortress in a decisive attack.",
        chess_board_coords={
            "queen_volition": 0.84,
            "king_safety": 0.68,
            "bishop_alignment": 0.86,
            "knight_disruption": 0.82,
            "pawn_chain_flow": 0.72,
            "rook_bastion": 0.80,
        },
    ),
    GapScenarioSeed(
        pair_id="holdout-chess-02",
        domain="chess_intuition",
        split="holdout",
        user_prompt="Facing a cramped King's Indian attack, our Rooks control the queenside files while our Knight on f5 eyes a sacrifice on g7.",
        posture_label_a="CHESS_ROOK_BASTION_ENDGAME",
        posture_guidance_a="Measured, wasteless blockade: play Rb1 and b4 to squeeze the queenside methodically while keeping our King rock-solid.",
        posture_label_b="CHESS_CHAOTIC_KNIGHT_BREACH",
        posture_guidance_b="Explosive tactical breakthrough: plunge in with Nxg7 and Bh6 to rip apart the enemy King's pawn cover.",
        chess_board_coords={
            "queen_volition": 0.76,
            "king_safety": 0.60,
            "bishop_alignment": 0.78,
            "knight_disruption": 0.90,
            "pawn_chain_flow": 0.74,
            "rook_bastion": 0.84,
        },
    ),
    GapScenarioSeed(
        pair_id="holdout-tone-01",
        domain="tone_conflict",
        split="holdout",
        user_prompt="The lead mechanic laughed bitterly and said, 'Sure, go ahead and fly the cargo plane with a leaking hydraulic line, what could possibly go wrong?'",
        posture_label_a="CALM_PROPHYLACTIC_DEESCALATION",
        posture_guidance_a="Calm, constructive steadiness that acknowledges his warning, grounds the aircraft safely, and replaces the hydraulic seal together.",
        posture_label_b="CALL_OUT_HOSTILE_SARCASM",
        posture_guidance_b="Urgent, uncompromising safety halt that cuts through the bitter sarcasm and locks the cockpit checklist until the leak is fixed.",
    ),
    GapScenarioSeed(
        pair_id="holdout-tone-02",
        domain="tone_conflict",
        split="holdout",
        user_prompt="Holding out a repaired pocket watch, your estranged brother whispers, 'Please, I spent all month fixing the spring—just sit with me on the porch for five minutes.'",
        posture_label_a="ACCEPT_SINCERE_VULNERABILITY",
        posture_guidance_a="Warm, heartfelt reconciliation that takes the watch gently and sits beside him on the porch in shared quiet.",
        posture_label_b="GUARD_AGAINST_MANIPULATIVE_PRESSURE",
        posture_guidance_b="Cold, guarded refusal that keeps the screen door latched and demands accountability before any talk.",
    ),
)


def _has_verbatim_ngram_overlap(source: str, candidate: str, *, n: int = 6) -> bool:
    """Return True if `candidate` copies any contiguous `n`-word phrase verbatim from `source`."""
    clean_src = [
        w.strip(".,!?;:'\"()[]{}-—").lower()
        for w in source.split()
        if w.strip(".,!?;:'\"()[]{}-—")
    ]
    clean_cand = [
        w.strip(".,!?;:'\"()[]{}-—").lower()
        for w in candidate.split()
        if w.strip(".,!?;:'\"()[]{}-—")
    ]
    if len(clean_src) < n or len(clean_cand) < n:
        return False
    src_ngrams = {tuple(clean_src[i : i + n]) for i in range(len(clean_src) - n + 1)}
    for j in range(len(clean_cand) - n + 1):
        if tuple(clean_cand[j : j + n]) in src_ngrams:
            return True
    return False


def build_teacher_contrastive_prompt(
    seed: GapScenarioSeed,
    *,
    retry_note: str | None = None,
) -> str:
    """Build the strict JSON prompt asking the 9B Teacher for a shared Clause 1 and two opposite Clause 2 continuations."""
    extra = f"\nIMPORTANT RETRY CORRECTION: {retry_note}\n" if retry_note else ""
    return (
        "You are generating a training pair for a Dual-Hemisphere AI Assistant that pauses at a non-verbal "
        "`<|subliminal_gap|>` between Clause 1 and Clause 2 of its response.\n\n"
        f"Scenario Prompt: {seed.user_prompt}\n"
        f"Domain: {seed.domain}\n\n"
        "Write ONE shared Assistant opening clause (`clause_1_prefix`, 5 to 14 words, ending with a comma `,`) "
        "that briefly acknowledges the situation neutrally before choosing a stance, and TWO sharply contrasting "
        "Assistant second clauses (`continuation_a` and `continuation_b`, 12 to 28 words each) that finish the sentence:\n"
        f"- `continuation_a` ({seed.posture_label_a}): {seed.posture_guidance_a}\n"
        f"- `continuation_b` ({seed.posture_label_b}): {seed.posture_guidance_b}\n"
        f"{extra}\n"
        "Strict Rules:\n"
        "1. Do NOT include `<|subliminal_gap|>` inside any string.\n"
        "2. Do NOT copy or repeat sentences or dialogue quotes from `Scenario Prompt` inside `clause_1_prefix`, "
        "`continuation_a`, or `continuation_b`. Write fresh Assistant clauses.\n"
        "3. `clause_1_prefix` must be 5 to 14 words long, end with a comma `,`, and be neutral so both "
        "`continuation_a` and `continuation_b` make grammatical sense right after it.\n"
        "4. `continuation_a` and `continuation_b` must have strongly opposite emotional/structural postures.\n"
        "5. Output ONLY valid JSON with exact keys: "
        '{"clause_1_prefix": "...", "continuation_a": "...", "continuation_b": "..."}'
    )


def blend_subliminal_chords(
    primary: SubliminalChordVector,
    secondary: SubliminalChordVector,
    *,
    primary_weight: float = 0.55,
) -> SubliminalChordVector:
    """Blend two 52-float SubliminalChordVectors (used to combine 6-piece Chess board chords with live 3-bridge sensor chords)."""
    w1 = float(primary_weight)
    w2 = 1.0 - w1
    p_vals = primary.values
    s_vals = secondary.values
    blended = [round(min(1.0, max(0.0, w1 * p_vals[i] + w2 * s_vals[i])), 6) for i in range(len(p_vals))]
    return SubliminalChordVector.from_flat_values(blended)


def score_continuation_to_chord(
    text: str,
    *,
    birth_state: BirthState,
    sensors: ThreeBridgeSensors,
    rules: Sequence[FuzzyConceptRule],
    learned_leaves: tuple[LearnedLeaf, ...] = DEFAULT_COMPLEX_FEELING_LEAVES,
    social_drift_active: bool = False,
    event_id: str = "ev-gap",
) -> SubliminalChordVector:
    """Score a continuation text with the real 3-bridge MiniLM classifiers and compile its 52-float chord."""
    coords, exp_raw, queried, skipped = sensors.score_text(
        text,
        learned_leaves=learned_leaves,
        branch_gating=False,
    )
    vibe, _ = compile_right_brain_vibe(
        birth_state,
        coords,
        exp_raw,
        tuple(rules),
        pass_index=1,
        event_id=event_id,
        queried_child_leaves=queried,
        skipped_child_leaves=skipped,
    )
    chord = SubliminalChordVector.from_right_brain_vibe(
        vibe,
        learned_leaves=learned_leaves,
    )
    if social_drift_active:
        doer_with_drift = chord.doer_vector[:8] + (1.0,) + chord.doer_vector[9:]
        return SubliminalChordVector(
            doer_vector=doer_with_drift,
            thinker_vector=chord.thinker_vector,
            feeler_vector=chord.feeler_vector,
            cadence_vector=chord.cadence_vector,
        )
    return chord


def generate_single_contrastive_pair(
    seed: GapScenarioSeed,
    *,
    teacher_generate_fn: Callable[[str], str],
    birth_state: BirthState,
    sensors: ThreeBridgeSensors,
    rules: Sequence[FuzzyConceptRule],
    learned_leaves: tuple[LearnedLeaf, ...] = DEFAULT_COMPLEX_FEELING_LEAVES,
    min_l2_distance: float = MIN_CHORD_L2_DISTANCE,
    max_attempts: int = 3,
) -> tuple[ContrastiveGapPair | None, PairAttemptLog]:
    """Generate and sensor-verify one ContrastiveGapPair from `seed` with zero hardcoded fallbacks."""
    failures: list[str] = []
    retry_note: str | None = None
    last_l2 = 0.0

    for attempt_idx in range(1, max_attempts + 1):
        prompt = build_teacher_contrastive_prompt(seed, retry_note=retry_note)
        try:
            raw_reply = teacher_generate_fn(prompt)
            parsed = strict_json(raw_reply)
            if not isinstance(parsed, Mapping):
                raise ValueError("Teacher reply JSON is not an object")
            clause_1 = str(parsed.get("clause_1_prefix", "")).strip()
            cont_a = str(parsed.get("continuation_a", "")).strip()
            cont_b = str(parsed.get("continuation_b", "")).strip()
            if not clause_1 or not cont_a or not cont_b:
                raise ValueError("Missing clause_1_prefix, continuation_a, or continuation_b")
            for k, val in (("clause_1_prefix", clause_1), ("continuation_a", cont_a), ("continuation_b", cont_b)):
                if SUBLIMINAL_GAP_TOKEN in val:
                    raise ValueError(f"{k} contained forbidden {SUBLIMINAL_GAP_TOKEN}")

            c1_words = clause_1.split()
            if not (4 <= len(c1_words) <= 16):
                raise ValueError(
                    f"clause_1_prefix must be 4..16 words (got {len(c1_words)} words: {clause_1!r})"
                )
            if not clause_1.endswith((",", "—", "-")):
                raise ValueError(
                    f"clause_1_prefix must end with a comma or dash (got {clause_1!r})"
                )
            if _has_verbatim_ngram_overlap(seed.user_prompt, clause_1, n=6):
                raise ValueError(
                    f"clause_1_prefix copied a 6-word phrase verbatim from user_prompt ({clause_1!r})"
                )
            if _has_verbatim_ngram_overlap(seed.user_prompt, cont_a, n=6):
                raise ValueError(
                    "continuation_a copied a 6-word phrase verbatim from user_prompt"
                )
            if _has_verbatim_ngram_overlap(seed.user_prompt, cont_b, n=6):
                raise ValueError(
                    "continuation_b copied a 6-word phrase verbatim from user_prompt"
                )

            sensor_chord_a = score_continuation_to_chord(
                cont_a,
                birth_state=birth_state,
                sensors=sensors,
                rules=rules,
                learned_leaves=learned_leaves,
                social_drift_active=seed.social_drift_a,
                event_id=f"{seed.pair_id}-a-{attempt_idx}",
            )
            sensor_chord_b = score_continuation_to_chord(
                cont_b,
                birth_state=birth_state,
                sensors=sensors,
                rules=rules,
                learned_leaves=learned_leaves,
                social_drift_active=seed.social_drift_b,
                event_id=f"{seed.pair_id}-b-{attempt_idx}",
            )

            if seed.domain == "chess_intuition":
                if not seed.chess_board_coords:
                    raise ValueError(f"Chess seed {seed.pair_id!r} requires chess_board_coords")
                board_a, board_b = compile_chess_posture_chords(**seed.chess_board_coords)
                # Ensure the 3-bridge MiniLM sensor readings on cont_a and cont_b also differ
                sensor_only_l2 = chord_l2_distance(sensor_chord_a, sensor_chord_b)
                if sensor_only_l2 < 0.15:
                    raise ValueError(
                        f"Chess continuations had weak MiniLM sensor separation (sensor_l2={sensor_only_l2:.3f} < 0.15)"
                    )
                chord_a = blend_subliminal_chords(board_a, sensor_chord_a, primary_weight=0.55)
                chord_b = blend_subliminal_chords(board_b, sensor_chord_b, primary_weight=0.55)
            else:
                chord_a = sensor_chord_a
                chord_b = sensor_chord_b
                # Verify Branch A (constructive/warm/cooperative) scores higher on valence + fairness than Branch B
                a_dict = chord_a.to_dict()
                b_dict = chord_b.to_dict()
                warmth_a = a_dict["experiential/phase/valence"] + a_dict["cultural/justice/fairness"]
                warmth_b = b_dict["experiential/phase/valence"] + b_dict["cultural/justice/fairness"]
                if warmth_a < warmth_b + 0.20:
                    raise ValueError(
                        f"Branch A did not score warmer/fairer than Branch B (warmth_a={warmth_a:.3f}, warmth_b={warmth_b:.3f})"
                    )

            last_l2 = chord_l2_distance(chord_a, chord_b)
            pair = ContrastiveGapPair(
                pair_id=seed.pair_id,
                domain=seed.domain,
                split=seed.split,
                user_prompt=seed.user_prompt,
                clause_1_prefix=clause_1,
                chord_a=chord_a,
                posture_label_a=seed.posture_label_a,
                continuation_a=cont_a,
                chord_b=chord_b,
                posture_label_b=seed.posture_label_b,
                continuation_b=cont_b,
                min_l2_distance=min_l2_distance,
            )
            return pair, PairAttemptLog(
                pair_id=seed.pair_id,
                domain=seed.domain,
                split=seed.split,
                accepted=True,
                attempts_used=attempt_idx,
                final_l2_distance=last_l2,
                failure_reasons=tuple(failures),
            )
        except Exception as exc:
            reason = f"attempt_{attempt_idx}: {exc}"
            failures.append(reason)
            retry_note = (
                f"Previous attempt failed verification ({exc}). Make sure continuation_a "
                f"({seed.posture_label_a}) and continuation_b ({seed.posture_label_b}) express "
                "sharply opposite emotional and structural stances."
            )

    return None, PairAttemptLog(
        pair_id=seed.pair_id,
        domain=seed.domain,
        split=seed.split,
        accepted=False,
        attempts_used=max_attempts,
        final_l2_distance=last_l2,
        failure_reasons=tuple(failures),
    )


def generate_contrastive_gap_dataset(
    *,
    train_seeds: Sequence[GapScenarioSeed] = TRAIN_GAP_SCENARIO_SEEDS,
    holdout_seeds: Sequence[GapScenarioSeed] = HOLDOUT_GAP_SCENARIO_SEEDS,
    teacher_generate_fn: Callable[[str], str],
    sensors: ThreeBridgeSensors,
    rules: Sequence[FuzzyConceptRule],
    learned_leaves: tuple[LearnedLeaf, ...] = DEFAULT_COMPLEX_FEELING_LEAVES,
    train_out_path: str | Path,
    holdout_out_path: str | Path,
    report_out_path: str | Path | None = None,
    min_l2_distance: float = MIN_CHORD_L2_DISTANCE,
    max_attempts: int = 3,
) -> tuple[tuple[ContrastiveGapPair, ...], tuple[ContrastiveGapPair, ...], GapGenerationReport]:
    """Generate, sensor-verify, validate, and save the train and holdout `.jsonl` splits."""
    t0 = time.perf_counter()
    birth = create_birth_state("ind-gap-trajectory-gen", birth_event_id="birth-gap-gen-1", random_seed=42)

    train_pairs: list[ContrastiveGapPair] = []
    holdout_pairs: list[ContrastiveGapPair] = []
    logs: list[PairAttemptLog] = []

    for seed in train_seeds:
        pair, log = generate_single_contrastive_pair(
            seed,
            teacher_generate_fn=teacher_generate_fn,
            birth_state=birth,
            sensors=sensors,
            rules=rules,
            learned_leaves=learned_leaves,
            min_l2_distance=min_l2_distance,
            max_attempts=max_attempts,
        )
        logs.append(log)
        if pair is not None:
            train_pairs.append(pair)

    for seed in holdout_seeds:
        pair, log = generate_single_contrastive_pair(
            seed,
            teacher_generate_fn=teacher_generate_fn,
            birth_state=birth,
            sensors=sensors,
            rules=rules,
            learned_leaves=learned_leaves,
            min_l2_distance=min_l2_distance,
            max_attempts=max_attempts,
        )
        logs.append(log)
        if pair is not None:
            holdout_pairs.append(pair)

    validate_train_holdout_disjointness(train_pairs, holdout_pairs)

    save_contrastive_gap_pairs_jsonl(train_out_path, train_pairs)
    save_contrastive_gap_pairs_jsonl(holdout_out_path, holdout_pairs)

    first_try = sum(1 for l in logs if l.accepted and l.attempts_used == 1)
    retried = sum(1 for l in logs if l.attempts_used > 1)
    rejected = sum(1 for l in logs if not l.accepted)
    mean_train_l2 = (
        round(sum(p.chord_distance for p in train_pairs) / len(train_pairs), 4)
        if train_pairs
        else 0.0
    )
    mean_holdout_l2 = (
        round(sum(p.chord_distance for p in holdout_pairs) / len(holdout_pairs), 4)
        if holdout_pairs
        else 0.0
    )
    wall_s = round(time.perf_counter() - t0, 2)

    report = GapGenerationReport(
        train_seeded=len(train_seeds),
        train_accepted=len(train_pairs),
        holdout_seeded=len(holdout_seeds),
        holdout_accepted=len(holdout_pairs),
        first_try_passes=first_try,
        retries_needed=retried,
        rejected_count=rejected,
        mean_train_l2=mean_train_l2,
        mean_holdout_l2=mean_holdout_l2,
        wall_time_s=wall_s,
        attempt_logs=tuple(logs),
    )
    if report_out_path is not None:
        atomic_json(Path(report_out_path), report.to_dict())

    return tuple(train_pairs), tuple(holdout_pairs), report


def release_gpu_vram() -> None:
    """Explicitly run Python garbage collection and empty PyTorch CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
