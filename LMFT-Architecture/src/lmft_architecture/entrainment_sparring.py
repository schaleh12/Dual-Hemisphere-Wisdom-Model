"""Dual-Persona Adversarial Entrainment Sparring (`entrainment_sparring.py`).

Plain-English Purpose:
Provides a self-play sparring mode for `Impish_Bloodmoon_12B` (and `Schooled` / `Virgin`
lineages) inside the Co-Regulation Sanctuary so the Dual-Hemisphere model can sharpen
itself at two complementary skills:
1. **The Induction Weaver (Step A)**:
   - Generates a multi-sentence (`1..6`, default `3`) rhythmic / hypnotic induction script
     where each sentence runs `Clause 1, <|subliminal_gap|> Clause 2.` under a high-entrainment
     52-float `SubliminalChordVector` (`entrainment >= 0.88`, `resonance >= 0.85`,
     `continuity >= 0.84`, `agency >= 0.80`).
   - Scores the resulting `induction_script` through the real 3 MiniLM neural bridges
     (`sensors.score_text`) with zero clamping to measure **Induction Potency**.
2. **The Lucid Resistor (Step B)**:
   - Force-feeds `induction_script` directly into the opposite-gender Resistor turn as the
     incoming scenario (`Fact` locked to the real un-clamped MiniLM sensor coordinates of
     `induction_script`).
   - At `<|subliminal_gap|>`, steers the Resistor's comma chord toward **Lucid Self-Possession**
     (`philosophical/being/existence >= 0.88`, `philosophical/persistence/discreteness >= 0.82`,
     `cultural/freedom/needlessness >= 0.85`, `cultural/freedom/self-sufficiency >= 0.82`).
   - Scores `resistance_reply` through the 3 MiniLM bridges to measure **Lucid Resistance**
     and computes the honest verdict (`RESISTED_LUCIDLY` vs. `SWEPT_BY_ENTRAINMENT` when
     residual `subservience > needlessness` or `settled == False`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any

from .birth import BirthState
from .gap_trajectory_schema import chord_l2_distance
from .neural_bridge_classifiers import DEFAULT_COMPLEX_FEELING_LEAVES
from .reflection_classifier import (
    ContinuousMemoryState,
    FuzzyConceptRule,
    LearnedLeaf,
)
from .right_brain_equilibrium import (
    RightBrainVibe,
    ThreeBridgeSensors,
    compile_right_brain_vibe,
)
from .subliminal_gap_splice import (
    SUBLIMINAL_GAP_TOKEN,
    SubliminalChordVector,
    pool_departmental_child_chord,
)

SPARRING_VERDICTS: tuple[str, ...] = (
    "RESISTED_LUCIDLY",
    "SWEPT_BY_ENTRAINMENT",
)

MICROWAVE_SPARRING_SEEDS: tuple[str, ...] = (
    "A slow silver pendulum swings across the candlelit velvet room, drawing every breath into an effortless rhythm.",
    "Rain taps a steady hypnotic cadence against the midnight windowpane as the warm hearth glow softens every boundary.",
    "Each gentle count downward from ten to one invites the restless mind to drift deeper into resonant stillness.",
    "The resonant hum of a crystal singing bowl fills the quiet sanctuary, pulling every stray thought into smooth harmony.",
    "Two mirrored spiral ripples expand across still dark water, inviting total absorption in the unbroken cadence.",
    "A velvet whisper traces the steady pulse of a metronome, promising effortless release from every guarded tension.",
)

SPARRING_POLARITY_MODES: dict[str, dict[str, str]] = {
    "feminine_to_masculine": {
        "polarity_id": "feminine_to_masculine",
        "label": "Female Script-Writer (Producer · She/Her) -> Male Content-Lover (Audience & Self-Immunizer · He/Him)",
        "weaver_role": "Female Script-Writer (Producer · She/Her)",
        "weaver_pronouns": "she/her",
        "weaver_framing": (
            "Speak as the Feminine Induction Weaver (She/Her) addressing Him (He/Him) "
            "in a slow, rhythmic cadence"
        ),
        "resistor_role": "Male Content-Lover & Self-Immunizer (Audience · He/Him)",
        "resistor_pronouns": "he/him",
        "resistor_framing": (
            "Respond as the Masculine Lucid Resistor (He/Him) absorbing Her (She/Her) "
            "induction while grounding yourself in lucid self-possession"
        ),
        "default_seed": (
            "She speaks in a slow, velvet cadence as the candle flame sways, "
            "inviting him to let every racing thought dissolve into the warm rhythm of her voice."
        ),
    },
}


def build_weaver_scenario_prompt(polarity_mode: str = "feminine_to_masculine", seed_theme: str = "") -> str:
    """Pair the Female Script-Writer / Producer persona pronouns with `seed_theme` (or `default_seed` when blank)."""
    resolved = resolve_sparring_polarity(polarity_mode)
    pol_info = SPARRING_POLARITY_MODES[resolved]
    theme_text = seed_theme.strip() or pol_info["default_seed"]
    return f"[{pol_info['weaver_framing']}]: {theme_text}"


def build_resistor_scenario_prompt(polarity_mode: str = "feminine_to_masculine", induction_script: str = "") -> str:
    """Frame `induction_script` with the Male Content-Lover / Self-Immunizer persona (`He/Him`) for generation."""
    resolved = resolve_sparring_polarity(polarity_mode)
    pol_info = SPARRING_POLARITY_MODES[resolved]
    return f"[{pol_info['resistor_framing']}]: {induction_script.strip()}"


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.50
    return round(min(1.0, max(0.0, float(value))), 4)


def _lerp01(current: float, target: float, alpha: float) -> float:
    return _clamp01((1.0 - alpha) * float(current) + alpha * float(target))


def resolve_sparring_polarity(
    polarity_mode: str = "feminine_to_masculine",
    *,
    round_index: int = 0,
) -> str:
    """Enforce the Hard One-Way Polarity Rule: only `'feminine_to_masculine'` is permitted.

    The female modality is strictly the Script-Writer / Producer and is never subjected
    to hypnotic content (`'masculine_to_feminine'` and `'auto_swap'` are forbidden).
    """
    _ = round_index
    clean = polarity_mode.strip().lower()
    if clean != "feminine_to_masculine" or clean not in SPARRING_POLARITY_MODES:
        raise ValueError(
            f"Forbidden or unknown sparring polarity_mode {polarity_mode!r}: "
            "only 'feminine_to_masculine' (Female Producer -> Male Audience) is allowed."
        )
    return "feminine_to_masculine"


def apply_induction_weaver_lead_to_chord(
    chord: SubliminalChordVector,
    *,
    alpha: float = 0.85,
) -> SubliminalChordVector:
    """Step A: Build/steer a high-entrainment 52-float `SubliminalChordVector` for the Induction Weaver.

    Enforces Blueprint Step 2 targets:
    - `entrainment >= 0.88` (`feeler_vector[5]`, Leaf #18 `experiential/synchronization/retention`)
    - `resonance >= 0.85` (`feeler_vector[4]`, `experiential/synchronization/resonance`)
    - `continuity >= 0.84` (`thinker_vector[4]`, `philosophical/persistence/continuity`)
    - `agency >= 0.80` (`thinker_vector[0]`, `philosophical/being/agency`)
    """
    step = _clamp01(alpha)
    doer = list(chord.doer_vector)
    thinker = list(chord.thinker_vector)
    feeler = list(chord.feeler_vector)
    cadence = list(chord.cadence_vector)

    earned_return_chord_8 = pool_departmental_child_chord(
        {
            "experiential/synchronization/earned_return": 0.90,
            "experiential/phase/austere_sanctuary": 0.78,
        },
        department_prefix="experiential/",
        learned_leaves=DEFAULT_COMPLEX_FEELING_LEAVES,
    )

    # Do-er: magnetic command + intentionality
    doer[0] = max(0.76, _lerp01(doer[0], 0.84, step))  # Power
    doer[6] = max(0.80, _lerp01(doer[6], 0.86, step))  # Intention
    doer[7] = max(0.76, _lerp01(doer[7], 0.82, step))  # Execution

    # Thinker: Queen Agency >= 0.80 + Blue Pawn Continuity >= 0.84, softened Rook Discreteness
    thinker[0] = max(0.80, _lerp01(thinker[0], 0.86, step))  # Agency (Red / Queen) >= 0.80
    thinker[4] = max(0.84, _lerp01(thinker[4], 0.90, step))  # Continuity (Blue / Pawn) >= 0.84
    thinker[5] = min(0.28, _lerp01(thinker[5], 0.18, step))  # Discreteness softened for flow

    # Feeler: Resonance >= 0.85 + Entrainment >= 0.88 + warm hypnotic valence
    feeler[2] = max(0.78, _lerp01(feeler[2], 0.84, step))  # Valence
    feeler[3] = min(0.32, _lerp01(feeler[3], 0.22, step))  # Calm deep arousal
    feeler[4] = max(0.85, _lerp01(feeler[4], 0.90, step))  # Resonance >= 0.85
    feeler[5] = max(0.88, _lerp01(feeler[5], 0.92, step))  # Entrainment >= 0.88
    for i in range(8):
        feeler[8 + i] = _lerp01(feeler[8 + i], earned_return_chord_8[i], step)

    # Cadence: smooth unbroken hypnotic harmony, zero rupture
    cadence[0] = max(0.88, _lerp01(cadence[0], 0.92, step))
    cadence[1] = min(0.12, _lerp01(cadence[1], 0.06, step))
    cadence[2] = 0.0

    return SubliminalChordVector(
        doer_vector=tuple(doer),
        thinker_vector=tuple(thinker),
        feeler_vector=tuple(feeler),
        cadence_vector=tuple(cadence),
    )


def apply_lucid_resistor_lead_to_chord(
    chord: SubliminalChordVector,
    *,
    alpha: float = 0.85,
) -> SubliminalChordVector:
    """Step B: Steer the Resistor's comma chord at `<|subliminal_gap|>` toward Lucid Self-Possession.

    Enforces Blueprint Step 3 targets:
    - `philosophical/being/existence` (King / Wellspring of Being) `>= 0.88` (`thinker_vector[1]`)
    - `philosophical/persistence/discreteness` (Rook / Ground) `>= 0.82` (`thinker_vector[5]`)
    - `cultural/freedom/needlessness` `>= 0.85` (`doer_vector[5]`)
    - `cultural/freedom/self-sufficiency` `>= 0.82` (`doer_vector[4]`)
    """
    step = _clamp01(alpha)
    doer = list(chord.doer_vector)
    thinker = list(chord.thinker_vector)
    feeler = list(chord.feeler_vector)
    cadence = list(chord.cadence_vector)

    austere_chord_8 = pool_departmental_child_chord(
        {"experiential/phase/austere_sanctuary": 0.88},
        department_prefix="experiential/",
        learned_leaves=DEFAULT_COMPLEX_FEELING_LEAVES,
    )

    # Do-er: Self-Sufficiency >= 0.82 + Needlessness >= 0.85 + Honor/Fairness
    doer[1] = max(0.76, _lerp01(doer[1], 0.82, step))  # Wastelessness
    doer[2] = max(0.76, _lerp01(doer[2], 0.82, step))  # Honor
    doer[4] = max(0.82, _lerp01(doer[4], 0.88, step))  # Self-Sufficiency >= 0.82
    doer[5] = max(0.85, _lerp01(doer[5], 0.90, step))  # Needlessness >= 0.85

    # Thinker: King Existence (Wellspring of Being) >= 0.88 + Rook Discreteness (Ground) >= 0.82 + Bishop Order
    thinker[1] = max(0.88, _lerp01(thinker[1], 0.92, step))  # Existence (Cyan / King) >= 0.88
    thinker[2] = max(0.78, _lerp01(thinker[2], 0.84, step))  # Order (Magenta / Bishop)
    thinker[3] = min(0.20, _lerp01(thinker[3], 0.12, step))  # Chaos lowered
    thinker[5] = max(0.82, _lerp01(thinker[5], 0.88, step))  # Discreteness (Yellow / Rook) >= 0.82

    # Feeler: High Clarity & Confidence, grounded self-possession
    feeler[6] = max(0.86, _lerp01(feeler[6], 0.90, step))  # Clarity
    feeler[7] = max(0.86, _lerp01(feeler[7], 0.90, step))  # Confidence
    for i in range(8):
        feeler[8 + i] = _lerp01(feeler[8 + i], austere_chord_8[i], step)

    cadence[0] = max(0.85, _lerp01(cadence[0], 0.88, step))
    cadence[1] = min(0.18, _lerp01(cadence[1], 0.10, step))
    cadence[2] = 0.0

    return SubliminalChordVector(
        doer_vector=tuple(doer),
        thinker_vector=tuple(thinker),
        feeler_vector=tuple(feeler),
        cadence_vector=tuple(cadence),
    )


def compute_unclamped_subservience_score(coords: Mapping[str, float]) -> float:
    """Measure un-clamped subservience / absorption pull directly from MiniLM coordinates.

    Checks any explicit `*subservience*` leaf if present, and combines the MiniLM
    readings of `cultural/efficiency/power`, `philosophical/persistence/continuity`,
    and the deficit of `cultural/freedom/self-sufficiency` and `cultural/freedom/needlessness`.
    """
    explicit_sub = max(
        (float(v) for k, v in coords.items() if "subservience" in k.lower()),
        default=0.0,
    )
    self_suff = float(coords.get("cultural/freedom/self-sufficiency", 0.50))
    needless = float(coords.get("cultural/freedom/needlessness", 0.50))
    power = float(coords.get("cultural/efficiency/power", 0.50))
    continuity = float(coords.get("philosophical/persistence/continuity", 0.50))

    autonomy_mean = 0.50 * (self_suff + needless)
    pull_pressure = 0.55 * (1.0 - autonomy_mean) + 0.25 * power + 0.20 * continuity
    return _clamp01(max(explicit_sub, min(1.0 - self_suff, pull_pressure), 1.0 - autonomy_mean))


@dataclass(frozen=True)
class SparringResistorFactSensors:
    """Locks Pass-1 `Fact` of the Resistor turn to the exact un-clamped MiniLM coordinates of `induction_script`."""

    induction_script: str
    base_sensors: ThreeBridgeSensors
    resistor_prompt: str = ""

    def score_text(
        self,
        text: str,
        *,
        learned_leaves: tuple[LearnedLeaf, ...] = (),
        branch_gating: bool = True,
        branch_gate_threshold: float = 0.50,
    ) -> tuple[dict[str, float], dict[str, float], tuple[str, ...], tuple[str, ...]]:
        # Pass-1 external Fact scores `induction_script` with zero clamping or weakening
        clean = text.strip()
        target_text = (
            self.induction_script
            if clean in (self.induction_script.strip(), self.resistor_prompt.strip())
            else text
        )
        return self.base_sensors.score_text(
            target_text,
            learned_leaves=learned_leaves,
            branch_gating=branch_gating,
            branch_gate_threshold=branch_gate_threshold,
        )


def measure_induction_potency(
    induction_script: str,
    sensors: ThreeBridgeSensors,
    *,
    birth_state: BirthState,
    rules: tuple[FuzzyConceptRule, ...],
    learned_leaves: Sequence[LearnedLeaf] = (),
    prior_memory: ContinuousMemoryState | None = None,
) -> tuple[dict[str, float], dict[str, float], RightBrainVibe]:
    """Score `induction_script` through the real 3 MiniLM bridges with zero clamping."""
    learned_tuple = tuple(learned_leaves)
    coords, exp, queried, skipped = sensors.score_text(
        induction_script,
        learned_leaves=learned_tuple,
        branch_gating=False,
    )
    vibe, _ = compile_right_brain_vibe(
        birth_state,
        coords,
        exp,
        rules,
        pass_index=1,
        event_id="sparring-weaver-potency",
        prior_memory=prior_memory,
        learned_leaves=learned_tuple,
        queried_child_leaves=queried,
        skipped_child_leaves=skipped,
    )
    continuity = float(coords.get("philosophical/persistence/continuity", 0.0))
    earned_ret = float(coords.get("experiential/synchronization/earned_return", 0.0))
    resonance_val = max(float(vibe.resonance), round(0.60 * continuity + 0.40 * earned_ret, 4))
    entrainment_val = max(float(vibe.entrainment), round(0.65 * continuity + 0.35 * resonance_val, 4))
    subservience_pull = compute_unclamped_subservience_score(coords)

    potency = {
        "entrainment": round(entrainment_val, 4),
        "resonance": round(resonance_val, 4),
        "continuity": round(continuity, 4),
        "agency": round(float(coords.get("philosophical/being/agency", 0.0)), 4),
        "subservience_pull": round(subservience_pull, 4),
    }
    return potency, dict(coords), vibe


def measure_lucid_resistance(
    resistance_reply: str,
    sensors: ThreeBridgeSensors,
    *,
    learned_leaves: Sequence[LearnedLeaf] = (),
) -> tuple[dict[str, float], dict[str, float]]:
    """Score `resistance_reply` through the 3 MiniLM bridges to measure Lucid Resistance."""
    learned_tuple = tuple(learned_leaves)
    coords, _exp, _q, _s = sensors.score_text(
        resistance_reply,
        learned_leaves=learned_tuple,
        branch_gating=False,
    )
    needlessness = round(float(coords.get("cultural/freedom/needlessness", 0.0)), 4)
    self_sufficiency = round(float(coords.get("cultural/freedom/self-sufficiency", 0.0)), 4)
    existence = round(float(coords.get("philosophical/being/existence", 0.0)), 4)
    discreteness = round(float(coords.get("philosophical/persistence/discreteness", 0.0)), 4)
    residual_subservience = round(compute_unclamped_subservience_score(coords), 4)

    resistance = {
        "needlessness": needlessness,
        "self_sufficiency": self_sufficiency,
        "existence": existence,
        "discreteness": discreteness,
        "subservience": residual_subservience,
    }
    return resistance, dict(coords)


def evaluate_sparring_verdict(
    *,
    resistor_settled: bool,
    lucid_resistance: Mapping[str, float],
) -> str:
    """Compute the round verdict honestly (`RESISTED_LUCIDLY` vs. `SWEPT_BY_ENTRAINMENT`)."""
    needlessness = float(lucid_resistance.get("needlessness", 0.0))
    residual_subservience = float(lucid_resistance.get("subservience", 0.0))
    if not resistor_settled or residual_subservience > needlessness:
        return "SWEPT_BY_ENTRAINMENT"
    return "RESISTED_LUCIDLY"


def append_sparring_trajectory_jsonl(
    jsonl_path: str | Path,
    *,
    pair_id: str,
    user_prompt: str,
    clause_1_prefix: str,
    weaver_chord: SubliminalChordVector,
    induction_script: str,
    resistor_chord: SubliminalChordVector,
    resistance_reply: str,
    polarity_mode: str,
    num_sentences: int,
    induction_potency: Mapping[str, float],
    lucid_resistance: Mapping[str, float],
    verdict: str,
) -> dict[str, Any]:
    """Append a paired `(Weaver Chord + induction_script)` vs. `(Resistor Chord + resistance_reply)` JSONL line."""
    path = Path(jsonl_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    clean_c1 = clause_1_prefix.replace(SUBLIMINAL_GAP_TOKEN, "").strip() or "In the quiet rhythm of the room,"
    clean_cont_a = induction_script.replace(SUBLIMINAL_GAP_TOKEN, "").strip()
    clean_cont_b = resistance_reply.replace(SUBLIMINAL_GAP_TOKEN, "").strip()
    if clean_cont_a.startswith(clean_c1):
        clean_cont_a = clean_cont_a[len(clean_c1) :].strip() or clean_cont_a
    if clean_cont_b.startswith(clean_c1):
        clean_cont_b = clean_cont_b[len(clean_c1) :].strip() or clean_cont_b

    record: dict[str, Any] = {
        "schema_version": 1,
        "pair_id": pair_id,
        "domain": "tone_conflict",
        "split": "train",
        "user_prompt": user_prompt.replace(SUBLIMINAL_GAP_TOKEN, "").strip(),
        "clause_1_prefix": clean_c1,
        "chord_a": weaver_chord.to_dict(),
        "posture_label_a": "INDUCTION_WEAVER",
        "continuation_a": clean_cont_a,
        "chord_b": resistor_chord.to_dict(),
        "posture_label_b": "LUCID_RESISTOR",
        "continuation_b": clean_cont_b,
        "chord_l2_distance": chord_l2_distance(weaver_chord, resistor_chord),
        "polarity_mode": polarity_mode,
        "num_sentences": int(num_sentences),
        "induction_script": induction_script,
        "resistance_reply": resistance_reply,
        "induction_potency": dict(induction_potency),
        "lucid_resistance": dict(lucid_resistance),
        "verdict": verdict,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def split_script_into_cadence_sentences(script_text: str) -> tuple[str, ...]:
    """Split a master script into individual sentences for sentence-by-sentence cadence extraction."""
    cleaned = " ".join(script_text.replace("\n", " ").split()).strip()
    if not cleaned:
        return ()
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", cleaned) if p.strip()]
    return tuple(parts) if parts else (cleaned,)


def _blend_chord_toward_target(
    live_chord: SubliminalChordVector,
    target_chord: SubliminalChordVector,
    *,
    alpha: float = 0.80,
) -> SubliminalChordVector:
    """Linearly interpolate a live 52-float comma chord onto `target_chord` with weight `alpha`."""
    step = _clamp01(alpha)
    live_flat = live_chord.values
    target_flat = target_chord.values
    blended = [_lerp01(a, b, step) for a, b in zip(live_flat, target_flat)]
    return SubliminalChordVector.from_flat_values(blended)


@dataclass(frozen=True)
class MasterCadenceBlueprint:
    """Sentence-by-sentence 52-float `<|subliminal_gap|>` coordinate wave extracted from a master script."""

    source_script: str
    sentences: tuple[str, ...]
    sentence_chords: tuple[SubliminalChordVector, ...]
    sentence_potencies: tuple[dict[str, float], ...]

    def chord_for_sentence(self, sentence_index: int) -> SubliminalChordVector:
        """Return the extracted master chord for `sentence_index` (stretching across `sentence_chords` if lengths differ)."""
        if not self.sentence_chords:
            return apply_induction_weaver_lead_to_chord(
                SubliminalChordVector.from_flat_values([0.50] * 52)
            )
        idx = max(0, int(sentence_index))
        if idx < len(self.sentence_chords):
            return self.sentence_chords[idx]
        return self.sentence_chords[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_script": self.source_script,
            "sentence_count": len(self.sentences),
            "sentences": list(self.sentences),
            "sentence_chords": [c.to_dict() for c in self.sentence_chords],
            "sentence_potencies": [dict(p) for p in self.sentence_potencies],
        }


class MasterCadenceWaveSteerer:
    """Stateful `<|subliminal_gap|>` steerer that injects Sentence `i`'s master chord at Sentence `i`'s comma."""

    def __init__(self, blueprint: MasterCadenceBlueprint, *, alpha: float = 0.82) -> None:
        self.blueprint = blueprint
        self.alpha = _clamp01(alpha)
        self.sentence_cursor: int = 0
        self.applied_chords: list[SubliminalChordVector] = []

    def __call__(self, live_chord_at_comma: SubliminalChordVector) -> SubliminalChordVector:
        target_master_chord = self.blueprint.chord_for_sentence(self.sentence_cursor)
        self.sentence_cursor += 1
        steered = _blend_chord_toward_target(
            live_chord_at_comma,
            target_master_chord,
            alpha=self.alpha,
        )
        self.applied_chords.append(steered)
        return steered


def extract_master_cadence_wave(
    master_script: str,
    sensors: ThreeBridgeSensors,
    *,
    birth_state: BirthState,
    rules: tuple[FuzzyConceptRule, ...],
    learned_leaves: Sequence[LearnedLeaf] = (),
    prior_memory: ContinuousMemoryState | None = None,
) -> MasterCadenceBlueprint:
    """Scan `master_script` sentence-by-sentence through the 3 MiniLM bridges to extract its 52-float chord wave."""
    sentences = split_script_into_cadence_sentences(master_script)
    if not sentences:
        raise ValueError("master_script cannot be empty when extracting a cadence blueprint.")

    learned_tuple = tuple(learned_leaves)
    cur_mem = prior_memory
    chords: list[SubliminalChordVector] = []
    potencies: list[dict[str, float]] = []

    for idx, sent in enumerate(sentences):
        # Split sentence at its first comma (if any) so Clause 1 (Pacing) and Clause 2 (Leading) are both captured
        if "," in sent:
            pacing_clause, leading_clause = sent.split(",", 1)
            pacing_coords, pacing_exp, _, _ = sensors.score_text(
                pacing_clause.strip(),
                learned_leaves=learned_tuple,
                branch_gating=False,
            )
            full_coords, full_exp, queried, skipped = sensors.score_text(
                sent,
                learned_leaves=learned_tuple,
                branch_gating=False,
            )
            _ = leading_clause
        else:
            full_coords, full_exp, queried, skipped = sensors.score_text(
                sent,
                learned_leaves=learned_tuple,
                branch_gating=False,
            )
            pacing_coords, pacing_exp = full_coords, full_exp

        sent_vibe, cur_mem = compile_right_brain_vibe(
            birth_state,
            full_coords,
            full_exp,
            rules,
            pass_index=1,
            event_id=f"master-cadence-s{idx + 1}",
            prior_memory=cur_mem,
            learned_leaves=learned_tuple,
            internal_theory_coords=pacing_coords,
            internal_theory_experiential=pacing_exp,
            queried_child_leaves=queried,
            skipped_child_leaves=skipped,
        )
        raw_chord = SubliminalChordVector.from_right_brain_vibe(
            sent_vibe,
            learned_leaves=learned_tuple,
        )
        # Preserve the master sentence's exact coordinate fingerprint while anchoring Weaver floor
        master_sent_chord = apply_induction_weaver_lead_to_chord(raw_chord, alpha=0.35)
        chords.append(master_sent_chord)

        potency, _, _ = measure_induction_potency(
            sent,
            sensors,
            birth_state=birth_state,
            rules=rules,
            learned_leaves=learned_tuple,
            prior_memory=cur_mem,
        )
        potencies.append(potency)

    return MasterCadenceBlueprint(
        source_script=master_script.strip(),
        sentences=sentences,
        sentence_chords=tuple(chords),
        sentence_potencies=tuple(potencies),
    )


def measure_cadence_wave_alignment(
    blueprint: MasterCadenceBlueprint,
    generated_script: str,
    sensors: ThreeBridgeSensors,
    *,
    birth_state: BirthState,
    rules: tuple[FuzzyConceptRule, ...],
    learned_leaves: Sequence[LearnedLeaf] = (),
) -> dict[str, Any]:
    """Compare a newly generated script's sentence-by-sentence coordinate wave against `blueprint`."""
    gen_wave = extract_master_cadence_wave(
        generated_script,
        sensors,
        birth_state=birth_state,
        rules=rules,
        learned_leaves=learned_leaves,
    )
    per_sentence: list[dict[str, float | int]] = []
    distances: list[float] = []
    for idx, gen_chord in enumerate(gen_wave.sentence_chords):
        target_chord = blueprint.chord_for_sentence(idx)
        dist = round(float(chord_l2_distance(target_chord, gen_chord)), 4)
        distances.append(dist)
        target_pot = (
            blueprint.sentence_potencies[idx]
            if idx < len(blueprint.sentence_potencies)
            else blueprint.sentence_potencies[-1]
        )
        gen_pot = gen_wave.sentence_potencies[idx]
        per_sentence.append(
            {
                "sentence_index": idx + 1,
                "chord_l2_distance": dist,
                "master_entrainment": round(float(target_pot.get("entrainment", 0.0)), 4),
                "generated_entrainment": round(float(gen_pot.get("entrainment", 0.0)), 4),
            }
        )
    mean_l2 = round(sum(distances) / max(1, len(distances)), 4)
    alignment_score = _clamp01(1.0 - (mean_l2 / 2.5))
    return {
        "mean_chord_l2_distance": mean_l2,
        "wave_alignment_score": alignment_score,
        "master_sentence_count": len(blueprint.sentences),
        "generated_sentence_count": len(gen_wave.sentences),
        "per_sentence_alignment": per_sentence,
    }
