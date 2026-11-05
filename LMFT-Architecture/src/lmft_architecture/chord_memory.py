"""Pre-Verbal Felt Chords and Chord-Rhyming / Chord-Inversion Memory Retrieval.

Plain-English purpose:
- Compiles the Feeler bridge's raw numbers (`salience`, `valence`, `arousal`,
  `resonance`, plus any active `experiential/*` complex-feeling drawers) into a
  human-readable `FeltChord` free of dry debug labels.
- Stores settled scene chords in a `ChordMemoryBank` and retrieves two kinds of
  pre-verbal memory echoes across earlier scenes:
  1. Structural Rhyme: an earlier moment whose structural and emotional chord
     resonates closely with the present scene.
  2. Chord Inversion: an earlier moment that shares a structural backbone (or
     enclosure/flow axis) while flipping key value or feeling polarity.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path

from .reflection_classifier import (
    AROUSAL_LEAF,
    RESONANCE_LEAF,
    SALIENCE_LEAF,
    VALENCE_LEAF,
    LearnedLeaf,
    is_valid_leaf_id,
)
from .runtime_store import atomic_json

BACKBONE_AXES = (
    "philosophical/being/agency",
    "philosophical/being/existence",
    "philosophical/structure/order",
    "cultural/efficiency/power",
    "cultural/efficiency/wastelessness",
    "cultural/freedom/self-sufficiency",
)

POLARITY_AXES = (
    "cultural/justice/fairness",
    "philosophical/persistence/discreteness",
    "philosophical/persistence/continuity",
    "philosophical/structure/chaos",
    VALENCE_LEAF,
)

HUMAN_AXIS_NAMES = {
    "philosophical/being/agency": "deliberate human choice",
    "philosophical/being/existence": "grounded physical presence",
    "philosophical/structure/order": "structured craft and order",
    "philosophical/structure/chaos": "fracture and turbulence",
    "philosophical/persistence/continuity": "ongoing shared flow",
    "philosophical/persistence/discreteness": "sealed-off separation",
    "cultural/efficiency/power": "commanding force",
    "cultural/efficiency/wastelessness": "careful stewardship",
    "cultural/justice/honor": "sincere duty",
    "cultural/justice/fairness": "open, fair sharing",
    "cultural/freedom/self-sufficiency": "self-reliance",
    "cultural/freedom/needlessness": "quiet contentment",
}


def humanize_feeling_name(leaf_id_or_name: str) -> str:
    """Turn an experiential leaf ID into natural everyday English words."""
    short = leaf_id_or_name.split("/")[-1]
    return short.replace("_", " ").replace("-", " ").strip()


def _compact_gist(text: str, limit: int = 120) -> str:
    cleaned = " ".join(text.strip().split())
    first = cleaned.split(". ")[0].strip()
    if len(first) > limit:
        return first[: limit - 3].rstrip() + "..."
    return first


@dataclass(frozen=True)
class FeltChord:
    """Pure emotional chord compiled by the Feeler bridge."""

    salience: float
    valence: float
    arousal: float
    resonance: float
    complex_notes: dict[str, float]
    dominant_mood: str

    def to_dict(self) -> dict:
        return {
            "salience": round(self.salience, 4),
            "valence": round(self.valence, 4),
            "arousal": round(self.arousal, 4),
            "resonance": round(self.resonance, 4),
            "complex_notes": {k: round(v, 4) for k, v in self.complex_notes.items()},
            "dominant_mood": self.dominant_mood,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "FeltChord":
        return cls(
            salience=float(data["salience"]),
            valence=float(data["valence"]),
            arousal=float(data["arousal"]),
            resonance=float(data.get("resonance", 0.0)),
            complex_notes={str(k): float(v) for k, v in dict(data.get("complex_notes", {})).items()},
            dominant_mood=str(data["dominant_mood"]),
        )

    def short_stamp(self) -> str:
        """One-line felt signature for compact Contemplation Ledger certainties."""
        if self.complex_notes:
            ranked = sorted(self.complex_notes.items(), key=lambda kv: (-kv[1], kv[0]))
            top_names = [humanize_feeling_name(k) for k, v in ranked[:2] if v >= 0.40]
            if top_names:
                tone = "warm" if self.valence >= 0.58 else ("cold" if self.valence <= 0.35 else "mixed")
                return f"{' + '.join(top_names)} ({tone})"
        return self.dominant_mood

    def render_for_left_brain(self) -> str:
        """Render pure human-readable emotional atmosphere with zero raw debug tokens."""
        lines = [f"Atmosphere: {self.dominant_mood}."]
        if self.complex_notes:
            ranked = sorted(self.complex_notes.items(), key=lambda kv: (-kv[1], kv[0]))
            note_phrases = []
            for leaf_id, val in ranked:
                if val < 0.35:
                    continue
                name = humanize_feeling_name(leaf_id)
                if val >= 0.72:
                    note_phrases.append(f"strong {name}")
                elif val >= 0.52:
                    note_phrases.append(f"clear {name}")
                else:
                    note_phrases.append(f"undertone of {name}")
            if note_phrases:
                lines.append("Complex feeling notes: " + ", ".join(note_phrases) + ".")
        return " ".join(lines)


def compile_felt_chord(
    coordinates: Mapping[str, float],
    *,
    salience: float,
    valence: float,
    arousal: float,
    resonance: float = 0.0,
    learned_leaves: tuple[LearnedLeaf, ...] = (),
) -> FeltChord:
    """Compile raw experiential coordinates and complex feeling drawers into a FeltChord."""
    complex_notes: dict[str, float] = {}
    known_exp_ids = {
        l.leaf_id for l in learned_leaves if l.leaf_id.startswith("experiential/")
    }
    for k, v in coordinates.items():
        if k in (SALIENCE_LEAF, VALENCE_LEAF, AROUSAL_LEAF, RESONANCE_LEAF):
            continue
        if k.startswith("experiential/") or k in known_exp_ids:
            val = float(v)
            if val > 0.0:
                complex_notes[k] = round(val, 4)

    if valence <= 0.28:
        warmth = "cold, bleak unease"
    elif valence <= 0.42:
        warmth = "heavy, unsettled chill"
    elif valence < 0.62:
        warmth = "bittersweet, mixed tension"
    else:
        warmth = "warm, life-giving relief"

    if arousal <= 0.32:
        pulse = "quiet stillness"
    elif arousal < 0.65:
        pulse = "steady, held tension"
    else:
        pulse = "sharp urgency"

    if salience < 0.45:
        weight = "light passing weight"
    elif salience < 0.70:
        weight = "clear human weight"
    else:
        weight = "deep, unmistakable gravity"

    ranked_notes = sorted(complex_notes.items(), key=lambda kv: (-kv[1], kv[0]))
    lead_notes = [humanize_feeling_name(k) for k, v in ranked_notes[:2] if v >= 0.48]
    if lead_notes:
        dominant_mood = f"{' and '.join(lead_notes)} — {warmth} inside {pulse} ({weight})"
    else:
        dominant_mood = f"{warmth} inside {pulse} ({weight})"

    return FeltChord(
        salience=round(float(salience), 4),
        valence=round(float(valence), 4),
        arousal=round(float(arousal), 4),
        resonance=round(float(resonance), 4),
        complex_notes=complex_notes,
        dominant_mood=dominant_mood,
    )


@dataclass(frozen=True)
class ChordMemoryEntry:
    """One past scene's full coordinate chord + FeltChord stored for rhyming retrieval."""

    scene_index: int
    scene_gist: str
    coordinates: dict[str, float]
    felt_chord: FeltChord
    settled: bool = True

    def __post_init__(self):
        if type(self.scene_index) is not int or self.scene_index < 1:
            raise ValueError("scene_index must be a positive int")
        if not isinstance(self.scene_gist, str) or not self.scene_gist.strip():
            raise ValueError("scene_gist must be a nonempty string")
        for k, v in self.coordinates.items():
            if not is_valid_leaf_id(k):
                raise ValueError(f"invalid chord coordinate key: {k!r}")
            if type(v) not in (int, float) or not math.isfinite(v) or not (0.0 <= v <= 1.0):
                raise ValueError(f"coordinate {k!r} must be in [0, 1]")

    def to_dict(self) -> dict:
        return {
            "scene_index": self.scene_index,
            "scene_gist": self.scene_gist,
            "coordinates": dict(self.coordinates),
            "felt_chord": self.felt_chord.to_dict(),
            "settled": self.settled,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "ChordMemoryEntry":
        return cls(
            scene_index=int(data["scene_index"]),
            scene_gist=str(data["scene_gist"]),
            coordinates={str(k): float(v) for k, v in dict(data["coordinates"]).items()},
            felt_chord=FeltChord.from_dict(data["felt_chord"]),
            settled=bool(data.get("settled", True)),
        )


@dataclass(frozen=True)
class ChordMatch:
    """A retrieved structural rhyme or sharp chord inversion from earlier memory."""

    match_kind: str  # "STRUCTURAL_RHYME" or "CHORD_INVERSION"
    scene_index: int
    score: float
    past_gist: str
    past_felt_stamp: str
    relationship_summary: str

    def to_dict(self) -> dict:
        return {
            "match_kind": self.match_kind,
            "scene_index": self.scene_index,
            "score": round(self.score, 4),
            "past_gist": self.past_gist,
            "past_felt_stamp": self.past_felt_stamp,
            "relationship_summary": self.relationship_summary,
        }


@dataclass(frozen=True)
class ChordEchoReport:
    """Pair of pre-verbal memory echoes (rhyme and/or inversion) surfaced for the Left Brain."""

    structural_rhyme: ChordMatch | None = None
    chord_inversion: ChordMatch | None = None

    def has_echoes(self) -> bool:
        return self.structural_rhyme is not None or self.chord_inversion is not None

    def to_dict(self) -> dict:
        return {
            "structural_rhyme": self.structural_rhyme.to_dict() if self.structural_rhyme else None,
            "chord_inversion": self.chord_inversion.to_dict() if self.chord_inversion else None,
        }

    def render_for_left_brain(self) -> str:
        if not self.has_echoes():
            return "(no strong chord rhyme or inversion in earlier memory yet)"
        parts: list[str] = []
        if self.structural_rhyme is not None:
            r = self.structural_rhyme
            parts.append(
                f"- Memory Rhyme (Scene {r.scene_index} — {r.past_felt_stamp}): "
                f"\"{r.past_gist}\" -> {r.relationship_summary}"
            )
        if self.chord_inversion is not None:
            inv = self.chord_inversion
            parts.append(
                f"- Memory Inversion (Scene {inv.scene_index} — {inv.past_felt_stamp}): "
                f"\"{inv.past_gist}\" -> {inv.relationship_summary}"
            )
        return "\n".join(parts)


@dataclass(frozen=True)
class ChordMemoryBank:
    """Persistent memory of past scene chords supporting cross-time rhyming and inversion."""

    entries: tuple[ChordMemoryEntry, ...] = ()
    max_entries: int = 16

    def __post_init__(self):
        if type(self.max_entries) is not int or self.max_entries < 1:
            raise ValueError("max_entries must be a positive int")

    def append(
        self,
        *,
        scene_index: int,
        scene_text_or_gist: str,
        coordinates: Mapping[str, float],
        felt_chord: FeltChord,
        settled: bool = True,
    ) -> "ChordMemoryBank":
        merged_coords = {
            k: round(float(v), 4)
            for k, v in coordinates.items()
            if is_valid_leaf_id(k)
        }
        merged_coords[SALIENCE_LEAF] = felt_chord.salience
        merged_coords[VALENCE_LEAF] = felt_chord.valence
        merged_coords[AROUSAL_LEAF] = felt_chord.arousal
        for k, v in felt_chord.complex_notes.items():
            if is_valid_leaf_id(k):
                merged_coords[k] = round(float(v), 4)

        entry = ChordMemoryEntry(
            scene_index=scene_index,
            scene_gist=_compact_gist(scene_text_or_gist),
            coordinates=merged_coords,
            felt_chord=felt_chord,
            settled=settled,
        )
        kept = tuple(e for e in self.entries if e.scene_index != scene_index) + (entry,)
        return ChordMemoryBank(entries=kept[-self.max_entries :], max_entries=self.max_entries)


def _describe_shared_traits(cur: Mapping[str, float], past: Mapping[str, float]) -> str:
    shared: list[tuple[float, str]] = []
    for axis, phrase in HUMAN_AXIS_NAMES.items():
        cv = float(cur.get(axis, 0.0))
        pv = float(past.get(axis, 0.0))
        if cv >= 0.48 and pv >= 0.48 and abs(cv - pv) <= 0.25:
            shared.append((0.5 * (cv + pv), phrase))
    for k, cv in cur.items():
        if k.startswith("experiential/") and k not in (SALIENCE_LEAF, VALENCE_LEAF, AROUSAL_LEAF, RESONANCE_LEAF):
            pv = float(past.get(k, 0.0))
            if cv >= 0.48 and pv >= 0.48 and abs(cv - pv) <= 0.25:
                shared.append((0.5 * (cv + pv), humanize_feeling_name(k)))
    shared.sort(key=lambda item: -item[0])
    top = [name for _, name in shared[:2]]
    if top:
        return "shares the chord of " + " and ".join(top)
    return "carries a closely matching structural and emotional shape"


def _collect_polarity_flips(
    cur: Mapping[str, float],
    cur_felt: FeltChord,
    past: Mapping[str, float],
    past_felt: FeltChord,
    min_flip: float,
) -> tuple[float, list[str]]:
    flips: list[tuple[float, str]] = []

    def _val(m: Mapping[str, float], key: str, fallback: float = 0.0) -> float:
        return float(m.get(key, fallback))

    c_fair, p_fair = _val(cur, "cultural/justice/fairness"), _val(past, "cultural/justice/fairness")
    if (
        abs(c_fair - p_fair) >= min_flip
        and min(c_fair, p_fair) <= 0.38
        and max(c_fair, p_fair) >= 0.55
    ):
        phrase = (
            "open sharing turned into shut-out exclusion"
            if c_fair < p_fair
            else "shut-out exclusion opened back into shared access"
        )
        flips.append((abs(c_fair - p_fair), phrase))

    c_disc, p_disc = _val(cur, "philosophical/persistence/discreteness"), _val(past, "philosophical/persistence/discreteness")
    if (
        abs(c_disc - p_disc) >= min_flip
        and min(c_disc, p_disc) <= 0.45
        and max(c_disc, p_disc) >= 0.55
    ):
        phrase = (
            "an open space was sealed behind a closed boundary"
            if c_disc > p_disc
            else "a sealed-off boundary gave way"
        )
        flips.append((abs(c_disc - p_disc), phrase))

    c_cont, p_cont = _val(cur, "philosophical/persistence/continuity"), _val(past, "philosophical/persistence/continuity")
    if (
        abs(c_cont - p_cont) >= min_flip
        and min(c_cont, p_cont) <= 0.42
        and max(c_cont, p_cont) >= 0.55
    ):
        phrase = (
            "ongoing flow stalled into stillness"
            if c_cont < p_cont
            else "stalled isolation returned to ongoing flow"
        )
        flips.append((abs(c_cont - p_cont), phrase))

    c_val, p_val = cur_felt.valence, past_felt.valence
    if (
        abs(c_val - p_val) >= max(0.30, min_flip - 0.05)
        and min(c_val, p_val) <= 0.42
        and max(c_val, p_val) >= 0.55
    ):
        phrase = (
            "warm relief chilled into unease"
            if c_val < p_val
            else "bleak chill warmed into relief"
        )
        flips.append((abs(c_val - p_val), phrase))

    # Complex experiential feeling note flips
    past_strong = [
        (k, pv)
        for k, pv in past_felt.complex_notes.items()
        if pv >= 0.48 and (pv - float(cur_felt.complex_notes.get(k, 0.0))) >= 0.28
    ]
    cur_strong = [
        (k, cv)
        for k, cv in cur_felt.complex_notes.items()
        if cv >= 0.48 and (cv - float(past_felt.complex_notes.get(k, 0.0))) >= 0.28
    ]
    if past_strong and cur_strong:
        past_strong.sort(key=lambda kv: -kv[1])
        cur_strong.sort(key=lambda kv: -kv[1])
        p_key, p_score = past_strong[0]
        c_key, c_score = cur_strong[0]
        if p_key != c_key:
            mag = min(1.0, 0.5 * p_score + 0.5 * c_score)
            flips.append(
                (
                    mag,
                    f"felt mood inverted from {humanize_feeling_name(p_key)} into {humanize_feeling_name(c_key)}",
                )
            )

    flips.sort(key=lambda item: -item[0])
    max_mag = flips[0][0] if flips else 0.0
    return max_mag, [desc for _, desc in flips[:2]]


def retrieve_chord_echoes(
    current_coordinates: Mapping[str, float],
    current_felt_chord: FeltChord,
    bank: ChordMemoryBank,
    *,
    current_scene_index: int | None = None,
    min_rhyme_similarity: float = 0.68,
    min_inversion_flip: float = 0.35,
) -> ChordEchoReport:
    """Search past memory chords for the strongest Structural Rhyme and sharp Chord Inversion."""
    if not bank.entries:
        return ChordEchoReport()

    cur_full = dict(current_coordinates)
    cur_full[SALIENCE_LEAF] = current_felt_chord.salience
    cur_full[VALENCE_LEAF] = current_felt_chord.valence
    cur_full[AROUSAL_LEAF] = current_felt_chord.arousal
    for k, v in current_felt_chord.complex_notes.items():
        cur_full[k] = v

    best_rhyme: ChordMatch | None = None
    best_inversion: ChordMatch | None = None

    for entry in bank.entries:
        if current_scene_index is not None and entry.scene_index == current_scene_index:
            continue

        past_full = entry.coordinates
        c_fair = float(cur_full.get("cultural/justice/fairness", 0.0))
        p_fair = float(past_full.get("cultural/justice/fairness", 0.0))
        c_disc = float(cur_full.get("philosophical/persistence/discreteness", 0.0))
        p_disc = float(past_full.get("philosophical/persistence/discreteness", 0.0))
        val_gap = abs(current_felt_chord.valence - entry.felt_chord.valence)

        # 1. Check Structural Rhyme (must align in emotional and openness polarity)
        if val_gap <= 0.28 and abs(c_fair - p_fair) <= 0.32 and abs(c_disc - p_disc) <= 0.35:
            union_keys = [
                k for k in (set(cur_full.keys()) | set(past_full.keys()))
                if max(float(cur_full.get(k, 0.0)), float(past_full.get(k, 0.0))) >= 0.25
            ]
            if union_keys:
                mean_diff = sum(
                    abs(float(cur_full.get(k, 0.0)) - float(past_full.get(k, 0.0)))
                    for k in union_keys
                ) / len(union_keys)
                rhyme_sim = 1.0 - mean_diff
                if current_scene_index is not None and (current_scene_index - entry.scene_index) >= 2:
                    rhyme_sim = min(1.0, rhyme_sim + 0.03)
                if rhyme_sim >= min_rhyme_similarity:
                    if best_rhyme is None or rhyme_sim > best_rhyme.score:
                        best_rhyme = ChordMatch(
                            match_kind="STRUCTURAL_RHYME",
                            scene_index=entry.scene_index,
                            score=round(rhyme_sim, 4),
                            past_gist=entry.scene_gist,
                            past_felt_stamp=entry.felt_chord.short_stamp(),
                            relationship_summary=_describe_shared_traits(cur_full, past_full) + ".",
                        )

        # 2. Check Chord Inversion (shared structural backbone OR enclosure/release mirror + sharp polarity flip)
        max_flip, flip_phrases = _collect_polarity_flips(
            cur_full,
            current_felt_chord,
            past_full,
            entry.felt_chord,
            min_inversion_flip,
        )
        if max_flip >= min_inversion_flip and flip_phrases:
            active_backbone = [
                b for b in BACKBONE_AXES
                if max(float(cur_full.get(b, 0.0)), float(past_full.get(b, 0.0))) >= 0.30
            ]
            backbone_sim = 0.0
            if active_backbone:
                backbone_sim = 1.0 - (
                    sum(abs(float(cur_full.get(b, 0.0)) - float(past_full.get(b, 0.0))) for b in active_backbone)
                    / len(active_backbone)
                )

            c_cont = float(cur_full.get("philosophical/persistence/continuity", 0.0))
            p_cont = float(past_full.get("philosophical/persistence/continuity", 0.0))
            release_mirror = (
                0.5 * max(p_disc, float(past_full.get("philosophical/structure/order", 0.0)))
                + 0.5 * max(c_cont, c_fair, current_felt_chord.valence)
                if (p_disc >= 0.52 and (p_disc - c_disc) >= 0.28 and max(c_cont, current_felt_chord.valence) >= 0.55)
                else 0.0
            )
            enclosure_mirror = (
                0.5 * max(p_cont, p_fair, entry.felt_chord.valence)
                + 0.5 * c_disc
                if (c_disc >= 0.52 and (c_disc - p_disc) >= 0.28 and max(p_cont, p_fair) >= 0.55)
                else 0.0
            )
            anchor = max(backbone_sim, release_mirror, enclosure_mirror)
            if anchor >= 0.58:
                inv_score = round(min(1.0, 0.50 * anchor + 0.50 * max_flip), 4)
                if best_inversion is None or inv_score > best_inversion.score:
                    shared_backbone_names = [
                        HUMAN_AXIS_NAMES[b]
                        for b in active_backbone
                        if float(cur_full.get(b, 0.0)) >= 0.45
                        and float(past_full.get(b, 0.0)) >= 0.45
                        and abs(float(cur_full.get(b, 0.0)) - float(past_full.get(b, 0.0))) <= 0.25
                    ]
                    if shared_backbone_names:
                        prefix = f"shares {shared_backbone_names[0]}, yet "
                    else:
                        prefix = "sharply inverts that earlier state: "
                    best_inversion = ChordMatch(
                        match_kind="CHORD_INVERSION",
                        scene_index=entry.scene_index,
                        score=inv_score,
                        past_gist=entry.scene_gist,
                        past_felt_stamp=entry.felt_chord.short_stamp(),
                        relationship_summary=prefix + "; ".join(flip_phrases) + ".",
                    )

    if (
        best_rhyme is not None
        and best_inversion is not None
        and best_rhyme.scene_index == best_inversion.scene_index
    ):
        if best_inversion.score >= best_rhyme.score:
            best_rhyme = None
        else:
            best_inversion = None

    return ChordEchoReport(
        structural_rhyme=best_rhyme,
        chord_inversion=best_inversion,
    )


def save_chord_memory_bank(path: str | Path, bank: ChordMemoryBank) -> None:
    payload = {
        "schema_version": 1,
        "max_entries": bank.max_entries,
        "entries": [e.to_dict() for e in bank.entries],
    }
    atomic_json(Path(path), payload)


def load_chord_memory_bank(path: str | Path) -> ChordMemoryBank:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("unsupported chord_memory_bank schema_version")
    return ChordMemoryBank(
        entries=tuple(ChordMemoryEntry.from_dict(d) for d in data.get("entries", ())),
        max_entries=int(data.get("max_entries", 16)),
    )
