"""Two-Layer Contemplation (Clarity / Confidence) Context Ledger and Raw Thought Stream.

Plain-English purpose:
Manages the conscious Left Brain's continuous train of thought under
`Psychological -> Contemplation (Clarity / Confidence)` using two on-disk files:
1. Raw Thought Stream (`.jsonl` on disk):
   Archives every pass's full Left-Brain self-talk draft, self-critique, and
   Right-Brain subliminal annotation for audit and future Sleep-Mode review.
2. Live Context Ledger (`.json` on disk and fed to `Qwen3-4B`):
   Keeps `Qwen3-4B` sharp and never context-drunk by splitting memory into:
   - Settled Certainties (bounded 1-line compacted takeaways + felt chord stamp;
     raw draft blather stripped once settled).
   - Active Uncertainties (1–2 open unresolved questions stamped with the Right
     Brain's subliminal red-pen margin tag showing Clarity, Confidence, and the
     Feeler's felt reaction so the Left Brain can debate and revise its prior hunch).
"""

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path

from .chord_memory import FeltChord
from .runtime_store import atomic_json

CONTEMPLATION_BRANCH = "psychological/contemplation"


def _compact_one_line(text: str, limit: int = 125) -> str:
    cleaned = " ".join(text.strip().split())
    sentences = [s.strip() for s in cleaned.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    first = sentences[0] if sentences else cleaned
    if len(first) > limit:
        return first[: limit - 3].rstrip() + "..."
    return first


def _level_word(score: float) -> str:
    if score < 0.30:
        return "low"
    if score < 0.58:
        return "partial"
    return "clear"


@dataclass(frozen=True)
class SubliminalMarginTag:
    """Right-Brain subliminal red-pen annotation stamped onto an unresolved hunch."""

    clarity: float
    confidence: float
    felt_reaction: str
    red_pen_prompt: str

    def __post_init__(self):
        for label, val in (("clarity", self.clarity), ("confidence", self.confidence)):
            if type(val) not in (int, float) or not math.isfinite(val) or not (0.0 <= val <= 1.0):
                raise ValueError(f"{label} must be a finite float in [0, 1]")
        if not isinstance(self.felt_reaction, str) or not self.felt_reaction.strip():
            raise ValueError("felt_reaction must be a nonempty string")
        if not isinstance(self.red_pen_prompt, str) or not self.red_pen_prompt.strip():
            raise ValueError("red_pen_prompt must be a nonempty string")

    def to_dict(self) -> dict:
        return {
            "clarity": round(float(self.clarity), 4),
            "confidence": round(float(self.confidence), 4),
            "felt_reaction": self.felt_reaction,
            "red_pen_prompt": self.red_pen_prompt,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "SubliminalMarginTag":
        return cls(
            clarity=float(data["clarity"]),
            confidence=float(data["confidence"]),
            felt_reaction=str(data["felt_reaction"]),
            red_pen_prompt=str(data["red_pen_prompt"]),
        )

    def render_for_left_brain(self) -> str:
        c_word = _level_word(self.clarity)
        conf_word = _level_word(self.confidence)
        return (
            f"[Red-Pen Margin | Clarity: {c_word} ({self.clarity:.2f}), "
            f"Confidence: {conf_word} ({self.confidence:.2f}), "
            f"Felt reaction: {self.felt_reaction} | Note: {self.red_pen_prompt}]"
        )


def build_subliminal_margin_tag(
    *,
    clarity: float,
    confidence: float,
    felt_chord: FeltChord,
    has_dissonance: bool = False,
    has_feeler_veto: bool = False,
    has_exclusion_tension: bool = False,
    prior_hunch_exists: bool = False,
) -> SubliminalMarginTag:
    """Translate Right-Brain disagreement into a concise, domain-general coordinate margin note."""
    felt_reaction = felt_chord.short_stamp()
    if has_dissonance:
        red_pen = (
            f"Candidate framing clashed with depressed external coordinates "
            f"(clarity={clarity:.2f}, confidence={confidence:.2f}); ground the revision in observed facts."
        )
    elif has_feeler_veto:
        red_pen = (
            f"Candidate framing left low-valence experiential tension unresolved ({felt_reaction}); "
            f"revise to account for the felt impact."
        )
    elif has_exclusion_tension and prior_hunch_exists:
        red_pen = (
            f"High control/structure is colliding with a depressed value coordinate ({felt_reaction}); "
            f"sharpen your earlier draft around that imbalance."
        )
    elif has_exclusion_tension:
        red_pen = (
            f"Cross-branch value tension detected ({felt_reaction}); "
            f"weigh the structural control against the human impact."
        )
    elif prior_hunch_exists:
        red_pen = (
            f"Prior draft left low clarity ({clarity:.2f}); "
            f"question what shifted in the scene and state a sharper revision."
        )
    else:
        red_pen = (
            f"Known rules do not cleanly settle this coordinate profile ({felt_reaction}); "
            f"weigh the felt atmosphere against the memory echo."
        )

    return SubliminalMarginTag(
        clarity=round(float(clarity), 4),
        confidence=round(float(confidence), 4),
        felt_reaction=felt_reaction,
        red_pen_prompt=red_pen,
    )


@dataclass(frozen=True)
class RawThoughtRecord:
    """One uncompacted self-talk entry archived to `.jsonl` under Psychological -> Contemplation."""

    scene_index: int
    pass_index: int
    clarity: float
    confidence: float
    raw_self_talk_draft: str
    felt_chord_stamp: str
    subliminal_margin_tag: SubliminalMarginTag
    epistemic_state: str
    settled_on_this_pass: bool
    prior_hunch_debated: str | None = None
    branch_owner: str = CONTEMPLATION_BRANCH

    def to_dict(self) -> dict:
        return {
            "branch_owner": self.branch_owner,
            "scene_index": self.scene_index,
            "pass_index": self.pass_index,
            "clarity": round(float(self.clarity), 4),
            "confidence": round(float(self.confidence), 4),
            "raw_self_talk_draft": self.raw_self_talk_draft,
            "prior_hunch_debated": self.prior_hunch_debated,
            "felt_chord_stamp": self.felt_chord_stamp,
            "subliminal_margin_tag": self.subliminal_margin_tag.to_dict(),
            "epistemic_state": self.epistemic_state,
            "settled_on_this_pass": self.settled_on_this_pass,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "RawThoughtRecord":
        return cls(
            scene_index=int(data["scene_index"]),
            pass_index=int(data["pass_index"]),
            clarity=float(data["clarity"]),
            confidence=float(data["confidence"]),
            raw_self_talk_draft=str(data["raw_self_talk_draft"]),
            prior_hunch_debated=data.get("prior_hunch_debated"),
            felt_chord_stamp=str(data["felt_chord_stamp"]),
            subliminal_margin_tag=SubliminalMarginTag.from_dict(data["subliminal_margin_tag"]),
            epistemic_state=str(data["epistemic_state"]),
            settled_on_this_pass=bool(data["settled_on_this_pass"]),
            branch_owner=str(data.get("branch_owner", CONTEMPLATION_BRANCH)),
        )


def append_raw_thought_record(path: str | Path, record: RawThoughtRecord) -> None:
    """Append one raw Left-Brain self-talk pass to the on-disk `.jsonl` archive."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
    with target.open("a", encoding="utf-8") as fh:
        fh.write(line)


def load_raw_thought_stream(path: str | Path) -> tuple[RawThoughtRecord, ...]:
    """Read back all archived raw self-talk entries from the `.jsonl` stream."""
    target = Path(path)
    if not target.exists():
        return ()
    records: list[RawThoughtRecord] = []
    for raw_line in target.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        records.append(RawThoughtRecord.from_dict(json.loads(stripped)))
    return tuple(records)


@dataclass(frozen=True)
class SettledCertainty:
    """Compacted 1-line takeaway + felt chord stamp once a scene settles."""

    scene_index: int
    takeaway: str
    felt_chord_stamp: str
    clarity: float
    confidence: float

    def to_dict(self) -> dict:
        return {
            "scene_index": self.scene_index,
            "takeaway": self.takeaway,
            "felt_chord_stamp": self.felt_chord_stamp,
            "clarity": round(float(self.clarity), 4),
            "confidence": round(float(self.confidence), 4),
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "SettledCertainty":
        return cls(
            scene_index=int(data["scene_index"]),
            takeaway=str(data["takeaway"]),
            felt_chord_stamp=str(data["felt_chord_stamp"]),
            clarity=float(data["clarity"]),
            confidence=float(data["confidence"]),
        )


@dataclass(frozen=True)
class ActiveUncertainty:
    """Bounded unresolved hunch + subliminal red-pen margin note kept open for self-debate."""

    scene_index: int
    prior_hunch: str
    open_question: str
    margin_tag: SubliminalMarginTag

    def to_dict(self) -> dict:
        return {
            "scene_index": self.scene_index,
            "prior_hunch": self.prior_hunch,
            "open_question": self.open_question,
            "margin_tag": self.margin_tag.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "ActiveUncertainty":
        return cls(
            scene_index=int(data["scene_index"]),
            prior_hunch=str(data["prior_hunch"]),
            open_question=str(data["open_question"]),
            margin_tag=SubliminalMarginTag.from_dict(data["margin_tag"]),
        )


@dataclass(frozen=True)
class ContemplationLedger:
    """Live `.json` context ledger separating Settled Certainties from Active Uncertainties."""

    settled_certainties: tuple[SettledCertainty, ...] = ()
    active_uncertainties: tuple[ActiveUncertainty, ...] = ()
    max_certainties: int = 5
    max_uncertainties: int = 2
    branch_owner: str = CONTEMPLATION_BRANCH

    def __post_init__(self):
        if type(self.max_certainties) is not int or self.max_certainties < 1:
            raise ValueError("max_certainties must be a positive int")
        if type(self.max_uncertainties) is not int or self.max_uncertainties < 1:
            raise ValueError("max_uncertainties must be a positive int")

    def record_settled(
        self,
        *,
        scene_index: int,
        takeaway_source: str,
        felt_chord: FeltChord,
        clarity: float,
        confidence: float,
        resolved_uncertainty_scenes: tuple[int, ...] = (),
    ) -> "ContemplationLedger":
        """Compact a settled scene into a 1-line certainty and clear any resolved uncertainties."""
        cert = SettledCertainty(
            scene_index=scene_index,
            takeaway=_compact_one_line(takeaway_source),
            felt_chord_stamp=felt_chord.short_stamp(),
            clarity=round(float(clarity), 4),
            confidence=round(float(confidence), 4),
        )
        cleared_ids = {scene_index, *resolved_uncertainty_scenes}
        kept_certs = tuple(c for c in self.settled_certainties if c.scene_index != scene_index) + (cert,)
        remaining_unc = tuple(u for u in self.active_uncertainties if u.scene_index not in cleared_ids)
        return ContemplationLedger(
            settled_certainties=kept_certs[-self.max_certainties :],
            active_uncertainties=remaining_unc[-self.max_uncertainties :],
            max_certainties=self.max_certainties,
            max_uncertainties=self.max_uncertainties,
            branch_owner=self.branch_owner,
        )

    def record_uncertainty(
        self,
        *,
        scene_index: int,
        raw_hunch: str,
        open_question: str,
        margin_tag: SubliminalMarginTag,
    ) -> "ContemplationLedger":
        """Store or revise an unresolved hunch stamped with the Right Brain's red-pen margin tag."""
        unc = ActiveUncertainty(
            scene_index=scene_index,
            prior_hunch=_compact_one_line(raw_hunch, limit=135),
            open_question=_compact_one_line(open_question, limit=135),
            margin_tag=margin_tag,
        )
        others = tuple(u for u in self.active_uncertainties if u.scene_index != scene_index)
        updated = (others + (unc,))[-self.max_uncertainties :]
        return ContemplationLedger(
            settled_certainties=self.settled_certainties,
            active_uncertainties=updated,
            max_certainties=self.max_certainties,
            max_uncertainties=self.max_uncertainties,
            branch_owner=self.branch_owner,
        )

    def render_for_left_brain(self) -> str:
        """Render clean, bounded context for Qwen3-4B with zero raw code paths or blather."""
        cert_lines = [
            f"- Scene {c.scene_index} [{c.felt_chord_stamp}]: {c.takeaway}"
            for c in self.settled_certainties
        ] or ["(none settled yet)"]

        unc_lines = [
            f"- Scene {u.scene_index} open hunch: \"{u.prior_hunch}\"\n"
            f"  Open question: {u.open_question}\n"
            f"  {u.margin_tag.render_for_left_brain()}"
            for u in self.active_uncertainties
        ] or ["(no unresolved hunches right now)"]

        return (
            "SETTLED CERTAINTIES (compacted 1-line takeaways — do not repeat wording):\n"
            + "\n".join(cert_lines)
            + "\n\nACTIVE UNCERTAINTIES (open questions for self-critique & revision):\n"
            + "\n".join(unc_lines)
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "branch_owner": self.branch_owner,
            "max_certainties": self.max_certainties,
            "max_uncertainties": self.max_uncertainties,
            "settled_certainties": [c.to_dict() for c in self.settled_certainties],
            "active_uncertainties": [u.to_dict() for u in self.active_uncertainties],
        }


def save_contemplation_ledger(path: str | Path, ledger: ContemplationLedger) -> None:
    """Save the live Contemplation Context Ledger atomically as `.json`."""
    atomic_json(Path(path), ledger.to_dict())


def load_contemplation_ledger(path: str | Path) -> ContemplationLedger:
    """Load the live Contemplation Context Ledger from `.json`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("unsupported contemplation_ledger schema_version")
    return ContemplationLedger(
        settled_certainties=tuple(
            SettledCertainty.from_dict(c) for c in data.get("settled_certainties", ())
        ),
        active_uncertainties=tuple(
            ActiveUncertainty.from_dict(u) for u in data.get("active_uncertainties", ())
        ),
        max_certainties=int(data.get("max_certainties", 5)),
        max_uncertainties=int(data.get("max_uncertainties", 2)),
        branch_owner=str(data.get("branch_owner", CONTEMPLATION_BRANCH)),
    )
