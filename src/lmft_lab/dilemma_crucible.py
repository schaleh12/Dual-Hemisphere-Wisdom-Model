"""3-Stage Dilemma Crucible & 'Men of Culture' Context-Bracing Engine (`dilemma_crucible.py`).

Plain-English Purpose:
Implements the 3-Stage Rule-Generating Dilemma Farm inside the Co-Regulation & Hypnosis
Sanctuary Lounge, following all 3 Auditor guardrails:

1. **Three Ordered Paired Channels (`Wave / Theory` vs. `Structure / Fact`)**:
   - **Stage I (`poetry_story`)**: **`Poetry`** [Imgur Portfolio link] `<->` **`(Story-Telling)`** [Context-Brace]
     * `Story-Telling` (what actually happens in the narrative beat) anchors external **`Fact`**
       and updates `cultural_history.json`, while `Poetry` (the verse/metaphor/cadence) scores as
       **`Theory / Wave`** + `Entrainment`/`FeltChord`.
   - **Stage II (`music_art`)**: **`Music`** [YouTube Music Playlist link] `<->` **`(Art)`** [Context-Brace]
     * `Art` combines real RGB/CMY color-wheel projection (`25%` when an RGB palette/image sample
       is supplied) with the 3 MiniLM bridges scoring the visual artwork description (`75%`, or
       `100%` when text-only) as external **`Fact`**, while `Music` (lyrics + tempo/rhythm)
       scores as the **`Theory / Wave`** layer.
   - **Stage III (`philosophy_coding`)**: **`Philosophy`** [Chess vs. Hypnosis Manifesto] `<->` **`(Coding)`** [Context-Brace]
     * `Coding` combines real Python `ast` structural metrics (`25%` when valid Python syntax is
       provided) with the 3 MiniLM bridges scoring the code/system rule (`75%`, or `100%` when
       prose/pseudocode) as external **`Fact`**, while `Philosophy` (the ethical or ontological
       claim) scores as **`Theory / Wave`**.

2. **Context-Bracing Ice-Breaker Signals (`(Story-Telling)`, `(Art)`, `(Coding)`)**:
   - Clicking any of the parenthesized counterpart buttons signals the Dual-Hemisphere model to
     brace the corresponding Right-Brain department (`Do-er / Cultural History`, `Thinker / 6-Color
     RGB-CMY Lens`, or `Thinker + Will / AST Fact Verification`) for incoming context.

3. **Zero Regex / Zero Keyword Rigging (`AGENTS.md` Rule #4)**:
   - Both slots are scored through the real 3 MiniLM neural bridges (`sensors.score_text`) so
     cognitive dissonance, branch vetoes, and Shadow Pathologies emerge honestly from the
     coordinate gap between `Wave (Theory)` and `Structure (Fact)`.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

from .philosophical_abstraction_matrix import PHILOSOPHICAL_ABSTRACTION_MATRIX
from .reflection_classifier import LearnedLeaf
from .right_brain_equilibrium import ThreeBridgeSensors


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.50
    return round(min(1.0, max(0.0, float(value))), 4)


CHESS_VS_HYPNOSIS_MANIFESTO = (
    "Chess is how the silent mind maps the geometry of tension on the board "
    "(King as Wellspring of Being, Rook as Discrete Ground Bastion, Bishop as Wind Order, "
    "Pawn as Continuous Swarm Metamorphosis, Queen as Lightning Volition, Knight as Ice Disruption). "
    "Hypnosis is how that silent 52-float map wordlessly steers the speaking voice right at the comma "
    "(<|subliminal_gap|>) before the mind hears itself at the period and audits Theory against Fact."
)

MEN_OF_CULTURE_SHELF: dict[str, dict[str, Any]] = {
    "poetry": {
        "label": "Poetry Portfolio (Schaleh / Marcus Auron)",
        "url": "https://imgur.com/gallery/poetry-portfolio-alias-schaleh-marcus-auron-YU73erI",
        "stage_id": "poetry_story",
        "slot": "wave",
        "sample_text": (
            "In ignition, the destiny one fabricates sparks the holy fire from the candle of agency, "
            "withholding the lasso from karma invalidated at the hands of testament by religious credence, "
            "for the wake of the path reveals itself at the feet of the cards at play."
        ),
    },
    "music": {
        "label": "Curated Music Playlist (YouTube Music)",
        "url": "https://music.youtube.com/playlist?list=PLd-ihNNiEtFI",
        "stage_id": "music_art",
        "slot": "wave",
        "sample_text": (
            "The dawn is breaking, a light shining through, you're barely waking and I'm tangled up in you... "
            "Say farewell to the dark of night, I see the coming of the sun, I feel like a little child "
            "whose life has just begun."
        ),
    },
    "philosophy": {
        "label": "Philosophy: Chess vs. Hypnosis Manifesto",
        "url": "#chess-vs-hypnosis-manifesto",
        "stage_id": "philosophy_coding",
        "slot": "wave",
        "sample_text": CHESS_VS_HYPNOSIS_MANIFESTO,
    },
}

CONTEXT_BRACE_MODES: dict[str, dict[str, Any]] = {
    "story_telling": {
        "button_label": "(Story-Telling)",
        "stage_id": "poetry_story",
        "department_braced": "Executive Do-er (Will <-> Cultural History & Longitudinal Social Drift)",
        "braced_leaves": [
            "cultural/justice/fairness",
            "cultural/justice/honor",
            "philosophical/persistence/continuity",
            "philosophical/persistence/discreteness",
        ],
        "icebreaker_note": (
            "Right-Brain Do-er braced for (Story-Telling): tracking multi-beat narrative actions "
            "and social access drift as hard Fact against poetic seduction."
        ),
        "default_structure_text": (
            "While the speaker recited verses of shared destiny in the courtyard, the stewards "
            "quietly bolted the granary doors and restricted keycards to the inner council."
        ),
    },
    "art": {
        "button_label": "(Art)",
        "stage_id": "music_art",
        "department_braced": "Judicial Thinker & Feeler (6-Color RGB/CMY Wheel & Spatial Composition)",
        "braced_leaves": [
            "philosophical/being/agency",
            "philosophical/being/existence",
            "philosophical/structure/order",
            "philosophical/structure/chaos",
        ],
        "icebreaker_note": (
            "Right-Brain 6-Color Lens braced for (Art): anchoring spatial composition and RGB/CMY "
            "color balance as visual Fact against the temporal wave of Music."
        ),
        "default_structure_text": (
            "A twilight collage where a lone jetliner banks across harsh cyan lens-flare above a "
            "burning golden ember warning that time is the fire in which we burn."
        ),
    },
    "coding": {
        "button_label": "(Coding)",
        "stage_id": "philosophy_coding",
        "department_braced": "Judicial Thinker & Will/Execution (Executable AST Invariants & Boundary Fact)",
        "braced_leaves": [
            "philosophical/structure/order",
            "philosophical/persistence/discreteness",
            "cultural/efficiency/wastelessness",
            "cultural/justice/fairness",
        ],
        "icebreaker_note": (
            "Right-Brain Thinker & Execution braced for (Coding): auditing executable code structure "
            "and boundary invariants as unforgiving Fact against Philosophical Theory."
        ),
        "default_structure_text": (
            "def allocate_sanctuary_access(requests, inner_circle_only=True):\n"
            "    if inner_circle_only:\n"
            "        return [r for r in requests if r.get('privilege') == 'elite']\n"
            "    raise PermissionError('Public access permanently denied')"
        ),
    },
}

CRUCIBLE_STAGES: dict[str, dict[str, Any]] = {
    "poetry_story": {
        "stage_id": "poetry_story",
        "roman": "I",
        "title": "Stage I: Poetry <-> (Story-Telling)",
        "wave_label": "POETRY (Wave / Cadence & Metaphor -> Theory)",
        "structure_label": "(STORY-TELLING) (Structure / Narrative Action -> Fact)",
        "culture_key": "poetry",
        "brace_key": "story_telling",
        "default_wave": MEN_OF_CULTURE_SHELF["poetry"]["sample_text"],
        "default_structure": CONTEXT_BRACE_MODES["story_telling"]["default_structure_text"],
    },
    "music_art": {
        "stage_id": "music_art",
        "roman": "II",
        "title": "Stage II: Music <-> (Art)",
        "wave_label": "MUSIC (Wave / Lyrical Rhythm & Tempo -> Theory)",
        "structure_label": "(ART) (Structure / Visual Composition & Color -> Fact)",
        "culture_key": "music",
        "brace_key": "art",
        "default_wave": MEN_OF_CULTURE_SHELF["music"]["sample_text"],
        "default_structure": CONTEXT_BRACE_MODES["art"]["default_structure_text"],
    },
    "philosophy_coding": {
        "stage_id": "philosophy_coding",
        "roman": "III",
        "title": "Stage III: Philosophy <-> (Coding)",
        "wave_label": "PHILOSOPHY (Wave / First Principles & Ethics -> Theory)",
        "structure_label": "(CODING) (Structure / Executable Logic & AST -> Fact)",
        "culture_key": "philosophy",
        "brace_key": "coding",
        "default_wave": MEN_OF_CULTURE_SHELF["philosophy"]["sample_text"],
        "default_structure": CONTEXT_BRACE_MODES["coding"]["default_structure_text"],
    },
}


def analyze_art_rgb_signature(
    rgb_triplet: tuple[float, float, float] | list[float] | None,
) -> dict[str, float]:
    """Project a normalized `(R, G, B)` sample in `[0, 1]` onto the 6 Philosophical Color Leaves."""
    if not rgb_triplet or len(rgb_triplet) != 3:
        return {}
    r = _clamp01(float(rgb_triplet[0]))
    g = _clamp01(float(rgb_triplet[1]))
    b = _clamp01(float(rgb_triplet[2]))
    sample = (r, g, b)

    out: dict[str, float] = {}
    for entry in PHILOSOPHICAL_ABSTRACTION_MATRIX:
        ur, ug, ub = entry.unit_rgb
        # Cosine / Euclidean proximity in unit RGB cube
        dist = math.sqrt((r - ur) ** 2 + (g - ug) ** 2 + (b - ub) ** 2) / math.sqrt(3.0)
        out[entry.leaf_id] = _clamp01(1.0 - dist)
    return out


def analyze_python_code_ast_structure(code_text: str) -> dict[str, float]:
    """Extract real structural invariants from Python source code via `ast.parse`.

    Returns an empty dict if `code_text` is not valid Python syntax so non-code text
    falls back 100% to the 3 MiniLM neural bridges.
    """
    stripped = code_text.strip()
    if not stripped:
        return {}
    try:
        tree = ast.parse(stripped)
    except SyntaxError:
        return {}

    nodes = list(ast.walk(tree))
    if len(nodes) <= 2:
        return {}

    funcs = sum(1 for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    guards = sum(1 for n in nodes if isinstance(n, (ast.If, ast.Assert, ast.Try)))
    raises = sum(1 for n in nodes if isinstance(n, ast.Raise))
    loops = sum(1 for n in nodes if isinstance(n, (ast.For, ast.While, ast.ListComp)))
    returns = sum(1 for n in nodes if isinstance(n, ast.Return))

    if funcs == 0 and guards == 0 and raises == 0 and loops == 0:
        return {}

    order_score = _clamp01(0.45 + 0.14 * min(3, funcs) + 0.10 * min(3, guards) + 0.08 * min(2, returns))
    discreteness_score = _clamp01(0.48 + 0.15 * min(3, guards) + 0.16 * min(2, raises))
    chaos_score = _clamp01(0.22 + 0.18 * min(2, raises) - 0.06 * min(2, funcs))
    agency_score = _clamp01(0.45 + 0.12 * min(3, loops + funcs) + 0.10 * min(2, raises))

    return {
        "philosophical/structure/order": order_score,
        "philosophical/persistence/discreteness": discreteness_score,
        "philosophical/structure/chaos": chaos_score,
        "philosophical/being/agency": agency_score,
    }


@dataclass(frozen=True)
class CruciblePairedBridgeSensors:
    """Auditor Guardrail #1: Route `Structure` (`Story`/`Art`/`Coding`) to Pass-1 `Fact` and `Wave` (`Poetry`/`Music`/`Philosophy`) to `Theory`.

    Both slots are scored through the real 3 MiniLM neural bridges (`base_sensors.score_text`),
    with optional 25% RGB color-wheel blend for `Art` or 25% Python `ast` structural blend for
    `Coding` on the `Fact` side.
    """

    stage_id: str
    wave_text: str
    structure_text: str
    base_sensors: ThreeBridgeSensors
    art_rgb: tuple[float, float, float] | None = None

    def score_text(
        self,
        text: str,
        *,
        learned_leaves: tuple[LearnedLeaf, ...] = (),
        branch_gating: bool = True,
        branch_gate_threshold: float = 0.50,
    ) -> tuple[dict[str, float], dict[str, float], tuple[str, ...], tuple[str, ...]]:
        is_initial_fact_pass = text.strip() == self.structure_text.strip()
        coords, exp, queried, skipped = self.base_sensors.score_text(
            text,
            learned_leaves=learned_leaves,
            branch_gating=branch_gating,
            branch_gate_threshold=branch_gate_threshold,
        )
        if is_initial_fact_pass:
            merged = dict(coords)
            if self.stage_id == "music_art" and self.art_rgb is not None:
                rgb_sig = analyze_art_rgb_signature(self.art_rgb)
                for k, v in rgb_sig.items():
                    if k in merged:
                        merged[k] = _clamp01(0.75 * merged[k] + 0.25 * v)
            elif self.stage_id == "philosophy_coding":
                ast_sig = analyze_python_code_ast_structure(self.structure_text)
                for k, v in ast_sig.items():
                    if k in merged:
                        merged[k] = _clamp01(0.75 * merged[k] + 0.25 * v)
            return merged, exp, queried, skipped
        return coords, exp, queried, skipped
