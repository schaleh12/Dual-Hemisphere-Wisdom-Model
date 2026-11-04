"""Local Desktop App: Chess Kibitzing Club & Co-Regulation Sanctuary (`sanctuary_club_app.py`).

Plain-English Purpose:
Provides a clean, dark-minimalist ("Antigravity-sleek, zero-fruitcake") local desktop application
powered by our Dual-Hemisphere LMFT engine (`run_three_legged_turn`), implementing all 4 Auditor
guardrails:
1. **Un-Rigged Sanctuary Pacing & Leading (`apply_coregulation_lead_to_chord`)**:
   - Keeps `vibe.coordinates` (`Fact`) 100% locked to the real MiniLM reading of the user's message
     so Clause 1 honestly *Paces* their state, and applies a graduated `alpha = 0.25` step on the
     spliced 52-float `SubliminalChordVector` right at the comma (`<|subliminal_gap|>`) so Clause 2
     *Leads* toward equilibrium (`tilt_reset`, `deep_focus`, or `night_wind_down`).
2. **Board-Anchored `Fact` vs. Chat `Theory` (`BoardAnchoredBridgeSensors`)**:
   - Uses `python-chess` (`evaluate_chess_board_posture`) to anchor the 6 Philosophical `Fact`
     coordinates to the live chessboard while scoring Cultural/Experiential leaves and proposed
     plans (`Theory`) through the 3 MiniLM bridges.
3. **Human-in-the-Loop Sleep Tagging (`[Coach]` & `[Savor Vibe]`) + Guest Kibitzer Chair**:
   - Lets the user flag any turn with **`[Coach / Queue for Sleep]`** (packaging an `UnresolvedSleepScene`
     for Nighttime REM Sleep) or bookmark a great turn with **`[Savor / Keep Vibe]`** (harvesting its
     chord into the 3-branch archives).
   - Includes a **Guest Kibitzer Chair (Gemini / External Club Member)** that scores guest comments
     through the exact same 3 MiniLM bridges + 6-Color / Shadow Pathology lens before our model replies.
4. **Single GPU Lock (`threading.Lock`) + `Cycle 2` Default + Fast `< 2s` Startup**:
   - Defaults to `models/left_brain_lora_cycle2/` (`sleep_cycle_index = 2`), lazy-loads `Qwen/Qwen3-4B`
     in the background so the desktop window (`msedge --app=http://127.0.0.1:8420`) opens in `< 2s`,
     and wraps GPU inference and **`Run REM Sleep`** in a single `threading.Lock()`.
"""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any
import urllib.error
import urllib.request
import webbrowser

import chess

from .birth import BirthState, create_birth_state
from .branch_archives import (
    EXECUTIVE_BRANCH,
    JUDICIAL_BRANCH,
    LEGISLATIVE_BRANCH,
    BranchClauseEntry,
    ThreeBranchArchives,
    harvest_into_branch_archives,
)
from .chess_kibitz_bridge import (
    SANCTUARY_MODES,
    ChessBoardPosture,
    evaluate_chess_board_posture,
    merge_board_fact_with_chat_sensors,
)
from .chord_memory import ChordMemoryBank, compile_felt_chord
from .contemplation_ledger import (
    ContemplationLedger,
    load_contemplation_ledger,
    save_contemplation_ledger,
)
from .runtime_store import atomic_json
from .neural_bridge_classifiers import (
    DEFAULT_COMPLEX_FEELING_LEAVES,
    NeuralThreeBridgeSuite,
)
from .philosophical_abstraction_matrix import (
    PHILOSOPHICAL_SHADOW_LEAVES,
    compute_philosophical_color_harmony,
    detect_shadow_pathologies,
)
from .reflection_classifier import (
    AROUSAL_LEAF,
    FOUNDATIONAL_LEAVES,
    NOVELTY_LEAF,
    RESONANCE_LEAF,
    RETENTION_LEAF,
    SALIENCE_LEAF,
    VALENCE_LEAF,
    ContinuousMemoryState,
    FuzzyConceptRule,
    LearnedLeaf,
    leaf_tier,
    load_learned_leaves,
    load_ruleset,
    save_ruleset,
)
from .right_brain_equilibrium import (
    EXPERIENTIAL_SENSOR_LEAVES,
    RIGHT_BRAIN_WILL_EDGES,
    RIGHT_BRAIN_WILL_VERSION,
    SettledDiary,
    ThreeBridgeSensors,
)
from .sleep_consolidation import (
    UnresolvedSleepScene,
    _anchor_target_leaves_from_cluster,
    _average_cluster_coordinates,
    _cluster_key_for_unresolved,
    _derive_concept_tags,
    refine_colliding_rules_during_sleep,
    run_sleep_consolidation,
)
from .three_legged_runtime import (
    SubliminalGapLeftBrain,
    ThreeLeggedTurnResult,
    run_three_legged_turn,
)
from .virgin_baseline import load_virgin_baseline_state

PRESET_CHESS_POSITIONS: dict[str, dict[str, str]] = {
    "start": {
        "label": "Standard Starting Position",
        "fen": chess.STARTING_FEN,
    },
    "exposed_king_attack": {
        "label": "Exposed King vs. Reckless Attack (Psychosis Test)",
        "fen": "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N1P/PPP2PP1/R1BQK2R w KQ - 1 7",
    },
    "naked_king_siege": {
        "label": "Stripped King Under Siege (Shadow Paranoia / Psychosis Test)",
        "fen": "4rrk1/pppb1ppp/2n5/3p4/3P1q2/2PB1N2/PP3PP/R2Q1K1R w - - 2 14".replace("\u10c3", "P"),
    },
    "knight_outpost_chaos": {
        "label": "Sharp Tactical Tension (Green Knight / Ice)",
        "fen": "r1bqkb1r/pp2pppp/2np1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",
    },
    "rook_seventh_bastion": {
        "label": "7th-Rank Rook Bastion Endgame (Yellow Rook / Ground)",
        "fen": "6k1/1R3ppp/4p3/8/8/4P3/5PPP/6K1 w - - 0 28",
    },
}


def _default_fallback_sensors() -> ThreeBridgeSensors:
    """Fast deterministic fallback sensors used only when neural bridge files are absent (e.g. unit tests)."""

    def _will(text: str, children: tuple[LearnedLeaf, ...] = ()) -> dict[str, float]:
        low = text.lower()
        fair = 0.20 if any(w in low for w in ("lock", "forbid", "deny", "seize", "bar")) else 0.65
        power = 0.78 if any(w in low for w in ("attack", "sacrifice", "force", "seize", "lock")) else 0.42
        out = {
            "cultural/efficiency/power": power,
            "cultural/efficiency/wastelessness": 0.62,
            "cultural/justice/honor": 0.58,
            "cultural/justice/fairness": fair,
            "cultural/freedom/self-sufficiency": 0.55,
            "cultural/freedom/needlessness": 0.68 if "calm" in low or "rest" in low else 0.45,
        }
        for c in children:
            out[c.leaf_id] = 0.50
        return out

    def _truth(text: str, children: tuple[LearnedLeaf, ...] = ()) -> dict[str, float]:
        low = text.lower()
        agency = 0.85 if any(w in low for w in ("queen", "sacrifice", "all-out", "attack", "strike")) else 0.48
        existence = 0.18 if any(w in low for w in ("ignore king", "forget defense", "reckless")) else 0.65
        order = 0.76 if any(w in low for w in ("castle", "defend", "steady", "structure", "calm")) else 0.52
        chaos = 0.74 if any(w in low for w in ("chaos", "shatter", "fork", "wild", "panic")) else 0.28
        out = {
            "philosophical/being/agency": agency,
            "philosophical/being/existence": existence,
            "philosophical/structure/order": order,
            "philosophical/structure/chaos": chaos,
            "philosophical/persistence/continuity": 0.62,
            "philosophical/persistence/discreteness": 0.54,
        }
        for c in children:
            out[c.leaf_id] = 0.45
        return out

    def _feeler(text: str, children: tuple[LearnedLeaf, ...] = ()) -> dict[str, float]:
        low = text.lower()
        stressed = any(w in low for w in ("tilt", "angry", "racing", "panic", "frustrated", "blunder"))
        out = {
            SALIENCE_LEAF: 0.72 if stressed else 0.48,
            VALENCE_LEAF: 0.28 if stressed else 0.68,
            AROUSAL_LEAF: 0.76 if stressed else 0.36,
        }
        for c in children:
            out[c.leaf_id] = 0.65 if "austere_sanctuary" in c.leaf_id and not stressed else 0.30
        return out

    return ThreeBridgeSensors(
        will_cultural=_will,
        truth_philosophical=_truth,
        contemplation_experiential=_feeler,
    )


def _default_fallback_rules() -> tuple[FuzzyConceptRule, ...]:
    return (
        FuzzyConceptRule(
            rule_id="rule_grounded_coordination",
            concept_name="grounded_coordination",
            description="Balanced structure, sheltered King existence, and fair cooperation.",
            target_leaves={
                "philosophical/being/existence": 0.62,
                "philosophical/structure/order": 0.60,
            },
        ),
        FuzzyConceptRule(
            rule_id="rule_sanctuary_equilibrium",
            concept_name="sanctuary_equilibrium",
            description="Calm co-regulation with steady order and low chaos.",
            target_leaves={
                "cultural/justice/fairness": 0.60,
                "philosophical/structure/order": 0.58,
            },
        ),
    )


class SanctuaryClubRuntime:
    """Thread-safe controller for the Chess Kibitzing Club + Co-Regulation Sanctuary app."""

    def __init__(
        self,
        repo_root: str | Path | None = None,
        *,
        state_dir: str | Path | None = None,
        birth_state: BirthState | None = None,
        sensors: ThreeBridgeSensors | None = None,
        rules: tuple[FuzzyConceptRule, ...] | None = None,
        learned_leaves: tuple[LearnedLeaf, ...] | None = None,
        left_brain: SubliminalGapLeftBrain | None = None,
        teacher: Any | None = None,
        lazy_load_gpu: bool = True,
    ) -> None:
        self.repo_root = (
            Path(repo_root).resolve()
            if repo_root is not None
            else Path(__file__).resolve().parents[2]
        )
        self.state_dir = (
            Path(state_dir).resolve()
            if state_dir is not None
            else (self.repo_root / "outputs" / "club_user_state")
        )
        self.teacher = teacher
        self.gpu_lock = threading.Lock()
        self.rem_sleep_in_progress = False
        self.left_brain_loading = False
        self.left_brain_status = "LAZY_STANDBY"
        self.active_lineage = "schooled"  # "schooled" (Cycle 2 + user state) or "virgin" (Cycle 0 snapshot)
        self.active_checkpoint_label = "Cycle 2 (models/left_brain_lora_cycle2)"
        self.sleep_cycle_index = 2

        self.birth_state = birth_state or create_birth_state(
            "lmft-sanctuary-club",
            birth_event_id="club-birth-0",
            random_seed=42,
        )

        # Load Right-Brain rules & learned leaves (preferring persisted state_dir when present)
        if rules is not None:
            self.rules = tuple(rules)
        else:
            self.rules = self._load_default_rules()

        if learned_leaves is not None:
            self.learned_leaves = tuple(learned_leaves)
        else:
            self.learned_leaves = self._load_default_learned_leaves()

        # Load 3 MiniLM bridges (~35 ms)
        if sensors is not None:
            self.sensors = sensors
        else:
            self.sensors = self._load_default_sensors()

        self.left_brain: SubliminalGapLeftBrain | None = left_brain
        if self.left_brain is not None:
            self.left_brain_status = "READY"
            if self.left_brain.callosum_metadata is not None:
                self.sleep_cycle_index = int(self.left_brain.callosum_metadata.sleep_cycle_index)

        # Live session state
        self.board = chess.Board()
        self.move_history_san: list[str] = []
        self.board_posture: ChessBoardPosture = evaluate_chess_board_posture(self.board)
        self.active_room: str = "chess"  # "chess" or "sanctuary"
        self.sanctuary_mode: str = "tilt_reset"
        self.scene_counter: int = 0

        self.memory: ContinuousMemoryState | None = None
        self.diary = SettledDiary()
        self.ledger = self._load_initial_ledger()
        self.chord_memory = ChordMemoryBank()
        self.archives = self._load_initial_archives()

        self.sleep_queue: list[UnresolvedSleepScene] = []
        self.savored_turn_ids: list[int] = self._load_initial_savored_ids()
        self.chat_messages: list[dict[str, Any]] = []
        self._turn_records_by_id: dict[int, dict[str, Any]] = {}

        if lazy_load_gpu and self.left_brain is None:
            # Start background warm-up thread so UI opens in < 1s
            t = threading.Thread(target=self._background_warm_left_brain, daemon=True)
            t.start()

    def _load_default_rules(self) -> tuple[FuzzyConceptRule, ...]:
        user_ruleset = self.state_dir / "club_ruleset.json"
        candidates = [user_ruleset] + [
            self.repo_root / rel
            for rel in (
                "outputs/rem_sleep_consolidated_ruleset.json",
                "outputs/sleep_consolidated_ruleset.json",
                "outputs/continuous_tiered_ruleset.json",
            )
        ]
        for p in candidates:
            if p.exists():
                try:
                    loaded = load_ruleset(p)
                    if loaded:
                        return loaded
                except Exception:
                    pass
        return _default_fallback_rules()

    def _load_default_learned_leaves(self) -> tuple[LearnedLeaf, ...]:
        base: list[LearnedLeaf] = []
        user_ruleset = self.state_dir / "club_ruleset.json"
        candidates = [user_ruleset] + [
            self.repo_root / rel
            for rel in (
                "outputs/rem_sleep_consolidated_ruleset.json",
                "outputs/sleep_consolidated_ruleset.json",
                "outputs/continuous_tiered_ruleset.json",
            )
        ]
        for p in candidates:
            if p.exists():
                try:
                    base.extend(load_learned_leaves(p))
                    break
                except Exception:
                    pass
        existing_ids = {l.leaf_id for l in base}
        for leaf in DEFAULT_COMPLEX_FEELING_LEAVES + PHILOSOPHICAL_SHADOW_LEAVES:
            if leaf.leaf_id not in existing_ids:
                base.append(leaf)
                existing_ids.add(leaf.leaf_id)
        return tuple(base)

    def _load_initial_archives(self) -> ThreeBranchArchives:
        user_arch_dir = self.state_dir / "branch_archives"
        if (user_arch_dir / "philosophical_lexicon.json").exists():
            try:
                return ThreeBranchArchives.load_all(user_arch_dir)
            except Exception:
                pass
        default_arch_dir = self.repo_root / "outputs" / "branch_archives"
        if (default_arch_dir / "philosophical_lexicon.json").exists():
            try:
                return ThreeBranchArchives.load_all(default_arch_dir)
            except Exception:
                pass
        return ThreeBranchArchives()

    def _load_initial_ledger(self) -> ContemplationLedger:
        user_ledger = self.state_dir / "contemplation_ledger.json"
        if user_ledger.exists():
            try:
                return load_contemplation_ledger(user_ledger)
            except Exception:
                pass
        return ContemplationLedger()

    def _load_initial_savored_ids(self) -> list[int]:
        meta_path = self.state_dir / "club_meta.json"
        if meta_path.exists():
            try:
                raw = json.loads(meta_path.read_text(encoding="utf-8"))
                self.scene_counter = max(self.scene_counter, int(raw.get("scene_counter", 0)))
                return [int(x) for x in raw.get("savored_turn_ids", [])]
            except Exception:
                pass
        return []

    def _persist_user_state(self) -> None:
        """Persist ruleset, 3-branch archives, contemplation ledger, and savored turn metadata to disk."""
        if self.active_lineage == "virgin":
            # Never overwrite the user's schooled state directory while inspecting the Cycle 0 Virgin Baseline
            return
        self.state_dir.mkdir(parents=True, exist_ok=True)
        save_ruleset(
            self.state_dir / "club_ruleset.json",
            tuple(self.rules),
            tuple(self.learned_leaves),
        )
        self.archives.save_all(self.state_dir / "branch_archives")
        save_contemplation_ledger(self.state_dir / "contemplation_ledger.json", self.ledger)
        atomic_json(
            self.state_dir / "club_meta.json",
            {
                "savored_turn_ids": list(self.savored_turn_ids),
                "scene_counter": int(self.scene_counter),
            },
        )

    def switch_brain_lineage(self, lineage: str) -> dict[str, Any]:
        """Switch between the `schooled` brain (Cycle 2 + user coaching) and `virgin` (Cycle 0 Core Architecture snapshot)."""
        mode = lineage.strip().lower()
        if mode not in ("schooled", "virgin"):
            raise ValueError(f"Unknown lineage mode {lineage!r}; expected 'schooled' or 'virgin'.")
        if mode == "virgin":
            virgin = load_virgin_baseline_state(repo_root=self.repo_root)
            self.active_lineage = "virgin"
            self.sleep_cycle_index = 0
            self.active_checkpoint_label = "Virgin Baseline (Cycle 0 — Core Architecture)"
            self.rules = virgin.rules
            self.learned_leaves = virgin.learned_leaves
            self.archives = virgin.archives
            self.ledger = virgin.ledger
            self.memory = None
            self.diary = SettledDiary()
            self.chord_memory = ChordMemoryBank()
            self.chat_messages.append(
                {
                    "id": f"lineage-{int(time.time())}",
                    "sender": "system",
                    "role_label": "Lineage Switch",
                    "text": (
                        "Switched to VIRGIN BASELINE (Cycle 0 — Core Architecture): "
                        "2 foundational rules, 10 structural Tier-2 leaves, 0 story precedents."
                    ),
                }
            )
        else:
            self.active_lineage = "schooled"
            self.sleep_cycle_index = (
                int(self.left_brain.callosum_metadata.sleep_cycle_index)
                if (self.left_brain is not None and self.left_brain.callosum_metadata is not None)
                else 2
            )
            self.active_checkpoint_label = f"Cycle {self.sleep_cycle_index} (models/left_brain_lora_cycle2)"
            self.rules = self._load_default_rules()
            self.learned_leaves = self._load_default_learned_leaves()
            self.archives = self._load_initial_archives()
            self.ledger = self._load_initial_ledger()
            self.savored_turn_ids = self._load_initial_savored_ids()
            self.memory = None
            self.chat_messages.append(
                {
                    "id": f"lineage-{int(time.time())}",
                    "sender": "system",
                    "role_label": "Lineage Switch",
                    "text": (
                        f"Switched to SCHOOLED BRAIN (Cycle {self.sleep_cycle_index}): "
                        f"{len(self.rules)} rules, {len(self.archives.judicial.ratified_chords)} ratified chords."
                    ),
                }
            )
        return self.get_state_snapshot()

    def _load_default_sensors(self) -> ThreeBridgeSensors:
        bridges_dir = self.repo_root / "models" / "right_brain_bridges"
        if bridges_dir.exists():
            try:
                suite = NeuralThreeBridgeSuite.load_all(bridges_dir)
                return suite.as_three_bridge_sensors()
            except Exception:
                pass
        return _default_fallback_sensors()

    def _resolve_checkpoint_dir(self) -> Path | None:
        """Auditor Guardrail #4: Prefer Cycle 2 weights (`models/left_brain_lora_cycle2/`)."""
        c2 = self.repo_root / "models" / "left_brain_lora_cycle2"
        if (c2 / "corpus_callosum_meta.json").exists():
            self.active_checkpoint_label = "Cycle 2 (models/left_brain_lora_cycle2)"
            self.sleep_cycle_index = 2
            return c2
        c1 = self.repo_root / "models" / "left_brain_lora"
        if (c1 / "corpus_callosum_meta.json").exists():
            self.active_checkpoint_label = "Cycle 1 (models/left_brain_lora)"
            self.sleep_cycle_index = 1
            return c1
        return None

    def _background_warm_left_brain(self) -> None:
        ckpt_dir = self._resolve_checkpoint_dir()
        if ckpt_dir is None:
            self.left_brain_status = "GEAR_1_ONLY (No LoRA checkpoint found)"
            return
        with self.gpu_lock:
            if self.left_brain is not None or self.rem_sleep_in_progress:
                return
            self.left_brain_loading = True
            self.left_brain_status = "LOADING_QWEN_4B..."
            try:
                self.left_brain = SubliminalGapLeftBrain.from_checkpoint(
                    ckpt_dir,
                    max_clause_1_tokens=14,
                    max_post_gap_tokens=36,
                )
                if self.left_brain.callosum_metadata is not None:
                    self.sleep_cycle_index = int(self.left_brain.callosum_metadata.sleep_cycle_index)
                self.left_brain_status = f"READY (Cycle {self.sleep_cycle_index})"
            except Exception as exc:
                self.left_brain_status = f"GEAR_1_FALLBACK ({type(exc).__name__})"
            finally:
                self.left_brain_loading = False

    def _synthesize_gear_1_note(
        self,
        turn: ThreeLeggedTurnResult,
        *,
        room: str,
        board_posture: ChessBoardPosture | None = None,
    ) -> str:
        """Provide a concise status note when running in Gear-1-only mode or fast reflex."""
        vibe = turn.final_vibe
        if room == "sanctuary":
            return (
                f"Pacing your current state (arousal={vibe.arousal:.2f}, valence={vibe.valence:.2f}), "
                f"we step steadily toward {self.sanctuary_mode.replace('_', ' ')} equilibrium."
            )
        if board_posture is not None:
            if board_posture.board_shadow_warnings:
                w = board_posture.board_shadow_warnings[0]
                return (
                    f"Evaluating {board_posture.dominant_piece} ({board_posture.dominant_color}), "
                    f"watch out for {w.shadow_pathology}—{w.trigger_reason}."
                )
            return (
                f"Position anchored on {board_posture.dominant_piece} ({board_posture.dominant_color} / "
                f"{board_posture.dominant_element}), keeping King shelter at {board_posture.philosophical_coordinates['philosophical/being/existence']:.2f}."
            )
        return f"Right-Brain equilibrium settled ({turn.outcome}, clarity={vibe.clarity:.2f})."

    def _execute_turn(
        self,
        scenario_text: str,
        *,
        room: str,
        force_spoken_reply: bool,
        sanctuary_target_mode: str | None = None,
        clause_1_prefix: str | None = None,
    ) -> tuple[ThreeLeggedTurnResult, float, dict[str, Any]]:
        """Run one thread-safe 3-Gear turn respecting Auditor Guardrails #1, #2, and #4."""
        t0 = time.perf_counter()
        self.scene_counter += 1
        scene_idx = self.scene_counter

        active_sensors: Any = self.sensors
        if room == "chess":
            self.board_posture = evaluate_chess_board_posture(self.board)
            active_sensors = merge_board_fact_with_chat_sensors(self.board_posture, self.sensors)

        # Acquire GPU lock if we aren't currently in REM sleep; otherwise fall back cleanly to Gear 1
        acquired = self.gpu_lock.acquire(blocking=not self.rem_sleep_in_progress, timeout=0.05 if self.rem_sleep_in_progress else 30.0)
        try:
            active_lb = self.left_brain if (acquired and not self.rem_sleep_in_progress) else None
            turn = run_three_legged_turn(
                self.birth_state,
                scenario_text,
                active_sensors,
                self.rules,
                scene_index=scene_idx,
                subliminal_left_brain=active_lb,
                clause_1_prefix=clause_1_prefix,
                learned_leaves=self.learned_leaves,
                prior_memory=self.memory,
                diary=self.diary,
                chord_memory=self.chord_memory,
                contemplation_ledger=self.ledger,
                branch_archives=self.archives,
                max_passes=2,
                sanctuary_target_mode=sanctuary_target_mode,
                sanctuary_lead_alpha=0.25,
                force_spoken_reply=force_spoken_reply,
            )
        finally:
            if acquired:
                self.gpu_lock.release()

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        self.memory = turn.next_memory
        self.diary = turn.next_diary
        self.ledger = turn.next_ledger
        self.chord_memory = turn.next_chord_memory
        self.archives = turn.next_archives
        if turn.unresolved_sleep_scene is not None:
            self.sleep_queue.append(turn.unresolved_sleep_scene)

        vibe = turn.final_vibe
        trace_info: dict[str, Any] = {}
        if turn.spliced_traces:
            st = turn.spliced_traces[-1]
            trace_info = {
                "clause_1_prefix": st.clause_1_prefix,
                "post_gap_continuation": st.post_gap_continuation,
                "comma_scored_live": st.comma_scored_live,
                "comma_vibe_posture": st.comma_vibe_posture,
                "comma_dissonance_notes": list(st.comma_dissonance_notes),
                "spliced_cadence_vector": [round(float(x), 4) for x in st.spliced_chord.cadence_vector],
                "spliced_thinker_vector": [round(float(x), 4) for x in st.spliced_chord.thinker_vector[:8]],
                "spliced_feeler_vector": [round(float(x), 4) for x in st.spliced_chord.feeler_vector[:8]],
            }

        telemetry = {
            "scene_index": scene_idx,
            "room": room,
            "gear_used": turn.gear_used,
            "settled": turn.settled,
            "outcome": turn.outcome,
            "elapsed_ms": elapsed_ms,
            "fact": round(float(vibe.fact), 4),
            "theory": round(float(vibe.theory), 4),
            "clarity": round(float(vibe.clarity), 4),
            "confidence": round(float(vibe.confidence), 4),
            "valence": round(float(vibe.valence), 4),
            "arousal": round(float(vibe.arousal), 4),
            "resonance": round(float(vibe.resonance), 4),
            "entrainment": round(float(vibe.retention), 4),
            "dissonance_notes": list(vibe.self_registry.dissonance_notes),
            "fact_coordinates": {k: round(float(v), 4) for k, v in vibe.coordinates.items()},
            "theory_coordinates": {k: round(float(v), 4) for k, v in vibe.theory_coordinates.items()},
            "sibling_color_balances": [b.to_dict() for b in turn.sibling_color_balances],
            "shadow_pathology_warnings": [w.to_dict() for w in turn.shadow_pathology_warnings],
            "spliced_trace": trace_info,
            "sanctuary_target_mode": sanctuary_target_mode,
        }

        self._turn_records_by_id[scene_idx] = {
            "scene_index": scene_idx,
            "scenario_text": scenario_text,
            "turn": turn,
            "telemetry": telemetry,
        }
        return turn, elapsed_ms, telemetry

    # -------------------------------------------------------------------------
    # Public Actions for HTTP Endpoints & Tests
    # -------------------------------------------------------------------------
    def get_state_snapshot(self) -> dict[str, Any]:
        self.board_posture = evaluate_chess_board_posture(self.board)
        legal_uci = [m.uci() for m in self.board.legal_moves]
        return {
            "active_room": self.active_room,
            "active_lineage": self.active_lineage,
            "sanctuary_mode": self.sanctuary_mode,
            "sleep_cycle_index": self.sleep_cycle_index,
            "active_checkpoint_label": self.active_checkpoint_label,
            "left_brain_status": (
                "REM_SLEEP_IN_PROGRESS..." if self.rem_sleep_in_progress else self.left_brain_status
            ),
            "rem_sleep_in_progress": self.rem_sleep_in_progress,
            "sleep_queue_count": len(self.sleep_queue),
            "savored_count": len(self.savored_turn_ids),
            "rules_count": len(self.rules),
            "ratified_chords_count": len(self.archives.judicial.ratified_chords),
            "board": {
                "fen": self.board.fen(),
                "turn": "w" if self.board.turn == chess.WHITE else "b",
                "in_check": self.board.is_check(),
                "is_game_over": self.board.is_game_over(),
                "move_history_san": list(self.move_history_san),
                "legal_moves_uci": legal_uci,
                "posture": self.board_posture.to_dict(),
                "presets": PRESET_CHESS_POSITIONS,
            },
            "chat_messages": list(self.chat_messages[-60:]),
            "gemini_api_configured": bool(os.environ.get("GEMINI_API_KEY", "").strip()),
        }

    def handle_chess_move(self, move_uci_or_san: str) -> dict[str, Any]:
        """Make a move on the chessboard; Gear 1 updates silently, Gear 2 speaks on tugs/warnings."""
        raw = move_uci_or_san.strip()
        move: chess.Move | None = None
        try:
            cand = chess.Move.from_uci(raw)
            if cand in self.board.legal_moves:
                move = cand
        except ValueError:
            pass
        if move is None:
            try:
                move = self.board.parse_san(raw)
            except ValueError as exc:
                raise ValueError(f"Illegal or unrecognized chess move: {raw!r}") from exc

        san_str = self.board.san(move)
        mover = "White" if self.board.turn == chess.WHITE else "Black"
        self.board.push(move)
        self.move_history_san.append(san_str)
        self.board_posture = evaluate_chess_board_posture(self.board)

        move_desc = (
            f"{mover} played {san_str}. {self.board_posture.board_summary}"
        )
        has_board_warning = bool(self.board_posture.board_shadow_warnings) or self.board.is_check()
        turn, elapsed_ms, telemetry = self._execute_turn(
            move_desc,
            room="chess",
            force_spoken_reply=has_board_warning,
        )

        # Only post to the chat room if Gear 2 spoke or a board shadow/check warning fired
        if turn.final_conscious_note or has_board_warning:
            reply_text = turn.final_conscious_note or self._synthesize_gear_1_note(
                turn,
                room="chess",
                board_posture=self.board_posture,
            )
            self.chat_messages.append(
                {
                    "id": telemetry["scene_index"],
                    "sender": "companion",
                    "role_label": f"LMFT Kibitzer (after {san_str})",
                    "text": reply_text,
                    "telemetry": telemetry,
                    "coached": False,
                    "savored": False,
                }
            )
        return self.get_state_snapshot()

    def handle_set_fen(self, fen_or_preset: str) -> dict[str, Any]:
        key = fen_or_preset.strip()
        if key in PRESET_CHESS_POSITIONS:
            fen = PRESET_CHESS_POSITIONS[key]["fen"]
        else:
            fen = key
        self.board = chess.Board(fen)
        self.move_history_san.clear()
        self.board_posture = evaluate_chess_board_posture(self.board)
        return self.get_state_snapshot()

    def handle_chat_message(
        self,
        text: str,
        *,
        room: str | None = None,
        sanctuary_mode: str | None = None,
        clause_1_prefix: str | None = None,
    ) -> dict[str, Any]:
        """Handle a user message in either the Chess Club or the Co-Regulation Sanctuary."""
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Message text cannot be empty")

        if room in ("chess", "sanctuary"):
            self.active_room = room
        if sanctuary_mode in SANCTUARY_MODES:
            self.sanctuary_mode = sanctuary_mode

        self.chat_messages.append(
            {
                "id": f"user-{self.scene_counter + 1}",
                "sender": "user",
                "role_label": "You",
                "text": clean_text,
            }
        )

        target_mode = self.sanctuary_mode if self.active_room == "sanctuary" else None
        turn, _elapsed_ms, telemetry = self._execute_turn(
            clean_text,
            room=self.active_room,
            force_spoken_reply=True,
            sanctuary_target_mode=target_mode,
            clause_1_prefix=clause_1_prefix,
        )
        reply_text = turn.final_conscious_note or self._synthesize_gear_1_note(
            turn,
            room=self.active_room,
            board_posture=self.board_posture if self.active_room == "chess" else None,
        )
        self.chat_messages.append(
            {
                "id": telemetry["scene_index"],
                "sender": "companion",
                "role_label": (
                    f"Sanctuary Guide ({self.sanctuary_mode.replace('_', ' ').title()})"
                    if self.active_room == "sanctuary"
                    else "LMFT Companion"
                ),
                "text": reply_text,
                "telemetry": telemetry,
                "coached": False,
                "savored": False,
            }
        )
        return self.get_state_snapshot()

    def handle_guest_kibitzer(
        self,
        guest_text: str = "",
        *,
        guest_name: str = "Gemini Guest",
        auto_query_gemini: bool = False,
    ) -> dict[str, Any]:
        """Auditor Guardrail #3: Score an external Guest Kibitzer comment through our 3 MiniLM bridges + 6-Color lens before replying."""
        resolved_text = guest_text.strip()
        if auto_query_gemini and not resolved_text:
            resolved_text = self._query_gemini_guest_api()
        if not resolved_text:
            raise ValueError("Provide guest_text or set GEMINI_API_KEY to query Gemini automatically.")

        # Score the guest's comment through the exact same 3 MiniLM bridges & 6-Color / Shadow lens
        g_coords, g_exp, _, _ = self.sensors.score_text(
            resolved_text,
            learned_leaves=self.learned_leaves,
        )
        g_balances = compute_philosophical_color_harmony(g_coords)
        g_shadows = detect_shadow_pathologies(
            g_coords,
            fact=self.board_posture.philosophical_coordinates.get("philosophical/being/existence", 0.50),
            arousal=float(g_exp.get("experiential/phase/arousal", 0.50)),
            resonance=float(g_exp.get("experiential/synchronization/resonance", 0.50)),
        )
        self.chat_messages.append(
            {
                "id": f"guest-{self.scene_counter + 1}",
                "sender": "guest",
                "role_label": guest_name,
                "text": resolved_text,
                "guest_analysis": {
                    "sibling_color_balances": [b.to_dict() for b in g_balances],
                    "shadow_pathology_warnings": [w.to_dict() for w in g_shadows],
                },
            }
        )

        # Now let our Dual-Hemisphere companion respond to the Guest Kibitzer anchored to the real board!
        turn, _elapsed_ms, telemetry = self._execute_turn(
            resolved_text,
            room="chess",
            force_spoken_reply=True,
        )
        reply_text = turn.final_conscious_note or self._synthesize_gear_1_note(
            turn,
            room="chess",
            board_posture=self.board_posture,
        )
        self.chat_messages.append(
            {
                "id": telemetry["scene_index"],
                "sender": "companion",
                "role_label": "LMFT Companion (replying to Guest)",
                "text": reply_text,
                "telemetry": telemetry,
                "coached": False,
                "savored": False,
            }
        )
        return self.get_state_snapshot()

    def _query_gemini_guest_api(self) -> str:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        prompt = (
            f"You are a sharp, concise chess club kibitzer. Current board FEN: {self.board.fen()} "
            f"({self.board_posture.board_summary}). Give a 1-sentence tactical or positional observation."
        )
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return str(data["candidates"][0]["content"]["parts"][0]["text"]).strip()
        except urllib.error.URLError as exc:
            raise ValueError(f"Gemini API request failed: {exc}") from exc

    def flag_turn_for_coaching(self, scene_index: int, user_note: str = "") -> dict[str, Any]:
        """Auditor Guardrail #3 (`[Coach / Queue for Sleep]`): Queue a turn as an `UnresolvedSleepScene` for Nighttime REM Sleep."""
        rec = self._turn_records_by_id.get(int(scene_index))
        if rec is None:
            raise ValueError(f"Unknown turn scene_index: {scene_index}")
        turn: ThreeLeggedTurnResult = rec["turn"]
        vibe = turn.final_vibe
        note_suffix = f" [Coach note: {user_note.strip()}]" if user_note.strip() else ""
        sleep_scene = UnresolvedSleepScene(
            scene_index=int(scene_index),
            scenario_text=f"{rec['scenario_text']}{note_suffix}",
            external_coordinates=dict(vibe.coordinates),
            felt_chord_notes=dict(vibe.felt_chord.complex_notes) if vibe.felt_chord is not None else {},
            valence=vibe.valence,
            salience=vibe.salience,
            arousal=vibe.arousal,
            left_brain_hunch=turn.final_conscious_note or rec["scenario_text"],
            social_drift_detected=False,
            active_concept_collision=vibe.active_concept,
        )
        self.sleep_queue = [s for s in self.sleep_queue if s.scene_index != int(scene_index)]
        self.sleep_queue.append(sleep_scene)
        for msg in self.chat_messages:
            if msg.get("id") == int(scene_index):
                msg["coached"] = True
                msg["coach_note"] = user_note.strip()
        return self.get_state_snapshot()

    def savor_turn_vibe(self, scene_index: int) -> dict[str, Any]:
        """Auditor Guardrail #3 (`[Savor / Keep Vibe]`): Bookmark a turn, reinforce its chord in the 3-branch archives, and persist to disk."""
        rec = self._turn_records_by_id.get(int(scene_index))
        if rec is None:
            raise ValueError(f"Unknown turn scene_index: {scene_index}")
        turn: ThreeLeggedTurnResult = rec["turn"]
        vibe = turn.final_vibe
        if vibe.felt_chord is not None:
            self.archives = harvest_into_branch_archives(
                self.archives,
                scene_index=int(scene_index),
                scenario_text=rec["scenario_text"],
                coordinates=vibe.coordinates,
                felt_chord=vibe.felt_chord,
                left_brain_reply=turn.final_conscious_note,
                settled=True,
                deliberation=None,
            )
        if int(scene_index) not in self.savored_turn_ids:
            self.savored_turn_ids.append(int(scene_index))
        for msg in self.chat_messages:
            if msg.get("id") == int(scene_index):
                msg["savored"] = True
        self._persist_user_state()
        return self.get_state_snapshot()

    def _consolidate_sleep_clusters_locally(
        self,
        queued_scenes: tuple[UnresolvedSleepScene, ...],
    ) -> tuple[tuple[str, ...], tuple[FuzzyConceptRule, ...], tuple[str, ...]]:
        """Refine colliding rules and forge new coordinate-anchored rules + 3-branch clauses when Teacher LLM is offline."""
        refined_rules, refined_ids = refine_colliding_rules_during_sleep(
            self.rules,
            queued_scenes,
        )
        active_rules = list(refined_rules)
        existing_names = {r.concept_name for r in active_rules}
        forged_rules: list[FuzzyConceptRule] = []
        harvested_clause_ids: list[str] = []

        clusters: dict[str, list[UnresolvedSleepScene]] = {}
        for s in queued_scenes:
            ckey = _cluster_key_for_unresolved(s)
            if ckey is not None:
                clusters.setdefault(ckey, []).append(s)

        cur_archives = self.archives
        cur_ledger = self.ledger

        for ckey, group in clusters.items():
            avg_coords = _average_cluster_coordinates(group)
            target_leaves = _anchor_target_leaves_from_cluster(ckey, avg_coords)
            rep = group[0]
            rep_scene = rep.scene_index

            base_name = f"coached_{ckey}"
            concept_name = base_name
            if concept_name in existing_names:
                concept_name = f"{base_name}_s{rep_scene}"

            has_tier2_or_chaos = (
                any(leaf_tier(k) >= 2 for k in target_leaves)
                or float(target_leaves.get("philosophical/structure/chaos", 0.0)) >= 0.45
            )
            rule_low = 0.55 if has_tier2_or_chaos else 0.72
            rule_high = 0.78 if has_tier2_or_chaos else 0.85

            description = (
                f"Consolidated REM coaching rule for {ckey.replace('_', ' ')}: "
                f"{rep.scenario_text[:120]}"
            )
            new_rule = FuzzyConceptRule(
                rule_id=f"rule-sleep-s{rep_scene}-{concept_name[:28]}",
                concept_name=concept_name,
                description=description,
                target_leaves=target_leaves,
                low=rule_low,
                high=rule_high,
            )
            active_rules.append(new_rule)
            forged_rules.append(new_rule)
            existing_names.add(concept_name)

            cult_sig = {k: v for k, v in avg_coords.items() if k.startswith("cultural/")}
            phil_sig = {k: v for k, v in avg_coords.items() if k.startswith("philosophical/")}
            exp_sig = {k: v for k, v in avg_coords.items() if k.startswith("experiential/")}
            exp_sig[SALIENCE_LEAF] = round(sum(s.salience for s in group) / len(group), 4)
            exp_sig[VALENCE_LEAF] = round(sum(s.valence for s in group) / len(group), 4)
            exp_sig[AROUSAL_LEAF] = round(sum(s.arousal for s in group) / len(group), 4)

            avg_fc = compile_felt_chord(
                avg_coords,
                salience=exp_sig[SALIENCE_LEAF],
                valence=exp_sig[VALENCE_LEAF],
                arousal=exp_sig[AROUSAL_LEAF],
                resonance=0.65,
                learned_leaves=tuple(self.learned_leaves),
            )
            exec_tag, leg_tag, jud_tag = _derive_concept_tags(avg_coords, avg_fc)

            exec_text = f"Coordinate action under {concept_name} ({exec_tag}): {rep.left_brain_hunch[:100]}"
            leg_text = f"Sense and pace {leg_tag} posture during {concept_name}: {rep.scenario_text[:100]}"
            jud_text = f"Verify {jud_tag} equilibrium and ratified chord for {concept_name}."

            exec_clause = BranchClauseEntry(
                clause_id=f"exec-sleep-s{rep_scene}-{exec_tag}",
                branch=EXECUTIVE_BRANCH,
                text=exec_text,
                signature_coords=cult_sig,
                source_scene=rep_scene,
                concept_tag=exec_tag,
            )
            leg_clause = BranchClauseEntry(
                clause_id=f"leg-sleep-s{rep_scene}-{leg_tag}",
                branch=LEGISLATIVE_BRANCH,
                text=leg_text,
                signature_coords=exp_sig,
                source_scene=rep_scene,
                concept_tag=leg_tag,
            )
            jud_clause = BranchClauseEntry(
                clause_id=f"jud-sleep-s{rep_scene}-{jud_tag}",
                branch=JUDICIAL_BRANCH,
                text=jud_text,
                signature_coords=phil_sig,
                source_scene=rep_scene,
                concept_tag=jud_tag,
            )
            harvested_clause_ids.extend([exec_clause.clause_id, leg_clause.clause_id, jud_clause.clause_id])

            next_exec = cur_archives.executive.record_scene(
                scene_index=rep_scene,
                coordinates=avg_coords,
                action_summary=exec_text,
                harvested_clause=exec_clause,
            )
            next_leg = cur_archives.legislative.record_scene(
                scene_index=rep_scene,
                sensory_snapshot=leg_text,
                felt_chord=avg_fc,
                coordinates=avg_coords,
                harvested_clause=leg_clause,
            )
            next_jud = cur_archives.judicial.record_clause_and_chord(
                harvested_clause=jud_clause,
                ratified_chord=new_rule,
            )
            cur_archives = ThreeBranchArchives(
                executive=next_exec,
                legislative=next_leg,
                judicial=next_jud,
            )
            for s in group:
                cur_ledger = cur_ledger.record_settled(
                    scene_index=s.scene_index,
                    takeaway_source=description,
                    felt_chord=avg_fc,
                    clarity=0.85,
                    confidence=0.90,
                    resolved_uncertainty_scenes=(s.scene_index,),
                )

        self.rules = tuple(active_rules)
        self.archives = cur_archives
        self.ledger = cur_ledger
        return refined_ids, tuple(forged_rules), tuple(harvested_clause_ids)

    def handle_run_rem_sleep(self) -> dict[str, Any]:
        """Auditor Guardrail #4: Run REM Sleep consolidation inside `self.gpu_lock` and persist forged rules + archives to disk."""
        with self.gpu_lock:
            self.rem_sleep_in_progress = True
            try:
                queued_count = len(self.sleep_queue)
                forged_count = 0
                if queued_count > 0:
                    if self.teacher is not None:
                        report = run_sleep_consolidation(
                            self.teacher,
                            tuple(self.sleep_queue),
                            self.rules,
                            self.archives,
                            self.ledger,
                            learned_leaves=self.learned_leaves,
                        )
                        self.rules = report.updated_rules
                        self.archives = report.updated_archives
                        self.ledger = report.updated_ledger
                        refined_ids = report.refined_rule_ids
                        forged_count = len(report.forged_rules)
                    else:
                        refined_ids, forged_rules, _clause_ids = (
                            self._consolidate_sleep_clusters_locally(tuple(self.sleep_queue))
                        )
                        forged_count = len(forged_rules)
                else:
                    refined_ids = ()
                self.sleep_queue.clear()
                self._persist_user_state()
                summary = (
                    f"REM Sleep Cycle complete: consolidated {queued_count} queued scene(s), "
                    f"forged {forged_count} new rule(s), refined {len(refined_ids)} colliding rule(s), "
                    f"and persisted {len(self.savored_turn_ids)} savored vibe(s) to disk."
                )
                self.chat_messages.append(
                    {
                        "id": f"rem-{int(time.time())}",
                        "sender": "system",
                        "role_label": "Nighttime REM Sleep",
                        "text": summary,
                    }
                )
            finally:
                self.rem_sleep_in_progress = False
        return self.get_state_snapshot()


# ============================================================================
# ZERO-FRUITCAKE ANTIGRAVITY-STYLE SINGLE-PAGE UI
# ============================================================================

SINGLE_PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LMFT Club — Chess & Co-Regulation Sanctuary</title>
<style>
  :root {
    --bg-root: #0d0f12;
    --bg-panel: #12151a;
    --bg-elevated: #171b22;
    --bg-input: #0f1217;
    --border: #222730;
    --border-subtle: #1a1e26;
    --text-main: #e2e8f0;
    --text-muted: #8b95a5;
    --text-dim: #5a6372;
    --accent: #38bdf8;
    --board-light: #2a303c;
    --board-dark: #181c24;
    --board-sel: #1e3a5f;
    --font-sans: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
    --font-mono: "JetBrains Mono", "Cascadia Code", "Consolas", monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg-root);
    color: var(--text-main);
    font-family: var(--font-sans);
    font-size: 13px;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    user-select: none;
  }
  /* Top Header Bar */
  header {
    height: 40px;
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 14px;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: var(--text-main);
  }
  .brand-tag {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-muted);
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    padding: 2px 7px;
    border-radius: 3px;
  }
  .mode-tabs {
    display: flex;
    gap: 4px;
  }
  .tab-btn {
    background: transparent;
    color: var(--text-muted);
    border: 1px solid transparent;
    padding: 5px 12px;
    font-size: 12px;
    font-family: var(--font-mono);
    cursor: pointer;
    border-radius: 3px;
  }
  .tab-btn.active {
    background: var(--bg-elevated);
    color: var(--text-main);
    border-color: var(--border);
  }
  /* Main 2-Pane Split */
  main {
    flex: 1;
    display: grid;
    grid-template-columns: 460px 1fr;
    overflow: hidden;
  }
  .left-pane {
    background: var(--bg-panel);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    padding: 14px;
    gap: 12px;
    overflow-y: auto;
  }
  .right-pane {
    background: var(--bg-root);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .section-label {
    font-family: var(--font-mono);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-dim);
    margin-bottom: 6px;
  }
  /* Chessboard */
  .board-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
  }
  .chess-grid {
    width: 384px;
    height: 384px;
    display: grid;
    grid-template-columns: repeat(8, 1fr);
    grid-template-rows: repeat(8, 1fr);
    border: 1px solid var(--border);
  }
  .sq {
    width: 48px;
    height: 48px;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    cursor: pointer;
    font-size: 30px;
  }
  .sq.light { background: var(--board-light); }
  .sq.dark { background: var(--board-dark); }
  .sq.selected { background: var(--board-sel); }
  .sq.legal-dot::after {
    content: "";
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: rgba(56, 189, 248, 0.55);
    position: absolute;
  }
  .piece-w { color: #f8fafc; filter: drop-shadow(0 1px 2px rgba(0,0,0,0.85)); }
  .piece-b { color: #94a3b8; filter: drop-shadow(0 1px 2px rgba(0,0,0,0.95)); }
  .controls-row {
    width: 100%;
    display: flex;
    gap: 6px;
    align-items: center;
  }
  select, input[type="text"] {
    background: var(--bg-input);
    color: var(--text-main);
    border: 1px solid var(--border);
    padding: 6px 9px;
    font-size: 12px;
    font-family: var(--font-mono);
    border-radius: 3px;
    outline: none;
  }
  select:focus, input[type="text"]:focus {
    border-color: #3b4454;
  }
  .btn {
    background: var(--bg-elevated);
    color: var(--text-main);
    border: 1px solid var(--border);
    padding: 6px 10px;
    font-size: 11px;
    font-family: var(--font-mono);
    border-radius: 3px;
    cursor: pointer;
    white-space: nowrap;
  }
  .btn:hover { background: #202631; }
  /* Color Harmony Bars */
  .telemetry-card {
    background: var(--bg-root);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 10px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .color-row {
    display: flex;
    flex-direction: column;
    gap: 3px;
  }
  .color-row-head {
    display: flex;
    justify-content: space-between;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-muted);
  }
  .bar-track {
    height: 6px;
    background: var(--bg-elevated);
    border-radius: 2px;
    overflow: hidden;
    display: flex;
  }
  .shadow-banner {
    font-family: var(--font-mono);
    font-size: 11px;
    padding: 6px 8px;
    border: 1px solid #5c2828;
    background: #1f1214;
    color: #fca5a5;
    border-radius: 3px;
  }
  /* Sanctuary View */
  .sanctuary-box {
    display: none;
    flex-direction: column;
    gap: 14px;
  }
  .pacer-track {
    height: 36px;
    background: var(--bg-root);
    border: 1px solid var(--border);
    border-radius: 3px;
    position: relative;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 12px;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-muted);
  }
  .pacer-fill {
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 50%;
    background: rgba(56, 189, 248, 0.08);
    border-right: 1px solid rgba(56, 189, 248, 0.45);
    animation: breathe 8s ease-in-out infinite alternate;
  }
  @keyframes breathe {
    0% { width: 18%; }
    48% { width: 50%; }
    52% { width: 50%; }
    100% { width: 88%; }
  }
  /* Chat Stream */
  .chat-stream {
    flex: 1;
    overflow-y: auto;
    padding: 16px 20px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    user-select: text;
  }
  .msg {
    border: 1px solid var(--border);
    background: var(--bg-panel);
    border-radius: 3px;
    padding: 10px 13px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .msg.user {
    background: #10141b;
    border-color: #1d2430;
  }
  .msg.guest {
    background: #12171a;
    border-color: #213540;
  }
  .msg-head {
    display: flex;
    justify-content: space-between;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-dim);
  }
  .msg-text {
    line-height: 1.5;
    color: var(--text-main);
    font-size: 13px;
  }
  .msg-foot {
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-dim);
    border-top: 1px solid var(--border-subtle);
    padding-top: 6px;
  }
  .foot-link {
    background: none;
    border: none;
    color: var(--text-muted);
    font-family: var(--font-mono);
    font-size: 11px;
    cursor: pointer;
    padding: 0;
  }
  .foot-link:hover { color: var(--text-main); text-decoration: underline; }
  .foot-link.active-flag { color: var(--accent); }
  .nerve-drawer {
    display: none;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-muted);
    background: var(--bg-root);
    border: 1px solid var(--border);
    padding: 8px 10px;
    border-radius: 3px;
    line-height: 1.45;
    white-space: pre-wrap;
  }
  /* Chat Composer */
  .composer {
    background: var(--bg-panel);
    border-top: 1px solid var(--border);
    padding: 10px 14px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .composer-main {
    display: flex;
    gap: 8px;
  }
  .composer-main input { flex: 1; }
  .guest-bar {
    display: none;
    gap: 8px;
    align-items: center;
  }
  .guest-bar input { flex: 1; }
  /* Bottom Status Bar */
  footer {
    height: 28px;
    background: var(--bg-panel);
    border-top: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 14px;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-muted);
  }
  .status-group {
    display: flex;
    align-items: center;
    gap: 14px;
  }
</style>
</head>
<body>
  <header>
    <div class="brand">
      <span>LMFT SANCTUARY &amp; CHESS CLUB</span>
      <span class="brand-tag" id="cycle-badge">CYCLE 2</span>
      <select id="lineage-select" style="padding:3px 8px; font-size:11px; font-family:var(--font-mono);" onchange="switchLineage(this.value)" title="Switch between Schooled (Cycle 2 + User Coaching) and Virgin Baseline (Cycle 0 — Core Architecture)">
        <option value="schooled">LINEAGE: SCHOOLED (CYCLE 2)</option>
        <option value="virgin">LINEAGE: VIRGIN BASELINE (CYCLE 0)</option>
      </select>
    </div>
    <div class="mode-tabs">
      <button class="tab-btn active" id="tab-chess" onclick="switchRoom('chess')">CHESS TABLE</button>
      <button class="tab-btn" id="tab-sanctuary" onclick="switchRoom('sanctuary')">CO-REGULATION SANCTUARY</button>
    </div>
  </header>

  <main>
    <section class="left-pane">
      <!-- CHESS VIEW -->
      <div id="chess-view" style="display:flex; flex-direction:column; gap:12px;">
        <div class="section-label">LIVE BOARD (FACT ANCHOR)</div>
        <div class="board-wrap">
          <div class="chess-grid" id="chess-grid"></div>
          <div class="controls-row">
            <select id="preset-select" style="flex:1;" onchange="loadPreset(this.value)">
              <option value="start">Standard Starting Position</option>
              <option value="exposed_king_attack">Exposed King vs. Attack (Psychosis Test)</option>
              <option value="naked_king_siege">Stripped King Under Siege</option>
              <option value="knight_outpost_chaos">Sharp Tactical Tension (Knight / Ice)</option>
              <option value="rook_seventh_bastion">7th-Rank Rook Bastion Endgame</option>
            </select>
            <button class="btn" onclick="loadPreset('start')">Reset</button>
          </div>
          <div class="controls-row">
            <input type="text" id="fen-input" placeholder="Paste FEN or move (e.g. e4, Nf3)..." style="flex:1;">
            <button class="btn" onclick="applyManualMoveOrFen()">Apply</button>
          </div>
        </div>
      </div>

      <!-- SANCTUARY VIEW -->
      <div id="sanctuary-view" class="sanctuary-box">
        <div class="section-label">CO-REGULATION PACING &amp; LEADING (COMMA SPLICE)</div>
        <div class="pacer-track">
          <div class="pacer-fill"></div>
          <span style="z-index:1;">CLAUSE 1: PACE (FACT)</span>
          <span style="z-index:1; color:var(--accent);">, &lt;|subliminal_gap|&gt;</span>
          <span style="z-index:1;">CLAUSE 2: LEAD (ALPHA=0.25)</span>
        </div>
        <div class="controls-row">
          <span style="font-family:var(--font-mono); font-size:11px; color:var(--text-muted);">TARGET LEAD:</span>
          <select id="sanctuary-mode-select" style="flex:1;" onchange="changeSanctuaryMode(this.value)">
            <option value="tilt_reset">Tilt Reset (Red/Green -&gt; Cyan Water / Magenta Wind)</option>
            <option value="deep_focus">Deep Focus (Magenta Bishop Order + Yellow Rook Bastion)</option>
            <option value="night_wind_down">Night Wind-Down (Needlessness + Austere Sanctuary)</option>
          </select>
        </div>
        <div class="telemetry-card">
          <div class="section-label">HOW THIS ROOM WORKS</div>
          <div style="color:var(--text-muted); line-height:1.45; font-size:12px;">
            1. <b>Clause 1 (Before Comma)</b>: Matches your un-rigged external state (never forces fake calm).<br>
            2. <b>At the Comma (&lt;|subliminal_gap|&gt;)</b>: Steps the 52-float nerve cord by <code>alpha=0.25</code> toward your chosen target.<br>
            3. <b>Clause 2 (After Comma)</b>: Guides cadence one step toward equilibrium.
          </div>
        </div>
      </div>

      <!-- SHARED 3-PAIR COLOR HARMONY & SHADOW TELEMETRY -->
      <div class="telemetry-card">
        <div class="section-label">5-TIER PHILOSOPHICAL COLOR HARMONY (RGB / CMY)</div>
        <div id="color-bars-container" style="display:flex; flex-direction:column; gap:8px;"></div>
        <div id="shadow-warnings-container" style="display:flex; flex-direction:column; gap:6px;"></div>
      </div>
    </section>

    <section class="right-pane">
      <div class="chat-stream" id="chat-stream"></div>
      <div class="composer">
        <div class="composer-main">
          <input type="text" id="chat-input" placeholder="Talk to companion, discuss a chess plan, or check in for Sanctuary pacing..." onkeydown="if(event.key==='Enter') sendChat()">
          <button class="btn" onclick="sendChat()">Send</button>
          <button class="btn" onclick="toggleGuestBar()">Guest Chair</button>
        </div>
        <div class="guest-bar" id="guest-bar">
          <input type="text" id="guest-input" placeholder="Paste Gemini / Guest comment to score through Right-Brain lens...">
          <button class="btn" onclick="sendGuest(false)">Score &amp; Reply</button>
          <button class="btn" id="btn-gemini-api" onclick="sendGuest(true)">Ask Gemini API</button>
        </div>
      </div>
    </section>
  </main>

  <footer>
    <div class="status-group">
      <span id="status-left-brain">LEFT BRAIN: INITIALIZING</span>
      <span id="status-rules">RULES: 0</span>
      <span id="status-savored">SAVORED: 0</span>
    </div>
    <div class="status-group">
      <span id="status-sleep-queue">SLEEP QUEUE: 0</span>
      <button class="btn" onclick="runRemSleep()">Run REM Sleep</button>
    </div>
  </footer>

<script>
const PIECE_GLYPHS = {
  'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
  'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
};
const COLOR_HEX = {
  'Red': '#ef4444',
  'Cyan': '#06b6d4',
  'Magenta': '#d946ef',
  'Green': '#22c55e',
  'Blue': '#3b82f6',
  'Yellow': '#eab308'
};

let appState = null;
let selectedSquare = null;
let openDrawers = {};

async function apiPost(path, body) {
  const res = await fetch(path, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body || {})
  });
  const data = await res.json();
  if (!res.ok) {
    alert(data.error || 'Request failed');
    return null;
  }
  appState = data;
  renderAll();
  return data;
}

async function fetchState() {
  const res = await fetch('/api/state');
  appState = await res.json();
  renderAll();
}

function switchRoom(room) {
  appState.active_room = room;
  document.getElementById('tab-chess').classList.toggle('active', room === 'chess');
  document.getElementById('tab-sanctuary').classList.toggle('active', room === 'sanctuary');
  document.getElementById('chess-view').style.display = room === 'chess' ? 'flex' : 'none';
  document.getElementById('sanctuary-view').style.display = room === 'sanctuary' ? 'flex' : 'none';
}

function changeSanctuaryMode(mode) {
  if (appState) appState.sanctuary_mode = mode;
}

function toggleGuestBar() {
  const bar = document.getElementById('guest-bar');
  bar.style.display = bar.style.display === 'flex' ? 'none' : 'flex';
}

function fenToGrid(fen) {
  const rows = fen.split(' ')[0].split('/');
  const out = [];
  for (let r = 0; r < 8; r++) {
    const row = [];
    for (const ch of rows[r]) {
      if (ch >= '1' && ch <= '8') {
        for (let k = 0; k < parseInt(ch, 10); k++) row.push(null);
      } else {
        row.push(ch);
      }
    }
    out.push(row);
  }
  return out;
}

function renderBoard() {
  if (!appState || !appState.board) return;
  const gridEl = document.getElementById('chess-grid');
  gridEl.innerHTML = '';
  const matrix = fenToGrid(appState.board.fen);
  const legalFromSelected = new Set();
  if (selectedSquare) {
    for (const uci of appState.board.legal_moves_uci) {
      if (uci.startsWith(selectedSquare)) {
        legalFromSelected.add(uci.slice(2, 4));
      }
    }
  }
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const fileChar = String.fromCharCode(97 + c);
      const rankChar = String(8 - r);
      const sqName = fileChar + rankChar;
      const sqDiv = document.createElement('div');
      sqDiv.className = 'sq ' + ((r + c) % 2 === 0 ? 'light' : 'dark');
      if (selectedSquare === sqName) sqDiv.classList.add('selected');
      if (legalFromSelected.has(sqName)) sqDiv.classList.add('legal-dot');
      const piece = matrix[r][c];
      if (piece) {
        const span = document.createElement('span');
        span.className = piece === piece.toUpperCase() ? 'piece-w' : 'piece-b';
        span.textContent = PIECE_GLYPHS[piece] || piece;
        sqDiv.appendChild(span);
      }
      sqDiv.onclick = () => onSquareClick(sqName, piece);
      gridEl.appendChild(sqDiv);
    }
  }
}

async function onSquareClick(sqName, piece) {
  if (!selectedSquare) {
    if (piece) {
      selectedSquare = sqName;
      renderBoard();
    }
    return;
  }
  if (selectedSquare === sqName) {
    selectedSquare = null;
    renderBoard();
    return;
  }
  const candUci = selectedSquare + sqName;
  const legal = appState.board.legal_moves_uci || [];
  let matched = legal.find(u => u === candUci || u === candUci + 'q');
  if (matched) {
    selectedSquare = null;
    await apiPost('/api/chess/move', {move: matched});
  } else if (piece) {
    selectedSquare = sqName;
    renderBoard();
  } else {
    selectedSquare = null;
    renderBoard();
  }
}

async function loadPreset(val) {
  selectedSquare = null;
  await apiPost('/api/chess/fen', {fen: val});
}

async function applyManualMoveOrFen() {
  const inp = document.getElementById('fen-input');
  const val = inp.value.trim();
  if (!val) return;
  inp.value = '';
  if (val.includes('/')) {
    await apiPost('/api/chess/fen', {fen: val});
  } else {
    await apiPost('/api/chess/move', {move: val});
  }
}

function renderColorBars() {
  if (!appState) return;
  let balances = appState.board.posture.sibling_color_balances || [];
  let warnings = appState.board.posture.board_shadow_warnings || [];
  const msgs = appState.chat_messages || [];
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (msgs[i].telemetry && msgs[i].telemetry.sibling_color_balances) {
      if (appState.active_room === 'sanctuary' || msgs[i].telemetry.room === appState.active_room) {
        balances = msgs[i].telemetry.sibling_color_balances;
        warnings = msgs[i].telemetry.shadow_pathology_warnings || [];
        break;
      }
    }
  }
  const container = document.getElementById('color-bars-container');
  container.innerHTML = '';
  for (const b of balances) {
    const actA = b.activation_a ?? 0.5;
    const actB = b.activation_b ?? 0.5;
    const aPct = Math.round((actA / Math.max(0.01, actA + actB)) * 100);
    const bPct = 100 - aPct;
    const row = document.createElement('div');
    row.className = 'color-row';
    row.innerHTML = `
      <div class="color-row-head">
        <span>${b.color_a} (${(b.leaf_a || '').split('/').pop()} ${actA.toFixed(2)})</span>
        <span>${(b.parent_branch || '').toUpperCase()}</span>
        <span>${b.color_b} (${(b.leaf_b || '').split('/').pop()} ${actB.toFixed(2)})</span>
      </div>
      <div class="bar-track">
        <div style="width:${aPct}%; background:${COLOR_HEX[b.color_a] || '#38bdf8'};"></div>
        <div style="width:${bPct}%; background:${COLOR_HEX[b.color_b] || '#94a3b8'};"></div>
      </div>
    `;
    container.appendChild(row);
  }
  const warnBox = document.getElementById('shadow-warnings-container');
  warnBox.innerHTML = '';
  for (const w of warnings) {
    const d = document.createElement('div');
    d.className = 'shadow-banner';
    d.textContent = `SHADOW ALERT: ${w.shadow_pathology.toUpperCase()} (${w.chess_piece} / ${w.color_name || ''}) — ${w.trigger_reason}`;
    warnBox.appendChild(d);
  }
}

function toggleDrawer(id) {
  openDrawers[id] = !openDrawers[id];
  renderChat();
}

async function coachTurn(sceneIdx) {
  const note = prompt('Optional 1-line coaching note for Nighttime REM Sleep:', '') || '';
  await apiPost('/api/turn/coach', {scene_index: sceneIdx, note: note});
}

async function savorTurn(sceneIdx) {
  await apiPost('/api/turn/savor', {scene_index: sceneIdx});
}

function renderChat() {
  if (!appState) return;
  const box = document.getElementById('chat-stream');
  box.innerHTML = '';
  const msgs = appState.chat_messages || [];
  if (msgs.length === 0) {
    const empty = document.createElement('div');
    empty.style.color = 'var(--text-dim)';
    empty.style.fontFamily = 'var(--font-mono)';
    empty.textContent = 'Make a move on the chessboard or type a message below to begin.';
    box.appendChild(empty);
    return;
  }
  for (const m of msgs) {
    const card = document.createElement('div');
    card.className = 'msg ' + (m.sender || '');
    const t = m.telemetry;
    let footHtml = '';
    let drawerHtml = '';
    if (t) {
      const isOpen = !!openDrawers[m.id];
      const trace = t.spliced_trace || {};
      const drawerLines = [
        `GEAR: ${t.gear_used} | OUTCOME: ${t.outcome} | FACT: ${t.fact} | THEORY: ${t.theory} | CLARITY: ${t.clarity}`,
        `FEELER: Valence=${t.valence}, Arousal=${t.arousal}, Resonance=${t.resonance}, Entrainment=${t.entrainment}`,
      ];
      if (trace.clause_1_prefix) {
        drawerLines.push(`CLAUSE 1 (AT COMMA): "${trace.clause_1_prefix}" [Posture=${trace.comma_vibe_posture || 'N/A'}]`);
        drawerLines.push(`CLAUSE 2 (POST-GAP): "${trace.post_gap_continuation}"`);
        drawerLines.push(`CADENCE VECTOR [Harmony, Dissonance, Rupture]: ${JSON.stringify(trace.spliced_cadence_vector)}`);
      }
      if (t.dissonance_notes && t.dissonance_notes.length) {
        drawerLines.push(`DISSONANCE CAUGHT: ${t.dissonance_notes.join(', ')}`);
      }
      footHtml = `
        <div class="msg-foot">
          <span>${t.gear_used.replace('GEAR_1_FAST_RIGHT_BRAIN', 'GEAR 1 · REFLEX').replace('GEAR_2_SUBLIMINAL_GAP_SPLICE', 'GEAR 2 · COMMA-SPLICED')} · ${t.elapsed_ms}ms</span>
          <button class="foot-link" onclick="toggleDrawer(${m.id})">[${isOpen ? 'Hide' : 'Inspect'} Nerve Cord]</button>
          <button class="foot-link ${m.coached ? 'active-flag' : ''}" onclick="coachTurn(${m.id})">[${m.coached ? 'Queued for Sleep' : 'Coach / Queue for Sleep'}]</button>
          <button class="foot-link ${m.savored ? 'active-flag' : ''}" onclick="savorTurn(${m.id})">[${m.savored ? 'Vibe Savored' : 'Savor Vibe'}]</button>
        </div>
      `;
      drawerHtml = `<div class="nerve-drawer" style="display:${isOpen ? 'block' : 'none'};">${drawerLines.join('\n')}</div>`;
    } else if (m.guest_analysis) {
      const gw = m.guest_analysis.shadow_pathology_warnings || [];
      const warnTxt = gw.length ? `Shadow Alert: ${gw.map(x => x.shadow_pathology).join(', ')}` : 'No Shadow Pathology triggered';
      footHtml = `<div class="msg-foot"><span>RIGHT-BRAIN GUEST X-RAY · ${warnTxt}</span></div>`;
    }
    card.innerHTML = `
      <div class="msg-head">
        <span>${m.role_label || m.sender}</span>
        <span>#${m.id}</span>
      </div>
      <div class="msg-text">${m.text}</div>
      ${footHtml}
      ${drawerHtml}
    `;
    box.appendChild(card);
  }
  box.scrollTop = box.scrollHeight;
}

async function sendChat() {
  const inp = document.getElementById('chat-input');
  const text = inp.value.trim();
  if (!text) return;
  inp.value = '';
  const modeSel = document.getElementById('sanctuary-mode-select').value;
  await apiPost('/api/chat', {
    text: text,
    room: appState ? appState.active_room : 'chess',
    sanctuary_mode: modeSel
  });
}

async function sendGuest(autoGemini) {
  const inp = document.getElementById('guest-input');
  const text = inp.value.trim();
  inp.value = '';
  await apiPost('/api/guest', {
    text: text,
    auto_query_gemini: !!autoGemini
  });
}

async function runRemSleep() {
  await apiPost('/api/sleep/run', {});
}

async function switchLineage(mode) {
  await apiPost('/api/lineage', { lineage: mode });
}

function renderAll() {
  if (!appState) return;
  document.getElementById('cycle-badge').textContent = `CYCLE ${appState.sleep_cycle_index}`;
  const linSel = document.getElementById('lineage-select');
  if (linSel && appState.active_lineage) linSel.value = appState.active_lineage;
  document.getElementById('status-left-brain').textContent = `LEFT BRAIN: ${appState.left_brain_status}`;
  document.getElementById('status-rules').textContent = `RULES: ${appState.rules_count} | RATIFIED CHORDS: ${appState.ratified_chords_count}`;
  document.getElementById('status-savored').textContent = `SAVORED: ${appState.savored_count}`;
  document.getElementById('status-sleep-queue').textContent = `SLEEP QUEUE: ${appState.sleep_queue_count}`;
  renderBoard();
  renderColorBars();
  renderChat();
}

fetchState();
setInterval(fetchState, 3000);
</script>
</body>
</html>
"""


def create_club_http_server(
    runtime: SanctuaryClubRuntime,
    *,
    host: str = "127.0.0.1",
    port: int = 8420,
) -> ThreadingHTTPServer:
    """Create a thread-safe local HTTP server for the Sanctuary & Chess Club UI."""

    class ClubRequestHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return  # Keep terminal quiet

        def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                body = SINGLE_PAGE_HTML.encode("utf-8")
                self.send_response(int(HTTPStatus.OK))
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/api/state":
                self._send_json(runtime.get_state_snapshot())
                return
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            raw_bytes = self.rfile.read(length) if length > 0 else b"{}"
            try:
                data = json.loads(raw_bytes.decode("utf-8"))
                if self.path == "/api/chess/move":
                    snap = runtime.handle_chess_move(str(data.get("move", "")))
                    self._send_json(snap)
                    return
                if self.path == "/api/chess/fen":
                    snap = runtime.handle_set_fen(str(data.get("fen", "start")))
                    self._send_json(snap)
                    return
                if self.path == "/api/chat":
                    snap = runtime.handle_chat_message(
                        str(data.get("text", "")),
                        room=str(data.get("room", "chess")),
                        sanctuary_mode=str(data.get("sanctuary_mode", "tilt_reset")),
                    )
                    self._send_json(snap)
                    return
                if self.path == "/api/guest":
                    snap = runtime.handle_guest_kibitzer(
                        str(data.get("text", "")),
                        guest_name=str(data.get("guest_name", "Gemini Guest")),
                        auto_query_gemini=bool(data.get("auto_query_gemini", False)),
                    )
                    self._send_json(snap)
                    return
                if self.path == "/api/turn/coach":
                    snap = runtime.flag_turn_for_coaching(
                        int(data["scene_index"]),
                        user_note=str(data.get("note", "")),
                    )
                    self._send_json(snap)
                    return
                if self.path == "/api/turn/savor":
                    snap = runtime.savor_turn_vibe(int(data["scene_index"]))
                    self._send_json(snap)
                    return
                if self.path == "/api/sleep/run":
                    snap = runtime.handle_run_rem_sleep()
                    self._send_json(snap)
                    return
                if self.path == "/api/lineage":
                    snap = runtime.switch_brain_lineage(str(data.get("lineage", "schooled")))
                    self._send_json(snap)
                    return
                self._send_json({"error": f"Unknown endpoint {self.path}"}, HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    return ThreadingHTTPServer((host, port), ClubRequestHandler)


def launch_desktop_window(url: str) -> None:
    """Open `url` in a native frameless Edge app window on Windows, falling back to browser."""
    if sys.platform.startswith("win"):
        edge_candidates = [
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        ]
        for exe in edge_candidates:
            if Path(exe).exists():
                try:
                    subprocess.Popen([exe, f"--app={url}", "--window-size=1280,820"])
                    return
                except Exception:
                    pass
    webbrowser.open(url)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the LMFT Chess Kibitzing Club & Co-Regulation Sanctuary app.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8420, help="Bind port (default: 8420)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the desktop window")
    parser.add_argument("--no-gpu", action="store_true", help="Run in fast Gear-1 Right-Brain mode without loading Qwen3-4B")
    args = parser.parse_args(argv)

    runtime = SanctuaryClubRuntime(lazy_load_gpu=not args.no_gpu)
    server = create_club_http_server(runtime, host=args.host, port=args.port)
    actual_port = server.server_port
    url = f"http://{args.host}:{actual_port}"
    print(f"LMFT Sanctuary & Chess Club running at {url}")
    if not args.no_browser:
        launch_desktop_window(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down LMFT Sanctuary & Chess Club...")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
