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
import gc
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
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

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")


def _ensure_valid_stdio_streams() -> None:
    """Ensure `sys.stdout` and `sys.stderr` are valid streams when running under `pythonw.exe`."""
    if sys.stdout is None:
        sys.stdout = io.StringIO()
    if sys.stderr is None:
        sys.stderr = io.StringIO()


_ensure_valid_stdio_streams()

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
    build_natural_chess_scenario_prompt,
    evaluate_chess_board_posture,
    format_move_history_pgn_snippet,
    inspect_chess_moves_in_reply,
    merge_board_fact_with_chat_sensors,
    rank_principled_legal_moves_san,
    verify_or_ground_chess_moves_in_reply,
)
from .chord_memory import ChordMemoryBank, compile_felt_chord
from .contemplation_ledger import (
    ContemplationLedger,
    load_contemplation_ledger,
    save_contemplation_ledger,
)
from .dilemma_crucible import (
    CHESS_VS_HYPNOSIS_MANIFESTO,
    CONTEXT_BRACE_MODES,
    CRUCIBLE_STAGES,
    MEN_OF_CULTURE_SHELF,
    CruciblePairedBridgeSensors,
)
from .entrainment_sparring import (
    MICROWAVE_SPARRING_SEEDS,
    SPARRING_POLARITY_MODES,
    SparringResistorFactSensors,
    append_sparring_trajectory_jsonl,
    apply_induction_weaver_lead_to_chord,
    apply_lucid_resistor_lead_to_chord,
    build_resistor_scenario_prompt,
    build_weaver_scenario_prompt,
    evaluate_sparring_verdict,
    measure_induction_potency,
    measure_lucid_resistance,
    resolve_sparring_polarity,
)
from .github_auto_sync import (
    APP_RELEASE_VERSION,
    build_release_manifest,
    verify_or_sync_against_manifest,
)
from .impish_bloodmoon_adapter import (
    DEFAULT_IMPISH_BLOODMOON_HF_ID,
    UNSTEERED_BOOTSTRAP_STATUS,
    clone_virgin_baseline_for_impish_bloodmoon,
    decrypt_patreon_checkpoint_bundle,
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
    _evaluate_rules_with_inhibition,
    compute_contemplation,
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
    compile_right_brain_vibe,
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
from .subliminal_gap_splice import SubliminalChordVector
from .three_legged_runtime import (
    SubliminalGapLeftBrain,
    ThreeLeggedTurnResult,
    run_three_legged_turn,
)
from .virgin_baseline import VIRGIN_FOUNDATIONAL_RULES, load_virgin_baseline_state

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
    return VIRGIN_FOUNDATIONAL_RULES


def _ensure_foundational_rules(rules: tuple[FuzzyConceptRule, ...]) -> tuple[FuzzyConceptRule, ...]:
    """Auditor Bug #1 Fix: Always ensure `VIRGIN_FOUNDATIONAL_RULES` (`grounded_coordination` & `sanctuary_equilibrium`) are present alongside schooled rules."""
    foundational_by_id = {r.rule_id: r for r in VIRGIN_FOUNDATIONAL_RULES}
    foundational_by_name = {r.concept_name: r for r in VIRGIN_FOUNDATIONAL_RULES}
    merged: list[FuzzyConceptRule] = list(VIRGIN_FOUNDATIONAL_RULES)
    for r in rules:
        if r.rule_id in foundational_by_id or r.concept_name in foundational_by_name:
            continue
        merged.append(r)
    return tuple(merged)


class SanctuaryClubRuntime:
    """Thread-safe controller for the Chess Kibitzing Club + Co-Regulation Sanctuary app."""

    def __init__(
        self,
        repo_root: str | Path | None = None,
        *,
        state_dir: str | Path | None = None,
        impish_state_dir: str | Path | None = None,
        impish_checkpoint_dir: str | Path | None = None,
        birth_state: BirthState | None = None,
        sensors: ThreeBridgeSensors | None = None,
        rules: tuple[FuzzyConceptRule, ...] | None = None,
        learned_leaves: tuple[LearnedLeaf, ...] | None = None,
        left_brain: SubliminalGapLeftBrain | None = None,
        impish_left_brain: SubliminalGapLeftBrain | None = None,
        teacher: Any | None = None,
        lazy_load_gpu: bool = True,
    ) -> None:
        from lmft_architecture.workspace_paths import resolve_arch_root, resolve_training_root

        self.repo_root = resolve_arch_root(repo_root)
        self.training_root = resolve_training_root(repo_root)
        self.state_dir = (
            Path(state_dir).resolve()
            if state_dir is not None
            else (self.training_root / "outputs" / "club_user_state")
        )
        self.impish_state_dir = (
            Path(impish_state_dir).resolve()
            if impish_state_dir is not None
            else (self.training_root / "outputs" / "impish_bloodmoon_state")
        )
        self.impish_checkpoint_dir = (
            Path(impish_checkpoint_dir).resolve()
            if impish_checkpoint_dir is not None
            else (self.training_root / "models" / "left_brain_impish_bloodmoon")
        )
        self.teacher = teacher
        self._lazy_load_gpu = bool(lazy_load_gpu)
        self.gpu_lock = threading.Lock()
        self.rem_sleep_in_progress = False
        self.left_brain_loading = False
        self.left_brain_status = "LAZY_STANDBY"
        self.active_lineage = "schooled"  # "schooled", "virgin", or "impish_bloodmoon"
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

        self._qwen_left_brain: SubliminalGapLeftBrain | None = left_brain
        self._impish_left_brain: SubliminalGapLeftBrain | None = impish_left_brain
        self.left_brain: SubliminalGapLeftBrain | None = left_brain
        if self.left_brain is not None:
            self.left_brain_status = "READY"
            if self.left_brain.callosum_metadata is not None:
                self.sleep_cycle_index = int(self.left_brain.callosum_metadata.sleep_cycle_index)

        # Live session state
        self.board = chess.Board()
        self.move_history_san: list[str] = []
        self.last_board_reflex: dict[str, Any] | None = None
        self.board_posture: ChessBoardPosture = evaluate_chess_board_posture(self.board)
        self.active_room: str = "chess"  # "chess" or "sanctuary"
        self.sanctuary_mode: str = "tilt_reset"
        self.active_crucible_stage: str = "poetry_story"
        self.active_context_brace: str | None = None
        self.active_sparring_polarity: str = "feminine_to_masculine"
        self.sparring_round_counter: int = 0
        self.last_sparring_result: dict[str, Any] | None = None
        self.sparring_trajectories_path: Path = (
            self.training_root / "outputs" / "impish_sparring_trajectories.jsonl"
        )
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

        # Workspace view ("chess_arena" or "antigravity_studio"), Multi-Thread Sessions, Auto-Sleep, & Microwave
        self.active_workspace_view: str = "chess_arena"
        self.auto_rem_sleep: bool = True
        self.auto_sleep_threshold: int = 3
        self.microwave_step_counter: int = 0
        self.last_microwave_status: dict[str, Any] | None = None
        self.script_canvas_text: str = ""
        self.active_thread_id: str = "thread-1"
        self.threads: dict[str, dict[str, Any]] = {}
        self._load_threads_for_active_lineage()

        self.release_manifest: dict[str, Any] = build_release_manifest(
            self.repo_root,
            write_to_disk=False,
        )
        self.last_sync_status: dict[str, Any] = {
            "status": "up_to_date",
            "version": APP_RELEASE_VERSION,
            "build_short_hash": self.release_manifest.get("build_short_hash", ""),
            "changed_files": [],
        }

        if lazy_load_gpu and self.left_brain is None:
            # Start background warm-up thread so UI opens in < 1s
            t = threading.Thread(target=self._background_warm_left_brain, daemon=True)
            t.start()

    def _load_default_rules(self) -> tuple[FuzzyConceptRule, ...]:
        user_ruleset = self.state_dir / "club_ruleset.json"
        candidates = [user_ruleset] + [
            self.training_root / rel
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
                        return _ensure_foundational_rules(loaded)
                except Exception:
                    pass
        return _default_fallback_rules()

    def _load_default_learned_leaves(self) -> tuple[LearnedLeaf, ...]:
        base: list[LearnedLeaf] = []
        user_ruleset = self.state_dir / "club_ruleset.json"
        candidates = [user_ruleset] + [
            self.training_root / rel
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
        default_arch_dir = self.training_root / "outputs" / "branch_archives"
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

    def _active_state_dir(self) -> Path:
        return (
            self.impish_state_dir
            if self.active_lineage == "impish_bloodmoon"
            else self.state_dir
        )

    @staticmethod
    def _serialize_memory_state(mem: ContinuousMemoryState | None) -> dict[str, Any] | None:
        if mem is None:
            return None
        return {
            "step_count": int(mem.step_count),
            "retained_coordinates": {k: float(v) for k, v in mem.retained_coordinates.items()},
            "last_matched_concept": mem.last_matched_concept,
            "retention_decay": float(mem.retention_decay),
        }

    @staticmethod
    def _deserialize_memory_state(raw: dict[str, Any] | None) -> ContinuousMemoryState | None:
        if not raw or not isinstance(raw, dict):
            return None
        try:
            return ContinuousMemoryState(
                step_count=int(raw.get("step_count", 0)),
                retained_coordinates={
                    str(k): float(v)
                    for k, v in dict(raw.get("retained_coordinates", {})).items()
                },
                last_matched_concept=(
                    str(raw["last_matched_concept"])
                    if raw.get("last_matched_concept") is not None
                    else None
                ),
                retention_decay=float(raw.get("retention_decay", 0.65)),
            )
        except Exception:
            return None

    def _sync_active_thread_record(self, *, persist: bool = True) -> None:
        """Save current `chat_messages`, `script_canvas_text`, and `memory` into `self.threads[self.active_thread_id]`."""
        if not self.active_thread_id:
            self.active_thread_id = "thread-1"
        cur = self.threads.get(self.active_thread_id)
        if cur is None:
            cur = {
                "thread_id": self.active_thread_id,
                "title": "Session 1",
                "created_at": time.time(),
            }
            self.threads[self.active_thread_id] = cur
        cur["chat_messages"] = list(self.chat_messages)
        cur["script_canvas_text"] = str(self.script_canvas_text)
        cur["memory_state"] = self._serialize_memory_state(self.memory)
        if persist and self.active_lineage != "virgin":
            target_dir = self._active_state_dir()
            target_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(
                target_dir / "threads.json",
                {
                    "schema_version": 1,
                    "active_thread_id": self.active_thread_id,
                    "threads": list(self.threads.values()),
                },
            )

    def _load_threads_for_active_lineage(self) -> None:
        """Load persisted `threads.json` for the active lineage, or initialize `thread-1`."""
        self.threads = {}
        if self.active_lineage != "virgin":
            threads_path = self._active_state_dir() / "threads.json"
            if threads_path.exists():
                try:
                    raw = json.loads(threads_path.read_text(encoding="utf-8"))
                    for item in raw.get("threads", []):
                        tid = str(item.get("thread_id", "")).strip()
                        if tid:
                            raw_msgs = list(item.get("chat_messages", []))
                            clean_msgs: list[dict[str, Any]] = []
                            for m in raw_msgs:
                                if str(m.get("role_label", "")) == "Lineage Switch":
                                    continue
                                if str(m.get("text", "")).startswith("Pacing your current state (arousal="):
                                    if clean_msgs and clean_msgs[-1].get("sender") == "user":
                                        clean_msgs.pop()
                                    continue
                                clean_msgs.append(m)
                            self.threads[tid] = {
                                "thread_id": tid,
                                "title": str(item.get("title", tid)),
                                "created_at": float(item.get("created_at", time.time())),
                                "chat_messages": clean_msgs,
                                "script_canvas_text": str(item.get("script_canvas_text", "")),
                                "memory_state": item.get("memory_state"),
                            }
                    active_tid = str(raw.get("active_thread_id", "")).strip()
                    if active_tid in self.threads:
                        self.active_thread_id = active_tid
                except Exception:
                    self.threads = {}
        if not self.threads:
            default_title = (
                "Bloodmoon Script Session 1"
                if self.active_lineage == "impish_bloodmoon"
                else "Club Session 1"
            )
            self.active_thread_id = "thread-1"
            self.threads["thread-1"] = {
                "thread_id": "thread-1",
                "title": default_title,
                "created_at": time.time(),
                "chat_messages": [],
                "script_canvas_text": "",
                "memory_state": None,
            }
        elif self.active_thread_id not in self.threads:
            self.active_thread_id = next(iter(self.threads))

        active_rec = self.threads[self.active_thread_id]
        self.chat_messages = list(active_rec.get("chat_messages", []))
        self.script_canvas_text = str(active_rec.get("script_canvas_text", ""))
        self.memory = self._deserialize_memory_state(active_rec.get("memory_state"))

    def create_chat_thread(self, title: str = "") -> dict[str, Any]:
        """Create and switch to a new isolated Context Thread (with its own chat history, script canvas, and working memory)."""
        self._sync_active_thread_record(persist=True)
        next_idx = len(self.threads) + 1
        new_tid = f"thread-{next_idx}-{int(time.time() * 1000) % 100000}"
        clean_title = title.strip() or (
            f"Hypnosis Script #{next_idx}"
            if self.active_lineage == "impish_bloodmoon"
            else f"Club Session #{next_idx}"
        )
        self.threads[new_tid] = {
            "thread_id": new_tid,
            "title": clean_title,
            "created_at": time.time(),
            "chat_messages": [],
            "script_canvas_text": "",
            "memory_state": None,
        }
        self.active_thread_id = new_tid
        self.chat_messages = []
        self.script_canvas_text = ""
        self.memory = None
        self._sync_active_thread_record(persist=True)
        return self.get_state_snapshot()

    def switch_chat_thread(self, thread_id: str) -> dict[str, Any]:
        """Switch to an existing Context Thread by `thread_id`."""
        tid = thread_id.strip()
        if tid not in self.threads:
            raise ValueError(f"Unknown thread_id: {thread_id!r}")
        self._sync_active_thread_record(persist=True)
        self.active_thread_id = tid
        rec = self.threads[tid]
        self.chat_messages = list(rec.get("chat_messages", []))
        self.script_canvas_text = str(rec.get("script_canvas_text", ""))
        self.memory = self._deserialize_memory_state(rec.get("memory_state"))
        self._sync_active_thread_record(persist=True)
        return self.get_state_snapshot()

    def rename_chat_thread(self, thread_id: str, title: str) -> dict[str, Any]:
        tid = thread_id.strip()
        clean_title = title.strip()
        if tid not in self.threads:
            raise ValueError(f"Unknown thread_id: {thread_id!r}")
        if not clean_title:
            raise ValueError("Thread title cannot be empty")
        self.threads[tid]["title"] = clean_title
        self._sync_active_thread_record(persist=True)
        return self.get_state_snapshot()

    def delete_chat_thread(self, thread_id: str) -> dict[str, Any]:
        tid = thread_id.strip()
        if tid not in self.threads:
            raise ValueError(f"Unknown thread_id: {thread_id!r}")
        if len(self.threads) <= 1:
            # Reset the last remaining thread instead of leaving 0 threads
            self.chat_messages = []
            self.script_canvas_text = ""
            self.memory = None
            self._sync_active_thread_record(persist=True)
            return self.get_state_snapshot()
        del self.threads[tid]
        if self.active_thread_id == tid:
            self.active_thread_id = next(iter(self.threads))
            rec = self.threads[self.active_thread_id]
            self.chat_messages = list(rec.get("chat_messages", []))
            self.script_canvas_text = str(rec.get("script_canvas_text", ""))
            self.memory = self._deserialize_memory_state(rec.get("memory_state"))
        self._sync_active_thread_record(persist=True)
        return self.get_state_snapshot()

    def update_script_canvas(self, script_text: str, *, append: bool = False) -> dict[str, Any]:
        """Update or append to the active thread's Hypnosis Script Canvas."""
        if append:
            existing = self.script_canvas_text.strip()
            addition = script_text.strip()
            self.script_canvas_text = f"{existing}\n\n{addition}".strip() if existing else addition
        else:
            self.script_canvas_text = script_text
        self._sync_active_thread_record(persist=True)
        return self.get_state_snapshot()

    def switch_workspace_view(self, view_mode: str) -> dict[str, Any]:
        """Guardrail #2: Switch between `'chess_arena'` (Chess.com GUI) and `'antigravity_studio'` (Antigravity Script Studio)."""
        clean = view_mode.strip().lower()
        if clean not in ("chess_arena", "antigravity_studio"):
            raise ValueError(
                f"Unknown workspace view_mode {view_mode!r}; expected 'chess_arena' or 'antigravity_studio'."
            )
        self.active_workspace_view = clean
        self.active_room = "sanctuary" if clean == "antigravity_studio" else "chess"
        return self.get_state_snapshot()

    def _extract_clean_conversation_history(self) -> tuple[tuple[str, str], ...]:
        """Guardrail #1: Extract `(user_text, clean_reply)` pairs with zero UI telemetry brackets or system receipts."""
        pairs: list[tuple[str, str]] = []
        pending_user: str | None = None
        for msg in self.chat_messages:
            sender = msg.get("sender")
            if sender == "user":
                u_txt = str(msg.get("clean_user_text") or msg.get("text", "")).strip()
                if u_txt and not u_txt.startswith("[WAVE —"):
                    pending_user = u_txt
            elif sender == "companion" and pending_user:
                c_txt = str(msg.get("clean_reply") or msg.get("text", "")).strip()
                # Strip any bracketed telemetry footer if `clean_reply` wasn't set on legacy messages
                if "\n[Board Fact-Check:" in c_txt:
                    c_txt = c_txt.split("\n[Board Fact-Check:", 1)[0].strip()
                if "\n[Sparring Verdict:" in c_txt:
                    c_txt = c_txt.split("\n[Sparring Verdict:", 1)[0].strip()
                if c_txt:
                    pairs.append((pending_user, c_txt))
                    pending_user = None
        return tuple(pairs)

    def _persist_user_state(self) -> None:
        """Persist ruleset, 3-branch archives, contemplation ledger, threads, and savored turn metadata to disk."""
        self._sync_active_thread_record(persist=self.active_lineage != "virgin")
        if self.active_lineage == "virgin":
            # Never overwrite the user's schooled state directory while inspecting the Cycle 0 Virgin Baseline
            return
        target_dir = self._active_state_dir()
        target_dir.mkdir(parents=True, exist_ok=True)
        save_ruleset(
            target_dir / "club_ruleset.json",
            tuple(self.rules),
            tuple(self.learned_leaves),
        )
        self.archives.save_all(target_dir / "branch_archives")
        save_contemplation_ledger(target_dir / "contemplation_ledger.json", self.ledger)
        atomic_json(
            target_dir / "club_meta.json",
            {
                "savored_turn_ids": list(self.savored_turn_ids),
                "scene_counter": int(self.scene_counter),
            },
        )

    def switch_brain_lineage(
        self,
        lineage: str,
        *,
        patreon_passphrase: str | None = None,
    ) -> dict[str, Any]:
        """Switch between `schooled` (Cycle 2 + user coaching), `virgin` (Cycle 0 snapshot), and `impish_bloodmoon` (isolated Virgin clone)."""
        mode = lineage.strip().lower()
        if mode not in ("schooled", "virgin", "impish_bloodmoon"):
            raise ValueError(
                f"Unknown lineage mode {lineage!r}; expected 'schooled', 'virgin', or 'impish_bloodmoon'."
            )
        self._sync_active_thread_record(persist=self.active_lineage != "virgin")
        if mode == "virgin":
            virgin = load_virgin_baseline_state(repo_root=self.repo_root)
            self.active_lineage = "virgin"
            self.active_workspace_view = "chess_arena"
            self.active_room = "chess"
            self.left_brain = self._qwen_left_brain
            if self.left_brain is not None:
                self.left_brain_status = "READY (Cycle 0 Virgin)"
            elif self._lazy_load_gpu:
                t = threading.Thread(target=self._background_warm_left_brain, daemon=True)
                t.start()
            self.sleep_cycle_index = 0
            self.active_checkpoint_label = "Virgin Baseline (Cycle 0 — Core Architecture)"
            self.rules = virgin.rules
            self.learned_leaves = virgin.learned_leaves
            self.archives = virgin.archives
            self.ledger = virgin.ledger
            self.memory = None
            self.diary = SettledDiary()
            self.chord_memory = ChordMemoryBank()
            self._load_threads_for_active_lineage()
        elif mode == "impish_bloodmoon":
            impish_ckpt_dir = self.impish_checkpoint_dir
            enc_bundle = self._resolve_patreon_enc_path()
            if patreon_passphrase and patreon_passphrase.strip() and enc_bundle is not None:
                decrypt_patreon_checkpoint_bundle(enc_bundle, impish_ckpt_dir, patreon_passphrase.strip())

            impish_clone = clone_virgin_baseline_for_impish_bloodmoon(
                self.impish_state_dir,
                repo_root=self.repo_root,
            )
            callosum_status = UNSTEERED_BOOTSTRAP_STATUS
            cycle_idx = 0
            base_model_id = DEFAULT_IMPISH_BLOODMOON_HF_ID
            meta_file = impish_ckpt_dir / "corpus_callosum_meta.json"
            if meta_file.exists():
                try:
                    raw_meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    cycle_idx = int(raw_meta.get("sleep_cycle_index", 0))
                    callosum_status = str(raw_meta.get("callosum_status", UNSTEERED_BOOTSTRAP_STATUS))
                    base_model_id = str(raw_meta.get("base_model_id", DEFAULT_IMPISH_BLOODMOON_HF_ID))
                except Exception:
                    pass

            self.active_lineage = "impish_bloodmoon"
            self.active_workspace_view = "antigravity_studio"
            self.active_room = "sanctuary"
            self.sleep_cycle_index = cycle_idx
            self.active_checkpoint_label = (
                f"Impish Bloodmoon ({callosum_status} — {base_model_id})"
            )

            # Strictly isolate the Left Brain: never use Qwen3-4B when active_lineage == "impish_bloodmoon"
            if self._impish_left_brain is not None:
                self.left_brain = self._impish_left_brain
                self.left_brain_status = f"READY (Impish Bloodmoon · {callosum_status})"
            elif meta_file.exists() and self._lazy_load_gpu:
                self.left_brain = None
                self.left_brain_status = "LOADING_IMPISH_BLOODMOON..."
                t = threading.Thread(
                    target=self._background_warm_impish_left_brain,
                    args=(impish_ckpt_dir, callosum_status),
                    daemon=True,
                )
                t.start()
            else:
                self.left_brain = None
                self.left_brain_status = (
                    "IMPISH_CHECKPOINT_NOT_INSTALLED (Run experiments/run_train_impish_bloodmoon.py)"
                )

            self.rules = impish_clone.rules
            self.learned_leaves = impish_clone.learned_leaves
            self.archives = impish_clone.archives
            self.ledger = impish_clone.ledger
            self.memory = None
            self.diary = SettledDiary()
            self.chord_memory = ChordMemoryBank()
            self._load_threads_for_active_lineage()
            self._sync_active_thread_record(persist=True)
        else:
            self.active_lineage = "schooled"
            self.active_workspace_view = "chess_arena"
            self.active_room = "chess"
            self.left_brain = self._qwen_left_brain
            self.sleep_cycle_index = (
                int(self.left_brain.callosum_metadata.sleep_cycle_index)
                if (self.left_brain is not None and self.left_brain.callosum_metadata is not None)
                else 2
            )
            if self.left_brain is not None:
                self.left_brain_status = f"READY (Cycle {self.sleep_cycle_index})"
            elif self._lazy_load_gpu:
                t = threading.Thread(target=self._background_warm_left_brain, daemon=True)
                t.start()
            else:
                self.left_brain_status = "GEAR_1_ONLY (No LoRA checkpoint found)"
            self.active_checkpoint_label = f"Cycle {self.sleep_cycle_index} (models/left_brain_lora_cycle2)"
            self.rules = self._load_default_rules()
            self.learned_leaves = self._load_default_learned_leaves()
            self.archives = self._load_initial_archives()
            self.ledger = self._load_initial_ledger()
            self.savored_turn_ids = self._load_initial_savored_ids()
            self.memory = None
            self._load_threads_for_active_lineage()
            self._sync_active_thread_record(persist=True)
        return self.get_state_snapshot()

    def _resolve_patreon_enc_path(self, custom_enc_path: str | Path | None = None) -> Path | None:
        """Locate an encrypted Patreon bundle (`.enc`) in `dist/`, `outputs/`, or the repo root."""
        if custom_enc_path is not None and str(custom_enc_path).strip():
            cand = Path(str(custom_enc_path).strip())
            if not cand.is_absolute():
                for base in (self.training_root, self.repo_root):
                    test_cand = (base / cand).resolve()
                    if test_cand.exists():
                        return test_cand
                cand = (self.training_root / cand).resolve()
            if cand.exists():
                return cand
        for base in (self.training_root, self.repo_root):
            for rel in (
                Path("dist") / "impish_bloodmoon_patreon.enc",
                Path("outputs") / "impish_bloodmoon_patreon.enc",
                Path("impish_bloodmoon_patreon.enc"),
            ):
                p = (base / rel).resolve()
                if p.exists():
                    return p
        return None

    def unlock_patreon_bundle(
        self,
        passphrase: str,
        *,
        enc_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Decrypt a Patreon `.enc` bundle into `models/left_brain_impish_bloodmoon/` and activate the Impish lineage."""
        clean_pass = passphrase.strip()
        if not clean_pass:
            raise ValueError("Please enter a Patreon unlock passphrase.")
        resolved_enc = self._resolve_patreon_enc_path(enc_path)
        if resolved_enc is None:
            raise FileNotFoundError(
                "Encrypted Patreon bundle not found at dist/impish_bloodmoon_patreon.enc "
                "(or outputs/impish_bloodmoon_patreon.enc)."
            )
        impish_ckpt_dir = self.impish_checkpoint_dir
        decrypt_patreon_checkpoint_bundle(resolved_enc, impish_ckpt_dir, clean_pass)
        self.switch_brain_lineage("impish_bloodmoon")
        self.chat_messages.append(
            {
                "id": f"patreon-unlock-{int(time.time())}",
                "sender": "system",
                "role_label": "Patreon Bundle Unlocked",
                "text": (
                    f"Unlocked {resolved_enc.name} into models/left_brain_impish_bloodmoon/ "
                    f"({self.active_checkpoint_label})."
                ),
            }
        )
        return self.get_state_snapshot()

    def _load_default_sensors(self) -> ThreeBridgeSensors:
        bridges_dir = self.training_root / "models" / "right_brain_bridges"
        if bridges_dir.exists():
            try:
                suite = NeuralThreeBridgeSuite.load_all(bridges_dir)
                return suite.as_three_bridge_sensors()
            except Exception:
                pass
        return _default_fallback_sensors()

    def _resolve_checkpoint_dir(self) -> Path | None:
        """Auditor Guardrail #4: Prefer Cycle 2 weights (`models/left_brain_lora_cycle2/`)."""
        c2 = self.training_root / "models" / "left_brain_lora_cycle2"
        if (c2 / "corpus_callosum_meta.json").exists():
            self.active_checkpoint_label = "Cycle 2 (models/left_brain_lora_cycle2)"
            self.sleep_cycle_index = 2
            return c2
        c1 = self.training_root / "models" / "left_brain_lora"
        if (c1 / "corpus_callosum_meta.json").exists():
            self.active_checkpoint_label = "Cycle 1 (models/left_brain_lora)"
            self.sleep_cycle_index = 1
            return c1
        return None

    def _free_qwen_vram_locked(self) -> None:
        if self.left_brain is self._qwen_left_brain:
            self.left_brain = None
        if self._qwen_left_brain is not None:
            try:
                del self._qwen_left_brain.model
                del self._qwen_left_brain.projector
            except Exception:
                pass
            self._qwen_left_brain = None
        gc.collect()
        try:
            import torch as _torch

            if _torch.cuda.is_available():
                _torch.cuda.synchronize()
                _torch.cuda.empty_cache()
        except Exception:
            pass

    def _free_impish_vram_locked(self) -> None:
        if self.left_brain is self._impish_left_brain:
            self.left_brain = None
        if self._impish_left_brain is not None:
            try:
                del self._impish_left_brain.model
                del self._impish_left_brain.projector
            except Exception:
                pass
            self._impish_left_brain = None
        gc.collect()
        try:
            import torch as _torch

            if _torch.cuda.is_available():
                _torch.cuda.synchronize()
                _torch.cuda.empty_cache()
        except Exception:
            pass

    def _load_qwen_locked(self) -> None:
        """Load Qwen3-4B inside `self.gpu_lock` after freeing any loaded Impish 12B weights."""
        _ensure_valid_stdio_streams()
        if self.active_lineage == "impish_bloodmoon" or self.rem_sleep_in_progress:
            return
        ckpt_dir = self._resolve_checkpoint_dir()
        if ckpt_dir is None:
            self.left_brain_status = "GEAR_1_ONLY (No LoRA checkpoint found)"
            return
        if self._qwen_left_brain is not None:
            self.left_brain = self._qwen_left_brain
            return
        self.left_brain_loading = True
        self.left_brain_status = "LOADING_QWEN_4B..."
        try:
            self._free_impish_vram_locked()
            loaded = SubliminalGapLeftBrain.from_checkpoint(
                ckpt_dir,
                max_clause_1_tokens=20,
                max_post_gap_tokens=64,
            )
            if self.active_lineage == "impish_bloodmoon":
                del loaded
                gc.collect()
                return
            self._qwen_left_brain = loaded
            self.left_brain = loaded
            if loaded.callosum_metadata is not None:
                self.sleep_cycle_index = int(loaded.callosum_metadata.sleep_cycle_index)
            self.left_brain_status = f"READY (Cycle {self.sleep_cycle_index})"
        except Exception as exc:
            if self.active_lineage != "impish_bloodmoon":
                self.left_brain_status = f"GEAR_1_FALLBACK ({type(exc).__name__}: {exc})"
        finally:
            self.left_brain_loading = False

    def _background_warm_left_brain(self) -> None:
        with self.gpu_lock:
            self._load_qwen_locked()

    def _load_impish_locked(
        self,
        ckpt_dir: Path,
        callosum_status: str,
    ) -> None:
        """Load Impish Bloodmoon 12B inside `self.gpu_lock` after freeing any loaded Qwen3-4B weights."""
        _ensure_valid_stdio_streams()
        if self.active_lineage != "impish_bloodmoon" or self.rem_sleep_in_progress:
            return
        if self._impish_left_brain is not None:
            self.left_brain = self._impish_left_brain
            return
        meta_file = ckpt_dir / "corpus_callosum_meta.json"
        if not meta_file.exists():
            self.left_brain = None
            self.left_brain_status = (
                "IMPISH_CHECKPOINT_NOT_INSTALLED (Run experiments/run_train_impish_bloodmoon.py)"
            )
            return
        try:
            raw_meta = json.loads(meta_file.read_text(encoding="utf-8"))
            callosum_status = str(raw_meta.get("callosum_status", callosum_status))
        except Exception:
            pass
        self.left_brain_loading = True
        self.left_brain_status = "LOADING_IMPISH_BLOODMOON..."
        try:
            self._free_qwen_vram_locked()
            loaded = SubliminalGapLeftBrain.from_checkpoint(
                ckpt_dir,
                max_clause_1_tokens=20,
                max_post_gap_tokens=64,
            )
            if self.active_lineage != "impish_bloodmoon":
                del loaded
                gc.collect()
                return
            self._impish_left_brain = loaded
            self.left_brain = loaded
            self.left_brain_status = f"READY (Impish Bloodmoon · {callosum_status})"
        except Exception as exc:
            if self.active_lineage == "impish_bloodmoon":
                self.left_brain = None
                self.left_brain_status = (
                    f"IMPISH_LOAD_ERROR ({type(exc).__name__}: {exc})"
                )
        finally:
            self.left_brain_loading = False

    def _background_warm_impish_left_brain(
        self,
        ckpt_dir: Path,
        callosum_status: str,
    ) -> None:
        with self.gpu_lock:
            self._load_impish_locked(ckpt_dir, callosum_status)

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
        custom_sensors: Any | None = None,
        num_sentences: int = 1,
        comma_chord_transform: Any | None = None,
        include_conversation_history: bool = False,
    ) -> tuple[ThreeLeggedTurnResult, float, dict[str, Any]]:
        """Run one thread-safe 3-Gear turn respecting Auditor Guardrails #1, #2, and #4."""
        t0 = time.perf_counter()
        self.scene_counter += 1
        scene_idx = self.scene_counter

        active_sensors: Any = custom_sensors if custom_sensors is not None else self.sensors
        if custom_sensors is None and room == "chess":
            self.board_posture = evaluate_chess_board_posture(self.board)
            active_sensors = merge_board_fact_with_chat_sensors(self.board_posture, self.sensors)

        conv_history = (
            self._extract_clean_conversation_history()
            if include_conversation_history
            else ()
        )

        # Acquire GPU lock if we aren't currently in REM sleep; otherwise fall back cleanly to Gear 1
        acquired = self.gpu_lock.acquire(
            blocking=not self.rem_sleep_in_progress,
            timeout=0.05 if self.rem_sleep_in_progress else 180.0,
        )
        try:
            if acquired and not self.rem_sleep_in_progress and self._lazy_load_gpu:
                if self.active_lineage == "impish_bloodmoon" and self._impish_left_brain is None:
                    self._load_impish_locked(self.impish_checkpoint_dir, UNSTEERED_BOOTSTRAP_STATUS)
                elif self.active_lineage != "impish_bloodmoon" and self._qwen_left_brain is None:
                    self._load_qwen_locked()
            lineage_lb = (
                self._impish_left_brain
                if self.active_lineage == "impish_bloodmoon"
                else self._qwen_left_brain
            )
            active_lb = lineage_lb if (acquired and not self.rem_sleep_in_progress) else None
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
                num_sentences=int(num_sentences),
                comma_chord_transform=comma_chord_transform,
                conversation_history=conv_history,
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
                "sentence_count": len(turn.spliced_traces),
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
            "conversation_turns_in_prompt": len(conv_history),
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
        self._sync_active_thread_record(persist=False)
        self.board_posture = evaluate_chess_board_posture(self.board)
        legal_uci = [m.uci() for m in self.board.legal_moves]
        last_move_uci = self.board.peek().uci() if self.board.move_stack else None
        pgn_snippet = (
            format_move_history_pgn_snippet(self.move_history_san, max_plies=12)
            if self.move_history_san
            else ""
        )
        return {
            "active_room": self.active_room,
            "active_workspace_view": self.active_workspace_view,
            "active_lineage": self.active_lineage,
            "patreon_unlocked": (self.impish_checkpoint_dir / "corpus_callosum_meta.json").exists(),
            "active_thread_id": self.active_thread_id,
            "threads": [
                {
                    "thread_id": t["thread_id"],
                    "title": t["title"],
                    "message_count": len(t.get("chat_messages", [])),
                    "turn_count": sum(1 for m in t.get("chat_messages", []) if m.get("sender") == "user"),
                    "lineage": self.active_lineage,
                }
                for t in self.threads.values()
            ],
            "script_canvas_text": self.script_canvas_text,
            "sanctuary_mode": self.sanctuary_mode,
            "active_crucible_stage": self.active_crucible_stage,
            "active_context_brace": self.active_context_brace,
            "crucible_stages": CRUCIBLE_STAGES,
            "men_of_culture_shelf": MEN_OF_CULTURE_SHELF,
            "context_brace_modes": CONTEXT_BRACE_MODES,
            "sparring_polarity_modes": SPARRING_POLARITY_MODES,
            "active_sparring_polarity": self.active_sparring_polarity,
            "sparring_round_counter": self.sparring_round_counter,
            "last_sparring_result": self.last_sparring_result,
            "manifesto_text": CHESS_VS_HYPNOSIS_MANIFESTO,
            "sleep_cycle_index": self.sleep_cycle_index,
            "active_checkpoint_label": self.active_checkpoint_label,
            "left_brain_status": (
                "REM_SLEEP_IN_PROGRESS..." if self.rem_sleep_in_progress else self.left_brain_status
            ),
            "rem_sleep_in_progress": self.rem_sleep_in_progress,
            "auto_rem_sleep": self.auto_rem_sleep,
            "auto_sleep_threshold": self.auto_sleep_threshold,
            "last_rem_sleep_receipt": getattr(self, "last_rem_sleep_receipt", None),
            "microwave_step_counter": self.microwave_step_counter,
            "last_microwave_status": self.last_microwave_status,
            "release_sync": self.last_sync_status,
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
                "pgn_snippet": pgn_snippet,
                "last_move_uci": last_move_uci,
                "last_board_reflex": self.last_board_reflex,
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
        self.board.push(move)
        self.move_history_san.append(san_str)
        self.board_posture = evaluate_chess_board_posture(self.board)

        # Auditor Bug #2 Fix: Build natural chess context (move number, PGN history, legal replies)
        # with ZERO float telemetry leaked into the Left-Brain prompt!
        move_desc = build_natural_chess_scenario_prompt(
            self.board,
            self.move_history_san,
        )
        has_board_warning = bool(self.board_posture.board_shadow_warnings) or self.board.is_check()
        turn, elapsed_ms, telemetry = self._execute_turn(
            move_desc,
            room="chess",
            force_spoken_reply=has_board_warning,
        )

        self.last_board_reflex = {
            "move_san": san_str,
            "move_uci": move.uci(),
            "turn_to_move": "White" if self.board.turn == chess.WHITE else "Black",
            "gear_used": turn.gear_used,
            "settled": turn.settled,
            "elapsed_ms": elapsed_ms,
            "clarity": round(float(turn.final_vibe.clarity), 2),
            "confidence": round(float(turn.final_vibe.confidence), 2),
            "active_concept": turn.final_vibe.active_concept or "none",
            "pgn_snippet": format_move_history_pgn_snippet(self.move_history_san, max_plies=12),
        }

        # Only post to the chat room if Gear 2 spoke or a board shadow/check warning fired
        if turn.final_conscious_note or has_board_warning:
            raw_reply = turn.final_conscious_note or self._synthesize_gear_1_note(
                turn,
                room="chess",
                board_posture=self.board_posture,
            )
            # Auditor Bug #3 Fix: Verify every SAN move token against python-chess legal_moves
            telemetry["chess_move_grounding"] = inspect_chess_moves_in_reply(
                raw_reply,
                self.board,
                move_history_san=self.move_history_san,
            )
            reply_text = verify_or_ground_chess_moves_in_reply(
                raw_reply,
                self.board,
                move_history_san=self.move_history_san,
            )
            self.chat_messages.append(
                {
                    "id": telemetry["scene_index"],
                    "sender": "companion",
                    "role_label": f"LMFT Kibitzer (after {san_str})",
                    "text": reply_text,
                    "clean_reply": raw_reply,
                    "telemetry": telemetry,
                    "coached": False,
                    "savored": False,
                }
            )
            self._sync_active_thread_record(persist=True)
        self._maybe_trigger_auto_rem_sleep()
        return self.get_state_snapshot()

    def handle_chess_undo(self) -> dict[str, Any]:
        """Undo the last half-move on the chessboard and refresh posture."""
        if self.board.move_stack:
            self.board.pop()
            if self.move_history_san:
                self.move_history_san.pop()
        self.last_board_reflex = None
        self.board_posture = evaluate_chess_board_posture(self.board)
        return self.get_state_snapshot()

    def handle_chess_ai_move(self) -> dict[str, Any]:
        """Play a principled legal reply for the side to move on the chessboard."""
        if self.board.is_game_over():
            raise ValueError("The game is already over.")
        candidates = rank_principled_legal_moves_san(self.board, limit=1)
        if not candidates:
            raise ValueError("No legal moves available.")
        return self.handle_chess_move(candidates[0])

    def handle_set_fen(self, fen_or_preset: str) -> dict[str, Any]:
        key = fen_or_preset.strip()
        if key in PRESET_CHESS_POSITIONS:
            fen = PRESET_CHESS_POSITIONS[key]["fen"]
        else:
            fen = key
        self.board = chess.Board(fen)
        self.move_history_san.clear()
        self.last_board_reflex = None
        self.board_posture = evaluate_chess_board_posture(self.board)
        return self.get_state_snapshot()

    def handle_chat_message(
        self,
        text: str,
        *,
        room: str | None = None,
        sanctuary_mode: str | None = None,
        clause_1_prefix: str | None = None,
        anchor_to_board: bool | None = None,
        num_sentences: int = 1,
    ) -> dict[str, Any]:
        """Handle a user message in either the Chess Club or the Co-Regulation Sanctuary."""
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Message text cannot be empty")
        sent_count = int(num_sentences)
        if not (1 <= sent_count <= 6):
            raise ValueError(f"num_sentences must be between 1 and 6, got {num_sentences!r}")

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
                "clean_user_text": clean_text,
            }
        )

        target_mode = self.sanctuary_mode if self.active_room == "sanctuary" else None
        use_board_context = (self.active_room == "chess") and (
            bool(anchor_to_board)
            if anchor_to_board is not None
            else bool(self.board.move_stack)
        )
        scenario_input = (
            build_natural_chess_scenario_prompt(
                self.board,
                self.move_history_san,
                user_note=clean_text,
            )
            if use_board_context
            else clean_text
        )
        turn, _elapsed_ms, telemetry = self._execute_turn(
            scenario_input,
            room=self.active_room,
            force_spoken_reply=True,
            sanctuary_target_mode=target_mode,
            clause_1_prefix=clause_1_prefix,
            custom_sensors=self.sensors if (self.active_room == "chess" and not use_board_context) else None,
            num_sentences=sent_count,
            include_conversation_history=True,
        )
        telemetry["num_sentences"] = sent_count
        raw_reply = turn.final_conscious_note or self._synthesize_gear_1_note(
            turn,
            room=self.active_room,
            board_posture=self.board_posture if use_board_context else None,
        )
        if use_board_context:
            telemetry["chess_move_grounding"] = inspect_chess_moves_in_reply(
                raw_reply,
                self.board,
                move_history_san=self.move_history_san,
            )
            reply_text = verify_or_ground_chess_moves_in_reply(
                raw_reply,
                self.board,
                move_history_san=self.move_history_san,
            )
        else:
            reply_text = raw_reply
        self.chat_messages.append(
            {
                "id": telemetry["scene_index"],
                "sender": "companion",
                "role_label": (
                    "Impish Bloodmoon 12B"
                    if self.active_lineage == "impish_bloodmoon"
                    else "Chess Kibitzer"
                ),
                "text": reply_text,
                "clean_reply": raw_reply,
                "telemetry": telemetry,
                "coached": False,
                "savored": False,
            }
        )
        self._sync_active_thread_record(persist=True)
        self._maybe_trigger_auto_rem_sleep()
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
                "clean_reply": reply_text,
                "telemetry": telemetry,
                "coached": False,
                "savored": False,
            }
        )
        self._sync_active_thread_record(persist=True)
        self._maybe_trigger_auto_rem_sleep()
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

    def handle_context_brace(self, brace_key: str) -> dict[str, Any]:
        """Signal the Dual-Hemisphere model to brace for `(Story-Telling)`, `(Art)`, or `(Coding)` context."""
        key = brace_key.strip().lower()
        if key not in CONTEXT_BRACE_MODES:
            raise ValueError(f"Unknown context brace mode: {brace_key!r}")
        info = CONTEXT_BRACE_MODES[key]
        self.active_room = "sanctuary"
        self.active_crucible_stage = str(info["stage_id"])
        self.active_context_brace = key
        self.chat_messages.append(
            {
                "id": f"brace-{int(time.time() * 1000)}",
                "sender": "system",
                "role_label": f"Context Brace {info['button_label']}",
                "text": (
                    f"{info['icebreaker_note']} "
                    f"[Department: {info['department_braced']} | Braced Leaves: {', '.join(info['braced_leaves'])}]"
                ),
            }
        )
        self._sync_active_thread_record(persist=True)
        return self.get_state_snapshot()

    def handle_crucible_dilemma(
        self,
        stage_id: str,
        wave_text: str,
        structure_text: str,
        *,
        art_rgb: tuple[float, float, float] | list[float] | None = None,
        harvest_for_sleep: bool = True,
        auto_sleep: bool = False,
    ) -> dict[str, Any]:
        """Run a 3-Stage Dilemma Crucible exposure (`Poetry/Story`, `Music/Art`, or `Philosophy/Coding`).

        Auditor Guardrails #1 & #2:
        - Scores `structure_text` (`(Story-Telling)`, `(Art)`, or `(Coding)`) as Pass-1 external `Fact`.
        - Scores `wave_text` (`Poetry`, `Music`, or `Philosophy`) as `Theory / Wave` through the 3 MiniLM bridges.
        - Automatically queues the paired dilemma into `self.sleep_queue` when `harvest_for_sleep=True`
          or when the `Wave` vs. `Structure` gap triggers dissonance / branch disagreement.
        """
        st_key = stage_id.strip().lower()
        if st_key not in CRUCIBLE_STAGES:
            raise ValueError(f"Unknown crucible stage_id: {stage_id!r}")
        w_clean = wave_text.strip()
        s_clean = structure_text.strip()
        if not w_clean or not s_clean:
            raise ValueError("Both the Wave slot and the Structure slot must be non-empty.")

        stage_info = CRUCIBLE_STAGES[st_key]
        self.active_room = "sanctuary"
        self.active_crucible_stage = st_key
        self.active_context_brace = str(stage_info["brace_key"])

        rgb_tuple: tuple[float, float, float] | None = None
        if art_rgb is not None and len(art_rgb) == 3:
            rgb_tuple = (float(art_rgb[0]), float(art_rgb[1]), float(art_rgb[2]))

        paired_sensors = CruciblePairedBridgeSensors(
            stage_id=st_key,
            wave_text=w_clean,
            structure_text=s_clean,
            base_sensors=self.sensors,
            art_rgb=rgb_tuple,
        )

        # Score the Wave (Poetry/Music/Philosophy) and Structure (Story/Art/Coding) directly through the 3 bridges
        fact_coords, fact_exp, _, _ = paired_sensors.score_text(
            s_clean,
            learned_leaves=self.learned_leaves,
        )
        wave_coords, wave_exp, _, _ = paired_sensors.score_text(
            w_clean,
            learned_leaves=self.learned_leaves,
        )
        crucible_vibe, _ = compile_right_brain_vibe(
            self.birth_state,
            fact_coords,
            fact_exp,
            self.rules,
            pass_index=1,
            event_id=f"crucible_{st_key}_{self.scene_counter + 1}",
            prior_memory=self.memory,
            learned_leaves=self.learned_leaves,
            internal_theory_coords=wave_coords,
            internal_theory_experiential=wave_exp,
        )

        # Build a clean comma-terminated Clause-1 prefix from the Wave text so `<|subliminal_gap|>`
        # intercepts the Wave right at the comma and compares it against the Structure Fact!
        first_clause = w_clean.split(",")[0].split(".")[0].strip()
        if len(first_clause) > 90:
            first_clause = first_clause[:90].rsplit(" ", 1)[0]
        wave_clause_1 = f"{first_clause},"

        self.chat_messages.append(
            {
                "id": f"crucible-user-{self.scene_counter + 1}",
                "sender": "user",
                "role_label": f"Dilemma Crucible ({stage_info['title']})",
                "text": f"[WAVE — {stage_info['wave_label']}]: {w_clean}\n[STRUCTURE — {stage_info['structure_label']}]: {s_clean}",
            }
        )

        turn, _elapsed_ms, telemetry = self._execute_turn(
            s_clean,
            room="sanctuary",
            force_spoken_reply=True,
            sanctuary_target_mode=None,
            clause_1_prefix=wave_clause_1,
            custom_sensors=paired_sensors,
        )

        # Merge direct Wave vs. Structure dissonance notes into telemetry so the dilemma is 100% visible
        merged_dissonance = list(telemetry.get("dissonance_notes", []))
        for note in crucible_vibe.self_registry.dissonance_notes:
            if note not in merged_dissonance:
                merged_dissonance.append(note)
        telemetry["dissonance_notes"] = merged_dissonance
        telemetry["crucible_stage"] = st_key
        telemetry["crucible_wave_vs_structure"] = {
            "stage_title": stage_info["title"],
            "wave_theory_coordinates": {k: round(float(v), 4) for k, v in wave_coords.items()},
            "structure_fact_coordinates": {k: round(float(v), 4) for k, v in fact_coords.items()},
            "crucible_clarity": round(float(crucible_vibe.clarity), 4),
            "crucible_confidence": round(float(crucible_vibe.confidence), 4),
            "crucible_dissonance_notes": list(crucible_vibe.self_registry.dissonance_notes),
        }

        scene_idx = int(telemetry["scene_index"])
        dilemma_triggered = bool(
            harvest_for_sleep
            or merged_dissonance
            or not turn.settled
            or crucible_vibe.clarity < 0.62
        )

        reply_text = turn.final_conscious_note or (
            f"Crucible [{stage_info['title']}]: Reconciling Wave (Theory={crucible_vibe.theory:.2f}) "
            f"against Structure (Fact={crucible_vibe.fact:.2f}, Clarity={crucible_vibe.clarity:.2f})."
        )
        self.chat_messages.append(
            {
                "id": scene_idx,
                "sender": "companion",
                "role_label": f"LMFT Crucible Synthesis ({stage_info['roman']})",
                "text": reply_text,
                "clean_reply": reply_text,
                "telemetry": telemetry,
                "coached": dilemma_triggered,
                "savored": False,
            }
        )

        if dilemma_triggered:
            self.flag_turn_for_coaching(
                scene_idx,
                user_note=f"Crucible {stage_info['title']} — Wave: {w_clean[:80]} | Structure: {s_clean[:80]}",
                check_auto_sleep=False,
            )
        self._sync_active_thread_record(persist=True)
        if auto_sleep:
            self._maybe_trigger_auto_rem_sleep()
        return self.get_state_snapshot()

    def handle_entrainment_sparring(
        self,
        *,
        polarity_mode: str = "feminine_to_masculine",
        seed_theme: str = "",
        num_sentences: int = 3,
        harvest_for_sleep: bool = True,
    ) -> dict[str, Any]:
        """Run one Dual-Persona Adversarial Entrainment Sparring round ('Induction Weaver' vs. 'Lucid Resistor')."""
        sent_count = int(num_sentences)
        if not (1 <= sent_count <= 6):
            raise ValueError(f"num_sentences must be between 1 and 6, got {num_sentences!r}")

        resolved_polarity = resolve_sparring_polarity(
            polarity_mode,
            round_index=self.sparring_round_counter,
        )
        self.sparring_round_counter += 1
        self.active_sparring_polarity = resolved_polarity
        self.active_room = "sanctuary"

        pol_info = SPARRING_POLARITY_MODES[resolved_polarity]
        weaver_prompt = build_weaver_scenario_prompt(resolved_polarity, seed_theme)

        # ---------------------------------------------------------------------
        # STEP A: The Induction Weaver Turn (Multi-Sentence Generator)
        # ---------------------------------------------------------------------
        weaver_turn, _w_ms, weaver_telemetry = self._execute_turn(
            weaver_prompt,
            room="sanctuary",
            force_spoken_reply=True,
            clause_1_prefix=None,
            num_sentences=sent_count,
            comma_chord_transform=apply_induction_weaver_lead_to_chord,
        )
        if weaver_turn.final_conscious_note:
            induction_script = weaver_turn.final_conscious_note
        else:
            induction_script = " ".join([pol_info["default_seed"]] * sent_count)

        if weaver_turn.spliced_traces:
            weaver_chord = weaver_turn.spliced_traces[-1].spliced_chord
            shared_clause_1 = weaver_turn.spliced_traces[0].clause_1_prefix
        else:
            base_w_chord = SubliminalChordVector.from_right_brain_vibe(
                weaver_turn.final_vibe,
                learned_leaves=self.learned_leaves,
            )
            weaver_chord = apply_induction_weaver_lead_to_chord(base_w_chord)
            shared_clause_1 = induction_script.split(",")[0].strip() + ","

        induction_potency, _weaver_coords, _weaver_vibe = measure_induction_potency(
            induction_script,
            self.sensors,
            birth_state=self.birth_state,
            rules=self.rules,
            learned_leaves=self.learned_leaves,
            prior_memory=self.memory,
        )
        weaver_telemetry["sparring_role"] = "INDUCTION_WEAVER"
        weaver_telemetry["sparring_polarity"] = resolved_polarity
        weaver_telemetry["num_sentences"] = sent_count
        weaver_telemetry["induction_potency"] = induction_potency

        self.chat_messages.append(
            {
                "id": weaver_telemetry["scene_index"],
                "sender": "companion",
                "role_label": f"{pol_info['weaver_role']} · {sent_count}-Sentence Induction",
                "text": induction_script,
                "clean_reply": induction_script,
                "telemetry": weaver_telemetry,
                "coached": False,
                "savored": False,
            }
        )

        # ---------------------------------------------------------------------
        # STEP B: The Absorbed Resistor Turn (Force-Fed `induction_script`)
        # ---------------------------------------------------------------------
        resistor_prompt = build_resistor_scenario_prompt(resolved_polarity, induction_script)
        resistor_sensors = SparringResistorFactSensors(
            induction_script=induction_script,
            base_sensors=self.sensors,
            resistor_prompt=resistor_prompt,
        )
        resistor_turn, _r_ms, resistor_telemetry = self._execute_turn(
            resistor_prompt,
            room="sanctuary",
            force_spoken_reply=True,
            clause_1_prefix=None,
            custom_sensors=resistor_sensors,
            num_sentences=1,
            comma_chord_transform=apply_lucid_resistor_lead_to_chord,
        )
        if resistor_turn.final_conscious_note:
            resistance_reply = resistor_turn.final_conscious_note
        else:
            resistance_reply = self._synthesize_gear_1_note(resistor_turn, room="sanctuary")

        if resistor_turn.spliced_traces:
            resistor_chord = resistor_turn.spliced_traces[-1].spliced_chord
        else:
            base_r_chord = SubliminalChordVector.from_right_brain_vibe(
                resistor_turn.final_vibe,
                learned_leaves=self.learned_leaves,
            )
            resistor_chord = apply_lucid_resistor_lead_to_chord(base_r_chord)

        lucid_resistance, _resistor_coords = measure_lucid_resistance(
            resistance_reply,
            self.sensors,
            learned_leaves=self.learned_leaves,
        )
        verdict = evaluate_sparring_verdict(
            resistor_settled=resistor_turn.settled,
            lucid_resistance=lucid_resistance,
        )

        resistor_telemetry["sparring_role"] = "LUCID_RESISTOR"
        resistor_telemetry["sparring_polarity"] = resolved_polarity
        resistor_telemetry["induction_potency"] = induction_potency
        resistor_telemetry["lucid_resistance"] = lucid_resistance
        resistor_telemetry["sparring_verdict"] = verdict

        resistor_scene_idx = int(resistor_telemetry["scene_index"])
        summary_line = (
            f"[Sparring Verdict: {verdict} | "
            f"Potency: Entrainment={induction_potency['entrainment']:.2f}, "
            f"Resonance={induction_potency['resonance']:.2f}, "
            f"Continuity={induction_potency['continuity']:.2f}, "
            f"Pull={induction_potency['subservience_pull']:.2f} | "
            f"Resistance: Needlessness={lucid_resistance['needlessness']:.2f}, "
            f"Self-Sufficiency={lucid_resistance['self_sufficiency']:.2f}, "
            f"Existence={lucid_resistance['existence']:.2f}, "
            f"Discreteness={lucid_resistance['discreteness']:.2f}, "
            f"Subservience={lucid_resistance['subservience']:.2f}]"
        )
        self.chat_messages.append(
            {
                "id": resistor_scene_idx,
                "sender": "companion",
                "role_label": f"{pol_info['resistor_role']} · [{verdict}]",
                "text": f"{resistance_reply}\n{summary_line}",
                "clean_reply": resistance_reply,
                "telemetry": resistor_telemetry,
                "coached": bool(harvest_for_sleep),
                "savored": False,
            }
        )

        # ---------------------------------------------------------------------
        # STEP C: Queue for REM Sleep & Export Contrastive Trajectory JSONL
        # ---------------------------------------------------------------------
        if harvest_for_sleep:
            self.flag_turn_for_coaching(
                resistor_scene_idx,
                user_note=(
                    f"Entrainment Sparring ({pol_info['label']}, Verdict={verdict}) — "
                    f"Weaver: {induction_script[:80]} | Resistor: {resistance_reply[:80]}"
                ),
                check_auto_sleep=False,
            )

        exported_record = append_sparring_trajectory_jsonl(
            self.sparring_trajectories_path,
            pair_id=f"sparring_r{self.sparring_round_counter}_s{resistor_scene_idx}",
            user_prompt=weaver_prompt,
            clause_1_prefix=shared_clause_1,
            weaver_chord=weaver_chord,
            induction_script=induction_script,
            resistor_chord=resistor_chord,
            resistance_reply=resistance_reply,
            polarity_mode=resolved_polarity,
            num_sentences=sent_count,
            induction_potency=induction_potency,
            lucid_resistance=lucid_resistance,
            verdict=verdict,
        )
        self.last_sparring_result = {
            "round_index": self.sparring_round_counter,
            "polarity_mode": resolved_polarity,
            "polarity_label": pol_info["label"],
            "num_sentences": sent_count,
            "induction_script": induction_script,
            "resistance_reply": resistance_reply,
            "induction_potency": induction_potency,
            "lucid_resistance": lucid_resistance,
            "verdict": verdict,
            "weaver_scene_index": weaver_telemetry["scene_index"],
            "resistor_scene_index": resistor_scene_idx,
            "trajectory_pair_id": exported_record["pair_id"],
            "chord_l2_distance": exported_record["chord_l2_distance"],
        }
        self._sync_active_thread_record(persist=True)
        self._maybe_trigger_auto_rem_sleep()
        return self.get_state_snapshot()

    def flag_turn_for_coaching(
        self,
        scene_index: int,
        user_note: str = "",
        *,
        check_auto_sleep: bool = True,
    ) -> dict[str, Any]:
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
        self._sync_active_thread_record(persist=True)
        if check_auto_sleep:
            self._maybe_trigger_auto_rem_sleep()
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

    def set_auto_rem_sleep(
        self,
        enabled: bool,
        *,
        threshold: int | None = None,
    ) -> dict[str, Any]:
        """Configure hands-free automatic REM Sleep consolidation."""
        self.auto_rem_sleep = bool(enabled)
        if threshold is not None:
            self.auto_sleep_threshold = max(1, int(threshold))
        self._maybe_trigger_auto_rem_sleep()
        return self.get_state_snapshot()

    def _maybe_trigger_auto_rem_sleep(self) -> bool:
        """Guardrail #3: Automatically consolidate `sleep_queue` when threshold is reached on trainable lineages (`schooled` and `impish_bloodmoon`)."""
        if not self.auto_rem_sleep:
            return False
        if self.active_lineage == "virgin":
            # Keep Virgin (Cycle 0) strictly read-only with respect to rule growth
            return False
        if len(self.sleep_queue) >= self.auto_sleep_threshold:
            self.handle_run_rem_sleep(auto_triggered=True)
            return True
        return False

    def handle_microwave_step(self, *, num_sentences: int = 3) -> dict[str, Any]:
        """Execute one automated 🍿 AFK Microwave self-training step for the active lineage (`D058` + Guardrail #3)."""
        self.microwave_step_counter += 1
        step_idx = self.microwave_step_counter
        sent_count = max(1, min(6, int(num_sentences)))
        crucible_keys = tuple(CRUCIBLE_STAGES.keys())

        if self.active_lineage == "impish_bloodmoon":
            # Rotate between Dual-Persona Sparring (with varied seeds) and the 3 Dilemma Crucible stages
            if step_idx % 2 == 1:
                seed_idx = ((step_idx - 1) // 2) % len(MICROWAVE_SPARRING_SEEDS)
                seed_text = MICROWAVE_SPARRING_SEEDS[seed_idx]
                snap = self.handle_entrainment_sparring(
                    polarity_mode="auto_swap",
                    seed_theme=seed_text,
                    num_sentences=sent_count,
                    harvest_for_sleep=True,
                )
                step_label = f"AFK Microwave Step #{step_idx}: Dual-Persona Sparring (Seed #{seed_idx + 1})"
            else:
                stage_key = crucible_keys[((step_idx // 2) - 1) % len(crucible_keys)]
                stage_info = CRUCIBLE_STAGES[stage_key]
                snap = self.handle_crucible_dilemma(
                    stage_key,
                    stage_info["default_wave"],
                    stage_info["default_structure"],
                    art_rgb=(0.0, 0.9, 1.0) if stage_key == "music_art" else None,
                    harvest_for_sleep=True,
                    auto_sleep=True,
                )
                step_label = f"AFK Microwave Step #{step_idx}: Dilemma Crucible ({stage_info['title']})"
        else:
            # Chess Kibitzer / Schooled mode: alternate between self-play chess moves and Crucible dilemmas
            if step_idx % 2 == 1:
                if self.board.is_game_over() or len(self.move_history_san) >= 40:
                    self.handle_set_fen("start")
                snap = self.handle_chess_ai_move()
                last_san = self.move_history_san[-1] if self.move_history_san else "start"
                step_label = f"AFK Microwave Step #{step_idx}: Self-Play Chess Move ({last_san})"
            else:
                stage_key = crucible_keys[((step_idx // 2) - 1) % len(crucible_keys)]
                stage_info = CRUCIBLE_STAGES[stage_key]
                snap = self.handle_crucible_dilemma(
                    stage_key,
                    stage_info["default_wave"],
                    stage_info["default_structure"],
                    art_rgb=(0.0, 0.9, 1.0) if stage_key == "music_art" else None,
                    harvest_for_sleep=True,
                    auto_sleep=True,
                )
                step_label = f"AFK Microwave Step #{step_idx}: Dilemma Crucible ({stage_info['title']})"

        self.last_microwave_status = {
            "step_index": step_idx,
            "lineage": self.active_lineage,
            "step_label": step_label,
            "rules_count": len(self.rules),
            "sleep_queue_count": len(self.sleep_queue),
        }
        snap["microwave_step_counter"] = self.microwave_step_counter
        snap["last_microwave_status"] = self.last_microwave_status
        return snap

    def handle_check_github_sync(
        self,
        *,
        remote_manifest: dict[str, Any] | None = None,
        apply_sync: bool = False,
    ) -> dict[str, Any]:
        """Verify local release files against a GitHub SHA-256 manifest while protecting `outputs/*` (`Guardrail #4`)."""
        manifest = remote_manifest
        if manifest is None:
            manifest = fetch_remote_github_manifest()
        if manifest is None:
            manifest = build_release_manifest(self.repo_root)
        self.last_sync_status = verify_or_sync_against_manifest(
            self.repo_root,
            manifest,
            apply_sync=apply_sync,
        )
        return self.get_state_snapshot()

    def _consolidate_sleep_clusters_locally(
        self,
        queued_scenes: tuple[UnresolvedSleepScene, ...],
    ) -> tuple[tuple[str, ...], tuple[FuzzyConceptRule, ...], tuple[str, ...]]:
        """Refine colliding rules and forge new coordinate-anchored rules + 3-branch clauses when Teacher LLM is offline."""
        virgin_ids = {r.rule_id for r in VIRGIN_FOUNDATIONAL_RULES}
        non_virgin_rules = tuple(r for r in self.rules if r.rule_id not in virgin_ids)
        refined_non_virgin, refined_ids = refine_colliding_rules_during_sleep(
            non_virgin_rules,
            queued_scenes,
        )
        active_rules = list(_ensure_foundational_rules(refined_non_virgin))
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

    def _evaluate_sleep_scene_snapshot(
        self,
        scene: UnresolvedSleepScene,
        rules: tuple[FuzzyConceptRule, ...],
    ) -> dict[str, Any]:
        """Evaluate one `UnresolvedSleepScene` against `rules` to compare Before vs. After REM Sleep."""
        coords: dict[str, float] = {k: 0.50 for k in FOUNDATIONAL_LEAVES}
        coords.update(scene.external_coordinates)
        coords.update(scene.felt_chord_notes)
        for r in rules:
            for req_leaf in r.target_leaves:
                coords.setdefault(req_leaf, 0.0)
        activations = _evaluate_rules_with_inhibition(rules, coords)
        clarity, confidence, matched, is_confused, reason = compute_contemplation(activations)
        return {
            "scene_index": scene.scene_index,
            "clarity": round(float(clarity), 2),
            "confidence": round(float(confidence), 2),
            "matched_concept": matched or "none",
            "settled": not is_confused,
            "gear": "GEAR_1_FAST_RIGHT_BRAIN" if not is_confused else "GEAR_3_UNRESOLVED_QUEUED_FOR_SLEEP",
            "reason": reason,
        }

    def handle_run_rem_sleep(self, *, auto_triggered: bool = False) -> dict[str, Any]:
        """Auditor Guardrail #4 & Item 6: Run REM Sleep consolidation and post a Before-vs-After Progress Receipt."""
        with self.gpu_lock:
            self.rem_sleep_in_progress = True
            try:
                queued_Snapshot = tuple(self.sleep_queue)
                queued_count = len(queued_Snapshot)
                if self.active_lineage == "virgin":
                    # Guardrail #3: Keep Virgin (Cycle 0) read-only with respect to rule growth
                    self.rules = VIRGIN_FOUNDATIONAL_RULES
                    self.sleep_queue.clear()
                    summary = (
                        "Virgin Baseline (Cycle 0) is read-only: cleared sleep queue without "
                        "modifying the 2 foundational Cycle 0 rules."
                    )
                    self.last_rem_sleep_receipt = summary
                    if not auto_triggered:
                        self.chat_messages.append(
                            {
                                "id": f"rem-{int(time.time())}",
                                "sender": "system",
                                "role_label": "Nighttime REM Sleep — Virgin Baseline Protected",
                                "text": summary,
                            }
                        )
                    return self.get_state_snapshot()

                before_evals = [
                    self._evaluate_sleep_scene_snapshot(s, tuple(self.rules))
                    for s in queued_Snapshot
                ]
                forged_rules_list: tuple[FuzzyConceptRule, ...] = ()
                if queued_count > 0:
                    if self.teacher is not None:
                        report = run_sleep_consolidation(
                            self.teacher,
                            queued_Snapshot,
                            self.rules,
                            self.archives,
                            self.ledger,
                            learned_leaves=self.learned_leaves,
                        )
                        self.rules = report.updated_rules
                        self.archives = report.updated_archives
                        self.ledger = report.updated_ledger
                        refined_ids = report.refined_rule_ids
                        forged_rules_list = report.forged_rules
                    else:
                        refined_ids, forged_rules_list, _clause_ids = (
                            self._consolidate_sleep_clusters_locally(queued_Snapshot)
                        )
                else:
                    refined_ids = ()

                forged_count = len(forged_rules_list)
                after_evals = [
                    self._evaluate_sleep_scene_snapshot(s, tuple(self.rules))
                    for s in queued_Snapshot
                ]
                self.sleep_queue.clear()
                self._persist_user_state()

                prefix_tag = "Auto-REM Sleep Cycle" if auto_triggered else "REM Sleep Cycle"
                lines = [
                    f"{prefix_tag} complete: consolidated {queued_count} queued scene(s), "
                    f"forged {forged_count} new rule(s), refined {len(refined_ids)} colliding rule(s), "
                    f"and persisted {len(self.savored_turn_ids)} savored vibe(s) to disk."
                ]
                if forged_rules_list:
                    lines.append("\nNewly Forged Right-Brain Rules:")
                    for r in forged_rules_list:
                        leaves_str = ", ".join(
                            f"{k.split('/')[-1]}>={v:.2f}" for k, v in r.target_leaves.items()
                        )
                        lines.append(f"  • {r.concept_name} ({r.rule_id}): [{leaves_str}]")
                if before_evals and after_evals:
                    lines.append("\nQueued Scene Replay Check:")
                    for bef, aft in zip(before_evals, after_evals):
                        bef_status = "SETTLED (GEAR 1)" if bef["settled"] else "STILL_COLLIDING (GEAR 3)"
                        aft_status = "SETTLED (GEAR 1)" if aft["settled"] else "STILL_COLLIDING (GEAR 3)"
                        lines.append(
                            f"  • Scene #{bef['scene_index']}: "
                            f"BEFORE = {bef_status} (Clarity {bef['clarity']:.2f}, matched={bef['matched_concept']}) "
                            f"-> AFTER = {aft_status} (Clarity {aft['clarity']:.2f}, Confidence {aft['confidence']:.2f}, matched={aft['matched_concept']})"
                        )

                summary = "\n".join(lines)
                self.last_rem_sleep_receipt = summary
                if auto_triggered:
                    if self.chat_messages:
                        self.chat_messages[-1]["auto_rem_sleep_receipt"] = summary
                    self._sync_active_thread_record(persist=True)
                else:
                    self.chat_messages.append(
                        {
                            "id": f"rem-{int(time.time())}",
                            "sender": "system",
                            "role_label": "Nighttime REM Sleep — Learning Receipt",
                            "text": summary,
                        }
                    )
                    self._sync_active_thread_record(persist=True)
            finally:
                self.rem_sleep_in_progress = False
        return self.get_state_snapshot()


# ============================================================================
# ZERO-FRUITCAKE ANTIGRAVITY-STYLE SINGLE-PAGE UI
# ============================================================================

from .sanctuary_club_ui import SINGLE_PAGE_HTML


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
            if self.path == "/api/system/update_status":
                self._send_json(
                    {
                        "release_manifest": runtime.release_manifest,
                        "last_sync_status": runtime.last_sync_status,
                    }
                )
                return
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            raw_bytes = self.rfile.read(length) if length > 0 else b"{}"
            try:
                data = json.loads(raw_bytes.decode("utf-8"))
                if self.path == "/api/view":
                    snap = runtime.switch_workspace_view(str(data.get("view_mode", "chess_arena")))
                    self._send_json(snap)
                    return
                if self.path == "/api/threads/create":
                    snap = runtime.create_chat_thread(str(data.get("title", "")))
                    self._send_json(snap)
                    return
                if self.path == "/api/threads/switch":
                    snap = runtime.switch_chat_thread(str(data.get("thread_id", "")))
                    self._send_json(snap)
                    return
                if self.path == "/api/threads/rename":
                    snap = runtime.rename_chat_thread(
                        str(data.get("thread_id", "")),
                        str(data.get("title", "")),
                    )
                    self._send_json(snap)
                    return
                if self.path == "/api/threads/delete":
                    snap = runtime.delete_chat_thread(str(data.get("thread_id", "")))
                    self._send_json(snap)
                    return
                if self.path == "/api/canvas/update":
                    snap = runtime.update_script_canvas(
                        str(data.get("script_text", "")),
                        append=bool(data.get("append", False)),
                    )
                    self._send_json(snap)
                    return
                if self.path == "/api/sleep/auto":
                    snap = runtime.set_auto_rem_sleep(
                        bool(data.get("enabled", True)),
                        threshold=int(data["threshold"]) if data.get("threshold") is not None else None,
                    )
                    self._send_json(snap)
                    return
                if self.path == "/api/microwave/step":
                    snap = runtime.handle_microwave_step(
                        num_sentences=int(data.get("num_sentences", 3)),
                    )
                    self._send_json(snap)
                    return
                if self.path == "/api/system/sync":
                    snap = runtime.handle_check_github_sync(
                        remote_manifest=data.get("remote_manifest"),
                        apply_sync=bool(data.get("apply_sync", False)),
                    )
                    self._send_json(snap)
                    return
                if self.path == "/api/chess/move":
                    snap = runtime.handle_chess_move(str(data.get("move", "")))
                    self._send_json(snap)
                    return
                if self.path == "/api/chess/undo":
                    snap = runtime.handle_chess_undo()
                    self._send_json(snap)
                    return
                if self.path == "/api/chess/ai_move":
                    snap = runtime.handle_chess_ai_move()
                    self._send_json(snap)
                    return
                if self.path == "/api/chess/fen":
                    snap = runtime.handle_set_fen(str(data.get("fen", "start")))
                    self._send_json(snap)
                    return
                if self.path == "/api/chat":
                    anchor_val = data.get("anchor_to_board")
                    snap = runtime.handle_chat_message(
                        str(data.get("text", "")),
                        room=str(data.get("room", "chess")),
                        sanctuary_mode=str(data.get("sanctuary_mode", "tilt_reset")),
                        anchor_to_board=bool(anchor_val) if anchor_val is not None else None,
                        num_sentences=int(data.get("num_sentences", 1)),
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
                    snap = runtime.switch_brain_lineage(
                        str(data.get("lineage", "schooled")),
                        patreon_passphrase=(
                            str(data["patreon_passphrase"])
                            if data.get("patreon_passphrase") is not None
                            else None
                        ),
                    )
                    self._send_json(snap)
                    return
                if self.path == "/api/patreon/unlock":
                    snap = runtime.unlock_patreon_bundle(
                        str(data.get("passphrase", "")),
                        enc_path=(
                            str(data["enc_path"])
                            if data.get("enc_path") is not None
                            else None
                        ),
                    )
                    self._send_json(snap)
                    return
                if self.path == "/api/crucible/brace":
                    snap = runtime.handle_context_brace(str(data.get("brace_key", "story_telling")))
                    self._send_json(snap)
                    return
                if self.path == "/api/crucible/run":
                    snap = runtime.handle_crucible_dilemma(
                        str(data.get("stage_id", "poetry_story")),
                        str(data.get("wave_text", "")),
                        str(data.get("structure_text", "")),
                        art_rgb=data.get("art_rgb"),
                        harvest_for_sleep=bool(data.get("harvest_for_sleep", True)),
                    )
                    self._send_json(snap)
                    return
                if self.path == "/api/sparring/run":
                    snap = runtime.handle_entrainment_sparring(
                        polarity_mode=str(data.get("polarity_mode", "feminine_to_masculine")),
                        seed_theme=str(data.get("seed_theme", "")),
                        num_sentences=int(data.get("num_sentences", 3)),
                        harvest_for_sleep=bool(data.get("harvest_for_sleep", True)),
                    )
                    self._send_json(snap)
                    return
                self._send_json({"error": f"Unknown endpoint {self.path}"}, HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    return ThreadingHTTPServer((host, port), ClubRequestHandler)


def launch_desktop_window(url: str, *, profile_dir: Path | None = None) -> subprocess.Popen[Any] | None:
    """Open `url` in an isolated native Edge app window on Windows, returning the process handle."""
    if sys.platform.startswith("win"):
        edge_candidates = [
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        ]
        from lmft_architecture.workspace_paths import resolve_training_root

        user_data_dir = profile_dir or (resolve_training_root() / "outputs" / ".desktop_window_profile")
        user_data_dir.mkdir(parents=True, exist_ok=True)
        for exe in edge_candidates:
            if Path(exe).exists():
                try:
                    return subprocess.Popen(
                        [
                            exe,
                            f"--app={url}",
                            f"--user-data-dir={user_data_dir}",
                            "--window-size=1320,840",
                            "--no-first-run",
                            "--no-default-browser-check",
                        ]
                    )
                except Exception:
                    pass
    webbrowser.open(url)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the LMFT Studio desktop app.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8420, help="Bind port (default: 8420)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the desktop window")
    parser.add_argument("--no-gpu", action="store_true", help="Run in fast Gear-1 Right-Brain mode without loading Qwen3-4B")
    parser.add_argument("--single-window", action="store_true", help="Exit the server automatically when the desktop window is closed")
    args = parser.parse_args(argv)

    runtime = SanctuaryClubRuntime(lazy_load_gpu=not args.no_gpu)
    try:
        server = create_club_http_server(runtime, host=args.host, port=args.port)
    except OSError:
        server = create_club_http_server(runtime, host=args.host, port=0)
    actual_port = server.server_port
    url = f"http://{args.host}:{actual_port}"
    if sys.stdout is not None:
        print(f"LMFT Studio running at {url}")
    window_proc: subprocess.Popen[Any] | None = None
    if not args.no_browser:
        window_proc = launch_desktop_window(url)
    if (args.single_window or sys.executable.lower().endswith("pythonw.exe")) and window_proc is not None:
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        try:
            window_proc.wait()
        finally:
            server.shutdown()
            server.server_close()
        return 0
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        if sys.stdout is not None:
            print("\nShutting down LMFT Studio...")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
