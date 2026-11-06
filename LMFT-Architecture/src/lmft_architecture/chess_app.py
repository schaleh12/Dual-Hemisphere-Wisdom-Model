"""Standalone App #1: `LMFT-Chess.exe` — Chess Wisdom & Self-Kibitzer App (`chess_app.py`).

Plain-English Purpose:
Dedicated standalone desktop application for `Qwen/Qwen3-4B` + `python-chess` + the 40-node
Right-Brain Fuzzy Tree, completely isolated from `Impish Bloodmoon 12B`.

Enforces:
1. **Dedicated State Folder (`LMFT-Training/outputs/chess_kibitzer_state/`)**:
   - Reads and writes learned chess wisdom rules, 3-branch archives, contemplation ledger,
     and threads in `outputs/chess_kibitzer_state/` while keeping the `Cycle 0` Virgin
     Baseline strictly read-only.
2. **Simultaneous Integration in an Objective Harmony (`Pretext -> Subtext -> Context`)**:
   - **Pretext**: Prior expectation in the sliding conversation history
     (`format_sliding_conversation_history`), continuously accumulating both human dialogue
     and spontaneous self-kibitzing notes across moves.
   - **Subtext**: Live Pass-1 `Fact` from `BoardAnchoredBridgeSensors` (`python-chess`
     6-Piece/6-Color posture + MiniLM bridges) and the 52-float `SubliminalChordVector`
     spliced at `<|subliminal_gap|>`.
   - **Context**: Grounded continuation (`verify_or_ground_chess_moves_in_reply`),
     spontaneously appended to the live kibitz stream on human moves and AI self-play moves.
3. **End-of-Game REM Sleep**:
   - Whenever board reality (`Fact`) collides with the verbal plan (`Theory` unsettled or
     Shadow Pathology warning), logs the surprise to the Contemplation Ledger and
     consolidates new chess wisdom rules during REM Sleep.
"""

from __future__ import annotations

import argparse
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
    BitboardIntuitionSleeve,
    CandidateMoveEvaluation,
    ChessBoardPosture,
    ObjectiveCalibratedSparringEngine,
    _static_centipawns_for_color,
    build_human_move_coaching_report,
    build_natural_chess_scenario_prompt,
    compute_candidate_move_fuzzy_coordinates,
    compute_estimated_game_elo,
    compute_fide_elo_update,
    compute_move_accuracy_percent,
    evaluate_candidate_moves_with_right_brain,
    evaluate_chess_board_posture,
    format_move_history_pgn_snippet,
    inspect_chess_moves_in_reply,
    merge_board_fact_with_chat_sensors,
    verify_or_ground_chess_moves_in_reply,
)
from .chess_ui import CHESS_SINGLE_PAGE_HTML
from .chord_memory import ChordMemoryBank, compile_felt_chord
from .contemplation_ledger import (
    ContemplationLedger,
    load_contemplation_ledger,
    save_contemplation_ledger,
)
from .github_auto_sync import (
    APP_RELEASE_VERSION,
    build_release_manifest,
    fetch_remote_github_manifest,
    verify_or_sync_against_manifest,
)
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
    SettledDiary,
    ThreeBridgeSensors,
    compile_right_brain_vibe,
)
from .runtime_store import atomic_json
from .sleep_consolidation import (
    UnresolvedSleepScene,
    _anchor_target_leaves_from_cluster,
    _average_cluster_coordinates,
    _average_contrastive_reference_coordinates,
    _cluster_key_for_unresolved,
    _derive_concept_tags,
    normalize_contrastive_target_leaves_and_thresholds,
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

SINGLE_PAGE_HTML = CHESS_SINGLE_PAGE_HTML

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
        "fen": "4rrk1/pppb1ppp/2n5/3p4/3P1q2/2PB1N2/PP3PPP/R2Q1K1R w - - 2 14",
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


def _canonical_cluster_concept_name(concept_name: str) -> str:
    """Strip `_s<scene>` suffixes from auto-generated `hazard_*_cluster_sN` / `coached_*_cluster_sN` rules."""
    if concept_name.startswith(("hazard_", "coached_")) and "_cluster_s" in concept_name:
        head, tail = concept_name.rsplit("_cluster_s", 1)
        if tail.isdigit():
            return f"{head}_cluster"
    return concept_name


def _entrain_existing_cluster_rule(
    existing: FuzzyConceptRule,
    incoming: FuzzyConceptRule,
    canonical_name: str,
) -> FuzzyConceptRule:
    """Entrain an existing cluster rule with new REM sleep observations (`0.7 * old + 0.3 * new`)."""
    blended_leaves: dict[str, float] = {}
    for k, v in incoming.target_leaves.items():
        if k in existing.target_leaves:
            blended_leaves[k] = round(0.7 * float(existing.target_leaves[k]) + 0.3 * float(v), 4)
        else:
            blended_leaves[k] = round(float(v), 4)
    new_low = round(0.7 * float(existing.low) + 0.3 * float(incoming.low), 4)
    new_high = round(max(new_low + 0.02, 0.7 * float(existing.high) + 0.3 * float(incoming.high)), 4)
    clean_rule_id = (
        existing.rule_id
        if "_cluster_s" not in existing.concept_name
        else f"rule-sleep-{canonical_name[:32]}"
    )
    return FuzzyConceptRule(
        rule_id=clean_rule_id,
        concept_name=canonical_name,
        description=incoming.description,
        target_leaves=blended_leaves,
        low=new_low,
        high=new_high,
    )


def _ensure_foundational_rules(rules: tuple[FuzzyConceptRule, ...]) -> tuple[FuzzyConceptRule, ...]:
    foundational_by_id = {r.rule_id: r for r in VIRGIN_FOUNDATIONAL_RULES}
    foundational_by_name = {r.concept_name: r for r in VIRGIN_FOUNDATIONAL_RULES}
    merged: list[FuzzyConceptRule] = list(VIRGIN_FOUNDATIONAL_RULES)
    seen_by_canonical_name: dict[str, int] = {}
    seen_learned_sigs: dict[tuple[str, tuple[tuple[str, float], ...]], int] = {}
    for r in rules:
        if r.rule_id in foundational_by_id or r.concept_name in foundational_by_name:
            continue
        if r.concept_name.startswith("hazard_") and (
            r.concept_name.startswith("hazard_high_wastelessness_")
            or "[!! BRILLIANCY" in r.description
            or "[! GREAT_MOVE" in r.description
        ):
            continue
        if r.concept_name.startswith("hazard_"):
            # Strip inverted virtue, low-needlessness, or low-shadow leaves that contradict hazard polarity
            shadow_ids = {l.leaf_id for l in PHILOSOPHICAL_SHADOW_LEAVES}
            cleaned_haz_leaves = {
                k: float(v)
                for k, v in r.target_leaves.items()
                if not (
                    (k == "cultural/freedom/needlessness" and float(v) <= 0.30)
                    or (k in shadow_ids and float(v) < 0.50)
                    or (k in ("philosophical/being/existence", "cultural/freedom/self-sufficiency", "philosophical/structure/order") and float(v) >= 0.58)
                )
            }
            if len(cleaned_haz_leaves) < 2:
                continue
            if cleaned_haz_leaves != dict(r.target_leaves):
                r = FuzzyConceptRule(
                    rule_id=r.rule_id,
                    concept_name=r.concept_name,
                    description=r.description,
                    target_leaves=cleaned_haz_leaves,
                    low=r.low,
                    high=r.high,
                )
        elif r.concept_name.startswith("coached_"):
            shadow_ids = {l.leaf_id for l in PHILOSOPHICAL_SHADOW_LEAVES}
            virtue_ids = {
                "philosophical/being/existence",
                "philosophical/structure/order",
                "philosophical/persistence/continuity",
                "philosophical/persistence/discreteness",
                "cultural/efficiency/wastelessness",
                "cultural/freedom/self-sufficiency",
            }
            cleaned_coached_leaves = {
                k: float(v)
                for k, v in r.target_leaves.items()
                if k not in shadow_ids
                and not (k == "philosophical/structure/chaos" and float(v) >= 0.48)
                and not (k == "cultural/freedom/needlessness" and float(v) > 0.20)
                and not (k in virtue_ids and float(v) < 0.55)
            }
            min_req = 2 if len(r.target_leaves) >= 2 else 1
            if len(cleaned_coached_leaves) < min_req:
                continue
            if cleaned_coached_leaves != dict(r.target_leaves):
                r = FuzzyConceptRule(
                    rule_id=r.rule_id,
                    concept_name=r.concept_name,
                    description=r.description,
                    target_leaves=cleaned_coached_leaves,
                    low=r.low,
                    high=r.high,
                )
        if "Played " in r.description and " instead of best move " in r.description:
            after_played = r.description.split("Played ", 1)[1].split(" ", 1)[0].strip()
            after_best = r.description.split(" instead of best move ", 1)[1].split(" ", 1)[0].rstrip(".").strip()
            if after_played and after_played == after_best:
                continue
        canonical_name = _canonical_cluster_concept_name(r.concept_name)
        if canonical_name != r.concept_name:
            r = FuzzyConceptRule(
                rule_id=f"rule-sleep-{canonical_name[:32]}",
                concept_name=canonical_name,
                description=r.description,
                target_leaves=dict(r.target_leaves),
                low=r.low,
                high=r.high,
            )
        if canonical_name in seen_by_canonical_name:
            idx = seen_by_canonical_name[canonical_name]
            merged[idx] = _entrain_existing_cluster_rule(merged[idx], r, canonical_name)
            continue
        if canonical_name.startswith(("hazard_", "coached_")) and canonical_name.endswith("_cluster"):
            prefix = "hazard" if canonical_name.startswith("hazard_") else "coached"
            sig = (prefix, tuple(sorted((k, round(float(v), 4)) for k, v in r.target_leaves.items())))
            if sig in seen_learned_sigs:
                idx = seen_learned_sigs[sig]
                merged[idx] = _entrain_existing_cluster_rule(merged[idx], r, merged[idx].concept_name)
                continue
            seen_learned_sigs[sig] = len(merged)
        seen_by_canonical_name[canonical_name] = len(merged)
        merged.append(r)
    return tuple(merged)


def _merge_rulesets(
    primary: tuple[FuzzyConceptRule, ...],
    seed: tuple[FuzzyConceptRule, ...],
) -> tuple[FuzzyConceptRule, ...]:
    """Non-destructively merge `seed` rules into `primary` rules without losing local rules."""
    base = list(_ensure_foundational_rules(primary))
    seen_ids = {r.rule_id for r in base}
    seen_names = {r.concept_name for r in base}
    for r in seed:
        if r.rule_id in seen_ids or r.concept_name in seen_names:
            continue
        base.append(r)
        seen_ids.add(r.rule_id)
        seen_names.add(r.concept_name)
    return _ensure_foundational_rules(tuple(base))


class ChessKibitzerRuntime:
    """Thread-safe controller for `LMFT-Chess.exe` (Qwen3-4B Chess Wisdom & Self-Kibitzer App)."""

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
        from lmft_architecture.workspace_paths import resolve_arch_root, resolve_training_root

        self.repo_root = resolve_arch_root(repo_root)
        self.training_root = resolve_training_root(repo_root)
        self.state_dir = (
            Path(state_dir).resolve()
            if state_dir is not None
            else (self.training_root / "outputs" / "chess_kibitzer_state")
        )
        self.teacher = teacher
        self._lazy_load_gpu = bool(lazy_load_gpu)
        self.gpu_lock = threading.Lock()
        self.rem_sleep_in_progress = False
        self.left_brain_loading = False
        self.left_brain_status = "LAZY_STANDBY"
        self.active_lineage = "schooled"
        self.active_checkpoint_label = "Cycle 2 (models/left_brain_lora_cycle2)"
        self.sleep_cycle_index = 2

        self.birth_state = birth_state or create_birth_state(
            "lmft-chess-kibitzer",
            birth_event_id="chess-birth-0",
            random_seed=42,
        )

        self.rules = tuple(rules) if rules is not None else self._load_default_rules()
        self.learned_leaves = (
            tuple(learned_leaves)
            if learned_leaves is not None
            else self._load_default_learned_leaves()
        )
        self.sensors = sensors if sensors is not None else self._load_default_sensors()

        self._qwen_left_brain: SubliminalGapLeftBrain | None = left_brain
        self.left_brain: SubliminalGapLeftBrain | None = left_brain
        if self.left_brain is not None:
            self.left_brain_status = "READY"
            if self.left_brain.callosum_metadata is not None:
                self.sleep_cycle_index = int(self.left_brain.callosum_metadata.sleep_cycle_index)

        self.board = chess.Board()
        self.move_history_san: list[str] = []
        self.last_board_reflex: dict[str, Any] | None = None
        self.board_posture: ChessBoardPosture = evaluate_chess_board_posture(self.board)
        self.active_room: str = "chess"
        self.active_workspace_view: str = "chess_arena"
        self.sanctuary_mode: str = "tilt_reset"
        self.scene_counter: int = 0
        self.spontaneous_kibitz_enabled: bool = True

        self.memory: ContinuousMemoryState | None = None
        self.diary = SettledDiary()
        self.ledger = self._load_initial_ledger()
        self.chord_memory = ChordMemoryBank()
        self.archives = self._load_initial_archives()
        self.bitboard_sleeve = self._load_initial_bitboard_sleeve()

        self.sleep_queue: list[UnresolvedSleepScene] = []
        self._hazard_scene_indices: set[int] = set()
        self.savored_turn_ids: list[int] = self._load_initial_savored_ids()
        self.chat_messages: list[dict[str, Any]] = []
        self._turn_records_by_id: dict[int, dict[str, Any]] = {}

        self.auto_rem_sleep: bool = True
        self.auto_sleep_threshold: int = 3
        self.microwave_step_counter: int = 0
        self.last_microwave_status: dict[str, Any] | None = None
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
            t = threading.Thread(target=self._background_warm_left_brain, daemon=True)
            t.start()

    def _is_default_live_state_dir(self) -> bool:
        try:
            return self.state_dir.resolve() == (self.training_root / "outputs" / "chess_kibitzer_state").resolve()
        except Exception:
            return False

    def _load_default_rules(self) -> tuple[FuzzyConceptRule, ...]:
        user_ruleset = self.state_dir / "club_ruleset.json"
        is_live = self._is_default_live_state_dir()
        seed_rels = (
            (
                "outputs/club_user_state/club_ruleset.json",
                "outputs/rem_sleep_consolidated_ruleset.json",
                "outputs/sleep_consolidated_ruleset.json",
                "outputs/continuous_tiered_ruleset.json",
            )
            if is_live
            else (
                "outputs/club_user_state/club_ruleset.json",
                "outputs/continuous_tiered_ruleset.json",
            )
        )
        seed_candidates = [self.training_root / rel for rel in seed_rels]
        user_loaded: tuple[FuzzyConceptRule, ...] = ()
        if user_ruleset.exists():
            try:
                user_loaded = load_ruleset(user_ruleset)
            except Exception:
                user_loaded = ()
        if user_loaded:
            if is_live:
                for seed_p in seed_candidates[1:3]:
                    if seed_p.exists():
                        try:
                            user_loaded = _merge_rulesets(user_loaded, load_ruleset(seed_p))
                        except Exception:
                            pass
            return _ensure_foundational_rules(user_loaded)
        for p in seed_candidates:
            if p.exists():
                try:
                    loaded = load_ruleset(p)
                    if loaded:
                        return _ensure_foundational_rules(loaded)
                except Exception:
                    pass
        return VIRGIN_FOUNDATIONAL_RULES

    def _load_default_learned_leaves(self) -> tuple[LearnedLeaf, ...]:
        base: list[LearnedLeaf] = []
        user_ruleset = self.state_dir / "club_ruleset.json"
        candidates = [user_ruleset] + [
            self.training_root / rel
            for rel in (
                "outputs/club_user_state/club_ruleset.json",
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
        for fallback_rel in ("outputs/club_user_state/branch_archives", "outputs/branch_archives"):
            default_arch_dir = self.training_root / fallback_rel
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

    def _load_initial_bitboard_sleeve(self) -> BitboardIntuitionSleeve:
        user_sleeve = self.state_dir / "bitboard_sleeve_weights.json"
        if user_sleeve.exists():
            return BitboardIntuitionSleeve.load(user_sleeve)
        seed_sleeve = self.training_root / "outputs" / "bitboard_sleeve_weights.json"
        if self._is_default_live_state_dir() and seed_sleeve.exists():
            return BitboardIntuitionSleeve.load(seed_sleeve)
        return BitboardIntuitionSleeve()

    def _load_initial_savored_ids(self) -> list[int]:
        self.sparring_engine = ObjectiveCalibratedSparringEngine(
            use_stockfish_uci=self._lazy_load_gpu
        )
        self.lmft_elo: float = 1000.0
        self.opponent_target_elo: float = 1000.0
        self.elo_wins: int = 0
        self.elo_losses: int = 0
        self.elo_draws: int = 0
        self.games_completed_count: int = 0
        self.lmft_playing_white: bool = True
        self._current_game_adjudicated: bool = False
        self.last_elo_receipt: dict[str, Any] | None = None

        self.player_elo: float = 1000.0
        self.player_wins: int = 0
        self.player_losses: int = 0
        self.player_draws: int = 0
        self.player_games_completed: int = 0
        self.player_move_accuracies: list[float] = []
        self.last_player_coaching: dict[str, Any] | None = None
        self.last_player_elo_receipt: dict[str, Any] | None = None

        meta_path = self.state_dir / "club_meta.json"
        if meta_path.exists():
            try:
                raw = json.loads(meta_path.read_text(encoding="utf-8"))
                self.scene_counter = max(self.scene_counter, int(raw.get("scene_counter", 0)))
                self.lmft_elo = float(raw.get("lmft_elo", 1000.0))
                self.opponent_target_elo = float(raw.get("opponent_target_elo", self.lmft_elo))
                self.elo_wins = int(raw.get("elo_wins", 0))
                self.elo_losses = int(raw.get("elo_losses", 0))
                self.elo_draws = int(raw.get("elo_draws", 0))
                self.games_completed_count = int(raw.get("games_completed_count", 0))
                self.lmft_playing_white = bool(raw.get("lmft_playing_white", True))
                self.player_elo = float(raw.get("player_elo", 1000.0))
                self.player_wins = int(raw.get("player_wins", 0))
                self.player_losses = int(raw.get("player_losses", 0))
                self.player_draws = int(raw.get("player_draws", 0))
                self.player_games_completed = int(raw.get("player_games_completed", 0))
                return [int(x) for x in raw.get("savored_turn_ids", [])]
            except Exception:
                pass
        return []

    def _load_default_sensors(self) -> ThreeBridgeSensors:
        bridges_dir = self.training_root / "models" / "right_brain_bridges"
        if bridges_dir.exists():
            try:
                suite = NeuralThreeBridgeSuite.load_all(bridges_dir)
                return suite.as_three_bridge_sensors()
            except Exception:
                pass
        return _default_fallback_sensors()

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
        if not self.active_thread_id:
            self.active_thread_id = "thread-1"
        cur = self.threads.get(self.active_thread_id)
        if cur is None:
            cur = {
                "thread_id": self.active_thread_id,
                "title": "Chess Session 1",
                "created_at": time.time(),
            }
            self.threads[self.active_thread_id] = cur
        cur["chat_messages"] = list(self.chat_messages)
        cur["memory_state"] = self._serialize_memory_state(self.memory)
        if persist and self.active_lineage != "virgin":
            self.state_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(
                self.state_dir / "threads.json",
                {
                    "schema_version": 1,
                    "active_thread_id": self.active_thread_id,
                    "threads": list(self.threads.values()),
                },
            )

    def _load_threads_for_active_lineage(self) -> None:
        self.threads = {}
        if self.active_lineage != "virgin":
            threads_path = self.state_dir / "threads.json"
            if threads_path.exists():
                try:
                    raw = json.loads(threads_path.read_text(encoding="utf-8"))
                    for item in raw.get("threads", []):
                        tid = str(item.get("thread_id", "")).strip()
                        if tid:
                            self.threads[tid] = {
                                "thread_id": tid,
                                "title": str(item.get("title", tid)),
                                "created_at": float(item.get("created_at", time.time())),
                                "chat_messages": list(item.get("chat_messages", [])),
                                "memory_state": item.get("memory_state"),
                            }
                    active_tid = str(raw.get("active_thread_id", "")).strip()
                    if active_tid in self.threads:
                        self.active_thread_id = active_tid
                except Exception:
                    self.threads = {}
        if not self.threads:
            self.active_thread_id = "thread-1"
            self.threads["thread-1"] = {
                "thread_id": "thread-1",
                "title": "Chess Study & Kibitz 1",
                "created_at": time.time(),
                "chat_messages": [],
                "memory_state": None,
            }
        elif self.active_thread_id not in self.threads:
            self.active_thread_id = next(iter(self.threads))

        active_rec = self.threads[self.active_thread_id]
        self.chat_messages = list(active_rec.get("chat_messages", []))
        self.memory = self._deserialize_memory_state(active_rec.get("memory_state"))

    def create_chat_thread(self, title: str = "") -> dict[str, Any]:
        self._sync_active_thread_record(persist=True)
        next_idx = len(self.threads) + 1
        new_tid = f"thread-{next_idx}-{int(time.time() * 1000) % 100000}"
        clean_title = title.strip() or f"Chess Session #{next_idx}"
        self.threads[new_tid] = {
            "thread_id": new_tid,
            "title": clean_title,
            "created_at": time.time(),
            "chat_messages": [],
            "memory_state": None,
        }
        self.active_thread_id = new_tid
        self.chat_messages = []
        self.memory = None
        self._sync_active_thread_record(persist=True)
        return self.get_state_snapshot()

    def switch_chat_thread(self, thread_id: str) -> dict[str, Any]:
        tid = thread_id.strip()
        if tid not in self.threads:
            raise ValueError(f"Unknown thread_id: {thread_id!r}")
        self._sync_active_thread_record(persist=True)
        self.active_thread_id = tid
        rec = self.threads[tid]
        self.chat_messages = list(rec.get("chat_messages", []))
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
            self.chat_messages = []
            self.memory = None
            self._sync_active_thread_record(persist=True)
            return self.get_state_snapshot()
        del self.threads[tid]
        if self.active_thread_id == tid:
            self.active_thread_id = next(iter(self.threads))
            rec = self.threads[self.active_thread_id]
            self.chat_messages = list(rec.get("chat_messages", []))
            self.memory = self._deserialize_memory_state(rec.get("memory_state"))
        self._sync_active_thread_record(persist=True)
        return self.get_state_snapshot()

    def _extract_clean_conversation_history(self, *, max_turns: int = 8) -> tuple[tuple[str, str], ...]:
        """Extract continuous `(pretext_prompt, clean_reply)` pairs across both user dialogue and self-kibitzing."""
        pairs: list[tuple[str, str]] = []
        pending_user: str | None = None
        for msg in self.chat_messages:
            sender = msg.get("sender")
            if sender == "user":
                u_txt = str(msg.get("clean_user_text") or msg.get("text", "")).strip()
                if u_txt and not u_txt.startswith("[WAVE —"):
                    pending_user = u_txt
            elif sender == "companion":
                c_txt = str(msg.get("clean_reply") or msg.get("text", "")).strip()
                if " [Board Fact-Check:" in c_txt:
                    c_txt = c_txt.split(" [Board Fact-Check:", 1)[0].strip()
                if "\n[Board Fact-Check:" in c_txt:
                    c_txt = c_txt.split("\n[Board Fact-Check:", 1)[0].strip()
                if pending_user and c_txt:
                    pairs.append((pending_user, c_txt))
                    pending_user = None
                elif c_txt and msg.get("clean_pretext_prompt"):
                    pairs.append((str(msg["clean_pretext_prompt"]).strip(), c_txt))
        if max_turns > 0 and len(pairs) > max_turns:
            return tuple(pairs[-max_turns:])
        return tuple(pairs)

    def _persist_user_state(self) -> None:
        self._sync_active_thread_record(persist=self.active_lineage != "virgin")
        if self.active_lineage == "virgin":
            return
        self.state_dir.mkdir(parents=True, exist_ok=True)
        save_ruleset(
            self.state_dir / "club_ruleset.json",
            tuple(self.rules),
            tuple(self.learned_leaves),
        )
        self.archives.save_all(self.state_dir / "branch_archives")
        save_contemplation_ledger(self.state_dir / "contemplation_ledger.json", self.ledger)
        self.bitboard_sleeve.save(self.state_dir / "bitboard_sleeve_weights.json")
        atomic_json(
            self.state_dir / "club_meta.json",
            {
                "savored_turn_ids": list(self.savored_turn_ids),
                "scene_counter": int(self.scene_counter),
                "lmft_elo": float(self.lmft_elo),
                "opponent_target_elo": float(self.opponent_target_elo),
                "elo_wins": int(self.elo_wins),
                "elo_losses": int(self.elo_losses),
                "elo_draws": int(self.elo_draws),
                "games_completed_count": int(self.games_completed_count),
                "lmft_playing_white": bool(self.lmft_playing_white),
                "player_elo": float(self.player_elo),
                "player_wins": int(self.player_wins),
                "player_losses": int(self.player_losses),
                "player_draws": int(self.player_draws),
                "player_games_completed": int(self.player_games_completed),
            },
        )

    def export_master_release_seed(
        self,
        *,
        target_training_root: str | Path | None = None,
    ) -> dict[str, Any]:
        """Export the evolved ruleset and 3-branch archives into the shared GitHub/Release seed files."""
        target_root = (
            Path(target_training_root).resolve()
            if target_training_root is not None
            else self.training_root
        )
        out_dir = target_root / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        rem_seed_path = out_dir / "rem_sleep_consolidated_ruleset.json"
        sleep_seed_path = out_dir / "sleep_consolidated_ruleset.json"
        sleeve_seed_path = out_dir / "bitboard_sleeve_weights.json"
        save_ruleset(rem_seed_path, tuple(self.rules), tuple(self.learned_leaves))
        self.bitboard_sleeve.save(sleeve_seed_path)
        if target_training_root is not None:
            save_ruleset(sleep_seed_path, tuple(self.rules), tuple(self.learned_leaves))
            self.archives.save_all(out_dir / "branch_archives")
        if target_root.resolve() == self.training_root.resolve():
            self.release_manifest = build_release_manifest(self.repo_root, write_to_disk=True)
        return {
            "exported_rules_count": len(self.rules),
            "exported_learned_leaves_count": len(self.learned_leaves),
            "rem_seed_path": str(rem_seed_path),
            "sleep_seed_path": str(sleep_seed_path),
            "sleeve_seed_path": str(sleeve_seed_path),
        }

    def merge_master_seed_rules(
        self,
        *,
        seed_training_root: str | Path | None = None,
    ) -> dict[str, Any]:
        """Non-destructively merge any new rules from the shared GitHub/Release seed into the active ruleset."""
        if self.active_lineage == "virgin":
            return {"merged_rules_count": 0, "total_rules_count": len(self.rules)}
        seed_root = (
            Path(seed_training_root).resolve()
            if seed_training_root is not None
            else self.training_root
        )
        before_count = len(self.rules)
        merged_rules = tuple(self.rules)
        existing_leaf_ids = {l.leaf_id for l in self.learned_leaves}
        merged_leaves = list(self.learned_leaves)

        for rel in ("outputs/rem_sleep_consolidated_ruleset.json", "outputs/sleep_consolidated_ruleset.json"):
            seed_path = seed_root / rel
            if seed_path.exists():
                try:
                    merged_rules = _merge_rulesets(merged_rules, load_ruleset(seed_path))
                    for lf in load_learned_leaves(seed_path):
                        if lf.leaf_id not in existing_leaf_ids:
                            merged_leaves.append(lf)
                            existing_leaf_ids.add(lf.leaf_id)
                except Exception:
                    pass

        sleeve_seed = seed_root / "outputs" / "bitboard_sleeve_weights.json"
        if sleeve_seed.exists():
            try:
                seed_sleeve = BitboardIntuitionSleeve.load(sleeve_seed)
                if seed_sleeve.steps_distilled > self.bitboard_sleeve.steps_distilled:
                    for lk, sv in seed_sleeve.leaf_weights.items():
                        cur_v = float(self.bitboard_sleeve.leaf_weights.get(lk, sv))
                        self.bitboard_sleeve.leaf_weights[lk] = round(0.5 * cur_v + 0.5 * float(sv), 4)
                    for vk, sv in seed_sleeve.value_weights.items():
                        cur_v = float(self.bitboard_sleeve.value_weights.get(vk, sv))
                        self.bitboard_sleeve.value_weights[vk] = round(0.5 * cur_v + 0.5 * float(sv), 4)
                    for pk, sv in seed_sleeve.policy_weights.items():
                        cur_v = float(self.bitboard_sleeve.policy_weights.get(pk, sv))
                        self.bitboard_sleeve.policy_weights[pk] = round(0.5 * cur_v + 0.5 * float(sv), 4)
                    self.bitboard_sleeve.steps_distilled = max(
                        self.bitboard_sleeve.steps_distilled,
                        seed_sleeve.steps_distilled,
                    )
            except Exception:
                pass

        self.rules = merged_rules
        self.learned_leaves = tuple(merged_leaves)
        self._persist_user_state()
        return {
            "merged_rules_count": max(0, len(self.rules) - before_count),
            "total_rules_count": len(self.rules),
        }

    def switch_brain_lineage(
        self,
        lineage: str,
        *,
        patreon_passphrase: str | None = None,
    ) -> dict[str, Any]:
        """Switch between `'schooled'` (Cycle 2 + user coaching) and `'virgin'` (Cycle 0 read-only snapshot)."""
        _ = patreon_passphrase
        mode = lineage.strip().lower()
        if mode not in ("schooled", "virgin"):
            raise ValueError(
                f"Unknown lineage mode {lineage!r} for LMFT-Chess; expected 'schooled' or 'virgin'."
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
            return self.get_state_snapshot()

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

    def _resolve_checkpoint_dir(self) -> Path | None:
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

    def _load_qwen_locked(self) -> None:
        _ensure_valid_stdio_streams()
        if self.rem_sleep_in_progress:
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
            loaded = SubliminalGapLeftBrain.from_checkpoint(
                ckpt_dir,
                max_clause_1_tokens=20,
                max_post_gap_tokens=64,
            )
            self._qwen_left_brain = loaded
            self.left_brain = loaded
            if loaded.callosum_metadata is not None:
                self.sleep_cycle_index = int(loaded.callosum_metadata.sleep_cycle_index)
            self.left_brain_status = f"READY (Cycle {self.sleep_cycle_index})"
        except Exception as exc:
            self.left_brain_status = f"GEAR_1_FALLBACK ({type(exc).__name__}: {exc})"
        finally:
            self.left_brain_loading = False

    def _background_warm_left_brain(self) -> None:
        with self.gpu_lock:
            self._load_qwen_locked()

    def _synthesize_gear_1_note(
        self,
        turn: ThreeLeggedTurnResult,
        *,
        room: str,
        board_posture: ChessBoardPosture | None = None,
    ) -> str:
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
        skip_left_brain: bool = False,
    ) -> tuple[ThreeLeggedTurnResult, float, dict[str, Any]]:
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

        acquired = (
            False
            if skip_left_brain
            else self.gpu_lock.acquire(
                blocking=not self.rem_sleep_in_progress,
                timeout=0.05 if self.rem_sleep_in_progress else 180.0,
            )
        )
        try:
            if acquired and not self.rem_sleep_in_progress and self._lazy_load_gpu and self._qwen_left_brain is None:
                self._load_qwen_locked()
            active_lb = (
                self._qwen_left_brain
                if (acquired and not self.rem_sleep_in_progress and not skip_left_brain)
                else None
            )
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
            "pretext_subtext_context": {
                "pretext_turns": len(conv_history),
                "dominant_piece": self.board_posture.dominant_piece,
                "dominant_color": self.board_posture.dominant_color,
                "dominant_element": self.board_posture.dominant_element,
                "board_summary": self.board_posture.board_summary,
            },
        }
        self._turn_records_by_id[scene_idx] = {
            "scene_index": scene_idx,
            "scenario_text": scenario_text,
            "turn": turn,
            "telemetry": telemetry,
        }
        return turn, elapsed_ms, telemetry

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
        player_game_acc = (
            round(sum(self.player_move_accuracies) / len(self.player_move_accuracies), 1)
            if self.player_move_accuracies
            else None
        )
        player_est_elo = (
            compute_estimated_game_elo(player_game_acc, reference_elo=self.lmft_elo)
            if player_game_acc is not None
            else round(float(self.player_elo), 1)
        )
        return {
            "app_name": "LMFT-Chess",
            "active_room": self.active_room,
            "active_workspace_view": self.active_workspace_view,
            "active_lineage": self.active_lineage,
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
            "sanctuary_mode": self.sanctuary_mode,
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
            "elo": {
                "lmft_elo": round(float(self.lmft_elo), 1),
                "opponent_target_elo": round(float(self.opponent_target_elo), 1),
                "wins": int(self.elo_wins),
                "losses": int(self.elo_losses),
                "draws": int(self.elo_draws),
                "games_completed": int(self.games_completed_count),
                "lmft_color": "White" if self.lmft_playing_white else "Black",
                "sparring_engine_label": self.sparring_engine.engine_label,
                "last_elo_receipt": self.last_elo_receipt,
                "player_elo": round(float(self.player_elo), 1),
                "player_wins": int(self.player_wins),
                "player_losses": int(self.player_losses),
                "player_draws": int(self.player_draws),
                "player_games_completed": int(self.player_games_completed),
                "player_game_accuracy_pct": player_game_acc,
                "player_estimated_game_elo": player_est_elo,
                "last_player_coaching": self.last_player_coaching,
                "last_player_elo_receipt": self.last_player_elo_receipt,
            },
            "board": {
                "fen": self.board.fen(),
                "turn": "w" if self.board.turn == chess.WHITE else "b",
                "in_check": self.board.is_check(),
                "is_game_over": self.board.is_game_over(),
                "eval_white_cp": _static_centipawns_for_color(self.board, chess.WHITE),
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

    def handle_chess_move(
        self,
        move_uci_or_san: str,
        *,
        pre_deliberation: tuple[str, Any] | None = None,
        is_human_move: bool | None = None,
        skip_spontaneous_left_brain: bool = False,
    ) -> dict[str, Any]:
        """Make a move on the chessboard and spontaneously append to the continuous Pretext -> Subtext -> Context stream."""
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

        human_turn = (pre_deliberation is None) if is_human_move is None else bool(is_human_move)
        board_before = self.board.copy()
        mover_color = self.board.turn
        san_str = self.board.san(move)
        self.board.push(move)
        self.move_history_san.append(san_str)
        mover_posture = evaluate_chess_board_posture(self.board, perspective=mover_color)
        self.board_posture = mover_posture

        human_review: dict[str, Any] | None = None
        human_coaching: dict[str, Any] | None = None
        if human_turn:
            human_review = self.sparring_engine.audit_played_move_for_game_review(
                board_before,
                san_str,
                perspective=mover_color,
                learned_leaves=self.learned_leaves,
                bitboard_sleeve=self.bitboard_sleeve,
            )
            human_coaching = build_human_move_coaching_report(
                human_review,
                rules=self.rules,
                player_elo=self.player_elo,
                reference_elo=self.lmft_elo,
                prior_move_accuracies=self.player_move_accuracies,
            )
            self.player_move_accuracies.append(float(human_coaching["move_accuracy_pct"]))
            self.last_player_coaching = human_coaching

        move_desc = build_natural_chess_scenario_prompt(
            self.board,
            self.move_history_san,
        )
        has_board_warning = (
            bool(mover_posture.board_shadow_warnings)
            or bool(evaluate_chess_board_posture(self.board).board_shadow_warnings)
            or self.board.is_check()
        )
        conv_history = self._extract_clean_conversation_history()
        turn, elapsed_ms, telemetry = self._execute_turn(
            move_desc,
            room="chess",
            force_spoken_reply=has_board_warning and pre_deliberation is None and not skip_spontaneous_left_brain,
            include_conversation_history=True,
            skip_left_brain=(pre_deliberation is not None) or skip_spontaneous_left_brain,
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

        raw_reply = turn.final_conscious_note
        spliced_chord = SubliminalChordVector.from_right_brain_vibe(
            turn.final_vibe,
            learned_leaves=self.learned_leaves,
        )
        if pre_deliberation is not None:
            delib_text, delib_trace = pre_deliberation
            raw_reply = delib_text
            if delib_trace is not None and not telemetry.get("spliced_trace"):
                telemetry["spliced_trace"] = {
                    "sentence_count": 1,
                    "clause_1_prefix": delib_trace.clause_1_prefix,
                    "post_gap_continuation": delib_trace.post_gap_continuation,
                    "comma_scored_live": delib_trace.comma_scored_live,
                    "comma_vibe_posture": delib_trace.comma_vibe_posture,
                    "comma_dissonance_notes": list(delib_trace.comma_dissonance_notes),
                    "spliced_cadence_vector": [round(float(x), 4) for x in delib_trace.spliced_chord.cadence_vector],
                    "spliced_thinker_vector": [round(float(x), 4) for x in delib_trace.spliced_chord.thinker_vector[:8]],
                    "spliced_feeler_vector": [round(float(x), 4) for x in delib_trace.spliced_chord.feeler_vector[:8]],
                }
        elif (
            not raw_reply
            and not skip_spontaneous_left_brain
            and self._qwen_left_brain is not None
            and not self.rem_sleep_in_progress
        ):
            try:
                with self.gpu_lock:
                    spontaneous_text, spontaneous_trace = self._qwen_left_brain.generate_with_subliminal_splice(
                        move_desc,
                        spliced_chord,
                        conversation_history=conv_history,
                    )
                raw_reply = spontaneous_text
                if not telemetry.get("spliced_trace"):
                    telemetry["spliced_trace"] = {
                        "sentence_count": 1,
                        "clause_1_prefix": spontaneous_trace.clause_1_prefix,
                        "post_gap_continuation": spontaneous_trace.post_gap_continuation,
                        "comma_scored_live": spontaneous_trace.comma_scored_live,
                        "comma_vibe_posture": spontaneous_trace.comma_vibe_posture,
                        "comma_dissonance_notes": list(spontaneous_trace.comma_dissonance_notes),
                        "spliced_cadence_vector": [round(float(x), 4) for x in spontaneous_trace.spliced_chord.cadence_vector],
                        "spliced_thinker_vector": [round(float(x), 4) for x in spontaneous_trace.spliced_chord.thinker_vector[:8]],
                        "spliced_feeler_vector": [round(float(x), 4) for x in spontaneous_trace.spliced_chord.feeler_vector[:8]],
                    }
            except Exception:
                raw_reply = None

        if not raw_reply:
            raw_reply = self._synthesize_gear_1_note(
                turn,
                room="chess",
                board_posture=mover_posture,
            )

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
        telemetry["pretext_subtext_context"] = {
            "pretext_turns": len(conv_history),
            "pretext_prompt": move_desc,
            "dominant_piece": mover_posture.dominant_piece,
            "dominant_color": mover_posture.dominant_color,
            "dominant_element": mover_posture.dominant_element,
            "board_summary": mover_posture.board_summary,
            "spliced_thinker_vector": [round(float(x), 4) for x in spliced_chord.thinker_vector[:6]],
            "context_reply": reply_text,
        }
        if human_review is not None:
            telemetry["game_review"] = human_review
        if human_coaching is not None:
            telemetry["player_coaching"] = human_coaching
            if self.last_board_reflex is not None:
                self.last_board_reflex["player_coaching"] = human_coaching

        if self.spontaneous_kibitz_enabled or turn.final_conscious_note or has_board_warning:
            self.chat_messages.append(
                {
                    "id": telemetry["scene_index"],
                    "sender": "companion",
                    "role_label": f"Celestriad 1.0 (after {san_str})",
                    "text": reply_text,
                    "clean_reply": raw_reply,
                    "clean_pretext_prompt": move_desc,
                    "telemetry": telemetry,
                    "coached": False,
                    "savored": False,
                }
            )
            self._sync_active_thread_record(persist=True)

        # End-of-Game REM Sleep Surprise Logging: if board reality collides with Theory or triggers a shadow warning
        if (
            has_board_warning
            and turn.shadow_pathology_warnings
            and not skip_spontaneous_left_brain
            and (human_review is None or bool(human_review.get("is_hazard", False)))
        ):
            self.flag_turn_for_coaching(
                int(telemetry["scene_index"]),
                user_note=f"Board vs. Theory surprise after {san_str} ({mover_posture.board_summary})",
                check_auto_sleep=False,
                is_hazard=True,
            )

        if human_turn and self.board.is_game_over(claim_draw=True):
            self.adjudicate_human_game_outcome(human_color=mover_color)

        self._maybe_trigger_auto_rem_sleep()
        return self.get_state_snapshot()

    def handle_chess_undo(self) -> dict[str, Any]:
        if self.board.move_stack:
            self.board.pop()
            if self.move_history_san:
                self.move_history_san.pop()
        self._current_game_adjudicated = False
        self._current_human_game_adjudicated = False
        self.last_board_reflex = None
        self.board_posture = evaluate_chess_board_posture(self.board)
        return self.get_state_snapshot()

    def handle_chess_ai_move(self) -> dict[str, Any]:
        """Dual-Hemisphere Move-Selection / Kibitzing Hybrid: Right-Brain 40-node scan + Left-Brain `<|subliminal_gap|>` commit."""
        if self.board.is_game_over():
            raise ValueError("The game is already over.")
        all_evals = evaluate_candidate_moves_with_right_brain(
            self.board,
            self.rules,
            learned_leaves=self.learned_leaves,
            limit=max(6, self.board.legal_moves.count()),
            bitboard_sleeve=self.bitboard_sleeve,
        )
        if not all_evals:
            raise ValueError("No legal moves available.")

        top_candidates = all_evals[:6]
        eval_by_san: dict[str, CandidateMoveEvaluation] = {c.san: c for c in all_evals}

        # Opening diversity in early plies among nearly-tied sound moves so self-play explores distinct games
        sound_opening = [
            c
            for c in top_candidates[:4]
            if not c.is_blunder
            and not c.is_repetition
            and not c.shadow_warnings
            and (top_candidates[0].composite_score - c.composite_score) <= 0.35
        ]
        if len(self.move_history_san) < 6 and len(sound_opening) > 1:
            game_idx = int(getattr(self, "games_completed_count", 0))
            right_brain_choice = sound_opening[game_idx % len(sound_opening)]
        else:
            right_brain_choice = top_candidates[0]

        chosen_san = right_brain_choice.san
        selection_mode = "right_brain_equilibrium_choice"
        left_brain_proposed_san: str | None = None
        veto_reason: str | None = None
        vetoed_coords: dict[str, float] | None = None
        safe_choice_coords: dict[str, float] | None = None
        pre_delib_pair: tuple[str, Any] | None = None
        top_san_set = {c.san for c in top_candidates[:4]}

        # Pre-move `<|subliminal_gap|>` deliberation when the 4B Left Brain is active
        if self._qwen_left_brain is not None and not self.rem_sleep_in_progress:
            try:
                cand_list_str = ", ".join(c.san for c in top_candidates[:5])
                pre_prompt = build_natural_chess_scenario_prompt(
                    self.board,
                    self.move_history_san,
                    user_note=f"Choose the soundest move from ({cand_list_str}) and state why.",
                )
                pre_sensors = merge_board_fact_with_chat_sensors(self.board_posture, self.sensors)
                fact_coords, fact_exp, _, _ = pre_sensors.score_text(
                    pre_prompt,
                    learned_leaves=self.learned_leaves,
                )
                pre_vibe, _ = compile_right_brain_vibe(
                    self.birth_state,
                    fact_coords,
                    fact_exp,
                    self.rules,
                    pass_index=1,
                    event_id=f"pre_move_{self.scene_counter + 1}",
                    prior_memory=self.memory,
                    learned_leaves=self.learned_leaves,
                )
                pre_chord = SubliminalChordVector.from_right_brain_vibe(
                    pre_vibe,
                    learned_leaves=self.learned_leaves,
                )
                conv_history = self._extract_clean_conversation_history()
                with self.gpu_lock:
                    delib_text, delib_trace = self._qwen_left_brain.generate_with_subliminal_splice(
                        pre_prompt,
                        pre_chord,
                        conversation_history=conv_history,
                    )
                pre_delib_pair = (delib_text, delib_trace)
                mentioned = inspect_chess_moves_in_reply(
                    delib_text,
                    self.board,
                    move_history_san=self.move_history_san,
                )["legal_moves_mentioned"]
                if mentioned:
                    left_brain_proposed_san = str(mentioned[0])
                    cand_eval = eval_by_san.get(left_brain_proposed_san)
                    if cand_eval is not None:
                        is_suboptimal = (
                            left_brain_proposed_san not in top_san_set
                            or (right_brain_choice.composite_score - cand_eval.composite_score) > 0.40
                        )
                        if (
                            cand_eval.is_blunder
                            or cand_eval.is_repetition
                            or cand_eval.shadow_warnings
                            or is_suboptimal
                        ):
                            selection_mode = "right_brain_blunder_veto_queued_for_sleep"
                            veto_reason = (
                                f"Left-Brain proposed {left_brain_proposed_san} "
                                f"(blunder={cand_eval.is_blunder}, repetition={cand_eval.is_repetition}, "
                                f"shadows={list(cand_eval.shadow_warnings)}, score={cand_eval.composite_score:.2f}); "
                                f"Right-Brain played {right_brain_choice.san} (score={right_brain_choice.composite_score:.2f})."
                            )
                            try:
                                us_color = self.board.turn
                                vetoed_coords, _, _, _, _ = compute_candidate_move_fuzzy_coordinates(
                                    self.board,
                                    self.board.parse_san(left_brain_proposed_san),
                                    perspective=us_color,
                                    learned_leaves=self.learned_leaves,
                                    bitboard_sleeve=self.bitboard_sleeve,
                                )
                                safe_choice_coords, _, _, _, _ = compute_candidate_move_fuzzy_coordinates(
                                    self.board,
                                    self.board.parse_san(right_brain_choice.san),
                                    perspective=us_color,
                                    learned_leaves=self.learned_leaves,
                                    bitboard_sleeve=self.bitboard_sleeve,
                                )
                            except Exception:
                                vetoed_coords = None
                                safe_choice_coords = None
                        else:
                            chosen_san = left_brain_proposed_san
                            selection_mode = "left_brain_kibitz_commit"
            except Exception:
                pass

        ai_mover_color = self.board.turn
        snap = self.handle_chess_move(
            chosen_san,
            pre_deliberation=pre_delib_pair,
            is_human_move=False,
        )
        arbitration_record = {
            "selection_mode": selection_mode,
            "chosen_move_san": chosen_san,
            "right_brain_top_san": right_brain_choice.san,
            "left_brain_proposed_san": left_brain_proposed_san,
            "veto_reason": veto_reason,
            "top_candidates": [c.to_dict() for c in top_candidates[:5]],
        }
        self.last_hybrid_move_arbitration = arbitration_record
        if self.last_board_reflex is not None:
            self.last_board_reflex["hybrid_arbitration"] = arbitration_record
        if (
            veto_reason
            and self.chat_messages
            and (
                vetoed_coords is not None
                and float(vetoed_coords.get("cultural/efficiency/wastelessness", 0.72)) <= 0.35
            )
        ):
            last_scene_id = int(self.chat_messages[-1].get("id", self.scene_counter))
            self.flag_turn_for_coaching(
                last_scene_id,
                user_note=veto_reason,
                check_auto_sleep=True,
                is_hazard=True,
                override_coordinates=vetoed_coords,
                contrastive_reference_coordinates=safe_choice_coords,
                game_review_annotation=veto_reason,
            )
            snap = self.get_state_snapshot()
        if (
            self.player_move_accuracies
            and self.board.is_game_over(claim_draw=True)
            and not getattr(self, "_current_human_game_adjudicated", False)
        ):
            self.adjudicate_human_game_outcome(human_color=not ai_mover_color)
            snap = self.get_state_snapshot()
        snap["last_hybrid_move_arbitration"] = arbitration_record
        return snap

    def handle_set_fen(self, fen_or_preset: str) -> dict[str, Any]:
        key = fen_or_preset.strip()
        fen = PRESET_CHESS_POSITIONS[key]["fen"] if key in PRESET_CHESS_POSITIONS else key
        self.board = chess.Board(fen)
        self.move_history_san.clear()
        self.player_move_accuracies.clear()
        self.last_player_coaching = None
        self._current_game_adjudicated = False
        self._current_human_game_adjudicated = False
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
                "role_label": "Chess Kibitzer",
                "text": reply_text,
                "clean_reply": raw_reply,
                "telemetry": telemetry,
                "coached": False,
                "savored": False,
            }
        )
        if use_board_context and turn.shadow_pathology_warnings:
            self.flag_turn_for_coaching(
                int(telemetry["scene_index"]),
                user_note=f"Chat vs. Board surprise: {clean_text[:80]}",
                check_auto_sleep=False,
            )
        self._sync_active_thread_record(persist=True)
        self._maybe_trigger_auto_rem_sleep()
        return self.get_state_snapshot()

    def flag_turn_for_coaching(
        self,
        scene_index: int,
        user_note: str = "",
        *,
        check_auto_sleep: bool = True,
        is_hazard: bool = False,
        override_coordinates: dict[str, float] | None = None,
        contrastive_reference_coordinates: dict[str, float] | None = None,
        game_review_annotation: str | None = None,
    ) -> dict[str, Any]:
        rec = self._turn_records_by_id.get(int(scene_index))
        if rec is None:
            raise ValueError(f"Unknown turn scene_index: {scene_index}")
        turn: ThreeLeggedTurnResult = rec["turn"]
        vibe = turn.final_vibe
        note_suffix = f" [Coach note: {user_note.strip()}]" if user_note.strip() else ""
        scene_coords = dict(override_coordinates) if override_coordinates is not None else dict(vibe.coordinates)
        if is_hazard:
            self._hazard_scene_indices.add(int(scene_index))
            if override_coordinates is None:
                scene_coords["cultural/efficiency/wastelessness"] = min(
                    float(scene_coords.get("cultural/efficiency/wastelessness", 0.20)),
                    0.20,
                )
        else:
            self._hazard_scene_indices.discard(int(scene_index))
        sleep_scene = UnresolvedSleepScene(
            scene_index=int(scene_index),
            scenario_text=f"{rec['scenario_text']}{note_suffix}",
            external_coordinates=scene_coords,
            felt_chord_notes=dict(vibe.felt_chord.complex_notes) if vibe.felt_chord is not None else {},
            valence=vibe.valence,
            salience=vibe.salience,
            arousal=vibe.arousal,
            left_brain_hunch=turn.final_conscious_note or rec["scenario_text"],
            social_drift_detected=False,
            active_concept_collision=None if override_coordinates is not None else vibe.active_concept,
            is_hazard=bool(is_hazard),
            contrastive_reference_coordinates=(
                dict(contrastive_reference_coordinates)
                if contrastive_reference_coordinates is not None
                else None
            ),
            game_review_annotation=game_review_annotation or (user_note.strip() if user_note.strip() else None),
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
        self.auto_rem_sleep = bool(enabled)
        if threshold is not None:
            self.auto_sleep_threshold = max(1, int(threshold))
        self._maybe_trigger_auto_rem_sleep()
        return self.get_state_snapshot()

    def _maybe_trigger_auto_rem_sleep(self) -> bool:
        if not self.auto_rem_sleep or self.active_lineage == "virgin":
            return False
        if len(self.sleep_queue) >= self.auto_sleep_threshold:
            self.handle_run_rem_sleep(auto_triggered=True)
            return True
        return False

    def _adjudicate_and_record_game_outcome(self) -> dict[str, Any]:
        """Score a finished sparring game objectively, update FIDE Elo, match opponent target Elo, and rotate colors."""
        lmft_color = chess.WHITE if self.lmft_playing_white else chess.BLACK
        outcome = self.board.outcome(claim_draw=True)
        if outcome is not None:
            if outcome.winner is None:
                lmft_score = 0.5
                result_label = "DRAW"
            elif outcome.winner == lmft_color:
                lmft_score = 1.0
                result_label = "WIN"
            else:
                lmft_score = 0.0
                result_label = "LOSS"
        else:
            cp = self.sparring_engine.evaluate_position_centipawns(
                self.board,
                perspective=lmft_color,
            )
            if cp >= 300.0:
                lmft_score = 1.0
                result_label = "WIN (Endgame Material Adjudication)"
            elif cp <= -300.0:
                lmft_score = 0.0
                result_label = "LOSS (Endgame Material Adjudication)"
            else:
                lmft_score = 0.5
                result_label = "DRAW (Endgame Cap)"

        prev_elo = self.lmft_elo
        opp_elo = self.opponent_target_elo
        new_elo, delta = compute_fide_elo_update(prev_elo, opp_elo, lmft_score)
        self.lmft_elo = new_elo
        self.opponent_target_elo = new_elo
        if lmft_score > 0.75:
            self.elo_wins += 1
        elif lmft_score < 0.25:
            self.elo_losses += 1
        else:
            self.elo_draws += 1
        self.games_completed_count += 1
        played_color = "White" if self.lmft_playing_white else "Black"
        self.lmft_playing_white = not self.lmft_playing_white
        next_color = "White" if self.lmft_playing_white else "Black"

        sign = "+" if delta >= 0 else ""
        receipt = {
            "game_number": self.games_completed_count,
            "lmft_color": played_color,
            "result_label": result_label,
            "lmft_score": lmft_score,
            "previous_elo": prev_elo,
            "opponent_elo": opp_elo,
            "new_elo": new_elo,
            "elo_delta": delta,
            "next_lmft_color": next_color,
            "next_opponent_target_elo": self.opponent_target_elo,
        }
        self._current_game_adjudicated = True
        self.last_elo_receipt = receipt
        self.chat_messages.append(
            {
                "id": f"elo-{self.games_completed_count}-{int(time.time() * 1000) % 10000}",
                "sender": "system",
                "role_label": f"Objective Elo Arbiter — Game #{self.games_completed_count} ({result_label})",
                "text": (
                    f"LMFT ({played_color}) vs. {self.sparring_engine.engine_label} ({opp_elo:.0f} Elo): "
                    f"{result_label} -> LMFT Elo {prev_elo:.1f} -> {new_elo:.1f} ({sign}{delta:.1f}). "
                    f"Next game: LMFT rotates to {next_color} vs. {self.opponent_target_elo:.0f} Elo opponent."
                ),
            }
        )
        self._persist_user_state()
        return receipt

    def adjudicate_human_game_outcome(
        self,
        *,
        human_color: chess.Color = chess.WHITE,
    ) -> dict[str, Any]:
        """Score a Human vs. LMFT game, update the human player's FIDE Elo (`player_elo`), and record a coaching summary."""
        if getattr(self, "_current_human_game_adjudicated", False) and self.last_player_elo_receipt is not None:
            return self.last_player_elo_receipt
        outcome = self.board.outcome(claim_draw=True)
        if outcome is not None:
            if outcome.winner is None:
                human_score = 0.5
                result_label = "DRAW"
            elif outcome.winner == human_color:
                human_score = 1.0
                result_label = "WIN"
            else:
                human_score = 0.0
                result_label = "LOSS"
        else:
            cp = self.sparring_engine.evaluate_position_centipawns(
                self.board,
                perspective=human_color,
            )
            if cp >= 250.0:
                human_score = 1.0
                result_label = "WIN (Position Evaluation)"
            elif cp <= -250.0:
                human_score = 0.0
                result_label = "LOSS (Position Evaluation)"
            else:
                human_score = 0.5
                result_label = "DRAW (Balanced Position)"

        prev_elo = self.player_elo
        opp_elo = self.lmft_elo
        new_elo, delta = compute_fide_elo_update(prev_elo, opp_elo, human_score)
        self.player_elo = new_elo
        if human_score > 0.75:
            self.player_wins += 1
        elif human_score < 0.25:
            self.player_losses += 1
        else:
            self.player_draws += 1
        self.player_games_completed += 1
        mean_acc = (
            round(sum(self.player_move_accuracies) / len(self.player_move_accuracies), 1)
            if self.player_move_accuracies
            else 75.0
        )
        est_perf = compute_estimated_game_elo(mean_acc, reference_elo=opp_elo)
        sign = "+" if delta >= 0 else ""
        receipt = {
            "game_number": self.player_games_completed,
            "result_label": result_label,
            "human_score": human_score,
            "previous_player_elo": prev_elo,
            "lmft_opponent_elo": opp_elo,
            "new_player_elo": new_elo,
            "elo_delta": delta,
            "mean_accuracy_pct": mean_acc,
            "estimated_performance_elo": est_perf,
        }
        self._current_human_game_adjudicated = True
        self.last_player_elo_receipt = receipt
        self.chat_messages.append(
            {
                "id": f"player-elo-{self.player_games_completed}-{int(time.time() * 1000) % 10000}",
                "sender": "system",
                "role_label": f"Player Rating & Coach Arbiter — Game #{self.player_games_completed} ({result_label})",
                "text": (
                    f"Player vs. LMFT ({opp_elo:.0f} Elo): {result_label} · "
                    f"Accuracy {mean_acc:.1f}% (Performance Est. {est_perf:.0f} Elo) -> "
                    f"Player Elo {prev_elo:.1f} -> {new_elo:.1f} ({sign}{delta:.1f})."
                ),
            }
        )
        self._persist_user_state()
        return receipt

    def handle_microwave_step(self, *, num_sentences: int = 1) -> dict[str, Any]:
        """Execute one Sparring & Self-Kibitz step (`LMFT` vs. strength-calibrated `ObjectiveCalibratedSparringEngine`)."""
        _ = num_sentences
        self.microwave_step_counter += 1
        step_idx = self.microwave_step_counter
        if self.board.is_game_over(claim_draw=True) or len(self.move_history_san) >= 160:
            if not getattr(self, "_current_game_adjudicated", False):
                self._adjudicate_and_record_game_outcome()
                if self.sleep_queue and self.active_lineage != "virgin":
                    self.handle_run_rem_sleep(auto_triggered=True)
            self.handle_set_fen("start")

        lmft_color = chess.WHITE if self.lmft_playing_white else chess.BLACK
        is_lmft_turn = self.board.turn == lmft_color

        lmft_color_name = "White" if self.lmft_playing_white else "Black"
        stockfish_color_name = "Black" if self.lmft_playing_white else "White"

        if is_lmft_turn:
            board_before = self.board.copy()
            snap = self.handle_chess_ai_move()
            last_san = self.move_history_san[-1] if self.move_history_san else "?"
            if self.chat_messages:
                self.chat_messages[-1]["role_label"] = f"Celestriad 1.0 ({lmft_color_name} · after {last_san})"
            review = self.sparring_engine.audit_played_move_for_game_review(
                board_before,
                last_san,
                perspective=lmft_color,
                learned_leaves=self.learned_leaves,
                bitboard_sleeve=self.bitboard_sleeve,
            )
            if self.chat_messages:
                self.chat_messages[-1].setdefault("telemetry", {})["game_review"] = review
            already_vetoed = bool(
                getattr(self, "last_hybrid_move_arbitration", None)
                and self.last_hybrid_move_arbitration.get("veto_reason")
            )
            if self.chat_messages:
                last_scene_id = int(self.chat_messages[-1].get("id", self.scene_counter))
                if review["should_queue_for_sleep"] and (not already_vetoed or review["is_hazard"]):
                    self.flag_turn_for_coaching(
                        last_scene_id,
                        user_note=review["annotation_summary"],
                        check_auto_sleep=True,
                        is_hazard=bool(review["is_hazard"]),
                        override_coordinates=review["played_coordinates"],
                        contrastive_reference_coordinates=review["contrastive_reference_coordinates"],
                        game_review_annotation=review["annotation_summary"],
                    )
                elif not already_vetoed and not review["should_queue_for_sleep"]:
                    self._hazard_scene_indices.discard(last_scene_id)
                    self.sleep_queue = [s for s in self.sleep_queue if s.scene_index != last_scene_id]
            snap = self.get_state_snapshot()
            mover_tag = "LMFT"
        else:
            opp = self.sparring_engine.select_sparring_opponent_move(
                self.board,
                target_elo=self.opponent_target_elo,
            )
            snap = self.handle_chess_move(
                opp["move_san"],
                is_human_move=False,
                skip_spontaneous_left_brain=True,
            )
            if self.chat_messages:
                self.chat_messages[-1]["role_label"] = f"Stockfish 19 ({stockfish_color_name} · after {opp['move_san']})"
            snap = self.get_state_snapshot()
            mover_tag = f"SparringEngine({self.opponent_target_elo:.0f})"

        if self.board.is_game_over(claim_draw=True) and not getattr(self, "_current_game_adjudicated", False):
            self._adjudicate_and_record_game_outcome()
            if self.sleep_queue and self.active_lineage != "virgin":
                snap = self.handle_run_rem_sleep(auto_triggered=True)
            else:
                snap = self.get_state_snapshot()

        last_san = self.move_history_san[-1] if self.move_history_san else "start"
        step_label = f"Sparring Step #{step_idx} ({mover_tag}): {last_san}"
        self.last_microwave_status = {
            "step_index": step_idx,
            "lineage": self.active_lineage,
            "step_label": step_label,
            "mover": mover_tag,
            "lmft_elo": round(float(self.lmft_elo), 1),
            "opponent_target_elo": round(float(self.opponent_target_elo), 1),
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
        manifest = remote_manifest
        if manifest is None:
            try:
                manifest = fetch_remote_github_manifest()
            except Exception:
                manifest = None
        if manifest is None:
            manifest = build_release_manifest(self.repo_root)
        self.last_sync_status = verify_or_sync_against_manifest(
            self.repo_root,
            manifest,
            apply_sync=apply_sync,
        )
        if apply_sync and self.active_lineage != "virgin":
            merge_info = self.merge_master_seed_rules()
            self.last_sync_status["merged_rules_count"] = merge_info["merged_rules_count"]
        return self.get_state_snapshot()

    def _consolidate_sleep_clusters_locally(
        self,
        queued_scenes: tuple[UnresolvedSleepScene, ...],
    ) -> tuple[tuple[str, ...], tuple[FuzzyConceptRule, ...], tuple[str, ...]]:
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

        cur_archives = self.archives
        cur_ledger = self.ledger

        clusters: dict[tuple[str, bool], list[UnresolvedSleepScene]] = {}
        for s in queued_scenes:
            is_uncoached_collision = (
                (
                    s.active_concept_collision is not None
                    or (
                        s.scenario_text.startswith("Chess table —")
                        and "[Coach note:" not in s.scenario_text
                    )
                )
                and s.contrastive_reference_coordinates is None
                and not s.game_review_annotation
            )
            if is_uncoached_collision:
                cur_ledger = cur_ledger.record_settled(
                    scene_index=s.scene_index,
                    takeaway_source=f"Resolved rule collision ({s.active_concept_collision}) via sleep threshold refinement.",
                    felt_chord=compile_felt_chord(
                        s.external_coordinates,
                        salience=s.salience,
                        valence=s.valence,
                        arousal=s.arousal,
                        resonance=0.65,
                        learned_leaves=tuple(self.learned_leaves),
                    ),
                    clarity=0.85,
                    confidence=0.90,
                    resolved_uncertainty_scenes=(s.scene_index,),
                )
                continue
            is_s_haz = bool(s.is_hazard or s.scene_index in self._hazard_scene_indices)
            if is_s_haz and s.game_review_annotation and (
                "[!! BRILLIANCY" in s.game_review_annotation
                or "[! GREAT_MOVE" in s.game_review_annotation
            ):
                is_s_haz = False
            if is_s_haz:
                s.external_coordinates["cultural/efficiency/wastelessness"] = min(
                    0.25,
                    float(s.external_coordinates.get("cultural/efficiency/wastelessness", 0.25)),
                )
                s.external_coordinates["cultural/freedom/needlessness"] = max(
                    0.53,
                    float(s.external_coordinates.get("cultural/freedom/needlessness", 0.53)),
                )
            ckey = _cluster_key_for_unresolved(s)
            if ckey is not None:
                clusters.setdefault((ckey, is_s_haz), []).append(s)

        calm_book_coords: dict[str, float] = {k: 0.50 for k in FOUNDATIONAL_LEAVES}
        calm_book_coords.update(
            {
                "philosophical/being/existence": 0.83,
                "philosophical/structure/order": 0.87,
                "philosophical/being/agency": 0.51,
                "philosophical/structure/chaos": 0.22,
                "philosophical/persistence/continuity": 0.86,
                "philosophical/persistence/discreteness": 0.44,
                "cultural/efficiency/wastelessness": 0.68,
                "cultural/freedom/needlessness": 0.09,
            }
        )

        for (ckey, is_group_hazard), group in clusters.items():
            avg_coords = _average_cluster_coordinates(group)
            avg_ref_coords = _average_contrastive_reference_coordinates(group)
            if is_group_hazard:
                avg_coords["cultural/efficiency/wastelessness"] = min(
                    0.25,
                    float(avg_coords.get("cultural/efficiency/wastelessness", 0.25)),
                )
                avg_coords["cultural/freedom/needlessness"] = max(
                    0.53,
                    float(avg_coords.get("cultural/freedom/needlessness", 0.53)),
                )
            target_leaves, rule_low, rule_high = normalize_contrastive_target_leaves_and_thresholds(
                ckey,
                avg_coords,
                avg_ref_coords,
            )
            if is_group_hazard:
                target_leaves["cultural/efficiency/wastelessness"] = min(
                    0.25,
                    float(target_leaves.get("cultural/efficiency/wastelessness", 0.25)),
                )
                for bad_k in (
                    "cultural/freedom/needlessness",
                    "philosophical/being/existence",
                    "cultural/freedom/self-sufficiency",
                    "philosophical/structure/order",
                ):
                    if bad_k == "cultural/freedom/needlessness" and float(target_leaves.get(bad_k, 1.0)) <= 0.30:
                        target_leaves[bad_k] = 0.53
                    elif bad_k != "cultural/freedom/needlessness" and float(target_leaves.get(bad_k, 0.0)) >= 0.65:
                        target_leaves.pop(bad_k, None)
                if len(target_leaves) < 2:
                    target_leaves["cultural/freedom/needlessness"] = 0.53
            rep = group[0]
            rep_scene = rep.scene_index

            prefix = "hazard" if is_group_hazard else "coached"
            base_name = f"{prefix}_{ckey}"
            concept_name = base_name

            description = (
                f"Consolidated REM chess wisdom rule for {ckey.replace('_', ' ')}: "
                f"{(rep.game_review_annotation or rep.scenario_text)[:140]}"
            )
            new_rule = FuzzyConceptRule(
                rule_id=f"rule-sleep-{concept_name[:32]}",
                concept_name=concept_name,
                description=description,
                target_leaves=target_leaves,
                low=rule_low,
                high=rule_high,
            )
            if new_rule.tier == 1:
                probe_coords = dict(calm_book_coords)
                for req_leaf in new_rule.target_leaves:
                    probe_coords.setdefault(req_leaf, 0.0)
                if new_rule.evaluate(probe_coords) >= 0.35:
                    adjusted_leaves = dict(target_leaves)
                    if is_group_hazard:
                        adjusted_leaves["cultural/efficiency/wastelessness"] = 0.20
                        rule_low = max(rule_low, 0.68)
                        rule_high = max(rule_high, 0.84)
                    else:
                        rule_low = max(rule_low, 0.76)
                        rule_high = max(rule_high, 0.88)
                    new_rule = FuzzyConceptRule(
                        rule_id=new_rule.rule_id,
                        concept_name=new_rule.concept_name,
                        description=new_rule.description,
                        target_leaves=adjusted_leaves,
                        low=rule_low,
                        high=rule_high,
                    )
            dup_idx = next(
                (
                    idx
                    for idx, r in enumerate(active_rules)
                    if _canonical_cluster_concept_name(r.concept_name) == base_name
                    or (
                        r.concept_name.startswith(f"{prefix}_")
                        and _canonical_cluster_concept_name(r.concept_name).endswith("_cluster")
                        and dict(r.target_leaves) == dict(new_rule.target_leaves)
                    )
                ),
                None,
            )
            if dup_idx is not None:
                entrained_rule = _entrain_existing_cluster_rule(
                    active_rules[dup_idx],
                    new_rule,
                    base_name,
                )
                active_rules[dup_idx] = entrained_rule
                forged_rules.append(entrained_rule)
            else:
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

            cur_archives = ThreeBranchArchives(
                executive=cur_archives.executive.record_scene(
                    scene_index=rep_scene,
                    coordinates=avg_coords,
                    action_summary=exec_text,
                    harvested_clause=exec_clause,
                ),
                legislative=cur_archives.legislative.record_scene(
                    scene_index=rep_scene,
                    sensory_snapshot=leg_text,
                    felt_chord=avg_fc,
                    coordinates=avg_coords,
                    harvested_clause=leg_clause,
                ),
                judicial=cur_archives.judicial.record_clause_and_chord(
                    harvested_clause=jud_clause,
                    ratified_chord=new_rule,
                ),
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
        with self.gpu_lock:
            self.rem_sleep_in_progress = True
            try:
                queued_snapshot = tuple(self.sleep_queue)
                queued_count = len(queued_snapshot)
                if self.active_lineage == "virgin":
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
                    for s in queued_snapshot
                ]
                forged_rules_list: tuple[FuzzyConceptRule, ...] = ()
                if queued_count > 0:
                    for s in queued_snapshot:
                        if s.contrastive_reference_coordinates is not None or s.game_review_annotation:
                            is_s_haz = bool(s.is_hazard or s.scene_index in self._hazard_scene_indices)
                            if is_s_haz and s.game_review_annotation and (
                                "[!! BRILLIANCY" in s.game_review_annotation
                                or "[! GREAT_MOVE" in s.game_review_annotation
                            ):
                                is_s_haz = False
                            self.bitboard_sleeve.distill_post_mortem_scene(
                                s.external_coordinates,
                                s.contrastive_reference_coordinates,
                                is_hazard=is_s_haz,
                            )
                    if self.teacher is not None:
                        virgin_ids = {r.rule_id for r in VIRGIN_FOUNDATIONAL_RULES}
                        non_virgin_rules = tuple(r for r in self.rules if r.rule_id not in virgin_ids)
                        report = run_sleep_consolidation(
                            unresolved_scenes=queued_snapshot,
                            rules=non_virgin_rules,
                            learned_leaves=self.learned_leaves,
                            archives=self.archives,
                            ledger=self.ledger,
                            teacher=self.teacher,
                        )
                        self.rules = _ensure_foundational_rules(report.updated_rules)
                        self.archives = report.updated_archives
                        self.ledger = report.updated_ledger
                        refined_ids = report.refined_rule_ids
                        forged_rules_list = report.forged_rules
                    else:
                        refined_ids, forged_rules_list, _clause_ids = (
                            self._consolidate_sleep_clusters_locally(queued_snapshot)
                        )
                else:
                    refined_ids = ()

                forged_count = len(forged_rules_list)
                after_evals = [
                    self._evaluate_sleep_scene_snapshot(s, tuple(self.rules))
                    for s in queued_snapshot
                ]
                self.sleep_queue.clear()
                self._persist_user_state()
                if self._is_default_live_state_dir():
                    self.export_master_release_seed()

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


def create_chess_http_server(
    runtime: ChessKibitzerRuntime,
    *,
    host: str = "127.0.0.1",
    port: int = 8420,
) -> ThreadingHTTPServer:
    """Create a thread-safe local HTTP server for `LMFT-Chess.exe`."""

    class ChessRequestHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html") or self.path.startswith("/?") or self.path.startswith("/index.html?"):
                body = CHESS_SINGLE_PAGE_HTML.encode("utf-8")
                self.send_response(int(HTTPStatus.OK))
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
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
                if self.path == "/api/threads/create":
                    self._send_json(runtime.create_chat_thread(str(data.get("title", ""))))
                    return
                if self.path == "/api/threads/switch":
                    self._send_json(runtime.switch_chat_thread(str(data.get("thread_id", ""))))
                    return
                if self.path == "/api/threads/rename":
                    self._send_json(
                        runtime.rename_chat_thread(
                            str(data.get("thread_id", "")),
                            str(data.get("title", "")),
                        )
                    )
                    return
                if self.path == "/api/threads/delete":
                    self._send_json(runtime.delete_chat_thread(str(data.get("thread_id", ""))))
                    return
                if self.path == "/api/sleep/auto":
                    self._send_json(
                        runtime.set_auto_rem_sleep(
                            bool(data.get("enabled", True)),
                            threshold=int(data["threshold"]) if data.get("threshold") is not None else None,
                        )
                    )
                    return
                if self.path == "/api/microwave/step":
                    self._send_json(
                        runtime.handle_microwave_step(num_sentences=int(data.get("num_sentences", 1)))
                    )
                    return
                if self.path == "/api/system/sync":
                    self._send_json(
                        runtime.handle_check_github_sync(
                            remote_manifest=data.get("remote_manifest"),
                            apply_sync=bool(data.get("apply_sync", False)),
                        )
                    )
                    return
                if self.path == "/api/chess/move":
                    self._send_json(runtime.handle_chess_move(str(data.get("move", ""))))
                    return
                if self.path == "/api/chess/undo":
                    self._send_json(runtime.handle_chess_undo())
                    return
                if self.path == "/api/chess/ai_move":
                    self._send_json(runtime.handle_chess_ai_move())
                    return
                if self.path == "/api/chess/fen":
                    self._send_json(runtime.handle_set_fen(str(data.get("fen", "start"))))
                    return
                if self.path == "/api/chat":
                    anchor_val = data.get("anchor_to_board")
                    self._send_json(
                        runtime.handle_chat_message(
                            str(data.get("text", "")),
                            room=str(data.get("room", "chess")),
                            sanctuary_mode=str(data.get("sanctuary_mode", "tilt_reset")),
                            anchor_to_board=bool(anchor_val) if anchor_val is not None else None,
                            num_sentences=int(data.get("num_sentences", 1)),
                        )
                    )
                    return
                if self.path == "/api/turn/coach":
                    self._send_json(
                        runtime.flag_turn_for_coaching(
                            int(data["scene_index"]),
                            user_note=str(data.get("note", "")),
                        )
                    )
                    return
                if self.path == "/api/turn/savor":
                    self._send_json(runtime.savor_turn_vibe(int(data["scene_index"])))
                    return
                if self.path == "/api/sleep/run":
                    self._send_json(runtime.handle_run_rem_sleep())
                    return
                if self.path == "/api/seed/export":
                    self._send_json(runtime.export_master_release_seed())
                    return
                if self.path == "/api/player/adjudicate":
                    self._send_json(runtime.adjudicate_human_game_outcome())
                    return
                if self.path == "/api/lineage":
                    self._send_json(runtime.switch_brain_lineage(str(data.get("lineage", "schooled"))))
                    return
                self._send_json({"error": f"Unknown endpoint {self.path}"}, HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    return ThreadingHTTPServer((host, port), ChessRequestHandler)


create_club_http_server = create_chess_http_server


def launch_chess_desktop_window(url: str, *, profile_dir: Path | None = None) -> subprocess.Popen[Any] | None:
    if sys.platform.startswith("win"):
        edge_candidates = [
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        ]
        from lmft_architecture.workspace_paths import resolve_training_root

        user_data_dir = profile_dir or (resolve_training_root() / "outputs" / ".chess_window_profile")
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


def _wait_for_local_server_ready(url: str, *, timeout_sec: float = 3.0) -> bool:
    deadline = time.time() + timeout_sec
    state_url = f"{url.rstrip('/')}/api/state"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(state_url, timeout=0.5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.05)
    return False


def launch_native_pywebview_window(title: str, url: str) -> bool:
    """Open `url` inside a native Windows desktop window (`pywebview`) with no browser chrome or 127.0.0.1 title."""
    try:
        import webview  # type: ignore[import-untyped]
        from lmft_architecture.workspace_paths import resolve_training_root

        storage_dir = resolve_training_root() / "outputs" / ".chess_window_profile"
        storage_dir.mkdir(parents=True, exist_ok=True)
        webview.create_window(
            title,
            url,
            width=1360,
            height=860,
            min_size=(1020, 680),
            background_color="#0b1017",
        )
        webview.start(private_mode=False, storage_path=str(storage_dir))
        return True
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the LMFT-Chess desktop app.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8420, help="Bind port (default: 8420)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the desktop window")
    parser.add_argument("--no-gpu", action="store_true", help="Run in fast Gear-1 Right-Brain mode without loading Qwen3-4B")
    parser.add_argument("--single-window", action="store_true", help="Exit the server automatically when the desktop window is closed")
    args = parser.parse_args(argv)

    runtime = ChessKibitzerRuntime(lazy_load_gpu=not args.no_gpu)
    try:
        server = create_chess_http_server(runtime, host=args.host, port=args.port)
    except OSError:
        server = create_chess_http_server(runtime, host=args.host, port=0)
    actual_port = server.server_port
    url = f"http://{args.host}:{actual_port}"
    if sys.stdout is not None:
        print(f"LMFT-Chess running at {url}")

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    _wait_for_local_server_ready(url, timeout_sec=3.0)

    try:
        if not args.no_browser:
            window_url = f"{url}/?v={int(time.time())}"
            if launch_native_pywebview_window(
                "LMFT-Chess — Simultaneous Integration & Self-Kibitzer Arena (Qwen3-4B)",
                window_url,
            ):
                return 0
            t0 = time.time()
            window_proc = launch_chess_desktop_window(window_url)
            if (args.single_window or sys.executable.lower().endswith("pythonw.exe")) and window_proc is not None:
                window_proc.wait()
                # If Edge handed off to an existing background broker process in < 3s, keep server alive
                if (time.time() - t0) >= 3.0:
                    return 0
        while server_thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        if sys.stdout is not None:
            print("\nShutting down LMFT-Chess...")
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
