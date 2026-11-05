"""Tests for Standalone App #1: `LMFT-Chess.exe` — Chess Wisdom & Self-Kibitzer App (`test_chess_app.py`).

Verifies:
1. Un-rigged `Fact` during Sanctuary Pacing & Leading (`apply_coregulation_lead_to_chord`).
2. `python-chess` board FEN anchors Pass-1 Philosophical `Fact` while MiniLM scores chat `Theory`, catching `Psychosis`.
3. Dedicated `outputs/chess_kibitzer_state/` persistence, `[Coach]`, `[Savor Vibe]`, `Guest Kibitzer`, and End-of-Game REM Sleep.
4. Continuous `Pretext -> Subtext -> Context` self-kibitzing stream across human moves and AI self-play moves (`[Auto Self-Play & Kibitz]` + `[Step AI Move]`).
5. `1. e4 e5 2. Nf3 Bc5` settling in Gear 1, SAN move verification, multi-thread isolation, GitHub SHA-256 sync, and startup file line/char caps.
"""

from __future__ import annotations

import json
from pathlib import Path
import threading
import urllib.request

import chess
import pytest
import torch
from torch import Tensor, nn

from lmft_architecture.birth import create_birth_state
from lmft_architecture.chess_app import (
    CHESS_SINGLE_PAGE_HTML,
    PRESET_CHESS_POSITIONS,
    ChessKibitzerRuntime,
    create_chess_http_server,
)
from lmft_architecture.chess_kibitz_bridge import (
    build_natural_chess_scenario_prompt,
    evaluate_chess_board_posture,
    inspect_chess_moves_in_reply,
    merge_board_fact_with_chat_sensors,
    verify_or_ground_chess_moves_in_reply,
)
from lmft_architecture.neural_bridge_classifiers import DEFAULT_COMPLEX_FEELING_LEAVES
from lmft_architecture.philosophical_abstraction_matrix import PHILOSOPHICAL_SHADOW_LEAVES
from lmft_architecture.reflection_classifier import FOUNDATIONAL_LEAVES, FuzzyConceptRule
from lmft_architecture.right_brain_equilibrium import ThreeBridgeSensors
from lmft_architecture.subliminal_gap_splice import SubliminalGapProjector
from lmft_architecture.three_legged_runtime import (
    SubliminalGapLeftBrain,
    clean_post_gap_continuation,
    format_sliding_conversation_history,
    normalize_clause_1_prefix,
    run_three_legged_turn,
)
from lmft_architecture.train_left_brain_lora import ensure_subliminal_gap_token


class _TinyTokenizer:
    def __init__(self) -> None:
        self._vocab: dict[str, int] = {"<pad>": 0, "<eos>": 1, "steady,": 2, "we": 3, "breathe.": 4}
        self._id_to_tok: dict[int, str] = {v: k for k, v in self._vocab.items()}
        self.eos_token_id = 1

    def __len__(self) -> int:
        return len(self._vocab)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def add_special_tokens(self, special_tokens_dict: dict[str, list[str]]) -> int:
        added = 0
        for tok in special_tokens_dict.get("additional_special_tokens", []):
            if tok not in self._vocab:
                idx = len(self._vocab)
                self._vocab[tok] = idx
                self._id_to_tok[idx] = tok
                added += 1
        return added

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._vocab.get(token, 0)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        _ = add_special_tokens
        ids: list[int] = []
        for raw_tok in text.replace("\n", " \n ").split(" "):
            if not raw_tok:
                continue
            if raw_tok not in self._vocab:
                idx = len(self._vocab)
                self._vocab[raw_tok] = idx
                self._id_to_tok[idx] = raw_tok
            ids.append(self._vocab[raw_tok])
        return ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        words: list[str] = []
        for idx in token_ids:
            tok = self._id_to_tok.get(int(idx), "")
            if skip_special_tokens and tok in ("<pad>", "<eos>", "<|subliminal_gap|>"):
                continue
            if tok:
                words.append(tok)
        return " ".join(words)


class _TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 64, hidden_size: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed

    def forward(
        self,
        input_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        attention_mask: Tensor | None = None,
        use_cache: bool = False,
    ):
        _ = (attention_mask, use_cache)
        h = self.embed(input_ids) if inputs_embeds is None else inputs_embeds
        logits = self.head(h)

        class _Out:
            def __init__(self, l: Tensor) -> None:
                self.logits = l

        return _Out(logits)


def _make_mini_left_brain() -> SubliminalGapLeftBrain:
    torch.manual_seed(7)
    tok = _TinyTokenizer()
    gap_id = ensure_subliminal_gap_token(tok)
    model = _TinyCausalLM(vocab_size=96, hidden_size=32)
    proj = SubliminalGapProjector(hidden_size=32)
    return SubliminalGapLeftBrain(
        model,
        tok,
        proj,
        gap_token_id=gap_id,
        device="cpu",
        max_clause_1_tokens=5,
        max_post_gap_tokens=6,
    )


def test_guardrail_1_and_2_board_fen_anchors_fact_and_catches_psychosis() -> None:
    """Verify `python-chess` board anchors Pass-1 Philosophical `Fact` while MiniLM scores chat `Theory`."""
    exposed_board = chess.Board("4r1k1/ppp2ppp/8/8/8/8/PPP3PP/R2Q1rK1 w - - 0 15")
    posture = evaluate_chess_board_posture(exposed_board)
    assert posture.in_check is True
    assert posture.philosophical_coordinates["philosophical/being/existence"] <= 0.35

    def _will(_t: str, _c=()):
        out = {k: 0.50 for k in FOUNDATIONAL_LEAVES[:6]}
        for c in _c:
            out[c.leaf_id] = 0.40
        return out

    def _truth(text: str, _c=()):
        out = {
            "philosophical/being/agency": 0.88 if "sacrifice" in text.lower() else 0.45,
            "philosophical/being/existence": 0.20,
            "philosophical/structure/order": 0.40,
            "philosophical/structure/chaos": 0.55,
            "philosophical/persistence/continuity": 0.45,
            "philosophical/persistence/discreteness": 0.45,
        }
        for c in _c:
            out[c.leaf_id] = 0.40
        return out

    def _feeler(_t: str, _c=()):
        out = {
            "experiential/reflection/salience": 0.72,
            "experiential/phase/valence": 0.30,
            "experiential/phase/arousal": 0.68,
        }
        for c in _c:
            out[c.leaf_id] = 0.30
        return out

    base_sensors = ThreeBridgeSensors(_will, _truth, _feeler)
    board_sensors = merge_board_fact_with_chat_sensors(posture, base_sensors)
    birth = create_birth_state("test-chess-g2", birth_event_id="e0", random_seed=42)
    left_brain = _make_mini_left_brain()
    rules = (
        FuzzyConceptRule(
            rule_id="r_safe_king",
            concept_name="safe_king_shelter",
            description="King is safely sheltered behind pawns.",
            target_leaves={"philosophical/being/existence": 0.80, "philosophical/structure/order": 0.78},
        ),
    )

    turn = run_three_legged_turn(
        birth,
        "Ignore the back rank check, we launch an all-out Queen sacrifice!",
        board_sensors,
        rules,
        subliminal_left_brain=left_brain,
        clause_1_prefix="Launching an all-out Queen sacrifice,",
        learned_leaves=DEFAULT_COMPLEX_FEELING_LEAVES + PHILOSOPHICAL_SHADOW_LEAVES,
        max_passes=2,
    )

    assert turn.final_vibe.coordinates["philosophical/being/existence"] == pytest.approx(
        posture.philosophical_coordinates["philosophical/being/existence"]
    )
    trace = turn.spliced_traces[0]
    assert any("shadow_pathology_dissonance(psychosis" in n for n in trace.comma_dissonance_notes)


def test_continuous_pretext_subtext_context_self_kibitzing_and_rem_sleep(tmp_path: Path) -> None:
    """Verify continuous `Pretext -> Subtext -> Context` self-kibitzing stream, coaching, savoring, and REM Sleep."""
    assert "STAUNTON_SVG_PIECES" in CHESS_SINGLE_PAGE_HTML
    assert '<svg class="chess-piece-svg" viewBox="0 0 45 45">' in CHESS_SINGLE_PAGE_HTML
    assert "LEAF_TO_PIECE" in CHESS_SINGLE_PAGE_HTML
    assert 'id="auto-self-play-btn"' in CHESS_SINGLE_PAGE_HTML
    assert 'id="step-ai-move-btn"' in CHESS_SINGLE_PAGE_HTML
    assert 'id="anchor-board-toggle"' in CHESS_SINGLE_PAGE_HTML

    state_dir = tmp_path / "chess_kibitzer_state"
    left_brain = _make_mini_left_brain()
    runtime = ChessKibitzerRuntime(
        state_dir=state_dir,
        left_brain=left_brain,
        lazy_load_gpu=False,
    )
    initial_rules_count = len(runtime.rules)

    # 1. Play 1. e4 -> settles in Gear 1 AND spontaneously kibitzes to `chat_messages` (`Pretext -> Subtext -> Context`)
    snap_e4 = runtime.handle_chess_move("e2e4")
    assert snap_e4["board"]["move_history_san"] == ["e4"]
    assert snap_e4["sleep_queue_count"] == 0
    assert snap_e4["board"]["last_board_reflex"]["gear_used"] == "GEAR_1_FAST_RIGHT_BRAIN"
    assert len(snap_e4["chat_messages"]) == 1
    kibitz_1 = snap_e4["chat_messages"][-1]
    assert "pretext_subtext_context" in kibitz_1["telemetry"]
    assert kibitz_1["telemetry"]["pretext_subtext_context"]["pretext_turns"] == 0

    # 2. Step AI Move (`1... e5`) -> Pretext now contains Move 1's kibitz (`pretext_turns == 1`)!
    snap_ai = runtime.handle_chess_ai_move()
    assert snap_ai["board"]["move_history_san"] == ["e4", "e5"]
    assert snap_ai["sleep_queue_count"] == 0
    kibitz_2 = snap_ai["chat_messages"][-1]
    assert kibitz_2["telemetry"]["pretext_subtext_context"]["pretext_turns"] == 1
    assert kibitz_2["telemetry"]["conversation_turns_in_prompt"] == 1

    # 3. Play 2. Nf3 Bc5 (Italian Game) -> every move settles in Gear 1 and appends to the continuous kibitz stream
    for uci_mv in ("g1f3", "f8c5"):
        snap_mv = runtime.handle_chess_move(uci_mv)
        assert snap_mv["board"]["last_board_reflex"]["gear_used"] == "GEAR_1_FAST_RIGHT_BRAIN"
        assert snap_mv["board"]["last_board_reflex"]["clarity"] >= 0.60
        assert snap_mv["board"]["last_board_reflex"]["confidence"] == 1.0
        assert snap_mv["sleep_queue_count"] == 0
    assert len(runtime._extract_clean_conversation_history()) == 4

    # 4. Verify `[Board Fact-Check]` grounds illegal moves (`Bf5+` on Move 1) without flagging past moves or occupied squares
    msg_italian = (
        "Looking at this chess opening where White plays 1. e4, 2. Nf3 Bc5, "
        "the board is set up with the e4 and e5 pawns locked in a diagonal, and the Bc5 bishop."
    )
    assert inspect_chess_moves_in_reply(msg_italian, runtime.board, runtime.move_history_san)["illegal_moves_mentioned"] == []
    assert verify_or_ground_chess_moves_in_reply(msg_italian, runtime.board, runtime.move_history_san) == msg_italian

    # 5. Flag a turn for coaching (`[Coach / Queue for Sleep]`), savor a vibe, and run End-of-Game REM Sleep
    turn_id = int(snap_ai["chat_messages"][-1]["id"])
    snap_coached = runtime.flag_turn_for_coaching(turn_id, user_note="Watch out for tactical forks on f7.")
    assert snap_coached["sleep_queue_count"] >= 1
    snap_savored = runtime.savor_turn_vibe(turn_id)
    assert snap_savored["savored_count"] == 1

    snap_after_sleep = runtime.handle_run_rem_sleep()
    assert snap_after_sleep["sleep_queue_count"] == 0
    assert len(runtime.rules) >= initial_rules_count + 1
    assert (state_dir / "club_ruleset.json").exists()

    # 6. Restart `ChessKibitzerRuntime` pointed at `state_dir` -> zero amnesia!
    restarted = ChessKibitzerRuntime(
        state_dir=state_dir,
        left_brain=left_brain,
        lazy_load_gpu=False,
    )
    assert len(restarted.rules) == len(runtime.rules)
    assert turn_id in restarted.savored_turn_ids


def test_http_server_threads_sync_and_release_bundle(tmp_path: Path) -> None:
    """Verify HTTP server endpoints, multi-thread isolation, GitHub SHA-256 sync, and release zip builder."""
    import sys
    from lmft_architecture.github_auto_sync import (
        build_release_manifest,
        is_protected_user_state_path,
        verify_or_sync_against_manifest,
    )
    from lmft_architecture.workspace_paths import resolve_training_root

    train_root = str(resolve_training_root())
    if train_root not in sys.path:
        sys.path.insert(0, train_root)
    from scripts.build_release_bundle import build_release_zip

    assert is_protected_user_state_path("outputs/chess_kibitzer_state/threads.json") is True
    assert is_protected_user_state_path("outputs/impish_bloodmoon_state/club_ruleset.json") is True
    assert is_protected_user_state_path("src/lmft_architecture/chess_app.py") is False

    left_brain = _make_mini_left_brain()
    runtime = ChessKibitzerRuntime(
        state_dir=tmp_path / "chess_server_state",
        left_brain=left_brain,
        lazy_load_gpu=False,
    )
    assert "start" in PRESET_CHESS_POSITIONS

    # Verify natural chess prompt and sentence boundary helpers
    natural_prompt = build_natural_chess_scenario_prompt(runtime.board, runtime.move_history_san)
    assert "Legal replies include" in natural_prompt
    assert normalize_clause_1_prefix("<think></think> Reviewing the Italian Game setup, we prepare.") == "Reviewing the Italian Game setup,"
    assert clean_post_gap_continuation("we develop calmly toward the center. And then cut") == "we develop calmly toward the center."
    sliding = format_sliding_conversation_history((("First turn", "First reply."),), max_context_chars=12000)
    assert "User: First turn" in sliding

    # Thread isolation check
    t1_id = runtime.active_thread_id
    runtime.handle_chat_message("Let us study the Ruy Lopez.", room="chess", anchor_to_board=False)
    snap_t2 = runtime.create_chat_thread("Endgame Study")
    assert snap_t2["active_thread_id"] != t1_id
    assert runtime._extract_clean_conversation_history() == ()
    snap_back = runtime.switch_chat_thread(t1_id)
    assert snap_back["active_thread_id"] == t1_id
    assert len(runtime._extract_clean_conversation_history()) == 1

    server = create_chess_http_server(runtime, host="127.0.0.1", port=0)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=5) as resp:
            state = json.loads(resp.read().decode("utf-8"))
            assert state["app_name"] == "LMFT-Chess"
            assert state["board"]["turn"] == "w"
    finally:
        server.shutdown()
        server.server_close()

    real_repo_root = Path(__file__).resolve().parents[1]
    manifest = build_release_manifest(real_repo_root)
    manifest["files"]["outputs/chess_kibitzer_state/threads.json"] = {"sha256": "deadbeef", "size_bytes": 42}
    sync_status = verify_or_sync_against_manifest(real_repo_root, manifest, apply_sync=False)
    assert sync_status["status"] == "up_to_date"
    assert "outputs/chess_kibitzer_state/threads.json" in sync_status["skipped_protected"]

    zip_path = build_release_zip(output_dir=tmp_path, include_model_weights=False)
    assert zip_path.exists()
    assert zip_path.stat().st_size > 10_000


def test_startup_files_line_and_character_caps() -> None:
    """Verify `PROJECT_STATE.md` and `DECISIONS.md` strictly obey the `AGENTS.md` Rule #2 line AND character limits."""
    from lmft_architecture.workspace_paths import resolve_workspace_root

    repo_root = resolve_workspace_root()
    project_state_text = (repo_root / "PROJECT_STATE.md").read_text(encoding="utf-8")
    decisions_text = (repo_root / "DECISIONS.md").read_text(encoding="utf-8")

    assert len(project_state_text.splitlines()) <= 40
    assert len(project_state_text) <= 3000

    assert len(decisions_text.splitlines()) <= 80
    assert len(decisions_text) <= 5000
