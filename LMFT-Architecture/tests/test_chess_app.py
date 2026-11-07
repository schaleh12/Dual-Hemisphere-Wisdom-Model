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
    evaluate_candidate_moves_with_right_brain,
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
            if raw_tok in self._vocab:
                ids.append(self._vocab[raw_tok])
            else:
                idx = 6 + (sum(ord(ch) * (i + 1) for i, ch in enumerate(raw_tok)) % 88)
                ids.append(idx)
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

    # 1. Play 1. e4 -> settles silently in Gear 1 (`is_gear_1_fallback=True`, 0 Left-Brain calls)
    snap_e4 = runtime.handle_chess_move("e2e4")
    assert snap_e4["board"]["move_history_san"] == ["e4"]
    assert snap_e4["sleep_queue_count"] == 0
    assert snap_e4["board"]["last_board_reflex"]["gear_used"] == "GEAR_1_FAST_RIGHT_BRAIN"
    assert len(snap_e4["chat_messages"]) == 1
    kibitz_1 = snap_e4["chat_messages"][-1]
    assert kibitz_1["is_gear_1_fallback"] is True
    assert "pretext_subtext_context" in kibitz_1["telemetry"]
    assert kibitz_1["telemetry"]["pretext_subtext_context"]["pretext_turns"] == 0

    # 2. Step AI Move (`1... e5`) -> also plays silently in Gear 1 without polluting conversation history
    snap_ai = runtime.handle_chess_ai_move()
    assert snap_ai["board"]["move_history_san"] == ["e4", "e5"]
    assert snap_ai["sleep_queue_count"] == 0
    kibitz_2 = snap_ai["chat_messages"][-1]
    assert kibitz_2["is_gear_1_fallback"] is True
    assert kibitz_2["telemetry"]["pretext_subtext_context"]["pretext_turns"] == 0

    # 3. Play 2. Nf3 Bc5 (Italian Game) -> every move settles in Gear 1 and board receipts stay out of chat history
    for uci_mv in ("g1f3", "f8c5"):
        snap_mv = runtime.handle_chess_move(uci_mv)
        assert snap_mv["board"]["last_board_reflex"]["gear_used"] == "GEAR_1_FAST_RIGHT_BRAIN"
        assert snap_mv["board"]["last_board_reflex"]["clarity"] >= 0.60
        assert snap_mv["board"]["last_board_reflex"]["confidence"] == 1.0
        assert snap_mv["sleep_queue_count"] == 0
    assert len(runtime._extract_clean_conversation_history()) == 0

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
    assert snap_after_sleep["last_rem_sleep_self_talk"] is not None
    assert "REM Dream Self-Reflection:" in snap_after_sleep["last_rem_sleep_receipt"]
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

    assert 'id="celestriad-jukebox"' in CHESS_SINGLE_PAGE_HTML
    assert "toggleJukeboxMute()" in CHESS_SINGLE_PAGE_HTML
    assert 'id="space-center-modal-overlay"' in CHESS_SINGLE_PAGE_HTML
    assert "syncAndAutoUpdate()" in CHESS_SINGLE_PAGE_HTML

    server = create_chess_http_server(runtime, host="127.0.0.1", port=0)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=5) as resp:
            state = json.loads(resp.read().decode("utf-8"))
            assert state["app_name"] == "LMFT-Chess"
            assert state["board"]["turn"] == "w"
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/assets/celestial_seal_poem.jpg", timeout=5) as img_resp:
            img_bytes = img_resp.read()
            assert img_resp.status == 200
            assert img_resp.headers.get("Content-Type") == "image/jpeg"
            assert len(img_bytes) > 50_000
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/assets/icy_sea_board.png", timeout=5) as board_resp:
            board_bytes = board_resp.read()
            assert board_resp.status == 200
            assert board_resp.headers.get("Content-Type") == "image/png"
            assert len(board_bytes) > 100_000
    finally:
        server.shutdown()
        server.server_close()

    real_repo_root = Path(__file__).resolve().parents[1]
    manifest = build_release_manifest(real_repo_root, write_to_disk=False)
    assert "docs/assets/celestial_seal_poem.jpg" in manifest["files"]
    assert "dist/impish_bloodmoon_patreon.enc" not in manifest["files"]
    assert "dist/lmft-sanctuary-club-v0.1.0-win64.zip" not in manifest["files"]
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


def test_hybrid_move_selection_and_blunder_veto_sleep_queue(tmp_path: Path) -> None:
    """Verify Right-Brain candidate ranking (`self.rules`) + Left-Brain `<|subliminal_gap|>` commit and blunder veto."""
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 3")
    evals = evaluate_candidate_moves_with_right_brain(board, limit=30)
    eval_by_san = {e.san: e for e in evals}
    # Playing `Bxf7+` without support hangs the Bishop to `Kxf7` (`is_blunder = True`)
    assert eval_by_san["Bxf7+"].is_blunder is True
    # Castling `O-O` is safe and ranks near the top
    assert eval_by_san["O-O"].is_blunder is False
    assert eval_by_san["O-O"].composite_score > eval_by_san["Bxf7+"].composite_score

    # Verify that during chess play (`handle_chess_ai_move`), Celestriad selects moves silently via
    # Gear-1 Right-Brain intuition (`0` Left-Brain calls) and only wakes the Left Brain during REM Sleep (A)
    # or when asked a question by a human (B).
    class _StubLeftBrain:
        def __init__(self, reply_text: str) -> None:
            self.reply_text = reply_text
            self.callosum_metadata = None
            self.call_count = 0

        def generate_with_subliminal_splice(self, prompt: str, chord, **kwargs):
            _ = (prompt, kwargs)
            self.call_count += 1

            class _Trace:
                clause_1_prefix = ""
                post_gap_continuation = self.reply_text
                comma_scored_live = True
                comma_vibe_posture = "settled"
                comma_dissonance_notes: tuple[str, ...] = ()
                spliced_chord = chord

            return self.reply_text, _Trace()

    stub_commit = _StubLeftBrain("During sleep we see that premature f7 sacrifices leave our King exposed.")
    runtime_commit = ChessKibitzerRuntime(
        state_dir=tmp_path / "hybrid_commit_state",
        left_brain=stub_commit,  # type: ignore[arg-type]
        lazy_load_gpu=False,
    )
    runtime_commit.set_auto_rem_sleep(False)
    runtime_commit.handle_set_fen("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 3")
    snap_commit = runtime_commit.handle_chess_ai_move()
    arb_commit = snap_commit["last_hybrid_move_arbitration"]
    assert arb_commit["selection_mode"] == "gear_1_right_brain_intuition"
    assert arb_commit["chosen_move_san"] != "Bxf7+"
    assert stub_commit.call_count == 0

    # Verify that when a 1-ply blunder (`Bxf7+`) is audited and queued for REM sleep,
    # `handle_run_rem_sleep()` wakes the Left Brain to talk to itself (`call_count == 1`)
    # and consolidates a cautionary `hazard_` rule that lowers `Bxf7+`'s Right-Brain score.
    stub_veto = _StubLeftBrain("Reflecting in REM sleep, sacrificing on f7 without support hangs the bishop.")
    runtime_veto = ChessKibitzerRuntime(
        state_dir=tmp_path / "hybrid_veto_state",
        left_brain=stub_veto,  # type: ignore[arg-type]
        lazy_load_gpu=False,
    )
    runtime_veto.set_auto_rem_sleep(False)
    runtime_veto.handle_set_fen("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 3")
    before_bxf7_score = {
        e.san: e.composite_score
        for e in evaluate_candidate_moves_with_right_brain(board, runtime_veto.rules, limit=30)
    }["Bxf7+"]
    blunder_rev = runtime_veto.sparring_engine.audit_played_move_for_game_review(
        board,
        "Bxf7+",
        perspective=chess.WHITE,
    )
    runtime_veto.handle_chess_move("Bxf7+")
    assert stub_veto.call_count == 0
    last_id = int(runtime_veto.chat_messages[-1]["id"])
    runtime_veto.flag_turn_for_coaching(
        last_id,
        user_note=blunder_rev["annotation_summary"],
        check_auto_sleep=False,
        is_hazard=True,
        override_coordinates=blunder_rev["played_coordinates"],
        contrastive_reference_coordinates=blunder_rev["contrastive_reference_coordinates"],
        game_review_annotation=blunder_rev["annotation_summary"],
    )
    assert runtime_veto.sleep_queue
    sleep_snap = runtime_veto.handle_run_rem_sleep()
    assert stub_veto.call_count == 1
    assert sleep_snap["last_rem_sleep_self_talk"] == stub_veto.reply_text
    assert any(r.concept_name.startswith("hazard_") for r in runtime_veto.rules)
    after_bxf7_score = {
        e.san: e.composite_score
        for e in evaluate_candidate_moves_with_right_brain(board, runtime_veto.rules, limit=30)
    }["Bxf7+"]
    assert after_bxf7_score < before_bxf7_score


def test_strict_airgap_sparring_engine_and_objective_elo_update(tmp_path: Path) -> None:
    """Prove `LMFT` never queries the sparring engine for its own moves, and verify objective Elo + color rotation."""
    from lmft_architecture.chess_kibitz_bridge import compute_fide_elo_update

    win_elo, win_delta = compute_fide_elo_update(1000.0, 1000.0, 1.0)
    loss_elo, loss_delta = compute_fide_elo_update(1000.0, 1000.0, 0.0)
    assert win_elo == 1012.0 and win_delta == 12.0
    assert loss_elo == 988.0 and loss_delta == -12.0

    runtime = ChessKibitzerRuntime(
        state_dir=tmp_path / "elo_airgap_state",
        left_brain=_make_mini_left_brain(),
        lazy_load_gpu=False,
    )
    assert runtime.sparring_engine.opponent_moves_played == 0
    assert runtime.sparring_engine.post_move_audits_run == 0

    # Calling `handle_chess_ai_move()` directly must NEVER touch `sparring_engine` (strict anti-cheat air-gap)
    runtime.handle_chess_ai_move()
    assert runtime.sparring_engine.opponent_moves_played == 0
    assert runtime.sparring_engine.post_move_audits_run == 0

    # Reset board and run 2 sparring steps via `handle_microwave_step()`:
    # Step 1 = White (`LMFT`), Step 2 = Black (`SparringEngine`)
    runtime.handle_set_fen("start")
    step_1 = runtime.handle_microwave_step()
    assert step_1["last_microwave_status"]["mover"] == "LMFT"
    assert runtime.sparring_engine.opponent_moves_played == 0

    step_2 = runtime.handle_microwave_step()
    assert step_2["last_microwave_status"]["mover"].startswith("SparringEngine")
    assert runtime.sparring_engine.opponent_moves_played == 1

    # Verify Scholar's Mate win updates Elo to 1012.0, matches `opponent_target_elo`, and rotates `LMFT` to Black
    runtime.handle_set_fen("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    runtime.handle_chess_move("Qxf7#")
    assert runtime.board.is_checkmate() is True
    receipt = runtime._adjudicate_and_record_game_outcome()
    assert receipt["result_label"] == "WIN"
    assert runtime.lmft_elo == 1012.0
    assert runtime.opponent_target_elo == 1012.0
    assert runtime.elo_wins == 1
    assert runtime.lmft_playing_white is False

    # Verify the next `handle_microwave_step()` resets the board for Game #2 WITHOUT double-adjudicating Game #1
    step_next_game = runtime.handle_microwave_step()
    assert runtime.games_completed_count == 1
    assert runtime.lmft_elo == 1012.0
    assert runtime.lmft_playing_white is False
    assert step_next_game["last_microwave_status"]["mover"].startswith("SparringEngine")
    assert len(runtime.move_history_san) == 1


def test_game_review_referee_and_contrastive_fuzzy_rule_normalization(tmp_path: Path) -> None:
    """Verify Chess.com-style post-move Game Review (`??` vs. `!!`/`!`) and contrastive FuzzyConceptRule normalization with 9B Teacher collaboration."""
    from lmft_architecture.chess_kibitz_bridge import (
        ObjectiveCalibratedSparringEngine,
        evaluate_candidate_moves_with_right_brain,
    )
    from lmft_architecture.sleep_consolidation import normalize_contrastive_target_leaves_and_thresholds

    engine = ObjectiveCalibratedSparringEngine(use_stockfish_uci=False)
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 3")

    # 1. Post-move review of a 1-ply hanging sacrifice (`Bxf7+`) classifies as `BLUNDER_??`
    blunder_rev = engine.audit_played_move_for_game_review(
        board,
        "Bxf7+",
        perspective=chess.WHITE,
    )
    assert blunder_rev["classification"] == "BLUNDER_??"
    assert blunder_rev["glyph"] == "??"
    assert blunder_rev["is_hazard"] is True
    assert blunder_rev["should_queue_for_sleep"] is True
    assert engine.opponent_moves_played == 0

    # 2. Contrastive normalization between `Bxf7+` (played blunder) and `best_move_coordinates`
    #    must normalize `[rule_low, rule_high]` so the rule fires at 1.0 on the blunder and 0.0 on the best move
    target_leaves, r_low, r_high = normalize_contrastive_target_leaves_and_thresholds(
        "low_wastelessness_chaos_cluster",
        blunder_rev["played_coordinates"],
        blunder_rev["contrastive_reference_coordinates"],
    )
    from lmft_architecture.reflection_classifier import FuzzyConceptRule

    norm_hazard_rule = FuzzyConceptRule(
        rule_id="test-contrastive-hazard",
        concept_name="hazard_unsound_bishop_sac",
        description="Contrastive-normalized hazard rule separating unsound sacrifice from sound development.",
        target_leaves=target_leaves,
        low=r_low,
        high=r_high,
    )
    assert norm_hazard_rule.evaluate(blunder_rev["played_coordinates"]) >= 0.85
    assert norm_hazard_rule.evaluate(blunder_rev["contrastive_reference_coordinates"]) == 0.0

    # 3. Post-move review of a winning undefended knight capture classifies as `BRILLIANCY_!!` or `GREAT_MOVE_!`
    tactical_board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4n3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 4")
    brill_rev = engine.audit_played_move_for_game_review(
        tactical_board,
        "Nxe5",
        perspective=chess.WHITE,
    )
    assert brill_rev["classification"] in ("BRILLIANCY_!!", "GREAT_MOVE_!")
    assert brill_rev["is_hazard"] is False
    assert brill_rev["should_queue_for_sleep"] is True

    # 4. Verify 9B Sleep Teacher collaboration in `ChessKibitzerRuntime.handle_run_rem_sleep()`
    captured_payloads: list[dict] = []

    class _StubSleepTeacher:
        def generate(self, role: str, req: dict, tag: str, validator):
            _ = (role, tag)
            user_msg = json.loads(req["messages"][1]["content"])
            captured_payloads.append(user_msg)
            raw_json = json.dumps(
                {
                    "concept_name": "reckless_unanchored_exposure",
                    "description": "Over-extending aggressive force while leaving core stability undefended.",
                    "executive_clause_text": "Restrain premature expenditure when foundational protection is absent.",
                    "legislative_clause_text": "Sense the brittle tension of an unguarded structural commitment.",
                    "judicial_clause_text": "Verify that defensive continuity holds before committing decisive agency.",
                }
            )
            return validator(raw_json), {}

    runtime = ChessKibitzerRuntime(
        state_dir=tmp_path / "teacher_collab_state",
        left_brain=_make_mini_left_brain(),
        teacher=_StubSleepTeacher(),
        lazy_load_gpu=False,
    )
    runtime.set_auto_rem_sleep(False)
    runtime.handle_set_fen("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 3")
    before_evals = {
        e.san: e.composite_score
        for e in evaluate_candidate_moves_with_right_brain(runtime.board, runtime.rules, limit=30)
    }
    runtime.handle_chess_move("Bxf7+")
    last_id = int(runtime.chat_messages[-1]["id"])
    runtime.flag_turn_for_coaching(
        last_id,
        user_note=blunder_rev["annotation_summary"],
        check_auto_sleep=False,
        is_hazard=True,
        override_coordinates=blunder_rev["played_coordinates"],
        contrastive_reference_coordinates=blunder_rev["contrastive_reference_coordinates"],
        game_review_annotation=blunder_rev["annotation_summary"],
    )
    runtime.handle_run_rem_sleep()

    assert len(captured_payloads) == 1
    assert "contrastive_coordinate_delta_vs_best_alternative" in captured_payloads[0]
    assert "objective_referee_annotations" in captured_payloads[0]
    forged_hazard = [r for r in runtime.rules if r.concept_name.startswith("hazard_")]
    assert len(forged_hazard) == 1
    assert forged_hazard[0].concept_name == "hazard_reckless_unanchored_exposure"

    # Reset board to before `Bxf7+` and verify `Bxf7+` is strongly repelled while `O-O` is un-penalized
    runtime.handle_set_fen("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 3")
    after_evals = {
        e.san: e.composite_score
        for e in evaluate_candidate_moves_with_right_brain(runtime.board, runtime.rules, limit=30)
    }
    assert after_evals["Bxf7+"] < before_evals["Bxf7+"] - 0.50
    assert after_evals["O-O"] >= before_evals["O-O"] - 0.01

    # 5. Verify plateau guard when `v_act = 0.66 < v_ref = 0.85` (prevents `>= 0.65` plateau from erasing delta)
    plateau_targets, p_low, p_high = normalize_contrastive_target_leaves_and_thresholds(
        "moderate_cultural_structure_cluster",
        {"philosophical/being/existence": 0.66, "philosophical/structure/order": 0.66},
        {"philosophical/being/existence": 0.86, "philosophical/structure/order": 0.86},
    )
    plateau_rule = FuzzyConceptRule(
        rule_id="test-plateau-guard",
        concept_name="hazard_plateau_test",
        description="Verify upper plateau guard keeps target<=0.64 when v_act < v_ref.",
        target_leaves=plateau_targets,
        low=p_low,
        high=p_high,
    )
    assert plateau_rule.evaluate({"philosophical/being/existence": 0.66, "philosophical/structure/order": 0.66}) >= 0.85
    assert plateau_rule.evaluate({"philosophical/being/existence": 0.86, "philosophical/structure/order": 0.86}) == 0.0

    # 6. Verify `handle_microwave_step()` directly attaches `game_review` telemetry
    mw_snap = runtime.handle_microwave_step()
    assert "game_review" in mw_snap["chat_messages"][-1]["telemetry"]


def test_master_seed_sync_and_human_player_rating_coaching(tmp_path: Path) -> None:
    """Verify master seed export/merge across GitHub releases and live human player rating + custom Fuzzy-Tree coaching."""
    from lmft_architecture.chess_kibitz_bridge import (
        compute_estimated_game_elo,
        compute_move_accuracy_percent,
    )
    from lmft_architecture.github_auto_sync import is_protected_user_state_path
    from lmft_architecture.reflection_classifier import FuzzyConceptRule

    assert is_protected_user_state_path("outputs/rem_sleep_consolidated_ruleset.json") is False
    assert is_protected_user_state_path("outputs/sleep_consolidated_ruleset.json") is False
    assert is_protected_user_state_path("outputs/chess_kibitzer_state/club_ruleset.json") is True

    assert compute_move_accuracy_percent(0.0) == 100.0
    assert compute_move_accuracy_percent(200.0) < 30.0
    assert compute_estimated_game_elo(90.0, reference_elo=1000.0) > 1150.0

    # 1. Owner bakes a new rule in `owner_runtime` and exports it to the shared release seed
    from lmft_architecture.sleep_consolidation import normalize_contrastive_target_leaves_and_thresholds

    owner_runtime = ChessKibitzerRuntime(
        state_dir=tmp_path / "owner_state",
        left_brain=_make_mini_left_brain(),
        lazy_load_gpu=False,
    )
    sample_board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 3")
    sample_rev = owner_runtime.sparring_engine.audit_played_move_for_game_review(
        sample_board,
        "Bxf7+",
        perspective=chess.WHITE,
    )
    t_leaves, r_low, r_high = normalize_contrastive_target_leaves_and_thresholds(
        "low_wastelessness_chaos_cluster",
        sample_rev["played_coordinates"],
        sample_rev["contrastive_reference_coordinates"],
    )
    owner_baked_rule = FuzzyConceptRule(
        rule_id="rule-owner-oven-1",
        concept_name="hazard_premature_f7_sacrifice",
        description="Avoid premature bishop sacrifice on f7 when King is uncastled.",
        target_leaves=t_leaves,
        low=r_low,
        high=r_high,
    )
    owner_runtime.rules = (*owner_runtime.rules, owner_baked_rule)
    seed_root = tmp_path / "shared_release_seed"
    export_info = owner_runtime.export_master_release_seed(target_training_root=seed_root)
    assert export_info["exported_rules_count"] == len(owner_runtime.rules)
    assert (seed_root / "outputs" / "rem_sleep_consolidated_ruleset.json").exists()

    # 2. A downloader with their own local runtime merges the shared release seed non-destructively
    player_runtime = ChessKibitzerRuntime(
        state_dir=tmp_path / "player_state",
        left_brain=_make_mini_left_brain(),
        lazy_load_gpu=False,
    )
    player_custom_rule = FuzzyConceptRule(
        rule_id="rule-player-local-1",
        concept_name="coached_patient_king_safety",
        description="Personal rule saved by the local player.",
        target_leaves={"philosophical/being/existence": 0.64},
        low=0.40,
        high=0.70,
    )
    player_runtime.rules = (*player_runtime.rules[:2], player_custom_rule)
    merge_res = player_runtime.merge_master_seed_rules(seed_training_root=seed_root)
    assert merge_res["merged_rules_count"] >= 1
    rule_names = {r.concept_name for r in player_runtime.rules}
    assert "coached_patient_king_safety" in rule_names
    assert "hazard_premature_f7_sacrifice" in rule_names

    # 3. Human plays a blunder (`Bxf7+`) against `player_runtime`:
    #    AI assigns live move accuracy, estimated performance Elo, and custom coaching citing the synced rule!
    player_runtime.handle_set_fen("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 3")
    snap_blunder = player_runtime.handle_chess_move("Bxf7+")
    coaching = snap_blunder["elo"]["last_player_coaching"]
    assert coaching is not None
    assert coaching["classification"] == "BLUNDER_??"
    assert coaching["move_accuracy_pct"] < 35.0
    assert coaching["estimated_performance_elo"] < 700.0
    assert coaching["matched_wisdom_rule"] == "hazard_premature_f7_sacrifice"
    assert "Stronger was" in coaching["coaching_advice"]

    # 4. Human wins a game (Scholar's Mate) and `adjudicate_human_game_outcome()` updates `player_elo`
    player_runtime.handle_set_fen("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    player_runtime.handle_chess_move("Qxf7#")
    p_receipt = player_runtime.adjudicate_human_game_outcome()
    assert p_receipt["result_label"] == "WIN"
    assert player_runtime.player_elo == 1012.0
    assert player_runtime.player_wins == 1

    # 5. Human plays a non-first legal move (`e4`, whereas `legal[0]` is `Nh3`) and coaching grades `e4` as SOLID (not BRILLIANCY_!!)
    player_runtime.handle_set_fen("start")
    snap_e4 = player_runtime.handle_chess_move("e4")
    assert snap_e4["elo"]["last_player_coaching"]["played_san"] == "e4"
    assert snap_e4["elo"]["last_player_coaching"]["classification"] == "SOLID"
    assert "eval_white_cp" in snap_e4["board"]
    snap_e5 = player_runtime.handle_chess_move("e5")
    assert snap_e5["elo"]["last_player_coaching"]["played_san"] == "e5"
    assert snap_e5["elo"]["last_player_coaching"]["classification"] == "SOLID"

    # 6. Verify Chess.com-style 3-column UI, Eval Bar, 2-Button workflow, Jukebox, Space Center Tour, and Auto-Update exist in HTML
    for required_ui_token in (
        'id="eval-bar-container"',
        'id="mode-play-vs-ai-btn"',
        'id="auto-self-play-btn"',
        'id="new-user-tutorial-btn"',
        'id="tutorial-modal-overlay"',
        'id="space-center-tour-btn"',
        'id="space-center-modal-overlay"',
        'id="celestriad-jukebox"',
        'id="jukebox-mute-btn"',
        'id="game-review-tally-box"',
        "playHumanMoveAndAutoReply",
        "syncAndAutoUpdate",
        'isDest ? "legal-dest" : ""',
        "Celestriad 1.0",
        "Stockfish 19 (",
        "88CuKNXFAq8",
        "kpZFlSJS9ZI",
        "Ys-oKc88tXY",
        "oDEpcYBu7eU",
        "/assets/celestial_seal_poem.jpg",
    ):
        assert required_ui_token in CHESS_SINGLE_PAGE_HTML
    assert "LMFT_Grandmaster" not in CHESS_SINGLE_PAGE_HTML

    # 7. When LMFT delivers checkmate on its AI turn (`Qh4#` in Fool's Mate), `handle_chess_ai_move`
    #    automatically adjudicates the human's loss and updates `player_elo` and `player_losses`
    player_runtime.handle_set_fen("rnbqkbnr/pppp1ppp/8/4p3/8/5P2/PPPPP1PP/RNBQKBNR w KQkq - 0 2")
    player_runtime.handle_chess_move("g4")
    player_runtime.handle_chess_ai_move()
    assert player_runtime.board.is_checkmate() is True
    assert player_runtime.player_losses == 1
    assert player_runtime.player_elo < 1012.0

    # 8. Regression check: positions where candidate moves trigger `ShadowPathologyWarning` (`w.shadow_pathology`, `w.severity`)
    #    must evaluate cleanly without AttributeError during `handle_microwave_step()`
    player_runtime.handle_set_fen("r1b1k2r/1p3ppp/p7/2qpN3/5B2/4B3/PPP1PPPP/R2QKB1R w KQkq - 0 11")
    mw_shadow = player_runtime.handle_microwave_step()
    assert len(mw_shadow["board"]["move_history_san"]) == 1

    # 9. Regression check: `_ensure_foundational_rules` collapses `_cluster_s<N>` suffixes into their canonical
    #    cluster rule and entrains `target_leaves` (`0.7 * old + 0.3 * new`) rather than multiplying rule counts
    from lmft_architecture.chess_app import _ensure_foundational_rules

    dup_s52 = FuzzyConceptRule(
        rule_id="rule-sleep-s52-hazard_low_wastelessness",
        concept_name="hazard_low_wastelessness_chaos_cluster_s52",
        description="First blunder observation.",
        target_leaves={"cultural/efficiency/wastelessness": 0.25, "philosophical/structure/chaos": 0.60},
        low=0.60,
        high=0.80,
    )
    dup_s82 = FuzzyConceptRule(
        rule_id="rule-sleep-s82-hazard_low_wastelessness",
        concept_name="hazard_low_wastelessness_chaos_cluster_s82",
        description="Second blunder observation in same cluster.",
        target_leaves={"cultural/efficiency/wastelessness": 0.20, "philosophical/structure/chaos": 0.70},
        low=0.70,
        high=0.90,
    )
    collapsed = _ensure_foundational_rules((dup_s52, dup_s82))
    haz_clusters = [r for r in collapsed if r.concept_name.startswith("hazard_low_wastelessness_chaos_cluster")]
    assert len(haz_clusters) == 1
    assert haz_clusters[0].concept_name == "hazard_low_wastelessness_chaos_cluster"
    assert haz_clusters[0].target_leaves["philosophical/structure/chaos"] == pytest.approx(0.63)

    # 10. Regression check: poisoned `hazard_high_wastelessness_*` / Brilliancy-labeled hazard rules
    #     and self-contradictory `Played X instead of best move X` rules are stripped
    poisoned_brill = FuzzyConceptRule(
        rule_id="rule-sleep-hazard_high_wastelessness_chaos_",
        concept_name="hazard_high_wastelessness_chaos_cluster",
        description="Game Review [!! BRILLIANCY_!!]: Found Rd1 (+2.53 pawns vs. alternatives).",
        target_leaves={"cultural/efficiency/wastelessness": 0.25, "philosophical/structure/order": 0.71},
        low=0.835,
        high=0.9642,
    )
    self_contradictory_mistake = FuzzyConceptRule(
        rule_id="rule-sleep-s260-hazard_high_self_sufficiency",
        concept_name="hazard_high_self_sufficiency_chaos_cluster",
        description="Consolidated REM chess wisdom rule: Game Review [? MISTAKE_?]: Played Qxe3 (-1.03 pawns) instead of best move Qxe3 (wastelessness -0.47).",
        target_leaves={"cultural/efficiency/wastelessness": 0.25, "cultural/freedom/self-sufficiency": 0.78},
        low=0.72,
        high=0.9295,
    )
    cleaned_rules = _ensure_foundational_rules((dup_s52, poisoned_brill, self_contradictory_mistake))
    assert all(r.concept_name != "hazard_high_wastelessness_chaos_cluster" for r in cleaned_rules)
    assert all(r.concept_name != "hazard_high_self_sufficiency_chaos_cluster" for r in cleaned_rules)

    # 11. Regression check: `normalize_contrastive_target_leaves_and_thresholds` and `_ensure_foundational_rules`
    #     prevent all 6 `PHILOSOPHICAL_SHADOW_LEAVES` from entering `coached_` rules and prevent high virtue leaves
    #     or low shadow leaves from entering `hazard_` rules.
    from lmft_architecture.sleep_consolidation import normalize_contrastive_target_leaves_and_thresholds

    coached_targets, _, _ = normalize_contrastive_target_leaves_and_thresholds(
        {"cultural/efficiency/wastelessness": 0.90},
        {
            "cultural/efficiency/wastelessness": 0.92,
            "philosophical/persistence/mania_greed": 0.85,
            "philosophical/being/paranoia": 0.78,
            "philosophical/structure/prejudice": 0.75,
            "philosophical/persistence/narcissism": 0.80,
            "philosophical/structure/order": 0.82,
        },
        {
            "cultural/efficiency/wastelessness": 0.25,
            "philosophical/persistence/mania_greed": 0.20,
            "philosophical/being/paranoia": 0.20,
            "philosophical/structure/prejudice": 0.20,
            "philosophical/persistence/narcissism": 0.20,
            "philosophical/structure/order": 0.40,
        },
    )
    for shadow_key in (
        "philosophical/persistence/mania_greed",
        "philosophical/being/paranoia",
        "philosophical/structure/prejudice",
        "philosophical/persistence/narcissism",
    ):
        assert shadow_key not in coached_targets
    assert "philosophical/structure/order" in coached_targets

    hazard_targets, _, _ = normalize_contrastive_target_leaves_and_thresholds(
        {"cultural/efficiency/wastelessness": 0.25},
        {
            "cultural/efficiency/wastelessness": 0.25,
            "philosophical/being/existence": 0.62,
            "philosophical/being/psychosis": 0.35,
            "philosophical/structure/chaos": 0.78,
        },
        {
            "cultural/efficiency/wastelessness": 0.92,
            "philosophical/being/existence": 0.90,
            "philosophical/being/psychosis": 0.73,
            "philosophical/structure/chaos": 0.30,
        },
    )
    assert "philosophical/being/existence" not in hazard_targets
    assert "philosophical/being/psychosis" not in hazard_targets
    assert "philosophical/structure/chaos" in hazard_targets


def test_bitboard_sensor_psychological_will_execution_bids_and_rem_distillation(tmp_path: Path) -> None:
    """Verify Bitboard `Fact`/`Theory` sensor, `BitboardIntuitionSleeve` `Confidence`, `Execution` Action Bids, and REM Sleep distillation."""
    from lmft_architecture.chess_kibitz_bridge import (
        BitboardIntuitionSleeve,
        compute_candidate_move_fuzzy_coordinates,
        evaluate_candidate_moves_with_right_brain,
    )

    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 3")
    sleeve = BitboardIntuitionSleeve()

    coords_oo, blun_oo, _, _, _ = compute_candidate_move_fuzzy_coordinates(
        board,
        board.parse_san("O-O"),
        bitboard_sleeve=sleeve,
    )
    coords_bxf7, blun_bxf7, _, _, _ = compute_candidate_move_fuzzy_coordinates(
        board,
        board.parse_san("Bxf7+"),
        bitboard_sleeve=sleeve,
    )
    assert blun_oo is False and blun_bxf7 is True
    # 1. `psychological/truth/fact`, `theory`, `confidence`, `intention`, and `execution` vary dynamically across moves
    assert coords_oo["psychological/truth/fact"] > coords_bxf7["psychological/truth/fact"] + 0.25
    assert coords_oo["psychological/contemplation/confidence"] > coords_bxf7["psychological/contemplation/confidence"]
    assert coords_oo["psychological/will/execution"] > coords_bxf7["psychological/will/execution"]

    # 2. Evaluate all legal moves as competing Action Bids on `psychological/will/execution`
    before_bids = {
        e.san: e
        for e in evaluate_candidate_moves_with_right_brain(
            board,
            limit=30,
            bitboard_sleeve=sleeve,
        )
    }
    assert before_bids["O-O"].execution_bid > before_bids["Bxf7+"].execution_bid
    assert before_bids["O-O"].fact > before_bids["Bxf7+"].fact
    assert "execution_bid" in before_bids["O-O"].to_dict()
    assert "fact" in before_bids["O-O"].to_dict()
    assert "theory" in before_bids["O-O"].to_dict()

    # 3. Verify strict pre-move air-gap + post-mortem REM Sleep distillation of both Fuzzy Rules & Bitboard Sleeve
    runtime = ChessKibitzerRuntime(
        state_dir=tmp_path / "bitboard_will_state",
        left_brain=_make_mini_left_brain(),
        lazy_load_gpu=False,
    )
    runtime.set_auto_rem_sleep(False)
    runtime.handle_set_fen("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 3")
    runtime.handle_chess_ai_move()
    assert runtime.sparring_engine.opponent_moves_played == 0
    assert runtime.sparring_engine.post_move_audits_run == 0

    runtime.handle_set_fen("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 3")
    pre_sleep_evals = {
        e.san: e
        for e in evaluate_candidate_moves_with_right_brain(
            runtime.board,
            runtime.rules,
            limit=30,
            bitboard_sleeve=runtime.bitboard_sleeve,
        )
    }
    rev = runtime.sparring_engine.audit_played_move_for_game_review(
        runtime.board,
        "Bxf7+",
        perspective=chess.WHITE,
    )
    runtime.handle_chess_move("Bxf7+")
    last_id = int(runtime.chat_messages[-1]["id"])
    runtime.flag_turn_for_coaching(
        last_id,
        user_note=rev["annotation_summary"],
        check_auto_sleep=False,
        is_hazard=True,
        override_coordinates=rev["played_coordinates"],
        contrastive_reference_coordinates=rev["contrastive_reference_coordinates"],
        game_review_annotation=rev["annotation_summary"],
    )
    runtime.handle_run_rem_sleep()

    assert runtime.bitboard_sleeve.steps_distilled >= 1
    assert (tmp_path / "bitboard_will_state" / "bitboard_sleeve_weights.json").exists()

    runtime.handle_set_fen("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 3")
    post_sleep_evals = {
        e.san: e
        for e in evaluate_candidate_moves_with_right_brain(
            runtime.board,
            runtime.rules,
            limit=30,
            bitboard_sleeve=runtime.bitboard_sleeve,
        )
    }
    assert post_sleep_evals["Bxf7+"].execution_bid < pre_sleep_evals["Bxf7+"].execution_bid - 0.50
    assert post_sleep_evals["O-O"].execution_bid > pre_sleep_evals["O-O"].execution_bid


def test_intention_to_theory_and_confidence_rollout_pipeline() -> None:
    """Verify `Intention (Ply 1) -> Theory (Ply 2-3 Rollout) -> Confidence -> Execution` refutes 2-ply mate/fork traps."""
    # 1. 2-ply back-rank forced mate (`a4?? -> Rf1+! Rxf1 Rxf1#`): `h3` creates luft and parries it, while `a4` fails the Theory rollout
    two_ply_mate_board = chess.Board("5r1k/5rpp/8/8/8/P7/1P4PP/4R2K w - - 0 50")
    evals_mate = {
        e.san: e
        for e in evaluate_candidate_moves_with_right_brain(
            two_ply_mate_board,
            limit=25,
        )
    }
    assert evals_mate["a4"].is_blunder is True
    assert evals_mate["h3"].is_blunder is False
    assert evals_mate["h3"].theory > evals_mate["a4"].theory
    assert evals_mate["h3"].confidence > evals_mate["a4"].confidence
    assert evals_mate["h3"].clarity > evals_mate["a4"].clarity
    assert evals_mate["h3"].execution_bid > evals_mate["a4"].execution_bid

    # 2. 2-ply Knight royal fork (`a4?? -> Nc2+! Kd2 Nxa1`): `Rd1` parries it, while `a4` fails the Theory rollout
    royal_fork_board = chess.Board("4k3/8/8/8/3n4/P6P/1P4P1/R3K3 w - - 0 30")
    evals_fork = {
        e.san: e
        for e in evaluate_candidate_moves_with_right_brain(
            royal_fork_board,
            limit=25,
        )
    }
    assert evals_fork["a4"].is_blunder is True
    assert evals_fork["Rd1"].is_blunder is False
    assert evals_fork["Rd1"].theory > evals_fork["a4"].theory
    assert evals_fork["Rd1"].confidence > evals_fork["a4"].confidence
    assert evals_fork["Rd1"].clarity > evals_fork["a4"].clarity
    assert evals_fork["Rd1"].execution_bid > evals_fork["a4"].execution_bid

    # 3. Post-move Game Review (`audit_played_move_for_game_review`) also enriches `played_coordinates`
    #    and `best_move_coordinates` with the rolled-out `theory` and `confidence` coordinates for REM Sleep
    from lmft_architecture.chess_kibitz_bridge import ObjectiveCalibratedSparringEngine

    referee = ObjectiveCalibratedSparringEngine(use_stockfish_uci=False)
    rev_fork = referee.audit_played_move_for_game_review(
        royal_fork_board,
        "a4",
        perspective=chess.WHITE,
    )
    assert rev_fork["classification"] == "BLUNDER_??"
    assert rev_fork["is_hazard"] is True
    assert (
        rev_fork["best_move_coordinates"]["psychological/truth/theory"]
        > rev_fork["played_coordinates"]["psychological/truth/theory"]
    )
    assert (
        rev_fork["best_move_coordinates"]["psychological/contemplation/confidence"]
        > rev_fork["played_coordinates"]["psychological/contemplation/confidence"]
    )


def test_speaker_attribution_and_clean_slate_reset(tmp_path: Path) -> None:
    """Verify routine opening moves stay in quiet Gear 1, tactical crossroads wake Gear 2/3 with 2-sentence deliberation, Stockfish bubbles carry referee reviews, and `reset_learned_chess_state` wipes tainted rules."""
    left_brain = _make_mini_left_brain()
    runtime = ChessKibitzerRuntime(
        state_dir=tmp_path / "speaker_clean_slate",
        left_brain=left_brain,
        lazy_load_gpu=False,
    )
    # 1. Step 1 (Celestriad as White on calm Move 1): settles quietly in Gear 1 without forced Left-Brain banter
    s1 = runtime.handle_microwave_step()
    msg_cel = s1["chat_messages"][-1]
    assert msg_cel["sender"] == "companion"
    assert "Celestriad 1.0 (White" in msg_cel["role_label"]
    assert runtime.last_hybrid_move_arbitration["selection_mode"] == "gear_1_right_brain_intuition"
    assert "Gear 1 Right-Brain intuition" in msg_cel["text"]
    assert msg_cel["is_gear_1_fallback"] is True
    assert msg_cel["telemetry"]["fact_coordinates"]["cultural/efficiency/power"] >= 0.35

    # 2. Step 2 (Stockfish as Black): sender is `sparring_opponent`, text clearly states Stockfish played its move,
    #    Stockfish's bubble never contains Celestriad quotes, and carries `sparring_referee_review`
    s2 = runtime.handle_microwave_step()
    msg_sf = s2["chat_messages"][-1]
    assert msg_sf["sender"] == "sparring_opponent"
    assert "Stockfish 19 (" in msg_sf["role_label"]
    assert msg_sf["text"].startswith("Stockfish 19 (")
    assert "Position anchored on " not in msg_sf["text"]
    assert "sparring_referee_review" in msg_sf["telemetry"]
    assert runtime._extract_clean_conversation_history() == ()

    # 3. Even on a Move 3+ contested tactical crossroads, board moves (`handle_microwave_step`) stay in
    #    silent, fast Gear-1 Right-Brain mode without per-move Left-Brain chatter; when an actual human
    #    asks an insightful question (`handle_chat_message`), Gear 2/3 wakes up with multi-sentence `<|subliminal_gap|>` self-talk.
    runtime.handle_set_fen("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 3")
    assert runtime.white_clock_sec == 300.0 and runtime.black_clock_sec == 300.0
    s3 = runtime.handle_microwave_step()
    msg_move = s3["chat_messages"][-1]
    assert msg_move["sender"] == "companion"
    assert msg_move["is_gear_1_fallback"] is True
    assert runtime.last_hybrid_move_arbitration["selection_mode"] == "gear_1_right_brain_intuition"
    assert "White to move" in msg_move["clean_pretext_prompt"] or "Black to move" in msg_move["clean_pretext_prompt"]
    assert "FEN" not in msg_move["clean_pretext_prompt"]
    assert s3["board"]["white_clock_sec"] <= 300.0
    assert runtime._extract_clean_conversation_history() == ()

    s_human = runtime.handle_chat_message(
        "Why does sacrificing the bishop on f7 fail to break the king's shelter in this position?",
        room="chess",
        anchor_to_board=True,
    )
    msg_delib = s_human["chat_messages"][-1]
    assert msg_delib["sender"] == "companion"
    assert msg_delib["telemetry"]["insightful_inquiry_deliberation"] is True
    assert msg_delib["telemetry"]["spliced_trace"].get("clause_1_prefix") == ""
    assert msg_delib["telemetry"]["spliced_trace"].get("sentence_count", 0) >= 2
    assert msg_delib["telemetry"]["spliced_trace"].get("comma_scored_live") is True
    hist_pairs = runtime._extract_clean_conversation_history()
    assert len(hist_pairs) == 1

    # Short non-question remarks stay at 1 sentence (`insightful_inquiry_deliberation is False`)
    s_short = runtime.handle_chat_message("Good game so far.", room="chess", anchor_to_board=True)
    assert s_short["chat_messages"][-1]["telemetry"]["insightful_inquiry_deliberation"] is False
    assert s_short["chat_messages"][-1]["telemetry"]["num_sentences"] == 1
    assert "auto_rem_sleep_receipt" in CHESS_SINGLE_PAGE_HTML

    # Early-ply Gear-1 Right-Brain move selection rotates among nearly-tied sound openings across games
    runtime.handle_set_fen("start")
    runtime.games_completed_count = 0
    mv_g0 = runtime.handle_chess_ai_move()["last_hybrid_move_arbitration"]["chosen_move_san"]
    runtime.handle_set_fen("start")
    runtime.games_completed_count = 1
    mv_g1 = runtime.handle_chess_ai_move()["last_hybrid_move_arbitration"]["chosen_move_san"]
    assert mv_g0 != mv_g1

    # 4. If Celestriad's 5:00 clock runs out (0:00), no hardcoded instinct bailout saves it;
    #    it forfeits on time, queues the time-forfeit turn into `sleep_queue`, talks to itself
    #    during REM sleep, and consolidates a permanent clock-persistence hazard rule.
    sleep_q_before = len(runtime.sleep_queue)
    runtime.white_clock_sec = 0.0
    runtime.clock_flagged_loser = chess.WHITE
    outcome = runtime._adjudicate_and_record_game_outcome()
    assert outcome is not None
    assert "Time Forfeit 0:00" in outcome["result_label"]
    assert len(runtime.sleep_queue) == sleep_q_before + 1
    flagged_fork = runtime.sleep_queue[-1]
    assert flagged_fork.external_coordinates["philosophical/persistence/continuity"] < 0.15
    assert flagged_fork.contrastive_reference_coordinates["philosophical/persistence/continuity"] >= 0.50
    assert flagged_fork.contrastive_reference_coordinates["psychological/will/execution"] > 0.80
    assert "psychological/will/hesitation" not in flagged_fork.external_coordinates
    assert "psychological/will/hesitation" not in flagged_fork.contrastive_reference_coordinates

    rem_snap = runtime.handle_run_rem_sleep()
    assert rem_snap["last_rem_sleep_self_talk"] is not None
    assert "REM Dream Self-Reflection:" in rem_snap["last_rem_sleep_receipt"]
    assert any(
        r.concept_name.startswith("hazard_")
        and "philosophical/persistence/continuity" in r.target_leaves
        and "philosophical/persistence/discreteness" in r.target_leaves
        for r in runtime.rules
    )

    # 5. Clean slate reset restores foundational rules, resets BitboardIntuitionSleeve, clocks, & Elo to 1000.0
    runtime.lmft_elo = 1156.0
    runtime.bitboard_sleeve.steps_distilled = 15
    runtime.rules = (
        *runtime.rules,
        FuzzyConceptRule(
            rule_id="rule-tainted-1",
            concept_name="hazard_tainted_test_cluster",
            description="Tainted rule to wipe.",
            target_leaves={"cultural/efficiency/wastelessness": 0.20, "philosophical/structure/chaos": 0.70},
        ),
    )
    snap_reset = runtime.reset_learned_chess_state()
    assert snap_reset["elo"]["lmft_elo"] == 1000.0
    assert runtime.bitboard_sleeve.steps_distilled == 0
    assert all(not r.concept_name.startswith(("hazard_", "coached_")) for r in runtime.rules)
    assert snap_reset["rem_dream_chronicles"] == []


def test_longitudinal_rem_dream_chronicles_and_coordinate_vow_keeping(tmp_path: Path) -> None:
    """Verify multi-stanza REM dream chronicles and pure coordinate vow-tracking (`NEW_VOW` -> `BROKEN` -> `HONORED`)."""
    state_dir = tmp_path / "dream_chronicles_state"
    left_brain = _make_mini_left_brain()
    runtime = ChessKibitzerRuntime(
        state_dir=state_dir,
        left_brain=left_brain,
        lazy_load_gpu=False,
    )
    runtime.set_auto_rem_sleep(False)

    # Cycle 1: Queue two distinct tactical scenes so Deep REM Sleep generates a 2-stanza Match Chronicle (`NEW_VOW`)
    runtime.handle_set_fen("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 3")
    blunder_rev = runtime.sparring_engine.audit_played_move_for_game_review(
        runtime.board,
        "Bxf7+",
        perspective=chess.WHITE,
    )
    runtime.handle_chess_move("Bxf7+")
    id_1 = int(runtime.chat_messages[-1]["id"])
    runtime.flag_turn_for_coaching(
        id_1,
        user_note=blunder_rev["annotation_summary"],
        check_auto_sleep=False,
        is_hazard=True,
        override_coordinates=blunder_rev["played_coordinates"],
        contrastive_reference_coordinates=blunder_rev["contrastive_reference_coordinates"],
        game_review_annotation=blunder_rev["annotation_summary"],
    )
    runtime.handle_chess_move("Kxf7")
    id_2 = int(runtime.chat_messages[-1]["id"])
    runtime.flag_turn_for_coaching(
        id_2,
        user_note="King shelter disrupted after recapture.",
        check_auto_sleep=False,
        is_hazard=False,
    )

    snap_c1 = runtime.handle_run_rem_sleep()
    chronicles_1 = snap_c1["rem_dream_chronicles"]
    assert len(chronicles_1) == 1
    assert chronicles_1[0]["vow_outcome"] == "NEW_VOW"
    assert len(chronicles_1[0]["stanzas"]) == 2
    assert "\n\n" in (snap_c1["last_rem_sleep_self_talk"] or "")
    assert len(chronicles_1[0]["vowed_leaves"]) >= 1
    assert chronicles_1[0]["vow_sentence"]

    # Cycle 2: Repeat the same blunder on the vowed coordinate leaves -> `vow_outcome == "BROKEN"`
    runtime.handle_set_fen("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 3")
    runtime.handle_chess_move("Bxf7+")
    id_3 = int(runtime.chat_messages[-1]["id"])
    runtime.flag_turn_for_coaching(
        id_3,
        user_note=blunder_rev["annotation_summary"],
        check_auto_sleep=False,
        is_hazard=True,
        override_coordinates=blunder_rev["played_coordinates"],
        contrastive_reference_coordinates=blunder_rev["contrastive_reference_coordinates"],
        game_review_annotation=blunder_rev["annotation_summary"],
    )
    snap_c2 = runtime.handle_run_rem_sleep()
    chronicles_2 = snap_c2["rem_dream_chronicles"]
    assert len(chronicles_2) == 2
    assert chronicles_2[-1]["vow_outcome"] == "BROKEN"
    assert len(chronicles_2[-1]["vow_overlap_leaves"]) >= 1
    assert "Longitudinal Dream Vow Status: BROKEN" in snap_c2["last_rem_sleep_receipt"]

    # Cycle 3: Play a sound castling move (`O-O`) with `is_hazard=False` -> `vow_outcome == "HONORED"`
    runtime.handle_set_fen("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 3")
    sound_rev = runtime.sparring_engine.audit_played_move_for_game_review(
        runtime.board,
        "O-O",
        perspective=chess.WHITE,
    )
    runtime.handle_chess_move("O-O")
    id_4 = int(runtime.chat_messages[-1]["id"])
    runtime.flag_turn_for_coaching(
        id_4,
        user_note="Castled safely to honor King shelter.",
        check_auto_sleep=False,
        is_hazard=False,
        override_coordinates=sound_rev["played_coordinates"],
        contrastive_reference_coordinates=sound_rev["contrastive_reference_coordinates"],
        game_review_annotation="[! GREAT_MOVE]: Castled safely.",
    )
    snap_c3 = runtime.handle_run_rem_sleep()
    chronicles_3 = snap_c3["rem_dream_chronicles"]
    assert len(chronicles_3) == 3
    assert chronicles_3[-1]["vow_outcome"] == "HONORED"
    assert "Longitudinal Dream Vow Status: HONORED" in snap_c3["last_rem_sleep_receipt"]

    # Verify persistence across restart in `club_meta.json`
    reloaded = ChessKibitzerRuntime(
        state_dir=state_dir,
        left_brain=left_brain,
        lazy_load_gpu=False,
    )
    assert len(reloaded.rem_dream_chronicles) == 3
    assert reloaded.rem_dream_chronicles[-1]["vow_outcome"] == "HONORED"
    assert len(reloaded._get_active_vow_leaves()) >= 1

    # Verify live 3-branch archive recall and vow-guided move selection
    res_check = reloaded.archives.recall_resonant_clauses(
        sound_rev["played_coordinates"],
        min_similarity=0.40,
    )
    assert res_check["top_resonance_score"] >= 0.40
    reloaded.handle_set_fen("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 3")
    ai_vow_snap = reloaded.handle_chess_ai_move()
    arb = ai_vow_snap["last_hybrid_move_arbitration"]
    assert "vow_alignment_delta" in arb
    assert "guided_by_vow" in arb
    assert len(arb["active_vow_leaves"]) >= 1

    # Verify multi-stanza chat exports `stanza_chord_trajectory` and `archive_resonance` in telemetry
    chat_stanza_snap = reloaded.handle_chat_message(
        "How should we coordinate our pieces across the next two stanzas of play?",
        num_sentences=20,
    )
    last_chat_tel = chat_stanza_snap["chat_messages"][-1]["telemetry"]
    assert "stanza_chord_trajectory" in last_chat_tel
    assert len(last_chat_tel["stanza_chord_trajectory"]) >= 1
    assert "archive_resonance" in last_chat_tel

    # Verify Fix #4a & Fix #5 in `LMFT-Chess.exe`:
    # REM Sleep harvests `ContrastiveGapPair` trajectories (`L2 >= 0.35`) to `celestriad_gap_trajectories.jsonl`
    # and mints discriminative Tier-2 `LearnedLeaf` nodes exposed in `get_state_snapshot()`
    traj_path = state_dir / "celestriad_gap_trajectories.jsonl"
    assert traj_path.exists()
    traj_lines = [ln for ln in traj_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(traj_lines) >= 1
    first_pair = json.loads(traj_lines[0])
    assert first_pair["domain"] == "chess_intuition"
    assert first_pair["chord_l2_distance"] >= 0.35
    assert snap_c3["harvested_gap_pairs_count"] >= 1
    assert snap_c3["learned_leaves_count"] >= 1
    assert len(snap_c3["learned_leaves"][0]["anchor_weights"]) == 8

    # Verify newly minted `hazard_` LearnedLeaf discriminates live candidate board moves
    hz_leaves = [lf for lf in runtime.learned_leaves if lf.leaf_name.startswith("hazard_")]
    assert len(hz_leaves) >= 1
    hz_id = hz_leaves[0].leaf_id
    test_board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 3")
    from lmft_architecture.chess_kibitz_bridge import compute_candidate_move_fuzzy_coordinates

    b_coords, _, _, _, _ = compute_candidate_move_fuzzy_coordinates(
        test_board,
        test_board.parse_san("Bxf7+"),
        learned_leaves=runtime.learned_leaves,
    )
    s_coords, _, _, _, _ = compute_candidate_move_fuzzy_coordinates(
        test_board,
        test_board.parse_san("O-O"),
        learned_leaves=runtime.learned_leaves,
    )
    assert b_coords[hz_id] > s_coords[hz_id] + 0.15


def test_pure_python_installer_wizard_and_zero_unsigned_exes() -> None:
    """Verify the pure-Python 5-page Setup Wizard (`installer_wizard.py`) and confirm zero unsigned `.exe` wrappers exist."""
    import zipfile
    from lmft_architecture.installer_wizard import (
        build_custom_antigravity_prompt,
        check_install_folder_integrity,
        compute_core_sha256_report,
        detect_system_hardware,
        resolve_workspace_paths,
    )

    base_dir, arch_dir, train_dir, assets_dir = resolve_workspace_paths()
    assert (base_dir / "Launch-Setup-Wizard.bat").is_file()
    assert (base_dir / "Launch-LMFT-Chess.bat").is_file()
    assert (base_dir / "Launch-LMFT-Bloodmoon.bat").is_file()

    # Zero custom-compiled unsigned .exe wrappers in root
    for banned_exe in ("LMFT-Installer-Wizard.exe", "LMFT-Chess.exe", "LMFT-Bloodmoon.exe"):
        assert not (base_dir / banned_exe).exists()

    hw = detect_system_hardware()
    assert "gpu_name" in hw and "ram_gb" in hw and float(hw["ram_gb"]) > 0.0
    prompt = build_custom_antigravity_prompt(hw)
    from lmft_architecture.chess_kibitz_bridge import ObjectiveCalibratedSparringEngine

    assert "Google Antigravity" in prompt and "birth.py" in prompt
    assert "onSquarePointerDown" in CHESS_SINGLE_PAGE_HTML
    assert "optimisticallySnapMoveInDom" in CHESS_SINGLE_PAGE_HTML
    assert "drag-piece-ghost" in CHESS_SINGLE_PAGE_HTML
    assert "Sf905k89AsY" in CHESS_SINGLE_PAGE_HTML
    assert "poetry-portfolio-alias-schaleh-marcus-auron-YU73erI" in CHESS_SINGLE_PAGE_HTML
    assert "esper_theories_alphabet.jpg" in CHESS_SINGLE_PAGE_HTML
    assert (assets_dir / "esper_theories_alphabet.jpg").is_file()
    assert hasattr(ObjectiveCalibratedSparringEngine, "_popen_stockfish")

    sha_report = compute_core_sha256_report(base_dir, arch_dir, train_dir)
    assert "[SHA-256] birth.py" in sha_report
    assert "[SHA-256] installer_wizard.py" in sha_report

    integrity_report, core_ok = check_install_folder_integrity(base_dir, arch_dir, train_dir, assets_dir)
    assert core_ok is True
    assert "✅ [OK] 1. Core 40-Node Seed (birth.py)" in integrity_report

    zip_path = train_dir / "dist" / "lmft-sanctuary-club-v0.1.0-win64.zip"
    if zip_path.is_file():
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            assert not any(n.lower().endswith(".exe") for n in names)
            assert "Launch-Setup-Wizard.bat" in names
            assert "Launch-LMFT-Chess.bat" in names

















