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
    """Verify vow-free multi-stanza REM sleep reflection, projector protection (`harvested_gap_pairs_count == 0`), real `will_edges` move scoring, and non-saturating `BitboardIntuitionSleeve` distillation."""
    state_dir = tmp_path / "dream_chronicles_state"
    left_brain = _make_mini_left_brain()
    runtime = ChessKibitzerRuntime(
        state_dir=state_dir,
        left_brain=left_brain,
        lazy_load_gpu=False,
    )
    runtime.set_auto_rem_sleep(False)

    # Cycle 1: Queue two distinct tactical scenes so Deep REM Sleep generates a 2-stanza reflection without vow theater
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
    from dataclasses import replace as _dc_replace
    runtime.sleep_queue[-1] = _dc_replace(
        runtime.sleep_queue[-1],
        left_brain_hunch="Eyeing the vulnerable f7 square in the Italian Game, I sacrifice the Bishop on f7 to drag the King out.",
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
    assert snap_c1["rem_dream_chronicles"] == []
    assert "\n\n" in (snap_c1["last_rem_sleep_self_talk"] or "")
    assert "Longitudinal Dream Vow Status:" not in snap_c1["last_rem_sleep_receipt"]
    assert snap_c1["harvested_gap_pairs_count"] == 0
    assert not (state_dir / "live_projector_weights.pt").exists()

    # Cycle 2: Play a sound castling move (`O-O`) with `is_hazard=False` and run REM sleep
    runtime.handle_set_fen("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 3")
    sound_rev = runtime.sparring_engine.audit_played_move_for_game_review(
        runtime.board,
        "O-O",
        perspective=chess.WHITE,
    )
    runtime.handle_chess_move("O-O")
    runtime.chat_messages[-1]["telemetry"]["left_brain_hunch"] = (
        "Evaluating the Italian Game kingside shelter, I castle short with O-O to safeguard the King."
    )
    id_4 = int(runtime.chat_messages[-1]["id"])
    runtime.flag_turn_for_coaching(
        id_4,
        user_note="Castled safely to shelter the King.",
        check_auto_sleep=False,
        is_hazard=False,
        override_coordinates=sound_rev["played_coordinates"],
        contrastive_reference_coordinates=sound_rev["contrastive_reference_coordinates"],
        game_review_annotation="[! GREAT_MOVE]: Castled safely.",
    )
    snap_c3 = runtime.handle_run_rem_sleep()
    assert snap_c3["rem_dream_chronicles"] == []
    assert "Longitudinal Dream Vow Status:" not in snap_c3["last_rem_sleep_receipt"]
    assert snap_c3["harvested_gap_pairs_count"] == 0
    assert not (state_dir / "live_projector_weights.pt").exists()

    # Verify persistence across restart in `club_meta.json`
    reloaded = ChessKibitzerRuntime(
        state_dir=state_dir,
        left_brain=left_brain,
        lazy_load_gpu=False,
    )
    assert reloaded.rem_dream_chronicles == []

    # Verify live 3-branch archive recall and real `will_edges` move evaluation
    res_check = reloaded.archives.recall_resonant_clauses(
        sound_rev["played_coordinates"],
        min_similarity=0.40,
    )
    assert res_check["top_resonance_score"] >= 0.40
    reloaded.handle_set_fen("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 3")
    ai_snap = reloaded.handle_chess_ai_move()
    arb = ai_snap["last_hybrid_move_arbitration"]
    assert arb["vow_alignment_delta"] == 0.0
    assert arb["guided_by_vow"] is False
    assert arb["active_vow_leaves"] == []

    # Verify `evaluate_candidate_moves_with_right_brain` uses `will_edges` (`rb-will-0..8`)
    from lmft_architecture.right_brain_equilibrium import Association, RIGHT_BRAIN_WILL_EDGES

    eval_default = {
        e.san: e
        for e in evaluate_candidate_moves_with_right_brain(
            chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 3"),
            will_edges=RIGHT_BRAIN_WILL_EDGES,
            limit=30,
        )
    }
    custom_will = tuple(
        Association(
            e.id,
            e.source_id,
            e.target_id,
            0.25 if e.id in ("rb-will-2", "rb-will-5") else (-1.0 if e.id == "rb-will-8" else e.weight),
            e.confidence,
            e.plasticity,
            e.created_event_id,
            e.last_updated_event_id,
        )
        for e in RIGHT_BRAIN_WILL_EDGES
    )
    eval_custom = {
        e.san: e
        for e in evaluate_candidate_moves_with_right_brain(
            chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 3"),
            will_edges=custom_will,
            limit=30,
        )
    }
    assert eval_custom["O-O"].intention != eval_default["O-O"].intention
    assert eval_custom["O-O"].execution_bid != eval_default["O-O"].execution_bid

    # Verify `BitboardIntuitionSleeve.distill_post_mortem_scene` ignores tiny noise (`|r_val - p_val| < 0.15`)
    # and does not saturate `wastelessness` or `material_balance_norm` to `2.5` / `3.0` after 160 steps
    from lmft_architecture.chess_kibitz_bridge import BitboardIntuitionSleeve

    test_sleeve = BitboardIntuitionSleeve()
    for _ in range(160):
        test_sleeve.distill_post_mortem_scene(
            blunder_rev["played_coordinates"],
            blunder_rev["contrastive_reference_coordinates"],
            is_hazard=True,
            played_bitboard_features=blunder_rev["played_bitboard_features"],
            reference_bitboard_features=blunder_rev["reference_bitboard_features"],
        )
    assert test_sleeve.leaf_weights["cultural/efficiency/wastelessness"] < 2.2
    assert test_sleeve.value_weights["material_balance_norm"] < 2.5
    assert test_sleeve.policy_weights["move_net_safety"] < 2.5

    # Verify `<1150 Elo` Stockfish sparring filters out Queen-hanging blunders (`> 250 cp` drop)
    queen_hang_board = chess.Board("r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/3P1Q2/PPP2PPP/RNB1K1NR b KQkq - 2 4")
    sp_res = runtime.sparring_engine.select_sparring_opponent_move(
        queen_hang_board,
        target_elo=990.0,
    )
    assert sp_res["move_san"] not in ("Qg5", "Qh4", "Qf6")

    # Verify multi-stanza chat exports `stanza_chord_trajectory` and `archive_resonance` in telemetry
    chat_stanza_snap = reloaded.handle_chat_message(
        "How should we coordinate our pieces across the next two stanzas of play?",
        num_sentences=20,
    )
    last_chat_tel = chat_stanza_snap["chat_messages"][-1]["telemetry"]
    assert "stanza_chord_trajectory" in last_chat_tel
    assert len(last_chat_tel["stanza_chord_trajectory"]) >= 1
    assert "archive_resonance" in last_chat_tel

    # Verify discriminative Tier-2 `LearnedLeaf` nodes are still minted and exposed in `get_state_snapshot()`
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


def test_weight_space_circuit_sculpting_homeostatic_decay_and_clean_cluster_naming() -> None:
    """Verify 24-circuit REM sculpting, homeostatic rail unsticking, and zero recursive `hazard_high_hazard_` prefixing."""
    from lmft_architecture.chess_app import _ensure_foundational_rules
    from lmft_architecture.chess_kibitz_bridge import (
        DEFAULT_SLEEVE_VALUE_WEIGHTS,
        BitboardIntuitionSleeve,
        ObjectiveCalibratedSparringEngine,
    )
    from lmft_architecture.sleep_consolidation import (
        UnresolvedSleepScene,
        _cluster_key_for_unresolved,
        canonical_cluster_concept_name,
        normalize_contrastive_target_leaves_and_thresholds,
    )

    # 1. `audit_played_move_for_game_review` computes `played_bitboard_features`, `reference_bitboard_features`, and `bitboard_feature_delta`
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 3")
    referee = ObjectiveCalibratedSparringEngine(use_stockfish_uci=False)
    rev = referee.audit_played_move_for_game_review(board, "Bxf7+", perspective=chess.WHITE)
    assert "played_bitboard_features" in rev
    assert "reference_bitboard_features" in rev
    assert "bitboard_feature_delta" in rev
    assert set(rev["played_bitboard_features"].keys()) == set(DEFAULT_SLEEVE_VALUE_WEIGHTS.keys())

    # 2. `BitboardIntuitionSleeve.distill_post_mortem_scene` sculpts all 8 named bitboard circuits and applies homeostatic decay
    sleeve = BitboardIntuitionSleeve()
    sleeve.leaf_weights["philosophical/structure/order"] = 2.5
    sleeve.leaf_weights["philosophical/structure/chaos"] = -2.5
    orig_king_shield_v = sleeve.value_weights["king_shield_safety"]
    orig_center_p = sleeve.policy_weights["center_bitboard_control"]

    receipt = sleeve.distill_post_mortem_scene(
        {"philosophical/being/existence": 0.40, "psychological/truth/fact": 0.40},
        {"philosophical/being/existence": 0.40, "psychological/truth/fact": 0.70},
        is_hazard=True,
        learning_rate=0.10,
        played_bitboard_features={
            "material_balance_norm": 0.0,
            "attack_control_share": 0.45,
            "center_bitboard_control": 0.10,
            "king_shield_safety": 0.32,
            "piece_coordination_ratio": 0.40,
            "development_progress": 0.25,
            "move_net_safety": -0.20,
            "move_positional_gain": 0.0,
        },
        reference_bitboard_features={
            "material_balance_norm": 0.25,
            "attack_control_share": 0.55,
            "center_bitboard_control": 0.50,
            "king_shield_safety": 0.85,
            "piece_coordination_ratio": 0.75,
            "development_progress": 0.65,
            "move_net_safety": 0.60,
            "move_positional_gain": 0.45,
        },
    )
    assert "king_shield_safety" in receipt["updated_circuit_keys"]
    assert "center_bitboard_control" in receipt["updated_circuit_keys"]
    assert "piece_coordination_ratio" in receipt["updated_circuit_keys"]
    assert "development_progress" in receipt["updated_circuit_keys"]
    assert sleeve.value_weights["king_shield_safety"] > orig_king_shield_v
    assert sleeve.policy_weights["center_bitboard_control"] > orig_center_p
    # Homeostatic decay pulls saturated `±2.5` rails gently back toward default when leaf delta is 0
    assert sleeve.leaf_weights["philosophical/structure/order"] < 2.5
    assert sleeve.leaf_weights["philosophical/structure/chaos"] > -2.5

    # 3. `_cluster_key_for_unresolved` and `canonical_cluster_concept_name` never stack `hazard_high_hazard_` prefixes
    sc_with_tier2 = UnresolvedSleepScene(
        scene_index=999,
        scenario_text="Chess table — blunder test",
        external_coordinates={
            "cultural/efficiency/power": 0.62,
            "cultural/efficiency/wastelessness": 0.42,
            "cultural/freedom/hazard_low_self_sufficiency_chaos": 0.95,
            "philosophical/structure/hazard_high_power_chaos": 0.92,
            "philosophical/structure/order": 0.58,
            "philosophical/structure/chaos": 0.20,
        },
        felt_chord_notes={},
        valence=0.35,
        salience=0.80,
        arousal=0.65,
        left_brain_hunch="Test",
        is_hazard=True,
    )
    ckey = _cluster_key_for_unresolved(sc_with_tier2)
    assert ckey == "high_power_order_cluster"
    assert "hazard_" not in ckey

    assert (
        canonical_cluster_concept_name("hazard_high_hazard_high_power_chaos_chaos_cluster")
        == "hazard_high_power_chaos_cluster"
    )
    assert (
        canonical_cluster_concept_name("hazard_high_hazard_low_self_sufficiency_chaos_chaos_cluster")
        == "hazard_low_self_sufficiency_chaos_cluster"
    )
    assert (
        canonical_cluster_concept_name("hazard_high_hazard_high_hazard_high_hazard_high_chaos_cluster")
        == "hazard_high_power_chaos_cluster"
    )

    # 4. `coached_*` rules never attach `hazard_*` Tier-2 leaves in `normalize_contrastive_target_leaves_and_thresholds` or `_ensure_foundational_rules`
    c_targets, _, _ = normalize_contrastive_target_leaves_and_thresholds(
        "low_needlessness_continuity_cluster",
        {
            "cultural/efficiency/wastelessness": 0.85,
            "cultural/freedom/needlessness": 0.08,
            "cultural/efficiency/hazard_high_power_chaos": 0.88,
            "philosophical/persistence/continuity": 0.78,
        },
        {
            "cultural/efficiency/wastelessness": 0.30,
            "cultural/freedom/needlessness": 0.55,
            "cultural/efficiency/hazard_high_power_chaos": 0.15,
            "philosophical/persistence/continuity": 0.40,
        },
    )
    assert "cultural/efficiency/hazard_high_power_chaos" not in c_targets
    assert "philosophical/persistence/continuity" in c_targets

    tangled_coached = FuzzyConceptRule(
        rule_id="rule-sleep-coached_tangled",
        concept_name="coached_low_needlessness_order_cluster",
        description="Found Rg1.",
        target_leaves={
            "cultural/efficiency/wastelessness": 0.75,
            "cultural/freedom/needlessness": 0.08,
            "cultural/efficiency/hazard_high_power_chaos": 0.25,
        },
        low=0.60,
        high=0.85,
    )
    cleaned_tangled = [
        r for r in _ensure_foundational_rules((tangled_coached,)) if r.concept_name == "coached_low_needlessness_order_cluster"
    ]
    assert len(cleaned_tangled) == 1
    assert "cultural/efficiency/hazard_high_power_chaos" not in cleaned_tangled[0].target_leaves


def test_chess_runtime_will_edges_persistence_and_virgin_lock(tmp_path: Path) -> None:
    """Verify D066: ChessKibitzerRuntime sculpts and persists `will_edges.json` in Schooled mode while keeping Virgin locked."""
    from lmft_architecture.right_brain_equilibrium import RIGHT_BRAIN_WILL_EDGES
    from lmft_architecture.sleep_consolidation import UnresolvedSleepScene

    state_dir = tmp_path / "d066_will_state"
    runtime = ChessKibitzerRuntime(
        state_dir=state_dir,
        left_brain=_make_mini_left_brain(),
        lazy_load_gpu=False,
    )
    runtime.set_auto_rem_sleep(False)
    assert runtime.will_edges == RIGHT_BRAIN_WILL_EDGES

    runtime.handle_set_fen("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 3")
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
        played_bitboard_features=rev["played_bitboard_features"],
        reference_bitboard_features=rev["reference_bitboard_features"],
    )
    runtime.handle_run_rem_sleep()

    assert runtime.will_edges != RIGHT_BRAIN_WILL_EDGES
    assert (state_dir / "will_edges.json").exists()

    # Switching to Virgin locks `will_edges` to `RIGHT_BRAIN_WILL_EDGES`, and switching back restores sculpted edges
    sculpted_snapshot = runtime.will_edges
    runtime.switch_brain_lineage("virgin")
    assert runtime.will_edges == RIGHT_BRAIN_WILL_EDGES
    runtime.switch_brain_lineage("schooled")
    assert runtime.will_edges == sculpted_snapshot

    # Verify mid-game board moves (`handle_chess_move`) do NOT trigger mid-game Auto-REM sleep while game is active,
    # and `_harvest_rem_contrastive_gap_pairs` never splits fallback `"Legal replies include ..."` strings on a comma
    runtime.handle_set_fen("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 3")
    runtime.flag_turn_for_coaching(
        last_id,
        user_note=rev["annotation_summary"],
        check_auto_sleep=False,
        is_hazard=True,
        override_coordinates=rev["played_coordinates"],
        contrastive_reference_coordinates=rev["contrastive_reference_coordinates"],
        game_review_annotation=rev["annotation_summary"],
    )
    runtime.auto_rem_sleep = True
    runtime.auto_sleep_threshold = 1
    assert len(runtime.sleep_queue) == 1
    # Playing a mid-game move while `not board.is_game_over()` must NOT trigger auto-REM sleep mid-game
    runtime.handle_chess_move("O-O")
    assert len(runtime.sleep_queue) == 1

    # Running REM sleep skips harvesting when `left_brain_hunch` is a silent Gear-1 fallback without a real LLM comma-split sentence
    runtime.sleep_queue[-1] = UnresolvedSleepScene(
        scene_index=runtime.sleep_queue[-1].scene_index,
        scenario_text="Chess table — Move 3. White to move. Legal replies include Bxf7+, O-O, d3, Nc3.",
        external_coordinates=rev["played_coordinates"],
        felt_chord_notes={},
        valence=0.25,
        salience=0.85,
        arousal=0.72,
        left_brain_hunch="Unresolved scene tension: Chess table — Move 3. Legal replies include Ra4, Rg4, Qd2, Qe3.",
        is_hazard=True,
        contrastive_reference_coordinates=rev["contrastive_reference_coordinates"],
        game_review_annotation=rev["annotation_summary"],
    )
    runtime.handle_run_rem_sleep()
    assert len(runtime._last_harvested_gap_pairs) == 0

    # Verify `_generate_rem_sleep_self_talk` branches cleanly between 1st-person Blunder (`is_hazard=True`)
    # and 1st-person Brilliancy/Checkmate (`is_hazard=False`, e.g., `Found Qe8#`)
    captured_dream_calls: list[tuple[str, str | None]] = []
    orig_gen_splice = runtime._qwen_left_brain.generate_with_subliminal_splice

    def _spy_gen_splice(scenario_prompt: str, chord, **kwargs):
        c1 = kwargs.get("clause_1_override")
        captured_dream_calls.append((scenario_prompt, c1))
        return orig_gen_splice(scenario_prompt, chord, **kwargs)

    runtime._qwen_left_brain.generate_with_subliminal_splice = _spy_gen_splice  # type: ignore[method-assign]
    blunder_scene = UnresolvedSleepScene(
        scene_index=10,
        scenario_text="Move 3. White played Bxf7+.",
        external_coordinates=rev["played_coordinates"],
        felt_chord_notes={},
        valence=0.25,
        salience=0.85,
        arousal=0.72,
        left_brain_hunch="Test blunder",
        is_hazard=True,
        contrastive_reference_coordinates=rev["contrastive_reference_coordinates"],
        game_review_annotation="?? Blunder: played Bxf7+ instead of O-O.",
    )
    brilliancy_scene = UnresolvedSleepScene(
        scene_index=11,
        scenario_text="Move 39. White played Qe8#.",
        external_coordinates=rev["contrastive_reference_coordinates"],
        felt_chord_notes={},
        valence=0.88,
        salience=0.92,
        arousal=0.68,
        left_brain_hunch="Test checkmate",
        is_hazard=False,
        contrastive_reference_coordinates=rev["contrastive_reference_coordinates"],
        game_review_annotation="!! Brilliancy: Found 39. Qe8# checkmate.",
    )
    runtime._generate_rem_sleep_self_talk((blunder_scene,), (), num_sentences=2, max_stanzas=1)
    runtime._generate_rem_sleep_self_talk((brilliancy_scene,), (), num_sentences=2, max_stanzas=1)
    assert len(captured_dream_calls) == 2
    blunder_prompt, blunder_c1 = captured_dream_calls[0]
    brilliancy_prompt, brilliancy_c1 = captured_dream_calls[1]
    assert blunder_c1 == "Looking back at my move Bxf7+ on the board,"
    assert "overlooked the stronger continuation" in blunder_prompt
    assert brilliancy_c1 == "Savoring how my move Qe8# harmonized the board,"
    assert "delivered the winning continuation" in brilliancy_prompt


def test_zero_wire_clone_rule_ties_and_rem_vow_prompt_no_chat_bleed(tmp_path: Path) -> None:
    """Verify (1) no two `rule-sleep-*` rules in `club_ruleset.json` share 3-wire clones or tie at `Clarity == 0.00`, (2) `<=` receipt formatting for low target centers, and (3) `prev_chronicle` vow is folded into `sleep_prompt` with `conversation_history=()`."""
    from lmft_architecture.chess_app import _ensure_foundational_rules
    from lmft_architecture.reflection_classifier import (
        FOUNDATIONAL_LEAVES,
        _evaluate_rules_with_inhibition,
        compute_contemplation,
        load_learned_leaves,
        load_ruleset,
    )
    from lmft_architecture.sleep_consolidation import UnresolvedSleepScene, parse_cluster_defining_leaves

    # 1. Verify `club_ruleset.json` on disk starts clean with the 16 canonical leaves (zero duplicate
    #    `coached_*`/`hazard_*` leaves) and `VIRGIN_FOUNDATIONAL_RULES` (with zero duplicate 3-wire clones).
    repo_root = Path(__file__).resolve().parents[2]
    ruleset_path = repo_root / "LMFT-Training" / "outputs" / "chess_kibitzer_state" / "club_ruleset.json"
    rules = load_ruleset(ruleset_path)
    leaves = load_learned_leaves(ruleset_path)
    canonical_leaves = [lf for lf in leaves if not lf.leaf_name.startswith(("coached_", "hazard_", "immunized_"))]
    assert len(canonical_leaves) == 16
    assert len(rules) >= 2
    sleep_rules = [r for r in rules if r.rule_id.startswith("rule-sleep-")]

    seen_sigs: set[tuple[str, frozenset[str]]] = set()
    for r in sleep_rules:
        assert r.high <= 0.90 and r.low <= 0.75
        prefix, _, def_cult, def_phil = parse_cluster_defining_leaves(r.concept_name)
        if prefix in ("hazard", "coached") and def_cult and def_phil:
            sig = (prefix, frozenset(r.target_leaves.keys()))
            assert sig not in seen_sigs, f"Duplicate 3-wire cluster clone {r.concept_name} on {sig}"
            seen_sigs.add(sig)

    # 2. Verify `_ensure_foundational_rules` deduplicates two differently named cluster rules that watch
    #    the exact same 3 leaves into a single entrained rule without fabricating coordinates or ratcheting high >= 0.88.
    clone_a = FuzzyConceptRule(
        rule_id="rule-sleep-hazard_low_wastelessness_order_c",
        concept_name="hazard_low_wastelessness_order_cluster",
        description="Clone A",
        target_leaves={
            "cultural/efficiency/wastelessness": 0.25,
            "cultural/freedom/needlessness": 0.75,
            "philosophical/structure/order": 0.64,
        },
        low=0.58,
        high=0.82,
        polarity="repulsive",
    )
    clone_b = FuzzyConceptRule(
        rule_id="rule-sleep-hazard_high_power_order_cluster",
        concept_name="hazard_high_power_order_cluster",
        description="Clone B with identical 3 wires",
        target_leaves={
            "cultural/efficiency/wastelessness": 0.25,
            "cultural/freedom/needlessness": 0.75,
            "philosophical/structure/order": 0.64,
        },
        low=0.58,
        high=0.84,
        polarity="repulsive",
    )
    deduped = _ensure_foundational_rules((clone_a, clone_b))
    sleep_deduped = [r for r in deduped if r.rule_id.startswith("rule-sleep-")]
    assert len(sleep_deduped) == 1
    assert sleep_deduped[0].high < 0.88
    assert "cultural/efficiency/power" not in sleep_deduped[0].target_leaves

    # 3. Verify `sleep_prompt` strips raw PGN move numbers/results, omits vow theater, uses `conversation_history=()`,
    #    and `Newly Forged Right-Brain Rules` formats `<= 0.30` leaf targets with `<=`.
    left_brain = _make_mini_left_brain()
    runtime = ChessKibitzerRuntime(
        state_dir=tmp_path / "no_bleed_state",
        left_brain=left_brain,
        lazy_load_gpu=False,
    )
    captured_hist: list[tuple[tuple[str, str], ...]] = []
    captured_prompts: list[str] = []
    orig_splice = left_brain.generate_with_subliminal_splice

    def _spy_splice(prompt_text: str, chord, **kwargs):
        captured_prompts.append(prompt_text)
        captured_hist.append(tuple(kwargs.get("conversation_history") or ()))
        return orig_splice(prompt_text, chord, **kwargs)

    left_brain.generate_with_subliminal_splice = _spy_splice  # type: ignore[method-assign]
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 3")
    rev = runtime.sparring_engine.audit_played_move_for_game_review(board, "Bxf7+", perspective=chess.WHITE)
    sc = UnresolvedSleepScene(
        scene_index=99,
        scenario_text="Move 3. White played 3. Bxf7+ (7. Be2 Qd7 9: 0-1.).",
        external_coordinates=rev["played_coordinates"],
        felt_chord_notes={},
        valence=0.22,
        salience=0.84,
        arousal=0.76,
        left_brain_hunch="Test blunder",
        is_hazard=True,
        contrastive_reference_coordinates=rev["contrastive_reference_coordinates"],
        game_review_annotation=rev["annotation_summary"],
    )
    runtime.sleep_queue.append(sc)
    snap = runtime.handle_run_rem_sleep()
    assert captured_hist and all(h == () for h in captured_hist)
    assert all("vow" not in p.lower() for p in captured_prompts)
    assert all("0-1" not in p and "Move 3" not in p for p in captured_prompts)
    assert "Previous REM Sleep Vow" not in snap["last_rem_sleep_receipt"]
    assert "Longitudinal Dream Vow Status:" not in snap["last_rem_sleep_receipt"]
    assert "wastelessness<=0.25" in snap["last_rem_sleep_receipt"]

    # 4. Verify all 29 queued scenes from Game #135 (`Scene #15545..#15651`) settle with `Clarity >= 0.50` when sleep rules are loaded,
    #    preserved `rule_grounded_coordination` refinements survive `_ensure_foundational_rules`, and
    #    cross-family (`hazard_` vs `coached_`) ties still collide (`Clarity == 0.00, is_confused == True`).
    threads_path = repo_root / "LMFT-Training" / "outputs" / "chess_kibitzer_state" / "threads.json"
    if threads_path.exists():
        t_raw = json.loads(threads_path.read_text(encoding="utf-8"))
        msgs_135 = (t_raw.get("threads") or [{}])[0].get("chat_messages", [])
        scenes_135 = [
            (
                int(m["id"]),
                ((m.get("telemetry") or {}).get("game_review") or {}).get("played_coordinates")
                or (m.get("telemetry") or {}).get("fact_coordinates")
                or {},
                bool(((m.get("telemetry") or {}).get("game_review") or {}).get("is_hazard")),
            )
            for m in msgs_135
            if isinstance(m.get("id"), int)
            and 15545 <= int(m["id"]) <= 15651
            and ((m.get("telemetry") or {}).get("game_review") or {}).get("should_queue_for_sleep")
        ]
        if scenes_135 and sleep_rules:
            assert len(scenes_135) == 29
            for sid, p_coords, is_haz in scenes_135:
                sc_coords = {k: 0.50 for k in FOUNDATIONAL_LEAVES}
                for rr in rules:
                    for k in rr.target_leaves:
                        sc_coords.setdefault(k, 0.0)
                for k, v in p_coords.items():
                    sc_coords[k] = float(v)
                if is_haz:
                    sc_coords["cultural/efficiency/wastelessness"] = min(
                        0.25,
                        float(sc_coords.get("cultural/efficiency/wastelessness", 0.25)),
                    )
                    sc_coords["cultural/freedom/needlessness"] = max(
                        0.53,
                        float(sc_coords.get("cultural/freedom/needlessness", 0.53)),
                    )
                sc_scores = _evaluate_rules_with_inhibition(tuple(rules), sc_coords)
                c_val, conf_val, matched_name, is_conf, _ = compute_contemplation(sc_scores)
                assert not is_conf, f"Scene #{sid} remained confused (Clarity={c_val:.2f})"
                assert c_val >= 0.50
                assert conf_val == pytest.approx(1.0)
                assert matched_name is not None and matched_name.startswith(
                    "hazard_" if is_haz else "coached_"
                )

    refined_grounded = FuzzyConceptRule(
        rule_id="rule_grounded_coordination",
        concept_name="grounded_coordination",
        description="Refined grounded coordination with low chaos.",
        target_leaves={
            "philosophical/being/existence": 0.68,
            "philosophical/structure/order": 0.65,
            "philosophical/structure/chaos": 0.10,
        },
        low=0.60,
        high=0.84,
    )
    ensured_refined = _ensure_foundational_rules((refined_grounded,))
    by_c = {r.concept_name: r for r in ensured_refined}
    assert "philosophical/structure/chaos" in by_c["grounded_coordination"].target_leaves
    assert by_c["grounded_coordination"].high == pytest.approx(0.84)


def test_board_valence_evaluation_and_candidate_move_valence() -> None:
    """Verify `experiential/phase/valence` reflects the `+/-` chess board evaluation and candidate move quality."""
    from lmft_architecture.chess_kibitz_bridge import (
        compute_board_valence,
        compute_candidate_move_fuzzy_coordinates,
    )

    start_board = chess.Board()
    assert compute_board_valence(start_board, chess.WHITE) == pytest.approx(0.50, abs=0.05)
    assert compute_board_valence(start_board, chess.BLACK) == pytest.approx(0.50, abs=0.05)
    start_posture = evaluate_chess_board_posture(start_board)
    assert start_posture.board_valence == pytest.approx(0.50, abs=0.05)
    assert start_posture.to_dict()["board_valence"] == pytest.approx(0.50, abs=0.05)

    # White is up a Knight (+300 cp)
    piece_up_board = chess.Board("r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    val_white = compute_board_valence(piece_up_board, chess.WHITE)
    val_black = compute_board_valence(piece_up_board, chess.BLACK)
    assert val_white > 0.68
    assert val_black < 0.32

    # BoardAnchoredBridgeSensors grounds `experiential/phase/valence` on the board evaluation
    sensors = ThreeBridgeSensors(
        will_cultural=lambda _t, _c=(): {k: 0.04 for k in FOUNDATIONAL_LEAVES[:6]},
        truth_philosophical=lambda _t, _c=(): {k: 0.25 for k in FOUNDATIONAL_LEAVES[6:12]},
        contemplation_experiential=lambda _t, _c=(): {
            "experiential/reflection/salience": 0.45,
            "experiential/phase/valence": 0.11,
            "experiential/phase/arousal": 0.30,
        },
    )
    anchored_sensors = merge_board_fact_with_chat_sensors(start_posture, sensors)
    coords, exp_out, _q, _s = anchored_sensors.score_text(
        build_natural_chess_scenario_prompt(start_board)
    )
    assert exp_out["experiential/phase/valence"] == pytest.approx(0.50, abs=0.05)
    assert coords["experiential/phase/valence"] == pytest.approx(0.50, abs=0.05)
    for exp_key in (
        "experiential/phase/valence",
        "experiential/phase/arousal",
        "experiential/reflection/salience",
        "experiential/reflection/novelty",
        "experiential/synchronization/resonance",
        "experiential/synchronization/entrainment",
    ):
        assert exp_key in exp_out and 0.0 <= exp_out[exp_key] <= 1.0
        assert exp_key in coords and coords[exp_key] == pytest.approx(exp_out[exp_key])
    assert coords["cultural/justice/honor"] != 0.52
    assert coords["cultural/justice/fairness"] != 0.52

    # Candidate move fuzzy coordinates populate all 24 Founding Leaves
    all_24_founding_leaves = (
        "cultural/efficiency/power",
        "cultural/efficiency/wastelessness",
        "cultural/justice/honor",
        "cultural/justice/fairness",
        "cultural/freedom/self-sufficiency",
        "cultural/freedom/needlessness",
        "philosophical/being/agency",
        "philosophical/being/existence",
        "philosophical/structure/order",
        "philosophical/structure/chaos",
        "philosophical/persistence/continuity",
        "philosophical/persistence/discreteness",
        "experiential/reflection/novelty",
        "experiential/reflection/salience",
        "experiential/phase/valence",
        "experiential/phase/arousal",
        "experiential/synchronization/resonance",
        "experiential/synchronization/entrainment",
        "psychological/will/intention",
        "psychological/will/execution",
        "psychological/truth/theory",
        "psychological/truth/fact",
        "psychological/contemplation/clarity",
        "psychological/contemplation/confidence",
    )
    italian_board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 3")
    coords_bxf7, is_blunder_bxf7, _, _, _ = compute_candidate_move_fuzzy_coordinates(
        italian_board,
        italian_board.parse_san("Bxf7+"),
    )
    coords_oo, is_blunder_oo, _, _, _ = compute_candidate_move_fuzzy_coordinates(
        italian_board,
        italian_board.parse_san("O-O"),
    )
    for k24 in all_24_founding_leaves:
        assert k24 in coords_oo, f"Missing Founding Leaf {k24} in candidate move coordinates"
        assert 0.0 <= coords_oo[k24] <= 1.0
    assert is_blunder_bxf7 is True
    assert is_blunder_oo is False
    assert coords_bxf7["experiential/phase/valence"] <= 0.28
    assert coords_oo["experiential/phase/valence"] > coords_bxf7["experiential/phase/valence"] + 0.20
    assert coords_bxf7["cultural/justice/fairness"] <= 0.30
    assert coords_oo["cultural/justice/fairness"] > coords_bxf7["cultural/justice/fairness"]
    assert coords_oo["cultural/justice/honor"] > coords_bxf7["cultural/justice/honor"]
    assert coords_oo["experiential/synchronization/resonance"] > coords_bxf7["experiential/synchronization/resonance"]
    assert coords_oo["experiential/synchronization/entrainment"] > coords_bxf7["experiential/synchronization/entrainment"]

    # 2-fold move repetition drops `experiential/reflection/novelty <= 0.10` and `cultural/justice/honor`
    rep_board = chess.Board()
    for san_mv in ("Nf3", "Nf6", "Ng1"):
        rep_board.push_san(san_mv)
    coords_rep_ng8, _, _, _, _ = compute_candidate_move_fuzzy_coordinates(
        rep_board,
        rep_board.parse_san("Ng8"),
    )
    coords_dev_nc6, _, _, _, _ = compute_candidate_move_fuzzy_coordinates(
        rep_board,
        rep_board.parse_san("Nc6"),
    )
    assert coords_rep_ng8["experiential/reflection/novelty"] <= 0.10
    assert coords_dev_nc6["experiential/reflection/novelty"] > coords_rep_ng8["experiential/reflection/novelty"] + 0.25
    assert coords_dev_nc6["cultural/justice/honor"] > coords_rep_ng8["cultural/justice/honor"] + 0.20

    # Winning free-queen capture has high positive valence (>= 0.68) and high tactical self-sufficiency (>= 0.72)
    # even in the opening while other minor pieces sit on the back rank (knowing when all pieces are NOT required)
    free_queen_board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/3qP3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3")
    coords_nxd4, is_blunder_nxd4, _, _, _ = compute_candidate_move_fuzzy_coordinates(
        free_queen_board,
        free_queen_board.parse_san("Nxd4"),
    )
    assert is_blunder_nxd4 is False
    assert coords_nxd4["experiential/phase/valence"] >= 0.68
    assert coords_nxd4["cultural/justice/fairness"] >= 0.65
    assert coords_nxd4["cultural/freedom/self-sufficiency"] >= 0.72
    assert coords_nxd4["cultural/freedom/needlessness"] <= 0.08

    # Quiet opening position (Move 2: after 1. e4 e5 2. Nf3 Nc6):
    # Developing a new piece (`Bc4`) beats moving the same knight twice (`Ng5`) while 3 minor pieces sleep on rank 1
    quiet_opening = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    coords_bc4, _, _, _, _ = compute_candidate_move_fuzzy_coordinates(
        quiet_opening,
        quiet_opening.parse_san("Bc4"),
    )
    coords_ng5, _, _, _, _ = compute_candidate_move_fuzzy_coordinates(
        quiet_opening,
        quiet_opening.parse_san("Ng5"),
    )
    assert coords_bc4["cultural/justice/honor"] > coords_ng5["cultural/justice/honor"] + 0.12
    assert coords_bc4["cultural/freedom/self-sufficiency"] > coords_ng5["cultural/freedom/self-sufficiency"] + 0.12
    assert coords_ng5["cultural/freedom/needlessness"] > coords_bc4["cultural/freedom/needlessness"] + 0.15
    assert coords_bc4["experiential/synchronization/resonance"] > coords_ng5["experiential/synchronization/resonance"]

    # Non-chess socio-political rules (`violent_resource_seizure`) are stripped by `_ensure_foundational_rules`,
    # spurious high `honor >= 0.58` is stripped from `hazard_low_wastelessness_chaos_cluster` and repaired to 3 wires,
    # and Same-Family Cluster Winner-Take-All is never overwritten by the Soft-Min tie-breaker when `w_val < 0.90`.
    from lmft_architecture.chess_app import _ensure_foundational_rules
    from lmft_architecture.reflection_classifier import (
        FuzzyConceptRule,
        _evaluate_rules_with_inhibition,
        compute_contemplation,
    )

    polluted_rules = (
        FuzzyConceptRule(
            rule_id="rule-socio-1",
            concept_name="violent_resource_seizure",
            description="Non-chess socio-political rule from sleep consolidation test.",
            target_leaves={"cultural/efficiency/power": 0.85, "philosophical/being/agency": 0.95},
            low=0.55,
            high=0.85,
        ),
        FuzzyConceptRule(
            rule_id="rule-sleep-hazard_low_wastelessness_chaos_c",
            concept_name="hazard_low_wastelessness_chaos_cluster",
            description="Blunder cluster with spurious high honor wire.",
            target_leaves={
                "cultural/efficiency/wastelessness": 0.25,
                "cultural/justice/honor": 0.69,
                "philosophical/structure/chaos": 0.35,
            },
            low=0.58,
            high=0.88,
        ),
        FuzzyConceptRule(
            rule_id="rule-sleep-coached_high_power_agency_cluste",
            concept_name="coached_high_power_agency_cluster",
            description="Coached power agency cluster.",
            target_leaves={
                "cultural/efficiency/wastelessness": 0.85,
                "cultural/efficiency/power": 0.91,
                "philosophical/being/agency": 0.95,
            },
            low=0.58,
            high=0.88,
        ),
        FuzzyConceptRule(
            rule_id="rule-sleep-coached_high_wastelessness_agenc",
            concept_name="coached_high_wastelessness_agency_cluster",
            description="Coached wastelessness agency cluster.",
            target_leaves={
                "cultural/efficiency/wastelessness": 0.94,
                "cultural/justice/honor": 0.61,
                "philosophical/being/agency": 0.95,
            },
            low=0.58,
            high=0.88,
        ),
    )
    cleaned_polluted = _ensure_foundational_rules(polluted_rules)
    cleaned_names = {r.concept_name for r in cleaned_polluted}
    assert "violent_resource_seizure" not in cleaned_names
    haz_chaos = next(r for r in cleaned_polluted if r.concept_name == "hazard_low_wastelessness_chaos_cluster")
    assert "cultural/justice/honor" not in haz_chaos.target_leaves
    assert len(haz_chaos.target_leaves) == 2

    scene_15902_coords = {k: 0.50 for k in FOUNDATIONAL_LEAVES}
    scene_15902_coords.update(
        {
            "cultural/efficiency/wastelessness": 0.80,
            "cultural/efficiency/power": 0.88,
            "cultural/freedom/needlessness": 0.08,
            "cultural/justice/honor": 0.61,
            "philosophical/being/agency": 0.95,
            "philosophical/being/existence": 0.68,
            "philosophical/structure/order": 0.65,
            "philosophical/structure/chaos": 0.15,
        }
    )
    acts_15902 = _evaluate_rules_with_inhibition(cleaned_polluted, scene_15902_coords)
    clarity_15902, conf_15902, matched_15902, confused_15902, _ = compute_contemplation(acts_15902)
    assert confused_15902 is False
    assert clarity_15902 >= 0.50
    assert matched_15902 == "coached_high_power_agency_cluster"


def test_clean_machine_installer_crlf_sync_and_signed_contrastive_bitboard_sleeve(tmp_path: Path) -> None:
    """Verify all 6 clean-machine installer fixes and signed contrastive BitboardIntuitionSleeve updates."""
    from lmft_architecture.chess_kibitz_bridge import (
        BitboardIntuitionSleeve,
        compute_candidate_move_fuzzy_coordinates,
    )
    from lmft_architecture.github_auto_sync import (
        PRETRAINED_MODEL_CHECKPOINT_PATHS,
        compute_file_sha256,
        sync_missing_pretrained_models,
        verify_or_sync_against_manifest,
    )
    from lmft_architecture.installer_wizard import write_provision_venv_batch
    from lmft_architecture.workspace_paths import resolve_arch_root, resolve_training_root

    # 1. `write_provision_venv_batch` safely handles Windows usernames with spaces without `\"` escaping
    spaced_arch = tmp_path / "Users" / "First Last" / "LMFT-Celestriad"
    spaced_py = tmp_path / "Users" / "First Last" / "Python311" / "python.exe"
    bat_path = write_provision_venv_batch(spaced_arch, spaced_py, launch_after=True)
    assert bat_path.is_file()
    bat_text = bat_path.read_text(encoding="utf-8")
    assert '\\"' not in bat_text
    assert f'"{spaced_py}" -m venv "%~dp0.venv"' in bat_text
    assert '"%~dp0.venv\\Scripts\\python.exe" -m pip install -e "%~dp0"' in bat_text

    # 2. `pyproject.toml` supports Python 3.11-3.13, valid PyPI versions, and runtime dependencies
    arch_root = resolve_arch_root()
    pyproject_text = (arch_root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.11,<3.14"' in pyproject_text
    assert "[build-system]" in pyproject_text
    for banned_pin in ("numpy==2.5.3", "scipy==1.18.1", "networkx==3.6.1", "pytest==9.1.1"):
        assert banned_pin not in pyproject_text
    for req_pkg in ("chess", "transformers", "accelerate", "bitsandbytes", "huggingface-hub", "pywebview"):
        assert req_pkg in pyproject_text

    # 3. CRLF vs LF normalization in `compute_file_sha256` and `verify_or_sync_against_manifest`
    lf_bytes = b'{\n  "schema_version": 1\n}\n'
    crlf_bytes = b'{\r\n  "schema_version": 1\r\n}\r\n'
    f_lf = tmp_path / "sample_lf.json"
    f_crlf = tmp_path / "sample_crlf.json"
    f_lf.write_bytes(lf_bytes)
    f_crlf.write_bytes(crlf_bytes)
    assert compute_file_sha256(f_lf) == compute_file_sha256(f_crlf)

    # Verify `apply_sync=True` succeeds when manifest has CRLF hash and GitHub returns LF bytes (or vice versa)
    crlf_raw_sha = __import__("hashlib").sha256(crlf_bytes).hexdigest()
    fake_manifest = {
        "version": "0.2.0",
        "build_short_hash": "abc123456789",
        "files": {
            "outputs/bitboard_sleeve_weights.json": {
                "sha256": crlf_raw_sha,
                "size_bytes": len(crlf_bytes),
            }
        },
    }
    sync_root = tmp_path / "clean_install"
    res = verify_or_sync_against_manifest(
        sync_root,
        fake_manifest,
        file_fetcher=lambda _rel: lf_bytes,
        apply_sync=True,
    )
    assert res["status"] == "synced"
    assert "outputs/bitboard_sleeve_weights.json" in res["synced_files"]

    # 4. `sync_missing_pretrained_models` downloads missing model checkpoint files
    assert len(PRETRAINED_MODEL_CHECKPOINT_PATHS) >= 20
    dl = sync_missing_pretrained_models(sync_root, file_fetcher=lambda _rel: b"checkpoint-bytes")
    assert len(dl) == len(PRETRAINED_MODEL_CHECKPOINT_PATHS)
    assert sync_missing_pretrained_models(sync_root, file_fetcher=lambda _rel: b"checkpoint-bytes") == []

    # 5. Desktop shortcut uses ASCII hyphen and PowerShell environment variables
    wiz_src = (arch_root / "src" / "lmft_architecture" / "installer_wizard.py").read_text(encoding="utf-8")
    assert "Celestriad 1.0 - LMFT Chess.lnk" in wiz_src
    assert "$env:CELESTRIAD_LNK" in wiz_src
    assert "$env:CELESTRIAD_TARGET" in wiz_src

    # 6. `BitboardIntuitionSleeve` signed contrastive updates decrease over-weighted blunder features
    sleeve = BitboardIntuitionSleeve()
    start_attack_v = sleeve.value_weights["attack_control_share"]
    sleeve.distill_post_mortem_scene(
        {"philosophical/being/agency": 0.90, "psychological/truth/fact": 0.30},
        {"philosophical/being/agency": 0.45, "psychological/truth/fact": 0.80},
        is_hazard=True,
        learning_rate=0.12,
        played_bitboard_features={"attack_control_share": 0.92, "king_shield_safety": 0.25},
        reference_bitboard_features={"attack_control_share": 0.40, "king_shield_safety": 0.85},
    )
    assert sleeve.value_weights["attack_control_share"] < start_attack_v
    assert sleeve.value_weights["king_shield_safety"] > 0.95

    train_root = resolve_training_root()
    saved_path = train_root / "outputs" / "bitboard_sleeve_weights.json"
    saved_weights = json.loads(saved_path.read_text(encoding="utf-8"))
    assert saved_weights["value_weights"]["move_net_safety"] >= 1.50
    assert saved_weights["steps_distilled"] >= 0
    assert all(float(v) < 2.25 for v in saved_weights["value_weights"].values())
    assert all(float(v) < 2.25 for v in saved_weights["policy_weights"].values())
    assert all(abs(float(v)) < 2.25 for v in saved_weights["leaf_weights"].values())

    # 7. Forced Mate-in-2 (`best_board_3.is_checkmate()`) ranks #1 and blowout captures never flag `-75.34 pawn` blunders
    import chess
    from lmft_architecture.chess_kibitz_bridge import (
        ObjectiveCalibratedSparringEngine,
        evaluate_candidate_moves_with_right_brain,
    )

    mate_in_2_board = chess.Board("1k2r3/8/8/8/2P5/4q3/5PPP/R6K b - - 0 1")
    cands = evaluate_candidate_moves_with_right_brain(mate_in_2_board, limit=16)
    assert cands[0].san == "Qe1+"
    assert cands[0].composite_score > 40.0
    assert cands[0].theory == pytest.approx(0.98)

    blowout_board = chess.Board("1k2r3/3q4/8/8/3P4/3q4/5PPP/R6K b - - 0 1")
    partner = ObjectiveCalibratedSparringEngine(use_stockfish_uci=False)
    review = partner.audit_played_move_for_game_review(
        blowout_board,
        chess.Move.from_uci("d3e4"),
    )
    assert review["classification"] == "SOLID"
    assert not review["is_hazard"]
    assert not review["should_queue_for_sleep"]










