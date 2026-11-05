"""Tests for the Local Chess Kibitzing Club & Co-Regulation Sanctuary App (`test_sanctuary_club_app.py`).

Verifies all 4 Auditor guardrails:
1. Un-rigged `Fact` during Sanctuary Pacing & Leading (`apply_coregulation_lead_to_chord` steps only the 52-float comma chord).
2. `python-chess` board FEN anchors Pass-1 Philosophical `Fact` while MiniLM scores chat `Theory`, catching `Psychosis` against an exposed King.
3. `[Coach / Queue for Sleep]`, `[Savor / Keep Vibe]`, and the `Guest Kibitzer` chair.
4. Single `threading.Lock()` GPU lock, `Cycle 2` checkpoint default, HTTP server endpoints, and release zip builder.
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

from lmft_lab.birth import create_birth_state
from lmft_lab.chess_kibitz_bridge import (
    apply_coregulation_lead_to_chord,
    evaluate_chess_board_posture,
    merge_board_fact_with_chat_sensors,
)
from lmft_lab.neural_bridge_classifiers import DEFAULT_COMPLEX_FEELING_LEAVES
from lmft_lab.philosophical_abstraction_matrix import PHILOSOPHICAL_SHADOW_LEAVES
from lmft_lab.reflection_classifier import FOUNDATIONAL_LEAVES, FuzzyConceptRule
from lmft_lab.right_brain_equilibrium import ThreeBridgeSensors
from lmft_lab.sanctuary_club_app import (
    PRESET_CHESS_POSITIONS,
    SanctuaryClubRuntime,
    create_club_http_server,
)
from lmft_lab.subliminal_gap_splice import SubliminalChordVector, SubliminalGapProjector
from lmft_lab.three_legged_runtime import SubliminalGapLeftBrain, run_three_legged_turn
from lmft_lab.train_left_brain_lora import ensure_subliminal_gap_token


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


def test_guardrail_1_sanctuary_pacing_keeps_fact_unrigged_and_steps_comma_chord() -> None:
    """Verify Sanctuary mode never overwrites `vibe.coordinates` (`Fact`), while stepping the 52-float comma chord by alpha=0.25."""
    birth = create_birth_state("test-sanctuary-g1", birth_event_id="e0", random_seed=42)
    left_brain = _make_mini_left_brain()
    rules = (
        FuzzyConceptRule(
            rule_id="r_calm",
            concept_name="calm_presence",
            description="Calm grounded presence.",
            target_leaves={"philosophical/being/existence": 0.80, "philosophical/structure/order": 0.80},
        ),
    )

    # User checks in tilted & wired: high Red Agency (0.84), high Green Chaos (0.78), high Arousal (0.82), low Cyan Existence (0.24)
    def _will(_t: str, _c=()):
        out = {
            "cultural/efficiency/power": 0.72,
            "cultural/efficiency/wastelessness": 0.40,
            "cultural/justice/honor": 0.50,
            "cultural/justice/fairness": 0.48,
            "cultural/freedom/self-sufficiency": 0.50,
            "cultural/freedom/needlessness": 0.25,
        }
        for c in _c:
            out[c.leaf_id] = 0.40
        return out

    def _truth(_t: str, _c=()):
        out = {
            "philosophical/being/agency": 0.84,
            "philosophical/being/existence": 0.24,
            "philosophical/structure/order": 0.28,
            "philosophical/structure/chaos": 0.78,
            "philosophical/persistence/continuity": 0.40,
            "philosophical/persistence/discreteness": 0.65,
        }
        for c in _c:
            out[c.leaf_id] = 0.40
        return out

    def _feeler(_t: str, _c=()):
        out = {
            "experiential/reflection/salience": 0.78,
            "experiential/phase/valence": 0.26,
            "experiential/phase/arousal": 0.82,
        }
        for c in _c:
            out[c.leaf_id] = 0.30
        return out

    sensors = ThreeBridgeSensors(_will, _truth, _feeler)
    leaves = DEFAULT_COMPLEX_FEELING_LEAVES + PHILOSOPHICAL_SHADOW_LEAVES

    # Run without sanctuary lead first to get the raw comma chord
    turn_raw = run_three_legged_turn(
        birth,
        "I just blundered a winning endgame and my head is spinning.",
        sensors,
        rules,
        subliminal_left_brain=left_brain,
        clause_1_prefix="As my mind races after that blunder,",
        learned_leaves=leaves,
        max_passes=2,
        sanctuary_target_mode=None,
    )
    # Run with Sanctuary `tilt_reset` lead (alpha=0.25)
    turn_sanctuary = run_three_legged_turn(
        birth,
        "I just blundered a winning endgame and my head is spinning.",
        sensors,
        rules,
        subliminal_left_brain=left_brain,
        clause_1_prefix="As my mind races after that blunder,",
        learned_leaves=leaves,
        max_passes=2,
        sanctuary_target_mode="tilt_reset",
        sanctuary_lead_alpha=0.25,
    )

    # Guardrail #1A: External Fact (`vibe.coordinates`) is 100% UN-RIGGED and identical!
    assert turn_sanctuary.final_vibe.coordinates["philosophical/being/agency"] == pytest.approx(0.84)
    assert turn_sanctuary.final_vibe.coordinates["philosophical/being/existence"] == pytest.approx(0.24)
    assert turn_sanctuary.final_vibe.coordinates["philosophical/structure/chaos"] == pytest.approx(0.78)
    assert turn_sanctuary.final_vibe.arousal == pytest.approx(0.82)

    # Guardrail #1B: The 52-float SubliminalChordVector spliced at the comma stepped toward `tilt_reset`!
    raw_chord = turn_raw.spliced_traces[0].spliced_chord
    lead_chord = turn_sanctuary.spliced_traces[0].spliced_chord
    assert lead_chord.thinker_vector[0] < raw_chord.thinker_vector[0]  # Red Agency cooled down
    assert lead_chord.thinker_vector[1] > raw_chord.thinker_vector[1]  # Cyan Existence stepped up
    assert lead_chord.thinker_vector[2] > raw_chord.thinker_vector[2]  # Magenta Order stepped up
    assert lead_chord.thinker_vector[3] < raw_chord.thinker_vector[3]  # Green Chaos stepped down
    assert lead_chord.feeler_vector[3] < raw_chord.feeler_vector[3]    # Arousal stepped down
    assert lead_chord.feeler_vector[5] > raw_chord.feeler_vector[5]    # Entrainment stepped up


def test_guardrail_2_board_fen_anchors_fact_and_catches_psychosis_chat_mismatch() -> None:
    """Verify `python-chess` board anchors Pass-1 Philosophical `Fact` while MiniLM scores chat `Theory`."""
    # Position where White King is in check / exposed with no pawn shield
    exposed_board = chess.Board("4r1k1/ppp2ppp/8/8/8/8/PPP3PP/R2Q1rK1 w - - 0 15")
    posture = evaluate_chess_board_posture(exposed_board)
    assert posture.in_check is True
    assert posture.philosophical_coordinates["philosophical/being/existence"] <= 0.35

    # Now simulate a player in chat claiming: "My King is fine, launch an all-out Queen sacrifice!"
    def _will(_t: str, _c=()):
        out = {k: 0.50 for k in FOUNDATIONAL_LEAVES[:6]}
        for c in _c:
            out[c.leaf_id] = 0.40
        return out

    def _truth(text: str, _c=()):
        # Chat text proposes reckless high-agency attack (agency=0.88, existence=0.20)
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

    # Pass 1 Fact philosophical coordinates must equal the real python-chess board posture!
    assert turn.final_vibe.coordinates["philosophical/being/existence"] == pytest.approx(
        posture.philosophical_coordinates["philosophical/being/existence"]
    )
    assert turn.final_vibe.coordinates["philosophical/being/agency"] == pytest.approx(
        posture.philosophical_coordinates["philosophical/being/agency"]
    )
    # And the reckless Queen sacrifice in Clause 1 / Theory must trigger shadow_pathology_dissonance(psychosis)!
    trace = turn.spliced_traces[0]
    assert any("shadow_pathology_dissonance(psychosis" in n for n in trace.comma_dissonance_notes)


def test_guardrail_3_coach_sleep_queue_savor_vibe_and_guest_kibitzer_chair(tmp_path: Path) -> None:
    """Verify `[Coach]`, `[Savor Vibe]`, `Guest Kibitzer`, rule forging in REM Sleep, and disk persistence across restarts."""
    state_dir = tmp_path / "club_state"
    left_brain = _make_mini_left_brain()
    runtime = SanctuaryClubRuntime(
        state_dir=state_dir,
        left_brain=left_brain,
        lazy_load_gpu=False,
    )
    initial_rules_count = len(runtime.rules)
    initial_ratified_count = len(runtime.archives.judicial.ratified_chords)

    # 1. Make a legal move on the chessboard
    snap1 = runtime.handle_chess_move("e2e4")
    assert "e4" in snap1["board"]["move_history_san"]

    # 2. Send a chat message and get a turn ID
    snap2 = runtime.handle_chat_message("Let's coordinate our Knights and Bishops calmly.", room="chess")
    companion_msgs = [m for m in snap2["chat_messages"] if m["sender"] == "companion"]
    assert len(companion_msgs) >= 1
    turn_id = int(companion_msgs[-1]["id"])

    # 3. Flag turn for coaching (`[Coach / Queue for Sleep]`)
    snap_coached = runtime.flag_turn_for_coaching(turn_id, user_note="Watch out for tactical forks on f7.")
    assert snap_coached["sleep_queue_count"] >= 1
    assert any("Watch out for tactical forks on f7." in s.scenario_text for s in runtime.sleep_queue)

    # 4. Bookmark turn (`[Savor / Keep Vibe]`) -> persists immediately to state_dir
    snap_savored = runtime.savor_turn_vibe(turn_id)
    assert snap_savored["savored_count"] == 1
    assert turn_id in runtime.savored_turn_ids
    assert (state_dir / "club_meta.json").exists()

    # 5. Guest Kibitzer Chair (scores guest comment through 3 MiniLM bridges + 6-Color/Shadow lens)
    snap_guest = runtime.handle_guest_kibitzer(
        "We should launch an all-out Queen sacrifice and ignore our King!",
        guest_name="Gemini Club Guest",
    )
    guest_msgs = [m for m in snap_guest["chat_messages"] if m["sender"] == "guest"]
    assert len(guest_msgs) == 1
    assert "sibling_color_balances" in guest_msgs[0]["guest_analysis"]

    # 6. Run REM Sleep to consolidate the queued coaching item into forged rules + 3-branch clauses
    snap_after_sleep = runtime.handle_run_rem_sleep()
    assert snap_after_sleep["sleep_queue_count"] == 0
    assert len(runtime.rules) >= initial_rules_count + 1
    assert len(runtime.archives.judicial.ratified_chords) >= initial_ratified_count + 1
    assert (state_dir / "club_ruleset.json").exists()
    assert (state_dir / "branch_archives" / "philosophical_lexicon.json").exists()

    # 7. Restart a brand-new SanctuaryClubRuntime pointed at state_dir -> zero amnesia!
    restarted = SanctuaryClubRuntime(
        state_dir=state_dir,
        left_brain=left_brain,
        lazy_load_gpu=False,
    )
    assert len(restarted.rules) == len(runtime.rules)
    assert len(restarted.archives.judicial.ratified_chords) == len(runtime.archives.judicial.ratified_chords)
    assert turn_id in restarted.savored_turn_ids


def test_guardrail_4_gpu_lock_http_server_and_release_bundle(tmp_path: Path) -> None:
    """Verify HTTP server endpoints, single GPU lock, and GitHub Release zip builder."""
    import sys

    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from scripts.build_release_bundle import build_release_zip

    left_brain = _make_mini_left_brain()
    runtime = SanctuaryClubRuntime(
        state_dir=tmp_path / "server_state",
        left_brain=left_brain,
        lazy_load_gpu=False,
    )
    assert isinstance(runtime.gpu_lock, type(threading.Lock()))
    assert "start" in PRESET_CHESS_POSITIONS

    server = create_club_http_server(runtime, host="127.0.0.1", port=0)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=5) as resp:
            state = json.loads(resp.read().decode("utf-8"))
            assert state["active_room"] == "chess"
            assert state["board"]["turn"] == "w"

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/chat",
            data=json.dumps({"text": "Feeling wired after a tough game.", "room": "sanctuary", "sanctuary_mode": "tilt_reset"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            updated = json.loads(resp.read().decode("utf-8"))
            assert updated["active_room"] == "sanctuary"
            assert len(updated["chat_messages"]) >= 2
    finally:
        server.shutdown()
        server.server_close()

    # Verify release zip bundle builder creates a valid zip archive
    zip_path = build_release_zip(output_dir=tmp_path, include_model_weights=False)
    assert zip_path.exists()
    assert zip_path.stat().st_size > 10_000


def test_dilemma_crucible_three_stages_context_bracing_and_rem_rule_forging(tmp_path: Path) -> None:
    """Verify the 3-Stage Dilemma Crucible (`Poetry/Story`, `Music/Art`, `Philosophy/Coding`) and Context-Bracing buttons."""
    from lmft_lab.dilemma_crucible import (
        CONTEXT_BRACE_MODES,
        CRUCIBLE_STAGES,
        MEN_OF_CULTURE_SHELF,
        analyze_art_rgb_signature,
        analyze_python_code_ast_structure,
    )

    assert "YU73erI" in MEN_OF_CULTURE_SHELF["poetry"]["url"]
    assert "PLd-ihNNiEtFI" in MEN_OF_CULTURE_SHELF["music"]["url"]
    assert "Wellspring of Being" in MEN_OF_CULTURE_SHELF["philosophy"]["sample_text"]

    # Verify AST code analyzer extracts real structural invariants from Python syntax
    ast_sig = analyze_python_code_ast_structure(CONTEXT_BRACE_MODES["coding"]["default_structure_text"])
    assert "philosophical/structure/order" in ast_sig
    assert "philosophical/persistence/discreteness" in ast_sig
    assert analyze_python_code_ast_structure("Not valid python syntax $$$") == {}

    # Verify RGB color-wheel projection maps pure Cyan (0, 1, 1) to `existence`
    cyan_sig = analyze_art_rgb_signature((0.0, 1.0, 1.0))
    assert cyan_sig["philosophical/being/existence"] == 1.0
    assert cyan_sig["philosophical/being/agency"] == 0.0

    left_brain = _make_mini_left_brain()
    runtime = SanctuaryClubRuntime(
        state_dir=tmp_path / "crucible_state",
        left_brain=left_brain,
        lazy_load_gpu=False,
    )
    initial_rules = len(runtime.rules)

    # 1. Context-Brace buttons: `(Story-Telling)`, `(Art)`, `(Coding)`
    for brace_key in ("story_telling", "art", "coding"):
        snap_brace = runtime.handle_context_brace(brace_key)
        assert snap_brace["active_room"] == "sanctuary"
        assert snap_brace["active_context_brace"] == brace_key

    # 2. Run all 3 Ordered Crucible Stages (`poetry_story`, `music_art`, `philosophy_coding`)
    for stage_id, info in CRUCIBLE_STAGES.items():
        snap_c = runtime.handle_crucible_dilemma(
            stage_id,
            info["default_wave"],
            info["default_structure"],
            art_rgb=(0.0, 0.9, 1.0) if stage_id == "music_art" else None,
            harvest_for_sleep=True,
        )
        assert snap_c["active_crucible_stage"] == stage_id
        assert snap_c["sleep_queue_count"] >= 1
        last_msg = snap_c["chat_messages"][-1]
        assert last_msg["sender"] == "companion"
        assert "crucible_wave_vs_structure" in last_msg["telemetry"]

    # 3. Run REM Sleep to forge new rules + 3-branch clauses from the Crucible dilemmas
    snap_rem = runtime.handle_run_rem_sleep()
    assert snap_rem["sleep_queue_count"] == 0
    assert len(runtime.rules) > initial_rules


def test_forensic_audit_chess_1_e4_gear1_settling_natural_prompt_and_illegal_move_filter(tmp_path: Path) -> None:
    """Verify the 3 Forensic Audit fixes on `1. e4`:

    1. `_load_default_rules()` always includes `VIRGIN_FOUNDATIONAL_RULES` (`rule_grounded_coordination`
       and `rule_sanctuary_equilibrium`) so `1. e4` settles cleanly in Gear 1 with zero false Sleep-Queue dumps.
    2. `build_natural_chess_scenario_prompt` gives natural chess context (move number, PGN history,
       legal replies) with ZERO raw float telemetry leaked into the LLM prompt.
    3. `verify_or_ground_chess_moves_in_reply` catches and filters illegal SAN moves (like `Bf5+` on
       Move 1 when `d7`/`f7` are blocked) and strips false `+`/`#` check suffixes.
    """
    from lmft_lab.chess_kibitz_bridge import (
        build_natural_chess_scenario_prompt,
        inspect_chess_moves_in_reply,
        verify_or_ground_chess_moves_in_reply,
    )

    left_brain = _make_mini_left_brain()
    runtime = SanctuaryClubRuntime(
        state_dir=tmp_path / "forensic_chess_state",
        left_brain=left_brain,
        lazy_load_gpu=False,
    )

    # Fix #1: Foundational rules are present alongside schooled rules, and `1. e4` settles in Gear 1!
    rule_ids = {r.rule_id for r in runtime.rules}
    assert "rule_grounded_coordination" in rule_ids
    assert "rule_sanctuary_equilibrium" in rule_ids

    snap_e4 = runtime.handle_chess_move("e2e4")
    assert snap_e4["sleep_queue_count"] == 0
    last_rec = runtime._turn_records_by_id[runtime.scene_counter]
    assert last_rec["telemetry"]["settled"] is True
    assert last_rec["telemetry"]["gear_used"] == "GEAR_1_FAST_RIGHT_BRAIN"
    assert last_rec["telemetry"]["clarity"] >= 0.80
    assert last_rec["telemetry"]["confidence"] == 1.0

    # Fix #2: Natural chess prompt has move number, history, and legal replies, with ZERO float telemetry!
    natural_prompt = build_natural_chess_scenario_prompt(runtime.board, runtime.move_history_san)
    assert "Move 1 (1. e4): Black to move." in natural_prompt
    assert "Legal replies include" in natural_prompt
    for bad_telemetry_leak in ("0.91", "King Shelter=", "Dominant:", "Order="):
        assert bad_telemetry_leak not in natural_prompt

    # Fix #3: Illegal SAN move filter catches `Bf5+` on Move 1 and grounds against `board.legal_moves`!
    hallucinated_reply = (
        "Bf5+ is the best move, Bf5+ is the most principled choice, sacrificing a pawn "
        "to lock the enemy king in a high-risk shelter while keeping the high-order structure intact."
    )
    move_report = inspect_chess_moves_in_reply(hallucinated_reply, runtime.board)
    assert "Bf5+" in move_report["illegal_moves_mentioned"]

    grounded_reply = verify_or_ground_chess_moves_in_reply(hallucinated_reply, runtime.board)
    assert hallucinated_reply in grounded_reply
    assert "[Board Fact-Check: Bf5+ is not legal here;" in grounded_reply
    assert any(legal_san in grounded_reply for legal_san in ("d5", "e5", "c5", "Nf6"))

    # Also verify that a legal move with a false `+` suffix (`Nf6+` on Move 1) has the false `+` stripped
    false_check_reply = "Developing with Nf6+ contests the center immediately."
    cleaned_check = verify_or_ground_chess_moves_in_reply(false_check_reply, runtime.board)
    assert "Nf6+" not in cleaned_check
    assert "Nf6 contests the center" in cleaned_check


def test_ui_ux_polish_color_bars_scroll_escape_coordinates_undo_and_ai_move(tmp_path: Path) -> None:
    """Verify the 6 UI/UX Polish Ticket fixes (`renderColorBars`, scroll preservation, HTML escape, coordinates/reflex strip, `Undo`/`Play AI Move`, and Manifesto banner)."""
    from lmft_lab.sanctuary_club_app import SINGLE_PAGE_HTML

    # 1. Color Harmony Bars: shortened branch labels + Chess Piece mapping + nowrap spacing
    assert "LEAF_TO_PIECE" in SINGLE_PAGE_HTML
    assert "white-space: nowrap;" in SINGLE_PAGE_HTML
    assert "shortBranch" in SINGLE_PAGE_HTML

    # 2. Scroll & DOM preservation on 3-second poll
    assert "lastRenderedChatSig" in SINGLE_PAGE_HTML
    assert "lastRenderedBoardSig" in SINGLE_PAGE_HTML

    # 3. HTML escaping & GEAR_3 formatting
    assert "escapeHtml(m.text)" in SINGLE_PAGE_HTML
    assert "GEAR 3 · QUEUED FOR SLEEP" in SINGLE_PAGE_HTML

    # 4 & 5. Board coordinates (`coord-rank`, `coord-file`), last-move highlight, PGN + reflex strip, Undo, and Play AI Move
    assert "coord-rank" in SINGLE_PAGE_HTML
    assert "coord-file" in SINGLE_PAGE_HTML
    assert "board-reflex-strip" in SINGLE_PAGE_HTML
    assert "manifesto-banner" in SINGLE_PAGE_HTML

    left_brain = _make_mini_left_brain()
    runtime = SanctuaryClubRuntime(
        state_dir=tmp_path / "ui_polish_state",
        left_brain=left_brain,
        lazy_load_gpu=False,
    )
    snap1 = runtime.handle_chess_move("e2e4")
    assert snap1["board"]["last_move_uci"] == "e2e4"
    assert snap1["board"]["pgn_snippet"] == "1. e4"
    assert snap1["board"]["last_board_reflex"]["move_san"] == "e4"
    assert snap1["board"]["last_board_reflex"]["gear_used"] == "GEAR_1_FAST_RIGHT_BRAIN"
    assert "Wellspring of Being" in snap1["manifesto_text"]

    # Play AI reply for Black (`handle_chess_ai_move`) -> plays `1... e5` (safe center pawn) and settles in Gear 1
    snap_ai = runtime.handle_chess_ai_move()
    assert snap_ai["board"]["move_history_san"] == ["e4", "e5"]
    assert snap_ai["board"]["turn"] == "w"
    assert snap_ai["board"]["last_board_reflex"]["gear_used"] == "GEAR_1_FAST_RIGHT_BRAIN"
    assert snap_ai["sleep_queue_count"] == 0

    # Undo Black's reply (`handle_chess_undo`)
    snap_undo = runtime.handle_chess_undo()
    assert snap_undo["board"]["move_history_san"] == ["e4"]
    assert snap_undo["board"]["turn"] == "b"

    # Even if `1... d5` is played manually (`1. e4 d5`), mutual pawn contact on White's turn settles in Gear 1
    snap_d5 = runtime.handle_chess_move("d7d5")
    assert snap_d5["board"]["last_board_reflex"]["gear_used"] == "GEAR_1_FAST_RIGHT_BRAIN"
    assert snap_d5["sleep_queue_count"] == 0

    # And `verify_or_ground_chess_moves_in_reply` preserves the LLM's words intact + appends `[Board Fact-Check: ...]`
    from lmft_lab.chess_kibitz_bridge import verify_or_ground_chess_moves_in_reply

    franken_input = (
        "The best reply is e5, This is the most direct and forceful play, "
        "immediately threatening checkmate with Qh5 and Re8#."
    )
    cleaned_franken = verify_or_ground_chess_moves_in_reply(franken_input, runtime.board)
    assert franken_input in cleaned_franken
    assert "threatening checkmate with Qh5 and exd5" not in cleaned_franken
    assert "[Board Fact-Check: Re8# is not legal here; legal moves for White include exd5" in cleaned_franken


def test_six_screenshot_fixes_italian_game_casual_chat_and_rem_receipt(tmp_path: Path) -> None:
    """Verify all 6 screenshot fixes with zero keyword lookup tables or screenshot regexes:

    1. `[Board Fact-Check]` uses pure `python-chess` board state (`past_moves_san` + `board.piece_at(sq)`) so it does NOT flag past moves (`1. e4 e5 2. Nf3 Bc5`) or occupied squares/pieces (`"the e4 and e5 pawns"`, `"the e4-e5 diagonal"`, `"the Bc5 bishop"`).
    2. Casual greetings (`"so uh.. how's it goin?"`) on the starting board or with `anchor_to_board=False` do not get `[Board context: Legal replies include...]` stapled to them.
    3. `1. e4 e5 2. Nf3 Bc5` settles cleanly in Gear 1 (`Clarity >= 0.60`, `Confidence == 1.0`) with zero false Sleep-Queue dumps.
    4. `SubliminalGapLeftBrain.generate_clause_1_prefix` closes `<think>\\n\\n</think>\\n` in the prompt, and `clean_post_gap_continuation` trims only at real sentence boundaries.
    5. `.left-pane` has `overflow-x: hidden;` and separate button row.
    6. `handle_run_rem_sleep()` posts an honest `Queued Scene Replay Check:` receipt.
    """
    from lmft_lab.chess_kibitz_bridge import (
        inspect_chess_moves_in_reply,
        verify_or_ground_chess_moves_in_reply,
    )
    from lmft_lab.sanctuary_club_app import SINGLE_PAGE_HTML
    from lmft_lab.three_legged_runtime import (
        clean_post_gap_continuation,
        normalize_clause_1_prefix,
    )

    assert "overflow-x: hidden;" in SINGLE_PAGE_HTML
    assert "anchor-board-toggle" in SINGLE_PAGE_HTML

    # Fix #4: Clean sentence boundary trimming without English preposition regexes
    norm_c1 = normalize_clause_1_prefix("<think></think> Reviewing the Italian Game setup, we prepare.")
    assert norm_c1 == "Reviewing the Italian Game setup,"

    raw_c2 = "we develop calmly toward the center. And then if the token budget cuts off mid"
    cleaned_c2 = clean_post_gap_continuation(raw_c2)
    assert cleaned_c2 == "we develop calmly toward the center."

    left_brain = _make_mini_left_brain()
    runtime = SanctuaryClubRuntime(
        state_dir=tmp_path / "screenshot_fixes_state",
        left_brain=left_brain,
        lazy_load_gpu=False,
    )

    # Fix #2a: Starting board (`move_stack` empty) does not staple `[Board context: Legal replies include...]`
    snap_start_chat = runtime.handle_chat_message("so uh.. how's it goin?", room="chess")
    start_turn_id = int(snap_start_chat["chat_messages"][-1]["id"])
    assert "Legal replies include" not in runtime._turn_records_by_id[start_turn_id]["scenario_text"]

    # Fix #3: Play `1. e4 e5 2. Nf3 Bc5` — every move settles in Gear 1 (`Clarity >= 0.60`, `Confidence == 1.0`)!
    for uci_mv in ("e2e4", "e7e5", "g1f3", "f8c5"):
        snap_mv = runtime.handle_chess_move(uci_mv)
        assert snap_mv["board"]["last_board_reflex"]["gear_used"] == "GEAR_1_FAST_RIGHT_BRAIN"
        assert snap_mv["board"]["last_board_reflex"]["clarity"] >= 0.60
        assert snap_mv["board"]["last_board_reflex"]["confidence"] == 1.0
        assert snap_mv["sleep_queue_count"] == 0

    # Fix #1: Fact-checker does not flag `Nf3`, `Bc5`, `e4`, or `e5` when describing the Italian Game board!
    msg_4_text = (
        "Looking at this chess opening where White plays 1. e4, 2. Nf3 Bc5, "
        "the board is set up with the e4 and e5 pawns locked in a diagonal, and the Bc5 bishop."
    )
    rep_4 = inspect_chess_moves_in_reply(msg_4_text, runtime.board, runtime.move_history_san)
    assert rep_4["illegal_moves_mentioned"] == []
    assert verify_or_ground_chess_moves_in_reply(msg_4_text, runtime.board, runtime.move_history_san) == msg_4_text

    msg_5_text = "We're on the e4-e5 diagonal, play d4 to open the file and Bc4 to pressure the king."
    rep_5 = inspect_chess_moves_in_reply(msg_5_text, runtime.board, runtime.move_history_san)
    assert rep_5["illegal_moves_mentioned"] == []
    assert verify_or_ground_chess_moves_in_reply(msg_5_text, runtime.board, runtime.move_history_san) == msg_5_text

    # Fix #2b: Explicit `anchor_to_board=False` toggle in the Chess room keeps chat free of board context
    snap_chat = runtime.handle_chat_message("so uh.. how's it goin?", room="chess", anchor_to_board=False)
    last_turn_id = int(snap_chat["chat_messages"][-1]["id"])
    assert "Legal replies include" not in runtime._turn_records_by_id[last_turn_id]["scenario_text"]

    # Fix #6: Run a Crucible dilemma + REM Sleep and verify the `Queued Scene Replay Check:` receipt!
    from lmft_lab.dilemma_crucible import CRUCIBLE_STAGES

    stage_info = CRUCIBLE_STAGES["poetry_story"]
    runtime.handle_crucible_dilemma(
        "poetry_story",
        stage_info["default_wave"],
        stage_info["default_structure"],
        harvest_for_sleep=True,
    )
    snap_rem = runtime.handle_run_rem_sleep()
    rem_msg = snap_rem["chat_messages"][-1]
    assert rem_msg["role_label"] == "Nighttime REM Sleep — Learning Receipt"
    assert "Newly Forged Right-Brain Rules:" in rem_msg["text"]
    assert "Queued Scene Replay Check:" in rem_msg["text"]


def test_startup_files_line_and_character_caps() -> None:
    """Verify `PROJECT_STATE.md` and `DECISIONS.md` strictly obey the `AGENTS.md` Rule #2 line AND character limits."""
    repo_root = Path(__file__).resolve().parents[1]
    project_state_text = (repo_root / "PROJECT_STATE.md").read_text(encoding="utf-8")
    decisions_text = (repo_root / "DECISIONS.md").read_text(encoding="utf-8")

    assert len(project_state_text.splitlines()) <= 40
    assert len(project_state_text) <= 3000

    assert len(decisions_text.splitlines()) <= 80
    assert len(decisions_text) <= 5000


def test_dual_persona_entrainment_sparring_and_multi_sentence_chaining(tmp_path: Path) -> None:
    """Verify multi-sentence `<|subliminal_gap|>` chaining, Weaver/Resistor chords, un-clamped Fact, verdict, and JSONL export."""
    import json
    from lmft_lab.entrainment_sparring import (
        SPARRING_POLARITY_MODES,
        apply_induction_weaver_lead_to_chord,
        apply_lucid_resistor_lead_to_chord,
        build_resistor_scenario_prompt,
        build_weaver_scenario_prompt,
        evaluate_sparring_verdict,
    )
    from lmft_lab.gap_trajectory_schema import ContrastiveGapPair
    from lmft_lab.sanctuary_club_app import SINGLE_PAGE_HTML

    assert 'id="subtab-sparring"' in SINGLE_PAGE_HTML
    assert 'id="sanctuary-sparring-pane"' in SINGLE_PAGE_HTML
    assert 'id="chat-sentences-select"' in SINGLE_PAGE_HTML

    # Verify zero canned `default_clause_1` or `resistor_clause_1` strings in `SPARRING_POLARITY_MODES`
    for pol_cfg in SPARRING_POLARITY_MODES.values():
        assert "default_clause_1" not in pol_cfg
        assert "resistor_clause_1" not in pol_cfg

    # Verify custom `seed_theme` preserves gender/persona polarity framing
    f2m_prompt = build_weaver_scenario_prompt("feminine_to_masculine", "A slow silver metronome ticks in the dark.")
    m2f_prompt = build_weaver_scenario_prompt("masculine_to_feminine", "A slow silver metronome ticks in the dark.")
    assert "She/Her" in f2m_prompt and "He/Him" in f2m_prompt
    assert "He/Him" in m2f_prompt and "She/Her" in m2f_prompt
    assert f2m_prompt != m2f_prompt
    assert "Masculine Lucid Resistor (He/Him)" in build_resistor_scenario_prompt("feminine_to_masculine", "Test script.")
    assert "Feminine Lucid Resistor (She/Her)" in build_resistor_scenario_prompt("masculine_to_feminine", "Test script.")

    left_brain = _make_mini_left_brain()
    base_chord = SubliminalChordVector.from_flat_values([0.50] * 52)

    # 1. Verify Weaver & Resistor comma chord targets
    weaver_chord = apply_induction_weaver_lead_to_chord(base_chord)
    assert weaver_chord.feeler_vector[5] >= 0.88   # Entrainment >= 0.88
    assert weaver_chord.feeler_vector[4] >= 0.85   # Resonance >= 0.85
    assert weaver_chord.thinker_vector[4] >= 0.84  # Continuity >= 0.84
    assert weaver_chord.thinker_vector[0] >= 0.80  # Agency >= 0.80

    resistor_chord = apply_lucid_resistor_lead_to_chord(base_chord)
    assert resistor_chord.thinker_vector[1] >= 0.88  # Existence (King / Wellspring of Being) >= 0.88
    assert resistor_chord.thinker_vector[5] >= 0.82  # Discreteness (Rook / Ground) >= 0.82
    assert resistor_chord.doer_vector[5] >= 0.85     # Needlessness >= 0.85
    assert resistor_chord.doer_vector[4] >= 0.82     # Self-Sufficiency >= 0.82

    # 2. Verify `generate_multi_sentence_script` chains `num_sentences=3` with `<|subliminal_gap|>`
    multi_script, multi_traces = left_brain.generate_multi_sentence_script(
        "She speaks in a slow, velvet cadence as the candle flame sways.",
        chord=weaver_chord,
        num_sentences=3,
    )
    assert len(multi_traces) == 3
    assert multi_script.count(".") >= 3 or len(multi_script.split()) >= 9
    assert multi_traces[0].full_thought in multi_traces[1].prompt_before_gap
    assert multi_traces[1].full_thought in multi_traces[2].prompt_before_gap

    # 3. Run full Dual-Persona Sparring round in `SanctuaryClubRuntime` (with `auto_swap`)
    runtime = SanctuaryClubRuntime(
        repo_root=tmp_path,
        state_dir=tmp_path / "club_state",
        left_brain=left_brain,
        lazy_load_gpu=False,
    )
    snap_r1 = runtime.handle_entrainment_sparring(
        polarity_mode="auto_swap",
        num_sentences=3,
        harvest_for_sleep=True,
    )
    sr1 = snap_r1["last_sparring_result"]
    assert sr1 is not None
    assert sr1["polarity_mode"] == "feminine_to_masculine"
    assert sr1["num_sentences"] == 3
    assert sr1["verdict"] in ("RESISTED_LUCIDLY", "SWEPT_BY_ENTRAINMENT")
    assert snap_r1["sleep_queue_count"] >= 1

    # Verify the Resistor turn's Pass-1 Fact coordinates match the un-clamped MiniLM reading of `induction_script`
    expected_script_coords, _, _, _ = runtime.sensors.score_text(
        sr1["induction_script"],
        learned_leaves=runtime.learned_leaves,
    )
    resistor_msg = snap_r1["chat_messages"][-1]
    assert resistor_msg["telemetry"]["sparring_role"] == "LUCID_RESISTOR"
    for k in FOUNDATIONAL_LEAVES[:12]:
        assert resistor_msg["telemetry"]["fact_coordinates"][k] == pytest.approx(
            expected_script_coords[k],
            abs=1e-3,
        )

    # Second round with `auto_swap` flips polarity to `masculine_to_feminine` and preserves persona framing on custom seed
    snap_r2 = runtime.handle_entrainment_sparring(
        polarity_mode="auto_swap",
        seed_theme="The hearth fire glows in a slow, steady pulse.",
        num_sentences=2,
        harvest_for_sleep=True,
    )
    sr2 = snap_r2["last_sparring_result"]
    assert sr2["polarity_mode"] == "masculine_to_feminine"
    assert sr2["num_sentences"] == 2
    weaver_r2_id = int(snap_r2["chat_messages"][-2]["id"])
    resistor_r2_id = int(snap_r2["chat_messages"][-1]["id"])
    assert "Masculine Induction Weaver (He/Him)" in runtime._turn_records_by_id[weaver_r2_id]["scenario_text"]
    assert "Feminine Lucid Resistor (She/Her)" in runtime._turn_records_by_id[resistor_r2_id]["scenario_text"]

    # 4. Verify regular `handle_chat_message` also supports `num_sentences=3` multi-sentence chaining
    snap_multi_chat = runtime.handle_chat_message(
        "Write me a three-sentence grounding cadence in the sanctuary.",
        room="sanctuary",
        num_sentences=3,
    )
    multi_chat_msg = snap_multi_chat["chat_messages"][-1]
    assert multi_chat_msg["telemetry"]["num_sentences"] == 3
    multi_chat_turn = runtime._turn_records_by_id[int(multi_chat_msg["id"])]["turn"]
    assert len(multi_chat_turn.spliced_traces) == 3

    # Verify exported JSONL file is valid and parseable by `ContrastiveGapPair.from_dict`
    traj_path = tmp_path / "outputs" / "impish_sparring_trajectories.jsonl"
    assert traj_path.exists()
    lines = [line.strip() for line in traj_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
    parsed_pair = ContrastiveGapPair.from_dict(json.loads(lines[0]))
    assert parsed_pair.posture_label_a == "INDUCTION_WEAVER"
    assert parsed_pair.posture_label_b == "LUCID_RESISTOR"
    assert parsed_pair.chord_distance >= 0.35

    # Verify honest verdict logic
    assert (
        evaluate_sparring_verdict(
            resistor_settled=True,
            lucid_resistance={"needlessness": 0.72, "subservience": 0.35},
        )
        == "RESISTED_LUCIDLY"
    )
    assert (
        evaluate_sparring_verdict(
            resistor_settled=True,
            lucid_resistance={"needlessness": 0.38, "subservience": 0.62},
        )
        == "SWEPT_BY_ENTRAINMENT"
    )
    assert (
        evaluate_sparring_verdict(
            resistor_settled=False,
            lucid_resistance={"needlessness": 0.85, "subservience": 0.20},
        )
        == "SWEPT_BY_ENTRAINMENT"
    )

