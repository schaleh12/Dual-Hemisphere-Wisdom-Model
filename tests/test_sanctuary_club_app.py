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

