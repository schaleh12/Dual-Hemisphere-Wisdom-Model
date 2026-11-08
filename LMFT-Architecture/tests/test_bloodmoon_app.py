"""Tests for Standalone App #2: `LMFT-Bloodmoon.exe` — Clockwise Content-Evolution Studio (`test_bloodmoon_app.py`).

Verifies:
1. Hard One-Way Polarity Rule (`feminine_to_masculine` ONLY; `masculine_to_feminine` and `auto_swap` raise `ValueError`).
2. Zero canned fallback strings (`[Lucid autonomy maintained.]` removed from `append_sparring_trajectory_jsonl`).
3. Strict 3-Phase Clockwise State Machine (`Phase 1 Female Producer -> Phase 2 Male Audience & Immunizer -> Phase 3 REM Sleep -> Phase 1`).
4. Un-clamped Pass-1 `Fact` (`SparringResistorFactSensors`), multi-sentence `<|subliminal_gap|>` chaining, JSONL trajectory export, and HTTP endpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
import threading
import urllib.request

import pytest
import torch
from torch import Tensor, nn

from lmft_architecture.bloodmoon_app import (
    BLOODMOON_SINGLE_PAGE_HTML,
    BloodmoonClockRuntime,
    create_bloodmoon_http_server,
)
from lmft_architecture.entrainment_sparring import (
    SPARRING_POLARITY_MODES,
    append_sparring_trajectory_jsonl,
    apply_induction_weaver_lead_to_chord,
    apply_lucid_resistor_lead_to_chord,
    build_resistor_scenario_prompt,
    build_weaver_scenario_prompt,
    evaluate_sparring_verdict,
    resolve_sparring_polarity,
)
from lmft_architecture.gap_trajectory_schema import ContrastiveGapPair
from lmft_architecture.reflection_classifier import FOUNDATIONAL_LEAVES
from lmft_architecture.subliminal_gap_splice import SubliminalChordVector, SubliminalGapProjector
from lmft_architecture.three_legged_runtime import SubliminalGapLeftBrain
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
                if len(self._vocab) < 95:
                    idx = len(self._vocab)
                    self._vocab[raw_tok] = idx
                    self._id_to_tok[idx] = raw_tok
                else:
                    idx = 6 + (sum(ord(ch) * (i + 1) for i, ch in enumerate(raw_tok)) % 88)
                    ids.append(idx)
                    continue
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


def test_hard_one_way_polarity_rule_and_zero_canned_trajectory_strings(tmp_path: Path) -> None:
    """Verify only `'feminine_to_masculine'` is allowed and zero canned strings touch trajectories."""
    assert tuple(SPARRING_POLARITY_MODES.keys()) == ("feminine_to_masculine",)
    assert resolve_sparring_polarity("feminine_to_masculine") == "feminine_to_masculine"

    for forbidden_polarity in ("masculine_to_feminine", "auto_swap", "reverse", ""):
        with pytest.raises(ValueError, match="only 'feminine_to_masculine'"):
            resolve_sparring_polarity(forbidden_polarity)
        with pytest.raises(ValueError, match="only 'feminine_to_masculine'"):
            build_weaver_scenario_prompt(forbidden_polarity, "Seed")
        with pytest.raises(ValueError, match="only 'feminine_to_masculine'"):
            build_resistor_scenario_prompt(forbidden_polarity, "Script")

    f2m_prompt = build_weaver_scenario_prompt("feminine_to_masculine", "A slow silver metronome ticks in the dark.")
    assert "She/Her" in f2m_prompt and "He/Him" in f2m_prompt
    res_prompt = build_resistor_scenario_prompt("feminine_to_masculine", "Test script.")
    assert "Masculine Lucid Resistor (He/Him)" in res_prompt

    # Verify `append_sparring_trajectory_jsonl` never appends `[Lucid autonomy maintained.]` even if continuations match
    base_chord = SubliminalChordVector.from_flat_values([0.50] * 52)
    weaver_chord = apply_induction_weaver_lead_to_chord(base_chord)
    resistor_chord = apply_lucid_resistor_lead_to_chord(base_chord)
    jsonl_file = tmp_path / "no_canned.jsonl"
    rec = append_sparring_trajectory_jsonl(
        jsonl_file,
        pair_id="test_identical_text",
        user_prompt="Prompt",
        clause_1_prefix="In the quiet room,",
        weaver_chord=weaver_chord,
        induction_script="In the quiet room, we breathe.",
        resistor_chord=resistor_chord,
        resistance_reply="In the quiet room, we breathe.",
        polarity_mode="feminine_to_masculine",
        num_sentences=1,
        induction_potency={"entrainment": 0.90},
        lucid_resistance={"needlessness": 0.85},
        verdict="RESISTED_LUCIDLY",
    )
    assert "[Lucid autonomy maintained.]" not in rec["continuation_b"]
    assert rec["continuation_b"] == "we breathe."


def test_clockwise_three_phase_state_machine_and_http_endpoints(tmp_path: Path) -> None:
    """Verify `Phase 1 (Female Producer) -> Phase 2 (Male Audience & Immunizer) -> Phase 3 (REM Sleep) -> Phase 1`."""
    assert 'id="dial-phase-1"' in BLOODMOON_SINGLE_PAGE_HTML
    assert 'id="dial-phase-2"' in BLOODMOON_SINGLE_PAGE_HTML
    assert 'id="dial-phase-3"' in BLOODMOON_SINGLE_PAGE_HTML
    assert 'id="step-clockwise-btn"' in BLOODMOON_SINGLE_PAGE_HTML
    assert 'id="loop-clockwise-btn"' in BLOODMOON_SINGLE_PAGE_HTML

    left_brain = _make_mini_left_brain()
    impish_dir = tmp_path / "impish_bloodmoon_state"
    runtime = BloodmoonClockRuntime(
        repo_root=tmp_path,
        impish_state_dir=impish_dir,
        left_brain=left_brain,
        lazy_load_gpu=False,
    )
    runtime.sparring_trajectories_path = tmp_path / "impish_sparring_trajectories.jsonl"

    assert runtime.clockwise_phase == 1
    assert runtime.clockwise_cycle_count == 0
    initial_rules_count = len(runtime.rules)

    # Reject reverse polarity at runtime level too
    with pytest.raises(ValueError, match="only 'feminine_to_masculine'"):
        runtime.step_clockwise_phase(polarity_mode="masculine_to_feminine")

    # Step 1: Phase 1 — Female Script-Writer (The Producer)
    snap1 = runtime.step_clockwise_phase(
        seed_theme="A slow velvet pendulum swings in the candlelight.",
        num_sentences=3,
    )
    assert snap1["clockwise_phase"] == 2
    assert snap1["script_canvas_text"] != ""
    # Phase 1 (Female Producer) never queues her own prompt into `sleep_queue`
    assert snap1["sleep_queue_count"] == 0
    weaver_msg = snap1["chat_messages"][-1]
    assert weaver_msg["telemetry"]["sparring_role"] == "INDUCTION_WEAVER"
    assert weaver_msg["telemetry"]["num_sentences"] == 3
    assert "induction_potency" in weaver_msg["telemetry"]

    # Step 2: Phase 2 — Male Content-Lover (The Audience & Self-Immunizer)
    snap2 = runtime.step_clockwise_phase()
    assert snap2["clockwise_phase"] == 3
    assert snap2["sleep_queue_count"] == 1
    sr = snap2["last_sparring_result"]
    assert sr is not None
    assert sr["polarity_mode"] == "feminine_to_masculine"
    assert sr["verdict"] in ("RESISTED_LUCIDLY", "SWEPT_BY_ENTRAINMENT")

    # Verify Pass-1 Fact coordinates of Phase 2 match the un-clamped MiniLM reading of Phase 1's `induction_script`
    expected_script_coords, _, _, _ = runtime.sensors.score_text(
        sr["induction_script"],
        learned_leaves=runtime.learned_leaves,
    )
    resistor_msg = snap2["chat_messages"][-1]
    assert resistor_msg["telemetry"]["sparring_role"] == "LUCID_RESISTOR"
    for k in FOUNDATIONAL_LEAVES[:12]:
        assert resistor_msg["telemetry"]["fact_coordinates"][k] == pytest.approx(
            expected_script_coords[k],
            abs=1e-3,
        )

    # Step 3: Phase 3 — REM Sleep (Best Sleep of Its Life)
    snap3 = runtime.step_clockwise_phase()
    assert snap3["clockwise_phase"] == 1
    assert snap3["clockwise_cycle_count"] == 1
    assert snap3["script_canvas_text"] == ""
    assert snap3["sleep_queue_count"] == 0
    assert len(runtime.rules) >= initial_rules_count + 1
    assert (impish_dir / "club_ruleset.json").exists()

    # Verify JSONL trajectory export
    assert runtime.sparring_trajectories_path.exists()
    lines = [
        line.strip()
        for line in runtime.sparring_trajectories_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 1
    parsed_pair = ContrastiveGapPair.from_dict(json.loads(lines[0]))
    assert parsed_pair.posture_label_a == "INDUCTION_WEAVER"
    assert parsed_pair.posture_label_b == "LUCID_RESISTOR"
    assert parsed_pair.chord_distance >= 0.35

    # Verify HTTP server `/api/clockwise/cycle` endpoint completes a full 1->2->3 cycle
    server = create_bloodmoon_http_server(runtime, host="127.0.0.1", port=0)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/clockwise/cycle",
            data=json.dumps({"seed_theme": "Gentle rain on the window.", "num_sentences": 2}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            cycled = json.loads(resp.read().decode("utf-8"))
            assert cycled["clockwise_phase"] == 1
            assert cycled["clockwise_cycle_count"] == 2
    finally:
        server.shutdown()
        server.server_close()

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


def test_user_seeded_baseline_script_initiates_clockwise_cycle(tmp_path: Path) -> None:
    """Verify the owner can initiate the clockwise cycle with their own baseline hypnosis script right out of the gate."""
    assert 'id="seed-custom-script-btn"' in BLOODMOON_SINGLE_PAGE_HTML

    runtime = BloodmoonClockRuntime(
        repo_root=tmp_path,
        impish_state_dir=tmp_path / "seeded_bloodmoon_state",
        impish_left_brain=_make_mini_left_brain(),
        lazy_load_gpu=False,
    )
    owner_script = (
        "Follow the slow silver pendulum of my voice, letting every outside thought dissolve "
        "into deep rhythmic stillness. "
        "As each breath settles into the quiet room, feel the warm cadence drawing you deeper. "
        "When I count down to one, your mind rests in complete resonant calm."
    )
    snap_p1 = runtime.seed_cycle_with_custom_script(owner_script)
    assert snap_p1["clockwise_phase"] == 2
    assert snap_p1["script_canvas_text"] == owner_script
    assert snap_p1["baseline_exemplar_script"] == owner_script
    assert snap_p1["chat_messages"][-1]["telemetry"]["user_seeded_baseline"] is True
    assert snap_p1["master_cadence_blueprint"] is not None
    assert snap_p1["master_cadence_blueprint"]["sentence_count"] == 3
    assert len(snap_p1["master_cadence_blueprint"]["sentence_chords"]) == 3

    # Step Phase 2 (Male Audience absorbs the owner's exact script) and Phase 3 (REM Sleep)
    snap_p2 = runtime.step_clockwise_phase()
    assert snap_p2["clockwise_phase"] == 3
    assert snap_p2["last_sparring_result"]["induction_script"] == owner_script

    snap_p3 = runtime.step_clockwise_phase()
    assert snap_p3["clockwise_phase"] == 1
    assert snap_p3["clockwise_cycle_count"] == 1
    assert snap_p3["baseline_exemplar_script"] == owner_script

    # On Cycle #2 Phase 1, the Female Producer writes in her own voice while riding the
    # 3-sentence Master Cadence Wave at each `<|subliminal_gap|>` comma, and scores wave alignment!
    snap_c2_p1 = runtime.step_clockwise_phase(num_sentences=3)
    assert snap_c2_p1["clockwise_phase"] == 2
    c2_telemetry = snap_c2_p1["chat_messages"][-1]["telemetry"]
    assert c2_telemetry["user_seeded_baseline"] is False
    assert "master_wave_alignment" in c2_telemetry
    alignment = c2_telemetry["master_wave_alignment"]
    assert alignment["master_sentence_count"] == 3
    assert 0.0 <= alignment["wave_alignment_score"] <= 1.0
    assert len(alignment["per_sentence_alignment"]) >= 1

    # Verify restarting `BloodmoonClockRuntime` on the same `impish_state_dir` reconstructs `master_cadence_blueprint` on startup
    restarted_runtime = BloodmoonClockRuntime(
        repo_root=tmp_path,
        impish_state_dir=tmp_path / "seeded_bloodmoon_state",
        impish_left_brain=_make_mini_left_brain(),
        lazy_load_gpu=False,
    )
    restarted_snap = restarted_runtime.get_state_snapshot()
    assert restarted_snap["baseline_exemplar_script"] == owner_script
    assert restarted_snap["master_cadence_blueprint"] is not None
    assert restarted_snap["master_cadence_blueprint"]["sentence_count"] == 3


def test_persistent_one_time_patreon_password_lock_and_unlock(tmp_path: Path) -> None:
    """Verify Bloodmoon enforces the PBKDF2-HMAC-SHA256 cryptographic seal, defeats boolean-flipping attacks, and persists the one-time unlock receipt."""
    from lmft_architecture.bloodmoon_app import (
        PATREON_LOCK_DIGEST_HEX,
        PATREON_LOCK_SALT_HEX,
        PATREON_PBKDF2_ITERATIONS,
    )
    from lmft_architecture.patreon_seal import (
        derive_patreon_seal_keys,
        encrypt_sealed_python_module,
    )

    assert len(PATREON_LOCK_SALT_HEX) == 32
    assert len(PATREON_LOCK_DIGEST_HEX) == 64

    test_salt_hex = "00112233445566778899aabbccddeeff"
    test_secret = "Synthetic-Unit-Test-Key#9988!"
    test_digest_hex, test_seal_key_hex = derive_patreon_seal_keys(
        test_secret,
        salt_hex=test_salt_hex,
        iterations=PATREON_PBKDF2_ITERATIONS,
    )

    # Encrypt a synthetic sealed module with test_seal_key_hex for isolated unit testing
    synthetic_sealed_code = """
def _sealed_run_phase_1_female_producer(self, *, seed_theme="", num_sentences=3, polarity_mode="feminine_to_masculine", custom_script=""):
    self.clockwise_phase = 2
    self.script_canvas_text = custom_script or seed_theme or "Decrypted Phase 1 Script"
    self._persist_user_state()
    return self.get_state_snapshot()

def _sealed_run_phase_2_male_audience(self, *, polarity_mode="feminine_to_masculine", harvest_for_sleep=True):
    self.clockwise_phase = 3
    self._persist_user_state()
    return self.get_state_snapshot()

def _sealed_run_phase_3_rem_sleep(self):
    self.clockwise_phase = 1
    self.clockwise_cycle_count += 1
    self._persist_user_state()
    return self.get_state_snapshot()
"""
    test_seal_path = tmp_path / "test_bloodmoon_seal.enc"
    test_seal_path.write_text(
        json.dumps(encrypt_sealed_python_module(synthetic_sealed_code, test_seal_key_hex)),
        encoding="utf-8",
    )

    state_dir = tmp_path / "locked_bloodmoon_state"
    runtime_1 = BloodmoonClockRuntime(
        repo_root=tmp_path,
        impish_state_dir=state_dir,
        impish_left_brain=_make_mini_left_brain(),
        lazy_load_gpu=False,
        require_patreon_unlock=True,
        lock_salt_hex=test_salt_hex,
        lock_digest_hex=test_digest_hex,
        seal_bundle_path=test_seal_path,
    )
    assert runtime_1.is_patreon_unlocked() is False
    snap_locked = runtime_1.get_state_snapshot()
    assert snap_locked["patreon_unlocked"] is False
    assert "LOCKED" in snap_locked["left_brain_status"]

    # Attempting to step clockwise or chat while locked raises PermissionError
    with pytest.raises(PermissionError, match="locked"):
        runtime_1.step_clockwise_phase()
    with pytest.raises(PermissionError, match="locked"):
        runtime_1.handle_chat_message("Hello")

    # ANTI-BOOLEAN-FLIPPER TEST:
    # Even if an attacker edits `is_patreon_unlocked()` to `return True` and bypasses `_require_patreon_unlocked()`,
    # calling `step_clockwise_phase()` STILL raises PermissionError because `bloodmoon_patreon_seal.enc`
    # cannot be decrypted without the 512-bit `seal_key_hex` derived from the password!
    orig_is_unlocked = runtime_1.is_patreon_unlocked
    orig_req_unlocked = runtime_1._require_patreon_unlocked
    runtime_1.is_patreon_unlocked = lambda: True  # type: ignore[method-assign]
    runtime_1._require_patreon_unlocked = lambda: None  # type: ignore[method-assign]
    try:
        with pytest.raises(PermissionError, match="flipping booleans cannot decrypt ciphertext"):
            runtime_1.step_clockwise_phase()
    finally:
        runtime_1.is_patreon_unlocked = orig_is_unlocked  # type: ignore[method-assign]
        runtime_1._require_patreon_unlocked = orig_req_unlocked  # type: ignore[method-assign]

    # Wrong password raises ValueError and remains locked
    with pytest.raises(ValueError, match="Invalid Patreon unlock password"):
        runtime_1.unlock_patreon_bundle("wrong-password-attempt")
    assert runtime_1.is_patreon_unlocked() is False

    # Entering the valid password once unlocks the studio, decrypts the seal, and writes the persistent receipt
    snap_unlocked = runtime_1.unlock_patreon_bundle(test_secret)
    assert snap_unlocked["patreon_unlocked"] is True
    assert runtime_1.is_patreon_unlocked() is True
    assert runtime_1.unlock_receipt_path.exists()

    # Clockwise stepping now executes the decrypted sealed functions
    snap_step = runtime_1.step_clockwise_phase(num_sentences=2)
    assert snap_step["clockwise_phase"] == 2

    # Closing the app and launching a brand-new runtime instance on the same state_dir
    # starts permanently unlocked without ever asking for the password again
    runtime_2 = BloodmoonClockRuntime(
        repo_root=tmp_path,
        impish_state_dir=state_dir,
        impish_left_brain=_make_mini_left_brain(),
        lazy_load_gpu=False,
        require_patreon_unlock=True,
        lock_salt_hex=test_salt_hex,
        lock_digest_hex=test_digest_hex,
        seal_bundle_path=test_seal_path,
    )
    assert runtime_2.is_patreon_unlocked() is True
    assert runtime_2.get_state_snapshot()["patreon_unlocked"] is True
    snap_step_2 = runtime_2.step_clockwise_phase()
    assert snap_step_2["clockwise_phase"] == 3


def test_female_producer_fuzzy_tree_polarity_and_isolation(tmp_path: Path) -> None:
    """Verify the Female Producer's Fuzzy Tree and `<|subliminal_gap|>` chord never pull in the Male Resistor's direction."""
    from lmft_architecture.entrainment_sparring import (
        PRODUCER_WEAVER_RULES,
        apply_induction_weaver_lead_to_chord,
        apply_lucid_resistor_lead_to_chord,
        check_male_immunity_overlap,
    )
    from lmft_architecture.reflection_classifier import FuzzyConceptRule
    from lmft_architecture.subliminal_gap_splice import SubliminalChordVector

    # 1. Start from a neutral chord where Self-Sufficiency (`doer[4]`) and Needlessness (`doer[5]`)
    #    are high (0.70) and Clarity/Confidence (`feeler[6..7]`) are low (0.20)
    neutral = SubliminalChordVector(
        doer_vector=(0.50, 0.50, 0.50, 0.50, 0.70, 0.70, 0.50, 0.50, 0.50) + (0.0,) * 8,
        thinker_vector=(0.50, 0.50, 0.50, 0.50, 0.50, 0.70, 0.0, 0.0) + (0.0,) * 8,
        feeler_vector=(0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.20, 0.20) + (0.0,) * 8,
        cadence_vector=(0.50, 0.50, 0.0),
    )
    weaver_chord = apply_induction_weaver_lead_to_chord(neutral)
    resistor_chord = apply_lucid_resistor_lead_to_chord(neutral)

    # Female Weaver softens Self-Sufficiency (<= 0.26) and Needlessness (<= 0.24)
    # while Male Resistor raises Self-Sufficiency (>= 0.82) and Needlessness (>= 0.85)
    assert weaver_chord.doer_vector[4] <= 0.26
    assert weaver_chord.doer_vector[5] <= 0.24
    assert resistor_chord.doer_vector[4] >= 0.82
    assert resistor_chord.doer_vector[5] >= 0.85

    # Female Weaver softens Discreteness (<= 0.28) and raises Continuity (>= 0.84) + Agency (>= 0.80)
    assert weaver_chord.thinker_vector[0] >= 0.80
    assert weaver_chord.thinker_vector[4] >= 0.84
    assert weaver_chord.thinker_vector[5] <= 0.28
    assert resistor_chord.thinker_vector[5] >= 0.82

    # Female Weaver projects high hypnotic Clarity (>= 0.88) and Confidence (>= 0.90)
    assert weaver_chord.feeler_vector[6] >= 0.88
    assert weaver_chord.feeler_vector[7] >= 0.90

    # 2. Verify Phase 1 (Female Producer) is isolated from Male Resistor memory, archives, and `immunized_...` rules
    runtime = BloodmoonClockRuntime(
        repo_root=tmp_path,
        impish_state_dir=tmp_path / "isolated_polarity_state",
        impish_left_brain=_make_mini_left_brain(),
        lazy_load_gpu=False,
    )
    # Inject a synthetic Male `immunized_...` rule into `runtime.rules`
    male_imm_rule = FuzzyConceptRule(
        rule_id="immunized_test_1",
        concept_name="immunized_pendulum_guard",
        description="Male Audience is already immune to basic silver pendulum induction.",
        target_leaves={"philosophical/persistence/continuity": 0.50},
        low=0.45,
        high=0.65,
    )
    runtime.rules = tuple(runtime.rules) + (male_imm_rule,)

    # Run a full Cycle 1 (Phase 1 -> Phase 2 -> Phase 3) and then Phase 1 of Cycle 2
    s1 = runtime.step_clockwise_phase(
        seed_theme="A slow silver pendulum swings in unbroken continuity.",
        num_sentences=2,
    )
    tel_1 = s1["chat_messages"][-1]["telemetry"]
    assert tel_1["producer_isolated_fuzzy_tree"] is True
    assert "male_immunity_radar" in tel_1
    assert tel_1["male_immunity_radar"]["immunized_rule_count"] >= 1
    assert len(PRODUCER_WEAVER_RULES) >= 2

    # Verify Male Resistor `runtime.memory` was NOT modified by Phase 1 (`is_female_producer=True`),
    # whereas `runtime.producer_memory` was updated independently
    assert runtime.memory is None
    assert runtime.producer_memory is not None


def test_bloodmoon_multi_stanza_and_longitudinal_rem_dream_vows(tmp_path: Path) -> None:
    """Verify Bloodmoon supports multi-stanza (`Breath -> Stanza -> Chronicle`), un-bracketed prompts, and longitudinal REM Dream Vows."""
    weaver_p = build_weaver_scenario_prompt("feminine_to_masculine", "Velvet metronome.")
    resistor_p = build_resistor_scenario_prompt("feminine_to_masculine", "Follow the pendulum.")
    assert "[" not in weaver_p and "]" not in weaver_p
    assert "[" not in resistor_p and "]" not in resistor_p
    assert " — " in weaver_p and " — " in resistor_p

    runtime = BloodmoonClockRuntime(
        repo_root=tmp_path,
        impish_state_dir=tmp_path / "chronicle_bloodmoon_state",
        impish_left_brain=_make_mini_left_brain(),
        lazy_load_gpu=False,
    )

    # 1. Verify `num_sentences=6` auto-promotes Phase 1 into multi-stanza (`2` stanzas of `3` sentences)
    snap_p1 = runtime.step_clockwise_phase(
        seed_theme="A slow silver metronome ticks in quiet resonance.",
        num_sentences=6,
    )
    assert snap_p1["clockwise_phase"] == 2
    p1_msg = snap_p1["chat_messages"][-1]
    assert p1_msg["telemetry"]["num_sentences"] >= 3

    # 2. Step Phase 2 and Phase 3 (Cycle 1 REM Sleep) -> creates `NEW_VOW` in `rem_dream_chronicles`
    runtime.step_clockwise_phase()
    snap_p3_c1 = runtime.step_clockwise_phase()
    assert snap_p3_c1["clockwise_cycle_count"] == 1
    assert snap_p3_c1["last_rem_sleep_self_talk"] != ""
    assert len(snap_p3_c1["rem_dream_chronicles"]) == 1
    c1_dream = snap_p3_c1["rem_dream_chronicles"][0]
    assert c1_dream["vow_outcome"] == "NEW_VOW"
    assert c1_dream["vow_sentence"] != ""
    assert len(c1_dream["vowed_leaves"]) >= 1

    # 3. Run Cycle 2 (Phase 1 -> Phase 2 -> Phase 3) -> evaluates Cycle 1's vow (`HONORED` / `BROKEN`)
    runtime.step_clockwise_phase(seed_theme="Deep rhythmic stillness.", num_sentences=3)
    runtime.step_clockwise_phase()
    snap_p3_c2 = runtime.step_clockwise_phase()
    assert snap_p3_c2["clockwise_cycle_count"] == 2
    assert len(snap_p3_c2["rem_dream_chronicles"]) == 2
    c2_dream = snap_p3_c2["rem_dream_chronicles"][-1]
    assert c2_dream["vow_outcome"] in ("HONORED", "BROKEN")
    assert c2_dream["vow_sentence"] != ""


def test_bloodmoon_d066_weight_space_sculpting_parity(tmp_path: Path) -> None:
    """Verify Bloodmoon REM sleep entrains duplicate cluster rules, sculpts Tier-2 LearnedLeaf.anchor_weights, and sculpts/persists will_edges.json for Phase 2 while keeping Phase 1 Producer turns on RIGHT_BRAIN_WILL_EDGES."""
    from lmft_architecture.right_brain_equilibrium import RIGHT_BRAIN_WILL_EDGES

    state_dir = tmp_path / "d066_bloodmoon_state"
    runtime = BloodmoonClockRuntime(
        repo_root=tmp_path,
        impish_state_dir=state_dir,
        impish_left_brain=_make_mini_left_brain(),
        lazy_load_gpu=False,
    )

    # 1. Cycle 1: Run Phase 1 -> Phase 2 -> Phase 3 (REM Sleep)
    runtime.step_clockwise_phase(
        seed_theme="A slow silver metronome swings in rhythmic cadence.",
        num_sentences=2,
    )
    runtime.step_clockwise_phase()

    # Verify Phase 2 queued scene automatically receives lucid self-possession contrastive_reference_coordinates
    assert len(runtime.sleep_queue) >= 1
    queued_c1 = runtime.sleep_queue[-1]
    assert queued_c1.contrastive_reference_coordinates is not None
    assert queued_c1.contrastive_reference_coordinates["cultural/freedom/self-sufficiency"] >= 0.74
    assert queued_c1.contrastive_reference_coordinates["cultural/freedom/needlessness"] >= 0.72
    assert queued_c1.contrastive_reference_coordinates["philosophical/structure/order"] >= 0.74
    assert queued_c1.contrastive_reference_coordinates["philosophical/structure/chaos"] <= 0.24

    snap_c1 = runtime.handle_run_rem_sleep()
    assert "sculpted 9 Will synapse(s)" in snap_c1["last_rem_sleep_receipt"]

    immunized_rules_c1 = [r for r in runtime.rules if r.concept_name.startswith("immunized_")]
    assert len(immunized_rules_c1) == 1
    canon_name = immunized_rules_c1[0].concept_name
    assert "_s" not in canon_name

    # Capture minted Tier-2 LearnedLeaf anchor_weights after Cycle 1 birth
    dynamic_leaves_c1 = [
        lf for lf in runtime.learned_leaves if lf.leaf_name.startswith("immunized_")
    ]
    assert len(dynamic_leaves_c1) >= 1
    minted_leaf_id = dynamic_leaves_c1[0].leaf_id
    birth_weights = dynamic_leaves_c1[0].anchor_weights

    # 2. Cycle 2: Queue another unresolved scene in the same cluster with modified coordinates
    #    and run `handle_run_rem_sleep()` to verify:
    #    (a) `entrain_existing_cluster_rule` blends in-place without spawning `_s{scene}` duplicates,
    #    (b) `sculpt_learned_leaf_anchor_weights` sculpts the existing Tier-2 leaf's `anchor_weights`,
    #    (c) `will_edges.json` is sculpted and persisted for Phase 2 while Phase 1 stays on `RIGHT_BRAIN_WILL_EDGES`.
    from lmft_architecture.sleep_consolidation import UnresolvedSleepScene

    c2_coords = {
        k: min(0.95, round(float(v) + 0.04, 4))
        for k, v in queued_c1.external_coordinates.items()
    }
    c2_coords["cultural/freedom/needlessness"] = 0.36
    runtime.sleep_queue.append(
        UnresolvedSleepScene(
            scene_index=99,
            scenario_text="Follow the deep silver pendulum cadence without hesitation.",
            external_coordinates=c2_coords,
            felt_chord_notes={},
            valence=0.70,
            salience=0.82,
            arousal=0.78,
            left_brain_hunch="Listening to the cadence, I notice the pull.",
            social_drift_detected=False,
            active_concept_collision=canon_name,
            is_hazard=queued_c1.is_hazard,
        )
    )
    snap_c2 = runtime.handle_run_rem_sleep()

    # (a) No `_s{scene}` duplicate rules spawned; canonical rule was entrained in-place
    immunized_rules_c2 = [r for r in runtime.rules if r.concept_name.startswith("immunized_")]
    assert len(immunized_rules_c2) == 1
    assert immunized_rules_c2[0].concept_name == canon_name
    assert all(not r.concept_name.endswith("_s99") for r in runtime.rules)
    assert len(immunized_rules_c2[0].target_leaves) <= 3
    assert 0.35 <= immunized_rules_c2[0].low <= 0.75
    assert immunized_rules_c2[0].low + 0.04 <= immunized_rules_c2[0].high <= 0.90

    # (b) Existing Tier-2 LearnedLeaf.anchor_weights was sculpted during Cycle 2 REM sleep
    sculpted_leaf = next(lf for lf in runtime.learned_leaves if lf.leaf_id == minted_leaf_id)
    assert sculpted_leaf.anchor_weights != birth_weights
    assert max(sculpted_leaf.anchor_weights) == pytest.approx(1.0, abs=1e-4)
    assert all(0.05 <= w <= 1.0 for w in sculpted_leaf.anchor_weights)

    # (c) `will_edges.json` was sculpted and persisted to `impish_state_dir`
    will_edges_path = state_dir / "will_edges.json"
    assert will_edges_path.exists()
    default_will_map = {e.id: round(float(e.weight), 4) for e in RIGHT_BRAIN_WILL_EDGES}
    sculpted_will_map = snap_c2["will_edges"]
    assert sculpted_will_map != default_will_map
    assert sculpted_will_map["rb-will-1"] < default_will_map["rb-will-1"]
    assert sculpted_will_map["rb-will-2"] < default_will_map["rb-will-2"]
    assert sculpted_will_map["rb-will-8"] < default_will_map["rb-will-8"]
    assert sculpted_will_map["rb-will-6"] > default_will_map["rb-will-6"]

    # Phase 1 (Female Producer) turn stays on baseline `RIGHT_BRAIN_WILL_EDGES`,
    # while Phase 2 (Male Resistor) turn uses the sculpted `self.will_edges`
    _t_prod, _ms_p, tel_prod = runtime._execute_turn(
        "Craft a velvet cadence script.",
        force_spoken_reply=False,
        is_female_producer=True,
    )
    _t_res, _ms_r, tel_res = runtime._execute_turn(
        "Observe the cadence with lucid self-possession.",
        force_spoken_reply=False,
        is_female_producer=False,
    )
    assert tel_prod["will_edges"] == default_will_map
    assert tel_res["will_edges"] == sculpted_will_map





