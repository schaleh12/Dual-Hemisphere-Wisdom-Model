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
    weaver_msg = snap1["chat_messages"][-1]
    assert weaver_msg["telemetry"]["sparring_role"] == "INDUCTION_WEAVER"
    assert weaver_msg["telemetry"]["num_sentences"] == 3
    assert "induction_potency" in weaver_msg["telemetry"]

    # Step 2: Phase 2 — Male Content-Lover (The Audience & Self-Immunizer)
    snap2 = runtime.step_clockwise_phase()
    assert snap2["clockwise_phase"] == 3
    assert snap2["sleep_queue_count"] >= 1
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
        "into deep rhythmic stillness."
    )
    snap_p1 = runtime.seed_cycle_with_custom_script(owner_script)
    assert snap_p1["clockwise_phase"] == 2
    assert snap_p1["script_canvas_text"] == owner_script
    assert snap_p1["baseline_exemplar_script"] == owner_script
    assert snap_p1["chat_messages"][-1]["telemetry"]["user_seeded_baseline"] is True

    # Step Phase 2 (Male Audience absorbs the owner's exact script) and Phase 3 (REM Sleep)
    snap_p2 = runtime.step_clockwise_phase()
    assert snap_p2["clockwise_phase"] == 3
    assert snap_p2["last_sparring_result"]["induction_script"] == owner_script

    snap_p3 = runtime.step_clockwise_phase()
    assert snap_p3["clockwise_phase"] == 1
    assert snap_p3["clockwise_cycle_count"] == 1
    assert snap_p3["baseline_exemplar_script"] == owner_script


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


