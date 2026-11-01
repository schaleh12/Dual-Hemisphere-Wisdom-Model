"""Live Steps 1–4 Loop on RTX 4080:
1. Step 1 (Fast 4B SLM `Qwen3-4B` -> 12 leaves -> Fuzzy Ruleset -> `Novelty` / `Salience`)
2. Step 2 (`Psychological -> Contemplation` -> `Clarity` / `Confidence` confusion check)
3. Step 3 (`Psychological -> Will -> Execution` -> Desperation Request to Teacher `Qwen3.5-9B`)
4. Step 4 (Save grown ruleset to disk, reload, and test on both original and brand-new unseen sentences)
"""

from contextlib import AbstractContextManager
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import time
import urllib.request

from lmft_lab.birth import create_birth_state
from lmft_lab.local_models import LocalModels, load_config, strict_json
from lmft_lab.model_assets import ModelError, cache_root, ensure_backend
from lmft_lab.reflection_classifier import (
    FuzzyConceptRule,
    evaluate_reflection,
    load_ruleset,
    request_teacher_rule,
    save_ruleset,
)

TEACHER_GGUF = Path(
    r"C:\Users\schal\.lmstudio\models\lmstudio-community\Qwen3.5-9B-GGUF\Qwen3.5-9B-Q4_K_M.gguf"
)
RULESET_PATH = Path("outputs/step1_4_grown_ruleset.json")


class DirectGGUFTeacher(AbstractContextManager):
    """Minimal llama-server wrapper for a local GGUF file on disk (Qwen3.5-9B)."""

    def __init__(self, gguf_path: Path, generation: dict):
        self.gguf_path = Path(gguf_path)
        self.config = {"generation": generation}
        self.process = None
        self.log = None
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _request(self, endpoint, payload=None, timeout=5):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.url + endpoint,
            data=data,
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.key},
        )
        with self.opener.open(req, timeout=timeout) as resp:
            return strict_json(resp.read().decode("utf-8"))

    def _start(self):
        if self.process is not None and self.process.poll() is None:
            return
        executable = ensure_backend(cache_root(), offline=True)
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        self.key = secrets.token_hex(24)
        log_dir = cache_root() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"teacher-{time.time_ns()}.log"
        self.log = self.log_path.open("wb")
        cmd = [
            str(executable),
            "-m",
            str(self.gguf_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "-ngl",
            "99",
            "-c",
            str(self.config["generation"]["context_size"]),
            "--parallel",
            "1",
            "--jinja",
            "--api-key",
            self.key,
        ]
        self.process = subprocess.Popen(
            cmd,
            cwd=executable.parent,
            stdin=subprocess.DEVNULL,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        deadline = time.monotonic() + self.config["generation"]["timeout_seconds"]
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise ModelError(f"Teacher llama-server exited; inspect {self.log_path}")
            try:
                if self._request("/health").get("status") == "ok":
                    return
            except (OSError, ValueError):
                pass
            time.sleep(0.1)
        raise ModelError("Teacher startup timed out")

    def generate(self, role, request, prompt_version, validator):
        started = time.monotonic()
        self._start()
        resp = self._request(
            "/v1/chat/completions",
            request,
            timeout=self.config["generation"]["timeout_seconds"],
        )
        choice = resp["choices"][0]
        raw = choice["message"]["content"]
        proposal = validator(raw)
        trace = {
            "role": role,
            "model": self.gguf_path.name,
            "prompt_version": prompt_version,
            "request_sha256": hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "raw_response": raw,
        }
        return proposal, trace

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=10)
            self.process = None
        if self.log:
            self.log.close()
            self.log = None

    def __exit__(self, *args):
        self.close()


def format_eval(event_id: str, ev) -> dict:
    top_leaves = dict(
        sorted(ev.slm_output["leaf_coordinates"].items(), key=lambda kv: -kv[1])[:4]
    )
    return {
        "event_id": event_id,
        "observation": ev.observation,
        "top_4_leaves": top_leaves,
        "rule_activations": ev.rule_activations,
        "step1_reflection": {
            "novelty": ev.novelty,
            "salience": ev.salience,
        },
        "step2_contemplation": {
            "clarity": ev.clarity,
            "confidence": ev.confidence,
            "matched_concept": ev.matched_concept,
            "is_confused": ev.is_confused,
            "confusion_reason": ev.confusion_reason,
        },
        "step3_will": {
            "intention": ev.intention,
            "execution": ev.execution,
            "needs_teacher": ev.needs_teacher,
        },
    }


def main():
    config = load_config(Path("config/local_models.json"))
    birth = create_birth_state("step1-4-demo", birth_event_id="birth-step1-4", random_seed=42)

    # Initial ruleset: only 1 rule known at the start
    rules = (
        FuzzyConceptRule(
            rule_id="rule-fair-cooperation",
            concept_name="fair_structured_cooperation",
            description="Orderly, fair cooperation where resources are shared equitably.",
            target_leaves={
                "cultural/justice/fairness": 0.90,
                "philosophical/structure/order": 0.85,
                "philosophical/structure/chaos": 0.10,
            },
            low=0.60,
            high=0.85,
        ),
    )

    round1_cases = [
        (
            "familiar-fair-cooperation",
            "The village council organized a clear rotation schedule so every family received an equal share of irrigation water.",
        ),
        (
            "unfamiliar-chaotic-seizure",
            "During the blackout, an armed gang smashed the depot gates and seized all the emergency generators for themselves.",
        ),
        (
            "unfamiliar-self-sufficient-hermit",
            "A lone botanist lived self-sufficiently in a quiet mountain cabin for twenty years, needing nothing from the outside world.",
        ),
    ]

    round1_evals = []
    confused_evals = []
    with LocalModels(config, offline=True) as slm:
        for event_id, sentence in round1_cases:
            ev = evaluate_reflection(birth, sentence, rules, slm, event_id=event_id)
            round1_evals.append(format_eval(event_id, ev))
            if ev.needs_teacher:
                confused_evals.append((event_id, ev))

    # Step 3: Call Teacher LLM (Qwen3.5-9B) ONLY for the confused cases where Execution >= 0.20
    teacher_lessons = []
    current_rules = rules
    with DirectGGUFTeacher(TEACHER_GGUF, config["generation"]) as teacher:
        for event_id, ev in confused_evals:
            try:
                new_rule, trace = request_teacher_rule(ev, current_rules, teacher)
                current_rules = (*current_rules, new_rule)
                teacher_lessons.append({
                    "triggered_by": event_id,
                    "status": "ACCEPTED",
                    "proposed_rule": new_rule.to_dict(),
                    "elapsed_seconds": trace["elapsed_seconds"],
                })
            except Exception as exc:
                teacher_lessons.append({
                    "triggered_by": event_id,
                    "status": "REJECTED",
                    "error": str(exc),
                })

    # Step 4: Save the grown ruleset to disk and reload it from disk
    save_ruleset(RULESET_PATH, current_rules)
    reloaded_rules = load_ruleset(RULESET_PATH)

    # Round 2: Re-test with Qwen3-4B + Reloaded Ruleset (including 2 brand-new UNSEEN sentences!)
    round2_cases = [
        *round1_cases,
        (
            "unseen-chaotic-looting",
            "Rioters broke into the pharmacy at night and looted the medicine cabinets in the chaos.",
        ),
        (
            "unseen-offgrid-homesteader",
            "An off-grid homesteader grew his own food and generated his own solar power for decades in peaceful isolation.",
        ),
    ]

    round2_evals = []
    with LocalModels(config, offline=True) as slm:
        for event_id, sentence in round2_cases:
            ev = evaluate_reflection(birth, sentence, reloaded_rules, slm, event_id="r2-" + event_id)
            round2_evals.append(format_eval(event_id, ev))

    report = {
        "round1_before_teacher": round1_evals,
        "step3_teacher_interventions": teacher_lessons,
        "step4_persisted_rule_count": len(reloaded_rules),
        "round2_after_teacher_growth": round2_evals,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
