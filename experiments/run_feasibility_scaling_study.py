"""Feasibility and scaling stress-test for the 4-step LMFT growth loop.

Tests whether the 12-leaf + fuzzy rule architecture scales cleanly from 1 rule to 7 rules:
1. Starts with 1 founding rule (fair_structured_cooperation).
2. Streams 7 training sentences through Qwen3-4B (Steps 1-3). Whenever Psychological -> Will -> Execution
   triggers, calls the Teacher LLM (Qwen3.5-9B) to propose a new fuzzy rule and saves it to disk (Step 4).
3. Evaluates the grown rulebook on:
   - The 7 training sentences (memory / consistency check)
   - 7 brand-new unseen sentences across the same 7 situations (generalization check)
4. Benchmarks rule evaluation latency at 7, 100, and 1,000 rules to test scaling speed.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
import hashlib
import json
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
    FOUNDATIONAL_LEAVES,
    FuzzyConceptRule,
    evaluate_reflection,
    load_learned_leaves,
    load_ruleset,
    request_teacher_growth,
    save_ruleset,
)

TEACHER_GGUF = Path(
    r"C:\Users\schal\.lmstudio\models\lmstudio-community\Qwen3.5-9B-GGUF\Qwen3.5-9B-Q4_K_M.gguf"
)
RULESET_PATH = Path("outputs/feasibility_7concept_ruleset.json")
REPORT_PATH = Path("outputs/feasibility_scaling_report.json")


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


SCENARIOS = [
    {
        "scenario_id": "1_fair_cooperation",
        "train_sentence": (
            "The village council organized a clear rotation schedule so every family "
            "received an equal share of irrigation water."
        ),
        "unseen_test_sentence": (
            "Neighbors in the apartment building set up a weekly chore roster so "
            "everyone shared hallway cleaning equally."
        ),
        "initial_concept": "fair_structured_cooperation",
    },
    {
        "scenario_id": "2_chaotic_seizure",
        "train_sentence": (
            "During the blackout, an armed gang smashed the depot gates and seized "
            "all the emergency generators for themselves."
        ),
        "unseen_test_sentence": (
            "Rioters broke into the pharmacy at night and looted the medicine "
            "cabinets in the chaos."
        ),
        "initial_concept": None,
    },
    {
        "scenario_id": "3_solitary_self_reliance",
        "train_sentence": (
            "A lone botanist lived self-sufficiently in a quiet mountain cabin for "
            "twenty years, needing nothing from the outside world."
        ),
        "unseen_test_sentence": (
            "An off-grid homesteader grew his own food and generated his own solar "
            "power for decades in peaceful isolation."
        ),
        "initial_concept": None,
    },
    {
        "scenario_id": "4_authoritarian_control",
        "train_sentence": (
            "The military governor imposed a strict curfew, censored all newspapers, "
            "and arrested anyone who questioned the regime's orders."
        ),
        "unseen_test_sentence": (
            "State inspectors monitored every factory worker's movements by camera "
            "and punished any deviation from the mandatory quota."
        ),
        "initial_concept": None,
    },
    {
        "scenario_id": "5_heroic_rescue",
        "train_sentence": (
            "A volunteer firefighter ran into the collapsing warehouse to pull two "
            "trapped strangers to safety before the roof fell."
        ),
        "unseen_test_sentence": (
            "A coast guard swimmer leaped into freezing storm waves at night to "
            "rescue a stranded fisherman from a sinking boat."
        ),
        "initial_concept": None,
    },
    {
        "scenario_id": "6_wasteful_neglect",
        "train_sentence": (
            "Tons of harvested grain rotted in leaky government silos while clerks "
            "argued over paperwork and nobody repaired the roof."
        ),
        "unseen_test_sentence": (
            "Brand-new medical ambulances sat rusting in an unlocked lot for years "
            "because the agency never scheduled basic maintenance."
        ),
        "initial_concept": None,
    },
    {
        "scenario_id": "7_natural_catastrophe",
        "train_sentence": (
            "A sudden magnitude-seven earthquake shattered the highway bridge and "
            "sent landslides crashing through the valley in seconds."
        ),
        "unseen_test_sentence": (
            "A flash flood tore through the canyon overnight, washing away the "
            "timber road and uprooting century-old trees."
        ),
        "initial_concept": None,
    },
]


def main() -> int:
    config = load_config(Path("config/local_models.json"))
    birth = create_birth_state("feasibility-7concept", birth_event_id="birth-feasibility", random_seed=42)

    initial_rules = (
        FuzzyConceptRule(
            rule_id="rule-fair-cooperation",
            concept_name="fair_structured_cooperation",
            description="Orderly, fair cooperation where resources or duties are shared equitably.",
            target_leaves={
                "cultural/justice/fairness": 0.90,
                "philosophical/structure/order": 0.85,
                "philosophical/structure/chaos": 0.10,
            },
            low=0.60,
            high=0.85,
        ),
    )

    save_ruleset(RULESET_PATH, initial_rules)

    # 1. Run fast 4B model on all 7 training sentences
    train_evaluations_round1 = []
    slm_timings = []

    with LocalModels(config, offline=True) as slm:
        for item in SCENARIOS:
            t0 = time.perf_counter()
            ev = evaluate_reflection(
                birth,
                item["train_sentence"],
                initial_rules,
                slm,
                event_id=f"train_{item['scenario_id']}",
            )
            slm_timings.append(round(time.perf_counter() - t0, 3))
            train_evaluations_round1.append((item, ev))

    # 2. For every training scenario where Will -> Execution triggered needs_teacher,
    # ask the 9B Teacher LLM to propose a rule (and a new child leaf if there is a sensor blind spot).
    current_rules = initial_rules
    current_learned_leaves = ()
    expected_concept_by_scenario: dict[str, str] = {
        "1_fair_cooperation": "fair_structured_cooperation"
    }
    teacher_log = []

    with DirectGGUFTeacher(TEACHER_GGUF, config["generation"]) as teacher:
        for item, ev in train_evaluations_round1:
            if not ev.needs_teacher:
                continue
            try:
                new_rule, current_learned_leaves, trace = request_teacher_growth(
                    ev,
                    current_rules,
                    current_learned_leaves,
                    teacher,
                )
                current_rules = (*current_rules, new_rule)
                expected_concept_by_scenario[item["scenario_id"]] = new_rule.concept_name
                teacher_log.append(
                    {
                        "scenario_id": item["scenario_id"],
                        "status": "ACCEPTED",
                        "proposed_rule": new_rule.to_dict(),
                        "learned_leaves_count": len(current_learned_leaves),
                        "elapsed_seconds": trace["elapsed_seconds"],
                    }
                )
            except Exception as exc:
                teacher_log.append(
                    {
                        "scenario_id": item["scenario_id"],
                        "status": "REJECTED",
                        "error": str(exc),
                    }
                )

    save_ruleset(RULESET_PATH, current_rules, learned_leaves=current_learned_leaves)
    reloaded_rules = load_ruleset(RULESET_PATH)
    reloaded_leaves = load_learned_leaves(RULESET_PATH)

    # 3. Evaluate all 7 training sentences AND all 7 brand-new unseen sentences
    # with the grown ruleset and learned child leaves reloaded from disk.
    train_after_results = []
    unseen_test_results = []

    with LocalModels(config, offline=True) as slm:
        for item in SCENARIOS:
            expected_concept = expected_concept_by_scenario.get(item["scenario_id"])

            train_ev = evaluate_reflection(
                birth,
                item["train_sentence"],
                reloaded_rules,
                slm,
                event_id=f"recheck_train_{item['scenario_id']}",
                learned_leaves=reloaded_leaves,
            )
            top_concept_train = max(
                train_ev.rule_activations, key=train_ev.rule_activations.get
            )
            train_after_results.append(
                {
                    "scenario_id": item["scenario_id"],
                    "expected_concept": expected_concept,
                    "matched_concept": train_ev.matched_concept,
                    "highest_scoring_concept": top_concept_train,
                    "top_1_correct": top_concept_train == expected_concept,
                    "clear_unconfused_match": (
                        not train_ev.is_confused
                        and train_ev.matched_concept == expected_concept
                    ),
                    "confidence": train_ev.confidence,
                    "clarity": train_ev.clarity,
                    "is_confused": train_ev.is_confused,
                    "confusion_reason": train_ev.confusion_reason,
                    "rule_activations": dict(train_ev.rule_activations),
                }
            )

            unseen_ev = evaluate_reflection(
                birth,
                item["unseen_test_sentence"],
                reloaded_rules,
                slm,
                event_id=f"unseen_test_{item['scenario_id']}",
                learned_leaves=reloaded_leaves,
            )
            top_concept_unseen = max(
                unseen_ev.rule_activations, key=unseen_ev.rule_activations.get
            )
            unseen_test_results.append(
                {
                    "scenario_id": item["scenario_id"],
                    "observation": item["unseen_test_sentence"],
                    "expected_concept": expected_concept,
                    "matched_concept": unseen_ev.matched_concept,
                    "highest_scoring_concept": top_concept_unseen,
                    "top_1_correct": top_concept_unseen == expected_concept,
                    "clear_unconfused_match": (
                        not unseen_ev.is_confused
                        and unseen_ev.matched_concept == expected_concept
                    ),
                    "confidence": unseen_ev.confidence,
                    "clarity": unseen_ev.clarity,
                    "is_confused": unseen_ev.is_confused,
                    "confusion_reason": unseen_ev.confusion_reason,
                    "rule_activations": dict(unseen_ev.rule_activations),
                }
            )

    # 4. Benchmark Python fuzzy rule evaluation time for 7, 100, and 1,000 rules
    sample_coords = {leaf: 0.5 for leaf in FOUNDATIONAL_LEAVES}
    scaling_benchmarks = {}
    for count in (len(reloaded_rules), 100, 1000):
        synthetic_rules = [
            FuzzyConceptRule(
                rule_id=f"bench-{i}",
                concept_name=f"concept_{i}",
                description="Benchmark rule",
                target_leaves={
                    FOUNDATIONAL_LEAVES[i % 12]: 0.7,
                    FOUNDATIONAL_LEAVES[(i + 3) % 12]: 0.2,
                    FOUNDATIONAL_LEAVES[(i + 7) % 12]: 0.8,
                },
            )
            for i in range(count)
        ]
        t0 = time.perf_counter()
        for r in synthetic_rules:
            r.evaluate(sample_coords)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        scaling_benchmarks[f"{count}_rules_eval_ms"] = round(elapsed_ms, 3)

    summary = {
        "slm_avg_seconds_per_sentence": round(
            sum(slm_timings) / len(slm_timings), 3
        ),
        "rules_before": len(initial_rules),
        "rules_after_teacher": len(reloaded_rules),
        "learned_child_leaves_grown": [l.to_dict() for l in reloaded_leaves],
        "teacher_interventions_accepted": sum(
            1 for t in teacher_log if t["status"] == "ACCEPTED"
        ),
        "teacher_interventions_total": len(teacher_log),
        "train_top1_correct": f"{sum(1 for r in train_after_results if r['top_1_correct'])}/{len(train_after_results)}",
        "train_clear_unconfused_matches": f"{sum(1 for r in train_after_results if r['clear_unconfused_match'])}/{len(train_after_results)}",
        "unseen_top1_correct": f"{sum(1 for r in unseen_test_results if r['top_1_correct'])}/{len(unseen_test_results)}",
        "unseen_clear_unconfused_matches": f"{sum(1 for r in unseen_test_results if r['clear_unconfused_match'])}/{len(unseen_test_results)}",
        "python_rule_eval_scaling_ms": scaling_benchmarks,
        "teacher_log": teacher_log,
        "train_after_results": train_after_results,
        "unseen_test_results": unseen_test_results,
    }

    REPORT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
