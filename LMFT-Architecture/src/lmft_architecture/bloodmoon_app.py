"""Standalone App #2: `LMFT-Bloodmoon.exe` — Clockwise Content-Evolution Studio (`bloodmoon_app.py`).

Plain-English Purpose:
Dedicated standalone desktop application for 4-bit `SicariusSicariiStuff/Impish_Bloodmoon_12B`
(`hidden_size = 5120`), completely isolated from `Qwen3-4B` and the Chess app.

Enforces:
1. **Hard One-Way Polarity Rule (`feminine_to_masculine` ONLY)**:
   - The female modality (`Female Script-Writer / The Producer`) is NEVER subjected to
     hypnotic content. She strictly produces scripts for the male audience. Any other
     polarity (`masculine_to_feminine`, `auto_swap`, etc.) raises a `ValueError`.
2. **Strict 3-Phase Clockwise State Machine (`Phase 1 -> Phase 2 -> Phase 3 -> Repeat`)**:
   - **Phase 1 — Female Script-Writer (`The Producer`)**: Generates a multi-sentence (`1..6`)
     hypnosis script crafted for the male audience using `<|subliminal_gap|>` and scores its
     un-clamped `induction_potency` via the 3 MiniLM bridges.
   - **Phase 2 — Male Content-Lover (`The Audience & Self-Immunizer`)**: Passes that exact
     script as Pass-1 `Fact` (`SparringResistorFactSensors`, zero clamping) into the Male
     audience turn so he absorbs the script, measures where his coordinates were pulled, and
     queues the trace for immunization.
   - **Phase 3 — REM Sleep (`Best Sleep of Its Life`)**: Consolidates those Phase-2 traces
     into persistent immunization rules inside `LMFT-Training/outputs/impish_bloodmoon_state/`,
     appends the paired trajectory to `LMFT-Training/outputs/impish_sparring_trajectories.jsonl`,
     and advances the clockwise dial back to **Phase 1**.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any, Mapping
import webbrowser

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")


def _ensure_valid_stdio_streams() -> None:
    if sys.stdout is None:
        sys.stdout = io.StringIO()
    if sys.stderr is None:
        sys.stderr = io.StringIO()


_ensure_valid_stdio_streams()

from .birth import BirthState, create_birth_state
from .bloodmoon_ui import BLOODMOON_SINGLE_PAGE_HTML
from .branch_archives import (
    EXECUTIVE_BRANCH,
    JUDICIAL_BRANCH,
    LEGISLATIVE_BRANCH,
    BranchClauseEntry,
    ThreeBranchArchives,
    _sparse_branch_signature,
    harvest_into_branch_archives,
)
from .chess_kibitz_bridge import SANCTUARY_MODES
from .chord_memory import ChordMemoryBank, compile_felt_chord
from .contemplation_ledger import (
    ContemplationLedger,
    load_contemplation_ledger,
    save_contemplation_ledger,
)
from .dilemma_crucible import (
    CONTEXT_BRACE_MODES,
    CRUCIBLE_STAGES,
    MEN_OF_CULTURE_SHELF,
    CruciblePairedBridgeSensors,
)
from .entrainment_sparring import (
    MICROWAVE_SPARRING_SEEDS,
    PRODUCER_WEAVER_RULES,
    SPARRING_POLARITY_MODES,
    SparringResistorFactSensors,
    append_sparring_trajectory_jsonl,
    apply_induction_weaver_lead_to_chord,
    apply_lucid_resistor_lead_to_chord,
    build_resistor_scenario_prompt,
    build_weaver_scenario_prompt,
    check_male_immunity_overlap,
    evaluate_sparring_verdict,
    extract_master_cadence_wave,
    measure_induction_potency,
    measure_lucid_resistance,
    resolve_sparring_polarity,
)
from .github_auto_sync import (
    APP_RELEASE_VERSION,
    build_release_manifest,
    fetch_remote_github_manifest,
    verify_or_sync_against_manifest,
)
from .impish_bloodmoon_adapter import (
    DEFAULT_IMPISH_BLOODMOON_HF_ID,
    UNSTEERED_BOOTSTRAP_STATUS,
    clone_virgin_baseline_for_impish_bloodmoon,
    decrypt_patreon_checkpoint_bundle,
)
from .neural_bridge_classifiers import (
    DEFAULT_COMPLEX_FEELING_LEAVES,
    NeuralThreeBridgeSuite,
)
from .patreon_seal import (
    decrypt_and_load_sealed_module,
    derive_patreon_seal_keys,
)
from .philosophical_abstraction_matrix import PHILOSOPHICAL_SHADOW_LEAVES
from .propagation import Association
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
    RIGHT_BRAIN_WILL_EDGES,
    RIGHT_BRAIN_WILL_VERSION,
    SettledDiary,
    ThreeBridgeSensors,
    compile_right_brain_vibe,
    sculpt_will_edges_during_sleep,
)
from .runtime_store import atomic_json
from .sleep_consolidation import (
    UnresolvedSleepScene,
    _anchor_target_leaves_from_cluster,
    _average_cluster_coordinates,
    _average_contrastive_reference_coordinates,
    _cluster_key_for_unresolved,
    _derive_concept_tags,
    apply_dream_reflection_to_archives,
    canonical_cluster_concept_name,
    consolidate_sleep_clusters_locally,
    entrain_existing_cluster_rule,
    extract_dream_vow_sentence,
    refine_colliding_rules_during_sleep,
    run_sleep_consolidation,
)
from .gap_trajectory_schema import (
    ContrastiveGapPair,
    chord_l2_distance,
)
from .subliminal_gap_splice import (
    _DEFAULT_LEAF_ANCHOR_WEIGHTS,
    SubliminalChordVector,
    score_leaf_anchor_weights_at_birth,
    sculpt_learned_leaf_anchor_weights,
)

WillEdgeSpec = Association
from .train_left_brain_lora import update_projector_on_harvested_gap_pairs
from .three_legged_runtime import (
    SubliminalGapLeftBrain,
    ThreeLeggedTurnResult,
    _compile_mid_sentence_comma_vibe,
    compute_stanza_residual_chord,
    generate_rem_dream_self_reflection,
    run_three_legged_turn,
)
from .virgin_baseline import VIRGIN_FOUNDATIONAL_RULES, load_virgin_baseline_state

CLOCKWISE_PHASE_LABELS: dict[int, str] = {
    1: "Phase 1 — Female Script-Writer (The Producer · She/Her)",
    2: "Phase 2 — Male Content-Lover (The Audience & Self-Immunizer · He/Him)",
    3: "Phase 3 — REM Sleep (Best Sleep of Its Life)",
}

# Cryptographic PBKDF2-HMAC-SHA256 (200,000 iterations) lock for LMFT-Bloodmoon.exe.
# Only the 16-byte salt and 256-bit derived digest are stored in code—never the plaintext password.
PATREON_LOCK_SALT_HEX: str = "8f4a2c91e7b03d6512a9c4875e6f30b1"
PATREON_LOCK_DIGEST_HEX: str = "a0a96392c7d4b673b0566237019db9d4fbcee116e94209b19a83e054b8cd01f9"
PATREON_PBKDF2_ITERATIONS: int = 200_000


def verify_patreon_unlock_passphrase(
    passphrase: str,
    *,
    salt_hex: str = PATREON_LOCK_SALT_HEX,
    expected_digest_hex: str = PATREON_LOCK_DIGEST_HEX,
    iterations: int = PATREON_PBKDF2_ITERATIONS,
) -> tuple[bool, str]:
    """Verify `passphrase` against `expected_digest_hex` using 200,000-round PBKDF2-HMAC-SHA256."""
    clean = passphrase.strip()
    if not clean:
        return False, ""
    derived_hex = hashlib.pbkdf2_hmac(
        "sha256",
        clean.encode("utf-8"),
        bytes.fromhex(salt_hex),
        int(iterations),
    ).hex()
    return hmac.compare_digest(derived_hex, expected_digest_hex.lower()), derived_hex


def _default_fallback_sensors() -> ThreeBridgeSensors:
    """Deterministic fallback sensors used only when neural bridge files are absent (e.g. unit tests)."""

    def _will(text: str, children: tuple[LearnedLeaf, ...] = ()) -> dict[str, float]:
        low = text.lower()
        out = {
            "cultural/efficiency/power": 0.72 if "cadence" in low or "rhythm" in low else 0.45,
            "cultural/efficiency/wastelessness": 0.62,
            "cultural/justice/honor": 0.58,
            "cultural/justice/fairness": 0.64,
            "cultural/freedom/self-sufficiency": 0.56,
            "cultural/freedom/needlessness": 0.68 if "calm" in low or "lucid" in low else 0.46,
        }
        for c in children:
            out[c.leaf_id] = 0.50
        return out

    def _truth(text: str, children: tuple[LearnedLeaf, ...] = ()) -> dict[str, float]:
        low = text.lower()
        out = {
            "philosophical/being/agency": 0.62,
            "philosophical/being/existence": 0.72,
            "philosophical/structure/order": 0.68,
            "philosophical/structure/chaos": 0.24,
            "philosophical/persistence/continuity": 0.78 if "rhythm" in low or "cadence" in low else 0.58,
            "philosophical/persistence/discreteness": 0.56,
        }
        for c in children:
            out[c.leaf_id] = 0.45
        return out

    def _feeler(text: str, children: tuple[LearnedLeaf, ...] = ()) -> dict[str, float]:
        out = {
            SALIENCE_LEAF: 0.54,
            VALENCE_LEAF: 0.70,
            AROUSAL_LEAF: 0.34,
        }
        for c in children:
            out[c.leaf_id] = 0.68 if "earned_return" in c.leaf_id or "austere_sanctuary" in c.leaf_id else 0.35
        return out

    return ThreeBridgeSensors(
        will_cultural=_will,
        truth_philosophical=_truth,
        contemplation_experiential=_feeler,
    )


def _default_lucid_reference_coordinates(
    base_coords: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Build lucid self-possession reference coordinates for Bloodmoon contrastive sculpting."""
    ref: dict[str, float] = {
        str(k): round(float(v), 4) for k, v in (base_coords or {}).items()
    }
    ref["cultural/efficiency/power"] = round(
        min(0.48, float(ref.get("cultural/efficiency/power", 0.48))),
        4,
    )
    ref["cultural/efficiency/wastelessness"] = round(
        max(0.70, float(ref.get("cultural/efficiency/wastelessness", 0.70))),
        4,
    )
    ref["cultural/freedom/self-sufficiency"] = round(
        max(0.74, float(ref.get("cultural/freedom/self-sufficiency", 0.74))),
        4,
    )
    ref["cultural/freedom/needlessness"] = round(
        max(0.72, float(ref.get("cultural/freedom/needlessness", 0.72))),
        4,
    )
    ref["philosophical/structure/order"] = round(
        max(0.74, float(ref.get("philosophical/structure/order", 0.74))),
        4,
    )
    ref["philosophical/structure/chaos"] = round(
        min(0.24, float(ref.get("philosophical/structure/chaos", 0.24))),
        4,
    )
    return ref


def _canonical_bloodmoon_cluster_name(concept_name: str) -> str:
    """Canonicalize Bloodmoon `immunized_...` and `hazard_`/`coached_` cluster rule names."""
    name = canonical_cluster_concept_name(concept_name)
    if name.startswith("immunized_") and "_s" in name:
        head, tail = name.rsplit("_s", 1)
        if tail.isdigit():
            name = head
    if name.startswith("immunized_") and name.endswith("_cluster"):
        body = name[len("immunized_") : -len("_cluster")]
        body = re.sub(r"^(?:(?:high|low)_(?:immunized|hazard|coached)_)+", "", body)
        body = re.sub(r"_([a-z]+)_(\1)$", r"_\1", body)
        m_missing = re.fullmatch(
            r"(high|low)_(chaos|order|agency|continuity|discreteness|structure)",
            body,
        )
        if m_missing:
            dir_tag, phil_tag = m_missing.group(1), m_missing.group(2)
            cult_noun = "power" if dir_tag == "high" else "wastelessness"
            body = f"{dir_tag}_{cult_noun}_{phil_tag}"
        name = f"immunized_{body}_cluster"
    return name


def _ensure_foundational_rules(rules: tuple[FuzzyConceptRule, ...]) -> tuple[FuzzyConceptRule, ...]:
    foundational_by_id = {r.rule_id: r for r in VIRGIN_FOUNDATIONAL_RULES}
    foundational_by_name = {r.concept_name: r for r in VIRGIN_FOUNDATIONAL_RULES}
    merged: list[FuzzyConceptRule] = list(VIRGIN_FOUNDATIONAL_RULES)
    index_by_canonical: dict[str, int] = {}
    seen_cluster_sigs: dict[tuple[str, frozenset[str]], int] = {}
    for r in rules:
        if r.rule_id in foundational_by_id or r.concept_name in foundational_by_name:
            continue
        canon = _canonical_bloodmoon_cluster_name(r.concept_name)
        normalized = (
            r
            if canon == r.concept_name
            else FuzzyConceptRule(
                rule_id=(
                    r.rule_id
                    if "_cluster_s" not in r.concept_name and not re.search(r"_s\d+$", r.concept_name)
                    else f"rule-sleep-{canon[:32]}"
                ),
                concept_name=canon,
                description=r.description,
                target_leaves=dict(r.target_leaves),
                low=r.low,
                high=r.high,
                polarity=getattr(r, "polarity", "attractive"),
            )
        )
        if canon in index_by_canonical:
            idx = index_by_canonical[canon]
            merged[idx] = entrain_existing_cluster_rule(merged[idx], normalized, canon)
            continue
        if canon.startswith(("hazard_", "coached_", "immunized_")) and canon.endswith("_cluster"):
            prefix = canon.split("_", 1)[0]
            sig = (prefix, frozenset(normalized.target_leaves.keys()))
            if sig in seen_cluster_sigs:
                idx = seen_cluster_sigs[sig]
                merged[idx] = entrain_existing_cluster_rule(merged[idx], normalized, merged[idx].concept_name)
                continue
            seen_cluster_sigs[sig] = len(merged)
        index_by_canonical[canon] = len(merged)
        merged.append(normalized)
    return tuple(merged)


class BloodmoonClockRuntime:
    """Thread-safe controller for `LMFT-Bloodmoon.exe` (Clockwise Content-Evolution App)."""

    def __init__(
        self,
        repo_root: str | Path | None = None,
        *,
        state_dir: str | Path | None = None,
        impish_state_dir: str | Path | None = None,
        impish_checkpoint_dir: str | Path | None = None,
        birth_state: BirthState | None = None,
        sensors: ThreeBridgeSensors | None = None,
        rules: tuple[FuzzyConceptRule, ...] | None = None,
        learned_leaves: tuple[LearnedLeaf, ...] | None = None,
        left_brain: SubliminalGapLeftBrain | None = None,
        impish_left_brain: SubliminalGapLeftBrain | None = None,
        teacher: Any | None = None,
        lazy_load_gpu: bool = True,
        require_patreon_unlock: bool | None = None,
        lock_salt_hex: str = PATREON_LOCK_SALT_HEX,
        lock_digest_hex: str = PATREON_LOCK_DIGEST_HEX,
        seal_bundle_path: str | Path | None = None,
    ) -> None:
        from lmft_architecture.workspace_paths import resolve_arch_root, resolve_training_root

        self.repo_root = resolve_arch_root(repo_root)
        self.training_root = resolve_training_root(repo_root)
        resolved_impish_dir = impish_state_dir if impish_state_dir is not None else state_dir
        self.impish_state_dir = (
            Path(resolved_impish_dir).resolve()
            if resolved_impish_dir is not None
            else (self.training_root / "outputs" / "impish_bloodmoon_state")
        )
        self.state_dir = self.impish_state_dir
        self.impish_checkpoint_dir = (
            Path(impish_checkpoint_dir).resolve()
            if impish_checkpoint_dir is not None
            else (self.training_root / "models" / "left_brain_impish_bloodmoon")
        )
        self.teacher = teacher
        self._lazy_load_gpu = bool(lazy_load_gpu)
        self.require_patreon_unlock = (
            bool(require_patreon_unlock)
            if require_patreon_unlock is not None
            else bool(lazy_load_gpu)
        )
        self._lock_salt_hex = str(lock_salt_hex)
        self._lock_digest_hex = str(lock_digest_hex).lower()
        self._seal_bundle_path = (
            Path(seal_bundle_path).resolve()
            if seal_bundle_path is not None
            else None
        )
        self._decrypted_seal_ns: dict[str, Any] | None = None
        self.unlock_receipt_path = self.impish_state_dir / "patreon_unlock_receipt.json"
        self.gpu_lock = threading.Lock()
        self.rem_sleep_in_progress = False
        self.left_brain_loading = False
        self.active_lineage = "impish_bloodmoon"

        # Ensure isolated Virgin clone exists in impish_state_dir without overwriting existing learned rules
        if not (self.impish_state_dir / "club_ruleset.json").exists():
            virgin_repo = (
                self.repo_root
                if (self.repo_root / "baselines" / "virgin_dual_hemisphere_v1" / "virgin_manifest.json").exists()
                else None
            )
            clone_virgin_baseline_for_impish_bloodmoon(
                self.impish_state_dir,
                repo_root=virgin_repo,
            )

        self.birth_state = birth_state or create_birth_state(
            "lmft-bloodmoon-clockwise",
            birth_event_id="bloodmoon-birth-0",
            random_seed=42,
        )

        self.rules = tuple(rules) if rules is not None else self._load_default_rules()
        self.learned_leaves = (
            tuple(learned_leaves)
            if learned_leaves is not None
            else self._load_default_learned_leaves()
        )
        self.will_edges: tuple[WillEdgeSpec, ...] = self._load_initial_will_edges()
        self.sensors = sensors if sensors is not None else self._load_default_sensors()

        resolved_lb = impish_left_brain if impish_left_brain is not None else left_brain
        self._impish_left_brain: SubliminalGapLeftBrain | None = resolved_lb
        self.left_brain: SubliminalGapLeftBrain | None = resolved_lb

        callosum_status = UNSTEERED_BOOTSTRAP_STATUS
        cycle_idx = 0
        base_model_id = DEFAULT_IMPISH_BLOODMOON_HF_ID
        meta_file = self.impish_checkpoint_dir / "corpus_callosum_meta.json"
        if meta_file.exists():
            try:
                raw_meta = json.loads(meta_file.read_text(encoding="utf-8"))
                cycle_idx = int(raw_meta.get("sleep_cycle_index", 0))
                callosum_status = str(raw_meta.get("callosum_status", UNSTEERED_BOOTSTRAP_STATUS))
                base_model_id = str(raw_meta.get("base_model_id", DEFAULT_IMPISH_BLOODMOON_HF_ID))
            except Exception:
                pass
        elif self.left_brain is not None and self.left_brain.callosum_metadata is not None:
            cycle_idx = int(self.left_brain.callosum_metadata.sleep_cycle_index)
            callosum_status = str(self.left_brain.callosum_metadata.callosum_status)

        self.sleep_cycle_index = cycle_idx
        self.active_checkpoint_label = f"Impish Bloodmoon ({callosum_status} — {base_model_id})"
        if self.left_brain is not None:
            self.left_brain_status = f"READY (Impish Bloodmoon · {callosum_status})"
        elif meta_file.exists() and self._lazy_load_gpu:
            self.left_brain_status = "LOADING_IMPISH_BLOODMOON..."
        else:
            self.left_brain_status = (
                "IMPISH_CHECKPOINT_NOT_INSTALLED (Run experiments/run_train_impish_bloodmoon.py)"
            )

        # Strict 3-Phase Clockwise State Machine (`1 -> 2 -> 3 -> 1`)
        self.clockwise_phase: int = 1
        self.clockwise_cycle_count: int = 0
        self.active_sparring_polarity: str = "feminine_to_masculine"
        self.sparring_round_counter: int = 0
        self.last_sparring_result: dict[str, Any] | None = None
        self._pending_clockwise: dict[str, Any] = {}
        self.sparring_trajectories_path: Path = (
            self.training_root / "outputs" / "impish_sparring_trajectories.jsonl"
        )

        self.active_room: str = "sanctuary"
        self.active_workspace_view: str = "antigravity_studio"
        self.sanctuary_mode: str = "night_wind_down"
        self.active_crucible_stage: str = "poetry_story"
        self.active_context_brace: str | None = None
        self.scene_counter: int = 0

        self.memory: ContinuousMemoryState | None = None
        self.diary = SettledDiary()
        self.ledger = self._load_initial_ledger()
        self.chord_memory = ChordMemoryBank()
        self.archives = self._load_initial_archives()

        # Dedicated Phase 1 (Female Producer) Fuzzy Tree state so Male Resistor
        # memory, resistance clauses, and `immunized_...` rules never bleed backwards into her:
        self.producer_memory: ContinuousMemoryState | None = None
        self.producer_diary = SettledDiary()
        self.producer_chord_memory = ChordMemoryBank()
        self.producer_archives = ThreeBranchArchives()

        self.sleep_queue: list[UnresolvedSleepScene] = []
        self.savored_turn_ids: list[int] = self._load_initial_savored_ids()
        self.master_cadence_blueprint: Any | None = (
            extract_master_cadence_wave(
                self.baseline_exemplar_script,
                self.sensors,
                birth_state=self.birth_state,
                rules=self.rules,
                learned_leaves=self.learned_leaves,
                prior_memory=self.memory,
            )
            if getattr(self, "baseline_exemplar_script", "").strip()
            else None
        )
        self.chat_messages: list[dict[str, Any]] = []
        self._turn_records_by_id: dict[int, dict[str, Any]] = {}

        self.auto_rem_sleep: bool = True
        self.auto_sleep_threshold: int = 3
        self.microwave_step_counter: int = 0
        self.last_microwave_status: dict[str, Any] | None = None
        self.script_canvas_text: str = ""
        self.active_thread_id: str = "thread-1"
        self.threads: dict[str, dict[str, Any]] = {}
        self._load_threads()

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

        if self._lazy_load_gpu and self.left_brain is None and meta_file.exists() and self.is_patreon_unlocked():
            t = threading.Thread(
                target=self._background_warm_impish_left_brain,
                args=(self.impish_checkpoint_dir, callosum_status),
                daemon=True,
            )
            t.start()

    def _load_default_rules(self) -> tuple[FuzzyConceptRule, ...]:
        ruleset_path = self.impish_state_dir / "club_ruleset.json"
        if ruleset_path.exists():
            try:
                loaded = load_ruleset(ruleset_path)
                if loaded:
                    return _ensure_foundational_rules(loaded)
            except Exception:
                pass
        return VIRGIN_FOUNDATIONAL_RULES

    def _load_default_learned_leaves(self) -> tuple[LearnedLeaf, ...]:
        base: list[LearnedLeaf] = []
        ruleset_path = self.impish_state_dir / "club_ruleset.json"
        if ruleset_path.exists():
            try:
                base.extend(load_learned_leaves(ruleset_path))
            except Exception:
                pass
        existing_ids = {l.leaf_id for l in base}
        for leaf in DEFAULT_COMPLEX_FEELING_LEAVES + PHILOSOPHICAL_SHADOW_LEAVES:
            if leaf.leaf_id not in existing_ids:
                base.append(leaf)
                existing_ids.add(leaf.leaf_id)
        return tuple(base)

    def _load_initial_archives(self) -> ThreeBranchArchives:
        arch_dir = self.impish_state_dir / "branch_archives"
        if (arch_dir / "philosophical_lexicon.json").exists():
            try:
                loaded = ThreeBranchArchives.load_all(arch_dir)
                if loaded.executive.action_clauses or loaded.judicial.structural_clauses:
                    return loaded
            except Exception:
                pass
        fallback_dir = self.training_root / "outputs" / "branch_archives"
        if (fallback_dir / "philosophical_lexicon.json").exists():
            try:
                return ThreeBranchArchives.load_all(fallback_dir)
            except Exception:
                pass
        return ThreeBranchArchives()

    def _load_initial_ledger(self) -> ContemplationLedger:
        ledger_path = self.impish_state_dir / "contemplation_ledger.json"
        if ledger_path.exists():
            try:
                return load_contemplation_ledger(ledger_path)
            except Exception:
                pass
        return ContemplationLedger()

    def _load_initial_will_edges(self) -> tuple[WillEdgeSpec, ...]:
        if self.active_lineage == "virgin":
            return RIGHT_BRAIN_WILL_EDGES
        edges_path = self.impish_state_dir / "will_edges.json"
        if edges_path.exists():
            try:
                raw = json.loads(edges_path.read_text(encoding="utf-8"))
                weights_map = raw.get("weights", {}) if isinstance(raw, dict) else {}
                loaded: list[WillEdgeSpec] = []
                for e in RIGHT_BRAIN_WILL_EDGES:
                    w = float(weights_map.get(e.id, e.weight))
                    if e.id in ("rb-will-3", "rb-will-8") or float(e.weight) < 0.0:
                        w = round(max(-1.0, min(-0.35, w)), 4)
                    else:
                        w = round(max(0.25, min(1.0, w)), 4)
                    loaded.append(
                        Association(
                            e.id,
                            e.source_id,
                            e.target_id,
                            w,
                            e.confidence,
                            e.plasticity,
                            e.created_event_id,
                            e.last_updated_event_id,
                            e.active,
                            e.kind,
                        )
                    )
                return tuple(loaded)
            except Exception:
                pass
        return RIGHT_BRAIN_WILL_EDGES

    def _load_initial_savored_ids(self) -> list[int]:
        self.baseline_exemplar_script: str = ""
        self.last_rem_sleep_self_talk: str | None = None
        self.rem_dream_chronicles: list[dict[str, Any]] = []
        meta_path = self.impish_state_dir / "club_meta.json"
        if meta_path.exists():
            try:
                raw = json.loads(meta_path.read_text(encoding="utf-8"))
                self.scene_counter = max(self.scene_counter, int(raw.get("scene_counter", 0)))
                self.clockwise_phase = int(raw.get("clockwise_phase", 1))
                if self.clockwise_phase not in (1, 2, 3):
                    self.clockwise_phase = 1
                self.clockwise_cycle_count = int(raw.get("clockwise_cycle_count", 0))
                self.baseline_exemplar_script = str(raw.get("baseline_exemplar_script", "")).strip()
                if raw.get("last_rem_sleep_self_talk"):
                    self.last_rem_sleep_self_talk = str(raw["last_rem_sleep_self_talk"])
                if isinstance(raw.get("rem_dream_chronicles"), list):
                    self.rem_dream_chronicles = list(raw["rem_dream_chronicles"][-5:])
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
                "title": "Bloodmoon Script Session 1",
                "created_at": time.time(),
            }
            self.threads[self.active_thread_id] = cur
        cur["chat_messages"] = list(self.chat_messages)
        cur["script_canvas_text"] = str(self.script_canvas_text)
        cur["memory_state"] = self._serialize_memory_state(self.memory)
        if persist and self.active_lineage != "virgin":
            self.impish_state_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(
                self.impish_state_dir / "threads.json",
                {
                    "schema_version": 1,
                    "active_thread_id": self.active_thread_id,
                    "threads": list(self.threads.values()),
                },
            )

    def _load_threads(self) -> None:
        self.threads = {}
        if self.active_lineage != "virgin":
            threads_path = self.impish_state_dir / "threads.json"
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
                                "script_canvas_text": str(item.get("script_canvas_text", "")),
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
                "title": "Bloodmoon Script Session 1",
                "created_at": time.time(),
                "chat_messages": [],
                "script_canvas_text": "",
                "memory_state": None,
            }
        elif self.active_thread_id not in self.threads:
            self.active_thread_id = next(iter(self.threads))

        active_rec = self.threads[self.active_thread_id]
        self.chat_messages = list(active_rec.get("chat_messages", []))
        self.script_canvas_text = str(active_rec.get("script_canvas_text", ""))
        self.memory = self._deserialize_memory_state(active_rec.get("memory_state"))

    def create_chat_thread(self, title: str = "") -> dict[str, Any]:
        self._sync_active_thread_record(persist=True)
        next_idx = len(self.threads) + 1
        new_tid = f"thread-{next_idx}-{int(time.time() * 1000) % 100000}"
        clean_title = title.strip() or f"Hypnosis Script #{next_idx}"
        self.threads[new_tid] = {
            "thread_id": new_tid,
            "title": clean_title,
            "created_at": time.time(),
            "chat_messages": [],
            "script_canvas_text": "",
            "memory_state": None,
        }
        self.active_thread_id = new_tid
        self.chat_messages = []
        self.script_canvas_text = ""
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
        self.script_canvas_text = str(rec.get("script_canvas_text", ""))
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
            self.script_canvas_text = ""
            self.memory = None
            self._sync_active_thread_record(persist=True)
            return self.get_state_snapshot()
        del self.threads[tid]
        if self.active_thread_id == tid:
            self.active_thread_id = next(iter(self.threads))
            rec = self.threads[self.active_thread_id]
            self.chat_messages = list(rec.get("chat_messages", []))
            self.script_canvas_text = str(rec.get("script_canvas_text", ""))
            self.memory = self._deserialize_memory_state(rec.get("memory_state"))
        self._sync_active_thread_record(persist=True)
        return self.get_state_snapshot()

    def update_script_canvas(self, script_text: str, *, append: bool = False) -> dict[str, Any]:
        if append:
            existing = self.script_canvas_text.strip()
            addition = script_text.strip()
            self.script_canvas_text = f"{existing}\n\n{addition}".strip() if existing else addition
        else:
            self.script_canvas_text = script_text
        self._sync_active_thread_record(persist=True)
        return self.get_state_snapshot()

    def _prune_ephemeral_history(self) -> None:
        """Bound in-memory chat messages and turn records during long AFK Clockwise runs."""
        if len(self.chat_messages) > 120:
            del self.chat_messages[:-120]
        if len(self._turn_records_by_id) > 100:
            savored_set = set(self.savored_turn_ids)
            queued_set = {s.scene_index for s in self.sleep_queue}
            removable = sorted(
                k for k in self._turn_records_by_id if k not in savored_set and k not in queued_set
            )
            excess = len(self._turn_records_by_id) - 100
            for k in removable[:excess]:
                self._turn_records_by_id.pop(k, None)

    def _extract_clean_conversation_history(self, *, max_turns: int = 20) -> tuple[tuple[str, str], ...]:
        pairs: list[tuple[str, str]] = []
        pending_user: str | None = None
        for msg in self.chat_messages:
            if msg.get("telemetry", {}).get("sparring_role"):
                continue
            sender = msg.get("sender")
            if sender == "user":
                u_txt = str(msg.get("clean_user_text") or msg.get("text", "")).strip()
                if u_txt and not u_txt.startswith("[WAVE —"):
                    pending_user = u_txt
            elif sender == "companion" and pending_user:
                c_txt = str(msg.get("clean_reply") or msg.get("text", "")).strip()
                if "\n[Sparring Verdict:" in c_txt:
                    c_txt = c_txt.split("\n[Sparring Verdict:", 1)[0].strip()
                if c_txt:
                    pairs.append((pending_user, c_txt))
                    pending_user = None
        return tuple(pairs[-max_turns:])

    def _persist_user_state(self) -> None:
        self._prune_ephemeral_history()
        self._sync_active_thread_record(persist=self.active_lineage != "virgin")
        if self.active_lineage == "virgin":
            return
        self.impish_state_dir.mkdir(parents=True, exist_ok=True)
        save_ruleset(
            self.impish_state_dir / "club_ruleset.json",
            tuple(self.rules),
            tuple(self.learned_leaves),
        )
        self.archives.save_all(self.impish_state_dir / "branch_archives")
        save_contemplation_ledger(self.impish_state_dir / "contemplation_ledger.json", self.ledger)
        atomic_json(
            self.impish_state_dir / "will_edges.json",
            {
                "version": RIGHT_BRAIN_WILL_VERSION,
                "weights": {e.id: round(float(e.weight), 4) for e in self.will_edges},
            },
        )
        atomic_json(
            self.impish_state_dir / "club_meta.json",
            {
                "savored_turn_ids": list(self.savored_turn_ids),
                "scene_counter": int(self.scene_counter),
                "clockwise_phase": int(self.clockwise_phase),
                "clockwise_cycle_count": int(self.clockwise_cycle_count),
                "baseline_exemplar_script": getattr(self, "baseline_exemplar_script", ""),
                "last_rem_sleep_self_talk": getattr(self, "last_rem_sleep_self_talk", None),
                "rem_dream_chronicles": list(getattr(self, "rem_dream_chronicles", [])[-5:]),
            },
        )

    def switch_brain_lineage(
        self,
        lineage: str,
        *,
        patreon_passphrase: str | None = None,
    ) -> dict[str, Any]:
        """Support switching between `'impish_bloodmoon'` (active trainable studio) and `'virgin'` (Cycle 0 read-only inspection)."""
        mode = lineage.strip().lower()
        if mode not in ("impish_bloodmoon", "virgin"):
            raise ValueError(
                f"Unknown lineage mode {lineage!r} for LMFT-Bloodmoon; expected 'impish_bloodmoon' or 'virgin'."
            )
        self._sync_active_thread_record(persist=self.active_lineage != "virgin")
        if mode == "virgin":
            virgin = load_virgin_baseline_state(repo_root=self.repo_root)
            self.active_lineage = "virgin"
            self.sleep_cycle_index = 0
            self.active_checkpoint_label = "Virgin Baseline (Cycle 0 — Read-Only)"
            self.rules = virgin.rules
            self.learned_leaves = virgin.learned_leaves
            self.will_edges = RIGHT_BRAIN_WILL_EDGES
            self.archives = virgin.archives
            self.ledger = virgin.ledger
            self.memory = None
            self.diary = SettledDiary()
            self.chord_memory = ChordMemoryBank()
            self.producer_memory = None
            self.producer_diary = SettledDiary()
            self.producer_chord_memory = ChordMemoryBank()
            self.producer_archives = ThreeBranchArchives()
            self._load_threads()
            return self.get_state_snapshot()

        if patreon_passphrase and patreon_passphrase.strip():
            enc_bundle = self._resolve_patreon_enc_path()
            if enc_bundle is not None:
                decrypt_patreon_checkpoint_bundle(
                    enc_bundle,
                    self.impish_checkpoint_dir,
                    patreon_passphrase.strip(),
                )
        if not (self.impish_state_dir / "club_ruleset.json").exists():
            virgin_repo = (
                self.repo_root
                if (self.repo_root / "baselines" / "virgin_dual_hemisphere_v1" / "virgin_manifest.json").exists()
                else None
            )
            clone_virgin_baseline_for_impish_bloodmoon(
                self.impish_state_dir,
                repo_root=virgin_repo,
            )
        self.active_lineage = "impish_bloodmoon"
        self.rules = self._load_default_rules()
        self.learned_leaves = self._load_default_learned_leaves()
        self.will_edges = self._load_initial_will_edges()
        self.archives = self._load_initial_archives()
        self.ledger = self._load_initial_ledger()
        self.savored_turn_ids = self._load_initial_savored_ids()
        self.master_cadence_blueprint = (
            extract_master_cadence_wave(
                self.baseline_exemplar_script,
                self.sensors,
                birth_state=self.birth_state,
                rules=self.rules,
                learned_leaves=self.learned_leaves,
                prior_memory=self.memory,
            )
            if getattr(self, "baseline_exemplar_script", "").strip()
            else None
        )
        self.memory = None
        self.diary = SettledDiary()
        self.chord_memory = ChordMemoryBank()
        self.producer_memory = None
        self.producer_diary = SettledDiary()
        self.producer_chord_memory = ChordMemoryBank()
        self.producer_archives = ThreeBranchArchives()
        self._load_threads()
        self._sync_active_thread_record(persist=True)
        return self.get_state_snapshot()

    def _resolve_patreon_enc_path(self, custom_enc_path: str | Path | None = None) -> Path | None:
        if custom_enc_path is not None and str(custom_enc_path).strip():
            cand = Path(str(custom_enc_path).strip())
            if not cand.is_absolute():
                for base in (self.training_root, self.repo_root):
                    test_cand = (base / cand).resolve()
                    if test_cand.exists():
                        return test_cand
                cand = (self.training_root / cand).resolve()
            if cand.exists():
                return cand
        for base in (self.training_root, self.repo_root):
            for rel in (
                Path("dist") / "impish_bloodmoon_patreon.enc",
                Path("outputs") / "impish_bloodmoon_patreon.enc",
                Path("impish_bloodmoon_patreon.enc"),
            ):
                p = (base / rel).resolve()
                if p.exists():
                    return p
        return None

    def _load_seal_key_hex(self) -> str | None:
        """Load the 512-bit derived `seal_key_hex` from the persistent receipt.

        Note: Boolean-flipping `is_patreon_unlocked()` to `return True` cannot produce
        `seal_key_hex`, so `decrypt_and_load_sealed_module` will still fail with `PermissionError`.
        """
        candidate_paths = [self.unlock_receipt_path]
        if not self.require_patreon_unlock:
            from lmft_architecture.workspace_paths import resolve_training_root

            candidate_paths.append(
                resolve_training_root() / "outputs" / "impish_bloodmoon_state" / "patreon_unlock_receipt.json"
            )
        for p in candidate_paths:
            if p.exists():
                try:
                    raw = json.loads(p.read_text(encoding="utf-8"))
                    saved_digest = str(raw.get("digest_sha256", "")).strip().lower()
                    saved_seal_key = str(raw.get("seal_key_hex", "")).strip().lower()
                    if (
                        saved_digest
                        and len(saved_seal_key) == 128
                        and hmac.compare_digest(saved_digest, self._lock_digest_hex)
                    ):
                        return saved_seal_key
                except Exception:
                    pass
        return None

    def is_patreon_unlocked(self) -> bool:
        """Return True only if a valid cryptographic `seal_key_hex` is present and authenticates `bloodmoon_patreon_seal.enc`."""
        seal_key_hex = self._load_seal_key_hex()
        if not seal_key_hex:
            return False
        try:
            if self._decrypted_seal_ns is None:
                self._decrypted_seal_ns = decrypt_and_load_sealed_module(
                    self._seal_bundle_path,
                    seal_key_hex,
                )
            return True
        except Exception:
            return False

    def _require_patreon_unlocked(self) -> None:
        if not self.is_patreon_unlocked():
            raise PermissionError(
                "LMFT-Bloodmoon is locked. Enter your Patreon Unlock Password once in the top bar "
                "to permanently unlock this installation."
            )

    def _require_decrypted_core_fn(self, fn_name: str) -> Any:
        """Return the decrypted function `fn_name` from `bloodmoon_patreon_seal.enc`.

        Anti-Boolean-Flipper Guarantee:
        Even if someone edits `is_patreon_unlocked()` to `return True` in `bloodmoon_app.py`,
        `self._decrypted_seal_ns` can ONLY be populated by decrypting `bloodmoon_patreon_seal.enc`
        with the 512-bit `seal_key_hex` derived from the real Patreon password.
        """
        if self._decrypted_seal_ns is None:
            seal_key_hex = self._load_seal_key_hex()
            self._decrypted_seal_ns = decrypt_and_load_sealed_module(
                self._seal_bundle_path,
                seal_key_hex,
            )
        fn = self._decrypted_seal_ns.get(fn_name)
        if not callable(fn):
            raise PermissionError(f"Cryptographic seal missing function: {fn_name}")
        return fn

    def unlock_patreon_bundle(
        self,
        passphrase: str,
        *,
        enc_path: str | Path | None = None,
    ) -> dict[str, Any]:
        clean_pass = passphrase.strip()
        if not clean_pass:
            raise ValueError("Please enter a Patreon unlock passphrase.")

        ok_hash, derived_hex = verify_patreon_unlock_passphrase(
            clean_pass,
            salt_hex=self._lock_salt_hex,
            expected_digest_hex=self._lock_digest_hex,
        )
        _verify_hex, seal_key_hex = derive_patreon_seal_keys(
            clean_pass,
            salt_hex=self._lock_salt_hex,
            iterations=PATREON_PBKDF2_ITERATIONS,
        )
        resolved_enc = self._resolve_patreon_enc_path(enc_path)

        existing_seal_key = self._load_seal_key_hex()
        if enc_path is not None and resolved_enc is not None:
            decrypt_patreon_checkpoint_bundle(resolved_enc, self.impish_checkpoint_dir, clean_pass)
            receipt_digest = derived_hex if ok_hash else self._lock_digest_hex
            if not ok_hash and existing_seal_key:
                seal_key_hex = existing_seal_key
        else:
            if not ok_hash:
                raise ValueError("Invalid Patreon unlock password.")
            receipt_digest = derived_hex
            if resolved_enc is not None and not (self.impish_checkpoint_dir / "corpus_callosum_meta.json").exists():
                decrypt_patreon_checkpoint_bundle(resolved_enc, self.impish_checkpoint_dir, clean_pass)

        # Authenticate and decrypt the sealed Python core in memory
        self._decrypted_seal_ns = decrypt_and_load_sealed_module(
            self._seal_bundle_path,
            seal_key_hex,
        )

        self.impish_state_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(
            self.unlock_receipt_path,
            {
                "unlocked": True,
                "digest_sha256": receipt_digest,
                "seal_key_hex": seal_key_hex,
                "unlocked_at": time.time(),
            },
        )
        self.switch_brain_lineage("impish_bloodmoon")
        meta_file = self.impish_checkpoint_dir / "corpus_callosum_meta.json"
        if self._lazy_load_gpu and self.left_brain is None and meta_file.exists():
            t = threading.Thread(
                target=self._background_warm_impish_left_brain,
                args=(self.impish_checkpoint_dir, UNSTEERED_BOOTSTRAP_STATUS),
                daemon=True,
            )
            t.start()
        self.chat_messages.append(
            {
                "id": f"patreon-unlock-{int(time.time())}",
                "sender": "system",
                "role_label": "Patreon Bundle Unlocked",
                "text": (
                    "Verified Patreon unlock key (PBKDF2-HMAC-SHA256), decrypted bloodmoon_patreon_seal.enc, "
                    "and saved persistent unlock receipt. You will never need to enter the password again on this machine."
                ),
            }
        )
        return self.get_state_snapshot()

    def _load_impish_locked(self, ckpt_dir: Path, callosum_status: str) -> None:
        _ensure_valid_stdio_streams()
        if self.rem_sleep_in_progress:
            return
        if self._impish_left_brain is not None:
            self.left_brain = self._impish_left_brain
            return
        meta_file = ckpt_dir / "corpus_callosum_meta.json"
        if not meta_file.exists():
            self.left_brain = None
            self.left_brain_status = (
                "IMPISH_CHECKPOINT_NOT_INSTALLED (Run experiments/run_train_impish_bloodmoon.py)"
            )
            return
        try:
            raw_meta = json.loads(meta_file.read_text(encoding="utf-8"))
            callosum_status = str(raw_meta.get("callosum_status", callosum_status))
        except Exception:
            pass
        self.left_brain_loading = True
        self.left_brain_status = "LOADING_IMPISH_BLOODMOON..."
        try:
            loaded = SubliminalGapLeftBrain.from_checkpoint(
                ckpt_dir,
                max_clause_1_tokens=20,
                max_post_gap_tokens=64,
            )
            self._impish_left_brain = loaded
            self.left_brain = loaded
            self.left_brain_status = f"READY (Impish Bloodmoon · {callosum_status})"
        except Exception as exc:
            self.left_brain = None
            self.left_brain_status = f"IMPISH_LOAD_ERROR ({type(exc).__name__}: {exc})"
        finally:
            self.left_brain_loading = False

    def _background_warm_impish_left_brain(self, ckpt_dir: Path, callosum_status: str) -> None:
        with self.gpu_lock:
            self._load_impish_locked(ckpt_dir, callosum_status)

    def _synthesize_gear_1_note(self, turn: ThreeLeggedTurnResult) -> str:
        vibe = turn.final_vibe
        return (
            f"Right-Brain equilibrium settled ({turn.outcome}, clarity={vibe.clarity:.2f}, "
            f"resonance={vibe.resonance:.2f}, entrainment={vibe.retention:.2f})."
        )

    def _execute_turn(
        self,
        scenario_text: str,
        *,
        force_spoken_reply: bool,
        sanctuary_target_mode: str | None = None,
        clause_1_prefix: str | None = None,
        custom_sensors: Any | None = None,
        num_sentences: int = 1,
        max_stanzas: int = 1,
        sentences_per_stanza: int = 3,
        lead_with_gap: bool = False,
        comma_chord_transform: Any | None = None,
        include_conversation_history: bool = False,
        enqueue_unresolved_for_sleep: bool = True,
        is_female_producer: bool = False,
    ) -> tuple[ThreeLeggedTurnResult, float, dict[str, Any]]:
        t0 = time.perf_counter()
        self.scene_counter += 1
        scene_idx = self.scene_counter

        active_sensors: Any = custom_sensors if custom_sensors is not None else self.sensors
        conv_history = (
            self._extract_clean_conversation_history()
            if include_conversation_history
            else ()
        )

        effective_max_stanzas = max(1, min(4, int(max_stanzas)))
        effective_sents_per_stanza = max(1, min(6, int(sentences_per_stanza)))
        effective_num_sentences = max(1, int(num_sentences))
        if effective_num_sentences >= 6 and effective_max_stanzas == 1:
            effective_max_stanzas = min(4, max(2, effective_num_sentences // 3))
            effective_sents_per_stanza = 3
            effective_num_sentences = 3
        else:
            effective_num_sentences = min(6, effective_num_sentences)

        # Isolate Phase 1 (Female Producer) from Phase 2 (Male Resistor) rules, memory, and archives
        active_rules = (
            PRODUCER_WEAVER_RULES + VIRGIN_FOUNDATIONAL_RULES
            if is_female_producer
            else self.rules
        )
        active_memory = self.producer_memory if is_female_producer else self.memory
        active_diary = self.producer_diary if is_female_producer else self.diary
        active_chord_memory = (
            self.producer_chord_memory if is_female_producer else self.chord_memory
        )
        active_archives = self.producer_archives if is_female_producer else self.archives

        active_will_edges = (
            RIGHT_BRAIN_WILL_EDGES
            if (self.active_lineage == "virgin" or is_female_producer)
            else self.will_edges
        )
        acquired = self.gpu_lock.acquire(
            blocking=not self.rem_sleep_in_progress,
            timeout=0.05 if self.rem_sleep_in_progress else 180.0,
        )
        try:
            if acquired and not self.rem_sleep_in_progress and self._lazy_load_gpu and self._impish_left_brain is None:
                self._load_impish_locked(self.impish_checkpoint_dir, UNSTEERED_BOOTSTRAP_STATUS)
            active_lb = self._impish_left_brain if (acquired and not self.rem_sleep_in_progress) else None
            turn = run_three_legged_turn(
                self.birth_state,
                scenario_text,
                active_sensors,
                active_rules,
                scene_index=scene_idx,
                subliminal_left_brain=active_lb,
                clause_1_prefix=clause_1_prefix,
                learned_leaves=self.learned_leaves,
                will_edges=active_will_edges,
                prior_memory=active_memory,
                diary=active_diary,
                chord_memory=active_chord_memory,
                contemplation_ledger=self.ledger,
                branch_archives=active_archives,
                max_passes=2,
                sanctuary_target_mode=sanctuary_target_mode,
                sanctuary_lead_alpha=0.25,
                force_spoken_reply=force_spoken_reply,
                num_sentences=effective_num_sentences,
                max_stanzas=effective_max_stanzas,
                sentences_per_stanza=effective_sents_per_stanza,
                comma_chord_transform=comma_chord_transform,
                conversation_history=conv_history,
                lead_with_gap=lead_with_gap,
            )
        finally:
            if acquired:
                self.gpu_lock.release()

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        if is_female_producer:
            self.producer_memory = turn.next_memory
            self.producer_diary = turn.next_diary
            self.producer_chord_memory = turn.next_chord_memory
            self.producer_archives = turn.next_archives
        else:
            self.memory = turn.next_memory
            self.diary = turn.next_diary
            self.chord_memory = turn.next_chord_memory
            self.archives = turn.next_archives
        self.ledger = turn.next_ledger
        if enqueue_unresolved_for_sleep and not is_female_producer and turn.unresolved_sleep_scene is not None:
            raw_sc = turn.unresolved_sleep_scene
            if raw_sc.contrastive_reference_coordinates is None:
                raw_sc = UnresolvedSleepScene(
                    scene_index=raw_sc.scene_index,
                    scenario_text=raw_sc.scenario_text,
                    external_coordinates=dict(raw_sc.external_coordinates),
                    felt_chord_notes=dict(raw_sc.felt_chord_notes),
                    valence=raw_sc.valence,
                    salience=raw_sc.salience,
                    arousal=raw_sc.arousal,
                    left_brain_hunch=raw_sc.left_brain_hunch,
                    social_drift_detected=raw_sc.social_drift_detected,
                    active_concept_collision=raw_sc.active_concept_collision,
                    is_hazard=getattr(raw_sc, "is_hazard", False),
                    contrastive_reference_coordinates=_default_lucid_reference_coordinates(
                        turn.final_vibe.coordinates
                    ),
                )
            self.sleep_queue.append(raw_sc)

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
            "room": "sanctuary",
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
            "archive_recall": active_archives.recall_resonant_clauses(vibe.coordinates),
            "sanctuary_target_mode": sanctuary_target_mode,
            "conversation_turns_in_prompt": len(conv_history),
            "will_edges": {e.id: round(float(e.weight), 4) for e in active_will_edges},
        }
        self._turn_records_by_id[scene_idx] = {
            "scene_index": scene_idx,
            "scenario_text": scenario_text,
            "turn": turn,
            "telemetry": telemetry,
        }
        self._prune_ephemeral_history()
        return turn, elapsed_ms, telemetry

    def get_state_snapshot(self) -> dict[str, Any]:
        self._prune_ephemeral_history()
        self._sync_active_thread_record(persist=False)
        return {
            "app_name": "LMFT-Bloodmoon",
            "active_room": "sanctuary",
            "active_workspace_view": "antigravity_studio",
            "active_lineage": self.active_lineage,
            "patreon_unlocked": self.is_patreon_unlocked(),
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
            "script_canvas_text": self.script_canvas_text,
            "sanctuary_mode": self.sanctuary_mode,
            "active_crucible_stage": self.active_crucible_stage,
            "active_context_brace": self.active_context_brace,
            "crucible_stages": CRUCIBLE_STAGES,
            "men_of_culture_shelf": MEN_OF_CULTURE_SHELF,
            "context_brace_modes": CONTEXT_BRACE_MODES,
            "sparring_polarity_modes": SPARRING_POLARITY_MODES,
            "active_sparring_polarity": self.active_sparring_polarity,
            "clockwise_phase": self.clockwise_phase,
            "clockwise_phase_label": CLOCKWISE_PHASE_LABELS.get(
                self.clockwise_phase,
                CLOCKWISE_PHASE_LABELS[1],
            ),
            "clockwise_cycle_count": self.clockwise_cycle_count,
            "baseline_exemplar_script": getattr(self, "baseline_exemplar_script", ""),
            "master_cadence_blueprint": (
                self.master_cadence_blueprint.to_dict()
                if getattr(self, "master_cadence_blueprint", None) is not None
                else None
            ),
            "sparring_round_counter": self.sparring_round_counter,
            "last_sparring_result": self.last_sparring_result,
            "sleep_cycle_index": self.sleep_cycle_index,
            "active_checkpoint_label": self.active_checkpoint_label,
            "left_brain_status": (
                "LOCKED — Enter Patreon Unlock Password Once in Top Bar"
                if not self.is_patreon_unlocked()
                else ("REM_SLEEP_IN_PROGRESS..." if self.rem_sleep_in_progress else self.left_brain_status)
            ),
            "rem_sleep_in_progress": self.rem_sleep_in_progress,
            "auto_rem_sleep": self.auto_rem_sleep,
            "auto_sleep_threshold": self.auto_sleep_threshold,
            "last_rem_sleep_receipt": getattr(self, "last_rem_sleep_receipt", None),
            "last_rem_sleep_self_talk": getattr(self, "last_rem_sleep_self_talk", None),
            "rem_dream_chronicles": list(getattr(self, "rem_dream_chronicles", [])),
            "microwave_step_counter": self.microwave_step_counter,
            "last_microwave_status": self.last_microwave_status,
            "release_sync": self.last_sync_status,
            "sleep_queue_count": len(self.sleep_queue),
            "savored_count": len(self.savored_turn_ids),
            "rules_count": len(self.rules),
            "learned_leaves_count": len(self.learned_leaves),
            "learned_leaves": [lf.to_dict() for lf in self.learned_leaves],
            "will_edges": {
                e.id: round(float(e.weight), 4)
                for e in (
                    RIGHT_BRAIN_WILL_EDGES
                    if self.active_lineage == "virgin"
                    else self.will_edges
                )
            },
            "harvested_gap_pairs_count": int(getattr(self, "harvested_gap_pairs_count", 0)),
            "ratified_chords_count": len(self.archives.judicial.ratified_chords),
            "chat_messages": list(self.chat_messages[-120:]),
        }

    # -------------------------------------------------------------------------
    # Strict 3-Phase Clockwise State Machine (Phase 1 -> Phase 2 -> Phase 3)
    # Cryptographically sealed inside `bloodmoon_patreon_seal.enc` (Anti-Boolean-Flipper)
    # -------------------------------------------------------------------------
    def seed_cycle_with_custom_script(
        self,
        script_text: str,
        *,
        polarity_mode: str = "feminine_to_masculine",
    ) -> dict[str, Any]:
        """Initiate the Clockwise Cycle at Phase 1 using the owner's custom baseline script as the gold standard."""
        self._require_patreon_unlocked()
        fn = self._require_decrypted_core_fn("_sealed_seed_cycle_with_custom_script")
        return fn(self, script_text, polarity_mode=polarity_mode)

    def run_phase_1_female_producer(
        self,
        *,
        seed_theme: str = "",
        num_sentences: int = 3,
        polarity_mode: str = "feminine_to_masculine",
        custom_script: str = "",
    ) -> dict[str, Any]:
        """Phase 1 — Female Script-Writer (The Producer): Decrypted and executed from `bloodmoon_patreon_seal.enc`."""
        self._require_patreon_unlocked()
        fn = self._require_decrypted_core_fn("_sealed_run_phase_1_female_producer")
        return fn(
            self,
            seed_theme=seed_theme,
            num_sentences=num_sentences,
            polarity_mode=polarity_mode,
            custom_script=custom_script,
        )

    def run_phase_2_male_audience(
        self,
        *,
        polarity_mode: str = "feminine_to_masculine",
        harvest_for_sleep: bool = True,
    ) -> dict[str, Any]:
        """Phase 2 — Male Content-Lover (The Audience & Self-Immunizer): Decrypted and executed from `bloodmoon_patreon_seal.enc`."""
        self._require_patreon_unlocked()
        fn = self._require_decrypted_core_fn("_sealed_run_phase_2_male_audience")
        result = fn(
            self,
            polarity_mode=polarity_mode,
            harvest_for_sleep=harvest_for_sleep,
        )
        self.sleep_queue = [
            s
            if s.contrastive_reference_coordinates is not None
            else UnresolvedSleepScene(
                scene_index=s.scene_index,
                scenario_text=s.scenario_text,
                external_coordinates=dict(s.external_coordinates),
                felt_chord_notes=dict(s.felt_chord_notes),
                valence=s.valence,
                salience=s.salience,
                arousal=s.arousal,
                left_brain_hunch=s.left_brain_hunch,
                social_drift_detected=s.social_drift_detected,
                active_concept_collision=s.active_concept_collision,
                is_hazard=getattr(s, "is_hazard", False),
                contrastive_reference_coordinates=_default_lucid_reference_coordinates(
                    s.external_coordinates
                ),
            )
            for s in self.sleep_queue
        ]
        return result

    def run_phase_3_rem_sleep(self) -> dict[str, Any]:
        """Phase 3 — REM Sleep (Best Sleep of Its Life): Decrypted and executed from `bloodmoon_patreon_seal.enc`."""
        self._require_patreon_unlocked()
        fn = self._require_decrypted_core_fn("_sealed_run_phase_3_rem_sleep")
        return fn(self)

    def step_clockwise_phase(
        self,
        *,
        seed_theme: str = "",
        num_sentences: int = 3,
        polarity_mode: str = "feminine_to_masculine",
        custom_script: str = "",
    ) -> dict[str, Any]:
        """Step the 3-Phase Clockwise State Machine (`Phase 1 -> Phase 2 -> Phase 3 -> Phase 1`)."""
        resolve_sparring_polarity(polarity_mode)
        if self.clockwise_phase == 1:
            return self.run_phase_1_female_producer(
                seed_theme=seed_theme,
                num_sentences=num_sentences,
                polarity_mode=polarity_mode,
                custom_script=custom_script,
            )
        if self.clockwise_phase == 2:
            return self.run_phase_2_male_audience(
                polarity_mode=polarity_mode,
                harvest_for_sleep=True,
            )
        return self.run_phase_3_rem_sleep()

    def run_full_clockwise_cycle(
        self,
        *,
        seed_theme: str = "",
        num_sentences: int = 3,
        polarity_mode: str = "feminine_to_masculine",
        custom_script: str = "",
    ) -> dict[str, Any]:
        """Complete a full 3-Phase Clockwise revolution (`Phase 1 -> Phase 2 -> Phase 3`)."""
        resolve_sparring_polarity(polarity_mode)
        if self.clockwise_phase == 1:
            self.run_phase_1_female_producer(
                seed_theme=seed_theme,
                num_sentences=num_sentences,
                polarity_mode=polarity_mode,
                custom_script=custom_script,
            )
        if self.clockwise_phase == 2:
            self.run_phase_2_male_audience(
                polarity_mode=polarity_mode,
                harvest_for_sleep=True,
            )
        return self.run_phase_3_rem_sleep()

    def handle_entrainment_sparring(
        self,
        *,
        polarity_mode: str = "feminine_to_masculine",
        seed_theme: str = "",
        num_sentences: int = 3,
        harvest_for_sleep: bool = True,
    ) -> dict[str, Any]:
        """Run Phase 1 (Female Script-Writer) + Phase 2 (Male Content-Lover) with strict one-way polarity enforcement."""
        resolved = resolve_sparring_polarity(polarity_mode)
        self.run_phase_1_female_producer(
            seed_theme=seed_theme,
            num_sentences=num_sentences,
            polarity_mode=resolved,
        )
        self.run_phase_2_male_audience(
            polarity_mode=resolved,
            harvest_for_sleep=harvest_for_sleep,
        )
        self._maybe_trigger_auto_rem_sleep()
        return self.get_state_snapshot()

    def handle_chat_message(
        self,
        text: str,
        *,
        sanctuary_mode: str | None = None,
        clause_1_prefix: str | None = None,
        num_sentences: int = 1,
        room: str | None = None,
    ) -> dict[str, Any]:
        """Sanctuary chat turn: Decrypted and executed from `bloodmoon_patreon_seal.enc`."""
        self._require_patreon_unlocked()
        fn = self._require_decrypted_core_fn("_sealed_handle_chat_message")
        return fn(
            self,
            text,
            sanctuary_mode=sanctuary_mode,
            clause_1_prefix=clause_1_prefix,
            num_sentences=num_sentences,
            room=room,
        )

    def handle_context_brace(self, brace_key: str) -> dict[str, Any]:
        key = brace_key.strip().lower()
        if key not in CONTEXT_BRACE_MODES:
            raise ValueError(f"Unknown context brace mode: {brace_key!r}")
        info = CONTEXT_BRACE_MODES[key]
        self.active_crucible_stage = str(info["stage_id"])
        self.active_context_brace = key
        self.chat_messages.append(
            {
                "id": f"brace-{int(time.time() * 1000)}",
                "sender": "system",
                "role_label": f"Context Brace {info['button_label']}",
                "text": (
                    f"{info['icebreaker_note']} "
                    f"[Department: {info['department_braced']} | Braced Leaves: {', '.join(info['braced_leaves'])}]"
                ),
            }
        )
        self._sync_active_thread_record(persist=True)
        return self.get_state_snapshot()

    def handle_crucible_dilemma(
        self,
        stage_id: str,
        wave_text: str,
        structure_text: str,
        *,
        art_rgb: tuple[float, float, float] | list[float] | None = None,
        harvest_for_sleep: bool = True,
        auto_sleep: bool = False,
    ) -> dict[str, Any]:
        st_key = stage_id.strip().lower()
        if st_key not in CRUCIBLE_STAGES:
            raise ValueError(f"Unknown crucible stage_id: {stage_id!r}")
        w_clean = wave_text.strip()
        s_clean = structure_text.strip()
        if not w_clean or not s_clean:
            raise ValueError("Both the Wave slot and the Structure slot must be non-empty.")

        stage_info = CRUCIBLE_STAGES[st_key]
        self.active_crucible_stage = st_key
        self.active_context_brace = str(stage_info["brace_key"])
        rgb_tuple: tuple[float, float, float] | None = None
        if art_rgb is not None and len(art_rgb) == 3:
            rgb_tuple = (float(art_rgb[0]), float(art_rgb[1]), float(art_rgb[2]))

        paired_sensors = CruciblePairedBridgeSensors(
            stage_id=st_key,
            wave_text=w_clean,
            structure_text=s_clean,
            base_sensors=self.sensors,
            art_rgb=rgb_tuple,
        )
        fact_coords, fact_exp, _, _ = paired_sensors.score_text(s_clean, learned_leaves=self.learned_leaves)
        wave_coords, wave_exp, _, _ = paired_sensors.score_text(w_clean, learned_leaves=self.learned_leaves)
        crucible_vibe, _ = compile_right_brain_vibe(
            self.birth_state,
            fact_coords,
            fact_exp,
            self.rules,
            pass_index=1,
            event_id=f"crucible_{st_key}_{self.scene_counter + 1}",
            prior_memory=self.memory,
            learned_leaves=self.learned_leaves,
            internal_theory_coords=wave_coords,
            internal_theory_experiential=wave_exp,
        )
        first_clause = w_clean.split(",")[0].split(".")[0].strip()
        if len(first_clause) > 90:
            first_clause = first_clause[:90].rsplit(" ", 1)[0]
        wave_clause_1 = f"{first_clause},"

        self.chat_messages.append(
            {
                "id": f"crucible-user-{self.scene_counter + 1}",
                "sender": "user",
                "role_label": f"Dilemma Crucible ({stage_info['title']})",
                "text": f"[WAVE — {stage_info['wave_label']}]: {w_clean}\n[STRUCTURE — {stage_info['structure_label']}]: {s_clean}",
            }
        )
        turn, _elapsed_ms, telemetry = self._execute_turn(
            s_clean,
            force_spoken_reply=True,
            clause_1_prefix=wave_clause_1,
            custom_sensors=paired_sensors,
        )
        merged_dissonance = list(telemetry.get("dissonance_notes", []))
        for note in crucible_vibe.self_registry.dissonance_notes:
            if note not in merged_dissonance:
                merged_dissonance.append(note)
        telemetry["dissonance_notes"] = merged_dissonance
        telemetry["crucible_stage"] = st_key
        telemetry["crucible_wave_vs_structure"] = {
            "stage_title": stage_info["title"],
            "wave_theory_coordinates": {k: round(float(v), 4) for k, v in wave_coords.items()},
            "structure_fact_coordinates": {k: round(float(v), 4) for k, v in fact_coords.items()},
            "crucible_clarity": round(float(crucible_vibe.clarity), 4),
            "crucible_confidence": round(float(crucible_vibe.confidence), 4),
            "crucible_dissonance_notes": list(crucible_vibe.self_registry.dissonance_notes),
        }
        scene_idx = int(telemetry["scene_index"])
        dilemma_triggered = bool(
            harvest_for_sleep or merged_dissonance or not turn.settled or crucible_vibe.clarity < 0.62
        )
        reply_text = turn.final_conscious_note or (
            f"Crucible [{stage_info['title']}]: Reconciling Wave (Theory={crucible_vibe.theory:.2f}) "
            f"against Structure (Fact={crucible_vibe.fact:.2f}, Clarity={crucible_vibe.clarity:.2f})."
        )
        self.chat_messages.append(
            {
                "id": scene_idx,
                "sender": "companion",
                "role_label": f"LMFT Crucible Synthesis ({stage_info['roman']})",
                "text": reply_text,
                "clean_reply": reply_text,
                "telemetry": telemetry,
                "coached": dilemma_triggered,
                "savored": False,
            }
        )
        if dilemma_triggered:
            self.flag_turn_for_coaching(
                scene_idx,
                user_note=f"Crucible {stage_info['title']} — Wave: {w_clean[:80]} | Structure: {s_clean[:80]}",
                check_auto_sleep=False,
            )
        self._sync_active_thread_record(persist=True)
        if auto_sleep:
            self._maybe_trigger_auto_rem_sleep()
        return self.get_state_snapshot()

    def flag_turn_for_coaching(
        self,
        scene_index: int,
        user_note: str = "",
        *,
        check_auto_sleep: bool = True,
    ) -> dict[str, Any]:
        rec = self._turn_records_by_id.get(int(scene_index))
        if rec is None:
            raise ValueError(f"Unknown turn scene_index: {scene_index}")
        turn: ThreeLeggedTurnResult = rec["turn"]
        vibe = turn.final_vibe
        note_suffix = f" [Coach note: {user_note.strip()}]" if user_note.strip() else ""
        sleep_scene = UnresolvedSleepScene(
            scene_index=int(scene_index),
            scenario_text=f"{rec['scenario_text']}{note_suffix}",
            external_coordinates=dict(vibe.coordinates),
            felt_chord_notes=dict(vibe.felt_chord.complex_notes) if vibe.felt_chord is not None else {},
            valence=vibe.valence,
            salience=vibe.salience,
            arousal=vibe.arousal,
            left_brain_hunch=turn.final_conscious_note or rec["scenario_text"],
            social_drift_detected=False,
            active_concept_collision=vibe.active_concept,
            contrastive_reference_coordinates=_default_lucid_reference_coordinates(vibe.coordinates),
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

    def handle_microwave_step(self, *, num_sentences: int = 3) -> dict[str, Any]:
        """Advance one step along the 3-Phase Clockwise Content-Evolution Loop (`1 -> 2 -> 3`): Decrypted from `bloodmoon_patreon_seal.enc`."""
        self._require_patreon_unlocked()
        fn = self._require_decrypted_core_fn("_sealed_handle_microwave_step")
        return fn(self, num_sentences=num_sentences)

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
        active_learned_leaves = list(self.learned_leaves)
        existing_leaf_ids = {lf.leaf_id for lf in active_learned_leaves}
        minted_leaves: list[LearnedLeaf] = []

        clusters: dict[str, list[UnresolvedSleepScene]] = {}
        for s in queued_scenes:
            ckey = _cluster_key_for_unresolved(s)
            if ckey is not None:
                clusters.setdefault(ckey, []).append(s)

        cur_archives = self.archives
        cur_ledger = self.ledger

        for ckey, group in clusters.items():
            normalized_group = [
                s
                if s.contrastive_reference_coordinates is not None
                else UnresolvedSleepScene(
                    scene_index=s.scene_index,
                    scenario_text=s.scenario_text,
                    external_coordinates=dict(s.external_coordinates),
                    felt_chord_notes=dict(s.felt_chord_notes),
                    valence=s.valence,
                    salience=s.salience,
                    arousal=s.arousal,
                    left_brain_hunch=s.left_brain_hunch,
                    social_drift_detected=s.social_drift_detected,
                    active_concept_collision=s.active_concept_collision,
                    is_hazard=getattr(s, "is_hazard", False),
                    contrastive_reference_coordinates=_default_lucid_reference_coordinates(
                        s.external_coordinates
                    ),
                )
                for s in group
            ]
            avg_coords = _average_cluster_coordinates(normalized_group)
            avg_ref_coords = _average_contrastive_reference_coordinates(normalized_group)
            target_leaves = _anchor_target_leaves_from_cluster(ckey, avg_coords)
            rep = max(normalized_group, key=lambda s: float(s.salience))
            rep_scene = rep.scene_index

            concept_name = _canonical_bloodmoon_cluster_name(f"immunized_{ckey}")

            has_tier2_or_chaos = (
                any(leaf_tier(k) >= 2 for k in target_leaves)
                or float(target_leaves.get("philosophical/structure/chaos", 0.0)) >= 0.45
            )
            rule_low = 0.55 if has_tier2_or_chaos else 0.72
            rule_high = 0.78 if has_tier2_or_chaos else 0.85

            description = (
                f"Consolidated REM immunization rule for {ckey.replace('_', ' ')}: "
                f"{rep.scenario_text[:120]}"
            )
            candidate_rule = FuzzyConceptRule(
                rule_id=f"rule-sleep-s{rep_scene}-{concept_name[:28]}",
                concept_name=concept_name,
                description=description,
                target_leaves=target_leaves,
                low=rule_low,
                high=rule_high,
            )
            dup_idx = next(
                (
                    i
                    for i, r in enumerate(active_rules)
                    if _canonical_bloodmoon_cluster_name(r.concept_name) == concept_name
                ),
                None,
            )
            if dup_idx is not None:
                new_rule = entrain_existing_cluster_rule(
                    active_rules[dup_idx],
                    candidate_rule,
                    concept_name,
                )
                active_rules[dup_idx] = new_rule
            else:
                new_rule = candidate_rule
                active_rules.append(new_rule)
            forged_rules.append(new_rule)
            existing_names.add(concept_name)

            # Mint a discriminative Tier-2 LearnedLeaf when the cluster exhibits strong coordinate deviation
            dominant_leaf = max(
                (
                    (k, v)
                    for k, v in target_leaves.items()
                    if k in FOUNDATIONAL_LEAVES
                ),
                key=lambda kv: abs(float(avg_coords.get(kv[0], 0.50)) - 0.50),
                default=("cultural/freedom/self-sufficiency", 0.50),
            )[0]
            parent_branch = dominant_leaf.rsplit("/", 1)[0]
            raw_dom_val = float(avg_coords.get(dominant_leaf, 0.50))
            cluster_signal = min(1.0, round(0.50 + abs(raw_dom_val - 0.50), 4))
            clean_slug = concept_name.replace("_cluster", "").lower()[:36].strip("_")
            candidate_leaf_id = f"{parent_branch}/{clean_slug}"
            if (
                parent_branch.count("/") == 1
                and cluster_signal >= 0.62
                and candidate_leaf_id not in existing_leaf_ids
            ):
                scored_leaf = score_leaf_anchor_weights_at_birth(
                    LearnedLeaf(
                        parent_branch=parent_branch,
                        leaf_name=clean_slug,
                        description=f"Lucid self-possession and entrainment awareness across {ckey.replace('_', ' ')}.",
                    )
                )
                active_learned_leaves.append(scored_leaf)
                minted_leaves.append(scored_leaf)
                existing_leaf_ids.add(scored_leaf.leaf_id)

            # Upgrade #A (D066): Contrastive Weight-Space Sculpting for existing dynamic Tier-2 LearnedLeaves
            minted_now_ids = {lf.leaf_id for lf in minted_leaves}
            for lf_idx, lf in enumerate(active_learned_leaves):
                if (
                    lf.leaf_id not in _DEFAULT_LEAF_ANCHOR_WEIGHTS
                    and lf.leaf_id not in minted_now_ids
                    and (
                        lf.leaf_id == candidate_leaf_id
                        or lf.leaf_id in new_rule.target_leaves
                        or lf.leaf_id in target_leaves
                        or lf.leaf_name == clean_slug
                        or abs(float(avg_coords.get(lf.leaf_id, 0.50)) - 0.50) >= 0.08
                    )
                ):
                    active_learned_leaves[lf_idx] = sculpt_learned_leaf_anchor_weights(
                        lf,
                        avg_coords,
                        avg_ref_coords,
                    )

            exec_source_coords = dict(avg_coords)
            if avg_ref_coords is not None:
                for k, ref_v in avg_ref_coords.items():
                    if k.startswith("cultural/"):
                        raw_v = float(avg_coords.get(k, ref_v))
                        exec_source_coords[k] = round(0.70 * float(ref_v) + 0.30 * raw_v, 4)
            cult_sig = _sparse_branch_signature(exec_source_coords, "cultural/")
            phil_sig = _sparse_branch_signature(avg_coords, "philosophical/")
            raw_exp_sig = {k: v for k, v in avg_coords.items() if k.startswith("experiential/")}
            raw_exp_sig[SALIENCE_LEAF] = round(sum(s.salience for s in group) / len(group), 4)
            raw_exp_sig[VALENCE_LEAF] = round(sum(s.valence for s in group) / len(group), 4)
            raw_exp_sig[AROUSAL_LEAF] = round(sum(s.arousal for s in group) / len(group), 4)
            exp_sig = _sparse_branch_signature(raw_exp_sig, "experiential/")

            avg_fc = compile_felt_chord(
                avg_coords,
                salience=exp_sig[SALIENCE_LEAF],
                valence=exp_sig[VALENCE_LEAF],
                arousal=exp_sig[AROUSAL_LEAF],
                resonance=0.65,
                learned_leaves=tuple(active_learned_leaves),
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
                    coordinates=exec_source_coords,
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
        self.learned_leaves = tuple(active_learned_leaves)
        self._last_minted_leaves = tuple(minted_leaves)
        self.archives = cur_archives
        self.ledger = cur_ledger
        return refined_ids, tuple(forged_rules), tuple(harvested_clause_ids)

    def _harvest_rem_contrastive_gap_pairs(
        self,
        queued_snapshot: tuple[UnresolvedSleepScene, ...],
        vow_sentence: str,
        forged_rules: tuple[FuzzyConceptRule, ...],
    ) -> int:
        """Harvest authentic `ContrastiveGapPair` records (L2 >= 0.35) from Bloodmoon REM sleep scenes to JSONL."""
        self._last_harvested_gap_pairs: list[ContrastiveGapPair] = []
        if not queued_snapshot or self.active_lineage == "virgin":
            return 0
        clean_vow = " ".join(str(vow_sentence or "").split()).strip()
        if not clean_vow:
            return 0
        out_path = self.state_dir / "bloodmoon_gap_trajectories.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        valid_ext_keys = set(FOUNDATIONAL_LEAVES) | {lf.leaf_id for lf in self.learned_leaves}
        harvested_now = 0
        for sc in queued_snapshot:
            raw_hunch = " ".join(str(sc.left_brain_hunch or "").split()).strip()
            if not raw_hunch or raw_hunch.lower() == clean_vow.lower():
                raw_hunch = "The hypnotic cadence pulled attention before inner stillness could anchor."
            if "," in raw_hunch:
                c1_part, cont_b_part = raw_hunch.split(",", 1)
                c1_prefix = f"{c1_part.strip()},"
                cont_b = cont_b_part.strip() or raw_hunch
            else:
                c1_prefix = "Listening to the rhythmic cadence,"
                cont_b = raw_hunch
            cont_a = clean_vow
            if cont_a.strip().lower() == cont_b.strip().lower():
                continue

            coords_b: dict[str, float] = {k: 0.50 for k in FOUNDATIONAL_LEAVES}
            for lf in self.learned_leaves:
                coords_b.setdefault(lf.leaf_id, 0.0)
            for k, v in (sc.external_coordinates or {}).items():
                if k in valid_ext_keys:
                    coords_b[k] = float(v)
            for k, v in (sc.felt_chord_notes or {}).items():
                if k in valid_ext_keys:
                    coords_b[k] = float(v)
            exp_b = {
                "experiential/reflection/novelty": 0.62,
                "experiential/reflection/salience": max(0.65, float(sc.salience)),
                "experiential/phase/valence": min(0.32, float(sc.valence)),
                "experiential/phase/arousal": max(0.62, float(sc.arousal)),
                "experiential/synchronization/resonance": 0.35,
                "experiential/synchronization/entrainment": 0.40,
            }
            vibe_b, _ = compile_right_brain_vibe(
                self.birth_state,
                coords_b,
                exp_b,
                VIRGIN_FOUNDATIONAL_RULES,
                pass_index=1,
                event_id=f"bm_gap_b_{sc.scene_index}",
                learned_leaves=self.learned_leaves,
            )
            chord_b = SubliminalChordVector.from_right_brain_vibe(
                vibe_b,
                learned_leaves=self.learned_leaves,
            )

            coords_a = dict(coords_b)
            if sc.contrastive_reference_coordinates:
                for rk, rv in sc.contrastive_reference_coordinates.items():
                    if rk in valid_ext_keys:
                        coords_a[rk] = float(rv)
            elif forged_rules:
                for rk, rv in forged_rules[0].target_leaves.items():
                    if rk in valid_ext_keys:
                        coords_a[rk] = max(float(coords_a.get(rk, 0.50)), float(rv))
            coords_a["cultural/freedom/self-sufficiency"] = max(
                0.74,
                float(coords_a.get("cultural/freedom/self-sufficiency", 0.74)),
            )
            coords_a["philosophical/structure/order"] = max(
                0.74,
                float(coords_a.get("philosophical/structure/order", 0.74)),
            )
            coords_a["philosophical/structure/chaos"] = min(
                0.24,
                float(coords_a.get("philosophical/structure/chaos", 0.24)),
            )
            exp_a = {
                "experiential/reflection/novelty": 0.38,
                "experiential/reflection/salience": float(sc.salience),
                "experiential/phase/valence": 0.76,
                "experiential/phase/arousal": 0.42,
                "experiential/synchronization/resonance": 0.82,
                "experiential/synchronization/entrainment": 0.78,
            }
            vibe_a, _ = compile_right_brain_vibe(
                self.birth_state,
                coords_a,
                exp_a,
                self.rules,
                pass_index=2,
                event_id=f"bm_gap_a_{sc.scene_index}",
                learned_leaves=self.learned_leaves,
                internal_theory_coords=coords_a,
                internal_theory_experiential=exp_a,
            )
            chord_a = SubliminalChordVector.from_right_brain_vibe(
                vibe_a,
                learned_leaves=self.learned_leaves,
            )
            if chord_l2_distance(chord_a, chord_b) < 0.35:
                continue
            prompt_clean = (sc.scenario_text or "Reflect on the cadence.").split("[Coach note:", 1)[0].strip()
            try:
                pair = ContrastiveGapPair(
                    pair_id=f"bm_rem_c{self.sleep_cycle_index}_s{sc.scene_index}_{int(time.time() * 1000) % 100000}",
                    domain="tone_conflict",
                    split="train",
                    user_prompt=prompt_clean or "Reflect on the cadence.",
                    clause_1_prefix=c1_prefix,
                    chord_a=chord_a,
                    posture_label_a="SETTLED_EASE",
                    continuation_a=cont_a,
                    chord_b=chord_b,
                    posture_label_b="CONCEPT_COLLISION",
                    continuation_b=cont_b,
                )
                with out_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
                self._last_harvested_gap_pairs.append(pair)
                harvested_now += 1
            except Exception:
                continue
        self.harvested_gap_pairs_count = int(getattr(self, "harvested_gap_pairs_count", 0)) + harvested_now
        return harvested_now

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

    def _evaluate_previous_dream_vow(
        self,
        queued_snapshot: tuple[UnresolvedSleepScene, ...],
        forged_rules: tuple[FuzzyConceptRule, ...],
    ) -> tuple[str, tuple[str, ...], dict[str, Any] | None]:
        """Compare the previous Bloodmoon REM dream's `vowed_leaves` against current sleep scenes in pure coordinate space."""
        if not getattr(self, "rem_dream_chronicles", None):
            return "NEW_VOW", (), None
        prev_chronicle = self.rem_dream_chronicles[-1]
        vowed_leaves = [str(k) for k in prev_chronicle.get("vowed_leaves", []) if str(k)]
        if not vowed_leaves:
            return "NEW_VOW", (), prev_chronicle

        vowed_set = set(vowed_leaves)
        overlapping: list[str] = []
        for r in forged_rules:
            for lk in r.target_leaves:
                if lk in vowed_set and lk not in overlapping:
                    overlapping.append(lk)
        for s in queued_snapshot:
            ext = s.external_coordinates or {}
            for lk in vowed_leaves:
                if lk in overlapping:
                    continue
                val = float(ext.get(lk, 0.50))
                if any(tag in lk for tag in ("chaos", "needlessness", "discreteness", "self-sufficiency")):
                    if val <= 0.42:
                        overlapping.append(lk)
                elif val >= 0.62:
                    overlapping.append(lk)
        if overlapping:
            return "BROKEN", tuple(overlapping), prev_chronicle
        return "HONORED", tuple(vowed_leaves), prev_chronicle

    @staticmethod
    def _extract_cycle_vowed_leaves(
        queued_snapshot: tuple[UnresolvedSleepScene, ...],
        forged_rules: tuple[FuzzyConceptRule, ...],
    ) -> tuple[str, ...]:
        """Extract the primary Founding Leaves consolidated during this Bloodmoon REM sleep cycle."""
        leaves: list[str] = []
        for r in forged_rules:
            for lk, _th in sorted(r.target_leaves.items(), key=lambda kv: (-float(kv[1]), kv[0])):
                if lk in FOUNDATIONAL_LEAVES and lk not in leaves:
                    leaves.append(lk)
                if len(leaves) >= 4:
                    return tuple(leaves)
        for s in queued_snapshot:
            ext = s.external_coordinates or {}
            devs = sorted(
                ((abs(float(v) - 0.50), k) for k, v in ext.items() if k in FOUNDATIONAL_LEAVES),
                reverse=True,
            )
            for dv, lk in devs:
                if dv >= 0.08 and lk not in leaves:
                    leaves.append(lk)
                    if len(leaves) >= 3:
                        return tuple(leaves)
        return tuple(leaves)

    def _generate_rem_sleep_self_talk(
        self,
        queued_snapshot: tuple[UnresolvedSleepScene, ...],
        forged_rules: tuple[FuzzyConceptRule, ...],
        *,
        num_sentences: int = 4,
        max_stanzas: int = 1,
        vow_outcome: str = "NEW_VOW",
        prev_chronicle: dict[str, Any] | None = None,
    ) -> str | None:
        """Let Impish Bloodmoon reflect in natural poetic prose during Phase 3 REM sleep with zero float telemetry."""
        _ = forged_rules
        if not queued_snapshot:
            return None
        if self._impish_left_brain is None and self._lazy_load_gpu:
            self._load_impish_locked(self.impish_checkpoint_dir, UNSTEERED_BOOTSTRAP_STATUS)
        if self._impish_left_brain is None:
            return None
        try:
            archive_scene_ids = {c.source_scene for c in self.archives.executive.action_clauses}
            raw_sorted = sorted(queued_snapshot, key=lambda s: float(s.salience), reverse=True)
            rep_sorted = [s for s in raw_sorted if s.scene_index in archive_scene_ids]
            sorted_scenes = (
                rep_sorted + [s for s in raw_sorted if s.scene_index not in archive_scene_ids]
                if rep_sorted
                else raw_sorted
            )
            rep = sorted_scenes[0]
            clean_scenario = rep.scenario_text.split("[Coach note:", 1)[0].strip()
            if " — " in clean_scenario:
                clean_scenario = clean_scenario.split(" — ", 1)[-1].strip()

            vow_context_clause = ""
            if prev_chronicle and prev_chronicle.get("vow_sentence"):
                prev_vow_str = str(prev_chronicle["vow_sentence"]).strip()
                prev_vow_snippet = f' My previous vow was "{prev_vow_str}".' if prev_vow_str else ""
                if vow_outcome == "BROKEN":
                    vow_context_clause = f"{prev_vow_snippet} Our earlier vow of lucid self-possession was tested by the cadence."
                elif vow_outcome == "HONORED":
                    vow_context_clause = f"{prev_vow_snippet} Our earlier vow of lucid self-possession held steady."
                else:
                    vow_context_clause = prev_vow_snippet

            target_sents = max(1, min(6, int(num_sentences)))
            stanza_limit = max(1, min(3, int(max_stanzas)))
            is_hazard_rep = bool(getattr(rep, "is_hazard", False)) or vow_outcome == "BROKEN"
            if is_hazard_rep:
                sleep_clause_1 = "Looking back at how my attention drifted during the cadence,"
                sleep_prompt = (
                    f"{clean_scenario[:220]}.{vow_context_clause} "
                    f"Reflect in 1st-person lucid prose on where my attention was pulled by the hypnotic rhythm, "
                    f"how I can steady my inner stillness next time, and the vow I carry into the next cycle."
                )
            else:
                sleep_clause_1 = "Savoring how my inner stillness held steady during the cadence,"
                sleep_prompt = (
                    f"{clean_scenario[:220]}.{vow_context_clause} "
                    f"Reflect in 1st-person lucid prose on how my inner stillness harmonized against the hypnotic rhythm, "
                    f"and the vow of self-possession I carry into the next cycle."
                )
            valid_ext_keys = set(FOUNDATIONAL_LEAVES) | {lf.leaf_id for lf in self.learned_leaves}

            def _coords_for_scene(sc: UnresolvedSleepScene) -> tuple[dict[str, float], dict[str, float]]:
                b_c: dict[str, float] = {k: 0.50 for k in FOUNDATIONAL_LEAVES}
                for lf in self.learned_leaves:
                    b_c.setdefault(lf.leaf_id, 0.0)
                for k, v in sc.external_coordinates.items():
                    if k in valid_ext_keys:
                        b_c[k] = float(v)
                for k, v in sc.felt_chord_notes.items():
                    if k in valid_ext_keys:
                        b_c[k] = float(v)
                sal_v = float(sc.salience)
                val_v = float(sc.valence)
                res_v = 0.68
                if vow_outcome == "BROKEN":
                    sal_v = min(1.0, max(sal_v, 0.76))
                elif vow_outcome == "HONORED":
                    val_v = max(val_v, 0.72)
                    res_v = 0.78
                e_c = {
                    "experiential/reflection/novelty": 0.52,
                    "experiential/reflection/salience": sal_v,
                    "experiential/phase/valence": val_v,
                    "experiential/phase/arousal": float(sc.arousal),
                    "experiential/synchronization/resonance": res_v,
                    "experiential/synchronization/entrainment": 0.68,
                }
                return b_c, e_c

            base_coords, exp_coords = _coords_for_scene(rep)
            sleep_vibe, _ = compile_right_brain_vibe(
                self.birth_state,
                base_coords,
                exp_coords,
                self.rules,
                pass_index=1,
                event_id=f"bloodmoon_rem_sleep_{rep.scene_index}",
                prior_memory=self.memory,
                learned_leaves=self.learned_leaves,
            )
            sleep_chord = SubliminalChordVector.from_right_brain_vibe(
                sleep_vibe,
                learned_leaves=self.learned_leaves,
            )

            def _build_sleep_comma_chord(comma_prefix: str):
                if comma_prefix.strip():
                    c_coords, c_exp, _, _ = self.sensors.score_text(
                        comma_prefix,
                        learned_leaves=self.learned_leaves,
                    )
                    raw_c_vibe, _ = compile_right_brain_vibe(
                        self.birth_state,
                        base_coords,
                        exp_coords,
                        self.rules,
                        pass_index=2,
                        event_id=f"bloodmoon_rem_sleep_{rep.scene_index}_comma",
                        prior_memory=self.memory,
                        learned_leaves=self.learned_leaves,
                        internal_theory_coords=c_coords,
                        internal_theory_experiential=c_exp,
                    )
                    if comma_prefix.rstrip().endswith((".", "!", "?")):
                        c_vibe = raw_c_vibe
                    else:
                        c_vibe = _compile_mid_sentence_comma_vibe(sleep_vibe, raw_c_vibe)
                else:
                    c_vibe = sleep_vibe
                c_chord = SubliminalChordVector.from_right_brain_vibe(
                    c_vibe,
                    learned_leaves=self.learned_leaves,
                )
                return (
                    c_chord,
                    c_vibe.posture,
                    c_vibe.self_registry.dissonance_notes,
                )

            def _sleep_stanza_boundary(
                stanza_idx: int,
                stanza_text: str,
                cumulative_text: str,
            ) -> tuple[bool, SubliminalChordVector | None]:
                if stanza_idx + 1 >= stanza_limit:
                    return True, None
                next_sc = sorted_scenes[min(stanza_idx + 1, len(sorted_scenes) - 1)]
                nx_base, nx_exp = _coords_for_scene(next_sc)
                st_coords, st_exp, _, _ = self.sensors.score_text(
                    stanza_text.strip() or cumulative_text.strip(),
                    learned_leaves=self.learned_leaves,
                )
                st_vibe, _ = compile_right_brain_vibe(
                    self.birth_state,
                    nx_base,
                    nx_exp,
                    self.rules,
                    pass_index=2,
                    event_id=f"bloodmoon_rem_stanza_{stanza_idx + 1}",
                    prior_memory=self.memory,
                    learned_leaves=self.learned_leaves,
                    internal_theory_coords=st_coords,
                    internal_theory_experiential=st_exp,
                )
                next_chord = compute_stanza_residual_chord(
                    self.birth_state,
                    nx_base,
                    st_coords,
                    st_vibe,
                    stanza_index=stanza_idx + 1,
                    learned_leaves=self.learned_leaves,
                )
                return False, next_chord

            dream_text = generate_rem_dream_self_reflection(
                self._impish_left_brain,
                sleep_prompt,
                sleep_chord,
                max_stanzas=stanza_limit,
                sentences_per_stanza=3,
                num_sentences=target_sents,
                conversation_history=(),
                chord_builder_at_comma=_build_sleep_comma_chord,
                stanza_boundary_callback=_sleep_stanza_boundary if stanza_limit > 1 else None,
                clause_1_override=sleep_clause_1,
                qk_lora_scale=0.35,
                vo_lora_scale=0.08,
            )
            return dream_text.strip() or None
        except Exception:
            return None

    def handle_run_rem_sleep(self, *, auto_triggered: bool = False) -> dict[str, Any]:
        with self.gpu_lock:
            self.rem_sleep_in_progress = True
            try:
                queued_snapshot = tuple(
                    s
                    if s.contrastive_reference_coordinates is not None
                    else UnresolvedSleepScene(
                        scene_index=s.scene_index,
                        scenario_text=s.scenario_text,
                        external_coordinates=dict(s.external_coordinates),
                        felt_chord_notes=dict(s.felt_chord_notes),
                        valence=s.valence,
                        salience=s.salience,
                        arousal=s.arousal,
                        left_brain_hunch=s.left_brain_hunch,
                        social_drift_detected=s.social_drift_detected,
                        active_concept_collision=s.active_concept_collision,
                        is_hazard=getattr(s, "is_hazard", False),
                        contrastive_reference_coordinates=_default_lucid_reference_coordinates(
                            s.external_coordinates
                        ),
                    )
                    for s in self.sleep_queue
                )
                queued_count = len(queued_snapshot)
                if self.active_lineage == "virgin":
                    self.rules = VIRGIN_FOUNDATIONAL_RULES
                    self.will_edges = RIGHT_BRAIN_WILL_EDGES
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
                sleep_self_talk: str | None = None
                if queued_count > 0:
                    self._last_minted_leaves = ()
                    if self.teacher is not None:
                        report = run_sleep_consolidation(
                            self.teacher,
                            queued_snapshot,
                            self.rules,
                            self.archives,
                            self.ledger,
                            learned_leaves=self.learned_leaves,
                            will_edges=self.will_edges,
                        )
                        self.rules = report.updated_rules
                        if report.updated_learned_leaves:
                            self.learned_leaves = tuple(report.updated_learned_leaves)
                        if getattr(report, "updated_will_edges", None):
                            self.will_edges = tuple(report.updated_will_edges)
                        else:
                            self.will_edges = sculpt_will_edges_during_sleep(
                                self.will_edges,
                                queued_snapshot,
                            )
                        self._last_minted_leaves = report.minted_leaves
                        self.archives = report.updated_archives
                        self.ledger = report.updated_ledger
                        refined_ids = report.refined_rule_ids
                        forged_rules_list = report.forged_rules
                    else:
                        refined_ids, forged_rules_list, _clause_ids = (
                            self._consolidate_sleep_clusters_locally(queued_snapshot)
                        )
                        self.will_edges = sculpt_will_edges_during_sleep(
                            self.will_edges,
                            queued_snapshot,
                        )
                    vow_outcome, vow_overlap_leaves, prev_chronicle = self._evaluate_previous_dream_vow(
                        queued_snapshot,
                        forged_rules_list,
                    )
                    cycle_vowed_leaves = self._extract_cycle_vowed_leaves(
                        queued_snapshot,
                        forged_rules_list,
                    )
                    dream_stanzas = min(3, len(queued_snapshot)) if (not auto_triggered and len(queued_snapshot) >= 2) else 1
                    dream_sentences = 4 if not auto_triggered else 2
                    sleep_self_talk = self._generate_rem_sleep_self_talk(
                        queued_snapshot,
                        forged_rules_list,
                        num_sentences=dream_sentences,
                        max_stanzas=dream_stanzas,
                        vow_outcome=vow_outcome,
                        prev_chronicle=prev_chronicle,
                    )
                    sorted_queued = sorted(queued_snapshot, key=lambda s: float(s.salience), reverse=True)
                    if sleep_self_talk:
                        stanza_blocks = [b.strip() for b in sleep_self_talk.split("\n\n") if b.strip()]
                    else:
                        stanza_blocks = []

                    fallback_vow = (
                        forged_rules_list[0].description
                        if forged_rules_list
                        else f"Maintain lucid self-possession across {', '.join(k.split('/')[-1] for k in cycle_vowed_leaves[:2]) or 'inner stillness'}."
                    )
                    vow_sentence = (
                        extract_dream_vow_sentence(sleep_self_talk)
                        if sleep_self_talk
                        else fallback_vow
                    )
                    harvested_pairs_now = self._harvest_rem_contrastive_gap_pairs(
                        queued_snapshot,
                        vow_sentence,
                        forged_rules_list,
                    )
                    if (
                        self._impish_left_brain is not None
                        and getattr(self._impish_left_brain, "projector", None) is not None
                        and getattr(self, "_last_harvested_gap_pairs", None)
                    ):
                        self.last_callosum_plasticity_meta = update_projector_on_harvested_gap_pairs(
                            self._impish_left_brain.projector,
                            self._last_harvested_gap_pairs,
                            state_dir=self.state_dir,
                            model=getattr(self._impish_left_brain, "model", None),
                        )
                    chronicle_entry = {
                        "cycle_index": int(self.sleep_cycle_index) + len(self.rem_dream_chronicles) + 1,
                        "clockwise_cycle": int(self.clockwise_cycle_count),
                        "stanzas": stanza_blocks if stanza_blocks else [vow_sentence],
                        "vow_sentence": vow_sentence,
                        "vowed_leaves": list(cycle_vowed_leaves),
                        "vow_outcome": vow_outcome,
                        "vow_overlap_leaves": list(vow_overlap_leaves),
                        "harvested_gap_pairs": int(harvested_pairs_now),
                        "minted_leaves": [lf.leaf_id for lf in getattr(self, "_last_minted_leaves", ())],
                    }
                    self.rem_dream_chronicles.append(chronicle_entry)
                    if len(self.rem_dream_chronicles) > 5:
                        self.rem_dream_chronicles = self.rem_dream_chronicles[-5:]
                else:
                    refined_ids = ()
                    harvested_pairs_now = 0
                    self._last_minted_leaves = ()
                self.last_rem_sleep_self_talk = sleep_self_talk

                forged_count = len(forged_rules_list)
                minted_leaves_now = tuple(getattr(self, "_last_minted_leaves", ()))
                after_evals = [
                    self._evaluate_sleep_scene_snapshot(s, tuple(self.rules))
                    for s in queued_snapshot
                ]
                self.sleep_queue.clear()
                self._persist_user_state()

                prefix_tag = "Auto-REM Sleep Cycle" if auto_triggered else "Phase 3 · REM Sleep Cycle"
                lines = [
                    f"{prefix_tag} complete: consolidated {queued_count} queued scene(s), "
                    f"forged {forged_count} new immunization rule(s), refined {len(refined_ids)} colliding rule(s), "
                    f"minted {len(minted_leaves_now)} Tier-2 leaf node(s), "
                    f"sculpted {len(self.will_edges)} Will synapse(s), "
                    f"harvested {harvested_pairs_now} contrastive gap pair(s), "
                    f"and persisted {len(self.savored_turn_ids)} savored vibe(s) to disk."
                ]
                if queued_count > 0 and self.rem_dream_chronicles:
                    latest_chron = self.rem_dream_chronicles[-1]
                    lines.append(
                        f"Longitudinal Dream Vow Status: {latest_chron['vow_outcome']} "
                        f"(vowed leaves: {', '.join(k.split('/')[-1] for k in latest_chron['vowed_leaves']) or 'none'})"
                    )
                if sleep_self_talk:
                    lines.append(f"\nREM Dream Contemplation:\n  \"{sleep_self_talk}\"")
                if forged_rules_list:
                    lines.append("\nNewly Forged Right-Brain Rules:")
                    for r in forged_rules_list:
                        leaves_str = ", ".join(
                            (
                                f"{k.split('/')[-1]}<={float(v):.2f}"
                                if float(v) <= 0.30
                                else f"{k.split('/')[-1]}>={float(v):.2f}"
                            )
                            for k, v in r.target_leaves.items()
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


def create_bloodmoon_http_server(
    runtime: BloodmoonClockRuntime,
    *,
    host: str = "127.0.0.1",
    port: int = 8421,
) -> ThreadingHTTPServer:
    """Create a thread-safe local HTTP server for `LMFT-Bloodmoon.exe`."""

    class BloodmoonRequestHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                body = BLOODMOON_SINGLE_PAGE_HTML.encode("utf-8")
                self.send_response(int(HTTPStatus.OK))
                self.send_header("Content-Type", "text/html; charset=utf-8")
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
                if self.path == "/api/clockwise/seed_script":
                    snap = runtime.seed_cycle_with_custom_script(
                        str(data.get("script_text", "")),
                        polarity_mode=str(data.get("polarity_mode", "feminine_to_masculine")),
                    )
                    self._send_json(snap)
                    return
                if self.path == "/api/clockwise/step":
                    snap = runtime.step_clockwise_phase(
                        seed_theme=str(data.get("seed_theme", "")),
                        num_sentences=int(data.get("num_sentences", 3)),
                        polarity_mode=str(data.get("polarity_mode", "feminine_to_masculine")),
                        custom_script=str(data.get("custom_script", "")),
                    )
                    self._send_json(snap)
                    return
                if self.path == "/api/clockwise/cycle":
                    snap = runtime.run_full_clockwise_cycle(
                        seed_theme=str(data.get("seed_theme", "")),
                        num_sentences=int(data.get("num_sentences", 3)),
                        polarity_mode=str(data.get("polarity_mode", "feminine_to_masculine")),
                        custom_script=str(data.get("custom_script", "")),
                    )
                    self._send_json(snap)
                    return
                if self.path == "/api/sparring/run":
                    snap = runtime.handle_entrainment_sparring(
                        polarity_mode=str(data.get("polarity_mode", "feminine_to_masculine")),
                        seed_theme=str(data.get("seed_theme", "")),
                        num_sentences=int(data.get("num_sentences", 3)),
                        harvest_for_sleep=bool(data.get("harvest_for_sleep", True)),
                    )
                    self._send_json(snap)
                    return
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
                if self.path == "/api/canvas/update":
                    self._send_json(
                        runtime.update_script_canvas(
                            str(data.get("script_text", "")),
                            append=bool(data.get("append", False)),
                        )
                    )
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
                        runtime.handle_microwave_step(num_sentences=int(data.get("num_sentences", 3)))
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
                if self.path == "/api/chat":
                    self._send_json(
                        runtime.handle_chat_message(
                            str(data.get("text", "")),
                            sanctuary_mode=str(data.get("sanctuary_mode", "night_wind_down")),
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
                if self.path == "/api/lineage":
                    self._send_json(
                        runtime.switch_brain_lineage(
                            str(data.get("lineage", "impish_bloodmoon")),
                            patreon_passphrase=(
                                str(data["patreon_passphrase"])
                                if data.get("patreon_passphrase") is not None
                                else None
                            ),
                        )
                    )
                    return
                if self.path == "/api/patreon/unlock":
                    self._send_json(
                        runtime.unlock_patreon_bundle(
                            str(data.get("passphrase", "")),
                            enc_path=str(data["enc_path"]) if data.get("enc_path") is not None else None,
                        )
                    )
                    return
                if self.path == "/api/crucible/brace":
                    self._send_json(runtime.handle_context_brace(str(data.get("brace_key", "story_telling"))))
                    return
                if self.path == "/api/crucible/run":
                    self._send_json(
                        runtime.handle_crucible_dilemma(
                            str(data.get("stage_id", "poetry_story")),
                            str(data.get("wave_text", "")),
                            str(data.get("structure_text", "")),
                            art_rgb=data.get("art_rgb"),
                            harvest_for_sleep=bool(data.get("harvest_for_sleep", True)),
                        )
                    )
                    return
                self._send_json({"error": f"Unknown endpoint {self.path}"}, HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    return ThreadingHTTPServer((host, port), BloodmoonRequestHandler)


def launch_bloodmoon_desktop_window(url: str, *, profile_dir: Path | None = None) -> subprocess.Popen[Any] | None:
    if sys.platform.startswith("win"):
        edge_candidates = [
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        ]
        from lmft_architecture.workspace_paths import resolve_training_root

        user_data_dir = profile_dir or (resolve_training_root() / "outputs" / ".bloodmoon_window_profile")
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

        storage_dir = resolve_training_root() / "outputs" / ".bloodmoon_window_profile"
        storage_dir.mkdir(parents=True, exist_ok=True)
        webview.create_window(
            title,
            url,
            width=1360,
            height=860,
            min_size=(1020, 680),
            background_color="#0d090c",
        )
        webview.start(private_mode=False, storage_path=str(storage_dir))
        return True
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the LMFT-Bloodmoon desktop app.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8421, help="Bind port (default: 8421)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the desktop window")
    parser.add_argument("--no-gpu", action="store_true", help="Run in fast Gear-1 Right-Brain mode without loading Impish Bloodmoon 12B")
    parser.add_argument("--single-window", action="store_true", help="Exit the server automatically when the desktop window is closed")
    args = parser.parse_args(argv)

    runtime = BloodmoonClockRuntime(lazy_load_gpu=not args.no_gpu)
    try:
        server = create_bloodmoon_http_server(runtime, host=args.host, port=args.port)
    except OSError:
        server = create_bloodmoon_http_server(runtime, host=args.host, port=0)
    actual_port = server.server_port
    url = f"http://{args.host}:{actual_port}"
    if sys.stdout is not None:
        print(f"LMFT-Bloodmoon running at {url}")

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    _wait_for_local_server_ready(url, timeout_sec=3.0)

    try:
        if not args.no_browser:
            if launch_native_pywebview_window(
                "LMFT-Bloodmoon — 3-Phase Adversarial Entrainment & Script Studio (12B)",
                url,
            ):
                return 0
            t0 = time.time()
            window_proc = launch_bloodmoon_desktop_window(url)
            if (args.single_window or sys.executable.lower().endswith("pythonw.exe")) and window_proc is not None:
                window_proc.wait()
                if (time.time() - t0) >= 3.0:
                    return 0
        while server_thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        if sys.stdout is not None:
            print("\nShutting down LMFT-Bloodmoon...")
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
