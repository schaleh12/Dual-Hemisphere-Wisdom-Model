"""Offline 9B Sleep-Mode Consolidation for Unresolved Tensions & Branch Archives.

Plain-English Purpose:
When the system finishes an active wake cycle (Epoch 1), some scenes may remain in
`UNRESOLVED_TENSION` (`settled = False`) inside `contemplation_ledger.json` because:
1. No rule in the current notebook covers that structural/social/emotional pattern yet
   (for example, non-violent exclusionary gatekeeping or restorative barrier fracture).
2. An existing broad Tier-1 rule (such as `solitary_autarky`) collided with high-chaos
   or communal-sharing scenes because its `target_leaves` were underspecified.

During Sleep Mode (offline, with zero runtime pressure), the local 9B Teacher
(`Qwen3.5-9B-Q4_K_M.gguf`) audits the unresolved Epoch 1 records and:
- Refines any over-broad Tier-1 rule that misfired during Epoch 1 collisions.
- Forges general, domain-independent `FuzzyConceptRule` entries and 3-branch archive
  clauses (`Executive`, `Legislative`, `Judicial`) anchored to the observed Epoch 1
  training coordinates—strictly rejecting any story-specific vocabulary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import re

from .branch_archives import (
    EXECUTIVE_BRANCH,
    JUDICIAL_BRANCH,
    LEGISLATIVE_BRANCH,
    BranchClauseEntry,
    ThreeBranchArchives,
    _derive_concept_tags,
)
from .contemplation_ledger import ContemplationLedger, save_contemplation_ledger
from .local_models import strict_json
from .model_assets import ModelError
from .philosophical_abstraction_matrix import PHILOSOPHICAL_SHADOW_LEAVES
from .reflection_classifier import (
    AROUSAL_LEAF,
    FOUNDATIONAL_LEAVES,
    NOVELTY_LEAF,
    RESONANCE_LEAF,
    RETENTION_LEAF,
    SALIENCE_LEAF,
    VALENCE_LEAF,
    FuzzyConceptRule,
    LearnedLeaf,
    is_valid_leaf_id,
    leaf_tier,
    save_ruleset,
)

def assert_domain_general_text(
    text: str,
    field_name: str = "text",
    banned_substrings: Sequence[str] = (),
) -> str:
    """Verify that a Teacher-proposed name, description, or clause contains zero banned story nouns."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    lower = text.lower()
    for banned in banned_substrings:
        if banned.lower() in lower:
            raise ValueError(f"{field_name} contains story-specific term {banned!r}: {text!r}")
    return text.strip()


@dataclass(frozen=True)
class UnresolvedSleepScene:
    """One unresolved wake-cycle scene queued for offline 9B Teacher sleep review."""

    scene_index: int
    scenario_text: str
    external_coordinates: dict[str, float]
    felt_chord_notes: dict[str, float]
    valence: float
    salience: float
    arousal: float
    left_brain_hunch: str
    social_drift_detected: bool = False
    active_concept_collision: str | None = None
    is_hazard: bool = False
    contrastive_reference_coordinates: dict[str, float] | None = None
    game_review_annotation: str | None = None


@dataclass(frozen=True)
class SleepConsolidationReport:
    """Summary of offline 9B Teacher Sleep-Mode rule refinements and newly forged rules."""

    refined_rule_ids: tuple[str, ...]
    forged_rules: tuple[FuzzyConceptRule, ...]
    harvested_clause_ids: tuple[str, ...]
    resolved_scene_indices: tuple[int, ...]
    updated_rules: tuple[FuzzyConceptRule, ...]
    updated_archives: ThreeBranchArchives
    updated_ledger: ContemplationLedger

    def to_dict(self) -> dict:
        return {
            "refined_rule_ids": list(self.refined_rule_ids),
            "forged_rules": [r.to_dict() for r in self.forged_rules],
            "harvested_clause_ids": list(self.harvested_clause_ids),
            "resolved_scene_indices": list(self.resolved_scene_indices),
            "total_rules_after_sleep": len(self.updated_rules),
            "remaining_open_uncertainties": len(self.updated_ledger.active_uncertainties),
        }


def _cluster_key_for_unresolved(scene: UnresolvedSleepScene) -> str | None:
    """Group any unresolved wake-cycle scene by its coordinate profile across departments.

    Note: If a scene was only held back by historical `social_drift_detected` while its
    static fairness coordinate is moderate or high (`> 0.40`), we do NOT forge a static
    coordinate rule from it (since a static rule on moderate fairness inside a sealed
    room would falsely override the `social_drift` veto or fire on normal peaceful rooms).
    """
    coords = scene.external_coordinates
    if scene.social_drift_detected and float(coords.get("cultural/justice/fairness", 0.50)) > 0.40:
        return None

    cult = {
        k: float(v)
        for k, v in coords.items()
        if k.startswith("cultural/") and is_valid_leaf_id(k)
    }
    phil = {
        k: float(v)
        for k, v in coords.items()
        if k.startswith("philosophical/") and is_valid_leaf_id(k)
    }

    low_cult = sorted(
        ((k, v) for k, v in cult.items() if k in FOUNDATIONAL_LEAVES and v <= 0.35),
        key=lambda kv: (kv[1], kv[0]),
    )
    high_cult = sorted(
        ((k, v) for k, v in cult.items() if v >= 0.45),
        key=lambda kv: (-kv[1], kv[0]),
    )
    chaos = float(phil.get("philosophical/structure/chaos", 0.0))

    if low_cult and high_cult:
        low_extremity = 0.50 - low_cult[0][1]
        high_extremity = high_cult[0][1] - 0.50
        if low_extremity >= high_extremity:
            cult_tag = f"low_{low_cult[0][0].split('/')[-1].replace('-', '_')}"
        else:
            cult_tag = f"high_{high_cult[0][0].split('/')[-1].replace('-', '_')}"
    elif low_cult:
        cult_tag = f"low_{low_cult[0][0].split('/')[-1].replace('-', '_')}"
    elif high_cult:
        cult_tag = f"high_{high_cult[0][0].split('/')[-1].replace('-', '_')}"
    else:
        cult_tag = "moderate_cultural"

    if chaos >= 0.42:
        phil_tag = "chaos"
    else:
        high_phil = sorted(
            ((k, v) for k, v in phil.items() if k != "philosophical/being/existence" and v >= 0.45),
            key=lambda kv: (-kv[1], kv[0]),
        )
        if not high_phil:
            high_phil = sorted(phil.items(), key=lambda kv: (-kv[1], kv[0]))
        phil_tag = high_phil[0][0].split("/")[-1].replace("-", "_") if high_phil else "structure"

    return f"{cult_tag}_{phil_tag}_cluster"


def _average_cluster_coordinates(scenes: Sequence[UnresolvedSleepScene]) -> dict[str, float]:
    all_keys: set[str] = set()
    for s in scenes:
        all_keys.update(s.external_coordinates.keys())
        all_keys.update(s.felt_chord_notes.keys())
    avg: dict[str, float] = {}
    for k in sorted(all_keys):
        vals = [
            float(s.external_coordinates.get(k, s.felt_chord_notes.get(k, 0.0)))
            for s in scenes
        ]
        avg[k] = round(sum(vals) / len(vals), 4)
    return avg


def _average_contrastive_reference_coordinates(
    scenes: Sequence[UnresolvedSleepScene],
) -> dict[str, float] | None:
    """Average `contrastive_reference_coordinates` across scenes in a cluster when present."""
    ref_scenes = [s for s in scenes if s.contrastive_reference_coordinates]
    if not ref_scenes:
        return None
    all_keys: set[str] = set()
    for s in ref_scenes:
        assert s.contrastive_reference_coordinates is not None
        all_keys.update(s.contrastive_reference_coordinates.keys())
    avg_ref: dict[str, float] = {}
    for k in sorted(all_keys):
        vals = [
            float((s.contrastive_reference_coordinates or {}).get(k, 0.50))
            for s in ref_scenes
        ]
        avg_ref[k] = round(sum(vals) / len(vals), 4)
    return avg_ref


def _raw_rule_min_similarity(
    target_leaves: Mapping[str, float],
    coords: Mapping[str, float],
) -> float:
    """Compute the raw `min_similarity` inside `FuzzyConceptRule.evaluate` before `[low, high]` trapezoidal scaling."""
    if not target_leaves:
        return 0.0
    sims: list[float] = []
    for leaf_id, target_val in target_leaves.items():
        actual = float(coords.get(leaf_id, 0.50))
        t_val = float(target_val)
        if t_val >= 0.65 and actual >= t_val:
            sim = 1.0
        elif t_val <= 0.30 and actual <= t_val:
            sim = 1.0
        else:
            sim = 1.0 - abs(actual - t_val)
        sims.append(sim)
    return min(sims)


def normalize_contrastive_target_leaves_and_thresholds(
    cluster_key: str,
    avg_coords: Mapping[str, float],
    avg_ref_coords: Mapping[str, float] | None = None,
) -> tuple[dict[str, float], float, float]:
    """Select `target_leaves` and normalize `[low, high]` activation thresholds using contrastive Game Review coordinates.

    - When `avg_ref_coords` is `None`, preserves the standard `_anchor_target_leaves_from_cluster` behavior.
    - When `avg_ref_coords` is present (e.g., comparing a played Blunder `??` against the missed Best Move `!`,
      or a Brilliancy `!!` against an inferior alternative from the exact same board position):
      1. Selects the 2–3 leaves with the largest contrastive divergence `|avg_coords[k] - avg_ref_coords[k]|`.
      2. Sets each selected leaf's target value to `avg_coords[k]` so `min_similarity` peaks on `avg_coords`
         and drops by `max_delta` on `avg_ref_coords`.
      3. Normalizes `[rule_low, rule_high]` directly inside `[sim_ref, sim_target]` so the forged rule
         fires at `~1.0` on `avg_coords` and `0.0` on `avg_ref_coords`.
    """
    base_targets = _anchor_target_leaves_from_cluster(cluster_key, avg_coords)
    has_tier2_or_chaos = (
        any(leaf_tier(k) >= 2 for k in base_targets)
        or float(base_targets.get("philosophical/structure/chaos", 0.0)) >= 0.45
    )
    default_low = 0.55 if has_tier2_or_chaos else 0.72
    default_high = 0.78 if has_tier2_or_chaos else 0.85

    if not avg_ref_coords:
        return base_targets, default_low, default_high

    transient_exp = {
        SALIENCE_LEAF,
        VALENCE_LEAF,
        AROUSAL_LEAF,
        RESONANCE_LEAF,
        RETENTION_LEAF,
        NOVELTY_LEAF,
    }
    shadow_leaf_set = {leaf.leaf_id for leaf in PHILOSOPHICAL_SHADOW_LEAVES} | {
        "philosophical/being/tyrannical_volition",
        "philosophical/being/nihilistic_withdrawal",
        "philosophical/structure/dogmatic_rigidity",
        "philosophical/structure/fragmentation",
        "philosophical/persistence/stagnation",
        "philosophical/persistence/amnesia",
    }
    virtue_leaf_set = {
        "philosophical/being/existence",
        "philosophical/structure/order",
        "philosophical/persistence/continuity",
        "cultural/efficiency/wastelessness",
        "cultural/freedom/self-sufficiency",
    }
    has_wastelessness = "cultural/efficiency/wastelessness" in avg_coords
    wastelessness_val = float(avg_coords.get("cultural/efficiency/wastelessness", 0.50))
    explicit_hazard = has_wastelessness and wastelessness_val <= 0.35
    explicit_coached = has_wastelessness and wastelessness_val >= 0.65

    divergent: list[tuple[float, str, float, float]] = []
    for k, raw_v in avg_coords.items():
        if not is_valid_leaf_id(k) or k in transient_exp:
            continue
        if not k.startswith(("cultural/", "philosophical/", "experiential/")):
            continue
        v_act = float(raw_v)
        v_ref = float(avg_ref_coords.get(k, v_act))
        if explicit_coached:
            # A `coached_` rule must never target shadow pathologies or low virtue leaves
            if k in shadow_leaf_set:
                continue
            if k in virtue_leaf_set and (v_act < 0.55 or v_act < v_ref):
                continue
            if k == "cultural/freedom/needlessness" and v_act > 0.20:
                continue
            if k == "philosophical/structure/chaos" and v_act > 0.45:
                continue
        elif explicit_hazard:
            # A `hazard_` rule must never target high virtue leaves, low needlessness, or low shadow/chaos leaves
            if k in virtue_leaf_set and (v_act >= 0.58 or v_act >= v_ref):
                continue
            if k in shadow_leaf_set and (v_act < 0.50 or v_act <= v_ref):
                continue
            if k == "philosophical/structure/chaos" and (v_act < 0.50 or v_act <= v_ref):
                continue
            if k == "cultural/freedom/needlessness" and (v_act <= 0.30 or v_act < v_ref):
                continue
        abs_diff = abs(v_act - v_ref)
        if abs_diff >= 0.03:
            divergent.append((abs_diff, k, v_act, v_ref))

    divergent.sort(key=lambda item: (-item[0], item[1]))
    target_leaves: dict[str, float] = {}
    for idx, (abs_diff, k, v_act, v_ref) in enumerate(divergent):
        if idx >= 3 or (idx == 2 and abs_diff < 0.08):
            break
        t_val = round(min(0.95, max(0.05, v_act)), 2)
        if v_act < v_ref and t_val >= 0.65:
            t_val = 0.64
        elif v_act > v_ref and t_val <= 0.30:
            t_val = 0.31
        target_leaves[k] = t_val

    if len(target_leaves) < 2:
        for k, t_val in base_targets.items():
            if k not in target_leaves:
                target_leaves[k] = t_val
                if len(target_leaves) >= 2:
                    break

    sim_target = _raw_rule_min_similarity(target_leaves, avg_coords)
    sim_ref = _raw_rule_min_similarity(target_leaves, avg_ref_coords)
    gap = sim_target - sim_ref
    if gap >= 0.04:
        rule_low = round(max(0.35, min(0.92, sim_ref + 0.25 * gap)), 4)
        rule_high = round(max(rule_low + 0.02, min(0.98, sim_target - 0.15 * gap)), 4)
        return target_leaves, rule_low, rule_high

    return target_leaves, default_low, default_high


def _build_sleep_teacher_forge_request(
    cluster_key: str,
    scenes: Sequence[UnresolvedSleepScene],
    avg_coords: Mapping[str, float],
    generation: Mapping[str, object],
    banned_substrings: Sequence[str] = (),
) -> dict:
    """Build a strict JSON request for the 9B Teacher to name and describe an unresolved pattern."""
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "concept_name",
            "description",
            "executive_clause_text",
            "legislative_clause_text",
            "judicial_clause_text",
        ],
        "properties": {
            "concept_name": {"type": "string", "minLength": 5, "maxLength": 48},
            "description": {"type": "string", "minLength": 20, "maxLength": 220},
            "executive_clause_text": {"type": "string", "minLength": 20, "maxLength": 200},
            "legislative_clause_text": {"type": "string", "minLength": 20, "maxLength": 200},
            "judicial_clause_text": {"type": "string", "minLength": 20, "maxLength": 200},
        },
    }

    salient_coords = {
        k: v
        for k, v in avg_coords.items()
        if is_valid_leaf_id(k) and (v >= 0.45 or k == "cultural/justice/fairness")
    }
    hunches: list[str] = []
    review_notes: list[str] = []
    for s in scenes:
        if s.left_brain_hunch:
            masked = s.left_brain_hunch
            for b in banned_substrings:
                masked = re.sub(re.escape(b), "resource", masked, flags=re.IGNORECASE)
            hunches.append(masked)
        if s.game_review_annotation:
            review_notes.append(s.game_review_annotation)

    avg_ref_coords = _average_contrastive_reference_coordinates(scenes)
    contrastive_delta: dict[str, float] = {}
    if avg_ref_coords:
        for k, v in avg_coords.items():
            if is_valid_leaf_id(k) and k in avg_ref_coords:
                diff = round(float(v) - float(avg_ref_coords[k]), 4)
                if abs(diff) >= 0.03:
                    contrastive_delta[k] = diff

    system = (
        "You are the offline Sleep-Mode Teacher consolidating unresolved experiences into general, "
        "domain-independent conceptual definitions.\n"
        "CRITICAL RULES:\n"
        "1. Write in clear, plain everyday English.\n"
        "2. Your definitions and clauses MUST be completely general across human settings. "
        "NEVER use setting-specific nouns, concrete props, animals, or local weather from the notes; "
        "describe the underlying social, emotional, and structural pattern.\n"
        "3. `concept_name` must be lower_snake_case.\n"
        "4. Return JSON only. /no_think"
    )
    user_payload: dict[str, object] = {
        "pattern_id": cluster_key,
        "observed_coordinate_profile": salient_coords,
        "unresolved_reflection_notes": hunches,
        "instructions": (
            "Propose a general `concept_name`, a one-sentence `description`, and three general "
            "branch clauses (`executive_clause_text` for social access/duty, `legislative_clause_text` "
            "for the emotional atmosphere, and `judicial_clause_text` for the structural reality) "
            "using zero scene-specific nouns."
        ),
    }
    if contrastive_delta:
        user_payload["contrastive_coordinate_delta_vs_best_alternative"] = contrastive_delta
    if review_notes:
        user_payload["objective_referee_annotations"] = review_notes
    import json

    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, indent=2)},
        ],
        "temperature": 0,
        "seed": int(generation.get("seed", 42)),
        "max_tokens": 380,
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object", "schema": schema},
    }


def _anchor_target_leaves_from_cluster(
    cluster_key: str,
    avg_coords: Mapping[str, float],
) -> dict[str, float]:
    """Anchor a Sleep-Mode rule's target_leaves from the cluster's genuine extreme/salient leaves."""
    _ = cluster_key
    target_leaves: dict[str, float] = {}

    cult = {
        k: float(v)
        for k, v in avg_coords.items()
        if k.startswith("cultural/") and is_valid_leaf_id(k)
    }
    phil = {
        k: float(v)
        for k, v in avg_coords.items()
        if k.startswith("philosophical/") and is_valid_leaf_id(k)
    }
    exp_complex = {
        k: float(v)
        for k, v in avg_coords.items()
        if k.startswith("experiential/")
        and is_valid_leaf_id(k)
        and k not in (SALIENCE_LEAF, VALENCE_LEAF, AROUSAL_LEAF, RESONANCE_LEAF, RETENTION_LEAF, NOVELTY_LEAF)
    }

    # 1. Cultural anchor: select the leaf with the strongest extremity from neutral (0.50)
    low_cult = sorted(
        ((k, v) for k, v in cult.items() if k in FOUNDATIONAL_LEAVES and v <= 0.35),
        key=lambda kv: (kv[1], kv[0]),
    )
    high_cult = sorted(
        ((k, v) for k, v in cult.items() if v >= 0.50),
        key=lambda kv: (-kv[1], kv[0]),
    )
    if low_cult and high_cult:
        if (0.50 - low_cult[0][1]) >= (high_cult[0][1] - 0.50):
            k_low, v_low = low_cult[0]
            target_leaves[k_low] = round(min(0.20, max(0.14, v_low + 0.04)), 2)
        else:
            k_high, v_high = high_cult[0]
            target_leaves[k_high] = round(min(0.78, v_high), 2)
    elif low_cult:
        k_low, v_low = low_cult[0]
        target_leaves[k_low] = round(min(0.20, max(0.14, v_low + 0.04)), 2)
    elif high_cult:
        k_high, v_high = high_cult[0]
        target_leaves[k_high] = round(min(0.78, v_high), 2)

    # 2. Experiential complex feeling anchor (if an active complex feeling drawer >= 0.50 exists)
    top_exp = sorted(
        ((k, v) for k, v in exp_complex.items() if v >= 0.50),
        key=lambda kv: (-kv[1], kv[0]),
    )
    has_exp_anchor = False
    if top_exp:
        k_exp, v_exp = top_exp[0]
        target_leaves[k_exp] = round(min(0.60, v_exp), 2)
        has_exp_anchor = True

    # 3. Philosophical structural anchor(s)
    chaos = float(phil.get("philosophical/structure/chaos", 0.0))
    if chaos >= 0.45:
        target_leaves["philosophical/structure/chaos"] = round(min(0.65, chaos), 2)
    else:
        structural_phil = sorted(
            ((k, v) for k, v in phil.items() if k != "philosophical/being/existence" and v >= 0.55),
            key=lambda kv: (-kv[1], kv[0]),
        )
        if not structural_phil:
            structural_phil = sorted(
                ((k, v) for k, v in phil.items() if v >= 0.55),
                key=lambda kv: (-kv[1], kv[0]),
            )
        if structural_phil:
            k1, v1 = structural_phil[0]
            target_leaves[k1] = round(min(0.75, v1), 2)
            if not has_exp_anchor and len(structural_phil) >= 2 and structural_phil[1][1] >= 0.65:
                k2, v2 = structural_phil[1]
                target_leaves[k2] = round(min(0.70, v2), 2)

    if len(target_leaves) < 2:
        for k, v in sorted(avg_coords.items(), key=lambda kv: (-kv[1], kv[0])):
            if is_valid_leaf_id(k) and k not in target_leaves and v >= 0.45:
                target_leaves[k] = round(min(0.75, v), 2)
                if len(target_leaves) >= 2:
                    break

    return target_leaves


def _rule_collided_on_unresolved(
    rule: FuzzyConceptRule,
    unresolved_scenes: Sequence[UnresolvedSleepScene],
) -> bool:
    """Return True if `rule` was flagged as colliding or fired strongly on an unresolved scene."""
    for s in unresolved_scenes:
        if s.active_concept_collision == rule.concept_name:
            return True
        merged = {k: 0.50 for k in FOUNDATIONAL_LEAVES}
        merged.update(s.external_coordinates)
        merged.update(s.felt_chord_notes)
        for req_leaf in rule.target_leaves:
            merged.setdefault(req_leaf, 0.0)
        if rule.evaluate(merged) >= 0.50:
            return True
    return False


def refine_colliding_rules_during_sleep(
    rules: Sequence[FuzzyConceptRule],
    unresolved_scenes: Sequence[UnresolvedSleepScene],
) -> tuple[tuple[FuzzyConceptRule, ...], tuple[str, ...]]:
    """Tighten any over-broad Tier-1 rule that collided or misfired on unresolved wake-cycle scenes.

    Instead of checking for a specific rule name, this function inspects every rule in `rules`:
    if a Tier-1 rule collided on an unresolved scene (either via `active_concept_collision` or
    by firing `>= 0.50` on an unsettled scene), its moderate positive leaf targets are tightened
    and, if it specifies steady continuity/order without bounding structural chaos, a low-chaos
    bound (`philosophical/structure/chaos = 0.10`) is added so it no longer misfires on fractures.
    """
    if not unresolved_scenes:
        return tuple(rules), ()

    updated: list[FuzzyConceptRule] = []
    refined_ids: list[str] = []
    for r in rules:
        if r.tier == 1 and _rule_collided_on_unresolved(r, unresolved_scenes):
            new_targets = dict(r.target_leaves)
            for k, val in list(new_targets.items()):
                fval = float(val)
                if 0.40 <= fval <= 0.65:
                    new_targets[k] = 0.75

            has_steady_axis = (
                float(new_targets.get("philosophical/persistence/continuity", 0.0)) >= 0.60
                or float(new_targets.get("philosophical/structure/order", 0.0)) >= 0.65
            )
            if has_steady_axis and "philosophical/structure/chaos" not in new_targets:
                new_targets["philosophical/structure/chaos"] = 0.10

            refined_rule = FuzzyConceptRule(
                rule_id=r.rule_id,
                concept_name=r.concept_name,
                description=r.description,
                target_leaves=new_targets,
                low=max(r.low, 0.72),
                high=max(r.high, 0.88),
            )
            updated.append(refined_rule)
            refined_ids.append(r.rule_id)
        else:
            updated.append(r)
    return tuple(updated), tuple(refined_ids)


def run_sleep_consolidation(
    *,
    unresolved_scenes: Sequence[UnresolvedSleepScene],
    rules: Sequence[FuzzyConceptRule],
    learned_leaves: Sequence[LearnedLeaf],
    archives: ThreeBranchArchives,
    ledger: ContemplationLedger,
    teacher,
    teacher_role: str = "teacher",
    generation: Mapping[str, object] | None = None,
    ruleset_out_path: str | Path | None = None,
    archives_out_dir: str | Path | None = None,
    ledger_out_path: str | Path | None = None,
    banned_substrings: Sequence[str] = (),
) -> SleepConsolidationReport:
    """Execute an offline 9B Teacher Sleep-Mode consolidation pass over unresolved Epoch 1 scenes."""
    gen_cfg = dict(generation) if generation is not None else {"seed": 42}

    # 1. Refine over-broad Tier-1 rules that collided on Epoch 1 unresolved scenes
    refined_rules, refined_ids = refine_colliding_rules_during_sleep(rules, unresolved_scenes)
    active_rules: list[FuzzyConceptRule] = list(refined_rules)
    existing_names = {r.concept_name for r in active_rules}

    # 2. Group unresolved scenes into distinct coordinate clusters (keeping hazard vs. non-hazard separate)
    clusters: dict[tuple[str, bool], list[UnresolvedSleepScene]] = {}
    for s in unresolved_scenes:
        ckey = _cluster_key_for_unresolved(s)
        if ckey is not None:
            clusters.setdefault((ckey, bool(s.is_hazard)), []).append(s)

    forged_rules: list[FuzzyConceptRule] = []
    harvested_clause_ids: list[str] = []
    resolved_scene_indices: list[int] = []
    cur_archives = archives

    for (ckey, is_hazard_group), group in clusters.items():
        avg_coords = _average_cluster_coordinates(group)
        avg_ref_coords = _average_contrastive_reference_coordinates(group)
        req = _build_sleep_teacher_forge_request(
            ckey,
            group,
            avg_coords,
            gen_cfg,
            banned_substrings=banned_substrings,
        )

        def _validate_teacher_sleep_json(raw: str) -> dict[str, str]:
            try:
                cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                parsed = strict_json(cleaned)
                raw_name = str(parsed["concept_name"]).strip().lower()
                raw_name = re.sub(r"[^a-z0-9_]+", "_", raw_name).strip("_")
                if len(raw_name) < 4:
                    raise ValueError("concept_name too short")
                assert_domain_general_text(raw_name, "concept_name", banned_substrings)
                desc = assert_domain_general_text(str(parsed["description"]), "description", banned_substrings)
                exec_txt = assert_domain_general_text(
                    str(parsed["executive_clause_text"]),
                    "executive_clause_text",
                    banned_substrings,
                )
                leg_txt = assert_domain_general_text(
                    str(parsed["legislative_clause_text"]),
                    "legislative_clause_text",
                    banned_substrings,
                )
                jud_txt = assert_domain_general_text(
                    str(parsed["judicial_clause_text"]),
                    "judicial_clause_text",
                    banned_substrings,
                )
                return {
                    "concept_name": raw_name,
                    "description": desc,
                    "executive_clause_text": exec_txt,
                    "legislative_clause_text": leg_txt,
                    "judicial_clause_text": jud_txt,
                }
            except Exception as exc:
                raise ModelError(f"Sleep Teacher validation error on {ckey}: {exc}") from None

        proposal, _trace = teacher.generate(
            teacher_role,
            req,
            f"sleep-consolidate-{ckey}-v1",
            _validate_teacher_sleep_json,
        )

        target_leaves, rule_low, rule_high = normalize_contrastive_target_leaves_and_thresholds(
            ckey,
            avg_coords,
            avg_ref_coords,
        )
        concept_name = proposal["concept_name"]
        if is_hazard_group and not concept_name.startswith(("hazard_", "blunder_", "avoid_")):
            concept_name = f"hazard_{concept_name}"
        if concept_name in existing_names:
            concept_name = f"{concept_name}_{ckey.split('_')[0]}"

        rep_scene = group[0].scene_index
        new_rule = FuzzyConceptRule(
            rule_id=f"rule-sleep-s{rep_scene}-{concept_name[:28]}",
            concept_name=concept_name,
            description=proposal["description"],
            target_leaves=target_leaves,
            low=rule_low,
            high=rule_high,
        )
        active_rules.append(new_rule)
        forged_rules.append(new_rule)
        existing_names.add(concept_name)

        # File the Teacher-consolidated clauses into the 3 branch archives using observed coordinates
        cult_sig = {k: v for k, v in avg_coords.items() if k.startswith("cultural/")}
        phil_sig = {k: v for k, v in avg_coords.items() if k.startswith("philosophical/")}
        exp_sig = {
            k: v
            for k, v in avg_coords.items()
            if k.startswith("experiential/")
        }
        exp_sig[SALIENCE_LEAF] = round(sum(s.salience for s in group) / len(group), 4)
        exp_sig[VALENCE_LEAF] = round(sum(s.valence for s in group) / len(group), 4)
        exp_sig[AROUSAL_LEAF] = round(sum(s.arousal for s in group) / len(group), 4)

        from .chord_memory import compile_felt_chord

        avg_fc = compile_felt_chord(
            avg_coords,
            salience=exp_sig[SALIENCE_LEAF],
            valence=exp_sig[VALENCE_LEAF],
            arousal=exp_sig[AROUSAL_LEAF],
            resonance=0.65,
            learned_leaves=tuple(learned_leaves),
        )
        exec_tag, leg_tag, jud_tag = _derive_concept_tags(avg_coords, avg_fc)

        exec_clause = BranchClauseEntry(
            clause_id=f"exec-sleep-s{rep_scene}-{exec_tag}",
            branch=EXECUTIVE_BRANCH,
            text=proposal["executive_clause_text"],
            signature_coords=cult_sig,
            source_scene=rep_scene,
            concept_tag=exec_tag,
        )
        leg_clause = BranchClauseEntry(
            clause_id=f"leg-sleep-s{rep_scene}-{leg_tag}",
            branch=LEGISLATIVE_BRANCH,
            text=proposal["legislative_clause_text"],
            signature_coords=exp_sig,
            source_scene=rep_scene,
            concept_tag=leg_tag,
        )
        jud_clause = BranchClauseEntry(
            clause_id=f"jud-sleep-s{rep_scene}-{jud_tag}",
            branch=JUDICIAL_BRANCH,
            text=proposal["judicial_clause_text"],
            signature_coords=phil_sig,
            source_scene=rep_scene,
            concept_tag=jud_tag,
        )
        harvested_clause_ids.extend([exec_clause.clause_id, leg_clause.clause_id, jud_clause.clause_id])

        next_exec = cur_archives.executive.record_scene(
            scene_index=rep_scene,
            coordinates=avg_coords,
            action_summary=proposal["executive_clause_text"],
            harvested_clause=exec_clause,
        )
        next_leg = cur_archives.legislative.record_scene(
            scene_index=rep_scene,
            sensory_snapshot=proposal["legislative_clause_text"],
            felt_chord=avg_fc,
            coordinates=avg_coords,
            harvested_clause=leg_clause,
        )
        next_jud = cur_archives.judicial.record_clause_and_chord(
            harvested_clause=jud_clause,
            ratified_chord=new_rule,
        )
        cur_archives = ThreeBranchArchives(
            executive=next_exec,
            legislative=next_leg,
            judicial=next_jud,
        )

        for s in group:
            resolved_scene_indices.append(s.scene_index)
            ledger = ledger.record_settled(
                scene_index=s.scene_index,
                takeaway_source=proposal["description"],
                felt_chord=avg_fc,
                clarity=0.85,
                confidence=0.90,
                resolved_uncertainty_scenes=(s.scene_index,),
            )

    if ruleset_out_path is not None:
        save_ruleset(ruleset_out_path, tuple(active_rules), tuple(learned_leaves))
    if archives_out_dir is not None:
        cur_archives.save_all(archives_out_dir)
    if ledger_out_path is not None:
        save_contemplation_ledger(ledger_out_path, ledger)

    return SleepConsolidationReport(
        refined_rule_ids=refined_ids,
        forged_rules=tuple(forged_rules),
        harvested_clause_ids=tuple(harvested_clause_ids),
        resolved_scene_indices=tuple(sorted(set(resolved_scene_indices))),
        updated_rules=tuple(active_rules),
        updated_archives=cur_archives,
        updated_ledger=ledger,
    )
