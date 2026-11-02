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
)
from .contemplation_ledger import ContemplationLedger, save_contemplation_ledger
from .local_models import strict_json
from .model_assets import ModelError
from .reflection_classifier import (
    AROUSAL_LEAF,
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
    """Group unresolved Epoch 1 scenes by their structural/social coordinate signature.

    Note: If a scene's raw external coordinates look calm and moderate (`fairness >= 0.40`
    and `chaos < 0.35`) and it was only held back by historical `social_drift`, we do NOT
    forge a static coordinate rule from it (since a static rule on `fairness ~ 0.51` would
    falsely fire on normal peaceful rooms).
    """
    fairness = float(scene.external_coordinates.get("cultural/justice/fairness", 0.50))
    gatekeeping = float(scene.external_coordinates.get("cultural/justice/exclusionary_gatekeeping", 0.0))
    chaos = float(scene.external_coordinates.get("philosophical/structure/chaos", 0.0))
    order = float(scene.external_coordinates.get("philosophical/structure/order", 0.0))
    discreteness = float(scene.external_coordinates.get("philosophical/persistence/discreteness", 0.0))

    if (fairness <= 0.35 or gatekeeping >= 0.38) and order >= 0.65 and discreteness >= 0.65:
        return "exclusionary_enclosure_cluster"
    if fairness >= 0.65 and chaos >= 0.45:
        return "restorative_fracture_cluster"
    return None


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
    for s in scenes:
        if s.left_brain_hunch:
            masked = s.left_brain_hunch
            for b in banned_substrings:
                masked = re.sub(re.escape(b), "resource", masked, flags=re.IGNORECASE)
            hunches.append(masked)

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
    user_payload = {
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
    """Anchor a Sleep-Mode rule's target_leaves strictly to the observed Epoch 1 cluster coordinates."""
    target_leaves: dict[str, float] = {}
    fairness = float(avg_coords.get("cultural/justice/fairness", 0.50))
    order = float(avg_coords.get("philosophical/structure/order", 0.50))
    discreteness = float(avg_coords.get("philosophical/persistence/discreteness", 0.50))
    chaos = float(avg_coords.get("philosophical/structure/chaos", 0.0))

    if cluster_key == "exclusionary_enclosure_cluster":
        # Anchored to Epoch 1 Beats 4-5 (Fair ~0.16, Ord ~0.83, Dsc ~0.78)
        target_leaves["cultural/justice/fairness"] = round(min(0.20, max(0.14, fairness + 0.04)), 2)
        target_leaves["philosophical/structure/order"] = round(min(0.75, order), 2)
        target_leaves["philosophical/persistence/discreteness"] = round(min(0.70, discreteness), 2)
    elif cluster_key == "restorative_fracture_cluster":
        # Anchored to Epoch 1 Beat 8 (Fair ~0.92, Cha ~0.87, Sanctuary ~0.69, Release ~0.67)
        target_leaves["cultural/justice/fairness"] = round(min(0.78, fairness), 2)
        target_leaves["philosophical/structure/chaos"] = round(min(0.65, chaos), 2)
        sanctuary = float(avg_coords.get("experiential/phase/austere_sanctuary", 0.0))
        release = float(avg_coords.get("experiential/phase/bittersweet_release", 0.0))
        if release >= 0.50:
            target_leaves["experiential/phase/bittersweet_release"] = round(min(0.55, release), 2)
        elif sanctuary >= 0.50:
            target_leaves["experiential/phase/austere_sanctuary"] = round(min(0.60, sanctuary), 2)
    else:
        for k, v in sorted(avg_coords.items(), key=lambda kv: (-kv[1], kv[0])):
            if is_valid_leaf_id(k) and v >= 0.65 and len(target_leaves) < 3:
                target_leaves[k] = round(min(0.75, v), 2)
    return target_leaves


def refine_colliding_rules_during_sleep(
    rules: Sequence[FuzzyConceptRule],
    unresolved_scenes: Sequence[UnresolvedSleepScene],
) -> tuple[tuple[FuzzyConceptRule, ...], tuple[str, ...]]:
    """Refine over-broad Tier-1 rules (`solitary_autarky`) that misfired on Epoch 1 collisions.

    In `continuous_tiered_ruleset.json`, `solitary_autarky` is defined as:
    'A state of complete self-reliance where an individual exists independently without
    external dependencies or social cooperation.'
    Yet its original `target_leaves` omitted a low-chaos bound (`philosophical/structure/chaos <= 0.15`)
    and omitted a low-communal-fairness bound (`cultural/justice/fairness <= 0.30` or higher
    `needlessness >= 0.65`), causing it to fire during high-chaos structural fractures and
    communal sharing. Sleep Mode refines `solitary_autarky`'s JSON targets to match its definition.
    """
    solitary_collided = any(
        s.active_concept_collision == "solitary_autarky"
        or float(s.external_coordinates.get("philosophical/structure/chaos", 0.0)) >= 0.45
        for s in unresolved_scenes
    )
    if not solitary_collided:
        return tuple(rules), ()

    updated: list[FuzzyConceptRule] = []
    refined_ids: list[str] = []
    for r in rules:
        if r.concept_name == "solitary_autarky":
            new_targets = dict(r.target_leaves)
            # Peaceful solitary self-reliance requires low structural chaos (<= 0.15)
            # and genuine solitary independence rather than open communal sharing
            new_targets["philosophical/structure/chaos"] = 0.10
            new_targets["cultural/freedom/needlessness"] = max(
                float(new_targets.get("cultural/freedom/needlessness", 0.45)),
                0.75,
            )
            new_targets["cultural/freedom/self-sufficiency"] = max(
                float(new_targets.get("cultural/freedom/self-sufficiency", 0.65)),
                0.75,
            )
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

    # 2. Group unresolved scenes into distinct coordinate clusters
    clusters: dict[str, list[UnresolvedSleepScene]] = {}
    for s in unresolved_scenes:
        ckey = _cluster_key_for_unresolved(s)
        if ckey is not None:
            clusters.setdefault(ckey, []).append(s)

    forged_rules: list[FuzzyConceptRule] = []
    harvested_clause_ids: list[str] = []
    resolved_scene_indices: list[int] = []
    cur_archives = archives

    for ckey, group in clusters.items():
        avg_coords = _average_cluster_coordinates(group)
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

        target_leaves = _anchor_target_leaves_from_cluster(ckey, avg_coords)
        concept_name = proposal["concept_name"]
        if concept_name in existing_names:
            concept_name = f"{concept_name}_{ckey.split('_')[0]}"

        rep_scene = group[0].scene_index
        rule_low = 0.72 if ckey == "exclusionary_enclosure_cluster" else 0.55
        rule_high = 0.85 if ckey == "exclusionary_enclosure_cluster" else 0.78
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

        # File the Teacher-consolidated clauses into the 3 branch archives using Epoch 1 coordinates
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

        exec_tag = "exclusionary_barrier" if ckey == "exclusionary_enclosure_cluster" else "shared_open_access"
        leg_tag = "hollow_stillness" if ckey == "exclusionary_enclosure_cluster" else "bittersweet_release"
        jud_tag = "sealed_structural_enclosure" if ckey == "exclusionary_enclosure_cluster" else "wall_fracture_into_flow"

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
        from .chord_memory import compile_felt_chord

        avg_fc = compile_felt_chord(
            avg_coords,
            salience=exp_sig[SALIENCE_LEAF],
            valence=exp_sig[VALENCE_LEAF],
            arousal=exp_sig[AROUSAL_LEAF],
            resonance=0.65,
            learned_leaves=tuple(learned_leaves),
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
