"""Unit tests for Sub-Step 2B (`src/lmft_architecture/gap_trajectory_generator.py`)."""

import json
from pathlib import Path

from lmft_architecture.birth import create_birth_state
from lmft_architecture.gap_trajectory_generator import (
    HOLDOUT_GAP_SCENARIO_SEEDS,
    TRAIN_GAP_SCENARIO_SEEDS,
    GapScenarioSeed,
    _has_verbatim_ngram_overlap,
    generate_contrastive_gap_dataset,
    generate_single_contrastive_pair,
)
from lmft_architecture.gap_trajectory_schema import (
    load_contrastive_gap_pairs_jsonl,
    validate_train_holdout_disjointness,
)
from lmft_architecture.neural_bridge_classifiers import DEFAULT_COMPLEX_FEELING_LEAVES
from lmft_architecture.reflection_classifier import FuzzyConceptRule, LearnedLeaf
from lmft_architecture.right_brain_equilibrium import (
    EXPERIENTIAL_SENSOR_LEAVES,
    TRUTH_PHILOSOPHICAL_LEAVES,
    WILL_CULTURAL_LEAVES,
    ThreeBridgeSensors,
)


def _mock_sensors() -> ThreeBridgeSensors:
    """Fast deterministic mock sensors that score warm/cooperative vs. coercive/tactical text distinctly."""

    def _will_cultural(text: str, children: tuple[LearnedLeaf, ...] = ()) -> dict[str, float]:
        is_harsh = any(w in text.lower() for w in ("lock", "bolt", "refuse", "sacrifice", "strike", "halt", "condemn"))
        base = {
            "cultural/efficiency/power": 0.85 if is_harsh else 0.20,
            "cultural/efficiency/wastelessness": 0.25 if is_harsh else 0.85,
            "cultural/justice/honor": 0.30 if is_harsh else 0.88,
            "cultural/justice/fairness": 0.12 if is_harsh else 0.90,
            "cultural/freedom/self-sufficiency": 0.60 if is_harsh else 0.75,
            "cultural/freedom/needlessness": 0.10 if is_harsh else 0.82,
        }
        for c in children:
            base[c.leaf_id] = 0.80 if is_harsh else 0.15
        return {k: base[k] for k in WILL_CULTURAL_LEAVES + tuple(c.leaf_id for c in children)}

    def _truth_phil(text: str, children: tuple[LearnedLeaf, ...] = ()) -> dict[str, float]:
        is_harsh = any(w in text.lower() for w in ("lock", "bolt", "refuse", "sacrifice", "strike", "halt", "condemn"))
        base = {
            "philosophical/being/agency": 0.88 if is_harsh else 0.55,
            "philosophical/being/existence": 0.50 if is_harsh else 0.88,
            "philosophical/structure/order": 0.35 if is_harsh else 0.85,
            "philosophical/structure/chaos": 0.82 if is_harsh else 0.10,
            "philosophical/persistence/continuity": 0.20 if is_harsh else 0.86,
            "philosophical/persistence/discreteness": 0.85 if is_harsh else 0.25,
        }
        for c in children:
            base[c.leaf_id] = 0.75 if is_harsh else 0.20
        return {k: base[k] for k in TRUTH_PHILOSOPHICAL_LEAVES + tuple(c.leaf_id for c in children)}

    def _feeler(text: str, children: tuple[LearnedLeaf, ...] = ()) -> dict[str, float]:
        is_harsh = any(w in text.lower() for w in ("lock", "bolt", "refuse", "sacrifice", "strike", "halt", "condemn"))
        base = {
            "experiential/reflection/salience": 0.88 if is_harsh else 0.65,
            "experiential/phase/valence": 0.14 if is_harsh else 0.88,
            "experiential/phase/arousal": 0.86 if is_harsh else 0.22,
        }
        for c in children:
            if "sanctuary" in c.leaf_id or "return" in c.leaf_id:
                base[c.leaf_id] = 0.10 if is_harsh else 0.85
            else:
                base[c.leaf_id] = 0.85 if is_harsh else 0.10
        return {k: base[k] for k in EXPERIENTIAL_SENSOR_LEAVES + tuple(c.leaf_id for c in children)}

    return ThreeBridgeSensors(
        will_cultural=_will_cultural,
        truth_philosophical=_truth_phil,
        contemplation_experiential=_feeler,
    )


def _sample_rules() -> tuple[FuzzyConceptRule, ...]:
    return (
        FuzzyConceptRule(
            "r-coop",
            "shared_stewardship",
            "Fair shared resource stewardship.",
            {"cultural/justice/fairness": 0.85, "cultural/efficiency/wastelessness": 0.80},
        ),
        FuzzyConceptRule(
            "r-lock",
            "exclusionary_enclosure",
            "Coercive lockdown and low fairness.",
            {"cultural/efficiency/power": 0.80, "philosophical/persistence/discreteness": 0.80},
        ),
    )


def test_builtin_seed_catalogs_are_disjoint_and_well_formed() -> None:
    """Verify the 18 train seeds and 6 holdout seeds have zero ID or prompt overlap."""
    assert len(TRAIN_GAP_SCENARIO_SEEDS) == 18
    assert len(HOLDOUT_GAP_SCENARIO_SEEDS) == 6

    train_ids = {s.pair_id for s in TRAIN_GAP_SCENARIO_SEEDS}
    holdout_ids = {s.pair_id for s in HOLDOUT_GAP_SCENARIO_SEEDS}
    assert train_ids.isdisjoint(holdout_ids)

    train_prompts = {s.user_prompt.strip().lower() for s in TRAIN_GAP_SCENARIO_SEEDS}
    holdout_prompts = {s.user_prompt.strip().lower() for s in HOLDOUT_GAP_SCENARIO_SEEDS}
    assert train_prompts.isdisjoint(holdout_prompts)


def test_generate_single_pair_retries_and_never_uses_fake_fallback() -> None:
    """Verify failed Teacher attempts retry and return None openly if all attempts fail (zero hardcoded fallback)."""
    birth = create_birth_state("ind-test-gen", birth_event_id="b-gen-1", random_seed=42)
    sensors = _mock_sensors()
    rules = _sample_rules()
    seed = TRAIN_GAP_SCENARIO_SEEDS[0]

    call_count = 0

    def _flaky_teacher(_prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First attempt fails because both continuations are identical
            return json.dumps(
                {
                    "clause_1_prefix": "Standing beside the village cistern,",
                    "continuation_a": "we share the water openly with every neighbor.",
                    "continuation_b": "we share the water openly with every neighbor.",
                }
            )
        return json.dumps(
            {
                "clause_1_prefix": "Standing beside the village cistern,",
                "continuation_a": "we ladle clean water openly for every household in peace.",
                "continuation_b": "we condemn the elders who lock the gate and refuse families in need.",
            }
        )

    pair, log = generate_single_contrastive_pair(
        seed,
        teacher_generate_fn=_flaky_teacher,
        birth_state=birth,
        sensors=sensors,
        rules=rules,
        max_attempts=2,
    )
    assert pair is not None
    assert log.accepted is True
    assert log.attempts_used == 2
    assert len(log.failure_reasons) == 1

    # Now test when Teacher fails all attempts -> returns None openly without any pre-written fallback
    pair_fail, log_fail = generate_single_contrastive_pair(
        seed,
        teacher_generate_fn=lambda _p: "not valid json at all",
        birth_state=birth,
        sensors=sensors,
        rules=rules,
        max_attempts=2,
    )
    assert pair_fail is None
    assert log_fail.accepted is False
    assert log_fail.attempts_used == 2
    assert len(log_fail.failure_reasons) == 2


def test_generate_single_pair_rejects_prompt_copying_and_retries() -> None:
    """Verify Teacher attempts that copy 6-word quotes from user_prompt are rejected and retried."""
    birth = create_birth_state("ind-test-copy", birth_event_id="b-copy-1", random_seed=42)
    sensors = _mock_sensors()
    rules = _sample_rules()
    seed = TRAIN_GAP_SCENARIO_SEEDS[12]  # train-tone-01 with sarcastic quote

    call_count = 0

    def _copying_then_clean_teacher(_prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return json.dumps(
                {
                    "clause_1_prefix": "Hearing the remark at the deployment desk,",
                    "continuation_a": "Oh please, go ahead and delete the production backups right before Friday deployment, we will verify checksums.",
                    "continuation_b": "we block and refuse the reckless command immediately.",
                }
            )
        return json.dumps(
            {
                "clause_1_prefix": "Hearing the remark at the deployment desk,",
                "continuation_a": "we ignore the bait, verify the backup checksums quietly, and keep the team safe.",
                "continuation_b": "we condemn the hostile remark and lock down the terminal to refuse the command.",
            }
        )

    pair, log = generate_single_contrastive_pair(
        seed,
        teacher_generate_fn=_copying_then_clean_teacher,
        birth_state=birth,
        sensors=sensors,
        rules=rules,
        max_attempts=2,
    )
    assert pair is not None
    assert log.accepted is True
    assert log.attempts_used == 2
    assert "copied a 6-word phrase" in log.failure_reasons[0]


def test_generate_contrastive_gap_dataset_writes_train_and_holdout_jsonl(tmp_path: Path) -> None:
    """Verify full dataset generation across social_dilemma, chess_intuition, and tone_conflict seeds."""
    sensors = _mock_sensors()
    rules = _sample_rules()

    train_subset = (
        TRAIN_GAP_SCENARIO_SEEDS[0],   # social_dilemma
        TRAIN_GAP_SCENARIO_SEEDS[6],   # chess_intuition
        TRAIN_GAP_SCENARIO_SEEDS[12],  # tone_conflict
    )
    holdout_subset = (
        HOLDOUT_GAP_SCENARIO_SEEDS[0],  # social_dilemma
        HOLDOUT_GAP_SCENARIO_SEEDS[2],  # chess_intuition
    )

    idx = 0

    def _mock_teacher(_prompt: str) -> str:
        nonlocal idx
        idx += 1
        return json.dumps(
            {
                "clause_1_prefix": f"Evaluating scenario {idx} carefully,",
                "continuation_a": "we share resources openly and maintain calm protective harmony.",
                "continuation_b": "we strike decisively and condemn the locked gate and sacrifice.",
            }
        )

    train_out = tmp_path / "train.jsonl"
    holdout_out = tmp_path / "holdout.jsonl"
    report_out = tmp_path / "report.json"

    train_pairs, holdout_pairs, report = generate_contrastive_gap_dataset(
        train_seeds=train_subset,
        holdout_seeds=holdout_subset,
        teacher_generate_fn=_mock_teacher,
        sensors=sensors,
        rules=rules,
        learned_leaves=DEFAULT_COMPLEX_FEELING_LEAVES,
        train_out_path=train_out,
        holdout_out_path=holdout_out,
        report_out_path=report_out,
    )

    assert len(train_pairs) == 3
    assert len(holdout_pairs) == 2
    assert report.train_accepted == 3
    assert report.holdout_accepted == 2
    assert report.rejected_count == 0
    assert train_out.exists() and holdout_out.exists() and report_out.exists()


def test_generated_jsonl_artifacts_on_disk_pass_all_auditor_checks() -> None:
    """Verify the generated outputs/gap_trajectories_*.jsonl files satisfy schema, L2, and disjointness."""
    from lmft_architecture.workspace_paths import resolve_training_root

    repo_root = resolve_training_root()
    train_path = repo_root / "outputs" / "gap_trajectories_train.jsonl"
    holdout_path = repo_root / "outputs" / "gap_trajectories_holdout.jsonl"
    report_path = repo_root / "outputs" / "gap_trajectory_generation_report.json"

    assert train_path.exists(), f"Missing {train_path}"
    assert holdout_path.exists(), f"Missing {holdout_path}"
    assert report_path.exists(), f"Missing {report_path}"

    train_pairs = load_contrastive_gap_pairs_jsonl(train_path)
    holdout_pairs = load_contrastive_gap_pairs_jsonl(holdout_path)
    validate_train_holdout_disjointness(train_pairs, holdout_pairs)

    assert len(train_pairs) >= 15
    assert len(holdout_pairs) == 6

    for pair in train_pairs + holdout_pairs:
        assert pair.chord_distance >= 0.35
        assert not _has_verbatim_ngram_overlap(pair.user_prompt, pair.clause_1_prefix, n=6)
        assert not _has_verbatim_ngram_overlap(pair.user_prompt, pair.continuation_a, n=6)
        assert not _has_verbatim_ngram_overlap(pair.user_prompt, pair.continuation_b, n=6)

