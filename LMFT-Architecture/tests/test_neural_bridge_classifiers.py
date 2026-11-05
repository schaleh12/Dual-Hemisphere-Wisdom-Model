"""Automated unit tests for the 3 independent Right-Brain Neural Bridge Classifiers."""

from pathlib import Path
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

from lmft_architecture.birth import create_birth_state
from lmft_architecture.neural_bridge_classifiers import (
    DEFAULT_COMPLEX_FEELING_LEAVES,
    DOER_BRIDGE,
    FEELER_BRIDGE,
    THINKER_BRIDGE,
    BridgePairExample,
    NeuralBridgeClassifier,
    NeuralThreeBridgeSuite,
    evaluate_bridge_classifier,
    load_trained_neural_bridge_sensors,
    train_bridge_classifier,
)
from lmft_architecture.reflection_classifier import FuzzyConceptRule, LearnedLeaf
from lmft_architecture.right_brain_equilibrium import settle_scenario


def _make_tiny_classifier(bridge_name: str, seed: int = 7) -> NeuralBridgeClassifier:
    """Create a tiny fast offline Transformer cross-encoder for deterministic unit tests."""
    torch.manual_seed(seed)
    words = [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]",
        "fair", "open", "share", "coop", "crew", "warm", "relief", "calm",
        "lock", "shut", "exclude", "cold", "bleak", "force", "bribe", "order",
        "do", "er", "social", "value", "reading", "thinker", "structural",
        "reality", "feeler", "emotional", "atmospheric", "chord", "fairness",
        "equitable", "treatment", "power", "command", "valence", "salience", "arousal",
    ]
    vocab = {w: idx for idx, w in enumerate(words)}
    tok_obj = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok_obj.pre_tokenizer = Whitespace()
    tok_obj.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )
    cfg = BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.08,
        num_labels=1,
    )
    model = BertForSequenceClassification(cfg)
    return NeuralBridgeClassifier(
        bridge_name,
        fast_tok,
        model,
        device="cpu",
        max_length=64,
        backbone_name="tiny-unit-test-bert",
    )


def test_suite_rejects_shared_weights_and_trains_save_loads_and_scores_dynamic_drawers(tmp_path: Path):
    doer = _make_tiny_classifier(DOER_BRIDGE, seed=1)
    thinker = _make_tiny_classifier(THINKER_BRIDGE, seed=2)
    feeler = _make_tiny_classifier(FEELER_BRIDGE, seed=3)

    # Guard against fake "3 classifiers" sharing one underlying model object
    with pytest.raises(ValueError, match="genuinely separate"):
        NeuralThreeBridgeSuite(doer=doer, thinker=NeuralBridgeClassifier(THINKER_BRIDGE, doer.tokenizer, doer.model, device="cpu"), feeler=feeler)

    # Fine-tune Do-er on separable synthetic pairs and confirm loss drops & MAE improves
    train_pairs = [
        BridgePairExample("s1", "open fair share crew coop", "cultural/justice/fairness", "fairness open equitable treatment", 0.92),
        BridgePairExample("s2", "force lock shut exclude", "cultural/justice/fairness", "fairness open equitable treatment", 0.08),
        BridgePairExample("s2", "force lock shut exclude", "cultural/efficiency/power", "power command force", 0.90),
        BridgePairExample("s3", "calm warm relief", "cultural/efficiency/power", "power command force", 0.06),
    ] * 4

    before = evaluate_bridge_classifier(doer, train_pairs)
    report = train_bridge_classifier(
        doer,
        train_pairs,
        epochs=25,
        batch_size=8,
        learning_rate=2e-3,
        seed=11,
    )
    after = evaluate_bridge_classifier(doer, train_pairs)
    assert after["mae"] < before["mae"]
    assert after["pearson_r"] > 0.70
    assert report["epoch_losses"][-1] < report["epoch_losses"][0]

    # Save all 3 independent checkpoints and reload them cleanly into ThreeBridgeSensors
    suite = NeuralThreeBridgeSuite(doer=doer, thinker=thinker, feeler=feeler)
    ckpt_dir = tmp_path / "bridges"
    suite.save_all(ckpt_dir, metadata_by_bridge={DOER_BRIDGE: report})

    reloaded_suite = NeuralThreeBridgeSuite.load_all(ckpt_dir, device="cpu")
    assert reloaded_suite.doer.model is not reloaded_suite.thinker.model
    assert reloaded_suite.doer.model is not reloaded_suite.feeler.model

    sensors = load_trained_neural_bridge_sensors(ckpt_dir, device="cpu")

    # Verify newly grown Tier-2 and Tier-3 child drawers across all 3 bridges work without JSON prompts!
    new_cult = LearnedLeaf("cultural/justice", "bribe_gate", "bribe lock exclude")
    new_phil = LearnedLeaf("philosophical/structure", "wall_fracture", "structural fracture")
    new_tier3_exp = LearnedLeaf(
        "experiential/phase/bittersweet_release",
        "warm_thaw_relief",
        "warm relief open flow",
    )
    all_learned = DEFAULT_COMPLEX_FEELING_LEAVES + (new_cult, new_phil, new_tier3_exp)

    coords, exp_raw, _queried, _skipped = sensors.score_text(
        "open fair share crew coop warm relief",
        learned_leaves=all_learned,
        branch_gating=False,
    )
    assert 0.0 <= coords["cultural/justice/fairness"] <= 1.0
    assert 0.0 <= coords[new_cult.leaf_id] <= 1.0
    assert 0.0 <= coords[new_phil.leaf_id] <= 1.0
    assert 0.0 <= coords[new_tier3_exp.leaf_id] <= 1.0
    assert coords["cultural/justice/fairness"] > coords["cultural/efficiency/power"]
    assert set(exp_raw.keys()) == {
        "experiential/reflection/salience",
        "experiential/phase/valence",
        "experiential/phase/arousal",
    }

    # Verify full end-to-end settle_scenario execution with neural sensors
    birth = create_birth_state("rb-neural-unit", birth_event_id="b-1", random_seed=5)
    rule = FuzzyConceptRule(
        rule_id="r-coop",
        concept_name="fair_structured_cooperation",
        description="Fair shared cooperation.",
        target_leaves={"cultural/justice/fairness": 0.70},
    )
    eq = settle_scenario(
        birth,
        "open fair share crew coop warm relief",
        sensors,
        (rule,),
        learned_leaves=DEFAULT_COMPLEX_FEELING_LEAVES,
    )
    assert eq.felt_chord is not None
    assert eq.final_vibe.coordinates["cultural/justice/fairness"] > 0.60
