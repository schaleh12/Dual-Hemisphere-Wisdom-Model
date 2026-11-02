"""Three Dedicated Fast Right-Brain Neural Bridge Classifiers (Do-er, Thinker, Feeler).

Plain-English purpose:
Takes all classification duties off `Qwen3-4B` by using 3 genuinely separate,
lightweight neural pair-scoring classifiers on GPU (`~15-25 ms` total):
1. Do-er (`doer_will_cultural`): scores social duties, rules vs. exclusion,
   power, fairness, honor, wastelessness, self-sufficiency, needlessness, and
   any learned `cultural/*` child drawers.
2. Thinker (`thinker_truth_philosophical`): scores structural reality—agency,
   existence, order, chaos, continuity, sealed discreteness, and any learned
   `philosophical/*` child drawers.
3. Feeler (`feeler_contemplation_experiential`): scores felt salience, valence,
   arousal, and complex `experiential/*` feeling drawers (such as austere
   sanctuary, hollow stillness, bittersweet release, earned return) so complex
   feelings never collapse into flat 0.30 fallback scores.

Because each classifier scores `(trait_prompt, scene_text) -> [0.0, 1.0]`,
newly grown Tier-2 and Tier-3 `LearnedLeaf` drawers work immediately at runtime
from their plain-English descriptions without changing output head dimensions
or using LLM JSON prompts.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from .reflection_classifier import (
    AROUSAL_LEAF,
    SALIENCE_LEAF,
    VALENCE_LEAF,
    LearnedLeaf,
)
from .right_brain_equilibrium import (
    EXPERIENTIAL_SENSOR_LEAVES,
    TRUTH_PHILOSOPHICAL_LEAVES,
    WILL_CULTURAL_LEAVES,
    ThreeBridgeSensors,
)
from .runtime_store import atomic_json

DEFAULT_BRIDGE_BACKBONE = "cross-encoder/ms-marco-MiniLM-L-12-v2"

DOER_BRIDGE = "doer_will_cultural"
THINKER_BRIDGE = "thinker_truth_philosophical"
FEELER_BRIDGE = "feeler_contemplation_experiential"
BRIDGE_NAMES = (DOER_BRIDGE, THINKER_BRIDGE, FEELER_BRIDGE)

BRIDGE_ROLE_HEADERS = {
    DOER_BRIDGE: "Do-er Social & Value Reading",
    THINKER_BRIDGE: "Thinker Structural Reality Reading",
    FEELER_BRIDGE: "Feeler Emotional & Atmospheric Chord",
}

DOER_FOUNDING_DESCRIPTIONS: dict[str, str] = {
    "cultural/efficiency/power": (
        "Command, force, leverage, authority, or active capacity to compel outcomes and control access."
    ),
    "cultural/efficiency/wastelessness": (
        "Careful stewardship, conservation, thrift, and avoiding waste or squandering of resources."
    ),
    "cultural/justice/honor": (
        "Sincere integrity, kept duty, and honest moral responsibility (low when pride is self-serving or hypocritical)."
    ),
    "cultural/justice/fairness": (
        "Open, equitable, non-exclusionary sharing and fair treatment of others in need (low when rules, gates, or force shut people out)."
    ),
    "cultural/freedom/self-sufficiency": (
        "Independent capability, self-reliance, and standing on one's own effort without outside dependence."
    ),
    "cultural/freedom/needlessness": (
        "Peaceful contentment and sufficiency free from urgent lack, crisis, or threat."
    ),
}

THINKER_FOUNDING_DESCRIPTIONS: dict[str, str] = {
    "philosophical/being/agency": (
        "Deliberate purposeful choice, intentional human action, and active intervention."
    ),
    "philosophical/being/existence": (
        "Concrete physical presence, tangible material reality, and grounded bodily or structural state."
    ),
    "philosophical/structure/order": (
        "Structured arrangement, deliberate rules, built boundaries, discipline, or organized pattern."
    ),
    "philosophical/structure/chaos": (
        "Violent disruption, sudden fracture, uncontrolled breakdown, storm, or turbulent disorder."
    ),
    "philosophical/persistence/continuity": (
        "Ongoing flow, sustained rhythm, unbroken endurance, or shared continuation across time."
    ),
    "philosophical/persistence/discreteness": (
        "Sealed-off enclosure, locked separation, rigid isolation, or sharp boundary cutting things apart."
    ),
}

FEELER_FOUNDING_DESCRIPTIONS: dict[str, str] = {
    SALIENCE_LEAF: (
        "How consequential, urgent, emotionally weighty, or attention-demanding this moment is (low for trivial routine chatter)."
    ),
    VALENCE_LEAF: (
        "Genuine human warmth, life-giving relief, kindness, or constructive flourishing (0.0 = bleak, cruel, suffocating, or harmful; 1.0 = warm, restorative, life-affirming)."
    ),
    AROUSAL_LEAF: (
        "Immediate intensity, nervous energy, alarm, or physical tension in the moment (0.0 = deep quiet stillness; 1.0 = peak urgency or crisis)."
    ),
}

DEFAULT_COMPLEX_FEELING_LEAVES: tuple[LearnedLeaf, ...] = (
    LearnedLeaf(
        "experiential/phase",
        "austere_sanctuary",
        "Spare, humble, unpretentious shelter or care offered openly amidst harsh surroundings without fanfare or conditions.",
    ),
    LearnedLeaf(
        "experiential/phase",
        "hollow_stillness",
        "Cold, sterile, airless quiet where outward perfection or sealed isolation has drained away living warmth and human connection.",
    ),
    LearnedLeaf(
        "experiential/phase",
        "bittersweet_release",
        "Poignant simultaneous ache of something rigid breaking apart and life-giving relief as pent-up strain finally gives way.",
    ),
    LearnedLeaf(
        "experiential/synchronization",
        "earned_return",
        "Restored shared rhythm and quiet homecoming after a stretch of hardship, blockage, or estrangement.",
    ),
)


def format_trait_prompt(bridge_name: str, leaf_id: str, description: str) -> str:
    """Format a uniform natural-language trait query for founding and child leaves alike."""
    if bridge_name not in BRIDGE_ROLE_HEADERS:
        raise ValueError(f"unknown bridge_name: {bridge_name!r}")
    short_name = leaf_id.split("/")[-1].replace("_", " ").replace("-", " ")
    header = BRIDGE_ROLE_HEADERS[bridge_name]
    return f"[{header}] {short_name}: {description.strip()}"


@dataclass(frozen=True)
class BridgePairExample:
    """One supervised `(scene_text, trait_description) -> target_score` calibration pair."""

    scenario_id: str
    text: str
    leaf_id: str
    trait_description: str
    target_score: float

    def __post_init__(self):
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must be a nonempty string")
        if not isinstance(self.trait_description, str) or not self.trait_description.strip():
            raise ValueError("trait_description must be a nonempty string")
        if type(self.target_score) not in (int, float) or not math.isfinite(self.target_score) or not (0.0 <= self.target_score <= 1.0):
            raise ValueError(f"target_score must be in [0, 1], got {self.target_score!r}")


class _BridgePairDataset(Dataset):
    def __init__(self, bridge_name: str, examples: Sequence[BridgePairExample]):
        self.prompts = [
            format_trait_prompt(bridge_name, ex.leaf_id, ex.trait_description)
            for ex in examples
        ]
        self.texts = [ex.text for ex in examples]
        self.targets = [float(ex.target_score) for ex in examples]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[str, str, float]:
        return self.prompts[idx], self.texts[idx], self.targets[idx]


class NeuralBridgeClassifier:
    """Single dedicated neural bridge classifier scoring `(text, trait_descriptions)` in [0, 1]."""

    def __init__(
        self,
        bridge_name: str,
        tokenizer,
        model: torch.nn.Module,
        *,
        device: str | torch.device | None = None,
        max_length: int = 224,
        backbone_name: str = DEFAULT_BRIDGE_BACKBONE,
    ):
        if bridge_name not in BRIDGE_NAMES:
            raise ValueError(f"bridge_name must be one of {BRIDGE_NAMES}, got {bridge_name!r}")
        self.bridge_name = bridge_name
        self.tokenizer = tokenizer
        self.backbone_name = backbone_name
        self.max_length = max_length
        resolved_device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.device = resolved_device
        self.model = model.to(self.device)
        self.model.eval()
        self._last_text: str | None = None
        self._last_cache: dict[tuple[str, str], float] = {}

    @classmethod
    def from_pretrained_backbone(
        cls,
        bridge_name: str,
        backbone_name: str = DEFAULT_BRIDGE_BACKBONE,
        *,
        device: str | torch.device | None = None,
        max_length: int = 224,
        local_files_only: bool = False,
    ) -> "NeuralBridgeClassifier":
        tokenizer = AutoTokenizer.from_pretrained(backbone_name, local_files_only=local_files_only)
        model = AutoModelForSequenceClassification.from_pretrained(
            backbone_name,
            num_labels=1,
            ignore_mismatched_sizes=True,
            local_files_only=local_files_only,
        )
        return cls(
            bridge_name,
            tokenizer,
            model,
            device=device,
            max_length=max_length,
            backbone_name=backbone_name,
        )

    def clear_cache(self) -> None:
        self._last_text = None
        self._last_cache = {}

    def score_pairs(
        self,
        text: str,
        trait_descriptions: Mapping[str, str],
        *,
        use_text_cache: bool = True,
    ) -> dict[str, float]:
        """Score a batch of `(leaf_id: trait_description)` pairs against `text` in one pass."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a nonempty string")
        if not trait_descriptions:
            return {}

        if not use_text_cache or text != self._last_text:
            self._last_text = text
            self._last_cache = {}

        results: dict[str, float] = {}
        missing_keys: list[str] = []
        missing_prompts: list[str] = []

        for leaf_id, desc in trait_descriptions.items():
            cache_key = (leaf_id, desc)
            if use_text_cache and cache_key in self._last_cache:
                results[leaf_id] = self._last_cache[cache_key]
            else:
                missing_keys.append(leaf_id)
                missing_prompts.append(format_trait_prompt(self.bridge_name, leaf_id, desc))

        if missing_keys:
            texts_batch = [text] * len(missing_prompts)
            encoded = self.tokenizer(
                missing_prompts,
                texts_batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            self.model.eval()
            with torch.inference_mode():
                logits = self.model(**encoded).logits.squeeze(-1)
                if logits.ndim == 0:
                    logits = logits.unsqueeze(0)
                probs = torch.sigmoid(logits).detach().cpu().tolist()
            for leaf_id, prob in zip(missing_keys, probs):
                val = round(float(min(1.0, max(0.0, prob))), 4)
                results[leaf_id] = val
                if use_text_cache:
                    self._last_cache[(leaf_id, trait_descriptions[leaf_id])] = val

        return {k: results[k] for k in trait_descriptions}

    def save(self, directory: str | Path, *, metadata: Mapping | None = None) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(target)
        self.tokenizer.save_pretrained(target)
        meta_payload = {
            "schema_version": 1,
            "bridge_name": self.bridge_name,
            "backbone_name": self.backbone_name,
            "max_length": self.max_length,
            "metadata": dict(metadata) if metadata is not None else {},
        }
        atomic_json(target / "bridge_metadata.json", meta_payload)

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        device: str | torch.device | None = None,
    ) -> "NeuralBridgeClassifier":
        target = Path(directory)
        meta_path = target / "bridge_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"missing bridge_metadata.json in {target}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("schema_version") != 1:
            raise ValueError("unsupported bridge_metadata schema_version")
        tokenizer = AutoTokenizer.from_pretrained(target, local_files_only=True)
        config = AutoConfig.from_pretrained(target, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            target,
            config=config,
            local_files_only=True,
        )
        return cls(
            meta["bridge_name"],
            tokenizer,
            model,
            device=device,
            max_length=int(meta.get("max_length", 224)),
            backbone_name=str(meta.get("backbone_name", DEFAULT_BRIDGE_BACKBONE)),
        )


def _bridge_training_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Combined continuous BCE + SmoothL1 + pairwise contrast loss to prevent flat-mean collapse."""
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    smooth_l1 = F.smooth_l1_loss(probs, targets, beta=0.08)
    if probs.numel() >= 2:
        pred_diffs = probs.unsqueeze(0) - probs.unsqueeze(1)
        true_diffs = targets.unsqueeze(0) - targets.unsqueeze(1)
        pairwise = F.l1_loss(pred_diffs, true_diffs)
    else:
        pairwise = torch.tensor(0.0, device=logits.device)
    return bce + 0.50 * smooth_l1 + 0.30 * pairwise


def evaluate_bridge_classifier(
    classifier: NeuralBridgeClassifier,
    examples: Sequence[BridgePairExample],
    *,
    batch_size: int = 32,
) -> dict:
    """Evaluate MAE, RMSE, and Pearson correlation on unseen holdout pairs."""
    if not examples:
        raise ValueError("examples must be nonempty")

    classifier.model.eval()
    preds: list[float] = []
    trues: list[float] = []
    by_leaf: dict[str, list[float]] = {}

    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        prompts = [
            format_trait_prompt(classifier.bridge_name, ex.leaf_id, ex.trait_description)
            for ex in chunk
        ]
        texts = [ex.text for ex in chunk]
        encoded = classifier.tokenizer(
            prompts,
            texts,
            padding=True,
            truncation=True,
            max_length=classifier.max_length,
            return_tensors="pt",
        ).to(classifier.device)
        with torch.inference_mode():
            logits = classifier.model(**encoded).logits.squeeze(-1)
            if logits.ndim == 0:
                logits = logits.unsqueeze(0)
            batch_probs = torch.sigmoid(logits).detach().cpu().tolist()
        for ex, p in zip(chunk, batch_probs):
            p_clamped = float(min(1.0, max(0.0, p)))
            preds.append(p_clamped)
            trues.append(float(ex.target_score))
            by_leaf.setdefault(ex.leaf_id, []).append(abs(p_clamped - float(ex.target_score)))

    pred_arr = np.asarray(preds, dtype=np.float64)
    true_arr = np.asarray(trues, dtype=np.float64)
    abs_err = np.abs(pred_arr - true_arr)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((pred_arr - true_arr) ** 2)))
    if np.std(pred_arr) > 1e-8 and np.std(true_arr) > 1e-8:
        pearson_r = float(np.corrcoef(pred_arr, true_arr)[0, 1])
    else:
        pearson_r = 0.0

    per_leaf_mae = {k: round(float(np.mean(vals)), 4) for k, vals in sorted(by_leaf.items())}
    return {
        "bridge_name": classifier.bridge_name,
        "num_pairs": len(examples),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "pearson_r": round(pearson_r, 4),
        "pred_min": round(float(np.min(pred_arr)), 4),
        "pred_max": round(float(np.max(pred_arr)), 4),
        "pred_std": round(float(np.std(pred_arr)), 4),
        "per_leaf_mae": per_leaf_mae,
    }


def train_bridge_classifier(
    classifier: NeuralBridgeClassifier,
    train_examples: Sequence[BridgePairExample],
    *,
    epochs: int = 8,
    batch_size: int = 16,
    learning_rate: float = 3e-5,
    weight_decay: float = 0.01,
    seed: int = 42,
) -> dict:
    """Fine-tune one NeuralBridgeClassifier independently on its supervised calibration split."""
    if not train_examples:
        raise ValueError("train_examples must be nonempty")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset = _BridgePairDataset(classifier.bridge_name, train_examples)
    generator = torch.Generator()
    generator.manual_seed(seed)

    def _collate(batch):
        prompts, texts, targets = zip(*batch)
        enc = classifier.tokenizer(
            list(prompts),
            list(texts),
            padding=True,
            truncation=True,
            max_length=classifier.max_length,
            return_tensors="pt",
        )
        y = torch.tensor(targets, dtype=torch.float32)
        return enc, y

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=_collate,
    )

    optimizer = torch.optim.AdamW(
        classifier.model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    total_steps = max(1, epochs * len(loader))
    warmup_steps = max(1, int(0.10 * total_steps))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    epoch_losses: list[float] = []
    t0 = time.perf_counter()
    for _epoch in range(epochs):
        classifier.model.train()
        running_loss = 0.0
        steps_in_epoch = 0
        for enc, y in loader:
            enc = enc.to(classifier.device)
            y = y.to(classifier.device)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier.model(**enc).logits.squeeze(-1)
            if logits.ndim == 0:
                logits = logits.unsqueeze(0)
            loss = _bridge_training_loss(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.item())
            steps_in_epoch += 1
        epoch_losses.append(round(running_loss / max(1, steps_in_epoch), 5))

    classifier.model.eval()
    classifier.clear_cache()
    train_seconds = round(time.perf_counter() - t0, 3)
    train_eval = evaluate_bridge_classifier(classifier, train_examples, batch_size=max(16, batch_size))
    return {
        "bridge_name": classifier.bridge_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_seconds": train_seconds,
        "epoch_losses": epoch_losses,
        "train_metrics": train_eval,
    }


@dataclass(frozen=True)
class NeuralThreeBridgeSuite:
    """Container holding the 3 independent Right-Brain neural classifiers and wiring them into ThreeBridgeSensors."""

    doer: NeuralBridgeClassifier
    thinker: NeuralBridgeClassifier
    feeler: NeuralBridgeClassifier

    def __post_init__(self):
        if self.doer.bridge_name != DOER_BRIDGE:
            raise ValueError(f"doer must have bridge_name={DOER_BRIDGE!r}")
        if self.thinker.bridge_name != THINKER_BRIDGE:
            raise ValueError(f"thinker must have bridge_name={THINKER_BRIDGE!r}")
        if self.feeler.bridge_name != FEELER_BRIDGE:
            raise ValueError(f"feeler must have bridge_name={FEELER_BRIDGE!r}")
        if (
            self.doer.model is self.thinker.model
            or self.doer.model is self.feeler.model
            or self.thinker.model is self.feeler.model
        ):
            raise ValueError("Do-er, Thinker, and Feeler must be 3 genuinely separate neural models")

    def as_three_bridge_sensors(self) -> ThreeBridgeSensors:
        """Plug the 3 trained neural classifiers directly into ThreeBridgeSensors."""

        def score_will_cultural(text: str, children: tuple[LearnedLeaf, ...] = ()) -> Mapping[str, float]:
            traits = {k: DOER_FOUNDING_DESCRIPTIONS[k] for k in WILL_CULTURAL_LEAVES}
            for child in children:
                traits[child.leaf_id] = child.description
            return self.doer.score_pairs(text, traits, use_text_cache=True)

        def score_truth_philosophical(text: str, children: tuple[LearnedLeaf, ...] = ()) -> Mapping[str, float]:
            traits = {k: THINKER_FOUNDING_DESCRIPTIONS[k] for k in TRUTH_PHILOSOPHICAL_LEAVES}
            for child in children:
                traits[child.leaf_id] = child.description
            return self.thinker.score_pairs(text, traits, use_text_cache=True)

        def score_contemplation_experiential(
            text: str,
            children: tuple[LearnedLeaf, ...] = (),
        ) -> Mapping[str, float]:
            traits = {k: FEELER_FOUNDING_DESCRIPTIONS[k] for k in EXPERIENTIAL_SENSOR_LEAVES}
            for child in children:
                traits[child.leaf_id] = child.description
            return self.feeler.score_pairs(text, traits, use_text_cache=True)

        return ThreeBridgeSensors(
            will_cultural=score_will_cultural,
            truth_philosophical=score_truth_philosophical,
            contemplation_experiential=score_contemplation_experiential,
        )

    def benchmark_latency_ms(
        self,
        sample_text: str,
        *,
        learned_leaves: tuple[LearnedLeaf, ...] = DEFAULT_COMPLEX_FEELING_LEAVES,
        warmup_runs: int = 5,
        timed_runs: int = 25,
    ) -> dict[str, float]:
        """Measure warmed-up millisecond latency per bridge and end-to-end across all 3 bridges."""
        sensors = self.as_three_bridge_sensors()
        exp_children = tuple(l for l in learned_leaves if l.leaf_id.startswith("experiential/"))
        cult_children = tuple(l for l in learned_leaves if l.leaf_id.startswith("cultural/"))
        phil_children = tuple(l for l in learned_leaves if l.leaf_id.startswith("philosophical/"))

        doer_traits = {**DOER_FOUNDING_DESCRIPTIONS, **{c.leaf_id: c.description for c in cult_children}}
        thinker_traits = {**THINKER_FOUNDING_DESCRIPTIONS, **{c.leaf_id: c.description for c in phil_children}}
        feeler_traits = {**FEELER_FOUNDING_DESCRIPTIONS, **{c.leaf_id: c.description for c in exp_children}}

        for _ in range(warmup_runs):
            self.doer.score_pairs(sample_text, doer_traits, use_text_cache=False)
            self.thinker.score_pairs(sample_text, thinker_traits, use_text_cache=False)
            self.feeler.score_pairs(sample_text, feeler_traits, use_text_cache=False)

        def _time_one(fn) -> float:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(timed_runs):
                fn()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return round((time.perf_counter() - t0) * 1000.0 / timed_runs, 2)

        doer_ms = _time_one(lambda: self.doer.score_pairs(sample_text, doer_traits, use_text_cache=False))
        thinker_ms = _time_one(lambda: self.thinker.score_pairs(sample_text, thinker_traits, use_text_cache=False))
        feeler_ms = _time_one(lambda: self.feeler.score_pairs(sample_text, feeler_traits, use_text_cache=False))

        def _full_sensor_pass():
            self.doer.clear_cache()
            self.thinker.clear_cache()
            self.feeler.clear_cache()
            sensors.score_text(sample_text, learned_leaves=learned_leaves, branch_gating=True)

        total_ms = _time_one(_full_sensor_pass)
        return {
            "doer_will_cultural_ms": doer_ms,
            "thinker_truth_philosophical_ms": thinker_ms,
            "feeler_contemplation_experiential_ms": feeler_ms,
            "total_three_bridges_ms": total_ms,
        }

    def save_all(self, root_dir: str | Path, *, metadata_by_bridge: Mapping[str, Mapping] | None = None) -> None:
        base = Path(root_dir)
        meta = metadata_by_bridge or {}
        self.doer.save(base / DOER_BRIDGE, metadata=meta.get(DOER_BRIDGE))
        self.thinker.save(base / THINKER_BRIDGE, metadata=meta.get(THINKER_BRIDGE))
        self.feeler.save(base / FEELER_BRIDGE, metadata=meta.get(FEELER_BRIDGE))

    @classmethod
    def load_all(
        cls,
        root_dir: str | Path,
        *,
        device: str | torch.device | None = None,
    ) -> "NeuralThreeBridgeSuite":
        base = Path(root_dir)
        return cls(
            doer=NeuralBridgeClassifier.load(base / DOER_BRIDGE, device=device),
            thinker=NeuralBridgeClassifier.load(base / THINKER_BRIDGE, device=device),
            feeler=NeuralBridgeClassifier.load(base / FEELER_BRIDGE, device=device),
        )


def load_trained_neural_bridge_sensors(
    root_dir: str | Path = Path("models/right_brain_bridges"),
    *,
    device: str | torch.device | None = None,
) -> ThreeBridgeSensors:
    """Convenience loader returning ThreeBridgeSensors backed by the 3 saved neural checkpoints."""
    suite = NeuralThreeBridgeSuite.load_all(root_dir, device=device)
    return suite.as_three_bridge_sensors()
