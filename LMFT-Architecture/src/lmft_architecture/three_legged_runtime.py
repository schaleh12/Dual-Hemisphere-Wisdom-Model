"""Stage 4: Closed-Loop Dual-Hemisphere Runtime Integration (`three_legged_runtime.py`).

Plain-English Purpose:
Connects the Fast Right Brain (24-leaf fuzzy tree + 3 MiniLM neural bridges + 3-branch
archives) and the Slow Left Brain (`Qwen/Qwen3-4B` + `LoRALinear` + `SubliminalGapProjector`)
into a single 3-Gear runtime turn (`run_three_legged_turn`) with zero English stage
directions in the LLM prompt:

- **Gear 1 (Fast Right-Brain Reflex & 3-Branch Deliberation, ~5–30 ms, 0 LLM calls)**:
  Scores external reality once (`Fact` locked), runs 24-leaf propagation, and—if a
  tug occurs—lets the Executive, Legislative, and Judicial branches deliberate in
  coordinate space. Routine or precedent-matched scenes settle here with 0 LLM calls.
- **Gear 2 (Subliminal-Gap Spliced Contemplation, ~150–400 ms, 1–2 LLM passes)**:
  When the Right Brain remains in `TUG_LEFT_BRAIN` (or an open ledger thread triggers),
  `SubliminalGapLeftBrain`:
  1. Generates an opening clause (`clause_1_prefix`) from `"User: {scenario_text}\\nAssistant:"`
     and stops cleanly at the first comma `,` before committing to an action decision.
  2. Compiles the live 52-float `SubliminalChordVector` from `current_vibe` (`17` Do-er +
     `16` Thinker + `16` Feeler + `3` Cadence) and splices it into `<|subliminal_gap|>`
     as 4 virtual token embeddings (`[Do-er, Thinker, Feeler, Cadence]`) with **zero**
     English stage directions (`TacticalHail`, `Red-Pen Margin`, etc.) in the text prompt.
  3. Generates the post-gap Clause 2 continuation and re-scores the full 2-clause thought
     back through the 3 MiniLM bridges into `Truth / Theory` while keeping external `Fact`
     immutable.
- **Gear 3 (Memory Compaction, Archive Harvesting & Sleep Queueing)**:
  Compacts the turn into `ContemplationLedger`, `ChordMemoryBank`, `SettledDiary`, and
  (only when `settled=True`) `ThreeBranchArchives`. Any genuinely unsettled turn
  (`UNRESOLVED_TENSION`) is never harvested as a ratified rule and is instead packaged
  as an `UnresolvedSleepScene` for Stage 5 Sleep Mode.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
import re

import torch
from torch import Tensor, nn

from .birth import BirthState
from .propagation import Association
from .branch_archives import (
    SocialDriftReport,
    ThreeBranchArchives,
    ThreeClassifierDeliberation,
    check_open_thread_trigger,
    deliberate_across_three_branches,
    harvest_into_branch_archives,
)
from .chord_memory import (
    ChordMemoryBank,
    compile_felt_chord,
    retrieve_chord_echoes,
)
from .contemplation_ledger import (
    ContemplationLedger,
    RawThoughtRecord,
    append_raw_thought_record,
    save_contemplation_ledger,
)
from .reflection_classifier import (
    ContinuousMemoryState,
    FOUNDATIONAL_LEAVES,
    FuzzyConceptRule,
    LearnedLeaf,
)
from .right_brain_equilibrium import (
    DEFAULT_EQUILIBRIUM_TOLERANCE,
    RIGHT_BRAIN_WILL_EDGES,
    RIGHT_BRAIN_WILL_VERSION,
    DiaryEntry,
    EquilibriumPassTrace,
    EquilibriumResult,
    RightBrainVibe,
    SettledDiary,
    ThreeBridgeSensors,
    _make_margin_tag_for_vibe,
    _top_key_coordinates,
    build_tactical_hail,
    compile_right_brain_vibe,
    validate_plain_english_left_brain_reply,
)
from .sleep_consolidation import UnresolvedSleepScene
from .subliminal_gap_splice import (
    CULTURAL_FOUNDING_ORDER,
    EXPERIENTIAL_FOUNDING_ORDER,
    PHILOSOPHICAL_FOUNDING_ORDER,
    SUBLIMINAL_GAP_TOKEN,
    CorpusCallosumMetadata,
    SubliminalChordVector,
    SubliminalGapProjector,
    splice_subliminal_gaps_into_embeds,
)
from .train_left_brain_lora import (
    LoRAConfig,
    ensure_subliminal_gap_token,
    load_lora_and_callosum_checkpoint,
    sculpt_lora_rank_channels,
)

DEFAULT_QWEN_4B_HF_ID = "Qwen/Qwen3-4B"
DEFAULT_QWEN_8B_HF_ID = "Qwen/Qwen3-8B"
DEFAULT_MAX_CONTEXT_CHARS = 24_000


def format_sliding_conversation_history(
    conversation_history: Sequence[tuple[str, str]],
    *,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> str:
    """Format prior `(user_text, clean_assistant_reply)` pairs within `max_context_chars`."""
    if not conversation_history or max_context_chars <= 0:
        return ""
    formatted_turns: list[str] = []
    total_chars = 0
    for u_raw, a_raw in reversed(tuple(conversation_history)):
        u_clean = str(u_raw).strip()
        a_clean = str(a_raw).strip()
        if not u_clean or not a_clean:
            continue
        block = f"User: {u_clean}\nAssistant: {a_clean}"
        if formatted_turns and (total_chars + len(block) + 1 > max_context_chars):
            break
        if not formatted_turns and len(block) > max_context_chars:
            block = block[-max_context_chars:]
        formatted_turns.append(block)
        total_chars += len(block) + 1
    formatted_turns.reverse()
    return "\n".join(formatted_turns)

# Forbidden English stage-direction headers that must NEVER appear in a SubliminalGapLeftBrain prompt
FORBIDDEN_STAGE_DIRECTION_MARKERS: tuple[str, ...] = (
    "TACTICAL HAIL:",
    "FELT CHORD:",
    "PRE-VERBAL MEMORY ECHO:",
    "CONTEMPLATION LEDGER:",
    "SUBLIMINAL MARGIN STAMP:",
    "YOUR PRIOR HUNCH ON THIS SCENE",
)


def assert_zero_stage_directions_in_prompt(prompt_text: str) -> None:
    """Enforce Roadmap Invariant #3: zero English stage directions in the spliced LLM prompt."""
    for marker in FORBIDDEN_STAGE_DIRECTION_MARKERS:
        if marker in prompt_text:
            raise ValueError(
                f"SubliminalGapLeftBrain prompt contains forbidden English stage direction {marker!r}"
            )


_CHAT_TURN_BOUNDARY_MARKERS: tuple[str, ...] = (
    "\nUser:",
    " User:",
    "\nAssistant:",
    " Assistant:",
    "User:",
    "Assistant:",
    "<|im_end|>",
    "<|im_start|>",
)


def _strip_chat_turn_bleed(text: str) -> str:
    """Truncate `text` immediately before any chat turn boundary marker (`User:`, `Assistant:`, `<|im_end|>`, `<|im_start|>`)."""
    raw = str(text or "")
    cut_idx: int | None = None
    for marker in _CHAT_TURN_BOUNDARY_MARKERS:
        pos = raw.find(marker)
        if pos != -1 and (cut_idx is None or pos < cut_idx):
            cut_idx = pos
    if cut_idx is not None:
        raw = raw[:cut_idx]
    return raw


def normalize_clause_1_prefix(raw_prefix: str) -> str:
    """Clean an opening clause so it stops at a comma boundary before `<|subliminal_gap|>`."""
    cleaned = _strip_chat_turn_bleed(raw_prefix)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.replace("<think>", "").replace("</think>", "")
    cleaned = cleaned.replace(SUBLIMINAL_GAP_TOKEN, "")
    cleaned = " ".join(cleaned.replace("\n", " ").split()).strip()
    if not cleaned:
        return "Weighing the situation before us,"

    # If a comma already appears after at least 2 words, truncate right at the first comma
    if "," in cleaned:
        first_comma_idx = cleaned.index(",")
        before_comma = cleaned[:first_comma_idx].strip()
        if len(before_comma.split()) >= 2:
            return f"{before_comma},"

    # Replace any premature sentence-ending punctuation with a comma
    cleaned = cleaned.rstrip(".!?;:—- ").strip()
    if not cleaned or len(cleaned.split()) < 2:
        return "Weighing the situation before us,"
    return f"{cleaned},"


def clean_post_gap_continuation(raw_text: str) -> str:
    """Clean Clause-2 continuation and ensure it ends on a complete sentence boundary."""
    text = _strip_chat_turn_bleed(raw_text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("<think>", "").replace("</think>", "")
    text = " ".join(text.replace("\n", " ").split()).strip()
    if text.count("(") > text.count(")"):
        last_open = text.rfind("(")
        if last_open > 0:
            text = text[:last_open].rstrip(",;:- ").strip()
    if not text:
        return "we hold steady and observe the unfolding tension."

    if text.endswith((".", "!", "?")) and not re.search(r"\b\d+\.$", text):
        return text

    # If the token budget ran out mid-sentence after an earlier complete sentence boundary
    # (not a move number like `1.`), trim to that last complete sentence boundary.
    sentence_ends = [m.end() for m in re.finditer(r"(?<!\b\d)[.!?](?=\s|$)", text)]
    if sentence_ends:
        return text[: sentence_ends[-1]].strip()

    trimmed = text.rstrip(",;:- ").strip()
    if not trimmed:
        return "we hold steady and observe the unfolding tension."
    return f"{trimmed}."


@dataclass(frozen=True)
class SplicedThoughtTrace:
    """Record of one 2-clause Left-Brain thought generated across `<|subliminal_gap|>`."""

    pass_index: int
    clause_1_prefix: str
    prompt_before_gap: str
    spliced_chord: SubliminalChordVector
    post_gap_continuation: str
    full_thought: str
    comma_vibe_posture: str | None = None
    comma_dissonance_notes: tuple[str, ...] = ()
    comma_scored_live: bool = False

    def __post_init__(self) -> None:
        assert_zero_stage_directions_in_prompt(self.prompt_before_gap)
        if SUBLIMINAL_GAP_TOKEN not in self.prompt_before_gap:
            raise ValueError("prompt_before_gap must end with <|subliminal_gap|>")

    def to_dict(self) -> dict:
        return {
            "pass_index": self.pass_index,
            "clause_1_prefix": self.clause_1_prefix,
            "prompt_before_gap": self.prompt_before_gap,
            "spliced_chord": self.spliced_chord.to_dict(),
            "post_gap_continuation": self.post_gap_continuation,
            "full_thought": self.full_thought,
            "comma_vibe_posture": self.comma_vibe_posture,
            "comma_dissonance_notes": list(self.comma_dissonance_notes),
            "comma_scored_live": self.comma_scored_live,
        }


def _tokenizer_has_single_think_token(tokenizer) -> bool:
    """Return True only when `<think>` is a registered single token in `tokenizer` (e.g. Qwen3)."""
    if hasattr(tokenizer, "get_vocab"):
        vocab = tokenizer.get_vocab()
        if isinstance(vocab, dict):
            return "<think>" in vocab
    return False


def validate_pytorch_hf_model_source(hf_model_id_or_path: str | Path) -> str:
    """Reject `.gguf` quantized files when loading a PyTorch `<|subliminal_gap|>` model."""
    raw_str = str(hf_model_id_or_path).strip()
    if not raw_str:
        raise ValueError("hf_model_id_or_path cannot be empty")
    if raw_str.lower().endswith(".gguf"):
        raise ValueError(
            f"Cannot load {raw_str!r}: <|subliminal_gap|> embedding splicing and LoRA training require "
            "HuggingFace PyTorch weights (e.g. 'SicariusSicariiStuff/Impish_Bloodmoon_12B' or a local HF "
            "directory with config.json + model.safetensors), not a quantized .gguf file."
        )
    candidate_path = Path(raw_str)
    if candidate_path.exists() and candidate_path.is_dir():
        has_gguf = any(candidate_path.glob("*.gguf"))
        has_pt = (
            (candidate_path / "config.json").exists()
            or any(candidate_path.glob("*.safetensors"))
            or any(candidate_path.glob("*.bin"))
        )
        if has_gguf and not has_pt:
            raise ValueError(
                f"Directory {raw_str!r} contains only .gguf files: <|subliminal_gap|> embedding splicing "
                "and LoRA training require HuggingFace PyTorch safetensors weights, not .gguf."
            )
    return raw_str


@dataclass
class _PersistentTurnStreamState:
    """Persistent KV-cache & embedding stream state across Clause 1, `<|subliminal_gap|>`, and multi-sentence chaining."""

    cur_embeds: Tensor
    cur_mask: Tensor
    pending_step_embeds: Tensor | None = None
    past_kv: Any = None
    active_gap_spans: list[tuple[int, int]] = field(default_factory=list)
    masked_gap_spans: list[tuple[int, int]] = field(default_factory=list)
    generated_turn_token_ids: list[int] = field(default_factory=list)


def _apply_repetition_and_ngram_penalty(
    logits: Tensor,
    prior_token_ids: Sequence[int],
    *,
    repetition_penalty: float = 1.22,
    no_repeat_ngram_size: int = 3,
) -> None:
    """Apply standard HuggingFace repetition penalty and n-gram blocking to `logits` in-place (when `vocab_size > 1000`)."""
    vocab_size = int(logits.size(0))
    if vocab_size <= 1000 or not prior_token_ids:
        return
    if repetition_penalty > 1.0:
        recent_32 = [int(t) for t in prior_token_ids[-32:]]
        freq_32: dict[int, int] = {}
        for t in recent_32:
            freq_32[t] = freq_32.get(t, 0) + 1
        for tok_id in set(int(t) for t in prior_token_ids[-96:]):
            if 0 <= tok_id < vocab_size:
                val = float(logits[tok_id].item())
                pen = (repetition_penalty ** 2) if freq_32.get(tok_id, 0) >= 2 else repetition_penalty
                if val > 0.0:
                    logits[tok_id] = val / pen
                else:
                    logits[tok_id] = val * pen
    if no_repeat_ngram_size >= 2 and len(prior_token_ids) >= no_repeat_ngram_size - 1:
        prefix_len = no_repeat_ngram_size - 1
        tail = tuple(int(t) for t in prior_token_ids[-prefix_len:])
        limit = len(prior_token_ids) - prefix_len
        for idx in range(limit):
            if tuple(int(t) for t in prior_token_ids[idx : idx + prefix_len]) == tail:
                banned_id = int(prior_token_ids[idx + prefix_len])
                if 0 <= banned_id < vocab_size:
                    logits[banned_id] = -1e4
    recent_2gram_window = prior_token_ids[-48:]
    if len(recent_2gram_window) >= 2:
        last_tok = int(recent_2gram_window[-1])
        for idx in range(len(recent_2gram_window) - 1):
            if int(recent_2gram_window[idx]) == last_tok:
                banned_2gram_id = int(recent_2gram_window[idx + 1])
                if 0 <= banned_2gram_id < vocab_size:
                    logits[banned_2gram_id] = -1e4


def is_repetitive_sentence(sentence: str, *, max_word_repeats: int = 3) -> bool:
    """Return True when any 4+ character content word appears `>= max_word_repeats` times in `sentence`."""
    words = re.findall(r"[a-zA-Z]{4,}", str(sentence or "").lower())
    if not words:
        return False
    counts: dict[str, int] = {}
    for w in words:
        c = counts.get(w, 0) + 1
        if c >= max_word_repeats:
            return True
        counts[w] = c
    return False


def extract_dream_vow_sentence(dream_text: str, *, fallback: str = "") -> str:
    """Return the final non-repetitive Executive vow/takeaway sentence from a REM dream stanza or chronicle."""
    cleaned = " ".join(_strip_chat_turn_bleed(str(dream_text or "")).split()).strip()
    if not cleaned:
        return _strip_chat_turn_bleed(str(fallback or "")).strip()
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", cleaned)
        if s and s.strip()
    ]
    valid_sents = [
        sent
        for sent in sentences
        if not sent.endswith("..") and not is_repetitive_sentence(sent)
    ]
    for sent in reversed(valid_sents):
        low = sent.lower()
        if (
            len(sent.split()) >= 5
            and not low.startswith(("my previous vow", "our previous vow"))
            and any(k in low for k in ("vow", "promise", "carry"))
        ):
            return sent
    for sent in reversed(valid_sents):
        if len(sent.split()) >= 5 and not sent.lower().startswith(("my previous vow", "our previous vow")):
            return sent
    for sent in reversed(valid_sents):
        return sent
    return _strip_chat_turn_bleed(str(fallback or "")).strip()


class _AttenuatedLoRAScalingContext:
    """Context manager that temporarily scales `LoRALinear.scaling` (uniformly or split across `q_proj/k_proj` vs `v_proj/o_proj`) and restores it on exit."""

    def __init__(
        self,
        model: Any,
        factor: float = 0.0,
        *,
        qk_factor: float | None = None,
        vo_factor: float | None = None,
    ) -> None:
        self.model = model
        self.factor = float(factor)
        self.qk_factor = float(qk_factor) if qk_factor is not None else None
        self.vo_factor = float(vo_factor) if vo_factor is not None else None
        self._saved: list[tuple[Any, float]] = []

    def __enter__(self) -> _AttenuatedLoRAScalingContext:
        if self.model is not None:
            if hasattr(self.model, "named_modules"):
                iter_mods = list(self.model.named_modules())
            elif hasattr(self.model, "modules"):
                iter_mods = [("", m) for m in self.model.modules()]
            else:
                iter_mods = []
            for mod_name, mod in iter_mods:
                if hasattr(mod, "lora_a") and hasattr(mod, "lora_b") and hasattr(mod, "scaling"):
                    orig = float(getattr(mod, "scaling"))
                    self._saved.append((mod, orig))
                    short_name = str(mod_name).split(".")[-1]
                    if short_name in ("q_proj", "k_proj") and self.qk_factor is not None:
                        mult = self.qk_factor
                    elif short_name not in ("q_proj", "k_proj") and self.vo_factor is not None:
                        mult = self.vo_factor
                    else:
                        mult = self.factor
                    setattr(mod, "scaling", orig * mult)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        for mod, orig in self._saved:
            setattr(mod, "scaling", orig)
        self._saved.clear()


def attenuate_lora_scaling(
    model: Any,
    factor: float = 0.0,
    *,
    qk_factor: float | None = None,
    vo_factor: float | None = None,
) -> _AttenuatedLoRAScalingContext:
    """Return a context manager that temporarily multiplies `LoRALinear.scaling` across `model` (uniformly or split `q_proj/k_proj` vs `v_proj/o_proj`)."""
    return _AttenuatedLoRAScalingContext(
        model,
        factor=factor,
        qk_factor=qk_factor,
        vo_factor=vo_factor,
    )


def _mask_prior_virtual_gaps(state: _PersistentTurnStreamState) -> None:
    """Zero out `cur_mask` over prior sentences' 4 virtual `<|subliminal_gap|>` tokens while keeping all English words unmasked."""
    if not state.active_gap_spans:
        return
    mask_len = int(state.cur_mask.size(1))
    for g_start, g_end in state.active_gap_spans:
        if 0 <= g_start < mask_len:
            clamped_end = min(int(g_end), mask_len)
            if clamped_end > g_start:
                state.cur_mask[:, g_start:clamped_end] = 0
        state.masked_gap_spans.append((int(g_start), int(g_end)))
    state.active_gap_spans.clear()


def _append_pending_embeds(state: _PersistentTurnStreamState, new_embeds: Tensor) -> None:
    """Queue newly accepted token or virtual-gap embeddings `[1, K, H]` for the next forward step."""
    if new_embeds.size(1) == 0:
        return
    if state.pending_step_embeds is None:
        state.pending_step_embeds = new_embeds
    else:
        state.pending_step_embeds = torch.cat([state.pending_step_embeds, new_embeds], dim=1)


def _crop_stream_state(
    state: _PersistentTurnStreamState,
    target_cached_len: int,
    restored_pending: Tensor | None,
) -> None:
    """Rewind `state` to `target_cached_len` cached positions when post-processing trims or normalizes tokens."""
    state.cur_embeds = state.cur_embeds[:, :target_cached_len, :]
    state.cur_mask = state.cur_mask[:, :target_cached_len]
    state.active_gap_spans = [
        (gs, min(ge, target_cached_len))
        for gs, ge in state.active_gap_spans
        if gs < target_cached_len
    ]
    state.masked_gap_spans = [
        (gs, min(ge, target_cached_len))
        for gs, ge in state.masked_gap_spans
        if gs < target_cached_len
    ]
    if state.past_kv is not None:
        if target_cached_len <= 0:
            state.past_kv = None
        elif hasattr(state.past_kv, "crop"):
            try:
                state.past_kv.crop(target_cached_len)
            except Exception:
                state.past_kv = None
        elif isinstance(state.past_kv, tuple):
            try:
                state.past_kv = tuple(
                    (
                        layer_kv[0][..., :target_cached_len, :],
                        layer_kv[1][..., :target_cached_len, :],
                    )
                    + tuple(layer_kv[2:])
                    for layer_kv in state.past_kv
                )
            except Exception:
                state.past_kv = None
        else:
            state.past_kv = None
    state.pending_step_embeds = restored_pending


class SubliminalGapLeftBrain:
    """PyTorch Left Brain (CausalLM + LoRA + `SubliminalGapProjector`) steered strictly via `<|subliminal_gap|>`."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        projector: SubliminalGapProjector,
        *,
        gap_token_id: int | None = None,
        device: str | torch.device = "cpu",
        max_clause_1_tokens: int = 20,
        max_post_gap_tokens: int = 64,
        callosum_metadata: CorpusCallosumMetadata | None = None,
        lora_config: LoRAConfig | None = None,
        anchor_at_birth: bool = False,
    ) -> None:
        embed_weight = model.get_input_embeddings().weight
        model_hidden_size = int(embed_weight.shape[1])
        if model_hidden_size != int(projector.hidden_size):
            raise ValueError(
                f"Model embedding hidden_size ({model_hidden_size}) does not match "
                f"SubliminalGapProjector.hidden_size ({projector.hidden_size})"
            )
        self.model = model
        self.tokenizer = tokenizer
        self.projector = projector
        self.device = torch.device(device)
        self.gap_token_id = (
            int(gap_token_id)
            if gap_token_id is not None
            else ensure_subliminal_gap_token(tokenizer)
        )
        self.max_clause_1_tokens = int(max_clause_1_tokens)
        self.max_post_gap_tokens = int(max_post_gap_tokens)
        self.callosum_metadata = callosum_metadata
        self.lora_config = lora_config
        self.last_traces: list[SplicedThoughtTrace] = []
        if not getattr(self.projector, "semantically_anchored_at_birth", False):
            should_anchor = bool(anchor_at_birth) or (
                callosum_metadata is not None
                and int(getattr(callosum_metadata, "trained_trajectories_count", 0)) == 0
            )
            if should_anchor:
                self.projector.anchor_to_llm_embeddings(self.model, self.tokenizer)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        *,
        hf_model_id: str | None = None,
        device: str | torch.device | None = None,
        torch_dtype: torch.dtype | None = None,
        load_in_4bit: bool | None = None,
        max_clause_1_tokens: int = 20,
        max_post_gap_tokens: int = 64,
    ) -> "SubliminalGapLeftBrain":
        """Load a HuggingFace CausalLM + trained `LoRALinear` adapters + `SubliminalGapProjector` from disk."""
        import json as _json
        from transformers import AutoModelForCausalLM, AutoTokenizer

        ckpt_path = Path(checkpoint_dir)
        resolved_model_id = hf_model_id
        if not resolved_model_id:
            meta_file = ckpt_path / "corpus_callosum_meta.json"
            lora_file = ckpt_path / "lora_config.json"
            if meta_file.exists():
                raw_m = _json.loads(meta_file.read_text(encoding="utf-8"))
                resolved_model_id = str(raw_m.get("base_model_id", "")).strip() or None
            if not resolved_model_id and lora_file.exists():
                raw_l = _json.loads(lora_file.read_text(encoding="utf-8"))
                resolved_model_id = str(raw_l.get("base_model_id", "")).strip() or None
        resolved_model_id = validate_pytorch_hf_model_source(
            resolved_model_id or DEFAULT_QWEN_4B_HF_ID
        )

        resolved_device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        resolved_dtype = (
            torch_dtype
            if torch_dtype is not None
            else (torch.bfloat16 if resolved_device.type == "cuda" else torch.float32)
        )

        use_4bit = (
            bool(load_in_4bit)
            if load_in_4bit is not None
            else (
                resolved_device.type == "cuda"
                and any(tag in resolved_model_id.lower() for tag in ("8b", "9b", "12b", "impish_bloodmoon"))
            )
        )

        try:
            tokenizer = AutoTokenizer.from_pretrained(resolved_model_id, fix_mistral_regex=True)
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(resolved_model_id)
        gap_token_id = ensure_subliminal_gap_token(tokenizer)

        if use_4bit and resolved_device.type == "cuda":
            from transformers import BitsAndBytesConfig

            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=resolved_dtype,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                resolved_model_id,
                quantization_config=quant_cfg,
                device_map={"": str(resolved_device)},
                dtype=resolved_dtype,
                low_cpu_mem_usage=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                resolved_model_id,
                dtype=resolved_dtype,
            )

        vocab_len = len(tokenizer) if hasattr(tokenizer, "__len__") else len(tokenizer.get_vocab())
        if not getattr(model, "is_loaded_in_4bit", False):
            if vocab_len > model.get_input_embeddings().num_embeddings:
                try:
                    model.resize_token_embeddings(vocab_len, mean_resizing=False)
                except TypeError:
                    model.resize_token_embeddings(vocab_len)
            model.to(resolved_device)

        projector, meta, lora_cfg = load_lora_and_callosum_checkpoint(
            ckpt_path,
            model=model,
            device=resolved_device,
        )
        sculpt_lora_rank_channels(model, projector)
        model.eval()
        projector.eval()
        return cls(
            model=model,
            tokenizer=tokenizer,
            projector=projector,
            gap_token_id=gap_token_id,
            device=resolved_device,
            max_clause_1_tokens=max_clause_1_tokens,
            max_post_gap_tokens=max_post_gap_tokens,
            callosum_metadata=meta,
            lora_config=lora_cfg,
        )

    def _suppressed_token_ids(self) -> list[int]:
        """Collect token IDs for `<think>`, `</think>`, and `<|subliminal_gap|>` so generation stays clean."""
        suppressed: list[int] = [int(self.gap_token_id)]
        if _tokenizer_has_single_think_token(self.tokenizer) and hasattr(self.tokenizer, "encode"):
            for tag in ("<think>", "</think>", "<think>\n", "</think>\n"):
                ids = self.tokenizer.encode(tag, add_special_tokens=False)
                if len(ids) == 1 and ids[0] not in suppressed:
                    suppressed.append(int(ids[0]))
        return suppressed

    def _forward_step_with_optional_kv_cache(
        self,
        *,
        cur_embeds: Tensor,
        step_embeds: Tensor,
        cur_mask: Tensor,
        past_key_values: Any = None,
    ) -> tuple[Tensor, Any]:
        """Run one autoregressive step using KV-cache + `logits_to_keep=1` when supported, with fallback for stub models."""
        if past_key_values is not None:
            try:
                out = self.model(
                    inputs_embeds=step_embeds,
                    attention_mask=cur_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                return out.logits[0, -1, :].clone(), getattr(out, "past_key_values", None)
            except TypeError:
                pass
        try:
            out = self.model(
                inputs_embeds=cur_embeds,
                attention_mask=cur_mask,
                use_cache=True,
                logits_to_keep=1,
            )
        except TypeError:
            out = self.model(
                inputs_embeds=cur_embeds,
                attention_mask=cur_mask,
                use_cache=True,
            )
        return out.logits[0, -1, :].clone(), getattr(out, "past_key_values", None)

    def _encode_continuation_embeds(self, text: str) -> Tensor:
        """Encode continuation text (`' ' + text.strip()`) without duplicate BOS tokens and return `[1, K, H]` embeddings."""
        embed_layer = self.model.get_input_embeddings()
        embed_dtype = next(embed_layer.parameters()).dtype
        hidden_size = int(embed_layer.weight.shape[1])
        cleaned = text.strip()
        if not cleaned:
            return torch.empty((1, 0, hidden_size), dtype=embed_dtype, device=self.device)
        tok_ids = self.tokenizer.encode(" " + cleaned, add_special_tokens=False)
        if not tok_ids:
            tok_ids = self.tokenizer.encode(cleaned, add_special_tokens=False)
        if not tok_ids:
            return torch.empty((1, 0, hidden_size), dtype=embed_dtype, device=self.device)
        num_emb = int(embed_layer.weight.shape[0])
        safe_ids = [int(t) if 0 <= int(t) < num_emb else 0 for t in tok_ids]
        id_tensor = torch.tensor([safe_ids], dtype=torch.long, device=self.device)
        return embed_layer(id_tensor).to(dtype=embed_dtype)

    def _encode_stanza_break_embeds(self) -> Tensor:
        """Encode a paragraph break (`'\\n\\n'`) into `[1, K, H]` embeddings for multi-stanza streaming."""
        embed_layer = self.model.get_input_embeddings()
        embed_dtype = next(embed_layer.parameters()).dtype
        hidden_size = int(embed_layer.weight.shape[1])
        tok_ids = self.tokenizer.encode("\n\n", add_special_tokens=False)
        if not tok_ids:
            tok_ids = self.tokenizer.encode("\n", add_special_tokens=False)
        if not tok_ids:
            return torch.empty((1, 0, hidden_size), dtype=embed_dtype, device=self.device)
        num_emb = int(embed_layer.weight.shape[0])
        safe_ids = [int(t) if 0 <= int(t) < num_emb else 0 for t in tok_ids]
        id_tensor = torch.tensor([safe_ids], dtype=torch.long, device=self.device)
        return embed_layer(id_tensor).to(dtype=embed_dtype)

    def _init_turn_stream_state(
        self,
        base_prompt: str,
        *,
        prior_assistant_text: str = "",
    ) -> _PersistentTurnStreamState | None:
        """Initialize a persistent KV-cache stream state by encoding `base_prompt` once at turn start."""
        prior_clean = prior_assistant_text.strip()
        full_prompt = f"{base_prompt} {prior_clean}" if prior_clean else base_prompt
        prompt_ids = self.tokenizer.encode(full_prompt, add_special_tokens=True)
        if not prompt_ids:
            return None
        embed_layer = self.model.get_input_embeddings()
        embed_dtype = next(embed_layer.parameters()).dtype
        cur_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        prompt_embeds = embed_layer(cur_ids).to(dtype=embed_dtype)
        return _PersistentTurnStreamState(
            cur_embeds=prompt_embeds[:, :0, :],
            cur_mask=torch.ones((1, 0), dtype=torch.long, device=self.device),
            pending_step_embeds=prompt_embeds,
            past_kv=None,
        )

    def _step_stream_state(self, state: _PersistentTurnStreamState) -> Tensor:
        """Advance `state` by consuming `state.pending_step_embeds` and updating `state.past_kv`."""
        step_embeds = state.pending_step_embeds
        if step_embeds is None or step_embeds.size(1) == 0:
            step_embeds = state.cur_embeds[:, -1:, :]
        else:
            state.cur_embeds = torch.cat([state.cur_embeds, step_embeds], dim=1)
            state.cur_mask = torch.cat(
                [
                    state.cur_mask,
                    torch.ones((1, step_embeds.size(1)), dtype=torch.long, device=self.device),
                ],
                dim=1,
            )
            state.pending_step_embeds = None
        next_logits, state.past_kv = self._forward_step_with_optional_kv_cache(
            cur_embeds=state.cur_embeds,
            step_embeds=step_embeds,
            cur_mask=state.cur_mask,
            past_key_values=state.past_kv,
        )
        return next_logits

    @torch.no_grad()
    def generate_clause_1_prefix(
        self,
        scenario_text: str,
        *,
        clause_1_override: str | None = None,
        prior_assistant_text: str = "",
        conversation_history: Sequence[tuple[str, str]] = (),
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        _stream_state: _PersistentTurnStreamState | None = None,
    ) -> str:
        """Step 2a: Generate Clause 1 from `"User: {scenario_text}\\nAssistant:"` and stop at the first comma."""
        if clause_1_override is not None and clause_1_override.strip():
            norm_override = normalize_clause_1_prefix(clause_1_override)
            if _stream_state is not None:
                _append_pending_embeds(_stream_state, self._encode_continuation_embeds(norm_override))
                if hasattr(self.tokenizer, "encode"):
                    _stream_state.generated_turn_token_ids.extend(
                        int(t) for t in self.tokenizer.encode(" " + norm_override, add_special_tokens=False)
                    )
            return norm_override

        owns_stream = _stream_state is None
        was_training = self.model.training
        self.model.eval()
        state = _stream_state
        try:
            if state is None:
                history_prefix = format_sliding_conversation_history(
                    conversation_history,
                    max_context_chars=max_context_chars,
                )
                history_block = f"{history_prefix}\n" if history_prefix else ""
                base_prompt = f"{history_block}User: {scenario_text.strip()}\nAssistant:"
                state = self._init_turn_stream_state(
                    base_prompt,
                    prior_assistant_text=prior_assistant_text,
                )
                if state is None:
                    return "Weighing the situation before us,"

            c1_start_cached_len = int(state.cur_embeds.size(1))

            embed_layer = self.model.get_input_embeddings()
            embed_dtype = next(embed_layer.parameters()).dtype
            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            suppressed_ids = self._suppressed_token_ids()
            generated_ids: list[int] = []

            for step_idx in range(self.max_clause_1_tokens):
                next_logits = self._step_stream_state(state)
                if step_idx == 0:
                    c1_start_cached_len = int(state.cur_embeds.size(1))
                for bad_id in suppressed_ids:
                    if 0 <= bad_id < next_logits.size(0):
                        next_logits[bad_id] = -1e9
                if step_idx == 0 and eos_id is not None and 0 <= int(eos_id) < next_logits.size(0):
                    next_logits[int(eos_id)] = -1e9
                _apply_repetition_and_ngram_penalty(
                    next_logits,
                    (state.generated_turn_token_ids if state is not None else []) + generated_ids,
                )

                next_id = int(torch.argmax(next_logits).item())
                if eos_id is not None and next_id == int(eos_id):
                    break
                generated_ids.append(next_id)
                next_tok_tensor = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
                next_embed = embed_layer(next_tok_tensor).to(dtype=embed_dtype)
                _append_pending_embeds(state, next_embed)

                decoded_so_far = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                if any(m in decoded_so_far for m in _CHAT_TURN_BOUNDARY_MARKERS):
                    break
                # Stop as soon as a comma or semicolon appears after at least 3 generated tokens
                if step_idx >= 2 and ("," in decoded_so_far or ";" in decoded_so_far):
                    break

            if state is not None and generated_ids:
                state.generated_turn_token_ids.extend(generated_ids)
            raw_clause_1 = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            norm_clause_1 = normalize_clause_1_prefix(raw_clause_1)
            if not owns_stream:
                raw_compact = " ".join(raw_clause_1.replace("\n", " ").split()).strip()
                if raw_compact != norm_clause_1:
                    _crop_stream_state(state, c1_start_cached_len, None)
                    _append_pending_embeds(state, self._encode_continuation_embeds(norm_clause_1))
            return norm_clause_1
        finally:
            if owns_stream and state is not None:
                state.past_kv = None
            if was_training:
                self.model.train()

    @torch.no_grad()
    def generate_two_clause_thought(
        self,
        scenario_text: str,
        chord: SubliminalChordVector | None = None,
        *,
        pass_index: int = 2,
        clause_1_prefix: str | None = None,
        prior_assistant_text: str = "",
        conversation_history: Sequence[tuple[str, str]] = (),
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        chord_builder_at_comma: Callable[
            [str],
            SubliminalChordVector | tuple[SubliminalChordVector, str | None, tuple[str, ...]],
        ]
        | None = None,
        lead_with_gap: bool = False,
        _stream_state: _PersistentTurnStreamState | None = None,
    ) -> SplicedThoughtTrace:
        """Step 2a-2c: Generate Clause 1 (or lead directly with `<|subliminal_gap|>`), splice 52-float `chord`, and generate continuation."""
        if chord is None and chord_builder_at_comma is None:
            raise ValueError("Either chord or chord_builder_at_comma must be provided")
        owns_stream = _stream_state is None
        was_model_training = self.model.training
        was_proj_training = self.projector.training
        self.model.eval()
        self.projector.eval()
        state = _stream_state
        try:
            history_prefix = format_sliding_conversation_history(
                conversation_history,
                max_context_chars=max_context_chars,
            )
            history_block = f"{history_prefix}\n" if history_prefix else ""
            prior_clean = prior_assistant_text.strip()
            base_prompt = f"{history_block}User: {scenario_text.strip()}\nAssistant:"
            if state is None:
                state = self._init_turn_stream_state(
                    base_prompt,
                    prior_assistant_text=prior_clean,
                )

            if lead_with_gap and not (clause_1_prefix is not None and clause_1_prefix.strip()):
                prefix = ""
            else:
                prefix = self.generate_clause_1_prefix(
                    scenario_text,
                    clause_1_override=clause_1_prefix,
                    prior_assistant_text=prior_clean,
                    conversation_history=conversation_history,
                    max_context_chars=max_context_chars,
                    _stream_state=state,
                )
            comma_vibe_posture: str | None = None
            comma_dissonance_notes: tuple[str, ...] = ()
            comma_scored_live = False
            if chord_builder_at_comma is not None:
                built = chord_builder_at_comma(prefix or prior_clean)
                comma_scored_live = True
                if isinstance(built, tuple):
                    active_chord, comma_vibe_posture, comma_dissonance_notes = built
                else:
                    active_chord = built
            else:
                assert chord is not None
                active_chord = chord

            assistant_prefix = f"{prior_clean} {prefix}".strip() if prior_clean else prefix
            if assistant_prefix:
                prompt_before_gap = (
                    f"{history_block}User: {scenario_text.strip()}\n"
                    f"Assistant: {assistant_prefix} {SUBLIMINAL_GAP_TOKEN}"
                )
            else:
                prompt_before_gap = (
                    f"{history_block}User: {scenario_text.strip()}\n"
                    f"Assistant: {SUBLIMINAL_GAP_TOKEN}"
                )
            assert_zero_stage_directions_in_prompt(prompt_before_gap)

            embed_layer = self.model.get_input_embeddings()
            embed_dtype = next(embed_layer.parameters()).dtype
            proj_param = next(self.projector.parameters(), None)
            proj_dtype = proj_param.dtype if proj_param is not None else torch.float32
            chord_batch = active_chord.to_tensor(device=self.device, dtype=proj_dtype).unsqueeze(0)
            gap_embeds = self.projector(chord_batch).to(dtype=embed_dtype)  # [1, 4, hidden_size]

            if state is None:
                state = _PersistentTurnStreamState(
                    cur_embeds=gap_embeds[:, :0, :],
                    cur_mask=torch.ones((1, 0), dtype=torch.long, device=self.device),
                    pending_step_embeds=gap_embeds,
                    past_kv=None,
                    active_gap_spans=[(0, int(gap_embeds.size(1)))],
                )
            else:
                _mask_prior_virtual_gaps(state)
                pending_len = (
                    int(state.pending_step_embeds.size(1))
                    if state.pending_step_embeds is not None
                    else 0
                )
                gap_start = int(state.cur_embeds.size(1)) + pending_len
                gap_end = gap_start + int(gap_embeds.size(1))
                state.active_gap_spans.append((gap_start, gap_end))
                _append_pending_embeds(state, gap_embeds)

            post_gap_start_cached_len = int(state.cur_embeds.size(1))

            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            suppressed_ids = self._suppressed_token_ids()
            generated_ids: list[int] = []

            for step_idx in range(self.max_post_gap_tokens):
                next_logits = self._step_stream_state(state)
                if step_idx == 0:
                    post_gap_start_cached_len = int(state.cur_embeds.size(1))
                for bad_id in suppressed_ids:
                    if 0 <= bad_id < next_logits.size(0):
                        next_logits[bad_id] = -1e9
                if step_idx == 0 and eos_id is not None and 0 <= int(eos_id) < next_logits.size(0):
                    next_logits[int(eos_id)] = -1e9
                _apply_repetition_and_ngram_penalty(
                    next_logits,
                    (state.generated_turn_token_ids if state is not None else []) + generated_ids,
                )

                next_id = int(torch.argmax(next_logits).item())
                if eos_id is not None and next_id == int(eos_id):
                    break
                generated_ids.append(next_id)
                next_tok_tensor = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
                next_embed = embed_layer(next_tok_tensor).to(dtype=embed_dtype)
                _append_pending_embeds(state, next_embed)

                decoded_cont = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                if any(m in decoded_cont for m in _CHAT_TURN_BOUNDARY_MARKERS):
                    break
                if (
                    step_idx >= 5
                    and decoded_cont.endswith((".", "!", "?"))
                    and not re.search(r"\b\d+\.$", decoded_cont)
                    and decoded_cont.count("(") <= decoded_cont.count(")")
                ):
                    break

            if state is not None and generated_ids:
                state.generated_turn_token_ids.extend(generated_ids)
            raw_post_gap = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            cleaned_post_gap = clean_post_gap_continuation(raw_post_gap)
            if not prefix and cleaned_post_gap:
                cleaned_post_gap = cleaned_post_gap.lstrip(",;:-–— ").strip()
            post_gap_text = cleaned_post_gap
            if not prefix and post_gap_text and post_gap_text[0].islower():
                post_gap_text = post_gap_text[0].upper() + post_gap_text[1:]
            elif (
                prefix
                and prefix.rstrip().endswith((",", ";"))
                and len(post_gap_text) >= 2
                and post_gap_text[0].isupper()
                and post_gap_text[1].islower()
            ):
                post_gap_text = post_gap_text[0].lower() + post_gap_text[1:]

            if not owns_stream and state is not None:
                raw_post_compact = " ".join(raw_post_gap.replace("\n", " ").split()).strip()
                if raw_post_compact != cleaned_post_gap:
                    _crop_stream_state(state, post_gap_start_cached_len, None)
                    _append_pending_embeds(state, self._encode_continuation_embeds(cleaned_post_gap))

            full_thought = f"{prefix} {post_gap_text}".strip()
            trace = SplicedThoughtTrace(
                pass_index=pass_index,
                clause_1_prefix=prefix,
                prompt_before_gap=prompt_before_gap,
                spliced_chord=active_chord,
                post_gap_continuation=post_gap_text,
                full_thought=full_thought,
                comma_vibe_posture=comma_vibe_posture,
                comma_dissonance_notes=comma_dissonance_notes,
                comma_scored_live=comma_scored_live,
            )
            self.last_traces.append(trace)
            if len(self.last_traces) > 50:
                del self.last_traces[:-50]
            return trace
        finally:
            if owns_stream:
                if state is not None:
                    state.past_kv = None
                if self.device.type == "cuda" and torch.cuda.is_available():
                    try:
                        if torch.cuda.memory_reserved(self.device) > 11 * (1024**3):
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
            if was_model_training:
                self.model.train()
            if was_proj_training:
                self.projector.train()

    @torch.no_grad()
    def generate_with_subliminal_splice(
        self,
        scenario_text: str,
        chord: SubliminalChordVector | None = None,
        *,
        num_sentences: int = 1,
        pass_index: int = 2,
        clause_1_prefix: str | None = None,
        clause_1_override: str | None = None,
        conversation_history: Sequence[tuple[str, str]] = (),
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        chord_builder_at_comma: Callable[
            [str],
            SubliminalChordVector | tuple[SubliminalChordVector, str | None, tuple[str, ...]],
        ]
        | None = None,
        stop_condition: Callable[[int, SplicedThoughtTrace, str], bool] | None = None,
        lead_with_gap: bool = False,
    ) -> tuple[str, SplicedThoughtTrace]:
        """Generate up to `num_sentences` `<|subliminal_gap|>` thoughts and return `(full_thought, trace)`."""
        effective_c1 = clause_1_override if clause_1_override is not None else clause_1_prefix
        if int(num_sentences) > 1 or stop_condition is not None:
            passage, traces = self.generate_multi_sentence_script(
                scenario_text,
                chord=chord,
                num_sentences=max(1, int(num_sentences)),
                pass_index=pass_index,
                clause_1_prefix=effective_c1,
                conversation_history=conversation_history,
                max_context_chars=max_context_chars,
                chord_builder_at_comma=chord_builder_at_comma,
                stop_condition=stop_condition,
                lead_with_gap=lead_with_gap,
            )
            self.last_sentence_count = len(traces)
            return passage, traces[-1]
        trace = self.generate_two_clause_thought(
            scenario_text,
            chord=chord,
            pass_index=pass_index,
            clause_1_prefix=effective_c1,
            conversation_history=conversation_history,
            max_context_chars=max_context_chars,
            chord_builder_at_comma=chord_builder_at_comma,
            lead_with_gap=lead_with_gap,
        )
        self.last_sentence_count = 1
        return trace.full_thought, trace

    @torch.no_grad()
    def generate_multi_sentence_script(
        self,
        scenario_text: str,
        chord: SubliminalChordVector | None = None,
        *,
        num_sentences: int = 3,
        pass_index: int = 2,
        clause_1_prefix: str | None = None,
        clause_1_overrides: Sequence[str | None] | None = None,
        conversation_history: Sequence[tuple[str, str]] = (),
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        chord_builder_at_comma: Callable[
            [str],
            SubliminalChordVector | tuple[SubliminalChordVector, str | None, tuple[str, ...]],
        ]
        | None = None,
        stop_condition: Callable[[int, SplicedThoughtTrace, str], bool] | None = None,
        lead_with_gap: bool = False,
    ) -> tuple[str, tuple[SplicedThoughtTrace, ...]]:
        """Chain up to `num_sentences` (`1..6`) `<|subliminal_gap|>` sentences into a continuous passage with a single persistent KV-cache."""
        count = int(num_sentences)
        if not (1 <= count <= 6):
            raise ValueError(f"num_sentences must be in [1, 6], got {num_sentences!r}")
        history_prefix = format_sliding_conversation_history(
            conversation_history,
            max_context_chars=max_context_chars,
        )
        history_block = f"{history_prefix}\n" if history_prefix else ""
        base_prompt = f"{history_block}User: {scenario_text.strip()}\nAssistant:"
        stream_state = self._init_turn_stream_state(base_prompt)
        completed_sentences: list[str] = []
        traces: list[SplicedThoughtTrace] = []
        try:
            for sent_idx in range(count):
                prior_text = " ".join(completed_sentences).strip()
                override_c1: str | None = None
                if clause_1_overrides is not None and sent_idx < len(clause_1_overrides):
                    override_c1 = clause_1_overrides[sent_idx]
                elif sent_idx == 0:
                    override_c1 = clause_1_prefix
                trace = self.generate_two_clause_thought(
                    scenario_text,
                    chord=chord,
                    pass_index=pass_index,
                    clause_1_prefix=override_c1,
                    prior_assistant_text=prior_text,
                    conversation_history=conversation_history,
                    max_context_chars=max_context_chars,
                    chord_builder_at_comma=chord_builder_at_comma,
                    lead_with_gap=lead_with_gap,
                    _stream_state=stream_state,
                )
                vocab_len = len(self.tokenizer) if hasattr(self.tokenizer, "__len__") else 0
                if vocab_len > 1000 and completed_sentences:
                    cand_words = set(re.findall(r"[a-z0-9]+", trace.full_thought.lower()))
                    is_dup = False
                    for prev in completed_sentences:
                        if trace.full_thought.strip().lower() == prev.strip().lower():
                            is_dup = True
                            break
                        prev_words = set(re.findall(r"[a-z0-9]+", prev.lower()))
                        if cand_words and prev_words:
                            overlap = len(cand_words & prev_words) / max(1, min(len(cand_words), len(prev_words)))
                            if overlap >= 0.72:
                                is_dup = True
                                break
                    if is_dup:
                        break
                completed_sentences.append(trace.full_thought)
                traces.append(trace)
                if stop_condition is not None and stop_condition(len(traces), trace, " ".join(completed_sentences).strip()):
                    break
            return " ".join(completed_sentences).strip(), tuple(traces)
        finally:
            if stream_state is not None:
                stream_state.past_kv = None
            if self.device.type == "cuda" and torch.cuda.is_available():
                try:
                    if torch.cuda.memory_reserved(self.device) > 11 * (1024**3):
                        torch.cuda.empty_cache()
                except Exception:
                    pass

    @torch.no_grad()
    def generate_multi_stanza_discourse(
        self,
        scenario_text: str,
        chord: SubliminalChordVector | None = None,
        *,
        max_stanzas: int = 2,
        sentences_per_stanza: int = 3,
        pass_index: int = 2,
        clause_1_prefix: str | None = None,
        clause_1_override: str | None = None,
        stanza_clause_1_overrides: Sequence[str | None] | None = None,
        conversation_history: Sequence[tuple[str, str]] = (),
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        chord_builder_at_comma: Callable[
            [str],
            SubliminalChordVector | tuple[SubliminalChordVector, str | None, tuple[str, ...]],
        ]
        | None = None,
        stanza_boundary_callback: Callable[
            [int, str, str],
            tuple[bool, SubliminalChordVector | None],
        ]
        | None = None,
        lead_with_gap: bool = False,
    ) -> tuple[str, tuple[str, ...], tuple[SplicedThoughtTrace, ...]]:
        """Generate `1..max_stanzas` stanzas (`sentences_per_stanza` sentences each) joined by `\\n\\n` over a single persistent `past_kv` cache."""
        if clause_1_override is not None:
            clause_1_prefix = clause_1_override
        stanza_limit = int(max_stanzas)
        sent_per_stanza = int(sentences_per_stanza)
        if not (1 <= stanza_limit <= 4):
            raise ValueError(f"max_stanzas must be in [1, 4], got {max_stanzas!r}")
        if not (1 <= sent_per_stanza <= 6):
            raise ValueError(f"sentences_per_stanza must be in [1, 6], got {sentences_per_stanza!r}")

        history_prefix = format_sliding_conversation_history(
            conversation_history,
            max_context_chars=max_context_chars,
        )
        history_block = f"{history_prefix}\n" if history_prefix else ""
        base_prompt = f"{history_block}User: {scenario_text.strip()}\nAssistant:"
        stream_state = self._init_turn_stream_state(base_prompt)
        completed_stanzas: list[str] = []
        all_traces: list[SplicedThoughtTrace] = []
        active_stanza_chord: SubliminalChordVector | None = chord
        chord_rotated_for_stanza = False
        try:
            for stanza_idx in range(stanza_limit):
                if stanza_idx > 0 and stream_state is not None:
                    _append_pending_embeds(stream_state, self._encode_stanza_break_embeds())
                stanza_sentences: list[str] = []
                for sent_idx in range(sent_per_stanza):
                    prior_parts = list(completed_stanzas)
                    if stanza_sentences:
                        prior_parts.append(" ".join(stanza_sentences))
                    prior_text = "\n\n".join(prior_parts).strip()
                    override_c1: str | None = None
                    if sent_idx == 0:
                        if stanza_clause_1_overrides is not None and stanza_idx < len(stanza_clause_1_overrides):
                            override_c1 = stanza_clause_1_overrides[stanza_idx]
                        elif stanza_idx == 0:
                            override_c1 = clause_1_prefix
                    use_rotated_chord = (
                        stanza_idx > 0
                        and chord_rotated_for_stanza
                        and active_stanza_chord is not None
                    )
                    trace = self.generate_two_clause_thought(
                        scenario_text,
                        chord=active_stanza_chord,
                        pass_index=pass_index,
                        clause_1_prefix=override_c1,
                        prior_assistant_text=prior_text,
                        conversation_history=conversation_history,
                        max_context_chars=max_context_chars,
                        chord_builder_at_comma=None if use_rotated_chord else chord_builder_at_comma,
                        lead_with_gap=lead_with_gap,
                        _stream_state=stream_state,
                    )
                    vocab_len = len(self.tokenizer) if hasattr(self.tokenizer, "__len__") else 0
                    if vocab_len > 1000 and (stanza_sentences or completed_stanzas):
                        cand_words = set(re.findall(r"[a-z0-9]+", trace.full_thought.lower()))
                        is_dup = False
                        prior_all_sents = [t.full_thought for t in all_traces]
                        for prev in prior_all_sents:
                            if trace.full_thought.strip().lower() == prev.strip().lower():
                                is_dup = True
                                break
                            prev_words = set(re.findall(r"[a-z0-9]+", prev.lower()))
                            if cand_words and prev_words:
                                overlap = len(cand_words & prev_words) / max(1, min(len(cand_words), len(prev_words)))
                                if overlap >= 0.72:
                                    is_dup = True
                                    break
                        if is_dup:
                            break
                    stanza_sentences.append(trace.full_thought)
                    all_traces.append(trace)

                stanza_text = " ".join(stanza_sentences).strip()
                if stanza_text:
                    completed_stanzas.append(stanza_text)
                cumulative_text = "\n\n".join(completed_stanzas).strip()
                if stanza_idx + 1 >= stanza_limit:
                    break
                chord_rotated_for_stanza = False
                if stanza_boundary_callback is not None:
                    should_stop, next_chord = stanza_boundary_callback(
                        stanza_idx,
                        stanza_text,
                        cumulative_text,
                    )
                    if should_stop:
                        break
                    if next_chord is not None:
                        active_stanza_chord = next_chord
                        chord_rotated_for_stanza = True
            self.last_sentence_count = len(all_traces)
            return "\n\n".join(completed_stanzas).strip(), tuple(completed_stanzas), tuple(all_traces)
        finally:
            if stream_state is not None:
                stream_state.past_kv = None
            if self.device.type == "cuda" and torch.cuda.is_available():
                try:
                    if torch.cuda.memory_reserved(self.device) > 11 * (1024**3):
                        torch.cuda.empty_cache()
                except Exception:
                    pass

    def generate_rem_dream_self_reflection(
        self,
        sleep_prompt: str,
        sleep_chord: SubliminalChordVector,
        *,
        max_stanzas: int = 1,
        num_sentences: int = 4,
        sentences_per_stanza: int = 3,
        conversation_history: Sequence[tuple[str, str]] = (),
        chord_builder_at_comma: Callable[
            [str],
            tuple[SubliminalChordVector, str, tuple[str, ...]] | SubliminalChordVector,
        ]
        | None = None,
        stanza_boundary_callback: Callable[
            [int, str, str],
            tuple[bool, SubliminalChordVector | None],
        ]
        | None = None,
        clause_1_override: str | None = None,
        stanza_clause_1_overrides: Sequence[str | None] | None = None,
        lead_with_gap: bool = True,
        lora_scale_factor: float | None = None,
        qk_lora_scale: float | None = 0.35,
        vo_lora_scale: float | None = 0.08,
    ) -> str:
        """Generate REM sleep self-reflection with split-projection LoRA scaling (`qk_lora_scale` vs `vo_lora_scale`) and optional 1st-person `clause_1_override`."""
        clean_c1 = str(clause_1_override).strip() if clause_1_override else None
        effective_lead_with_gap = bool(lead_with_gap)
        resolved_stanza_overrides: tuple[str | None, ...] | None = None
        if int(max_stanzas) > 1:
            if stanza_clause_1_overrides is not None:
                resolved_stanza_overrides = tuple(
                    str(s).strip() if s else None for s in stanza_clause_1_overrides
                )
            elif clean_c1:
                if int(max_stanzas) == 2:
                    resolved_stanza_overrides = (
                        clean_c1,
                        "Carrying this vow into our next game,",
                    )
                else:
                    resolved_stanza_overrides = (
                        clean_c1,
                        "Weighing how the board's structure shifted across the game,",
                        "Carrying this vow into our next game,",
                    )
        if lora_scale_factor is not None:
            ctx = attenuate_lora_scaling(self.model, factor=float(lora_scale_factor))
        else:
            ctx = attenuate_lora_scaling(
                self.model,
                factor=0.0,
                qk_factor=qk_lora_scale,
                vo_factor=vo_lora_scale,
            )
        with ctx:
            if int(max_stanzas) > 1 and hasattr(self, "generate_multi_stanza_discourse"):
                dream_text, _, _ = self.generate_multi_stanza_discourse(
                    sleep_prompt,
                    sleep_chord,
                    max_stanzas=int(max_stanzas),
                    sentences_per_stanza=int(sentences_per_stanza),
                    conversation_history=conversation_history,
                    chord_builder_at_comma=chord_builder_at_comma,
                    stanza_boundary_callback=stanza_boundary_callback,
                    clause_1_override=clean_c1,
                    stanza_clause_1_overrides=resolved_stanza_overrides,
                    lead_with_gap=effective_lead_with_gap,
                )
                return dream_text
            dream_text, _ = self.generate_with_subliminal_splice(
                sleep_prompt,
                sleep_chord,
                num_sentences=int(num_sentences),
                conversation_history=conversation_history,
                chord_builder_at_comma=chord_builder_at_comma,
                clause_1_override=clean_c1,
                lead_with_gap=effective_lead_with_gap,
            )
            return dream_text


def generate_rem_dream_self_reflection(
    left_brain: Any,
    sleep_prompt: str,
    sleep_chord: SubliminalChordVector,
    *,
    max_stanzas: int = 1,
    num_sentences: int = 4,
    sentences_per_stanza: int = 3,
    conversation_history: Sequence[tuple[str, str]] = (),
    chord_builder_at_comma: Callable[
        [str],
        tuple[SubliminalChordVector, str, tuple[str, ...]] | SubliminalChordVector,
    ]
    | None = None,
    stanza_boundary_callback: Callable[
        [int, str, str],
        tuple[bool, SubliminalChordVector | None],
    ]
    | None = None,
    clause_1_override: str | None = None,
    stanza_clause_1_overrides: Sequence[str | None] | None = None,
    lead_with_gap: bool = True,
    lora_scale_factor: float | None = None,
    qk_lora_scale: float | None = 0.35,
    vo_lora_scale: float | None = 0.08,
) -> str:
    """Generate REM sleep self-reflection on `left_brain` with split-projection LoRA scaling and optional 1st-person `clause_1_override`."""
    if hasattr(left_brain, "generate_rem_dream_self_reflection"):
        return left_brain.generate_rem_dream_self_reflection(
            sleep_prompt,
            sleep_chord,
            max_stanzas=max_stanzas,
            num_sentences=num_sentences,
            sentences_per_stanza=sentences_per_stanza,
            conversation_history=conversation_history,
            chord_builder_at_comma=chord_builder_at_comma,
            stanza_boundary_callback=stanza_boundary_callback,
            clause_1_override=clause_1_override,
            stanza_clause_1_overrides=stanza_clause_1_overrides,
            lead_with_gap=lead_with_gap,
            lora_scale_factor=lora_scale_factor,
            qk_lora_scale=qk_lora_scale,
            vo_lora_scale=vo_lora_scale,
        )
    clean_c1 = str(clause_1_override).strip() if clause_1_override else None
    effective_lead_with_gap = bool(lead_with_gap)
    resolved_stanza_overrides: tuple[str | None, ...] | None = None
    if int(max_stanzas) > 1:
        if stanza_clause_1_overrides is not None:
            resolved_stanza_overrides = tuple(
                str(s).strip() if s else None for s in stanza_clause_1_overrides
            )
        elif clean_c1:
            if int(max_stanzas) == 2:
                resolved_stanza_overrides = (
                    clean_c1,
                    "Carrying this vow into our next game,",
                )
            else:
                resolved_stanza_overrides = (
                    clean_c1,
                    "Weighing how the board's structure shifted across the game,",
                    "Carrying this vow into our next game,",
                )
    model_obj = getattr(left_brain, "model", None)
    if lora_scale_factor is not None:
        ctx = attenuate_lora_scaling(model_obj, factor=float(lora_scale_factor))
    else:
        ctx = attenuate_lora_scaling(
            model_obj,
            factor=0.0,
            qk_factor=qk_lora_scale,
            vo_factor=vo_lora_scale,
        )
    with ctx:
        if int(max_stanzas) > 1 and hasattr(left_brain, "generate_multi_stanza_discourse"):
            dream_text, _, _ = left_brain.generate_multi_stanza_discourse(
                sleep_prompt,
                sleep_chord,
                max_stanzas=int(max_stanzas),
                sentences_per_stanza=int(sentences_per_stanza),
                conversation_history=conversation_history,
                chord_builder_at_comma=chord_builder_at_comma,
                stanza_boundary_callback=stanza_boundary_callback,
                clause_1_override=clean_c1,
                stanza_clause_1_overrides=resolved_stanza_overrides,
                lead_with_gap=effective_lead_with_gap,
            )
            return dream_text
        dream_text, _ = left_brain.generate_with_subliminal_splice(
            sleep_prompt,
            sleep_chord,
            num_sentences=int(num_sentences),
            conversation_history=conversation_history,
            chord_builder_at_comma=chord_builder_at_comma,
            clause_1_override=clean_c1,
            lead_with_gap=effective_lead_with_gap,
        )
        return dream_text


@dataclass(frozen=True)
class ThreeLeggedTurnResult:
    """Complete result of one 3-Gear Closed-Loop Dual-Hemisphere Runtime Turn."""

    equilibrium: EquilibriumResult
    gear_used: str
    spliced_traces: tuple[SplicedThoughtTrace, ...] = ()
    unresolved_sleep_scene: UnresolvedSleepScene | None = None

    @property
    def settled(self) -> bool:
        return self.equilibrium.settled

    @property
    def outcome(self) -> str:
        return self.equilibrium.outcome

    @property
    def final_vibe(self) -> RightBrainVibe:
        return self.equilibrium.final_vibe

    @property
    def left_brain_calls(self) -> int:
        return self.equilibrium.left_brain_calls

    @property
    def final_conscious_note(self) -> str | None:
        return self.equilibrium.final_conscious_note

    @property
    def next_archives(self) -> ThreeBranchArchives:
        return self.equilibrium.next_archives

    @property
    def next_ledger(self) -> ContemplationLedger:
        return self.equilibrium.next_ledger

    @property
    def next_chord_memory(self) -> ChordMemoryBank:
        return self.equilibrium.next_chord_memory

    @property
    def next_memory(self) -> ContinuousMemoryState:
        return self.equilibrium.next_memory

    @property
    def next_diary(self) -> SettledDiary:
        return self.equilibrium.next_diary

    @property
    def sibling_color_balances(self):
        return self.final_vibe.sibling_color_balances

    @property
    def shadow_pathology_warnings(self):
        return self.final_vibe.shadow_pathology_warnings

    @property
    def abstraction_profile(self) -> dict:
        from .philosophical_abstraction_matrix import inspect_vibe_abstraction_profile

        return inspect_vibe_abstraction_profile(self.final_vibe)


def _compile_mid_sentence_comma_vibe(
    current_vibe: RightBrainVibe,
    comma_vibe: RightBrainVibe,
) -> RightBrainVibe:
    """Combine pre-pass `current_vibe` with live mid-sentence `comma_vibe` at the comma before `<|subliminal_gap|>`.

    - External `Fact` (`coordinates`, `fact`) remains strictly locked to the outside scene.
    - `Theory` (`theory`, `theory_coordinates`) incorporates the live MiniLM reading of Clause 1 (`"bla bla bla,"`),
      including any active Shadow Pathology leaves triggered by Clause 1.
    - Any `dissonance_notes` or `synthesis.conflicts` triggered by Clause 1 at the comma immediately raise
      `dissonance_pressure` and `rupture_drive` inside the 52-float nerve cord.
    - While the sentence is only half spoken at the comma, `clarity` and `harmony` are capped at
      `min(current_vibe, comma_vibe)` so a neutral setup clause never prematurely relaxes the tug before Clause 2.
    """
    merged_dissonance = tuple(
        dict.fromkeys(
            list(current_vibe.self_registry.dissonance_notes)
            + list(comma_vibe.self_registry.dissonance_notes)
        )
    )
    merged_conflicts = tuple(
        dict.fromkeys(
            list(current_vibe.synthesis.conflicts)
            + list(comma_vibe.synthesis.conflicts)
        )
    )
    merged_harmony = round(
        min(float(current_vibe.synthesis.harmony), float(comma_vibe.synthesis.harmony)),
        6,
    )
    merged_clarity = round(
        min(float(current_vibe.clarity), float(comma_vibe.clarity)),
        6,
    )

    merged_theory_coords = dict(current_vibe.theory_coordinates)
    for tk, tv in comma_vibe.theory_coordinates.items():
        if tk not in merged_theory_coords or tk.startswith("philosophical/"):
            merged_theory_coords[tk] = max(float(merged_theory_coords.get(tk, 0.0)), float(tv))

    merged_theory = (
        0.0
        if merged_dissonance
        else round(max(float(current_vibe.theory), float(comma_vibe.theory)), 6)
    )
    merged_epistemic = (
        "SELF_RATIONALIZATION_DISSONANCE"
        if merged_dissonance
        else comma_vibe.self_registry.epistemic_state
    )
    merged_registry = replace(
        comma_vibe.self_registry,
        epistemic_state=merged_epistemic,
        dissonance_notes=merged_dissonance,
    )
    merged_synthesis = replace(
        comma_vibe.synthesis,
        harmony=merged_harmony,
        conflicts=merged_conflicts,
    )
    merged_tug_reasons = tuple(
        dict.fromkeys(
            list(current_vibe.tug_reasons)
            + list(comma_vibe.tug_reasons)
            + list(merged_dissonance)
        )
    )
    merged_posture = (
        "TUG_LEFT_BRAIN"
        if (merged_dissonance or current_vibe.posture == "TUG_LEFT_BRAIN" or comma_vibe.posture == "TUG_LEFT_BRAIN")
        else comma_vibe.posture
    )

    return replace(
        comma_vibe,
        posture=merged_posture,
        tug_reasons=merged_tug_reasons,
        theory_coordinates=merged_theory_coords,
        synthesis=merged_synthesis,
        self_registry=merged_registry,
        clarity=merged_clarity,
        theory=merged_theory,
    )


def compute_stanza_residual_chord(
    birth_state: BirthState,
    external_coords: Mapping[str, float],
    stanza_theory_coords: Mapping[str, float],
    stanza_vibe: RightBrainVibe,
    stanza_index: int = 1,
    *,
    deliberation: ThreeClassifierDeliberation | None = None,
    social_drift: SocialDriftReport | None = None,
    learned_leaves: Sequence[LearnedLeaf] = (),
) -> SubliminalChordVector:
    """Compute the next stanza's `SubliminalChordVector` from true coordinate deficits across the 24 Founding Leaves.

    - Zero fake dissonance spikes (`AGENTS.md` Rule 4): shifts departmental weights strictly
      proportionally to `deficit[k] = max(0.0, float(external_coords[k]) - float(stanza_theory_coords[k]))`
      and genuine `stanza_vibe.self_registry.dissonance_notes`.
    - Natural 3-Branch stanza rotation (`Legislative/Feeler -> Judicial/Thinker -> Executive/Do-er`)
      weighted by each branch's actual neglected coordinate deficit.
    """
    _ = birth_state
    base_chord = SubliminalChordVector.from_right_brain_vibe(
        stanza_vibe,
        deliberation=deliberation,
        social_drift=social_drift,
        learned_leaves=learned_leaves,
    )
    deficits: dict[str, float] = {}
    all_founding = (
        CULTURAL_FOUNDING_ORDER
        + PHILOSOPHICAL_FOUNDING_ORDER
        + EXPERIENTIAL_FOUNDING_ORDER
    )
    for k in all_founding:
        ext_val = float(external_coords.get(k, stanza_vibe.coordinates.get(k, 0.0)))
        th_val = float(stanza_theory_coords.get(k, 0.0))
        deficits[k] = max(0.0, ext_val - th_val)

    feeler_def = sum(deficits[k] for k in EXPERIENTIAL_FOUNDING_ORDER) / len(EXPERIENTIAL_FOUNDING_ORDER)
    thinker_def = sum(deficits[k] for k in PHILOSOPHICAL_FOUNDING_ORDER) / len(PHILOSOPHICAL_FOUNDING_ORDER)
    doer_def = sum(deficits[k] for k in CULTURAL_FOUNDING_ORDER) / len(CULTURAL_FOUNDING_ORDER)
    max_def = max(deficits.values(), default=0.0)

    if int(stanza_index) <= 0:
        feeler_w, thinker_w, doer_w = 0.65, 0.35, 0.25
    elif int(stanza_index) == 1:
        feeler_w, thinker_w, doer_w = 0.30, 0.70, 0.40
    else:
        feeler_w, thinker_w, doer_w = 0.25, 0.40, 0.75

    doer_list = list(base_chord.doer_vector)
    for idx, k in enumerate(CULTURAL_FOUNDING_ORDER):
        doer_list[idx] = round(min(1.0, max(0.0, doer_list[idx] + doer_w * deficits[k])), 6)
    if int(stanza_index) >= 2:
        action_pull = 0.55 * max(doer_def, thinker_def)
        doer_list[6] = round(min(1.0, max(0.0, doer_list[6] + action_pull)), 6)
        doer_list[7] = round(min(1.0, max(0.0, doer_list[7] + action_pull)), 6)

    thinker_list = list(base_chord.thinker_vector)
    for idx, k in enumerate(PHILOSOPHICAL_FOUNDING_ORDER):
        thinker_list[idx] = round(min(1.0, max(0.0, thinker_list[idx] + thinker_w * deficits[k])), 6)
    if int(stanza_index) == 1:
        thinker_list[7] = round(
            min(1.0, max(0.0, thinker_list[7] + 0.50 * thinker_def)),
            6,
        )

    feeler_list = list(base_chord.feeler_vector)
    for idx, k in enumerate(EXPERIENTIAL_FOUNDING_ORDER):
        feeler_list[idx] = round(min(1.0, max(0.0, feeler_list[idx] + feeler_w * deficits[k])), 6)

    cadence_list = list(base_chord.cadence_vector)
    if stanza_vibe.self_registry.dissonance_notes or max_def > 0.20:
        dissonance_shift = 0.25 * max_def + (0.12 if stanza_vibe.self_registry.dissonance_notes else 0.0)
        cadence_list[1] = round(min(1.0, max(0.0, cadence_list[1] + dissonance_shift)), 6)

    return SubliminalChordVector(
        doer_vector=tuple(doer_list),
        thinker_vector=tuple(thinker_list),
        feeler_vector=tuple(feeler_list),
        cadence_vector=tuple(cadence_list),
    )


def select_pre_gap_branch_chord_rollout(
    base_chord: SubliminalChordVector,
    external_coords: Mapping[str, float],
    rules: Sequence[FuzzyConceptRule],
    *,
    ratified_chords: Sequence[FuzzyConceptRule] = (),
) -> SubliminalChordVector:
    """Evaluate 3 branch-led candidate chords in pure coordinate space (< 2 ms, 0 LLM calls).

    Simulates Executive-led, Legislative-led, and Judicial-led constructive theory shifts against
    known attractive and repulsive rules, nudging the opening `<|subliminal_gap|>` chord toward
    the highest-confidence branch without ever overwriting Pass-1 `Fact` (`external_coords`).
    """
    all_rules = tuple(rules) + tuple(ratified_chords)
    if not all_rules:
        return base_chord

    def _score_candidate(cand_coords: Mapping[str, float]) -> float:
        att_scores: list[float] = []
        rep_scores: list[float] = []
        for r in all_rules:
            if not all(k in cand_coords for k in r.target_leaves):
                continue
            val = float(r.evaluate(cand_coords))
            if getattr(r, "polarity", "attractive") == "repulsive":
                rep_scores.append(val)
            else:
                att_scores.append(val)
        att_sorted = sorted(att_scores, reverse=True)
        max_att = att_sorted[0] if att_sorted else 0.0
        second_att = att_sorted[1] if len(att_sorted) > 1 else 0.0
        clarity_att = max(0.0, max_att - second_att)
        max_rep = max(rep_scores, default=0.0)
        return round(max_att + 0.35 * clarity_att - 0.65 * max_rep, 6)

    base_score = _score_candidate(external_coords)
    branch_lifts: dict[str, tuple[str, ...]] = {
        "executive": (
            "cultural/justice/fairness",
            "cultural/efficiency/wastelessness",
            "cultural/freedom/self-sufficiency",
        ),
        "legislative": (
            "cultural/freedom/needlessness",
            "philosophical/being/existence",
        ),
        "judicial": (
            "philosophical/structure/order",
            "philosophical/persistence/continuity",
            "philosophical/being/agency",
        ),
    }
    best_branch: str | None = None
    best_score = base_score
    for branch_name, lift_keys in branch_lifts.items():
        cand = dict(external_coords)
        for k in lift_keys:
            cand[k] = min(1.0, round(float(cand.get(k, 0.0)) + 0.18, 4))
        sc = _score_candidate(cand)
        if sc > best_score + 0.02:
            best_score = sc
            best_branch = branch_name

    if best_branch is None:
        return base_chord

    doer_list = list(base_chord.doer_vector)
    thinker_list = list(base_chord.thinker_vector)
    feeler_list = list(base_chord.feeler_vector)
    for k in branch_lifts[best_branch]:
        if k in CULTURAL_FOUNDING_ORDER:
            idx = CULTURAL_FOUNDING_ORDER.index(k)
            doer_list[idx] = round(min(1.0, doer_list[idx] + 0.12), 6)
        elif k in PHILOSOPHICAL_FOUNDING_ORDER:
            idx = PHILOSOPHICAL_FOUNDING_ORDER.index(k)
            thinker_list[idx] = round(min(1.0, thinker_list[idx] + 0.12), 6)
    if best_branch == "legislative" and len(feeler_list) >= 3:
        feeler_list[2] = round(min(1.0, feeler_list[2] + 0.10), 6)

    return SubliminalChordVector(
        doer_vector=tuple(doer_list),
        thinker_vector=tuple(thinker_list),
        feeler_vector=tuple(feeler_list),
        cadence_vector=base_chord.cadence_vector,
    )


def run_three_legged_turn(
    birth_state: BirthState,
    scenario_text: str,
    sensors: ThreeBridgeSensors,
    rules: tuple[FuzzyConceptRule, ...],
    *,
    scene_index: int = 1,
    subliminal_left_brain: SubliminalGapLeftBrain | None = None,
    clause_1_prefix: str | None = None,
    learned_leaves: Sequence[LearnedLeaf] = (),
    prior_memory: ContinuousMemoryState | None = None,
    diary: SettledDiary | None = None,
    chord_memory: ChordMemoryBank | None = None,
    contemplation_ledger: ContemplationLedger | None = None,
    branch_archives: ThreeBranchArchives | None = None,
    raw_thought_stream_path: str | Path | None = None,
    contemplation_ledger_path: str | Path | None = None,
    branch_archives_dir: str | Path | None = None,
    require_self_critique: bool = False,
    max_passes: int = 3,
    equilibrium_tolerance: float = DEFAULT_EQUILIBRIUM_TOLERANCE,
    branch_gating: bool = True,
    sanctuary_target_mode: str | None = None,
    sanctuary_lead_alpha: float = 0.25,
    force_spoken_reply: bool = False,
    num_sentences: int = 1,
    max_stanzas: int = 1,
    sentences_per_stanza: int = 3,
    comma_chord_transform: Callable[[SubliminalChordVector], SubliminalChordVector] | None = None,
    conversation_history: Sequence[tuple[str, str]] = (),
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    lead_with_gap: bool = False,
    will_edges: Sequence[Association] | None = None,
) -> ThreeLeggedTurnResult:
    """Execute one 3-Gear Closed-Loop Dual-Hemisphere Runtime Turn.

    - **Gear 1 (`GEAR_1_FAST_RIGHT_BRAIN`)**: Scores external reality (`Fact` locked) and
      checks fast reflex + 3-branch coordinate deliberation (`0` LLM calls when settled).
    - **Gear 2 (`GEAR_2_SUBLIMINAL_GAP_SPLICE`)**: When a tug or open thread needs the Left
      Brain, generates a 2-clause thought with `SubliminalChordVector` spliced into
      `<|subliminal_gap|>` (zero English stage directions in the prompt) and re-scores
      the thought back into `Truth / Theory`.
    - **Gear 3 (`GEAR_3_UNRESOLVED_QUEUED_FOR_SLEEP` when unsettled)**: Updates memory,
      ledger, and archives (only harvesting when `settled=True`), and exports an
      `UnresolvedSleepScene` whenever `settled=False`.
    """
    if type(max_passes) is not int or not (1 <= max_passes <= 10):
        raise ValueError("max_passes must be an int in [1, 10]")

    learned_tuple = tuple(learned_leaves)
    active_diary = diary if diary is not None else SettledDiary()
    active_chord_memory = chord_memory if chord_memory is not None else ChordMemoryBank()
    active_ledger = (
        contemplation_ledger if contemplation_ledger is not None else ContemplationLedger()
    )
    active_archives = branch_archives if branch_archives is not None else ThreeBranchArchives()
    active_will_edges = tuple(will_edges) if will_edges is not None else RIGHT_BRAIN_WILL_EDGES

    # =========================================================================
    # GEAR 1: Fast Right-Brain Reflex & 3-Branch Deliberation (Fact Locked)
    # =========================================================================
    external_coords, external_exp, queried_1, skipped_1 = sensors.score_text(
        scenario_text,
        learned_leaves=learned_tuple,
        branch_gating=branch_gating,
    )
    social_drift: SocialDriftReport = active_archives.executive.detect_social_drift(external_coords)

    vibe_1, next_mem = compile_right_brain_vibe(
        birth_state,
        external_coords,
        external_exp,
        rules,
        pass_index=1,
        event_id=f"scene-{scene_index}-pass-1",
        prior_memory=prior_memory,
        learned_leaves=learned_tuple,
        internal_theory_coords=None,
        internal_theory_experiential=None,
        queried_child_leaves=queried_1,
        skipped_child_leaves=skipped_1,
        social_drift=social_drift,
        ratified_chords=active_archives.judicial.ratified_chords,
        will_edges=active_will_edges,
    )

    initial_felt_chord = vibe_1.felt_chord or compile_felt_chord(
        external_coords,
        salience=vibe_1.salience,
        valence=vibe_1.valence,
        arousal=vibe_1.arousal,
        resonance=vibe_1.resonance,
        learned_leaves=learned_tuple,
    )
    chord_echoes = retrieve_chord_echoes(
        external_coords,
        initial_felt_chord,
        active_chord_memory,
        current_scene_index=scene_index,
    )
    open_thread = check_open_thread_trigger(
        active_ledger,
        chord_echoes,
        initial_felt_chord,
        external_coords,
        social_drift=social_drift,
    )

    traces: list[EquilibriumPassTrace] = [EquilibriumPassTrace(pass_index=1, vibe=vibe_1)]
    current_vibe = vibe_1
    prev_theory_coords: dict[str, float] = {}
    left_calls = 0
    latest_reply: str | None = None
    last_hail = None
    raw_thoughts: list[RawThoughtRecord] = []
    spliced_traces: list[SplicedThoughtTrace] = []
    deliberation: ThreeClassifierDeliberation | None = None
    audience_mode = "PASS1_REFLEX_OR_EASE"
    gear_used = "GEAR_1_FAST_RIGHT_BRAIN"

    should_speak = bool(
        force_spoken_reply
        or sanctuary_target_mode is not None
        or comma_chord_transform is not None
        or int(num_sentences) > 1
        or int(max_stanzas) > 1
    )
    if current_vibe.posture == "TUG_LEFT_BRAIN" or open_thread.triggered or should_speak:
        # Step 1B: Right-Brain 3-Branch Coordinate Deliberation (~15-30 ms, 0 LLM calls)
        deliberation = deliberate_across_three_branches(
            birth_state,
            external_coords,
            external_exp,
            initial_felt_chord,
            sensors,
            active_archives,
            active_will_edges,
            RIGHT_BRAIN_WILL_VERSION,
            scene_index=scene_index,
            learned_leaves=learned_tuple,
            social_drift=social_drift,
            branch_gating=branch_gating,
        )

        if (
            deliberation.ratified
            and not should_speak
            and (not open_thread.triggered or subliminal_left_brain is None)
            and max_passes >= 2
        ):
            audience_mode = "RIGHT_BRAIN_3_CLASSIFIER_DELIBERATION"
            active_ratified = (
                (deliberation.newly_ratified_chord,)
                if deliberation.newly_ratified_chord is not None
                else active_archives.judicial.ratified_chords
            )
            vibe_delib, next_mem = compile_right_brain_vibe(
                birth_state,
                external_coords,
                external_exp,
                rules,
                pass_index=2,
                event_id=f"scene-{scene_index}-pass-2-delib",
                prior_memory=prior_memory,
                learned_leaves=learned_tuple,
                internal_theory_coords=deliberation.bid_coordinates,
                internal_theory_experiential=deliberation.bid_experiential,
                queried_child_leaves=queried_1,
                skipped_child_leaves=skipped_1,
                social_drift=social_drift,
                ratified_chords=active_ratified,
                will_edges=active_will_edges,
            )
            max_delta_delib = round(
                max(
                    (
                        abs(
                            float(deliberation.bid_coordinates.get(k, 0.0))
                            - float(external_coords.get(k, 0.0))
                        )
                        for k in deliberation.bid_coordinates
                    ),
                    default=0.0,
                ),
                6,
            )
            traces.append(
                EquilibriumPassTrace(
                    pass_index=2,
                    vibe=vibe_delib,
                    max_theory_delta=max_delta_delib,
                )
            )
            current_vibe = vibe_delib

        # =====================================================================
        # GEAR 2: Live `<|subliminal_gap|>` Spliced Contemplation
        # =====================================================================
        elif subliminal_left_brain is not None and max_passes >= 2:
            gear_used = "GEAR_2_SUBLIMINAL_GAP_SPLICE"
            audience_mode = (
                "LEFT_BRAIN_SUBLIMINAL_GAP_OPEN_THREAD"
                if open_thread.triggered
                else "LEFT_BRAIN_SUBLIMINAL_GAP_SPLICE"
            )
            pass_ratified_chords = (
                (deliberation.newly_ratified_chord,)
                if (deliberation is not None and deliberation.newly_ratified_chord is not None)
                else active_archives.judicial.ratified_chords
            )
            cached_clause_1_prefix: str | None = clause_1_prefix
            cached_comma_scores: dict[str, tuple[dict[str, float], dict[str, float]]] = {}

            for p in range(2, max_passes + 1):
                prior_hunch_for_pass = latest_reply
                pre_pass_margin = _make_margin_tag_for_vibe(
                    current_vibe,
                    prior_hunch_exists=prior_hunch_for_pass is not None,
                )
                # Build TacticalHail for telemetry/audit only—NEVER passed into the LLM prompt!
                hail = build_tactical_hail(
                    current_vibe,
                    chord_echoes=chord_echoes,
                    contemplation_ledger=active_ledger,
                    prior_self_talk_hunch=prior_hunch_for_pass,
                    subliminal_margin_tag=pre_pass_margin,
                )
                last_hail = hail

                def _build_chord_at_comma(
                    comma_prefix: str,
                    *,
                    _pass_idx: int = p,
                    _base_vibe: RightBrainVibe = current_vibe,
                ) -> tuple[SubliminalChordVector, str | None, tuple[str, ...]]:
                    if not comma_prefix.strip():
                        mid_comma_vibe = _base_vibe
                    else:
                        if comma_prefix in cached_comma_scores:
                            comma_coords, comma_exp = cached_comma_scores[comma_prefix]
                        else:
                            comma_coords, comma_exp, _q_c, _s_c = sensors.score_text(
                                comma_prefix,
                                learned_leaves=learned_tuple,
                                branch_gating=branch_gating,
                            )
                            cached_comma_scores[comma_prefix] = (dict(comma_coords), dict(comma_exp))
                        raw_comma_vibe, _ = compile_right_brain_vibe(
                            birth_state,
                            external_coords,
                            external_exp,
                            rules,
                            pass_index=_pass_idx,
                            event_id=f"scene-{scene_index}-pass-{_pass_idx}-comma",
                            prior_memory=prior_memory,
                            learned_leaves=learned_tuple,
                            internal_theory_coords=comma_coords,
                            internal_theory_experiential=comma_exp,
                            queried_child_leaves=queried_1,
                            skipped_child_leaves=skipped_1,
                            social_drift=social_drift,
                            ratified_chords=pass_ratified_chords,
                            will_edges=active_will_edges,
                        )
                        if comma_prefix.rstrip().endswith((".", "!", "?")):
                            mid_comma_vibe = raw_comma_vibe
                        else:
                            mid_comma_vibe = _compile_mid_sentence_comma_vibe(_base_vibe, raw_comma_vibe)
                    if _pass_idx >= 3 and prev_theory_coords:
                        chord_at_comma = compute_stanza_residual_chord(
                            birth_state,
                            external_coords,
                            prev_theory_coords,
                            mid_comma_vibe,
                            stanza_index=_pass_idx - 2,
                            deliberation=deliberation,
                            social_drift=social_drift,
                            learned_leaves=learned_tuple,
                        )
                    else:
                        chord_at_comma = SubliminalChordVector.from_right_brain_vibe(
                            mid_comma_vibe,
                            deliberation=deliberation,
                            social_drift=social_drift,
                            learned_leaves=learned_tuple,
                        )
                        if _pass_idx == 2 and mid_comma_vibe.posture == "TUG_LEFT_BRAIN":
                            chord_at_comma = select_pre_gap_branch_chord_rollout(
                                chord_at_comma,
                                external_coords,
                                rules,
                                ratified_chords=pass_ratified_chords,
                            )
                    if sanctuary_target_mode is not None:
                        from .chess_kibitz_bridge import apply_coregulation_lead_to_chord

                        chord_at_comma = apply_coregulation_lead_to_chord(
                            chord_at_comma,
                            sanctuary_target_mode,
                            alpha=sanctuary_lead_alpha,
                        )
                    if comma_chord_transform is not None:
                        chord_at_comma = comma_chord_transform(chord_at_comma)
                    return (
                        chord_at_comma,
                        mid_comma_vibe.posture,
                        mid_comma_vibe.self_registry.dissonance_notes,
                    )

                stanza_scored_records: list[tuple[dict[str, float], dict[str, float]]] = []
                if int(max_stanzas) > 1:
                    prev_stanza_theory_coords: dict[str, float] = {}

                    def _stanza_boundary_callback(
                        stanza_idx: int,
                        stanza_text: str,
                        cumulative_text: str,
                        *,
                        _pass_idx: int = p,
                    ) -> tuple[bool, SubliminalChordVector | None]:
                        nonlocal prev_stanza_theory_coords
                        eval_text = stanza_text.strip() or cumulative_text.strip()
                        st_coords, st_exp, _, _ = sensors.score_text(
                            eval_text,
                            learned_leaves=learned_tuple,
                            branch_gating=branch_gating,
                        )
                        stanza_scored_records.append((dict(st_coords), dict(st_exp)))
                        st_vibe, _ = compile_right_brain_vibe(
                            birth_state,
                            external_coords,
                            external_exp,
                            rules,
                            pass_index=_pass_idx,
                            event_id=f"scene-{scene_index}-pass-{_pass_idx}-stanza-{stanza_idx + 1}",
                            prior_memory=prior_memory,
                            learned_leaves=learned_tuple,
                            internal_theory_coords=st_coords,
                            internal_theory_experiential=st_exp,
                            queried_child_leaves=queried_1,
                            skipped_child_leaves=skipped_1,
                            social_drift=social_drift,
                            ratified_chords=pass_ratified_chords,
                            will_edges=active_will_edges,
                        )
                        if (
                            st_vibe.posture in ("SETTLED_EASE", "AT_EASE", "WORDLESS_REFLEX", "THEORY_SYNTHESIS")
                            and float(st_vibe.clarity) >= 0.68
                            and not st_vibe.self_registry.dissonance_notes
                        ):
                            return True, None
                        if stanza_idx >= 1 and prev_stanza_theory_coords:
                            stanza_delta = max(
                                (
                                    abs(float(st_coords.get(k, 0.0)) - float(prev_stanza_theory_coords.get(k, 0.0)))
                                    for k in set(st_coords) | set(prev_stanza_theory_coords)
                                ),
                                default=0.0,
                            )
                            if stanza_delta < 0.04:
                                return True, None
                        if stanza_idx + 1 >= int(max_stanzas):
                            return True, None
                        prev_stanza_theory_coords = dict(st_coords)
                        next_chord = compute_stanza_residual_chord(
                            birth_state,
                            external_coords,
                            st_coords,
                            st_vibe,
                            stanza_index=stanza_idx + 1,
                            deliberation=deliberation,
                            social_drift=social_drift,
                            learned_leaves=learned_tuple,
                        )
                        if sanctuary_target_mode is not None:
                            from .chess_kibitz_bridge import apply_coregulation_lead_to_chord

                            next_chord = apply_coregulation_lead_to_chord(
                                next_chord,
                                sanctuary_target_mode,
                                alpha=sanctuary_lead_alpha,
                            )
                        if comma_chord_transform is not None:
                            next_chord = comma_chord_transform(next_chord)
                        return False, next_chord

                    multi_text, _stanza_list, multi_traces = subliminal_left_brain.generate_multi_stanza_discourse(
                        scenario_text,
                        max_stanzas=int(max_stanzas),
                        sentences_per_stanza=int(sentences_per_stanza),
                        pass_index=p,
                        clause_1_prefix=cached_clause_1_prefix,
                        conversation_history=conversation_history,
                        max_context_chars=max_context_chars,
                        chord_builder_at_comma=_build_chord_at_comma,
                        stanza_boundary_callback=_stanza_boundary_callback,
                        lead_with_gap=lead_with_gap,
                    )
                    if len(stanza_scored_records) < len(_stanza_list):
                        for extra_st in _stanza_list[len(stanza_scored_records) :]:
                            if extra_st.strip():
                                ex_c, ex_e, _, _ = sensors.score_text(
                                    extra_st.strip(),
                                    learned_leaves=learned_tuple,
                                    branch_gating=branch_gating,
                                )
                                stanza_scored_records.append((dict(ex_c), dict(ex_e)))
                    if clause_1_prefix is None and multi_traces and not lead_with_gap:
                        cached_clause_1_prefix = multi_traces[0].clause_1_prefix
                    spliced_traces.extend(multi_traces)
                    reply_text = validate_plain_english_left_brain_reply(multi_text)
                elif int(num_sentences) > 1:
                    multi_text, multi_traces = subliminal_left_brain.generate_multi_sentence_script(
                        scenario_text,
                        num_sentences=int(num_sentences),
                        pass_index=p,
                        clause_1_prefix=cached_clause_1_prefix,
                        conversation_history=conversation_history,
                        max_context_chars=max_context_chars,
                        chord_builder_at_comma=_build_chord_at_comma,
                        lead_with_gap=lead_with_gap,
                    )
                    if clause_1_prefix is None and multi_traces and not lead_with_gap:
                        cached_clause_1_prefix = multi_traces[0].clause_1_prefix
                    spliced_traces.extend(multi_traces)
                    reply_text = validate_plain_english_left_brain_reply(multi_text)
                else:
                    spliced_trace = subliminal_left_brain.generate_two_clause_thought(
                        scenario_text,
                        pass_index=p,
                        clause_1_prefix=cached_clause_1_prefix,
                        conversation_history=conversation_history,
                        max_context_chars=max_context_chars,
                        chord_builder_at_comma=_build_chord_at_comma,
                        lead_with_gap=lead_with_gap,
                    )
                    if clause_1_prefix is None and not lead_with_gap:
                        cached_clause_1_prefix = spliced_trace.clause_1_prefix
                    spliced_traces.append(spliced_trace)
                    reply_text = validate_plain_english_left_brain_reply(spliced_trace.full_thought)
                left_calls += 1
                latest_reply = reply_text

                # Step 2d: Score arrived Truth / Theory while keeping external Fact (external_coords) immutable!
                # When multi-stanza discourse generated >= 2 stanzas, apply Culminating Synthesis Weighting
                # (0.70 * final_stanza + 0.30 * mean_prior_stanzas) so Stanza 1's diagnostic setup never
                # dilutes Stanza 2/3's arrived resolution.
                if len(stanza_scored_records) >= 2:
                    final_coords, final_exp = stanza_scored_records[-1]
                    prior_records = stanza_scored_records[:-1]
                    n_prior = float(len(prior_records))
                    all_coord_keys = set(final_coords)
                    for pr_c, _ in prior_records:
                        all_coord_keys.update(pr_c)
                    theory_coords = {
                        k: round(
                            0.70 * float(final_coords.get(k, 0.0))
                            + 0.30 * (sum(float(pr_c.get(k, 0.0)) for pr_c, _ in prior_records) / n_prior),
                            6,
                        )
                        for k in all_coord_keys
                    }
                    all_exp_keys = set(final_exp)
                    for _, pr_e in prior_records:
                        all_exp_keys.update(pr_e)
                    theory_exp = {
                        k: round(
                            0.70 * float(final_exp.get(k, 0.0))
                            + 0.30 * (sum(float(pr_e.get(k, 0.0)) for _, pr_e in prior_records) / n_prior),
                            6,
                        )
                        for k in all_exp_keys
                    }
                elif len(stanza_scored_records) == 1:
                    theory_coords = dict(stanza_scored_records[0][0])
                    theory_exp = dict(stanza_scored_records[0][1])
                else:
                    theory_coords, theory_exp, _q_p, _s_p = sensors.score_text(
                        reply_text,
                        learned_leaves=learned_tuple,
                        branch_gating=branch_gating,
                    )

                vibe_p, next_mem = compile_right_brain_vibe(
                    birth_state,
                    external_coords,
                    external_exp,
                    rules,
                    pass_index=p,
                    event_id=f"scene-{scene_index}-pass-{p}",
                    prior_memory=prior_memory,
                    learned_leaves=learned_tuple,
                    internal_theory_coords=theory_coords,
                    internal_theory_experiential=theory_exp,
                    queried_child_leaves=queried_1,
                    skipped_child_leaves=skipped_1,
                    social_drift=social_drift,
                    ratified_chords=pass_ratified_chords,
                    will_edges=active_will_edges,
                )

                if prev_theory_coords:
                    max_delta = round(
                        max(
                            (
                                abs(float(theory_coords.get(k, 0.0)) - float(prev_theory_coords.get(k, 0.0)))
                                for k in theory_coords
                            ),
                            default=0.0,
                        ),
                        6,
                    )
                else:
                    max_delta = round(
                        max(
                            (
                                abs(float(theory_coords.get(k, 0.0)) - float(external_coords.get(k, 0.0)))
                                for k in theory_coords
                            ),
                            default=0.0,
                        ),
                        6,
                    )

                must_continue_for_critique = (
                    (require_self_critique or open_thread.triggered) and p == 2 and max_passes >= 3
                )
                settled_this_pass = (
                    vibe_p.posture in ("AT_EASE", "WORDLESS_REFLEX", "THEORY_SYNTHESIS")
                    and not must_continue_for_critique
                )
                post_pass_fc = vibe_p.felt_chord or initial_felt_chord
                post_pass_margin = _make_margin_tag_for_vibe(vibe_p, prior_hunch_exists=True)
                thought_record = RawThoughtRecord(
                    scene_index=scene_index,
                    pass_index=p,
                    clarity=vibe_p.clarity,
                    confidence=vibe_p.confidence,
                    raw_self_talk_draft=reply_text,
                    prior_hunch_debated=prior_hunch_for_pass,
                    felt_chord_stamp=post_pass_fc.short_stamp(),
                    subliminal_margin_tag=post_pass_margin,
                    epistemic_state=vibe_p.self_registry.epistemic_state,
                    settled_on_this_pass=settled_this_pass,
                )
                raw_thoughts.append(thought_record)
                if raw_thought_stream_path is not None:
                    append_raw_thought_record(raw_thought_stream_path, thought_record)

                if not settled_this_pass:
                    active_ledger = active_ledger.record_uncertainty(
                        scene_index=scene_index,
                        raw_hunch=reply_text,
                        open_question=post_pass_margin.red_pen_prompt,
                        margin_tag=post_pass_margin,
                    )

                traces[-1] = EquilibriumPassTrace(
                    pass_index=traces[-1].pass_index,
                    vibe=traces[-1].vibe,
                    left_brain_reply=reply_text,
                    max_theory_delta=traces[-1].max_theory_delta,
                    tactical_hail=hail,
                )
                traces.append(
                    EquilibriumPassTrace(
                        pass_index=p,
                        vibe=vibe_p,
                        max_theory_delta=max_delta,
                    )
                )
                current_vibe = vibe_p
                prev_theory_coords = dict(theory_coords)

                if settled_this_pass:
                    break
                if p >= 3 and max_delta <= equilibrium_tolerance:
                    break

    # =========================================================================
    # GEAR 3: Memory Compaction, Archive Harvesting & Unresolved Sleep Queueing
    # =========================================================================
    final_fc = current_vibe.felt_chord or initial_felt_chord
    unresolved_sleep_scene: UnresolvedSleepScene | None = None

    if current_vibe.posture in ("AT_EASE", "WORDLESS_REFLEX", "THEORY_SYNTHESIS"):
        settled = True
        outcome = current_vibe.posture
        needs_new_instinct = False
        resolved_scenes = (
            (open_thread.target_scene_index,)
            if (open_thread.triggered and open_thread.target_scene_index is not None)
            else ()
        )
        takeaway_src = (
            latest_reply
            if latest_reply is not None
            else (
                deliberation.candidate_bid_text
                if (deliberation is not None and deliberation.ratified and deliberation.candidate_bid_text)
                else scenario_text
            )
        )
        active_ledger = active_ledger.record_settled(
            scene_index=scene_index,
            takeaway_source=takeaway_src,
            felt_chord=final_fc,
            clarity=current_vibe.clarity,
            confidence=current_vibe.confidence,
            resolved_uncertainty_scenes=resolved_scenes,
        )
    else:
        settled = False
        outcome = "UNRESOLVED_TENSION"
        needs_new_instinct = True
        gear_used = "GEAR_3_UNRESOLVED_QUEUED_FOR_SLEEP"
        final_margin = _make_margin_tag_for_vibe(
            current_vibe,
            prior_hunch_exists=latest_reply is not None,
        )
        hunch_text = (
            latest_reply
            if latest_reply is not None
            else f"Unresolved scene tension: {scenario_text}"
        )
        active_ledger = active_ledger.record_uncertainty(
            scene_index=scene_index,
            raw_hunch=hunch_text,
            open_question=final_margin.red_pen_prompt,
            margin_tag=final_margin,
        )
        unresolved_sleep_scene = UnresolvedSleepScene(
            scene_index=scene_index,
            scenario_text=scenario_text,
            external_coordinates=dict(current_vibe.coordinates),
            felt_chord_notes=dict(final_fc.complex_notes),
            valence=current_vibe.valence,
            salience=current_vibe.salience,
            arousal=current_vibe.arousal,
            left_brain_hunch=hunch_text,
            social_drift_detected=bool(social_drift.drift_detected),
            active_concept_collision=current_vibe.active_concept,
        )

    if contemplation_ledger_path is not None:
        save_contemplation_ledger(contemplation_ledger_path, active_ledger)

    next_archives = harvest_into_branch_archives(
        active_archives,
        scene_index=scene_index,
        scenario_text=scenario_text,
        coordinates=current_vibe.coordinates,
        felt_chord=final_fc,
        left_brain_reply=latest_reply,
        settled=settled,
        deliberation=deliberation,
        social_drift=social_drift,
    )
    if branch_archives_dir is not None:
        next_archives.save_all(branch_archives_dir)

    updated_chord_memory = active_chord_memory.append(
        scene_index=scene_index,
        scene_text_or_gist=scenario_text,
        coordinates=current_vibe.coordinates,
        felt_chord=final_fc,
        settled=settled,
    )

    diary_entry = DiaryEntry(
        scene_index=scene_index,
        outcome=outcome,
        active_concept=current_vibe.active_concept,
        key_coordinates=_top_key_coordinates(current_vibe.coordinates),
        conscious_note=latest_reply,
        epistemic_state=current_vibe.self_registry.epistemic_state,
    )
    updated_diary = active_diary.append(diary_entry)

    equilibrium = EquilibriumResult(
        scene_index=scene_index,
        scenario_text=scenario_text,
        settled=settled,
        outcome=outcome,
        final_vibe=current_vibe,
        passes=tuple(traces),
        left_brain_calls=left_calls,
        final_conscious_note=latest_reply,
        needs_new_instinct=needs_new_instinct,
        next_memory=next_mem,
        next_diary=updated_diary,
        felt_chord=final_fc,
        chord_echoes=chord_echoes,
        last_tactical_hail=last_hail,
        raw_thoughts=tuple(raw_thoughts),
        next_chord_memory=updated_chord_memory,
        next_ledger=active_ledger,
        deliberation=deliberation,
        social_drift=social_drift,
        open_thread=open_thread,
        audience_mode=audience_mode,
        next_archives=next_archives,
    )
    return ThreeLeggedTurnResult(
        equilibrium=equilibrium,
        gear_used=gear_used,
        spliced_traces=tuple(spliced_traces),
        unresolved_sleep_scene=unresolved_sleep_scene,
    )
