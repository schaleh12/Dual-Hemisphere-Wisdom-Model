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

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import re

import torch
from torch import Tensor, nn

from .birth import BirthState
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
)

DEFAULT_QWEN_4B_HF_ID = "Qwen/Qwen3-4B"

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


def normalize_clause_1_prefix(raw_prefix: str) -> str:
    """Clean an opening clause so it stops at a comma boundary before `<|subliminal_gap|>`."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_prefix, flags=re.DOTALL)
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

    # Replace any premature sentence-ending period/exclamation/question mark with a comma
    cleaned = cleaned.rstrip(".!?;:—- ").strip()
    if not cleaned:
        return "Weighing the situation before us,"
    return f"{cleaned},"


@dataclass(frozen=True)
class SplicedThoughtTrace:
    """Record of one 2-clause Left-Brain thought generated across `<|subliminal_gap|>`."""

    pass_index: int
    clause_1_prefix: str
    prompt_before_gap: str
    spliced_chord: SubliminalChordVector
    post_gap_continuation: str
    full_thought: str

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
        }


class SubliminalGapLeftBrain:
    """PyTorch Left Brain (`Qwen/Qwen3-4B` + LoRA + `SubliminalGapProjector`) steered strictly via `<|subliminal_gap|>`."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        projector: SubliminalGapProjector,
        *,
        gap_token_id: int | None = None,
        device: str | torch.device = "cpu",
        max_clause_1_tokens: int = 14,
        max_post_gap_tokens: int = 40,
        callosum_metadata: CorpusCallosumMetadata | None = None,
        lora_config: LoRAConfig | None = None,
    ) -> None:
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

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        *,
        hf_model_id: str = DEFAULT_QWEN_4B_HF_ID,
        device: str | torch.device | None = None,
        torch_dtype: torch.dtype | None = None,
        max_clause_1_tokens: int = 14,
        max_post_gap_tokens: int = 40,
    ) -> "SubliminalGapLeftBrain":
        """Load `Qwen/Qwen3-4B` + trained `LoRALinear` adapters + `SubliminalGapProjector` from disk."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

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

        tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        gap_token_id = ensure_subliminal_gap_token(tokenizer)
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            dtype=resolved_dtype,
        )
        if len(tokenizer) > model.get_input_embeddings().num_embeddings:
            model.resize_token_embeddings(len(tokenizer))
        model.to(resolved_device)
        projector, meta, lora_cfg = load_lora_and_callosum_checkpoint(
            checkpoint_dir,
            model=model,
            device=resolved_device,
        )
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
        if hasattr(self.tokenizer, "encode"):
            for tag in ("<think>", "</think>", "<think>\n", "</think>\n"):
                ids = self.tokenizer.encode(tag, add_special_tokens=False)
                if len(ids) == 1 and ids[0] not in suppressed:
                    suppressed.append(int(ids[0]))
        return suppressed

    @torch.no_grad()
    def generate_clause_1_prefix(
        self,
        scenario_text: str,
        *,
        clause_1_override: str | None = None,
    ) -> str:
        """Step 2a: Generate Clause 1 from `"User: {scenario_text}\\nAssistant:"` and stop at the first comma."""
        if clause_1_override is not None and clause_1_override.strip():
            return normalize_clause_1_prefix(clause_1_override)

        was_training = self.model.training
        self.model.eval()
        try:
            prompt = f"User: {scenario_text.strip()}\nAssistant:"
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
            if not prompt_ids:
                return "Weighing the situation before us,"

            embed_layer = self.model.get_input_embeddings()
            embed_dtype = next(embed_layer.parameters()).dtype
            cur_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
            cur_embeds = embed_layer(cur_ids).to(dtype=embed_dtype)
            cur_mask = torch.ones((1, cur_embeds.size(1)), dtype=torch.long, device=self.device)

            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            suppressed_ids = self._suppressed_token_ids()
            generated_ids: list[int] = []

            for step_idx in range(self.max_clause_1_tokens):
                out = self.model(
                    inputs_embeds=cur_embeds,
                    attention_mask=cur_mask,
                    use_cache=False,
                )
                next_logits = out.logits[0, -1, :].clone()
                for bad_id in suppressed_ids:
                    if 0 <= bad_id < next_logits.size(0):
                        next_logits[bad_id] = -1e9
                if step_idx == 0 and eos_id is not None and 0 <= int(eos_id) < next_logits.size(0):
                    next_logits[int(eos_id)] = -1e9

                next_id = int(torch.argmax(next_logits).item())
                if eos_id is not None and next_id == int(eos_id):
                    break
                generated_ids.append(next_id)

                decoded_so_far = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                # Stop as soon as a comma or semicolon appears after at least 3 generated tokens
                if step_idx >= 2 and ("," in decoded_so_far or ";" in decoded_so_far):
                    break

                next_tok_tensor = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
                next_embed = embed_layer(next_tok_tensor).to(dtype=embed_dtype)
                cur_embeds = torch.cat([cur_embeds, next_embed], dim=1)
                cur_mask = torch.cat(
                    [cur_mask, torch.ones((1, 1), dtype=torch.long, device=self.device)],
                    dim=1,
                )

            raw_clause_1 = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return normalize_clause_1_prefix(raw_clause_1)
        finally:
            if was_training:
                self.model.train()

    @torch.no_grad()
    def generate_two_clause_thought(
        self,
        scenario_text: str,
        chord: SubliminalChordVector,
        *,
        pass_index: int = 2,
        clause_1_prefix: str | None = None,
    ) -> SplicedThoughtTrace:
        """Step 2a-2c: Generate Clause 1, splice 52-float `chord` into `<|subliminal_gap|>`, and generate Clause 2."""
        was_model_training = self.model.training
        was_proj_training = self.projector.training
        self.model.eval()
        self.projector.eval()
        try:
            prefix = self.generate_clause_1_prefix(
                scenario_text,
                clause_1_override=clause_1_prefix,
            )
            prompt_before_gap = (
                f"User: {scenario_text.strip()}\n"
                f"Assistant: {prefix} {SUBLIMINAL_GAP_TOKEN}"
            )
            assert_zero_stage_directions_in_prompt(prompt_before_gap)

            pre_ids_list = self.tokenizer.encode(prompt_before_gap, add_special_tokens=True)
            if int(self.gap_token_id) not in pre_ids_list:
                pre_ids_list.append(int(self.gap_token_id))
            pre_ids = torch.tensor([pre_ids_list], dtype=torch.long, device=self.device)
            pre_mask = torch.ones_like(pre_ids)

            embed_layer = self.model.get_input_embeddings()
            embed_dtype = next(embed_layer.parameters()).dtype
            chord_batch = chord.to_tensor(device=self.device, dtype=torch.float32).unsqueeze(0)

            spliced = splice_subliminal_gaps_into_embeds(
                pre_ids,
                embed_layer,
                self.projector,
                chord_batch,
                gap_token_id=self.gap_token_id,
                attention_mask=pre_mask,
                labels=None,
            )
            cur_embeds = spliced.inputs_embeds.to(dtype=embed_dtype)
            cur_mask = spliced.attention_mask

            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            suppressed_ids = self._suppressed_token_ids()
            generated_ids: list[int] = []

            for step_idx in range(self.max_post_gap_tokens):
                out = self.model(
                    inputs_embeds=cur_embeds,
                    attention_mask=cur_mask,
                    use_cache=False,
                )
                next_logits = out.logits[0, -1, :].clone()
                for bad_id in suppressed_ids:
                    if 0 <= bad_id < next_logits.size(0):
                        next_logits[bad_id] = -1e9
                if step_idx == 0 and eos_id is not None and 0 <= int(eos_id) < next_logits.size(0):
                    next_logits[int(eos_id)] = -1e9

                next_id = int(torch.argmax(next_logits).item())
                if eos_id is not None and next_id == int(eos_id):
                    break
                generated_ids.append(next_id)

                decoded_cont = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                if step_idx >= 5 and decoded_cont.endswith((".", "!", "?")):
                    break

                next_tok_tensor = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
                next_embed = embed_layer(next_tok_tensor).to(dtype=embed_dtype)
                cur_embeds = torch.cat([cur_embeds, next_embed], dim=1)
                cur_mask = torch.cat(
                    [cur_mask, torch.ones((1, 1), dtype=torch.long, device=self.device)],
                    dim=1,
                )

            post_gap_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            post_gap_text = re.sub(r"<think>.*?</think>", "", post_gap_text, flags=re.DOTALL)
            post_gap_text = post_gap_text.replace("<think>", "").replace("</think>", "")
            post_gap_text = " ".join(post_gap_text.replace("\n", " ").split()).strip()
            if not post_gap_text:
                post_gap_text = "we hold steady and observe the unfolding tension."

            full_thought = f"{prefix} {post_gap_text}".strip()
            trace = SplicedThoughtTrace(
                pass_index=pass_index,
                clause_1_prefix=prefix,
                prompt_before_gap=prompt_before_gap,
                spliced_chord=chord,
                post_gap_continuation=post_gap_text,
                full_thought=full_thought,
            )
            self.last_traces.append(trace)
            return trace
        finally:
            if was_model_training:
                self.model.train()
            if was_proj_training:
                self.projector.train()


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

    if current_vibe.posture == "TUG_LEFT_BRAIN" or open_thread.triggered:
        # Step 1B: Right-Brain 3-Branch Coordinate Deliberation (~15-30 ms, 0 LLM calls)
        deliberation = deliberate_across_three_branches(
            birth_state,
            external_coords,
            external_exp,
            initial_felt_chord,
            sensors,
            active_archives,
            RIGHT_BRAIN_WILL_EDGES,
            RIGHT_BRAIN_WILL_VERSION,
            scene_index=scene_index,
            learned_leaves=learned_tuple,
            social_drift=social_drift,
            branch_gating=branch_gating,
        )

        if (
            deliberation.ratified
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

                chord_p = SubliminalChordVector.from_right_brain_vibe(
                    current_vibe,
                    deliberation=deliberation,
                    social_drift=social_drift,
                    learned_leaves=learned_tuple,
                )
                spliced_trace = subliminal_left_brain.generate_two_clause_thought(
                    scenario_text,
                    chord_p,
                    pass_index=p,
                    clause_1_prefix=clause_1_prefix,
                )
                spliced_traces.append(spliced_trace)
                reply_text = validate_plain_english_left_brain_reply(spliced_trace.full_thought)
                left_calls += 1
                latest_reply = reply_text

                # Step 2d: Re-score the completed 2-clause thought through the 3 MiniLM bridges
                # into Truth / Theory while keeping external Fact (external_coords) immutable!
                theory_coords, theory_exp, _q_p, _s_p = sensors.score_text(
                    reply_text,
                    learned_leaves=learned_tuple,
                    branch_gating=branch_gating,
                )

                pass_ratified_chords = (
                    (deliberation.newly_ratified_chord,)
                    if (deliberation is not None and deliberation.newly_ratified_chord is not None)
                    else active_archives.judicial.ratified_chords
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
