# LMFT-Architecture — Design Decisions (Rolling 5-Decision Window)

*(Keep only the 5 most recent decisions here, under 80 lines AND 5,000 characters in plain English. Earlier decisions `D001–D066` are archived in `LMFT-Architecture/docs/archive/`.)*

## D067: Semantic Birth Anchoring, Symmetric Updates & `will_edges` Scoring (`subliminal_gap_splice.py`, `sleep_consolidation.py`, `chess_app.py`, 2026-11-08)
- **Context**: Audit found 5 installer bugs, one-way ratchets, `0.25 / 0.53` stamps, a rigged `calm_baseline_score` gate, random static at `Cycle 0` birth, REM vow theater, and un-wired `will_edges` in move scoring.
- **Design Choice**: Fixed the installer bugs, replaced one-way ratchets with symmetric updates, removed coordinate stamps, replaced `calm_baseline_score` with `calm_deviation`, cut Dream Vow theater, wired `self.will_edges` into move scoring, and anchored `SubliminalGapProjector` (`256`-dim) to LLM input embeddings at Birth.
- **Result**: Raw holdout flips reached `6/6` on `Qwen3-8B` and `5/6` (`6/6` calibrated) on `Impish_Bloodmoon_12B` and `Qwen3-4B`.

## D068: Tier 2–4 Sub-Feature Leaf Genesis, Ply-5 Detective Walk & Two-Way Gear-2 Deliberation (`reflection_classifier.py`, `chess_kibitz_bridge.py`, `sleep_consolidation.py`, `chess_app.py`, 2026-11-09)
- **Context**: All 29 learned rules in `club_ruleset.json` stayed at Tier 1 while 82 Tier-2 `LearnedLeaf` nodes went unused because multi-ply traps shared nearly identical Tier-1 summary coordinates.
- **Design Choice**: Added a 16-float Sub-Sensor Bank (`compute_move_sub_sensors`), Tier 2–4 `LearnedLeaf` genesis with parent-gated (`parent >= 0.50`) sub-sensor projections and `Tier 4 > Tier 3 > Tier 2 > Tier 1` inhibition, the `Ply 1 -> Ply 3 -> Ply 5` Detective Walk, and two-way Gear-2 suspect reweighting.
- **Result**: Verified `48/48` passing tests across sleep, chess, subliminal gap, equilibrium, and Virgin baseline suites.

## D069: Universal Native 16-Bit `Impish_LLAMA_4B` Base Model & Full Clause-1 Supervision (`three_legged_runtime.py`, `train_left_brain_lora.py`, `chess_app.py`, `bloodmoon_app.py`, 2026-11-10)
- **Context**: `Qwen3` carried `<think>` tags and censorship filters, `Impish_Bloodmoon_12B` required 4-bit NF4 quantization, and `-100` pre-gap masking left Clause 1 un-trained up to the comma.
- **Design Choice**: Standardized on native 16-bit (`bfloat16`) `SicariusSicariiStuff/Impish_LLAMA_4B` (`hidden_size = 3072`, `32` layers), removed `-100` pre-gap masking in `build_spliced_training_forward`, and split checkpoints into `left_brain_baseline`, `left_brain_chess`, and `left_brain_bloodmoon`.
- **Result**: Achieved `6/6` raw holdout flips (`+4.4806` net swap margin, `6/6` greedy text changed) in `25s/epoch` at `11.92 GB` VRAM.

## D070: `LMFT-Scholar` 12-Book Seminar, Canopy Inspector & Graduation (`scholar_app.py`, `scholar_ui.py`, 2026-11-10)
- **Context**: Training directly inside chess or hypnosis specialized rules and the Callosum too early without general domain wisdom across the 12 founding boughs.
- **Design Choice**: Built `LMFT-Scholar` (`Launch-LMFT-Scholar.bat`, `models/left_brain_scholar/`, `outputs/scholar_state/`) with a 12-Book Literature Seminar (REM sleep after each essay), local Tier-2/3 canopy genesis (zero 9B Teacher reliance), a Visual Tree & Canopy Inspector, and `graduate_brain_to_target`.
- **Result**: Verified passing tests in `test_scholar_app.py` and full compatibility across `LMFT-Chess` and `LMFT-Bloodmoon`.

## D071: Norm-Locked ChatML Callosum, `QK`-Only LoRA & Self-Distilled Base-Voice Preservation (`subliminal_gap_splice.py`, `gap_trajectory_schema.py`, `train_left_brain_lora.py`, `three_legged_runtime.py`, `scholar_app.py`, 2026-11-10)
- **Context**: `LMFT-Scholar` produced garbled chat replies because `projector.output_norm` (`L2 = 55.5`) was `90x` louder than `Impish_LLAMA_4B` word embeddings (`0.6191`), `scholar_app.py` lacked LoRA loading and ChatML formatting, and `v_proj`/`o_proj` LoRA memorized training phrases on casual greetings.
- **Design Choice**: Locked `projector.output_norm` to the base model's word-embedding norm (`0.6191`), added native `prompt_format="chatml"` with `build_chatml_user_header`, removed all canned fallbacks, trained `models/left_brain_scholar/` with `QK`-only LoRA plus 40 self-distilled KL anchors from frozen `Impish_LLAMA_4B`, fixed REM Callosum persistence, and added non-blocking background GPU warmup plus Windows exclusive port-`8428` eviction so the Studio window opens immediately and never stacks behind a stale server.
- **Result**: Achieved `8/8` positive chord-swap shifts and `8/8` distinct greedy continuations on unseen holdouts (`+1.3773` comma / `+1.4355` turn-start net swap margin, `2/8` and `4/8` raw flips) while speaking naturally on greetings, literary dialogue, and Seminar essays.
