# LMFT-Architecture — Design Decisions (Rolling 5-Decision Window)

*(Keep only the 5 most recent decisions here, under 80 lines AND 5,000 characters in plain English. Earlier decisions `D001–D067` are archived in `LMFT-Architecture/docs/archive/`.)*

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

## D071: Norm-Locked ChatML Callosum, `QK`-Only LoRA & Non-Blocking Startup (`subliminal_gap_splice.py`, `train_left_brain_lora.py`, `three_legged_runtime.py`, `scholar_app.py`, 2026-11-10)
- **Context**: `LMFT-Scholar` gave garbled replies because `projector.output_norm` (`55.5`) was `90x` louder than `Impish_LLAMA_4B` word embeddings (`0.6191`), `v_proj`/`o_proj` LoRA memorized phrases on greetings, and synchronous GPU loading stalled window startup.
- **Design Choice**: Locked `projector.output_norm` to `0.6191`, added native `prompt_format="chatml"`, trained `models/left_brain_scholar/` with `QK`-only LoRA plus 40 self-distilled KL anchors, and added background GPU warmup plus Windows exclusive port-`8428` eviction.
- **Result**: Achieved `8/8` positive swap shifts (`+1.3773` comma / `+1.4355` turn-start net margin) with instant window startup and natural Gear-2 replies.

## D072: Genuine 4B-to-MiniLM Right-Brain Leaf Distillation & Unseen Holdout Gatekeeper (`scholar_app.py`, `scholar_ui.py`, `right_brain_equilibrium.py`, `sleep_consolidation.py`, 2026-11-10)
- **Context**: `LMFT-Scholar` previously grew junk Tier-2/3 leaves (`fairness_chaos_synthesis`, `power_existence_accord`, `coached_low_power`) from Python f-strings and Chess blunder templates without teaching the MiniLM bridges what new leaves meant.
- **Design Choice**: Removed all Python string-concatenation and Chess leaf templates from `scholar_app.py`, added contrasting literary foils to all 12 Seminar lessons, enabled Tier-4 sensor querying in `ThreeBridgeSensors.score_text`, fixed `learned_leaves.json` restart persistence, and wired REM Sleep to distill plain-English concepts and 10 modern examples from `Impish_LLAMA_4B`, fine-tune isolated plastic bridges in `outputs/scholar_state/right_brain_bridges/` with replay anchors, and roll back any candidate failing unseen holdout checks (`SceneA >= 0.48`, `margin >= 0.12`, `calm <= 0.38`, `drift <= 0.12`).
- **Result**: Live 5-step GPU Seminar run cleanly ratified `2/5` human leaves (`compassionate_covenant` at `+0.6074` unseen margin and `chosen_presence` at `+0.9138` unseen margin) while honestly rolling back `3/5` candidates whose founding-leaf drift (`0.13–0.22`) exceeded `0.12`.
