# LMFT-Lab — Design Decisions (Rolling 5-Decision Window)

*(Keep only the 5 most recent decisions here, under 80 lines AND 5,000 characters in plain English. Earlier decisions `D001–D052` are archived in `docs/archive/`.)*

## D053: 3-Stage Dilemma Crucible & 'Men of Culture' Context-Bracing (`dilemma_crucible.py`, 2026-11-04)
- **Context**: Needed a rule-generating dilemma farm in the Sanctuary pitting `Wave / Theory` against `Structure / Fact` across `1. Poetry <-> (Story-Telling)`, `2. Music <-> (Art)`, and `3. Philosophy <-> (Coding)`.
- **Design Choice**: Built `CruciblePairedBridgeSensors` to score both slots through the 3 MiniLM bridges (with 25% RGB color-wheel math on `Art` and 25% `ast.parse` invariants on `Coding`), wired the Men of Culture shelf and context-bracing buttons into the Sanctuary, and queued Crucible dilemmas for REM Sleep.
- **Result**: Verified by `test_dilemma_crucible_three_stages_context_bracing_and_rem_rule_forging`.

## D054: Chess Kibitzer Grounding, UI/UX Polish, & Zero Canned Templates (`chess_kibitz_bridge.py`, `sanctuary_club_app.py`, 2026-11-04)
- **Context**: Playtesting caught opening moves failing to settle in Gear 1, float telemetry leaking into the chess prompt, illegal move hallucinations (`Bf5+`, `Re8#`), and UI layout/scroll glitches.
- **Design Choice**: Calibrated `VIRGIN_FOUNDATIONAL_RULES` and opening posture so standard openings (`1. e4 e5 2. Nf3 Bc5`) settle in Gear 1; added `build_natural_chess_scenario_prompt` and `verify_or_ground_chess_moves_in_reply`; and added board coordinates, `[Undo]`/`[Play AI Move]`, and scroll-preserving poll updates.
- **Result**: Verified in `tests/test_sanctuary_club_app.py`.

## D055: Startup Context Hygiene & Removal of Screenshot Regex / Keyword Hacks (`AGENTS.md`, `three_legged_runtime.py`, `chess_kibitz_bridge.py`, 2026-11-04)
- **Context**: Auditor review caught startup file bloat (`32 KB`) and three band-aids in `commit 78345d1` (regexes, keyword lists, and Tier-2 leaf force-attach).
- **Design Choice**: Added strict line/character caps and Rules `4a–4c` to `AGENTS.md`; archived older state/decisions to `docs/archive/`; suppressed `Qwen3-4B` CoT leakage via `"Assistant: <think>\n\n</think>\n"`; restricted move-history checks to pure `python-chess` state; and removed Tier-2 force-attach in `_consolidate_sleep_clusters_locally`.
- **Result**: Enforced by `test_startup_files_line_and_character_caps`.

## D056: Impish Bloodmoon 12B Adapter, Isolated Virgin Clone, & Patreon Bundle Encryption (`impish_bloodmoon_adapter.py`, 2026-11-04)
- **Context**: Needed an isolated `Cycle 0` Virgin Baseline clone paired with `SicariusSicariiStuff/Impish_Bloodmoon_12B` (`hidden_size = 5120`) in 4-bit NF4 on a 16 GB RTX 4080, strict Left-Brain lineage separation, and Patreon `.enc` bundle encryption.
- **Design Choice**: Added 4-bit `BitsAndBytesConfig` support to `SubliminalGapLeftBrain`; built `clone_virgin_baseline_for_impish_bloodmoon` (`outputs/impish_bloodmoon_state/`), `train_impish_bloodmoon_callosum_and_lora` (`models/left_brain_impish_bloodmoon/`), and PBKDF2-HMAC-SHA256 `.enc` encryption; and added strict `_qwen_left_brain` / `_impish_left_brain` isolation plus `[Unlock .enc]` in `sanctuary_club_app.py`.
- **Result**: Trained 6 epochs on RTX 4080 (`6/6` positive holdout chord-swap shifts, `4/6` no-gap calibrated flips, `0.0` unplugged shift, `6/6` greedy text changes) and verified in `tests/test_impish_bloodmoon_adapter.py`.

## D057: Dual-Persona Adversarial Entrainment Sparring & Multi-Sentence Chaining (`entrainment_sparring.py`, `three_legged_runtime.py`, `sanctuary_club_app.py`, 2026-11-04)
- **Context**: Needed a self-play sparring mode in the Sanctuary where the model alternates between generating multi-sentence hypnotic induction scripts (`Induction Weaver`) and resisting hypnotic absorption (`Lucid Resistor`).
- **Design Choice**: Added `SubliminalGapLeftBrain.generate_multi_sentence_script` (`1..6` sentences); built `entrainment_sparring.py` with `feminine_to_masculine` / `masculine_to_feminine` / `auto_swap` polarity, Weaver (`entrainment >= 0.88`, `resonance >= 0.85`, `continuity >= 0.84`, `agency >= 0.80`) and Resistor (`existence >= 0.88`, `discreteness >= 0.82`, `needlessness >= 0.85`, `self-sufficiency >= 0.82`) comma chords, un-clamped `SparringResistorFactSensors`, honest `RESISTED_LUCIDLY` vs. `SWEPT_BY_ENTRAINMENT` verdicts, and `outputs/impish_sparring_trajectories.jsonl` export; and wired `ENTRAINMENT SPARRING (DUAL MODE)` into `sanctuary_club_app.py`.
- **Result**: Verified by `test_dual_persona_entrainment_sparring_and_multi_sentence_chaining`.
