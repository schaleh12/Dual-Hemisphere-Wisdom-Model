# Archived Decision D061

## D061: `Qwen/Qwen3-8B` 4-Bit Left Brain & Two-Tier REM Dream Contemplation (`chess_app.py`, `sleep_consolidation.py`, 2026-11-07)
- **Context**: `Qwen3-4B` (`2560`-d) had limited poetic depth during REM Sleep, and a 2-sentence reflection could not separately articulate felt tension, structural geometry, and execution for the 3-Branch Archives.
- **Design Choice**: Trained a 4-bit NF4 `Qwen/Qwen3-8B` (`4096`-d) `SubliminalGapProjector` + `LoRALinear` checkpoint (`models/left_brain_qwen3_8b/`); upgraded `_generate_rem_sleep_self_talk` to emit 2-sentence cat-naps and 4-sentence deep-sleep poems; and added `apply_dream_reflection_to_archives`.
- **Result**: Verified `5/6` holdout chord-swap shifts (`+0.88` margin, `0.0` unplugged shift) and `37/37` passing tests.
