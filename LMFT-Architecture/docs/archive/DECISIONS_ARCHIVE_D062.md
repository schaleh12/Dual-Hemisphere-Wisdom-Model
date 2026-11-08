# Archived Decision D062

## D062: Persistent `past_kv` Streaming, Multi-Stanza Discourse & Longitudinal REM Vows (`three_legged_runtime.py`, `chess_app.py`, 2026-11-07)
- **Context**: Discarding `past_kv` at every `<|subliminal_gap|>` caused `O(N^2)` prefill overhead, and REM dreams lacked multi-stanza structure or cross-cycle vow tracking.
- **Design Choice**: Added `_PersistentTurnStreamState` so prompt prefill runs once per turn and `past_kv` persists across `<|subliminal_gap|>` chords and `\n\n` stanza breaks; added `generate_multi_stanza_discourse` (`max_theory_delta < 0.04`) and `compute_stanza_residual_chord`; and added `rem_dream_chronicles` (`HONORED` / `BROKEN` / `NEW_VOW`).
- **Result**: Verified `40/40` passing tests across `test_three_legged_runtime.py`, `test_chess_app.py`, `test_sleep_consolidation.py`, and `test_virgin_baseline.py`.
