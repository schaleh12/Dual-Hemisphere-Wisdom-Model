

## D055: Startup Context Hygiene & Removal of Screenshot Regex / Keyword Hacks (`AGENTS.md`, `three_legged_runtime.py`, `chess_kibitz_bridge.py`, 2026-11-04)
- **Context**: Auditor review caught startup file bloat (`32 KB`) and three band-aids in `commit 78345d1` (regexes, keyword lists, and Tier-2 leaf force-attach).
- **Design Choice**: Added strict line/character caps and Rules `4a–4c` to `AGENTS.md`; archived older state/decisions to `docs/archive/`; suppressed `Qwen3-4B` CoT leakage via `"Assistant: <think>\n\n</think>\n"`; restricted move-history checks to pure `python-chess` state; and removed Tier-2 force-attach in `_consolidate_sleep_clusters_locally`.
- **Result**: Enforced by `test_startup_files_line_and_character_caps`.


## D056: Impish Bloodmoon 12B Adapter, Isolated Virgin Clone, & Patreon Bundle Encryption (`impish_bloodmoon_adapter.py`, 2026-11-04)
- **Context**: Needed an isolated `Cycle 0` Virgin Baseline clone paired with `SicariusSicariiStuff/Impish_Bloodmoon_12B` (`hidden_size = 5120`) in 4-bit NF4 on a 16 GB RTX 4080, strict Left-Brain lineage separation, and Patreon `.enc` bundle encryption.
- **Design Choice**: Added 4-bit `BitsAndBytesConfig` support to `SubliminalGapLeftBrain`; built `clone_virgin_baseline_for_impish_bloodmoon` (`outputs/impish_bloodmoon_state/`), `train_impish_bloodmoon_callosum_and_lora` (`models/left_brain_impish_bloodmoon/`), and PBKDF2-HMAC-SHA256 `.enc` encryption.
- **Result**: Trained 6 epochs on RTX 4080 (`6/6` positive holdout chord-swap shifts, `4/6` no-gap calibrated flips, `0.0` unplugged shift, `6/6` greedy text changes) and verified in `tests/test_impish_bloodmoon_adapter.py`.
