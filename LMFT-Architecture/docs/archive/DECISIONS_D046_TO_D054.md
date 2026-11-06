

## D055: Startup Context Hygiene & Removal of Screenshot Regex / Keyword Hacks (`AGENTS.md`, `three_legged_runtime.py`, `chess_kibitz_bridge.py`, 2026-11-04)
- **Context**: Auditor review caught startup file bloat (`32 KB`) and three band-aids in `commit 78345d1` (regexes, keyword lists, and Tier-2 leaf force-attach).
- **Design Choice**: Added strict line/character caps and Rules `4a–4c` to `AGENTS.md`; archived older state/decisions to `docs/archive/`; suppressed `Qwen3-4B` CoT leakage via `"Assistant: <think>\n\n</think>\n"`; restricted move-history checks to pure `python-chess` state; and removed Tier-2 force-attach in `_consolidate_sleep_clusters_locally`.
- **Result**: Enforced by `test_startup_files_line_and_character_caps`.
