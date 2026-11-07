
## D058: Sliding Context Threads, AFK Microwave Trainer, & GitHub SHA-256 Sync (`three_legged_runtime.py`, `github_auto_sync.py`, 2026-11-05)
- **Context**: Needed multi-turn conversation memory without polluting Pass-1 `Fact`, persistent session threads, hands-free REM Sleep + AFK training, and SHA-256 release verification.
- **Design Choice**: Added `format_sliding_conversation_history` (sliding window of clean dialogue while Pass-1 `Fact` scores only the current turn); added `threads.json` CRUD; added automatic REM Sleep at threshold (`Virgin` read-only); and built `github_auto_sync.py` (`Guardrail #4` protecting `outputs/*`).
- **Result**: Verified in `tests/test_chess_app.py` and `tests/test_bloodmoon_app.py`.
