# Archived Decision D059

## D059: Two Standalone Applications (`LMFT-Chess.exe` & `LMFT-Bloodmoon.exe`) (`chess_app.py`, `bloodmoon_app.py`, 2026-11-05)
- **Context**: Combining `Qwen3` Chess and `Impish Bloodmoon 12B` Hypnosis into one app caused UI clutter and VRAM swapping.
- **Design Choice**: Built **`LMFT-Chess.exe`** (`chess_app.py`, `chess_ui.py`) around `python-chess` in `outputs/chess_kibitzer_state/`, and **`LMFT-Bloodmoon.exe`** (`bloodmoon_app.py`, `bloodmoon_ui.py`) around `Impish_Bloodmoon_12B` with one-way `feminine_to_masculine` polarity and a 3-Phase Clockwise State Machine.
- **Result**: Verified across `tests/test_bloodmoon_app.py`, `tests/test_chess_app.py`, and `tests/test_virgin_baseline.py`.
