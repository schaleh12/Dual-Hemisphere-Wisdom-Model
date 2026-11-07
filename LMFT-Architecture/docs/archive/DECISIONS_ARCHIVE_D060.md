# Archived Decision D060

## D060: `Intention -> Theory -> Confidence -> Execution` Pipeline & REM Polarity (`chess_kibitz_bridge.py`, `sleep_consolidation.py`, 2026-11-06)
- **Context**: A 1-ply board sensor could not project opponent replies into `psychological/truth/theory`, and REM sleep attached low shadow leaves (`<= 0.45`) to `hazard_` rules.
- **Design Choice**: Wired `evaluate_candidate_moves_with_right_brain` to generate impulses on `intention` (`Ply 1`), project top intentions through `Ply 2–3` via the 40-node Tree + `BitboardIntuitionSleeve` to ground `theory` and `confidence`, and bid on `execution`; enforced strict virtue/shadow polarity in `sleep_consolidation.py`.
- **Result**: Verified `24/24` passing tests across `tests/test_chess_app.py` and `tests/test_sleep_consolidation.py`.
