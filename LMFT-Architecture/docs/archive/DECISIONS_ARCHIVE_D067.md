# Archived Decision D067

## D067: Semantic Birth Anchoring, Symmetric Updates & `will_edges` Scoring (`subliminal_gap_splice.py`, `sleep_consolidation.py`, `chess_app.py`, 2026-11-08)
- **Context**: Audit found 5 installer bugs, one-way ratchets, `0.25 / 0.53` stamps, a rigged `calm_baseline_score` gate, random static at `Cycle 0` birth, REM vow theater, and un-wired `will_edges` in move scoring.
- **Design Choice**: Fixed the installer bugs, replaced one-way ratchets with symmetric updates, removed coordinate stamps, replaced `calm_baseline_score` with `calm_deviation`, cut Dream Vow theater, wired `self.will_edges` into move scoring, and anchored `SubliminalGapProjector` (`256`-dim) to LLM input embeddings at Birth.
- **Result**: Raw holdout flips reached `6/6` on `Qwen3-8B` and `5/6` (`6/6` calibrated) on `Impish_Bloodmoon_12B` and `Qwen3-4B`.
