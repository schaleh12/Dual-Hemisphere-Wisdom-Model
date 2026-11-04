# Archived Design Decisions — D033 to D050

*(Archived from `DECISIONS.md` on 2026-11-04 to keep `DECISIONS.md` at a rolling 5-decision window under 80 lines and 5,000 characters.)*

## D033: Clean-Slate Reset at Commit #36 and 12-Leaf Constant-Prompt Architecture (2026-10-23)
- Reset to Commit #36 (`b41e485`) and kept `Qwen3-4B` scoring the constant 12 founding `Cultural` and `Philosophical` leaves while Python evaluates `FuzzyConceptRule` items in microseconds (`src/lmft_lab/reflection_classifier.py`).

## D034: Require All Rule Traits to Match (`min`) and Benchmark 7-Concept Scaling (2026-10-23)
- Changed `FuzzyConceptRule.evaluate()` to use fuzzy `AND` (`min` similarity across required traits), improving the 5-sentence test from `2/5` to `5/5` and scaling to 1,000 rules in 23.6 ms.

## D035: Isolated Multi-Tier Reverse-Abstraction (`Tier 1 -> Tier 2 -> Tier 3`) (2026-10-23)
- Kept Tier 1 (12 founding leaves) in an unchanging prompt and scored learned child leaves (`LearnedLeaf` at Tier 2 and Tier 3) in a separate pass with Specificity Inhibition (`Tier 3 > Tier 2 > Tier 1`).

## D036: Continuous-State Sequential Memory Across Experiential Leaves (2026-10-23)
- Added `ContinuousMemoryState` (`retention_decay = 0.65`) and `Resonance` across sequential turns.

## D037: 24-Leaf Executive Controller (`Truth -> Theory / Fact`) vs. Raw 4B Baseline (2026-10-23)
- Connected `psychological/truth/theory` and `psychological/truth/fact` to drive `ExecutiveDecision` (`3/4` Hybrid vs. `0/4` Raw `Qwen3-4B`).

## D038: Silent Subconscious (`STAY_SILENT`), Branch-Gated Child Querying, and Shoulder Fuzzy Rules (2026-11-01)
- Added `STAY_SILENT`, branch-gated Tier-2/Tier-3 drawer opening (`50%` skipped), and upper/lower shoulder fuzzy curves (`>= 0.65`, `<= 0.30`).

## D039: Right-Brain Dominant 3-Bridge Equilibrium Loop, Self-Registry, and Will-to-Theory Circuit (2026-11-01)
- Built `src/lmft_lab/right_brain_equilibrium.py` with `ThreeBridgeSensors` (Do-er, Thinker, Feeler), `InternalSelfRegistry` separating `EXTERNAL_WORLD` (`Fact`) from `INTERNAL_LEFT_BRAIN_THEORY` (`Theory`), and outbound `Will` sleeve-tugs.

## D040: Three Separate Neural Bridge Classifiers, Chord-Rhyming Tactical Hail, and Two-Layer Contemplation Ledger (2026-11-01)
- Trained 3 separate `MiniLM-L12` cross-encoder classifiers (`models/right_brain_bridges/*`, `31–42 ms` full pass), added `FeltChord` rhyming (`chord_memory.py`), and built the two-layer `ContemplationLedger` (`contemplation_ledger.py`).

## D041: 3-Branch Right-Brain Archives (Executive, Legislative, Judicial) and Dual-Hemisphere Self-Talk Competition (2026-11-01)
- Built persistent Executive, Legislative, and Judicial archives (`branch_archives.py`) with un-rigged `social_drift` Executive vetoes and zero coordinate clamping.

## D042: Offline 9B Sleep-Mode Consolidation (`sleep_consolidation.py`) for Unresolved Tensions and Rule Refinement (2026-11-01)
- Built `run_sleep_consolidation` using `Qwen3.5-9B-Q4_K_M.gguf` offline to refine colliding Tier-1 rules and forge domain-general rules (`7/8` Parable, `4/4` unseen Harbor Bakehouse with `0` LLM calls).

## D043: Stage 1 of the 'Three-Legged Race' `<|subliminal_gap|>` Non-Verbal Embedding Splice (2026-11-02)
- Built `SubliminalChordVector` (`52` floats across 4 disjoint departmental vectors: Do-er `17`, Thinker `16`, Feeler `16`, Cadence `3`) and `SubliminalGapProjector` (`subliminal_gap_splice.py`).

## D044: Stage 2 (`2A` & `2B`) Offline Contrastive `<|subliminal_gap|>` Trajectory Schema & 9B+MiniLM Dataset Generation (2026-11-02)
- Built `gap_trajectory_schema.py` and `gap_trajectory_generator.py`, generating `17/18` training pairs and `6/6` unseen holdout pairs with `Qwen3.5-9B` and the 3 MiniLM bridges.

## D045: Sub-Step 3 PyTorch `Qwen/Qwen3-4B` Left-Brain LoRA + Corpus Callosum Training & Holdout Falsification (2026-11-02)
- Built `train_left_brain_lora.py` (`LoRALinear` on top 18 layers + `SubliminalGapProjector`, `-100` pre-gap label masking, `models/left_brain_lora/`), achieving `5/6` positive holdout shift, `0.0` unplugged shift, and `5/6` greedy Clause 2 flip.

## D046: Staged De-Frankenstein Cleanup Across Sleep Mode, Branch Archives, Equilibrium, and Subliminal Gap Splice (`Surgery A -> B -> C`, 2026-11-03)
- Removed story-specific rule/leaf band-aids and legacy stemmers across `sleep_consolidation.py`, `branch_archives.py`, `right_brain_equilibrium.py`, `contemplation_ledger.py`, and `subliminal_gap_splice.py`.

## D047: Stage 4 Closed-Loop 3-Gear Runtime (`three_legged_runtime.py`) and Dual Prior-Calibrated Holdout Metrics (2026-11-03)
- Built `run_three_legged_turn` and `SubliminalGapLeftBrain` (`Gear 1` fast reflex, `Gear 2` comma-spliced contemplation with zero stage directions, `Gear 3` archive compaction and `UnresolvedSleepScene` export).

## D048: Stage 5 Nighttime REM Sleep Growth (`rem_sleep_cycle.py`: `sleep_cycle_index = 1 -> 2`, 2026-11-03)
- Built the 4-Phase Daytime -> REM Sleep -> Next-Day loop (`models/left_brain_lora_cycle2/`), raising Daytime Parable settlement from `5/8` to `7/8` and Unseen Bakehouse from `3/4` to `4/4`.

## D049: Calibrated 5-Tier Philosophical Abstraction Matrix Woven Into Live Tree, Vibe, and `<|subliminal_gap|>` (2026-11-03)
- Wove `philosophical_abstraction_matrix.py` (6 RGB/CMY colors, 6 Chess Pieces, and 6 Tier-2 Shadow Pathology leaves) into `RightBrainVibe`, `SubliminalChordVector`, and `ThreeLeggedTurnResult`.

## D050: Live Mid-Sentence Comma Scoring at `<|subliminal_gap|>` (`three_legged_runtime.py`, `subliminal_gap_splice.py`, 2026-11-03)
- Added `chord_builder_at_comma` and `_compile_mid_sentence_comma_vibe` so Clause 1 is scored live at the comma by the 3 MiniLM bridges before splicing the 52-float chord into `<|subliminal_gap|>`.

## D051: Local Chess Kibitzing Club & Co-Regulation Sanctuary App (`chess_kibitz_bridge.py`, `sanctuary_club_app.py`, 2026-11-03)
- **Context**: Needed a local desktop app (`Start-LMFT-Club.bat`) combining a `python-chess` Kibitzing Table and a Co-Regulation Sanctuary under a single GPU lock.
- **Design Choice**: `apply_coregulation_lead_to_chord` keeps `vibe.coordinates` (`Fact`) un-rigged while stepping the 52-float comma chord by `alpha = 0.25`; `evaluate_chess_board_posture` anchors Pass-1 Philosophical `Fact` to `python-chess`; and `SanctuaryClubRuntime` persists coached rules, 3-branch archives, and savored vibes to `outputs/club_user_state/`.
- **Result**: Verified by end-to-end tests in `tests/test_sanctuary_club_app.py` and packaged into `dist/lmft-sanctuary-club-v0.1.0-win64.zip`.
