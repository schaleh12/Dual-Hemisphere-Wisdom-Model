# LMFT-Lab — Design Decisions

*(Write all new entries in brief, plain everyday English. Historical decisions `D001–D032` from before the Commit #36 clean-slate reset are preserved in `docs/archive/DECISIONS_PRE_RESET_COMMIT36.md`.)*

## D033: Clean-Slate Reset at Commit #36 and 12-Leaf Constant-Prompt Architecture (2026-10-23)

- **Context**: Previous iterations after Commit #36 suffered from bloated bureaucratic documentation and rigged test rubrics. We reset the repository to Commit #36 (`b41e485`), archived the 1,480 lines of dense startup logs into `docs/archive/` so they do not pollute future AI context windows, and locked in strict Anti-Rigging and Context Hygiene rules in `AGENTS.md`.
- **Design Choice**: Instead of stuffing every learned rule into the `Qwen3-4B` prompt (which would choke the 4B model as rules grow), `Qwen3-4B` always scores the **same 12 foundational `Cultural` and `Philosophical` leaves**. Python then evaluates the growing list of fuzzy rules (`FuzzyConceptRule`) against those 12 numbers in microseconds.
- **Status**: Implemented in `src/lmft_lab/reflection_classifier.py` with unit tests in `tests/test_reflection_classifier.py`.

## D034: Require All Rule Traits to Match (`min`) and Benchmark 7-Concept Scaling (2026-10-23)

- **Context**: In our first 5-sentence live test, averaging a rule's traits (`mean`) allowed a peaceful hermit to match 2 out of 3 traits of an armed gang (`2/5` clear matches).
- **Design Choice**: Changed `FuzzyConceptRule.evaluate()` to use standard fuzzy `AND` (`min` similarity across a rule's required traits), so missing any key trait (such as `chaos`) drops the rule's score.
- **Result**: Improved the 5-sentence test from **`2/5` to `5/5` clear matches**. In a larger 7-concept (14-sentence) stress test (`experiments/run_feasibility_scaling_study.py`), rule evaluation scaled to **1,000 rules in 23.6 milliseconds**, and the tree generalized cleanly to **4/7** unseen sentences (`5/7` top-1) while exposing two clear next-step bottlenecks: 4B-vs-9B leaf scale alignment and single-example incidental traits.

## D035: Isolated Multi-Tier Reverse-Abstraction (`Tier 1 -> Tier 2 -> Tier 3`) (2026-10-23)

- **Context**: When we tried adding newly learned child leaves directly into the same `Qwen3-4B` prompt as the 12 founding leaves, `Qwen3-4B` shifted the numbers it gave to the original 12 leaves, hurting accuracy (`2/7` unseen matches).
- **Design Choice**: Keep **Tier 1** (the 12 founding `Cultural` and `Philosophical` leaves) in a locked, unchanging prompt, and score learned child leaves (`LearnedLeaf` at **Tier 2** and **Tier 3**) in a separate focused pass (`make_tier2_child_request`). When a deeper tier rule fires ($\ge 0.50$), Specificity Inhibition discounts shallower parent/trunk rules by `1.0 - 0.5 * max_deeper` so specific sub-classifications cleanly supersede broad parent rules.
- **Result**: Raised the 7-concept stress test (`outputs/feasibility_scaling_report.json`) to **`6/6` Teacher rules accepted**, **`7/7` top-1 correct (`6/7` clear matches)** on unseen test sentences, and verified automatic **Tier-2 $\rightarrow$ Tier-3** sub-branch splitting in `outputs/continuous_tiered_report.json`.

## D036: Continuous-State Sequential Memory Across Experiential Leaves (2026-10-23)

- **Context**: Previously, the tree reset its working memory to zero after every sentence and left the 4 `Experiential -> Phase / Synchronization` leaves (`Valence`, `Arousal`, `Resonance`, `Retention`) unused.
- **Design Choice**: Added `ContinuousMemoryState` in `src/lmft_lab/reflection_classifier.py`, which retains a decaying trace (`retention_decay = 0.65`) of active leaf coordinates across turns and computes `Resonance` (penalizing sudden disruption spikes). When a brief follow-up sentence in an ongoing scene is underspecified on its own but resonates ($\ge 0.55$) with recent memory, the retained trace carries forward without overriding sudden new events.
- **Result**: Verified on a live 3-turn GPU sequence (`experiments/run_continuous_tiered_demo.py`): a quiet post-rescue follow-up sentence improved from `0.00` (confused) statelessly to **`0.96` confidence/clarity** statefully, while a sudden earthquake on Turn 3 dropped `Resonance` to `0.31` and switched immediately to the earthquake rule (`0.80` confidence/clarity).

## D037: 24-Leaf Executive Controller (`Truth -> Theory / Fact`) vs. Raw 4B Baseline (2026-10-23)

- **Context**: To test whether the Fuzzy Tree should hold executive decision-making privileges over the 4B model, we connected the final two founding leaves (`psychological/truth/theory` and `psychological/truth/fact`) and built a closed-loop `ExecutiveDecision` controller (`COMMIT_FAST`, `VERIFY_HYPOTHESIS`, `ESCALATE_TO_TEACHER`).
- **Design Choice**: `Fact` measures how strongly the current sentence alone proves a rule (`min(raw_confidence, raw_clarity)`), while `Theory` measures working-memory carryover (`confidence - raw_confidence`) or partial single-frontrunner evidence. The resulting `ExecutiveDecision` constrains `Qwen3-4B`'s spoken response (`generate_executive_response`) and triggers 9B Teacher learning when `ESCALATE_TO_TEACHER` fires.
- **Result**: In a head-to-head GPU test (`experiments/run_executive_hybrid_eval.py`, `outputs/executive_hybrid_report.json`), the **24-Leaf LM/FT Hybrid scored `3/4`** (and automatically resolved the 4th unfamiliar bureaucratic-paperwork case via the 9B Teacher into `redundant_procedural_burden`), whereas **Raw `Qwen3-4B` Alone scored `0/4`** (losing the rescue context on Turn 2 and hallucinating `fair_structured_cooperation` on bureaucratic red tape instead of escalating).
