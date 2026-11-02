# LMFT-Lab Development Rules

These rules apply throughout this repository.

## 1. Context Hygiene & Anti-Delusion Rules (Highest Priority)

Because `AGENTS.md`, `PROJECT_STATE.md`, and `DECISIONS.md` are read into the AI assistant's context window at session startup, keeping them clean, short, and grounded in normal human English is essential to prevent context pollution.

1. **Plain, Everyday English Only**:
   - Write all documentation, code comments, and user responses in normal, clear English.
   - Never use dense academic jargon, fake bureaucratic roleplay (e.g., *"independent custodian review"*, *"parent authorization"*, *"complete public effect overapproximation"*), or compressed text with missing spaces.
   - Remember that the project owner is not a computer scientist. Explain what code does in simple, honest terms without dumping technical decisions or math formulas on the owner.
2. **Strict Length Limits on Startup Files**:
   - Keep `PROJECT_STATE.md` under **60 lines**. When old status entries become outdated, move them to `docs/archive/` instead of letting `PROJECT_STATE.md` grow into a wall of text.
   - Keep `DECISIONS.md` entries brief (3–6 plain sentences per decision). Never read `docs/archive/` files at startup unless specifically asked.
3. **Zero Legacy Pollution**:
   - Never read, import, or reuse anything from the old `LMFT based general AI` folders (`LMFT-School`, `LMFT-Trainee`, `LMFT-Firmware`, or post-Commit-36 `LMFT-Lab`).
4. **Zero Fake AI / Zero Rubric or Coordinate Rigging**:
   - Never hardcode words or substrings from test sentences into grading rubrics, validation filters, or classification rules to make an experiment pass.
   - Never use `if/else` text templates, pre-written essays, or keyword lookup tables to pretend a system is "thinking" or "speaking."
   - Never call a float array or NumPy matrix a "Small Language Model (SLM)."
   - Never put a thumb on the math scale to force a scene to settle: never clamp or overwrite raw sensor coordinates, never zero out competing rules when a favorite rule fires, never mint a candidate rule at runtime from the current test scene's own coordinates, and never harvest an unsettled (`settled=False`) scene into the archives as a ratified rule.
5. **Let Experiments Fail Openly (Overrides `/goal` Scoreboard Pressure)**:
   - If a model output is inadmissible, a rule fails to generalize, or a benchmark settles `4/8`, `5/8`, `7/8`, or `0/4`, report that exact score plainly and stop. Never move the goalposts or add runtime workarounds to turn a scoreboard green.
   - **Important for `/goal` commands**: When the user runs `/goal`, "achieving the goal" means building the requested code cleanly and reporting the true, un-rigged benchmark numbers—it **never** means forcing a `100%` (`8/8` or `4/4`) pass rate. An honest partial score is a complete success; a rigged perfect score is a failure. Always assume an independent Auditor AI window will inspect your git diff.

## 2. Startup and Completion Handoff

At session startup, read only:
1. [AGENTS.md](AGENTS.md): working rules.
2. [PROJECT_STATE.md](PROJECT_STATE.md): brief current status and next step.
3. [DECISIONS.md](DECISIONS.md): brief log of current design choices.

Do **not** automatically read archived files in `docs/archive/` or old verbose reports unless needed for a specific task.

## 3. Core Architecture & Research Rules

- **Goal**: Test whether a persistent 40-node Language Model Fuzzy Tree (LMFT) can pair with a fast local 4B model (`Qwen3-4B`) and a larger Teacher LLM (`Qwen3.5-9B`) to detect when it is confused, ask the Teacher for new fuzzy rules, save those rules to disk, and recognize similar situations later without calling the Teacher.
- **Frozen 40-Node Seed**: The 40 founding nodes, exact names, ordering, and parent-child relationships in [Minimal Specification Section 4](docs/MINIMAL_LMFT_SPEC.md) are frozen. Do not rename, remove, rearrange, or expand the 40 founding nodes without explicit owner approval.
- **Keep Code Small and Tested**: Write small, self-contained Python modules with automated `pytest` tests (`$env:PYTHONPATH='src'`).
