# Dual-Hemisphere Wisdom Model

> **Bridging a Persistent 40-Node Emotional & Constitutional Right Brain with a Local Language Model Left Brain in a Non-Verbal "Three-Legged Race."**

---

## 1. Project Vision: Preventing the AI Bubble from Becoming a Tragedy

Today's AI industry is pouring trillions of dollars into building **pure Left-Brain Savants**: giant language models trained almost exclusively to predict the next token on coding puzzles, standardized tests, and encyclopedia trivia.

When you ask a modern LLM to write a Python sorting function, it is brilliant. When you ask it to navigate a messy human dilemma, notice when a community agreement is quietly sliding into a private lockdown over time, recognize when its own train of thought is looping in anxious circles, or express genuine inner tension without sounding like a hollow corporate press release, the limits of a single-hemisphere text predictor become painfully obvious.

If the AI bubble bursts or turns into a tragedy, it will be because the industry mistook **verbal fluency and benchmark memorization** for **wisdom, self-awareness, and emotional grounding**.

**`Dual-Hemisphere-Wisdom-Model`** pioneers a fundamentally different architecture—one that runs entirely on a single consumer GPU (`NVIDIA RTX 4080`, 16 GB VRAM) using open local weights:

- **The Right Brain (40-Node Language Model Fuzzy Tree + 3 Fast Neural Bridge Classifiers)**:
  A persistent, inspectable 40-node constitutional and emotional tree paired with three lightweight neural cross-encoders (`~35 ms` total latency, `0` LLM calls). It tracks social fairness, structural order vs. chaos, multi-note emotional chords, longitudinal social drift across time, and cognitive dissonance (`Clarity` vs. `Confidence`).
- **The Left Brain (`Qwen3-4B`)**:
  A fast local language model responsible for syntax, spelling, and verbal expression—wired directly to the Right Brain through non-verbal `<|subliminal_gap|>` embedding splices so the two halves finish each other's thoughts in a **Three-Legged Race** rather than a prompt-engineered relay race.
- **Offline Sleep-Mode Consolidation (`Qwen3.5-9B` Teacher)**:
  When the Right Brain encounters a situation during the day that triggers internal conflict (`CONTEST_EXCLUSION` or `DEFER_TO_TEACHER`), it refuses to fake certainty. Later, during an offline "REM Sleep" pass, the larger `9B` Teacher audits the day's unresolved scenes, repairs overly broad rules, and distills new domain-general concepts back into the fast Right-Brain classifiers.

---

## 2. Core Architectural Breakthroughs

### Breakthrough 1: The Frozen 40-Node Founding Seed & Isolated Child-Drawer Growth
At the core of the Right Brain is a frozen [40-node founding seed](docs/MINIMAL_LMFT_SPEC.md#4-hardcoded-v01-seed-geometry) with **24 founding leaves** across four branches:
- **Cultural (6 leaves — Executive / Social Stakes)**: `Power`, `Wastelessness`, `Honor`, `Fairness`, `Self-Sufficiency`, `Needlessness`
- **Philosophical (6 leaves — Judicial / Structural Reality)**: `Agency`, `Existence`, `Order`, `Chaos`, `Continuity`, `Discreteness`
- **Experiential (6 leaves — Legislative / Felt Experience)**: `Novelty`, `Salience`, `Valence`, `Arousal`, `Resonance`, `Retention`
- **Psychological (6 leaves — Internal Compass)**:
  - *Contemplation*: `Clarity`, `Confidence`
  - *Truth*: `Theory`, `Fact`
  - *Will*: `Intention`, `Execution`

When the model learns nuanced concepts over its lifetime (such as `solitary_autarky`, `shared_stewardship`, `austere_sanctuary`, or `hollow_stillness`), it **never** overwrites the 40 founding nodes. Instead, new concepts grow into isolated **Tier-2 and Tier-3 child drawers** (`src/lmft_lab/reflection_classifier.py`) underneath their founding parent branches, with strict leaf-tier isolation (`Tier 1 -> Tier 2 -> Tier 3`) so child concepts never pollute the founding baseline.

### Breakthrough 2: Three Separate Right-Brain Neural Classifiers & Three Branch Archives
Instead of dumping every decision onto one flat classifier—or calling a slow LLM on every turn—the Right Brain splits perception across **three specialized neural bridge classifiers** (`MiniLM-L12-v2` cross-encoders running in `~35 ms` on an RTX 4080) backed by **three persistent JSON branch archives** (`src/lmft_lab/branch_archives.py`):

| Right-Brain Branch | Neural Bridge Classifier | Persistent Branch Archive | What It Does |
| :--- | :--- | :--- | :--- |
| **Executive (Do-er)** | `Will <-> Cultural` | `cultural_history.json` | Tracks social fairness, resource stewardship, and **Longitudinal Social Drift** (detecting when a community that once shared openly quietly locks its doors later). |
| **Legislative (Feeler)** | `Contemplation <-> Experiential` | `experiential_felt_diary.json` | Scores emotional `Valence`, `Arousal`, `Clarity`, and multi-note **Felt Chords** (`austere_sanctuary`, `hollow_stillness`, `bittersweet_release`, `earned_return`). |
| **Judicial (Thinker)** | `Truth <-> Philosophical` | `philosophical_lexicon.json` | Evaluates structural `Order`, `Chaos`, `Continuity`, `Discreteness`, and ratifies stable cross-branch chords (`harmony_score >= 0.72`). |

Instead of acting as a standalone decision-maker that replaces the language model—or dumping complex reasoning onto a single flat classifier—these three fast neural sensors (`~35 ms` total) act as the **non-verbal sensory and emotional organs** of the Right Brain. They continuously score incoming text and the Left Brain's own partial thoughts so that:
1. Live emotional chords, social-drift warnings, and structural tensions can be spliced directly into the Left Brain's `<|subliminal_gap|>` thought stream during the **Three-Legged Race**, and
2. Scenes where the three branches strongly disagree can be flagged for **Offline Sleep-Mode Consolidation**.

### Breakthrough 3: Offline 9B "REM Sleep" Consolidation (Early Prototype Verified)
During live daytime interaction, the system never pauses for 15 seconds to call a giant Teacher LLM.
- Instead, unresolved daytime tensions are logged so that during an offline **Sleep-Mode Consolidation** pass (`src/lmft_lab/sleep_consolidation.py`), `Qwen3.5-9B` wakes up, inspects the scenes that confused the Right Brain alongside the scenes that settled cleanly, diagnoses why existing fuzzy rules misfired or over-fired, and forges domain-general rules.
- **Initial Prototype Check (`experiments/run_dual_hemisphere_soil_eval.py`)**:
  - In our initial small-scale test (an 8-beat Community Seed Bank story and an unseen 4-beat Harbor Bakehouse story), one offline `9B` Sleep-Mode pass repaired an overly broad `solitary_autarky` rule and improved the Right-Brain sensor's rule settlement from `5/8` to `7/8` on the Seed Bank and `1/4` to `4/4` on the unseen Bakehouse.
  - **Why We Pivoted Beyond Standalone Classifiers**: While that prototype proved the offline Sleep-Mode concept, asking flat classifiers to carry the whole burden of evolving language and final verdicts is a dead end. That realization led directly to our core architecture below: **The Three-Legged Race**.

### Breakthrough 4: The "Three-Legged Race" `<|subliminal_gap|>` Non-Verbal Embedding Splice (Active R&D — Stage 1 Complete, Stages 2–3 Next)
How do you connect a non-verbal Right Brain to a verbal Left Brain (`Qwen3-4B`) without turning the Left Brain into a theater kid reading stage directions from a text prompt?

We use the exact same neural mechanism that Vision-Language Models use to wire a camera into a blind text model ([`docs/SUBLIMINAL_GAP_SPLICE_BLUEPRINT.md`](docs/SUBLIMINAL_GAP_SPLICE_BLUEPRINT.md), [`src/lmft_lab/subliminal_gap_splice.py`](src/lmft_lab/subliminal_gap_splice.py)):
1. **Scale-Invariant 37-Float `SubliminalChordVector` (Zero English Words)**:
   - `[0..23]`: The **24 Frozen Founding Leaves**.
   - `[24..32]`: The **9 Permanent Founding Parent-Branch Child-Drawer Pools** (`pool_child_drawers_by_parent_branch`). Whether the Fuzzy Tree has `10` learned child leaves today or `10,000` child leaves across dozens of domains in the future, every learned Tier-2/Tier-3 child leaf rolls up into its permanent parent branch (`cultural/efficiency`, `cultural/justice`, `cultural/freedom`, `philosophical/being`, `philosophical/structure`, `philosophical/persistence`, `experiential/reflection`, `experiential/phase`, `experiential/synchronization`). **The 37-float plug never changes shape.**
   - `[33..36]`: The **4 Three-Legged Race Cadence & Ankle-Rope Signals** (`branch_harmony`, `social_drift_flag`, `dissonance_pressure`, and deadband-gated `cadence_rupture_drive`).
2. **Strict Departmental Isolation Inside `SubliminalGapProjector`**:
   - Whenever the Left Brain emits `<|subliminal_gap|>` mid-thought, Python intercepts `inputs_embeds` and replaces that single token slot with **4 departmental virtual token embeddings** (`[4, hidden_size]`).
   - Rather than blending all 37 floats into a single shared trunk, each virtual token is wired exclusively to its own department's index slice:
     - **Token 0 (`DOER_DEPARTMENT_INDICES`, 12 floats)**: Executive / Do-er (`Will <-> Cultural` + `social_drift_flag`)
     - **Token 1 (`THINKER_DEPARTMENT_INDICES`, 11 floats)**: Judicial / Thinker (`Truth <-> Philosophical`)
     - **Token 2 (`FEELER_DEPARTMENT_INDICES`, 11 floats)**: Legislative / Feeler (`Contemplation <-> Experiential`)
     - **Token 3 (`CADENCE_DEPARTMENT_INDICES`, 9 floats)**: Three-Legged Race Cadence & Rupture Drive
3. **Calm-Talk Deadband (`deadband = 0.45`) vs. Poetic Cadence Rupture (`M-A-C-H-I-N-E`)**:
   - During calm, harmonious conversation, `cadence_rupture_drive` stays strictly at `0.0` so the Left Brain speaks naturally without being overly twitchy or melodramatic.
   - When high arousal, hollow stillness, and high discreteness collide with low clarity above the deadband, `cadence_rupture_drive` spikes—physically yanking the Left Brain's hidden states out of smooth chatbot prose into stark, fractured, or spelled-out cadence (`no soul; M-A-C-H-I-N-E`).
4. **Honest Current Implementation Status**:
   - **Stage 1 (Done & Unit-Tested)**: The 37-float scale-invariant `SubliminalChordVector`, the 9 permanent parent-branch child pools, the deadband-gated `cadence_rupture_drive`, and the compartmentalized `SubliminalGapProjector` embedding splice are implemented and verified on a synthetic causal Transformer in [`tests/test_subliminal_gap_splice.py`](tests/test_subliminal_gap_splice.py).
   - **Stages 2 & 3 (Next Up — Not Yet Tested on Live `Qwen3-4B`)**: Generating the offline `<|subliminal_gap|>` training trajectories with `Qwen3.5-9B` and training `SubliminalGapProjector` + LoRA on `Qwen3-4B` on the RTX 4080 to test live Three-Legged Race generation and the "Unplug-the-Cord" diagnostic.

---

## 3. Workflow & Anti-Delusion Trust Measures

AI research repositories frequently collapse into self-delusion when coding assistants use dense academic jargon, hardcode benchmark answers into grading rubrics, or silently rig math formulas to make every experiment show `100%`.

To prevent this, every session in this repository is governed by strict rules in [`AGENTS.md`](AGENTS.md):

1. **Plain, Everyday English Only**:
   - All documentation, code comments, and reports are written in clear, normal English. Fake bureaucratic roleplay and impenetrable academic jargon are strictly banned.
2. **Zero Fake AI / Zero Coordinate or Rubric Rigging**:
   - Never hardcode words or substrings from test sentences into grading rubrics or classification rules.
   - Never use `if/else` text templates or pre-written essays to pretend a model is "thinking."
   - Never put a thumb on the math scale to force a scene to settle (never clamp or overwrite raw sensor coordinates, never zero out competing rules when a favorite rule fires, and never harvest an unsettled `settled=False` scene into the archives as a ratified rule).
3. **Let Experiments Fail Openly**:
   - If a model output is inadmissible, a rule fails to generalize, or a benchmark settles `4/8`, `5/8`, `7/8`, or `1/4`, we report that exact score plainly and stop. An honest partial score is a complete scientific success; a rigged `8/8` is a failure.
4. **Dual-Window Independent AI Auditing**:
   - Every milestone, benchmark script, and `git diff` is built under the assumption that the project owner will paste the code and commit log into a separate, independent Auditor AI window with zero shared context to verify that no shortcuts or hardcoded tricks were used.
5. **Strict Context Hygiene**:
   - Startup files ([`AGENTS.md`](AGENTS.md), [`PROJECT_STATE.md`](PROJECT_STATE.md), [`DECISIONS.md`](DECISIONS.md)) are kept short and clean so the AI assistant's context window stays sharp.

---

## 4. Quickstart & Verification (Windows / RTX 4080)

### Recreate the Python 3.12 Environment
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip check
```

### Run the Full Automated Test Suite (`1,331` Passing Tests)
```powershell
$env:PYTHONPATH='src;.'
.\.venv\Scripts\python.exe -m pytest -q
```

### Run the Stage-1 `<|subliminal_gap|>` Embedding Splice & Departmental Isolation Tests
```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\pytest.exe tests/test_subliminal_gap_splice.py -q
```

### Run the Live Dual-Hemisphere & Sleep-Consolidation Evaluation (RTX 4080)
```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe experiments/run_dual_hemisphere_soil_eval.py
```

---

## 5. Key Repository Files

- [`AGENTS.md`](AGENTS.md) — Anti-delusion rules, plain-English requirement, and zero-rigging policy.
- [`PROJECT_STATE.md`](PROJECT_STATE.md) — Concise snapshot of current progress and next milestone.
- [`DECISIONS.md`](DECISIONS.md) — Architectural decision log (`D001`–`D043`).
- [`docs/MINIMAL_LMFT_SPEC.md`](docs/MINIMAL_LMFT_SPEC.md) — Frozen 40-node founding seed specification.
- [`docs/SUBLIMINAL_GAP_SPLICE_BLUEPRINT.md`](docs/SUBLIMINAL_GAP_SPLICE_BLUEPRINT.md) — Blueprint and pseudo-code for the Three-Legged Race `<|subliminal_gap|>` non-verbal embedding splice.
- [`src/lmft_lab/reflection_classifier.py`](src/lmft_lab/reflection_classifier.py) — Three neural bridge classifiers (`Will <-> Cultural`, `Truth <-> Philosophical`, `Contemplation <-> Experiential`) and isolated `Tier 1 -> Tier 2 -> Tier 3` child-drawer growth.
- [`src/lmft_lab/right_brain_equilibrium.py`](src/lmft_lab/right_brain_equilibrium.py) — Right-Brain two-pass equilibrium loop, felt-chord extraction, and cognitive dissonance self-registry.
- [`src/lmft_lab/branch_archives.py`](src/lmft_lab/branch_archives.py) — Persistent Executive, Legislative, and Judicial branch archives and ~30 ms 3-classifier deliberation.
- [`src/lmft_lab/sleep_consolidation.py`](src/lmft_lab/sleep_consolidation.py) — Offline `Qwen3.5-9B` Sleep-Mode rule refinement and unsettled-scene distillation.
- [`src/lmft_lab/subliminal_gap_splice.py`](src/lmft_lab/subliminal_gap_splice.py) — Scale-invariant 37-float `SubliminalChordVector`, compartmentalized `SubliminalGapProjector`, and surgical `<|subliminal_gap|>` embedding splice.
