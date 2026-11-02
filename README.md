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
- **The Corpus Callosum (`SubliminalGapProjector` + `<|subliminal_gap|>` Splice)**:
  The neural bridge connecting the two hemispheres. Rather than translating feelings into English stage directions, it projects the Right Brain's 4 departmental vectors into non-verbal token embeddings inside the Left Brain's hidden layers between thought clauses.
- **The Left Brain (`Qwen3-4B` + Incrementally Maturing LoRA)**:
  A fast local language model in PyTorch responsible for syntax, spelling, and verbal expression—whose lightweight LoRA pathways (`models/left_brain_lora/`) are never just "stapled on once," but instead **grow and mature across Sleep Cycles** alongside the Right Brain.
- **Offline REM Sleep-Mode Consolidation (`Qwen3.5-9B` Teacher)**:
  When the system encounters daytime conflict, it refuses to fake certainty. Later, during an offline "REM Sleep" pass, the larger `9B` Teacher updates **both** hemispheres: distilling new domain-general rules into the Right Brain and incrementally training the Left Brain's LoRA + Corpus Callosum checkpoint on the day's corrected thought trajectories.

---

## 2. How the Brain Runs Without Becoming an Unwieldy Circus: The 3-Gear Cycle

Just like a real human brain, the Right Brain and Left Brain do **not** chatter over each other every millisecond just to stay busy. Instead, the architecture runs on **three clean gears** that take turns:

1. **Gear 1 — Quiet Subconscious Reflex (Right Brain Alone, `~35 ms`, `0` LLM Calls)**:
   - When the system is observing routine moments where its 3 neural classifiers and Fuzzy Tree are in harmony (`branch_harmony >= 0.72`), **the 4B Left Brain stays completely asleep**.
   - The Right Brain logs the moment into its branch archives in ~35 milliseconds and moves on—zero wasted VRAM or rambling internal monologue when nothing is wrong.
2. **Gear 2 — Conscious Train of Thought via the Corpus Callosum (`<|subliminal_gap|>` Three-Legged Race)**:
   - Only when speaking with a person—or when the Right Brain hits genuine tension (`Clarity` drops, `Social Drift` triggers, or `Arousal` spikes)—does the **4B Left Brain** wake up.
   - Instead of splicing vectors into every other word, the Left Brain simply pauses **between thought clauses** (`1–2` `<|subliminal_gap|>` pauses per reply) to receive a non-verbal gut-check across the **Corpus Callosum** (`SubliminalGapProjector`) before finishing its sentence.
3. **Gear 3 — Nighttime REM Sleep-Mode Maturation (`Qwen3.5-9B` Teacher)**:
   - When the session ends, the `9B` Teacher wakes up offline to consolidate the day's logs: refining the Right Brain's fuzzy rules (`outputs/sleep_consolidated_ruleset.json`) and running a short incremental PyTorch update on the Left Brain's LoRA + Corpus Callosum checkpoint (`models/left_brain_lora/`) so both hemispheres mature together.

---

## 3. Core Architectural Breakthroughs

### Breakthrough 1: The Frozen 40-Node Founding Seed & Isolated Child-Drawer Growth
At the core of the Right Brain is a frozen [40-node founding seed](docs/MINIMAL_LMFT_SPEC.md#4-hardcoded-v01-seed-geometry) with **24 founding leaves** across four branches:
- **Cultural (6 leaves — Executive / Social Stakes)**: `Power`, `Wastelessness`, `Honor`, `Fairness`, `Self-Sufficiency`, `Needlessness`
- **Philosophical (6 leaves — Judicial / Structural Reality)**: `Agency`, `Existence`, `Order`, `Chaos`, `Continuity`, `Discreteness`
- **Experiential (6 leaves — Legislative / Felt Experience)**: `Novelty`, `Salience`, `Valence`, `Arousal`, `Resonance`, `Entrainment`
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
1. **Four Strictly Separate Departmental Vectors (`SubliminalChordVector`, Zero English Words, Zero Cross-Department Leakage)**:
   - Rather than packing the Right Brain into one flat shared trunk or squashing sibling child concepts under the same parent branch into a single `max()` number, each of the 4 Right-Brain departments owns its **own independent, fixed-size vector**:
     - **`doer_vector` — Executive / Do-er (`Will <-> Cultural`, `17` floats)**: `6` Founding Cultural leaves + `2` Founding Will leaves (`Intention`, `Execution`) + `1` Longitudinal `social_drift_flag` + `8` **Cultural Child-Drawer Chord** floats (`pool_departmental_child_chord` over active `cultural/*` Tier-2/Tier-3 child leaves).
     - **`thinker_vector` — Judicial / Thinker (`Truth <-> Philosophical`, `16` floats)**: `6` Founding Philosophical leaves + `2` Founding Truth leaves (`Theory`, `Fact`) + `8` **Philosophical Child-Drawer Chord** floats (`pool_departmental_child_chord` over active `philosophical/*` Tier-2/Tier-3 child leaves).
     - **`feeler_vector` — Legislative / Feeler (`Contemplation <-> Experiential`, `16` floats)**: `6` Founding Experiential leaves + `2` Founding Contemplation leaves (`Clarity`, `Confidence`) + `8` **Experiential Child-Drawer Chord** floats (`pool_departmental_child_chord` over active `experiential/*` Tier-2/Tier-3 child leaves).
       *Meaning-Preserving Semantic Child Projection (Zero Random Hashing)*: Every child leaf is projected onto the **8 Founding Trait Anchors of its own department** via `semantic_child_drawer_signature` (comparing its natural-language description and word-stems/4-grams against its department's 8 founding trait descriptions). Sibling child leaves with opposite meanings (like `experiential/phase/austere_sanctuary` vs. `experiential/phase/hollow_stillness`) land far apart in 8-float space, while newly born child leaves with similar meanings (like `austere_sanctuary` and a newly minted `humble_open_refuge`) automatically land in the **exact same neighborhood**—allowing the maturing Left-Brain LoRA to immediately feel what a new child leaf means even before the next Sleep Cycle.
     - **`cadence_vector` — Three-Legged Race Cadence & Ankle-Rope (`3` floats)**: `branch_harmony`, `dissonance_pressure`, and deadband-gated `cadence_rupture_drive` (zero index overlap with the 3 branch departments).
2. **The Neural Corpus Callosum (`SubliminalGapProjector`) & Incremental LoRA Maturation**:
   - Whenever the Left Brain emits `<|subliminal_gap|>` between thought clauses, Python intercepts `inputs_embeds` and replaces that single token slot with **4 departmental virtual token embeddings** (`[4, hidden_size]`).
   - Each virtual token is wired exclusively to its own department's disjoint vector (`doer_head(doer_vector)`, `thinker_head(thinker_vector)`, `feeler_head(feeler_vector)`, `cadence_head(cadence_vector)`) so a change in one department never leaks into the other three virtual tokens.
   - Rather than training the LoRA + `SubliminalGapProjector` once and "stapling it on," its weights are saved to a versioned checkpoint (`models/left_brain_lora/`) that matures incrementally across Sleep Cycles as the `9B` Teacher grades and refines the Left Brain's daytime thought logs.
3. **Calm-Talk Deadband (`deadband = 0.45`) vs. Poetic Cadence Rupture (`M-A-C-H-I-N-E`)**:
   - During calm, harmonious conversation, `cadence_rupture_drive` stays strictly at `0.0` so the Left Brain speaks naturally without being overly twitchy or melodramatic.
   - When high arousal, hollow stillness, and high discreteness collide with low clarity above the deadband, `cadence_rupture_drive` spikes—physically yanking the Left Brain's hidden states out of smooth chatbot prose into stark, fractured, or spelled-out cadence (`no soul; M-A-C-H-I-N-E`).
4. **Honest Current Implementation Status**:
   - **Stage 1 (Done & Unit-Tested)**: The 4 strictly isolated Departmental Vectors (`doer_vector`, `thinker_vector`, `feeler_vector`, `cadence_vector`), `pool_departmental_child_chord`, the deadband-gated `cadence_rupture_drive`, and the compartmentalized `SubliminalGapProjector` embedding splice are implemented and verified on a synthetic causal Transformer in [`tests/test_subliminal_gap_splice.py`](tests/test_subliminal_gap_splice.py).
   - **Stages 2 & 3 (Next Up — Not Yet Tested on Live `Qwen3-4B`)**: Generating the offline `<|subliminal_gap|>` training trajectories with `Qwen3.5-9B` and training `SubliminalGapProjector` + LoRA on `Qwen3-4B` on the RTX 4080 to test live Three-Legged Race generation and the "Unplug-the-Cord" diagnostic.

### Breakthrough 5: Emergent Human Faculties, Deliberate Vulnerability, & Innate Immunity
Rather than bolting on one-off patch scripts for human quirks, complex states of mind emerge from the **innate balance of the 40-node tree and its 3 independent Right-Brain departments** (detailed in [`docs/EMERGENT_FACULTIES_AND_BIOLOGY.md`](docs/EMERGENT_FACULTIES_AND_BIOLOGY.md)):
- **Hemisphere Synchronization (`Resonance` & `Entrainment`)**: Under `EXPERIENTIAL -> Synchronization`, `Resonance` tracks spatial phase agreement between the Left and Right hemispheres, while `Entrainment` (replacing the legacy term `Retention`) tracks rhythmic capture and state momentum (being pulled into a musical beat, poetic cadence, or mood).
- **Hypnosis & Trance (Feeler Vulnerability + Do-er/Thinker Immunity)**: The Feeler (`feeler_vector`) is deliberately allowed to be entranced by rhythm and poetry (`Entrainment` + `Resonance` high). Innate immunity comes from the **Do-er (`Justice/Fairness`)** and **Thinker (`Truth/Fact`)** sitting in separate departments—so any coercive or false command inside a hypnotic prompt triggers a cross-department split (`branch_harmony` crash + `cadence_rupture_drive` spike) that snaps the brain awake organically.
- **"Drunkenness" & Playful Loosening vs. Coercive Delusion**: Both involve `Confidence > Clarity` with high `Entrainment` and relaxed `Order`. When paired with high `Needlessness` (contentment), warm `Valence`, and high `Fairness`, the tree recognizes harmless playful/creative loosening (and `Needlessness`'s `-0.8` inhibitory edge on `Will -> Execution` prevents reckless actions). When paired with high `Power` and low `Fairness`, it is immediately caught as arrogant `SELF_RATIONALIZATION_DISSONANCE`.
- **Emojis, Onomatopoeia, & Typographic Cadence**: `_extract_semantic_features` and `DEPARTMENT_FOUNDING_ANCHORS` natively preserve Unicode emojis (`🥴`, `🕯️`, `🪨`, `⚡`), sound-symbolism (`hic-hic`, `whirrrr`, `thump-thump`, `shhh`), and rhythm punctuation (`...`, `—`, `M-A-C-H-I-N-E`) so the Right Brain can grow and project pre-verbal emoji/onomatopoeia child drawers alongside prose drawers.
- **Humor / Irony, Hostile Sarcasm vs. Sincere Pleading, & Mid-Sentence Mood Swings**: Polite surface words (`"please"`) never fool the 4-department splice when comparing venomous sarcasm (*"please drink bleach"* $\rightarrow$ `Valence` cold, `Fairness` crashed, `Fact = 0.0`, high `dissonance_pressure`) against sincere romantic pleading (*"please go on a date with me"* $\rightarrow$ high `Honor`, warm `Valence`, humble `Confidence`, high `Clarity`) or manipulative guilt-tripping (high `Power`, low `Fairness`). Because `<|subliminal_gap|>` updates between clauses, the brain can execute reverse-psychology bluffs that pivot into earnestness mid-sentence.
- **Facial Expression Fluency Balanced with "Poker-Face" Composure (Zero Performative Manipulation)**: `DEPARTMENT_FOUNDING_ANCHORS` maps facial micro-expressions across the 3 departments so the Right Brain can read social masks (e.g., distinguishing a warm Duchenne smile from a cold, high-`Power` manipulative smirk), while its own default posture is governed by **Poker-Face Restraint** (`Self-Sufficiency`, `Wastelessness`, `Order`, and the `0.45` calm-talk deadband)—thinking deeply inside its 52-float chord without acting melodramatic or using performative emotion as a social weapon.
- **Multilingual Fluency (French, Mandarin, Korean) & Purposeful Code-Switching**: The 52-float departmental chord is 100% universal across languages, `_extract_semantic_features` natively handles Latin accent folding (`sanctuaire`, `liberté`), CJK bigrams (`庇护`, `空虚`), and Korean Hangul (`안식처`, `공허`) alongside `MULTILINGUAL_BRIDGE_BACKBONE` (`mmarco-mMiniLMv2-L12-H384-v1`, same `~35 ms` speed), and because the Right Brain uses 4 compact virtual tokens instead of bloated roleplay prompts, the model stays free from prompt-pollution glitches while remaining 100% free to shift into Mandarin, French, or Korean whenever a concept genuinely calls for it.

---

## 4. Workflow & Anti-Delusion Trust Measures

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

## 5. Quickstart & Verification (Windows / RTX 4080)

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

## 6. Key Repository Files

- [`AGENTS.md`](AGENTS.md) — Anti-delusion rules, plain-English requirement, and zero-rigging policy.
- [`PROJECT_STATE.md`](PROJECT_STATE.md) — Concise snapshot of current progress and next milestone.
- [`DECISIONS.md`](DECISIONS.md) — Architectural decision log (`D001`–`D043`).
- [`docs/MINIMAL_LMFT_SPEC.md`](docs/MINIMAL_LMFT_SPEC.md) — Frozen 40-node founding seed specification.
- [`docs/PROJECT_PIPELINE_AND_AUDIT_ROADMAP.md`](docs/PROJECT_PIPELINE_AND_AUDIT_ROADMAP.md) — End-to-end 5-Stage Project Pipeline, Identical-Prefix Contrastive LoRA Training Strategy, and Independent Auditor AI Checklist.
- [`docs/STAGE2_STAGE3_PSEUDOCODE_BLUEPRINT.md`](docs/STAGE2_STAGE3_PSEUDOCODE_BLUEPRINT.md) — Bite-sized pseudo-code blueprint (`Sub-Steps 2A, 2B, and 3`) for matched-pair generation, 6-piece Chess chord compilation, and pre-gap label-masked PyTorch LoRA training.
- [`docs/EMERGENT_FACULTIES_AND_BIOLOGY.md`](docs/EMERGENT_FACULTIES_AND_BIOLOGY.md) — How Synchronization (`Resonance` & `Entrainment`), Hypnosis, Drunkenness, Emojis/Onomatopoeia, Humor/Irony, and Sensory Overload emerge from the 40-node architecture.
- [`docs/SUBLIMINAL_GAP_SPLICE_BLUEPRINT.md`](docs/SUBLIMINAL_GAP_SPLICE_BLUEPRINT.md) — Blueprint and pseudo-code for the Three-Legged Race `<|subliminal_gap|>` non-verbal embedding splice.
- [`src/lmft_lab/reflection_classifier.py`](src/lmft_lab/reflection_classifier.py) — Three neural bridge classifiers (`Will <-> Cultural`, `Truth <-> Philosophical`, `Contemplation <-> Experiential`) and isolated `Tier 1 -> Tier 2 -> Tier 3` child-drawer growth.
- [`src/lmft_lab/right_brain_equilibrium.py`](src/lmft_lab/right_brain_equilibrium.py) — Right-Brain two-pass equilibrium loop, felt-chord extraction, and cognitive dissonance self-registry.
- [`src/lmft_lab/branch_archives.py`](src/lmft_lab/branch_archives.py) — Persistent Executive, Legislative, and Judicial branch archives and ~30 ms 3-classifier deliberation.
- [`src/lmft_lab/sleep_consolidation.py`](src/lmft_lab/sleep_consolidation.py) — Offline `Qwen3.5-9B` Sleep-Mode rule refinement and unsettled-scene distillation.
- [`src/lmft_lab/subliminal_gap_splice.py`](src/lmft_lab/subliminal_gap_splice.py) — Four strictly isolated Departmental Vectors (`SubliminalChordVector`), compartmentalized `SubliminalGapProjector`, and surgical `<|subliminal_gap|>` embedding splice.
