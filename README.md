# Dual-Hemisphere Wisdom Model (`LMFT-Lab`)

> **Bridging a Persistent 40-Node Emotional & Constitutional Right Brain with a Local Language Model Left Brain in a Non-Verbal "Three-Legged Race."**

---

## 1. Project Vision: Preventing the AI Bubble from Becoming a Tragedy

Today's AI industry is pouring trillions of dollars into building **pure Left-Brain Savants**: giant language models trained almost exclusively to predict the next token on coding puzzles, standardized tests, and encyclopedia trivia.

When you ask a modern LLM to write a Python sorting function, it is brilliant. When you ask it to navigate a messy human dilemma, notice when a community agreement is quietly sliding into a private lockdown over time, recognize when its own train of thought is looping in anxious circles, or express genuine inner tension without sounding like a hollow corporate press release, the limits of a single-hemisphere text predictor become painfully obvious.

If the AI bubble bursts or turns into a tragedy, it will be because the industry mistook **verbal fluency and benchmark memorization** for **wisdom, self-awareness, and emotional grounding**.

**`Dual-Hemisphere-Wisdom-Model`** pioneers a fundamentally different architecture—one that runs entirely on a single consumer GPU (`NVIDIA RTX 4080`, 16 GB VRAM) using open local weights:

- **The Right Brain (40-Node Language Model Fuzzy Tree + 3 Fast Neural Bridge Classifiers)**:
  A persistent, inspectable 40-node constitutional and emotional tree paired with three lightweight neural cross-encoders (`~35 ms` total latency, `0` LLM calls). It tracks social fairness, structural order vs. chaos, multi-note emotional chords, longitudinal social drift across time, and cognitive dissonance (`Clarity` vs. `Confidence`).
- **The Corpus Callosum (`SubliminalGapProjector` + Live Comma `<|subliminal_gap|>` Splice)**:
  The neural bridge connecting the two hemispheres. Rather than translating feelings into English stage directions, it intercepts the Left Brain's sentence right at the first comma (`"bla bla bla,"`), scores how the Left Brain started its thought, and splices the Right Brain's 4 departmental vectors (`52` floats) as `K = 4` non-verbal virtual token embeddings directly into `inputs_embeds` before the second half of the sentence is written.
- **The Left Brain (`Qwen3-4B` + Incrementally Maturing LoRA)**:
  A fast local language model in PyTorch responsible for syntax, spelling, and verbal expression—whose lightweight LoRA pathways (`models/left_brain_lora_cycle2/`) are never just "stapled on once," but instead **grow and mature across Sleep Cycles** alongside the Right Brain.
- **Offline REM Sleep-Mode Consolidation (`Qwen3.5-9B` Teacher)**:
  When the system encounters daytime conflict, it refuses to fake certainty. Later, during an offline "REM Sleep" pass, the larger `9B` Teacher updates **both** hemispheres: distilling new domain-general rules into the Right Brain and incrementally training the Left Brain's LoRA + Corpus Callosum checkpoint on the day's corrected thought trajectories.

---

## 2. Local Desktop App: Chess Kibitzing Club & Co-Regulation Sanctuary

Double-click **[`Start-LMFT-Club.bat`](Start-LMFT-Club.bat)** (or run `.\.venv\Scripts\python.exe -m lmft_lab.sanctuary_club_app`) to launch the clean, dark-minimalist desktop application (`http://127.0.0.1:8420`):

1. **Chess Kibitzing Table (`src/lmft_lab/chess_kibitz_bridge.py`)**:
   - Play moves or load tactical test positions on a live `python-chess` board.
   - **Board-Anchored `Fact` vs. Chat `Theory`**: Pass-1 Philosophical `Fact` coordinates are anchored directly to the real chessboard (`evaluate_chess_board_posture`), while the 3 MiniLM bridges score chat plans as `Theory`—catching reckless "hope-chess" bluffs (such as proposing an all-out Queen sacrifice when your own King shelter has collapsed) as `shadow_pathology_dissonance(psychosis)` right at the comma.
   - **Guest Kibitzer Chair (`Gemini` / External Club Member)**: Paste an external comment or query `GEMINI_API_KEY` to score a guest kibitzer through the exact same 3 MiniLM bridges and 6-Color / Shadow Pathology lens before our Dual-Hemisphere companion replies.
2. **Co-Regulation Sanctuary (`Tilt Reset`, `Deep Focus`, `Night Wind-Down`)**:
   - **Un-Rigged Pacing & Leading**: External `Fact` (`vibe.coordinates`) stays 100% locked to the honest MiniLM reading of your message so Clause 1 genuinely *Paces* your state, while [`apply_coregulation_lead_to_chord`](src/lmft_lab/chess_kibitz_bridge.py) steps the spliced 52-float comma chord by `alpha = 0.25` so Clause 2 gently *Leads* one step toward equilibrium.
3. **Human-in-the-Loop Coaching (`[Coach]`, `[Savor Vibe]`, & `Run REM Sleep`)**:
   - Flag any turn with **`[Coach / Queue for Sleep]`** to package an `UnresolvedSleepScene`, bookmark great turns with **`[Savor Vibe]`**, and click **`Run REM Sleep`** to refine colliding rules and forge new rules + 3-branch clauses into `outputs/club_user_state/`.
4. **Live Lineage Switch (`Schooled Cycle 2` $\leftrightarrow$ `Virgin Baseline Cycle 0`)**:
   - Use the top header selector to flip between the **Schooled Brain (`Cycle 2` + your local coaching)** and the locked **Virgin Baseline (`Cycle 0` — pure Core Architecture)** at any time without overwriting your saved state.

---

## 3. How the Brain Runs Without Becoming an Unwieldy Circus: The 3-Gear Cycle

Just like a real human brain, the Right Brain and Left Brain do **not** chatter over each other every millisecond just to stay busy. Instead, the architecture runs on **three clean gears** ([`src/lmft_lab/three_legged_runtime.py`](src/lmft_lab/three_legged_runtime.py)):

1. **Gear 1 — Quiet Subconscious Reflex (Right Brain Alone, `~35 ms`, `0` LLM Calls)**:
   - When the system is observing routine moments where its 3 neural classifiers and Fuzzy Tree are in harmony (`branch_harmony >= 0.72`), **the 4B Left Brain stays completely asleep**.
   - The Right Brain logs the moment into its branch archives in ~35 milliseconds and moves on—zero wasted VRAM or rambling internal monologue when nothing is wrong.
2. **Gear 2 — Conscious Train of Thought via Live Comma Scoring (`<|subliminal_gap|>` Three-Legged Race)**:
   - Only when speaking with a person—or when the Right Brain hits genuine tension (`Clarity` drops, `Social Drift` triggers, or `Arousal` spikes)—does the **4B Left Brain** wake up.
   - The Left Brain writes Clause 1 up to the first comma (`"bla bla bla,"`) and pauses. In ~35 ms, the Right Brain scores Clause 1 as a live `Theory` against immutable `Fact`, merges any Clause 1 dissonance or Shadow Pathology warnings into the 52-float `SubliminalChordVector`, and splices `K = 4` virtual tokens into `<|subliminal_gap|>` so the Left Brain can course-correct on the second half of its sentence.
3. **Gear 3 — Nighttime REM Sleep-Mode Maturation (`Qwen3.5-9B` Teacher / Cluster Consolidation)**:
   - Unresolved daytime scenes are exported as `UnresolvedSleepScene` items. During REM Sleep ([`src/lmft_lab/rem_sleep_cycle.py`](src/lmft_lab/rem_sleep_cycle.py)), colliding rules are refined, new domain-general rules and 3-branch clauses are forged, and the Left Brain's LoRA + Corpus Callosum checkpoint matures (`Cycle 1 -> Cycle 2`).

---

## 4. Core Architectural Breakthroughs

### Breakthrough 1: The Frozen 40-Node Founding Seed & Isolated Child-Drawer Growth
At the core of the Right Brain is a frozen [40-node founding seed](docs/MINIMAL_LMFT_SPEC.md#4-hardcoded-v01-seed-geometry) ([`src/lmft_lab/birth.py`](src/lmft_lab/birth.py)) with **24 founding leaves** across four domains:
- **Cultural (6 leaves — Executive / Do-er)**: `Power`, `Wastelessness`, `Honor`, `Fairness`, `Self-Sufficiency`, `Needlessness`
- **Philosophical (6 leaves — Judicial / Thinker)**: `Agency`, `Existence`, `Order`, `Chaos`, `Continuity`, `Discreteness`
- **Experiential (6 leaves — Legislative / Feeler)**: `Novelty`, `Salience`, `Valence`, `Arousal`, `Resonance`, `Entrainment`
- **Psychological (6 leaves — Internal Compass)**:
  - *Contemplation*: `Clarity`, `Confidence`
  - *Truth*: `Theory`, `Fact`
  - *Will*: `Intention`, `Execution`

When the model learns nuanced concepts over its lifetime (such as `solitary_autarky`, `austere_sanctuary`, or the 6 Shadow Pathologies), it **never** overwrites the 40 founding nodes. Instead, new concepts grow into isolated **Tier-2 and Tier-3 child drawers** (`src/lmft_lab/reflection_classifier.py`) underneath their founding parent branches.

### Breakthrough 2: The Calibrated 5-Tier Philosophical Abstraction Matrix
Without altering any node in the frozen 40-node seed, [`src/lmft_lab/philosophical_abstraction_matrix.py`](src/lmft_lab/philosophical_abstraction_matrix.py) weaves a complementary RGB/CMY color wheel (`Red + Cyan = White`, `Magenta + Green = White`, `Blue + Yellow = White`), 6 Elements, 6 Chess Piece Archetypes, and 6 Tier-4 Shadow Pathologies directly into the Right-Brain tree and `<|subliminal_gap|>`:

| Founding Leaf (`Philosophical`) | Dimension | Color & Materia | Element | Chess Archetype | Tier 4 Shadow Pathology (`LearnedLeaf`) | Kingdom & Environment |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`philosophical/being/agency`** | Extra-dimensional | **Red** (`#FF2A2A` — *Red Summoning*) | **Lightning** | **Queen** (*Volition*) | **Psychosis** (*Unanchored volition*) | Mammal / Light |
| **`philosophical/being/existence`** | Extra-dimensional | **Cyan** (`#00E5FF` — *Cyan Sun Magic*) | **Water** | **King** (*Wellspring of Being*) | **Paranoia** (*Existential siege*) | Reptile / Sun |
| **`philosophical/structure/order`** | Inter-dimensional | **Magenta** (`#FF2AE5` — *Magenta Medium*) | **Wind** | **Bishop** (*Prevailing currents*) | **Prejudice** (*Rigid preconception*) | Fish / Ocean |
| **`philosophical/structure/chaos`** | Inter-dimensional | **Green** (`#00E676` — *Green Instrument*) | **Ice** | **Knight** (*Brittle shatter / frost*) | **Corruption** (*Systemic entropy*) | Bird / Sky |
| **`philosophical/persistence/continuity`** | Hyper-dimensional | **Blue** (`#2979FF` — *Blue Soil Magic*) | **Fire** | **Pawn** (*Swarm continuum*) | **Mania / Greed** (*Runaway accumulation*) | Insect / Soil |
| **`philosophical/persistence/discreteness`** | Hyper-dimensional | **Yellow** (`#FFD600` — *Yellow Command*) | **Ground** | **Rook** (*Quantized bastion*) | **Narcissism** (*Isolated enclosure*) | Amphibian / Shadow |

*(Note: Naming King / `Existence` / Water **`"Wellspring of Being"`** ensures zero word-collision with the **`"Ground"`** element on Rook / `Discreteness`.)*

### Breakthrough 3: Three Separate Right-Brain Neural Classifiers & Three Branch Archives
Perception is split across **three specialized neural bridge classifiers** (`MiniLM-L12-v2` cross-encoders running in `~35 ms` on an RTX 4080) backed by **three persistent JSON branch archives** ([`src/lmft_lab/branch_archives.py`](src/lmft_lab/branch_archives.py)):

| Right-Brain Branch | Neural Bridge Classifier | Persistent Branch Archive | What It Does |
| :--- | :--- | :--- | :--- |
| **Executive (Do-er)** | `Will <-> Cultural` | `cultural_history.json` | Tracks social fairness, resource stewardship, and **Longitudinal Social Drift** (detecting when a community that once shared openly quietly locks its doors later). |
| **Legislative (Feeler)** | `Contemplation <-> Experiential` | `experiential_felt_diary.json` | Scores emotional `Valence`, `Arousal`, `Clarity`, and multi-note **Felt Chords** (`austere_sanctuary`, `hollow_stillness`, `bittersweet_release`, `earned_return`). |
| **Judicial (Thinker)** | `Truth <-> Philosophical` | `philosophical_lexicon.json` | Evaluates structural `Order`, `Chaos`, `Continuity`, `Discreteness`, and ratifies stable cross-branch chords (`harmony_score >= 0.72`). |

### Breakthrough 4: The 52-Float `<|subliminal_gap|>` Embedding Splice & Stages 1–5 Results
1. **Four Strictly Disjoint Departmental Vectors (`17 + 16 + 16 + 3 = 52` floats, [`src/lmft_lab/subliminal_gap_splice.py`](src/lmft_lab/subliminal_gap_splice.py))**:
   - `doer_vector` (`17` floats), `thinker_vector` (`16` floats), `feeler_vector` (`16` floats), and `cadence_vector` (`3` floats) project into `K = 4` independent virtual token embeddings via `SubliminalGapProjector`.
2. **Identical-Prefix Contrastive LoRA Training (`Stage 2 & Stage 3`, [`src/lmft_lab/train_left_brain_lora.py`](src/lmft_lab/train_left_brain_lora.py))**:
   - Trained `Qwen/Qwen3-4B` (`bfloat16`, `8.127 GB` peak VRAM) on matched pairs with identical pre-gap words and `-100` pre-gap label masking (`outputs/left_brain_lora_eval_report.json`): achieved **`5/6`** positive chord-swap shift on unseen holdout pairs (`mean_net_chord_swap_margin = +0.7775`, `0.000000` unplugged shift, `5/6` greedy continuation change).
3. **Nighttime REM Sleep Growth (`Stage 5: Cycle 1 -> Cycle 2`, [`src/lmft_lab/rem_sleep_cycle.py`](src/lmft_lab/rem_sleep_cycle.py))**:
   - Sequential VRAM unloading kept PyTorch VRAM at `389.93 MB` while `Qwen3.5-9B` consolidated daytime sleep queues (`outputs/rem_sleep_cycle_report.json`).
   - Daytime Parable settlement rose from **`5/8`** (`Gear1=3, Gear2=2, Gear3=3`) on Day 1 to **`7/8`** (`Gear1=5, Gear2=2, Gear3=1`, holding Beat 6 honestly via `social_drift`) on Day 2; Unseen Harbor Bakehouse rose from **`3/4` to `4/4`**; and Cycle 2 unseen holdout margin improved from **`+0.7775` to `+0.8646`** (`zero_calibrated_both_flipped: 3/6 -> 4/6`).

---

## 5. Repository Organization & Separation of Accountability (`Lab` vs. `School` vs. `Local User State`)

To keep Core Architecture debugging, official curriculum training, and personal user tinkering cleanly separated, the repository is organized into distinct layers:

```text
LMFT-Lab/
├── Start-LMFT-Club.bat                  # One-click Windows launcher for the Chess & Sanctuary Desktop App
├── AGENTS.md                            # Anti-delusion, plain-English, and zero-rigging development rules
├── PROJECT_STATE.md                     # Concise (<60 lines) current status & benchmark summary
├── DECISIONS.md                         # Chronological engineering decision log (D001–D052)
├── baselines/
│   └── virgin_dual_hemisphere_v1/       # [LAB / CYCLE 0] SHA-256 locked "Virgin" Core Architecture snapshot
│       ├── virgin_manifest.json         # Pinned SHA-256 hashes (40 nodes, 10 Tier-2 leaves, 2 rules, 0 precedents)
│       ├── virgin_ruleset.json          # Foundational equilibrium rules + structural leaves only
│       ├── branch_archives/             # Empty Cycle 0 Executive, Legislative, and Judicial archives
│       └── contemplation_ledger.json    # Empty Cycle 0 Contemplation Ledger
├── src/lmft_lab/                        # Core Python package
│   ├── birth.py                         # Frozen 40-node founding seed (24 founding leaves)
│   ├── philosophical_abstraction_matrix.py # 5-Tier Color, Element, Chess Piece, & Shadow Pathology matrix
│   ├── virgin_baseline.py               # Cycle 0 Virgin Baseline verifier & loader
│   ├── reflection_classifier.py         # Fuzzy rules, shoulder curves, & Tier 1 -> Tier 2 -> Tier 3 child leaves
│   ├── neural_bridge_classifiers.py     # 3 fast MiniLM cross-encoder bridges (~35 ms)
│   ├── right_brain_equilibrium.py       # Two-pass Right-Brain equilibrium & dissonance self-registry
│   ├── branch_archives.py               # Executive, Legislative, & Judicial archives + 3-branch deliberation
│   ├── subliminal_gap_splice.py         # 52-float SubliminalChordVector & K=4 SubliminalGapProjector
│   ├── train_left_brain_lora.py         # Stage 3 Qwen3-4B LoRA + Corpus Callosum trainer & evaluator
│   ├── three_legged_runtime.py          # Stage 4 3-Gear runtime with live mid-sentence comma scoring (D050)
│   ├── rem_sleep_cycle.py               # Stage 5 Daytime -> REM Sleep -> Next-Day maturation loop
│   ├── chess_kibitz_bridge.py           # Live python-chess Fact sensor & Sanctuary Pacing/Leading helper
│   └── sanctuary_club_app.py            # Local HTTP server & Antigravity-style UI for Chess & Sanctuary
├── models/                              # [SCHOOL / WEIGHTS] Local neural checkpoints (git-ignored .pt weights)
│   ├── right_brain_bridges/             # 3 calibrated MiniLM Right-Brain bridge classifiers
│   ├── left_brain_lora/                 # Cycle 1 Qwen3-4B LoRA + SubliminalGapProjector
│   └── left_brain_lora_cycle2/          # Cycle 2 REM-matured Qwen3-4B LoRA + SubliminalGapProjector
├── outputs/                             # [SCHOOL & USER STATE] Rulesets, evaluation reports, and local app state
│   ├── rem_sleep_consolidated_ruleset.json # Official Cycle 2 consolidated Right-Brain ruleset
│   └── club_user_state/                 # Personal coached rules, archives & savored vibes (git-ignored)
├── scripts/
│   └── build_release_bundle.py          # Builds portable dist/lmft-sanctuary-club-v0.1.0-win64.zip
├── experiments/                         # Live GPU benchmark & curriculum runners
├── tests/                               # 1,369 automated pytest unit & integration tests
└── docs/                                # Architecture specifications, blueprints, and archived history
```

### Why This Layout Prevents Sync Headaches and Blame Confusion
1. **`baselines/virgin_dual_hemisphere_v1/` (`Cycle 0` — "Lab" / Core Architecture)**:
   - Locked with SHA-256 hashes (`verify_virgin_baseline_snapshot()`). If a bug happens in `Cycle 0`, we know immediately it is an oversight in the **Core Architecture** rather than bad training data.
2. **`outputs/rem_sleep_consolidated_ruleset.json` & `models/left_brain_lora_cycle2/` (`Cycle 2` — Official "School" Checkpoint)**:
   - Tracks the official repository's vetted REM-Sleep training progress.
3. **`outputs/club_user_state/` (Personal Local Coaching — Git-Ignored)**:
   - When you or anyone else uses the desktop app to coach turns, savor vibes, and run REM Sleep, all personal progress is saved automatically to `outputs/club_user_state/`. It survives app restarts, never pollutes official benchmarks, and never causes `git pull` conflicts unless someone intentionally forks the repo to publish their own lineage.

---

## 6. Quickstart & Verification (Windows / RTX 4080)

### 1. Launch the Desktop App (`Chess Club & Co-Regulation Sanctuary`)
```powershell
.\Start-LMFT-Club.bat
```

### 2. Run the Full Automated Test Suite (`1,369` Passing Tests)
```powershell
$env:PYTHONPATH='src;.'
.\.venv\Scripts\python.exe -m pytest -q
```

### 3. Run the Targeted Core & App Tests (`< 10 seconds`)
```powershell
$env:PYTHONPATH='src;.'
.\.venv\Scripts\python.exe -m pytest tests/test_virgin_baseline.py tests/test_philosophical_abstraction_matrix.py tests/test_three_legged_runtime.py tests/test_sanctuary_club_app.py -v
```

### 4. Build the Portable GitHub Release Bundle (`dist/lmft-sanctuary-club-v0.1.0-win64.zip`)
```powershell
.\.venv\Scripts\python.exe scripts/build_release_bundle.py
```

---

## 7. Workflow & Anti-Delusion Trust Measures ([`AGENTS.md`](AGENTS.md))

1. **Plain, Everyday English Only**: Zero dense academic jargon or fake bureaucratic roleplay.
2. **Zero Fake AI / Zero Coordinate or Rubric Rigging**: Never hardcode test words into rubrics, never clamp raw sensor coordinates (`Fact`), never zero out competing rules, and never harvest unsettled (`settled=False`) scenes into the archives as ratified rules.
3. **Let Experiments Fail Openly**: Honest partial scores (`4/8`, `5/8`, `7/8`, `3/4`) are reported plainly without moving goalposts.
4. **Dual-Window Independent AI Auditing**: Every commit is written to pass inspection by a separate, independent Auditor AI window.
