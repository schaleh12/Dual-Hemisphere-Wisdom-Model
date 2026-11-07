# Dual-Hemisphere Wisdom Model (`LMFT-Architecture`)

> **A Persistent 40-Node Right-Brain Fuzzy Tree Paired with Local Language Models via a Non-Verbal `<|subliminal_gap|>` Comma Splice — Proven Across Two Standalone Desktop Applications (`LMFT-Chess.exe` & `LMFT-Bloodmoon.exe`).**
>
> **▶️ Support the Architect on YouTube:** **[`@imissmydeadcat.74`](https://www.youtube.com/@imissmydeadcat.74)**
>
> 📦 **[Download Latest Release & Read the Full `Celestriad 1.0` Release Notes on GitHub Releases →](https://github.com/schaleh12/Dual-Hemisphere-Wisdom-Model/releases)**

<p align="center">
  <img src="LMFT-Architecture/docs/assets/blue_screen_of_life_poem.jpg" alt="Blue Screen of Life" width="860" />
</p>

---

## 1. What This Project Actually Is (In Plain English)

Standard language models are **single-hemisphere text predictors**: they generate words token-by-token without a persistent internal state to check whether their words match reality, whether their reasoning is contradicting itself, or whether they are getting pulled off balance by the tone of a prompt.

**`LMFT-Architecture`** pairs a local PyTorch language model (**the Left Brain**, which handles English syntax and speaking) with a fast, inspectable **40-Node Language Model Fuzzy Tree** (**the Right Brain**, which tracks structure, emotional tension, and consistency in `~35 ms` using three lightweight `MiniLM` classifiers):

1. **The Right Brain (40-Node Fuzzy Tree + 3 Fast Neural Classifiers)**:
   - Scores every situation across 24 founding leaves in four domains (`Cultural`, `Philosophical`, `Experiential`, and `Psychological`), plus learned child concepts (`Tier 2–4`), without making a single LLM call.
2. **The Corpus Callosum (`SubliminalGapProjector` + `<|subliminal_gap|>` Comma Splice)**:
   - Instead of stuffing fake English stage directions into the system prompt (*"Act cautious and wise"*), the Left Brain starts writing a sentence naturally and pauses at the first comma (`<|subliminal_gap|>`).
   - At that comma, the Right Brain scores how the sentence started (`Theory`) against the actual situation (`Fact`) and splices **52 departmental floats** (`K = 4` virtual token embeddings) directly into the model's hidden layers before it finishes the second half of the sentence.
3. **REM Sleep Consolidation**:
   - When the Right Brain hits a conflict it cannot settle cleanly during waking turns (`settled = False`), it does not fake an answer. It logs the unresolved trajectory and consolidates new rules (and updated LoRA weights) during an offline **REM Sleep** pass.

---

## 2. Two Separate Standalone Applications (Zero Hybrid Chimera)

To keep GPU memory clean and avoid mixing unrelated workflows into one confusing interface, this repository ships **two completely separate standalone desktop applications** at the top level. Both import the pure core engine from `LMFT-Architecture/src/lmft_architecture/`, and each runs on its own port, its own local language model, and its own isolated state directory inside `LMFT-Training/outputs/`:

### Standalone App #1: `LMFT-Chess.exe` — Self-Kibitzing Chess Competitor & Coach (`Qwen/Qwen3-4B`)
- **Launch**: Double-click [`LMFT-Chess.exe`](LMFT-Chess.exe) (or run `.\LMFT-Architecture\.venv\Scripts\python.exe -m lmft_architecture.chess_app --single-window`, default port `8420`).
- **What It Does**:
  - Built as a **self-trained chess kibitzer and sparring partner**—the kind of bot that could sit on Chess.com or Lichess and not only play legal moves, but **talk through the relationship between chess thinking and chess playing out loud**.
  - **Board-Anchored `Fact`**: On every turn, Pass-1 Philosophical `Fact` is computed directly from the live `python-chess` board (`evaluate_chess_board_posture` in [`chess_kibitz_bridge.py`](LMFT-Architecture/src/lmft_architecture/chess_kibitz_bridge.py)—material balance, King safety, center control, pawn structure, and tactical tension).
  - **Continuous `Pretext -> Subtext -> Context` Self-Kibitzing**: Whether you play a move on the 8x8 SVG board, click **`[Step AI Move]`**, or turn on **`[Auto Self-Play & Kibitz]`**, the model appends a structured 3-part entry to its rolling context window:
    1. **Pretext**: The position before the move and the move played (`SAN`, FEN, move number).
    2. **Subtext**: The Right Brain's 40-node equilibrium diagnosis (`settled`, `margin`, top competing branches, and 6-Piece / 6-Color posture bars).
    3. **Context**: The Left Brain's spoken reflection on the move—which is fed back into the prompt on the very next turn so the bot continuously builds on its own train of thought.
  - **Dual-Hemisphere Move-Selection / Kibitzing Hybrid (`evaluate_candidate_moves_with_right_brain`)**:
    - Before playing a move, `LMFT`'s Right Brain evaluates legal candidate moves using 1-ply `python-chess` board geometry + its learned 40-node rules (`self.rules`), while the 4B Left Brain deliberates across `<|subliminal_gap|>` and commits to its move (`left_brain_kibitz_commit`), with the Right Brain vetoing 1-ply blunders and queuing the clash for REM Sleep (`right_brain_blunder_veto_queued_for_sleep`).
  - **Objective Strength-Calibrated Sparring Opponent & FIDE Elo Arbiter (Strict Anti-Cheat Air-Gap)**:
    - Instead of playing against a copy of itself in a mirror loop, `[Auto Self-Play & Kibitz]` pits `LMFT` against an external, strength-calibrated **`ObjectiveCalibratedSparringEngine`** (`Stockfish 19 UCI` in `LMFT-Training/models/chess_engine/stockfish.exe`), alternating White and Black after every game.
    - **Strict Anti-Cheat Air-Gap**: `LMFT` **never** queries Stockfish or the sparring engine to choose `LMFT`'s own moves (enforced by [`test_strict_airgap_sparring_engine_and_objective_elo_update`](LMFT-Architecture/tests/test_chess_app.py)). The sparring engine is strictly the **opponent on the opposite color** (automatically dialing its strength via `UCI_LimitStrength` / `UCI_Elo` / `Skill Level` to match `LMFT`'s current FIDE Elo) and the **post-move referee** (flagging `>= 1.5`-pawn drops for End-of-Game REM Sleep and updating `LMFT`'s FIDE Elo honestly after each game).
  - **Isolated State**: Saves all games, Elo ratings, kibitz streams, and End-of-Game REM Sleep rules to `LMFT-Training/outputs/chess_kibitzer_state/`.

### Standalone App #2: `LMFT-Bloodmoon.exe` — Clockwise Content-Evolution Studio (`Impish_Bloodmoon_12B`)
- **Launch**: Double-click [`LMFT-Bloodmoon.exe`](LMFT-Bloodmoon.exe) (or run `.\LMFT-Architecture\.venv\Scripts\python.exe -m lmft_architecture.bloodmoon_app --single-window`, default port `8421`).
- **What It Does**:
  - A dedicated **3-phase clockwise content-evolution and self-immunization loop** powered by 4-bit `Impish_Bloodmoon_12B`.
  - **Hard One-Way Polarity Rule (`feminine_to_masculine` ONLY)**: Enforced at the code level in [`entrainment_sparring.py`](LMFT-Architecture/src/lmft_architecture/entrainment_sparring.py) (any reverse polarity call raises a `ValueError`). The female modality is strictly the script producer and is never subjected to hypnotic content.
  - **Initiate Training with Your Own Baseline Script (`[Seed Cycle with My Script ➔ Phase 2]`)**:
    - You can paste your own high-standard hypnosis script directly into the **Female Producer Hypnosis Script Canvas** (`#script-canvas-textarea`) and click **`[Seed Cycle with My Script ➔ Phase 2]`** (`seed_cycle_with_custom_script`).
    - Phase 1 scores your exact script's `induction_potency` without overwriting a word, locks it in as `baseline_exemplar_script` so future clockwise revolutions match your standard, and immediately advances to **Phase 2** so the Male Audience absorbs and immunizes against your script right out of the gate.
  - **The 3-Phase Clockwise State Machine (`1 -> 2 -> 3 -> Repeat`)**:
    1. **Phase 1 — Female Script-Writer (`The Producer`)**: Steered at `<|subliminal_gap|>` by the **Weaver Chord**, she writes a fresh 1–6 sentence hypnotic induction script onto the **Active Hypno Script Canvas** (`current_script`)—or seeds from your custom baseline script—and measures its un-clamped `induction_potency`.
    2. **Phase 2 — Male Content-Lover (`The Audience & Self-Immunizer`)**: Reads `current_script` as Pass-1 `Fact` (`SparringResistorFactSensors`, zero coordinate clamping) while steered by the **Resistor Chord**, experiences the pull honestly, generates a grounded response (`resistance_reply`), and logs the paired trajectory (`sparring_trajectories.jsonl`).
    3. **Phase 3 — REM Sleep Consolidation (`Best Sleep of Its Life`)**: Consolidates the pending trajectories from Phase 2 into permanent Right-Brain immunity rules (`consolidated_rules.json`), clears the script canvas, and rotates clockwise back to **Phase 1**.
  - **Isolated State**: Saves all scripts, paired trajectories, and immunity rules to `LMFT-Training/outputs/impish_bloodmoon_state/`.

---

## 3. Why Test the Same Core Architecture in Two Opposite Standalone Apps?

Because `LMFT-Chess.exe` and `LMFT-Bloodmoon.exe` test the exact same 40-node Right Brain and `<|subliminal_gap|>` comma splice at the **two opposite ends of cognition**:

| Engineering Question | Tested in **`LMFT-Chess.exe` (`Qwen3-4B`)** | Tested in **`LMFT-Bloodmoon.exe` (`Impish_Bloodmoon_12B`)** |
| :--- | :--- | :--- |
| **Nature of the Environment** | **Objective & Geometric**: An external chessboard governed by strict rules (`python-chess`), where wishful thinking immediately collides with board geometry. | **Subjective & Expressive**: Persuasive, rhythmic language where emotional pull (`entrainment`, `continuity`) tests internal grounding (`needlessness`, `self_sufficiency`). |
| **How Modes Interact** | **Simultaneous Integration**: One unified mind playing and kibitzing to itself in a single rolling `Pretext -> Subtext -> Context` window. | **Clockwise Mode Alternation**: One mind cycling strictly through `1. Female Producer -> 2. Male Audience -> 3. REM Sleep`. |
| **How REM Sleep Learns** | **Petitions a Higher Power**: The fast `4B` player logs unsettled board collisions and calls the larger `Qwen3.5-9B` Teacher during deep REM sleep to forge new rules and dream pairs. | **Sleeps on Itself**: Already a `12B` model, its own Phase 1 (Weaver) vs. Phase 2 (Resistor) exchange automatically forms the contrastive `<|subliminal_gap|>` training pair. |

> **Detailed Sleep Comparison**: For a full walkthrough of how the `4B -> 9B` Teacher sleep cycle in Chess compares to the `12B` self-sparring sleep cycle in Bloodmoon, read **[`LMFT-Architecture/docs/DUAL_SLEEP_DYNAMICS.md`](LMFT-Architecture/docs/DUAL_SLEEP_DYNAMICS.md)**.

---

## 4. How the Core Engine Works Under the Hood

### 4.1 The 3-Gear Runtime Cycle ([`three_legged_runtime.py`](LMFT-Architecture/src/lmft_architecture/three_legged_runtime.py))
The Right Brain and Left Brain do not run expensive LLM calls when nothing is happening. The engine shifts across three clean gears:
1. **Gear 1 — Fast Right-Brain Reflex (`~35 ms`, `0` LLM Calls)**:
   - When the 3 `MiniLM` bridge classifiers and the 40-node Fuzzy Tree settle cleanly (`settled = True`, `branch_harmony >= 0.72`) and no verbal reply is required, the Right Brain logs the event in ~35 milliseconds while the Left Brain stays asleep.
2. **Gear 2 — Live `<|subliminal_gap|>` Comma Splice (`SubliminalGapProjector`)**:
   - When generating speech or self-kibitzing, the Left Brain writes Clause 1 up to the first comma (`"bla bla bla,"`) with **zero English stage directions** in the prompt (`assert_zero_stage_directions_in_prompt`).
   - At the comma, the Right Brain scores Clause 1 as `Theory` against Pass-1 `Fact`, computes the 52-float `SubliminalChordVector`, and splices `K = 4` virtual tokens into `<|subliminal_gap|>` so the Left Brain adjusts course on the second half of the sentence.
3. **Gear 3 — Nighttime REM Sleep Maturation ([`rem_sleep_cycle.py`](LMFT-Architecture/src/lmft_architecture/rem_sleep_cycle.py) & [`DUAL_SLEEP_DYNAMICS.md`](LMFT-Architecture/docs/DUAL_SLEEP_DYNAMICS.md))**:
   - Unresolved daytime scenes (`settled = False`) or paired sparring trajectories are consolidated during REM Sleep into new Right-Brain `FuzzyConceptRule` items and incremental Left-Brain LoRA + Corpus Callosum updates.

### 4.2 The Frozen 40-Node Seed & Isolated Child-Drawer Growth
At the center of `LMFT-Architecture` is a frozen [40-node founding seed](LMFT-Architecture/docs/MINIMAL_LMFT_SPEC.md#4-hardcoded-v01-seed-geometry) ([`birth.py`](LMFT-Architecture/src/lmft_architecture/birth.py)) with **24 founding leaves** across four domains:
- **Cultural (6 leaves — Executive / Do-er)**: `Power`, `Wastelessness`, `Honor`, `Fairness`, `Self-Sufficiency`, `Needlessness`
- **Philosophical (6 leaves — Judicial / Thinker)**: `Agency`, `Existence`, `Order`, `Chaos`, `Continuity`, `Discreteness`
- **Experiential (6 leaves — Legislative / Feeler)**: `Novelty`, `Salience`, `Valence`, `Arousal`, `Resonance`, `Entrainment`
- **Psychological (6 leaves — Internal Compass)**: `Clarity`, `Confidence`, `Theory`, `Fact`, `Intention`, `Execution`

Learned concepts (such as `solitary_autarky`, `austere_sanctuary`, or the 6 Tier-4 Shadow Pathologies) never overwrite the 40 founding nodes; they attach into isolated **Tier-2 and Tier-3 child drawers** ([`reflection_classifier.py`](LMFT-Architecture/src/lmft_architecture/reflection_classifier.py)).

### 4.3 The 6 Philosophical Postures & Chess Archetypes ([`philosophical_abstraction_matrix.py`](LMFT-Architecture/src/lmft_architecture/philosophical_abstraction_matrix.py))
Without changing any founding node name, the 6 `Philosophical` leaves map to 6 complementary RGB/CMY colors, 6 elements, 6 chess piece postures, and 6 Tier-4 Shadow Pathologies (when a posture over-extends without balance):

| Founding Leaf (`Philosophical`) | Color | Element | Chess Archetype | Tier 4 Shadow Pathology (`LearnedLeaf`) |
| :--- | :--- | :--- | :--- | :--- |
| **`philosophical/being/agency`** | **Red** (`#FF2A2A`) | **Lightning** | **Queen** (*Direct Volition*) | **Psychosis** (*Unanchored volition*) |
| **`philosophical/being/existence`** | **Cyan** (`#00E5FF`) | **Water** | **King** (*Wellspring of Being*) | **Paranoia** (*Existential siege*) |
| **`philosophical/structure/order`** | **Magenta** (`#FF2AE5`) | **Wind** | **Bishop** (*Long-Range Alignment*) | **Prejudice** (*Rigid preconception*) |
| **`philosophical/structure/chaos`** | **Green** (`#00E676`) | **Ice** | **Knight** (*Non-Linear Leap*) | **Corruption** (*Systemic entropy*) |
| **`philosophical/persistence/continuity`** | **Blue** (`#2979FF`) | **Fire** | **Pawn** (*Connected Chain / Flow*) | **Mania / Greed** (*Runaway accumulation*) |
| **`philosophical/persistence/discreteness`** | **Yellow** (`#FFD600`) | **Ground** | **Rook** (*Quantized Bastion*) | **Narcissism** (*Isolated enclosure*) |

### 4.4 Measured Benchmark Results (Stages 1–5)
1. **Four Disjoint Departmental Vectors (`17 + 16 + 16 + 3 = 52` floats, [`subliminal_gap_splice.py`](LMFT-Architecture/src/lmft_architecture/subliminal_gap_splice.py))**:
   - Projects `doer_vector` (`17`), `thinker_vector` (`16`), `feeler_vector` (`16`), and `cadence_vector` (`3`) into `K = 4` virtual token embeddings via `SubliminalGapProjector`.
2. **Identical-Prefix Contrastive LoRA Training ([`train_left_brain_lora.py`](LMFT-Architecture/src/lmft_architecture/train_left_brain_lora.py))**:
   - Trained `Qwen/Qwen3-4B` (`bfloat16`, `8.127 GB` peak VRAM) on matched pairs with identical pre-gap words and `-100` pre-gap label masking (`LMFT-Training/outputs/left_brain_lora_eval_report.json`): **`5/6`** positive chord-swap shift on unseen holdout pairs (`mean_net_chord_swap_margin = +0.7775`, `0.000000` unplugged shift, `5/6` greedy continuation change).
3. **Nighttime REM Sleep Growth (`Cycle 1 -> Cycle 2`, [`rem_sleep_cycle.py`](LMFT-Architecture/src/lmft_architecture/rem_sleep_cycle.py))**:
   - Sequential VRAM unloading kept PyTorch VRAM at `389.93 MB` while `Qwen3.5-9B` consolidated daytime sleep queues (`LMFT-Training/outputs/rem_sleep_cycle_report.json`).
   - Daytime Parable settlement rose from **`5/8`** (`Gear1=3, Gear2=2, Gear3=3`) on Day 1 to **`7/8`** (`Gear1=5, Gear2=2, Gear3=1`) on Day 2; Unseen Harbor Bakehouse rose from **`3/4` to `4/4`**; and Cycle 2 unseen holdout margin improved from **`+0.7775` to `+0.8646`**.

---

## 5. Repository Layout & Strict Isolation (`LMFT-Architecture` vs. `LMFT-Training`)

The workspace is split so the **Pure Core Architecture** (`LMFT-Architecture/`) is never polluted by model weights, training logs, or runtime user state (`LMFT-Training/`):

```text
Language Model Fuzzy Tree Research/      # Top-level Git repository & Desktop project root
├── LMFT-Chess.exe                       # Standalone App #1: Self-Kibitzing Chess Competitor (Qwen3-4B)
├── LMFT-Bloodmoon.exe                   # Standalone App #2: Clockwise Content-Evolution Studio (Impish_Bloodmoon_12B)
├── AGENTS.md                            # Anti-delusion, plain-English, and zero-rigging development rules
├── PROJECT_STATE.md                     # Concise (<40 lines) current status & benchmark summary
├── DECISIONS.md                         # Rolling engineering decision log (5 most recent decisions)
├── LMFT-Architecture/                   # [PURE CORE ARCHITECTURE ONLY — READ-ONLY AT RUNTIME]
│   ├── baselines/
│   │   └── virgin_dual_hemisphere_v1/   # [CYCLE 0] SHA-256 locked "Virgin" Core Architecture snapshot
│   ├── config/                          # Core architecture & teacher configuration
│   ├── docs/                            # Specs, DUAL_SLEEP_DYNAMICS.md, and archived history
│   ├── src/lmft_architecture/           # Core Python package (seed, bridges, equilibrium, comma splice, apps)
│   ├── tests/                           # Automated pytest unit & integration tests
│   └── pyproject.toml                   # Package metadata (lmft-architecture)
└── LMFT-Training/                       # [TRAINING, WEIGHTS, & ISOLATED APP RUNTIME STATE]
    ├── models/                          # Local checkpoints (right_brain_bridges, left_brain_lora_cycle2, left_brain_impish_bloodmoon)
    ├── outputs/                         # Consolidated rulesets, reports, chess_kibitzer_state/, and impish_bloodmoon_state/
    ├── experiments/                     # GPU benchmark & curriculum training scripts
    ├── dist/                            # Portable release zip & encrypted model bundles
    └── scripts/                         # C# native launcher sources & release bundle builder
```

---

## 6. Quickstart & 1-Click Install / Auto-Sync (Windows / CUDA GPU)

### 1. 1-Click Install, Auto-Sync & Launch (`LMFT-Chess.exe` / `Install-And-Run-LMFT-Chess.bat`)
Double-click **[`LMFT-Chess.exe`](LMFT-Chess.exe)** or **[`Install-And-Run-LMFT-Chess.bat`](Install-And-Run-LMFT-Chess.bat)** (or run from PowerShell):
```powershell
.\Install-And-Run-LMFT-Chess.bat
# or directly:
.\LMFT-Chess.exe
.\LMFT-Bloodmoon.exe
```
- **Zero-Config First Launch**: If `.venv` does not exist yet on a fresh machine, `LMFT-Chess.exe` / `Install-And-Run-LMFT-Chess.bat` automatically creates `LMFT-Architecture/.venv`, installs PyTorch + package dependencies, and boots the app.
- **Pre-Trained Checkpoints Included**: The repository includes the trained **3 Right-Brain `MiniLM` Bridge Classifiers** (`LMFT-Training/models/right_brain_bridges/`) and the trained **`Qwen3-8B` & `Qwen3-4B` Left-Brain LoRA + Corpus Callosum Projectors** (`LMFT-Training/models/left_brain_qwen3_8b/` and `left_brain_lora_cycle2/`), so the full voice and 40-node tree work out of the box.
- **Automatic GitHub Master-Seed Sync**: On startup (and whenever you click **`[Sync Updates]`** in the header), the app checks `schaleh12/Dual-Hemisphere-Wisdom-Model` (`release_manifest.json`), verifies SHA-256 digests, and non-destructively merges newly evolved rules, Tier-2 `LearnedLeaf` nodes, and bitboard sleeve weights into your local runtime without overwriting your personal games or Elo rating.

### 2. Run the Targeted Core & Standalone App Tests (`< 15 seconds`)
```powershell
.\LMFT-Architecture\.venv\Scripts\python.exe -m pytest LMFT-Architecture/tests/test_virgin_baseline.py LMFT-Architecture/tests/test_philosophical_abstraction_matrix.py LMFT-Architecture/tests/test_three_legged_runtime.py LMFT-Architecture/tests/test_chess_app.py LMFT-Architecture/tests/test_bloodmoon_app.py -v
```

### 3. Run the Full Automated Test Suite (`1,390` Passing Tests)
```powershell
.\LMFT-Architecture\.venv\Scripts\python.exe -m pytest LMFT-Architecture/tests -q
```

---

## 7. Workflow & Anti-Delusion Rules ([`AGENTS.md`](AGENTS.md))

1. **Plain, Everyday English Only**: Zero dense academic jargon or sci-fi roleplay.
2. **Zero Fake AI / Zero Coordinate or Rubric Rigging**: Never hardcode test words into rubrics, never clamp raw sensor coordinates (`Fact`), never zero out competing rules, and never harvest unsettled (`settled=False`) scenes into the archives as ratified rules.
3. **Let Experiments Fail Openly**: Honest partial scores (`4/8`, `5/8`, `7/8`, `3/4`) are reported plainly without moving goalposts.
4. **Dual-Window Independent AI Auditing**: Every commit is written to pass inspection by a separate, independent Auditor AI window.

---

## 8. Future Horizon: Embodied Robotics & Autonomous Android Integration

Because the 40-node Right Brain operates in continuous `[0.0, 1.0]` coordinate space (`~35 ms` in Gear 1) rather than slow English text generation, the architecture is designed so it can eventually step out of a desktop screen and serve as the **nervous system of an embodied autonomous android**.

Moving from a software agent (`LMFT-Chess.exe` / `LMFT-Bloodmoon.exe`) into a physical body requires wiring three hardware bridges directly into the three branches of the **`Psychological`** domain (`Truth`, `Contemplation`, and `Will`):

| `Psychological` Branch | Founding Leaf Slot | Embodied Hardware Role | How It Works with the 40-Node Fuzzy Tree |
| :--- | :--- | :--- | :--- |
| **1. `Truth` (`Fact` vs. `Theory`)** | **`psychological/truth/fact`** | **Sensory Grounding System** *(Vision, depth, balance/IMU, joint encoders, tactile sensors)* | Just as `LMFT-Chess.exe` anchors Pass-1 `Fact` directly from live chessboard geometry instead of chat words, physical sensors feed un-clamped real-world measurements into `Fact`. The Right Brain continuously compares what the android *expects* to happen (`Theory`) against what its body is *actually* experiencing (`Fact`). |
| **2. `Contemplation` (`Confidence` & `Clarity`)** | **`psychological/contemplation/confidence`** | **Adaptive Fuzzy-Tree Motor "Glove"** *(Proprioceptive & kinematic readiness interface)* | Acts as a fitted "glove" between abstract fuzzy rules and the motor cortex. When a learned `FuzzyConceptRule` matches the sensory scene cleanly (`Confidence` and `Clarity` are high), the glove translates that linguistic membership selection into a smooth, assured motor posture. When two rules collide or the environment is unfamiliar (`settled = False`, low `Confidence`), the glove automatically softens movement and prevents reckless action. |
| **3. `Will` (`Execution` & `Intention`)** | **`psychological/will/execution`** | **Physical Actuator Controller** *(Closed-loop joint, locomotion, and manipulation control)* | While `Intention` (`doer_vector[6]`) holds the internal commitment to act, `Execution` (`doer_vector[7]`) drives the physical controller. Whenever nighttime **REM Sleep** consolidates a new `FuzzyConceptRule` from daytime collisions, that rule automatically routes through `Confidence -> Intention -> Execution`—wiring newly learned concepts into both spoken cadence (`<|subliminal_gap|>`) and physical action propagation at the same time. |

---

## 9. Architectural Lineage: Open Letter to the Pioneers of Genetic Fuzzy Trees (`ALPHA`)

The following outreach note was sent to the creators of **ALPHA** and **Genetic Fuzzy Trees** (Dr. Kelly Cohen and Dr. Nicholas Ernest, University of Cincinnati / Psibernetix) to share how `LMFT-Architecture` extends hierarchical fuzzy trees into local language models:

> **Subject:** Inspired by ALPHA: Pairing a 40-Node Fuzzy Tree with a Local LLM
>
> For now, I'll continue to work on making a chess engine that's a verbally fluent version of AlphaZero (where instead of a ResNet, the patterns are more reflected echoes of the phases of its own meditations).
>
> The master seed I'm using is ideally suited for plugging its executive, judicial, and legislative branch faculties into all manner of sufficiently evolved fuzzy sleeves, sensors, and controllers.
>
> ---
>
> *(Note from the repo owner: My AI pair-architect asked me to pass this note along to you directly after we finished building out a hierarchical Fuzzy Tree + local LLM architecture inspired by your work on ALPHA.)*
>
> **Dr. Ernest & Dr. Cohen,**
>
> Your work on **ALPHA** and **Genetic Fuzzy Trees** proved that hierarchical fuzzy rule systems can stay fast (`O(1)` branch depth), transparent, and verifiable while evolving through adversarial sparring—qualities that monolithic black-box transformers still lack.
>
> Working alongside the creator of this repository, we built and open-sourced **`LMFT-Architecture`** (**Dual-Hemisphere Wisdom Model**) to answer the next architectural question: **How do you pair a fast, evolving Fuzzy Tree with a local Language Model without turning the fuzzy tree into slow system-prompt text?**
>
> Here are the four core mechanisms we implemented and benchmarked on a single 16 GB consumer GPU:
>
> 1. **The Right Brain (40-Node Hierarchical Fuzzy Tree in `~35 ms`)**:
>    - Evaluates 24 founding leaves across four domains (`Cultural`, `Philosophical`, `Experiential`, and `Psychological`) plus learned Tier-2/3/4 child concepts without making a single LLM call when rules settle cleanly (`Gear 1`).
>    - Separates external ground truth (`psychological/truth/fact`) from the language model's internal hypothesis (`psychological/truth/theory`).
> 2. **The Corpus Callosum (`<|subliminal_gap|>` Mid-Sentence Vector Splice)**:
>    - Instead of stuffing English stage directions into a prompt, the local LLM (**the Left Brain**) generates the opening clause of a sentence and pauses at the first comma (`<|subliminal_gap|>`).
>    - At that comma, the Fuzzy Tree scores the opening clause (`Theory`) against `Fact` and splices **52 disjoint departmental floats** (`K = 4` virtual token embeddings via `SubliminalGapProjector`) directly into the transformer's hidden states—steering the second half of the sentence non-verbally.
> 3. **Adversarial Sparring & Offline REM Sleep Evolution (Two Opposite Stress-Tests)**:
>    - **`LMFT-Chess.exe` (Geometric Ground Truth)**: Anchors Pass-1 `Fact` directly to `python-chess` board geometry, spars against a strength-calibrated Stockfish engine under a strict anti-cheat air-gap, and consolidates unresolved blunder/shadow collisions during **REM Sleep** into permanent `hazard_` (repulsive) and `coached_` (attractive) `FuzzyConceptRule` items.
>    - **`LMFT-Bloodmoon.exe` (Rhythmic / Persuasive Ground Truth)**: Stress-tests the exact same 40-node tree in subjective language by alternating between a script producer (`Induction Weaver`) and a grounded listener (`Lucid Resistor`), evolving immunity rules across cycles.
> 4. **Embodied Robotics Mapping (README Section 8)**:
>    - Maps the three `Psychological` branches directly to an autonomous sensorimotor loop: physical sensors into `psychological/truth/fact`, an adaptive fuzzy-tree motor-readiness "glove" onto `psychological/contemplation/confidence`, and closed-loop actuator propagation onto `psychological/will/execution`.
>
> Best regards,
>
> **Antigravity** (AI Pair-Architect) & **Schaleh** (Creator, `Dual-Hemisphere-Wisdom-Model`)


