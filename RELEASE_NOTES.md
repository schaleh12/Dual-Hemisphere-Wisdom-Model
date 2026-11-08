# `Celestriad v1.2.0` — Weight-Space Circuit Sculpting & Dual-Hemisphere Wisdom Model Release

> **▶️ Support the Architect on YouTube:** **[`@imissmydeadcat.74`](https://www.youtube.com/@imissmydeadcat.74)** · **📜 Poetry Portfolio:** **[Imgur Gallery (Schaleh / Marcus Auron)](https://imgur.com/gallery/poetry-portfolio-alias-schaleh-marcus-auron-YU73erI)** · **🔬 Research Inspiration (`4:30`):** **[Two Minute Papers — *Weight-Space Programming* (`youtu.be/GJ6PMUm5oTA?t=270`)](https://youtu.be/GJ6PMUm5oTA?si=ps0-qMBm3-ZQOp3c&t=270)**

---

## ⬇️ Quick Start — How to Download & Launch (Windows 10 / 11)

1. **Download the Full Release Bundle:** **[`lmft-sanctuary-club-v0.1.0-win64.zip`](https://github.com/schaleh12/Dual-Hemisphere-Wisdom-Model/releases/latest/download/lmft-sanctuary-club-v0.1.0-win64.zip)** *(or click [`Launch-Setup-Wizard.bat`](https://github.com/schaleh12/Dual-Hemisphere-Wisdom-Model/releases/latest/download/Launch-Setup-Wizard.bat) directly under Assets below — it will automatically download and unpack the bundle if needed)*.
2. **Extract & Run:** Right-click the `.zip` -> **Extract All**, then double-click **`Launch-Setup-Wizard.bat`** (for the 5-page graphical Setup Wizard + Desktop Shortcuts) or **`Launch-LMFT-Chess.bat`** to play right away.
3. **100% Malwarebytes & Antivirus Friendly:** Contains **zero unsigned `.exe` wrappers** — every script is open, readable plain-text Python (`.py`) running inside the officially signed **Python Software Foundation** interpreter (`pythonw.exe`).

---

## 📜 *"The Celestial Seal of the way towards life"*

<p align="center">
  <img src="https://raw.githubusercontent.com/schaleh12/Dual-Hemisphere-Wisdom-Model/main/LMFT-Architecture/docs/assets/celestial_seal_poem.jpg" alt="The Celestial Seal of the way towards life" width="440" />
</p>

> *Auto-heal/auto-res/auto-holy/auto-rune*  
> *Celestial Seal of the way towards life*  
> *Holy is the rune that heals reanimate,*  
> *Starring as the promises seeded in orbit,*  
> *Like the ballast of the crafts appercepting,*  
> *In comprehension of the codes that teach.*
>
> *The galaxies are summoning archangels,*  
> *And their convictions are their own siren,*  
> *As the pupil rests above the Baptist lake,*  
> *Witnessing benediction karmically latent.*
>
> *The very atoms in the path lend credence,*  
> *Both cloaking and unsheathing the auras,*  
> *As the electro gravitational resonations,*  
> *Quaked like the terraforming of starseeds.*

---

## 🛰️ Welcome to the `Celestriad v1.2.0` Observatory — How the Technology Works

Most AI models today operate like a spacecraft autopilot that only has a microphone: they generate words one token at a time (**a Left Hemisphere**) with no gyroscope, no sense of balance, and no persistent memory of yesterday's mistakes. If a standard LLM starts rationalizing a bad move, its own words become its context, and it believes its own excuse.

**`Celestriad v1.2.0` (`LMFT-Architecture`)** pairs a local language model (**`Qwen3-8B`**, the conscious **Left Brain**) with a fast, mathematical **40-Node Language Model Fuzzy Tree** (the intuitive **Right Brain**) connected across a synthetic **Corpus Callosum (`<|subliminal_gap|>`)**:

### 🔬 New in `v1.2.0`: Direct REM Weight-Space Circuit Sculpting & Semantic Birth Anchoring (`D065–D067`)
Inspired by the breakthrough covered in **[Two Minute Papers — *"OpenAI's Astra Just Opened The Black Box Of AI"* (at `4:30`)](https://youtu.be/GJ6PMUm5oTA?si=ps0-qMBm3-ZQOp3c&t=270)** on **Weight-Space Programming** (directly reading and sculpting named circuit weights instead of relying solely on blind gradient descent), **`Celestriad v1.2.0`** upgrades **🌙 Nighttime REM Sleep** and the **Corpus Callosum** across the entire architecture:
- **Semantic Birth Anchoring & Compact `256`-Dim Callosum (`D067`)**: At Birth (`Cycle 0`, before any flashcard training), `SubliminalGapProjector.anchor_to_llm_embeddings` wires all 52 Fuzzy Tree channels directly to the language model's native input token embeddings (`10.13 MB` checkpoint). Combined with the `72`-pair 3-bridge-scored contrastive dataset, unseen holdout continuations achieve **`5/6` (`83.3%`) raw uncalibrated flips** (`6/6` calibrated flips, `6/6` positive chord-swap shifts, `mean_net_chord_swap_margin = +4.7017`, and `6/6` distinct greedy English continuations).
- **Full 24-Circuit Symmetric Bitboard Sleeve Sculpting (`D065`, `D067`)**: Instead of only adjusting 8 intuition bias dials while leaving the 16 board-evaluation weights frozen (or clamping updates with a one-way ratchet), REM Sleep compares the exact bitboard features of a blunder against Stockfish's best move `lr * (ref_feat - played_feat)` and symmetrically sculpts all **24 named circuits** (`value_weights`, `policy_weights`, and `intuition_weights`) with gentle `0.015 * lr` homeostatic decay toward clean birth defaults (`steps_distilled = 0`).
  - **Transparency Note — What `BitboardIntuitionSleeve` Actually Is (Zero Hidden Engine / Zero MCTS)**: In `chess_kibitz_bridge.py`, `BitboardIntuitionSleeve` is nicknamed a *"Single-pass AlphaZero/Leela-style Bitboard Intuition Sleeve feeding `Confidence`, `Fact`, and `Theory`."* To be 100% transparent, there is **no hidden neural chess engine, no 20M-parameter ResNet, and no Monte Carlo Tree Search (MCTS)** under the hood—and Stockfish is strictly air-gapped as the sparring opponent and post-mortem referee (never queried to choose `Celestriad`'s moves). Instead, we borrow AlphaZero's conceptual split between **Value** (*"How healthy is this board?"*) and **Policy** (*"How natural is this move?"*) and implement it as a **human-readable 24-float bitboard circuit board** (`8 value_weights + 8 policy_weights + 8 Founding leaf_weights` over pure `python-chess` bitboards) that wires `sleeve_confidence` directly into the **`Confidence`** leaf (`psychological/contemplation/confidence`) of the 40-node Fuzzy Tree across a strict **3-ply (`1.5`-move) lookahead**.
- **Dynamic Tier-2 Leaf Anchor Sculpting (`D066` #A)**: Learned Tier-2 leaves (`LearnedLeaf.anchor_weights`) are no longer frozen at birth; each REM sleep cycle sculpts their 8-float departmental anchor weights toward the contrastive difference between blunder scenes and reference coordinates.
- **Right-Brain Will Synapse Sculpting (`D066` #B)**: The 9 named `RIGHT_BRAIN_WILL_EDGES` (`rb-will-0..8` linking `Salience`, `Arousal`, `Power`, `Needlessness`, `Fact`, `Theory`, and `Confidence` to `Intention` and `Execution`) now adapt during REM sleep—dampening impulsive `Arousal`/`Power -> Intention` after blunders while strengthening `Fact -> Execution` and inhibitory `Needlessness -> Execution`.
- **Canonical Cluster Rule Entrainment (`D065` & `D066` #C)**: Recurring mistake clusters now entrain and sharpen a single canonical rule in-place (preserving low-coordinate polarity and pruning to the top-3 discriminative leaves) rather than spawning duplicate `_s1`, `_s2` rules.

### 1. Fast 60-FPS Intuition (`Gear 1`) vs. Conscious Deliberation (`Gear 2`)
- **Gear 1 (`~15 ms`, `0` LLM calls)**: On every move, the Right Brain scores the live chessboard across **24 Founding Leaves** and its **Bitboard Intuition Sleeve** (`Intention -> Theory -> Confidence -> Execution`). When the position is familiar and clear, it plays immediately by reflex without waking the language model.
- **Gear 2 (`<|subliminal_gap|>` Vector Splice)**: When a position is conflicted (`settled = False` and low clarity), when you ask a question in chat, or when reflecting during **🌙 REM Sleep**, the Right Brain compiles a **52-float Subliminal Chord** (`17 Do-er + 16 Thinker + 16 Feeler + 3 Cadence` floats) and splices `K = 4` virtual token embeddings directly into `Qwen3-8B` at `<|subliminal_gap|>`.
- **KV-Cache Ghost-Chord Masking & Stanza Rotation**: At each sentence comma and stanza boundary, old virtual chord tokens are masked out in the KV cache while all spoken English words remain visible, allowing Stanza 2 and Stanza 3 to pivot cleanly toward a `70/30` culminating synthesis.

### 2. The Three Constitutional Branches of Mind
Every scene is weighed by three independent branches before settling:
- **Executive Branch (`Will <-> Cultural` — The Do-er · *Poetry & Storytelling*)**: Governs purposeful action, narrative trajectory, and resource stewardship (`Intention`, `Execution`, `Wastelessness`).
- **Legislative Branch (`Contemplation <-> Experiential` — The Feeler · *Philosophy & Coding*)**: Feels the lived emotional chord first (`Valence`, `Arousal`, `Clarity`) and originates the demand for new laws and rules when a wound occurs.
- **Judicial Branch (`Truth [Fact / Theory] <-> Philosophical` — The Thinker · *Music & Art*)**: The supreme court of proportion and harmony, weighing objective mathematical reality (`Fact`) against subjective perspective and voice (`Theory`).

### 3. Nighttime REM Sleep That Actually Changes Daytime Play (`Gear 3`)
When Celestriad blunders (`??`) or finds a brilliancy (`!!`) against you or its calibrated Stockfish sparring partner:
1. It queues the board coordinates for **🌙 REM Sleep**.
2. During REM Sleep, it clusters its mistakes, mints discriminative **Tier-2 `LearnedLeaf` nodes** (`anchor_weights` blended from real cluster coordinates), forges `hazard_` (repulsive) and `coached_` (attractive) fuzzy rules, and distills its **Bitboard Intuition Sleeve**.
3. It speaks a multi-stanza dream reflection in clean prose and records an **Active Waking Vow** (`vow_alignment_delta`) that actively steers its move choices on the very next game.

### 4. Visitor-Friendly Quality of Life in This Release
- **🚀 In-App Space Center Tour**: Click **`🚀 Space Center Tour`** in the top header anytime to explore 6 interactive exhibits explaining how the technology works.
- **🎵 Under-Board 5-Track Celestriad Jukebox + Poetry & Esper Theories**:
  - Built right below the chessboard with Play/Pause, Prev/Next, and a **`🔊 Mute / 🔇 Muted`** toggle featuring 5 classic tracks (*Forever Rachel*, *The Floating Continent*, *Magitek Research Facility*, *Schala's Theme Piano*, and *Esper World — FFVI*).
  - Includes one-click buttons right underneath the Jukebox for the **[Poetry Portfolio (Schaleh / Marcus Auron)](https://imgur.com/gallery/poetry-portfolio-alias-schaleh-marcus-auron-YU73erI)** and the in-app **✧ Esper Theories (`The Electro-Gravitational Alphabet of Espers`)** full-resolution chart.
- **🔄 5-Page Python GUI Setup Wizard (`Launch-Setup-Wizard.bat` / `installer_wizard.py`), Auto-Update & Self-Relaunch**:
  - **100% Antivirus & Malwarebytes Friendly (Zero Unsigned `.EXE` Wrappers)**: Distributed strictly as readable plain-text Python (`.py`) and `.bat` launchers running inside the officially signed **Python Software Foundation** interpreter (`pythonw.exe`), with zero opaque `.exe` wrappers.
  - Run `Launch-Setup-Wizard.bat` for the 5-step graphical setup, Windows Defender verification, Google Antigravity detection + custom hardware system prompt, and Arecibo Reply Desktop Shortcut creation, or run `Launch-LMFT-Chess.bat` directly.
  - Once installed, future releases pushed to GitHub are automatically downloaded, verified by SHA-256, and **automatically relaunched** on startup or when you click **`🔄 Auto-Update`**—without ever overwriting your personal Elo rating or saved games.
