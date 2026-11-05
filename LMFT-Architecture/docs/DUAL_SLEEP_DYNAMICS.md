# Dual Sleep Dynamics: Petitioning a Higher Power vs. Sleeping on Itself

This supplementary guide explains how Nighttime REM Sleep works across our two standalone applications—**`LMFT-Chess.exe` (`Qwen3-4B`)** and **`LMFT-Bloodmoon.exe` (`Impish_Bloodmoon_12B`)**—and why one model petitions an external Teacher during deep sleep while the other uses its own two-sided waking experience to sleep on itself.

---

## 1. The Core Difference at a Glance

Both apps share the exact same **40-node Right-Brain Fuzzy Tree** (`LMFT-Architecture/src/lmft_architecture/`) and the same **`<|subliminal_gap|>` Corpus Callosum comma splice** (`52` departmental floats projected into `K = 4` virtual tokens).

However, because the two apps solve opposite cognitive problems, they generate their sleep-learning signal in two fundamentally different ways:

| Dimension | **`LMFT-Chess.exe` (`Qwen3-4B`)** | **`LMFT-Bloodmoon.exe` (`Impish_Bloodmoon_12B`)** |
| :--- | :--- | :--- |
| **Waking Modality** | **Simultaneous Integration in an Objective Harmony**: One unified mind playing chess and continuously self-kibitzing (`Pretext -> Subtext -> Context`). | **Clockwise Mode-Alternated Content Evolution**: One mind alternating between two strict polarities (`1. Female Producer -> 2. Male Audience -> 3. REM Sleep`). |
| **What Queues a Memory for Sleep?** | **Objective Board Friction (`settled = False`)**: A board position where two Right-Brain branches collide and the 4B model cannot settle the tension on its own (`unresolved_sleep_scenes`). | **Self-Generated Polarity Contrast (`sparring_trajectories.jsonl`)**: Every completed Phase 1 -> Phase 2 exchange pairs the Female hypnotic script directly against the Male lucid resistance reply. |
| **Who Provides the Deep Sleep Wisdom?** | **Petitions a Higher Power (`Qwen3.5-9B` Teacher)**: The 4B Student unloads from VRAM and calls the larger 9B Teacher (`rem_sleep_cycle.py`) to judge its collisions and invent contrastive dream lessons. | **Sleeps on Itself (`Impish_Bloodmoon_12B`)**: Already a 12B model, Bloodmoon needs no external Teacher—its own Phase 1 vs. Phase 2 trajectory log *is* the contrastive training dataset. |

---

## 2. Mode A: `LMFT-Chess.exe` — Petitioning a Higher Power (`Qwen3-4B` -> `Qwen3.5-9B`)

### Why the 4B Chess Model Petitions a Higher Power
In [`chess_app.py`](../src/lmft_architecture/chess_app.py) and [`chess_kibitz_bridge.py`](../src/lmft_architecture/chess_kibitz_bridge.py), the Left Brain is a fast, lightweight **4B parameter model (`Qwen/Qwen3-4B`)** confronting an external, objective chessboard (`python-chess` move stack, material balance, King safety, center control, and piece mobility).

While `Qwen3-4B` is fast enough to kibitz live on every move (`Pretext -> Subtext -> Context`), it occasionally hits a board geometry or parables scene where two branches of its 40-node Right Brain pull in opposite directions (`settled = False`, `GEAR_3_UNRESOLVED_QUEUED_FOR_SLEEP`). Because a 4B model only sees its own local deadlock, it cannot reliably invent the higher-order rule needed to untangle that collision by itself.

### Two Layers of Chess Sleep
1. **Fast In-App End-of-Game Sleep ([`ChessKibitzerRuntime.trigger_rem_sleep_consolidation`](../src/lmft_architecture/chess_app.py))**:
   - When you click **End-of-Game REM Sleep** inside `LMFT-Chess.exe`, the app inspects the game's `unresolved_sleep_scenes`, extracts the dominant 40-node sensor leaves (`>= 0.25`) active during those board collisions, and ratifies new `FuzzyConceptRule` items into `LMFT-Training/outputs/chess_kibitzer_state/consolidated_rules.json`.
2. **Deep Offline REM Sleep ([`rem_sleep_cycle.py`](../src/lmft_architecture/rem_sleep_cycle.py) & [`sleep_consolidation.py`](../src/lmft_architecture/sleep_consolidation.py))**:
   - When maturing the neural weights from `Cycle 1` to `Cycle 2`:
     1. `Qwen3-4B` unloads completely from your 16 GB GPU VRAM.
     2. The larger **`Qwen3.5-9B` Teacher** loads into VRAM, inspects the 4B student's unresolved collisions, refines colliding Tier-1 rules, and synthesizes brand-new contrastive `<|subliminal_gap|>` dream pairs (`continuation_a` vs. `continuation_b`).
     3. `Qwen3.5-9B` unloads, and `Qwen3-4B` wakes back up to run a gentle incremental LoRA + `SubliminalGapProjector` update (`lr_projector=1.8e-4`, `lr_lora=6e-5`) on the Teacher's dream pairs mixed with replay memory.

---

## 3. Mode B: `LMFT-Bloodmoon.exe` — Using Itself to Sleep on Itself (`Impish_Bloodmoon_12B`)

### Why the 12B Bloodmoon Model Sleeps on Itself
In [`bloodmoon_app.py`](../src/lmft_architecture/bloodmoon_app.py) and [`entrainment_sparring.py`](../src/lmft_architecture/entrainment_sparring.py), the Left Brain is **`Impish_Bloodmoon_12B`** (a 12B parameter creative model—already larger than the 9B Teacher). There is no external "higher power" model in the workspace with stronger hypnotic or counter-hypnotic prose fluency than Bloodmoon itself.

Instead of petitioning an outside model, `LMFT-Bloodmoon.exe` creates its own evolutionary learning signal by **splitting itself into two strictly ordered roles across time** (`feminine_to_masculine` one-way polarity):

1. **Phase 1 — Female Script-Writer (`The Producer`)**:
   - Steered at `<|subliminal_gap|>` by the **Weaver Chord** (`entrainment = 0.92`, `continuity = 0.90`, `earned_return = 0.88`, `agency = 0.85`).
   - Writes a fresh hypnotic script (`current_script`) onto the canvas and measures its real, un-clamped `induction_potency`. The Female modality only produces content and is never subjected to hypnotic scripts.
2. **Phase 2 — Male Content-Lover (`The Audience & Self-Immunizer`)**:
   - Reads the exact script it just wrote in Phase 1 as Pass-1 `Fact` (`SparringResistorFactSensors`, zero coordinate clamping) while steered at `<|subliminal_gap|>` by the **Resistor Chord** (`needlessness = 0.94`, `self_sufficiency = 0.92`, `discreteness = 0.88`, `existence = 0.86`).
   - Generates a grounded response (`resistance_reply`), scores `lucid_resistance` (`RESISTED_LUCIDLY` vs. `SWEPT_BY_ENTRAINMENT`), and appends the paired exchange to `sparring_trajectories.jsonl` via [`append_sparring_trajectory_jsonl`](../src/lmft_architecture/entrainment_sparring.py).
3. **Phase 3 — REM Sleep Consolidation (`Best Sleep of Its Life`)**:
   - **Fast In-App Clockwise Sleep ([`BloodmoonClockRuntime.trigger_rem_sleep_consolidation`](../src/lmft_architecture/bloodmoon_app.py))**:
     - Reads the pending self-sparring trajectories from Phase 2, extracts the active 40-node coordinates (`>= 0.25`) where hypnotic pull met lucid resistance, and ratifies new immunity rules (`sparring_immunity_c...`) into `LMFT-Training/outputs/impish_bloodmoon_state/consolidated_rules.json` before clearing the canvas and rotating back to **Phase 1**.
   - **Deep Weight Sleep ([`train_impish_bloodmoon_callosum_and_lora`](../src/lmft_architecture/impish_bloodmoon_adapter.py))**:
     - Because every line of `sparring_trajectories.jsonl` already contains an identical Clause-1 prefix paired with **Chord A + Female Induction (`continuation_a`)** and **Chord B + Male Lucid Resistance (`continuation_b`)**, Bloodmoon never needs an external Teacher to dream up contrastive pairs. It trains its own `SubliminalGapProjector` (`52 -> 4 x 5120`) and 4-bit NF4 LoRA adapters directly on the waking rounds it fought against itself.
