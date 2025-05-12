# Cognitive Architecture Outline

Simulates an **energy‑constrained explorer** that plans, acts, learns, and visualises its behaviour in a grid world.  
The agent threads together **per‑cell statistics, two Hopfield associative memories, and an EFE‑style planner** to maximise long‑term reward.

---

## 1  Memory System

| Layer | Purpose | Medium |
| --- | --- | --- |
| **Step memory** | Raw `StepExperience` events (position ➜ cost, surprise) | In‑RAM list, auto‑serialized to `save/memory.npy` :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1} |
| **Cell stats** | Running averages of cost, surprise, reward for each `(x,y)` | `Memory._cells : Dict[Coord, CellStats]` :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3} |
| **Per‑cell Hopfield** | Predicts **energy cost** of arbitrary cells given local geometry (+ shade) | `cell_hopfield` (key dim = 10, cap = 4096) :contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5} |
| **Cycle summaries** | Aggregate reward / cost / surprise per exploration cycle | `save/cycles.npy` |
| **Cycle Hopfield** | Predicts **future reward** from partial trajectories (legacy) | `cycle_hopfield` (key dim = 10) :contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7} |

### 1.1 Key Encodings  
*Cell cost keys* encode three direction vectors (origin→cell, goal→cell, self→cell) + shade (10‑D) before being written to the per‑cell Hopfield. :contentReference[oaicite:8]{index=8}:contentReference[oaicite:9]{index=9}

---

## 2  Planning

1. **Retrieve** cached plans that still start at the current position and fit the energy budget.  
2. **Generate** a new plan if none are suitable. The greedy expansion chooses the neighbour that minimises  

```

score = (cost + surprise) − α·reward − 0.01·dist\_to\_goal

```

(α = 1.0 by default). :contentReference[oaicite:10]{index=10}:contentReference[oaicite:11]{index=11}  
3. **Back‑track probes**: alternative neighbours become one‑step “probe” plans (flagged `backtrack=True`) to seed future retrieval. :contentReference[oaicite:12]{index=12}:contentReference[oaicite:13]{index=13}  
4. Each plan caches expected `(cost, surprise)` per step and derives  

```

expected\_reward = straight‑line\_distance(origin, last\_step) + unseen\_cells

```

Back‑tracks are slightly down‑weighted (× 0.8). :contentReference[oaicite:14]{index=14}:contentReference[oaicite:15]{index=15}  

---

## 3  Execution

* Consumes plan steps **once‑only** (steps are popped as the agent moves).  
* Calls `world.transition(prev, curr)` to obtain `cost` and discover **one‑time full‑energy restores** when moving between shades where `curr % prev == 0`. :contentReference[oaicite:16]{index=16}:contentReference[oaicite:17]{index=17}  
* Logs a `StepExperience`, updates per‑cell stats, and writes the true cost back into the per‑cell Hopfield.  
* Repaints the GUI every step.

Energy is reset and the agent teleports home at cycle end.

---

## 4  Learning & Reward

At cycle completion the agent:

1. Computes distance‑based reward **scaled** by efficiency relative to similar prior cycles. :contentReference[oaicite:18]{index=18}:contentReference[oaicite:19]{index=19}  
2. Stores a `CycleSummary`, serialises memories, and appends analytics rows for the GUI.  
3. Resets energy and begins a new planning loop.

---

## 5  World Dynamics

* 0‑9 shades map linearly to **1‑5 energy units** (`value_to_cost`). :contentReference[oaicite:20]{index=20}:contentReference[oaicite:21]{index=21}  
* *Restore rule*: full‑energy recharge happens **once per unique divisible pair** of successive shade values. :contentReference[oaicite:22]{index=22}:contentReference[oaicite:23]{index=23}  
* The world persists to `save/world.npy`; delete or press **Rebuild World** to regenerate.

---

## 6  User Interface & Analytics

* **PyQt 5** window with side panel: Run Cycles, Rebuild World, Reset Memory. :contentReference[oaicite:24]{index=24}:contentReference[oaicite:25]{index=25}  
* Grid canvas renders elevation + coloured trail per cycle; agent is a red dot.  
* Live Matplotlib charts (cost, surprise, energy) update every 200 ms. :contentReference[oaicite:26]{index=26}:contentReference[oaicite:27]{index=27}  
* Cycle stats table auto‑saves to `save/cycle_history.json`.

---

```

\[CYCLE START]
↓
\[PLAN RETRIEVAL / GENERATION]
↓
\[EXECUTE STEP ⇢ log exp. / update mem]
↓
\[CYCLE SUMMARY → Hopfield + disk]
↓
\[RESET & REPEAT]

```
