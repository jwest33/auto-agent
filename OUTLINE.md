# Cognitive Architecture Outline

Simulates an **energy‑constrained explorer** that learns, navigates, and visualizes its behavior in a grid world.  
The agent uses **step memory with directional vectors, surrounding cell information, and a Hopfield associative memory** to optimize exploration and goal-seeking behavior.

---

## 1  Memory System

| Layer | Purpose | Medium |
| --- | --- | --- |
| **Step memory** | Raw `StepExperience` events (position ➜ cost, surprise, energy before/after) | In‑RAM list, auto‑serialized to `save/memory.npy` |
| **Cell stats** | Running averages of cost, surprise, reward for each `(x,y)` | `Memory._cells : Dict[Coord, CellStats]` |
| **Per‑cell Hopfield** | Predicts **energy cost** of arbitrary cells given context features | `cell_hopfield` (key dim = 13, cap = 4096) |
| **Cycle summaries** | Aggregate reward / cost / surprise per exploration cycle | `save/cycles.npy` |

### 1.1 Key Encodings  
*Cell cost keys* encode a 13-dimensional feature vector including:
- Direction vectors (origin→cell, goal→cell, self→cell)
- Current cell shade (normalized)
- Surrounding cell shades (normalized)
- Current energy level (normalized)
- Distance to goal (normalized)

---

## 2  Decision Making

1. **Memory-based decision making** that evaluates potential moves based on:
   - Cost estimates from Hopfield memory
   - Expected surprise
   - Goal-seeking heuristic
   
2. **Move evaluation formula**:  
   ```
   score = (cost + surprise) - ALPHA * (1.0 / (goal_dist + 1.0))
   ```
   (ALPHA = 1.0 by default)

3. **Randomized exploration** with a 20% chance of selecting from the top 3 moves (with decreasing probability: 60%, 30%, 10%) instead of always selecting the best move.

---

## 3  Execution

* Agent chooses the next best move based on memory and goal-seeking behavior.
* Calls `world.transition(prev, curr)` to obtain `cost` and discover **one‑time full‑energy restores** when moving between shades where `curr % prev == 0`.
* Logs a `StepExperience`, updates per-cell stats, and writes the true cost back into the per-cell Hopfield.
* Repaints the GUI after each step.
* Limited to a maximum of 100 steps per cycle for safety.

Energy is reset and the agent teleports home at cycle end.

---

## 4  Learning & Reward

At cycle completion the agent:

1. Computes reward based on straight-line distance from origin to final position.
2. Stores a `CycleSummary`, serializes memories, and appends analytics rows for the GUI.
3. Resets energy, teleports home, and begins a new exploration cycle.

---

## 5  World Dynamics

* 0‑9 shades map linearly to energy costs.
* *Restore rule*: full‑energy recharge happens **once per unique divisible pair** of successive shade values.
* The world persists to `save/world.npy`; delete or press **Rebuild World** to regenerate.

---

## 6  User Interface & Analytics

* **PyQt 5** window with side panel: Run Cycles, Rebuild World, Reset Memory.
* Grid canvas renders elevation + coloured trail per cycle; agent is a red dot.
* Live Matplotlib charts (expected cost, surprise, energy, distance) update after each move.
* Cycle stats table auto‑saves to `save/cycle_history.json`.

---

```
[CYCLE START]
       ↓
[CHOOSE NEXT MOVE based on memory]
       ↓
[EXECUTE STEP ⇢ log exp. / update mem]
       ↓
[REPEAT UNTIL ENERGY DEPLETED OR MAX STEPS]
       ↓
[CYCLE SUMMARY → Save to disk]
       ↓
[RESET & REPEAT]
```
