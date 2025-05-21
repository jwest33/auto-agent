# Cognitive Architecture Outline

Simulates an **energy‑constrained autonomous agent** navigating a grid world.  
The agent learns from experience, predicts energy costs using associative memory, and visualizes decision-making and performance in real time.

---

## 1. Memory System

| Layer              | Purpose                                                      | Storage Medium                           |
|-------------------|--------------------------------------------------------------|------------------------------------------|
| **Step memory**    | Records detailed step data (position, cost, energy, surprise) | RAM → `save/memory.npy` (NumPy array)    |
| **Cycle summaries**| Tracks performance stats per cycle (reward, cost, surprise)  | RAM → `save/cycles.npy` (NumPy array)    |
| **Cell stats**     | Maintains running averages of cost and surprise per cell     | In-memory: `Dict[Coord, CellStats]`      |
| **Hopfield memory**| Predicts energy cost of unseen cells based on 12-d features  | Torch buffer: `HopfieldMemory` (4096 cap)|

### 1.1 Memory Key Encoding

Each cell is encoded as a 12D vector with:

- Direction vectors from origin, goal, and agent to the cell (6D)
- Normalized energy (1D)
- Distance to goal (1D)
- Cell shade and 4 neighboring shades (4D)

---

## 2. Decision Making

The agent evaluates all neighboring cells using:

- **Predicted cost** from Hopfield memory
- **Expected surprise** (from tabular cell stats)
- **Goal-seeking heuristic** (inverse goal distance)
- **Oscillation penalty** (backtracking or repeat visits)
- **Curiosity bonus** (uncertainty in Hopfield prediction)

### Move Scoring Formula:
score = cost + surprise + penalty - ALPHA * (1.0 / (goal_dist + 1.0)) - CURIOSITY_WEIGHT * uncertainty

---

## 3. Execution Loop

1. Choose the best move based on the scoring formula
2. Call `world.transition(prev, curr)` to move and get:
   - Actual cost
   - Potential full-energy restore if shade divisibility rule applies
3. Update agent position, memory, and GUI
4. Repeat until energy depletes or goal reached
5. Log `StepExperience` and append `CycleSummary`
6. Teleport to origin and reset energy for the next cycle

---

## 4. Learning & Adaptation

- All step/cycle data are serialized for persistence
- A **HyperScheduler** optimizes four parameters via hill-climbing:
  - `ALPHA`, `BACKTRACK_PENALTY`, `RECENT_VISIT_PENALTY`, `CURIOSITY_WEIGHT`
- Evolution history visualized in the analytics UI

---

## 5. GridWorld Dynamics

- Each cell has a shade (0–9) → cost from 1.0 to 5.0
- One-time **energy restore** when stepping into a cell where `new % prev == 0`
- Grid is saved in `save/world.npy` and reused between runs

---

## 6. User Interface (PyQt5)

- **Left Panel**: Grid canvas + control buttons
- **Right Panel**:
  - Cycle controls
  - Cycle history table
  - Live charts (surprise, energy, distance)
  - Hyper-parameter evolution
- **Analytics Tab**:
  - Cycle & step tables
  - Metric heatmaps
  - Memory network graph
  - Detailed memory queries by position

---

[CYCLE START]
↓
[EVALUATE MOVES from current position]
↓
[CHOOSE NEXT CELL using memory + goal + curiosity]
↓
[EXECUTE MOVE → update memory, UI, energy]
↓
[REPEAT UNTIL ENERGY DEPLETED or GOAL]
↓
[SAVE CYCLE SUMMARY + MEMORY]
↓
[RESET → TELEPORT HOME → NEXT CYCLE]
