## Cognitive Architecture Outline

Simulates a **planning and learning system** inspired by memory-based cognition. It operates in cycles, using past experiences and predictions to guide exploration in a grid-based world. The core loop includes **planning, executing, learning, and updating** behaviors.

---

## 1. **Memory System (Learning from Experience)**

The agent maintains a `Memory` object, which includes:

### 🔹 **Step-Based Memory**

* **StepExperience** records:

  * `position`, `shade`, `expected_cost`, `actual_cost`, `surprise`, `timestamp`
* Stored as raw experiences in a list for temporal tracking and statistical learning.

### 🔹 **Cell-Based Memory**

* Tracks per-cell stats using a dictionary of `CellStats`:

  * Running averages for cost, surprise, and reward.
  * Updated using **online learning** (incremental updates per visit).

### 🔹 **Cycle Summaries**

* After each exploration cycle, a `CycleSummary` is saved:

  * Total `reward`, `cost`, `surprise`, `energy_left`, and path `steps`.
* Summaries are encoded and saved into a **HopfieldMemory** (vector-based key-value store) to guide future predictions.

---

## 2. **Planning (Imagining Future Paths)**

### 🔹 **Plan Generation**

* If no prior plans exist, the agent generates one using:

  * A **greedy neighbor selection** process that balances:

    * **Expected Free Energy (EFE)**: `cost + surprise - α * reward`
  * Iteratively extends a plan until energy budget is exceeded.

### 🔹 **Plan Retrieval**

* Retrieves viable plans (stored in memory) that:

  * Begin at the current position.
  * Cost less than or equal to current energy.
* Selected based on highest **expected reward**.

### 🔹 **Plan Metrics**

* Plans include cached expectations of:

  * **Energy cost**, **surprise**, and **reward** (distance + novelty).
* Metrics are recalculated using memory lookups and Euclidean distance.

---

## 3. **Execution (Acting in the World)**

### 🔹 **Stepwise Execution**

* Executes each plan step‑by‑step, **popping** the current step off the front of the plan so finished plans are never replayed:

  * Transitions via `world.transition()` to determine actual cost and detect "restores".
  * Computes **surprise** as `|expected_cost - actual_cost|`.
  * Reduces energy or restores it under special conditions.
  * Resets `_prev_pos` at the start of every cycle to keep cost accounting consistent.

### 🔹 **Experience Logging**

* Each step appends a `StepExperience` to memory.
* These are used to:

  * Update per-cell statistics.
  * Accumulate data for real-time visual analytics.

---

## 4. **Learning and Adaptation**

### 🔹 **Cycle Finalization**

* At cycle end:

  * Computes final **reward** as `distance from origin + step count`.
  * Stores a `CycleSummary` in both raw history and HopfieldMemory.
  * Rewards per cell on the path are backpropagated to memory.
  * Memory is serialized to file.

### 🔹 **Hopfield Memory Use**

* HopfieldMemory is a fixed-capacity, dot-product-based **associative memory**.
* Used to **predict reward of new partial plans** by querying it with encoded path features (length, efficiency, neighbor stats, etc.).

---

## 5. **Motivation and Strategy**

### 🔹 **Exploration vs. Exploitation**

* Balances known rewards and surprises to explore intelligently.
* Prefers novel but promising directions by penalizing EFE and incentivizing potential rewards.

### 🔹 **Energy-Constrained Navigation**

* All planning and movement are constrained by a finite energy budget.
* Encourages strategic, efficient movement.

---

## Summary Diagram

```
[CYCLE START]
     ↓
[PLAN RETRIEVAL or GENERATION]
     ↓
[EXECUTE PLAN STEP BY STEP]
     ↓
[LOG EXPERIENCE & UPDATE MEMORY]
     ↓
[CYCLE SUMMARY → HopfieldMemory]
     ↓
[CYCLE END → Reset]
     ↓
[REPEAT]
```
