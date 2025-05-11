## **Agent Process Outline (with Cell-Level Surprise Association) ([Free Energy-Inspired](https://en.wikipedia.org/wiki/Free_energy_principle))**

### **Steps**

#### **1\. Perceive Environment**

* Observe grid cells in the local vicinity (vision radius or full map).

* Observe agent’s current position, internal energy level, and cell energy cost.

* Encode this into a belief state: a structured representation of internal and external variables for planning.


### **2\. Infer Current Needs and Goals**

* Compare expected internal energy state with current energy.

* If energy is full and the agent is at home, set the goal to **“explore further from origin”**.

* If energy is low, goal is automatically **fulfilled** (exploration ends).

* Goal is defined dynamically: **maximize Euclidean distance from origin before depletion**.

#### **3\. Retrieve and Evaluate Candidate Policies**

* Retrieve prior plans based on full paths or **regionally overlapping sub-paths**.

* Score plans by:

  * **total\_expected\_surprise \= sum(cell.expected\_surprise for cell in plan.steps)**

  * **expected\_energy\_cost \= sum(cell.expected\_energy\_cost for cell in plan.steps)**

  * **expected\_reward \= distance\_from\_origin \- total\_expected\_surprise**

#### **4\. Plan Generation or Adaptation**

* Newly generated plans should estimate cell-wise surprise using memory.

* Each step in the path includes:

  * (x, y) coordinates

  * expected\_energy\_cost

  * expected\_surprise\_at\_cell (looked up from memory or estimated as 0 if novel)

\`\`\`

`plan.expected_energy_cost = sum(cell.expected_energy_cost for cell in plan.steps)`

`plan.expected_surprise = sum(cell.expected_surprise_at_cell for cell in plan.steps)`

`plan.expected_reward = plan.distance_from_origin - plan.expected_surprise`

\`\`\`

#### **5\. Execute Plan with Continuous Inference**

At each step:

* Deduct energy based on actual cell cost:

  `agent.energy -= actual_cost`  
* Track:

  * `cell.observed_cost = actual_cost`

  * `cell.surprise = abs(expected_cost - actual_cost)`

* Log each cell visit in an **experience buffer** for memory updates.

Interrupt execution if:

* Agent's energy depletes

#### **6\. Post-Execution Evaluation**

* Teleport home.

* Aggregate:

  * `actual_path_cost = sum(actual_costs)`

  * `total_surprise = sum(cell.surprise for cell in path)`

  * `reward = distance_from_origin - total_surprise`

#### **7\. Memory Update**

#### **For each visited cell:**

* Store:

  * `(x, y)`

  * `shade`

  * `expected_cost`

  * `actual_cost`

  * `surprise`

  #### **For the plan:**

* Store:

  * Path steps

  * Total surprise

  * Context (start state, energy level)

  #### **Update:**

* `Cell Surprise Map`: for future per-cell surprise estimates.

* `Cell Cost Map`: smoothed terrain expectations (average cost).

* `Memory DB`: for plan recall and regional learning.

#### **8\. Core Modules and Components**

### **`Agent` Class**

* `origin` (x, y)

* `energy`

* `max_energy`

* `position`

* `teleport_home()`

* `calculate_distance_reward()`

### **`GridWorld` Class**

Represents the environment.

**Attributes**:

* `grid`: 2D matrix of cells  
* Each `cell` includes:  
  * `shade` (0–255)  
  * `energy_cost`  
  * `explored` (bool)

**Methods**:

* `get_neighbors(x, y)`  
* `render(agent_position)`  
* `calculate_path_cost(path)`  
  `get_shade(x, y)`  
* `get_energy_cost(x, y)`  
* `set_observed_cost(x, y, cost)`

  ### **`Plan` Class**

Stores:

* `steps`: list of (x, y) coordinates

* `expected_energy_cost`

* `expected_surprise`

* `expected_reward` (distance from origin)


  ### **`Memory` Class**

  #### **Plan Memory:**

* `plans: List[Plan]`

  #### **Cell Memory:**

* `cell_records: Dict[(x, y), List[CellExperience]]`


  #### **`CellExperience` object:**

* `position: (x, y)`

* `shade: int`

* `expected_cost: float`

* `actual_cost: float`

* `surprise: float`

* `timestamp` or `visit_count` (optional for smoothing)

## **GUI Integration Plan (PyQt)**

Will visualize:

* Grid view (grayscale) with agent position

* Step-wise plan animation

* Energy bar

* Cell surprise overlay / heatmap

* Memory viewer (plans \+ cell stats)

* Summary of distance, reward, surprise

## **Data Structures**

* `World grid`: 2D NumPy array or list-of-lists

* `Agent state`: `position`, `energy`, `memory`

* `Memory DB`: TinyDB, JSON, or in-memory dict

* `Plans`: stored with context hash and cell annotations
