# Free Energy Pathfinding Agent

**Free Energy Pathfinding Agent** is a desktop playground (built with PyQt 5) that demonstrates a simple agent driven by the *Free‑Energy Principle* (Free Energy). The app lets you watch the agent plan, act, learn, and update its internal model while exploring a grid world. You can step through any number of exploration cycles and view—*in real‑time*—per‑step charts that track the agent’s cost, surprise, and remaining energy. Colored trails and a legend visually tie each cycle’s path to its chart trace.


![App example](app-example.png)

---

## Conceptual Overview

### The World

* A `GridWorld` is generated with integer "shades" (0–9) representing terrain cost.
* Darker cells cost more energy to enter and are thus *riskier*.
* Every cycle the world “heals” previously visited cells, rewarding re‑exploration.

### The Agent

| Component            | Purpose                                                                                                         |
| -------------------- | --------------------------------------------------------------------------------------------------------------- |
| **Perception**       | Reads cell shade and energy feedback on every step.                                                             |
| **Internal Model**   | Encodes empirical priors on *expected* cost and surprise for each shade.                                        |
| **Memory**           | Logs a `CellExperience` (position ✓, shade, *expected* cost, *actual* cost, surprise, timestamp) on every move. |
| **Planner**          | Samples candidate paths, computes *expected free energy* for each, and selects the lowest‑EFE plan.             |
| **Policy Execution** | Follows the chosen plan until energy is exhausted or home is reached.                                           |

### Free‑Energy Principle (Free Energy)

The agent tries to **minimise its (expected) surprise** under an implicit generative model of the world.  In practice, the planner searches paths that trade off:

```
EFE  =  expected_cost   +   expected_surprise
      (energy budget)     (model prediction error)
```

* **Expected Cost** ≈ energy the agent predicts it will spend on the path.
* **Expected Surprise** ≈ discrepancy the agent *expects* between model‑predicted shade costs and true shade costs along the path.

The chosen path therefore advances the frontier while keeping the agent in familiar, affordable terrain.

---

### Controls

| UI Element         | Action                                           |
| ------------------ | ------------------------------------------------ |
| **Cycles** spinner | Number of full explore‑return cycles to run.     |
| **Run Cycles**     | Execute `n` cycles (chart updates in real time). |
| **Rebuild World**  | Generate a brand‑new grid world (clears charts). |
| **Reset Memory**   | Wipe the agent’s memory (keeps current world).   |

---

## Visual Analytics

| Chart             | What it shows                                     |                   |    |
| ----------------- | ------------------------------------------------- | ----------------- | -- |
| **Expected Cost** | Per‑step energy predicted before taking the step. |                   |    |
| **Surprise**      | Per‑step prediction error (                       | expected − actual | ). |
| **Energy Left**   | Agent’s remaining energy as the plan executes.    |                   |    |

* Each cycle is drawn with a unique hue in **both the grid trail and the charts**, making it easy to correlate behaviour and metrics.
* Lines grow point‑by‑point thanks to the wrapped `Memory.add_experience` callback, which appends metrics and triggers a redraw on every step.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
