# Hop to it! Hopfield Free‑Energy Pathfinder

**Hopfield Free‑Energy Pathfinder** is an interactive desktop playground (PyQt 5) where a single agent explores a procedurally generated grid world by minimising *Expected Free Energy (EFE)*. Watch the agent learn the environment, form plans, and refine a Hopfield associative memory — all visualised in real time.

![App example](app-example.jpg)

---

## Quick Start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py
```

| Library    | Tested ver. |
| ---------- | ----------- |
| PyQt5      |  5.15       |
| torch      |  2.2        |
| numpy      |  1.26       |
| matplotlib |  3.9        |

---

## How It Works

1. **GridWorld** assigns an energy cost (1‑5 u) to every cell. Moving between shades that divide evenly grants a **one‑time full‑energy restore**.

2. The **Agent** keeps two Hopfield memories: a per‑cell *cost* predictor and a per‑cycle *reward* predictor, plus tabular stats.

3. A greedy **Planner** expands candidate paths, scoring neighbours with

   `score = (cost + surprise) − α·reward − 0.01·distance_to_goal`.
   Alternative neighbours become one‑step “back‑track probes” to seed future re‑use.

4. The **Executor** walks the chosen plan step‑by‑step. Actual cost, surprise, and energy are logged; the GUI repaints in real time.

5. At cycle end the agent computes a reward proportional to **final distance scaled by efficiency** vs similar past cycles, writes a `CycleSummary`, and resets.

All data are persisted so learning continues across sessions.

---

## User Interface

| Control                               | Action                                                |
| ------------------------------------- | ----------------------------------------------------- |
| **Cycles:** spin box + **Run Cycles** | Run *n* full exploration cycles with live animation   |
| **Rebuild World**                     | Delete the saved grid and generate a new random world |
| **Reset Memory**                      | Wipe all experience, Hopfield memories, and analytics |

Right‑hand panels show a sortable history table and three time‑series charts:

* **Expected Cost per Step** (planner estimate)
* **Surprise per Step** (`|expected − actual cost|`)
* **Energy Left per Step**

---

## Configuration

You can override defaults with environment variables or CLI args (see `main.py`):

| Name         | Default  | Meaning                                 |
| ------------ | -------- | --------------------------------------- |
| `GRID_SIZE`  |  150     | Width = height of the square world      |
| `WORLD_SEED` | *random* | Seed for deterministic world generation |
| `MAX_ENERGY` |  100     | Starting energy per cycle               |

---

## Next Up

- Multi-cycle plan quantization with Q scaling.

## License

MIT © jwest33
