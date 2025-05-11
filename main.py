import json
import os
import random
import sys
from typing import List, Optional

import colorsys
from PyQt5 import QtCore, QtGui, QtWidgets

from agent import Agent, Plan, Memory, euclidean, MEMORY_PATH, StepExperience
from world import GridWorld, WORLD_PATH, BASE_CELL_COST, Coord

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Constants
DEFAULT_SIZE = 150
DEFAULT_SEED = None
CYCLE_HISTORY_PATH = "save/cycle_history.json"
APP_STYLES = """
QWidget { font-family: Arial, sans-serif; background-color: #f9f9f9; }
QPushButton { padding: 6px 12px; border-radius: 4px; background-color: #2E8B57; color: white; }
QPushButton:hover { background-color: #3CB371; }
QGroupBox { border: 1px solid #ccc; border-radius: 4px; margin-top: 10px; background-color: #fff; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px; font-weight: bold; }
QTableWidget { background-color: #fff; border: 1px solid #ccc; }
QHeaderView::section { background-color: #f0f0f0; padding: 4px; border: 1px solid #ddd; }
"""

class AgentWindow(QtWidgets.QMainWindow):
    def __init__(self, grid_size: int, seed: Optional[int]):
        super().__init__()
        self.setWindowTitle("Hopfield Agent Pathfinder")
        self.resize(1200, 800)
        self.central = QtWidgets.QWidget(); self.setCentralWidget(self.central)
        self.central.setStyleSheet(APP_STYLES)
        # layout panes
        main_layout = QtWidgets.QHBoxLayout(self.central)
        # grid canvas
        self.canvas = QtWidgets.QLabel(); self.canvas.setFixedSize(grid_size * 5, grid_size * 5); self.canvas.setFrameStyle(QtWidgets.QFrame.Box)
        main_layout.addWidget(self.canvas, stretch=3)
        # right panel
        right_panel = QtWidgets.QVBoxLayout(); main_layout.addLayout(right_panel, stretch=2)
        # controls group
        ctrl_group = QtWidgets.QGroupBox("Controls"); ctrl_layout = QtWidgets.QHBoxLayout(); ctrl_group.setLayout(ctrl_layout)
        self.spin_cycles = QtWidgets.QSpinBox(); self.spin_cycles.setMinimum(1); self.spin_cycles.setValue(1); self.spin_cycles.setPrefix("Cycles: ")
        self.run_btn = QtWidgets.QPushButton("Run Cycles"); self.run_btn.clicked.connect(self.on_run_cycles)
        self.reset_world_btn = QtWidgets.QPushButton("Rebuild World"); self.reset_world_btn.clicked.connect(self.on_rebuild_world)
        self.reset_mem_btn = QtWidgets.QPushButton("Reset Memory"); self.reset_mem_btn.clicked.connect(self.on_reset_memory)
        for w in (self.spin_cycles, self.run_btn, self.reset_world_btn, self.reset_mem_btn):
            ctrl_layout.addWidget(w)
        right_panel.addWidget(ctrl_group)
        # history table
        history_group = QtWidgets.QGroupBox("Cycle History"); hist_layout = QtWidgets.QVBoxLayout(); history_group.setLayout(hist_layout)
        self.table = QtWidgets.QTableWidget(0, 5); self.table.setHorizontalHeaderLabels(["#", "Reward", "Cost", "Surprise", "Energy"])
        hist_layout.addWidget(self.table); right_panel.addWidget(history_group)
        # charts
        charts = QtWidgets.QGroupBox("Analytics Charts"); ch_layout = QtWidgets.QVBoxLayout(); charts.setLayout(ch_layout)
        self.figure = plt.Figure(); self.canvas_chart = FigureCanvas(self.figure)
        self.ax_cost, self.ax_surprise, self.ax_energy = [self.figure.add_subplot(3, 1, i + 1) for i in range(3)]
        ch_layout.addWidget(self.canvas_chart); right_panel.addWidget(charts); right_panel.addStretch()
        # initialize world/agent
        self.world = GridWorld(grid_size, grid_size, random.Random(seed))
        self.agent = Agent((0, 0))
        self.cycles_run = 0
        self.current_plan: Optional[Plan] = None
        self.cycle_history: List[dict] = []
        self.load_cycle_history()
        # viz helpers
        self.trails: List[List[Coord]] = []
        self.step_history: List[list] = []
        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self.update_canvas); self.timer.start(200)
        self.update_canvas(); self.update_charts()

    def load_cycle_history(self):
        if os.path.exists(CYCLE_HISTORY_PATH):
            with open(CYCLE_HISTORY_PATH, "r") as f:
                self.cycle_history = json.load(f)
                for s in self.cycle_history:
                    self.add_history_row(s)
                if self.cycle_history:
                    self.cycles_run = self.cycle_history[-1]["cycle"]

    def save_cycle_history(self):
        os.makedirs(os.path.dirname(CYCLE_HISTORY_PATH), exist_ok=True)
        with open(CYCLE_HISTORY_PATH, "w") as f:
            json.dump(self.cycle_history, f, indent=2)

    def add_history_row(self, stats):
        r = self.table.rowCount(); self.table.insertRow(r)
        for i, k in enumerate(["cycle", "reward", "cost", "surprise", "energy"]):
            self.table.setItem(r, i, QtWidgets.QTableWidgetItem(f"{stats[k]:.2f}" if k != "cycle" else str(stats[k])))

    def on_run_cycles(self):
        self.trails.clear(); self.step_history.clear(); self.pending = self.spin_cycles.value(); self.run_next_cycle()

    def run_next_cycle(self):
        # 1. Finished?
        if self.pending <= 0:
            self.save_cycle_history()
            return

        # 2. Prepare UI / agent for a fresh cycle
        self.trails.append([])            # store coloured segments
        self.step_history.append([])      # per‑step stats for charts
        self.world.reset_cycle()
        self.agent.start_cycle()
        old_explored = self.world.explored.copy()

        # 3. Choose a plan
        self.current_plan = self.agent.choose_plan(self.world)

        # 4. Wrap Memory.add_experience so we can tap each StepExperience
        orig_add = self.agent.memory.add_experience

        def wrapped(exp: StepExperience):
            # Forward to the real recorder
            orig_add(exp)

            # Log data the charts need
            self.step_history[-1].append(
                {
                    "expected_cost": exp.expected_cost,
                    "actual_cost":   exp.actual_cost,
                    "surprise":      exp.surprise,
                    "energy":        self.agent.energy,
                }
            )

        self.agent.memory.add_experience = wrapped

        # 5. Execute the plan (updates UI via repaint_cb)
        self.agent.execute_plan(
            self.current_plan,
            self.world,
            repaint_cb=self.repaint_and_update_charts,
        )

        # 6. Restore original method
        self.agent.memory.add_experience = orig_add

        # 7. Summarise the cycle
        self.cycles_run += 1
        new_cells = int((~old_explored & self.world.explored).sum())
        dist       = euclidean(self.agent.origin, self.agent.position)
        reward     = dist + new_cells

        stats = {
            "cycle":    self.cycles_run,
            "reward":   reward,
            "cost":     self.current_plan.expected_energy_cost,
            "surprise": self.current_plan.expected_surprise,
            "energy":   self.agent.energy,
        }
        self.cycle_history.append(stats)
        self.add_history_row(stats)
        self.update_charts()

        # 8. Queue the next cycle
        self.pending -= 1
        QtCore.QTimer.singleShot(100, self.run_next_cycle)
        
    @staticmethod
    def format_trend(delta: float) -> str:
        if delta > 0:
            return f"▲ +{delta:.2f}"
        elif delta < 0:
            return f"▼ {delta:.2f}"
        else:
            return "→ 0.00"

    def on_rebuild_world(self):
        if os.path.exists(WORLD_PATH): os.remove(WORLD_PATH)
        if os.path.exists(CYCLE_HISTORY_PATH): os.remove(CYCLE_HISTORY_PATH)
        self.world = GridWorld(self.world.width, self.world.height)
        self.reset_all()
        self.update_canvas()

    def on_reset_memory(self):
        if os.path.exists(MEMORY_PATH): os.remove(MEMORY_PATH)
        if os.path.exists(CYCLE_HISTORY_PATH): os.remove(CYCLE_HISTORY_PATH)
        self.agent.memory = Memory()
        self.reset_all()
        self.update_canvas()

    def reset_all(self):
        self.cycles_run = 0
        self.current_plan = None
        self.cycle_history.clear()
        self.table.setRowCount(0)

    def repaint_and_update_charts(self):
        self.update_canvas(); self.update_charts()

    def update_canvas(self):
        sz = self.world.width * 5; pm = QtGui.QPixmap(sz, sz); pm.fill(QtGui.QColor("#f0f0f0"))
        p = QtGui.QPainter(pm); csize = 5
        if self.trails: self.trails[-1].append(self.agent.position)
        for y in range(self.world.height):
            for x in range(self.world.width):
                val = int(self.world.grid[y, x]); bright = int(255 - (val / 9.0) * 195)
                col = QtGui.QColor(bright, bright, bright, 180 if self.world.explored[y, x] else 255)
                p.fillRect(x * csize, y * csize, csize, csize, col)
        n = len(self.trails)
        for i, trail in enumerate(self.trails):
            if len(trail) < 2: continue
            hue = (i * 360 / n) / 360.0; rgb = colorsys.hsv_to_rgb(hue, 1, 0.78)
            pen = QtGui.QPen(QtGui.QColor(*[int(c * 255) for c in rgb], 120)); pen.setWidth(2); p.setPen(pen)
            for a, b in zip(trail, trail[1:]):
                p.drawLine(int((a[0] + 0.5) * csize), int((a[1] + 0.5) * csize), int((b[0] + 0.5) * csize), int((b[1] + 0.5) * csize))
        p.setPen(QtCore.Qt.NoPen); p.setBrush(QtGui.QColor(220, 20, 60))
        ax, ay = self.agent.position; p.drawEllipse(ax * csize, ay * csize, csize, csize); p.end(); self.canvas.setPixmap(pm)

    def update_charts(self):
        for ax in (self.ax_cost, self.ax_surprise, self.ax_energy): ax.clear()
        n = len(self.step_history)
        for idx, hist in enumerate(self.step_history):
            if not hist: continue
            hue = (idx * 360 / n) / 360.0; rgb = colorsys.hsv_to_rgb(hue, 1, 0.78)
            cols = [[d["expected_cost"] for d in hist], [d["surprise"] for d in hist], [d["energy"] for d in hist]]
            xs = list(range(len(hist)))
            for ax, dat in zip((self.ax_cost, self.ax_surprise, self.ax_energy), cols):
                ax.plot(xs, dat, color=rgb)
        self.ax_cost.set_title("Expected Cost per Step"); self.ax_surprise.set_title("Surprise per Step"); self.ax_energy.set_title("Energy Left per Step"); self.figure.tight_layout(); self.canvas_chart.draw()

def main():
    app = QtWidgets.QApplication(sys.argv); win = AgentWindow(grid_size=DEFAULT_SIZE, seed=DEFAULT_SEED); win.show(); sys.exit(app.exec_())


if __name__ == "__main__":
    main()
