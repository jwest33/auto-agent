import sys
import os
import json
import random
from typing import Optional, List

from PyQt5 import QtWidgets, QtGui, QtCore
from world import GridWorld, Coord, WORLD_PATH
from agent import Agent, Memory, Plan, MEMORY_PATH, euclidean

# --- Constants ---
DEFAULT_SIZE = 150
DEFAULT_SEED = None

CYCLE_HISTORY_PATH = "save/cycle_history.json"

APP_STYLES = """
QWidget {
    font-family: Arial, sans-serif;
    background-color: #f9f9f9;
}
QPushButton {
    padding: 6px 12px;
    border-radius: 4px;
    background-color: #2E8B57;
    color: white;
}
QPushButton:hover {
    background-color: #3CB371;
}
QGroupBox {
    border: 1px solid #ccc;
    border-radius: 4px;
    margin-top: 10px;
    background-color: #fff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center;
    padding: 0 3px;
    font-weight: bold;
}
QTableWidget {
    background-color: #fff;
    border: 1px solid #ccc;
}
QHeaderView::section {
    background-color: #f0f0f0;
    padding: 4px;
    border: 1px solid #ddd;
}
"""

class AgentWindow(QtWidgets.QMainWindow):
    def __init__(self, grid_size: int, seed: Optional[int]):
        super().__init__()
        self.setWindowTitle("FEP Agent Explorer")
        self.resize(1200, 800)
        self.central = QtWidgets.QWidget()
        self.setCentralWidget(self.central)
        self.central.setStyleSheet(APP_STYLES)

        # Initialize world and agent
        self.world = GridWorld(grid_size, grid_size, random.Random(seed))
        self.agent = Agent((0, 0))
        self.cycles_run = 0
        self.current_plan: Optional[Plan] = None
        self.cycle_history = []  # List of dicts with stats
        self.load_cycle_history()
        # Layouts
        main_layout = QtWidgets.QHBoxLayout(self.central)

        # --- Left: Grid View ---
        self.canvas = QtWidgets.QLabel()
        self.canvas.setFixedSize(grid_size * 5, grid_size * 5)
        self.canvas.setFrameStyle(QtWidgets.QFrame.Box)
        main_layout.addWidget(self.canvas, stretch=3)

        # --- Right: Controls & Analysis ---
        right_panel = QtWidgets.QVBoxLayout()
        main_layout.addLayout(right_panel, stretch=2)

        # Control Buttons and cycle input
        ctrl_group = QtWidgets.QGroupBox("Controls")
        ctrl_layout = QtWidgets.QHBoxLayout()
        ctrl_group.setLayout(ctrl_layout)
        self.spin_cycles = QtWidgets.QSpinBox()
        self.spin_cycles.setMinimum(1)
        self.spin_cycles.setValue(1)
        self.spin_cycles.setPrefix("Cycles: ")
        self.run_btn = QtWidgets.QPushButton("Run Cycles")
        self.run_btn.clicked.connect(self.on_run_cycles)
        self.reset_world_btn = QtWidgets.QPushButton("Rebuild World")
        self.reset_world_btn.clicked.connect(self.on_rebuild_world)
        self.reset_mem_btn = QtWidgets.QPushButton("Reset Memory")
        self.reset_mem_btn.clicked.connect(self.on_reset_memory)
        ctrl_layout.addWidget(self.spin_cycles)
        ctrl_layout.addWidget(self.run_btn)
        ctrl_layout.addWidget(self.reset_world_btn)
        ctrl_layout.addWidget(self.reset_mem_btn)
        right_panel.addWidget(ctrl_group)

        # Analysis Group
        self.analysis_group = QtWidgets.QGroupBox("Agent Analysis")
        analysis_layout = QtWidgets.QFormLayout()
        self.lbl_cycles = QtWidgets.QLabel("0")
        self.lbl_exp_reward = QtWidgets.QLabel("N/A")
        self.lbl_exp_cost = QtWidgets.QLabel("N/A")
        self.lbl_exp_surprise = QtWidgets.QLabel("N/A")
        self.lbl_energy = QtWidgets.QLabel("N/A")
        self.lbl_reward_trend = QtWidgets.QLabel("N/A")
        self.lbl_cost_trend = QtWidgets.QLabel("N/A")
        self.lbl_surprise_trend = QtWidgets.QLabel("N/A")
        analysis_layout.addRow("Cycles Run:", self.lbl_cycles)
        analysis_layout.addRow("Exp. Reward:", self.lbl_exp_reward)
        analysis_layout.addRow("Reward Trend:", self.lbl_reward_trend)
        analysis_layout.addRow("Exp. Cost:", self.lbl_exp_cost)
        analysis_layout.addRow("Cost Trend:", self.lbl_cost_trend)
        analysis_layout.addRow("Exp. Surprise:", self.lbl_exp_surprise)
        analysis_layout.addRow("Surprise Trend:", self.lbl_surprise_trend)
        analysis_layout.addRow("Energy Left:", self.lbl_energy)
        self.analysis_group.setLayout(analysis_layout)
        right_panel.addWidget(self.analysis_group)

        # Cycle History Table
        history_group = QtWidgets.QGroupBox("Cycle History")
        history_layout = QtWidgets.QVBoxLayout()
        history_group.setLayout(history_layout)
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Cycle", "Reward", "Cost", "Surprise", "Energy"])
        self.table.horizontalHeader().setStretchLastSection(True)
        history_layout.addWidget(self.table)
        right_panel.addWidget(history_group)
        right_panel.addStretch()
        
        self.trail: List[Coord] = []

        # Timer for refreshing view
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_canvas)
        self.timer.start(200)

        self.update_canvas()
            
    def load_cycle_history(self):
        if os.path.exists(CYCLE_HISTORY_PATH):
            with open(CYCLE_HISTORY_PATH, "r") as f:
                self.cycle_history = json.load(f)
                for stats in self.cycle_history:
                    self.add_history_row(stats)
                if self.cycle_history:
                    self.cycles_run = self.cycle_history[-1]['cycle']

    def save_cycle_history(self):
        with open(CYCLE_HISTORY_PATH, "w") as f:
            json.dump(self.cycle_history, f, indent=2)

    def on_run_cycles(self):
        self.trail.clear()
        self.pending_cycles = self.spin_cycles.value()
        self.run_next_cycle()

    def run_next_cycle(self):
        if self.pending_cycles <= 0:
            self.update_summary()
            self.save_cycle_history()
            return

        self.world.reset_cycle()
        old_explored = self.world.explored.copy()

        self.current_plan = self.agent.choose_plan(self.world)
        self.agent.execute_plan(
            self.current_plan,
            self.world,
            repaint_cb=self.update_canvas
        )
        self.cycles_run += 1

        # compute actual reward
        new_cells = int((~old_explored & self.world.explored).sum())
        dist = euclidean(self.agent.origin, self.agent.position)
        actual_reward = dist + new_cells

        stats = {
            'cycle': self.cycles_run,
            'reward': actual_reward,
            'cost': self.current_plan.expected_energy_cost,
            'surprise': self.current_plan.expected_surprise,
            'energy': self.agent.energy
        }
        self.cycle_history.append(stats)
        self.add_history_row(stats)

        # Defer next cycle to avoid blocking
        self.pending_cycles -= 1
        QtCore.QTimer.singleShot(100, self.run_next_cycle)

    def add_history_row(self, stats):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(stats['cycle'])))
        self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{stats['reward']:.2f}"))
        self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{stats['cost']:.2f}"))
        self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{stats['surprise']:.2f}"))
        self.table.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{stats['energy']:.2f}"))

    def update_summary(self):
        latest = self.cycle_history[-1]
        self.lbl_cycles.setText(str(latest['cycle']))
        self.lbl_exp_reward.setText(f"{latest['reward']:.2f}")
        self.lbl_exp_cost.setText(f"{latest['cost']:.2f}")
        self.lbl_exp_surprise.setText(f"{latest['surprise']:.2f}")
        self.lbl_energy.setText(f"{latest['energy']:.2f}")
        if len(self.cycle_history) > 1:
            prev = self.cycle_history[-2]
            self.lbl_reward_trend.setText(self.format_trend(latest['reward'] - prev['reward']))
            self.lbl_cost_trend.setText(self.format_trend(prev['cost'] - latest['cost']))
            self.lbl_surprise_trend.setText(self.format_trend(prev['surprise'] - latest['surprise']))
        else:
            self.lbl_reward_trend.setText("N/A")
            self.lbl_cost_trend.setText("N/A")
            self.lbl_surprise_trend.setText("N/A")

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
        self.reset_cycles_labels()

    def reset_cycles_labels(self):
        self.lbl_cycles.setText("0")
        self.lbl_exp_reward.setText("N/A")
        self.lbl_exp_cost.setText("N/A")
        self.lbl_exp_surprise.setText("N/A")
        self.lbl_reward_trend.setText("N/A")
        self.lbl_cost_trend.setText("N/A")
        self.lbl_surprise_trend.setText("N/A")
        self.lbl_energy.setText("N/A")

    def update_canvas(self):
        size = self.world.width * 5
        pixmap = QtGui.QPixmap(size, size)
        pixmap.fill(QtGui.QColor('#f0f0f0'))
        painter = QtGui.QPainter(pixmap)
        cell = 5

        # ← NEW: record current position
        self.trail.append(self.agent.position)

        # draw grid background and explored/restored cells
        for y in range(self.world.height):
            for x in range(self.world.width):
                val = int(self.world.grid[y, x])
                brightness = int(255 - (val / 9.0) * 195)
                if self.world.explored[y, x]:
                    color = QtGui.QColor(brightness, brightness, brightness, 180)
                else:
                    color = QtGui.QColor(brightness, brightness, brightness)
                painter.fillRect(x*cell, y*cell, cell, cell, color)

                if (x, y) in self.world.get_restored_cells():
                    pen = QtGui.QPen(QtGui.QColor(30, 144, 255))  # DodgerBlue
                    pen.setWidth(1)
                    painter.setPen(pen)
                    painter.drawRect(x*cell, y*cell, cell, cell)

        # ← NEW: draw the trail as a semi-transparent red line
        if len(self.trail) > 1:
            pen = QtGui.QPen(QtGui.QColor(255, 0, 0, 120))
            pen.setWidth(2)
            painter.setPen(pen)
            for (x1, y1), (x2, y2) in zip(self.trail, self.trail[1:]):
                # draw from center of one cell to the next
                painter.drawLine(
                    int((x1 + 0.5) * cell), int((y1 + 0.5) * cell),
                    int((x2 + 0.5) * cell), int((y2 + 0.5) * cell)
                )

        # draw agent on top
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(220, 20, 60))
        ax, ay = self.agent.position
        painter.drawEllipse(ax*cell, ay*cell, cell, cell)

        painter.end()
        self.canvas.setPixmap(pixmap)

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = AgentWindow(grid_size=DEFAULT_SIZE, seed=DEFAULT_SEED)
    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
