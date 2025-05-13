from __future__ import annotations
import math
import os
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
SAVE_DIR = BASE_DIR / "save"
MEMORY_PATH = SAVE_DIR / "memory.npy"
CYCLE_PATH = SAVE_DIR / "cycles.npy"
ORIGIN: Tuple[int, int] = (0, 0)

# Memory record dtype copied from agent.Memory._step_dtype
_STEP_DTYPE = np.dtype([
    ("x", "i4"), ("y", "i4"), ("shade", "i4"),
    ("expected_cost", "f4"), ("actual_cost", "f4"), ("surprise", "f4"),
    ("energy_before", "f4"), ("energy_after", "f4"),
    ("timestamp", "f8"),
])

# Cycle dtype copied from agent.Memory._cycle_dtype
_CYCLE_DTYPE = np.dtype([
    ("reward", "f4"), ("cost", "f4"), ("surprise", "f4"),
    ("energy_left", "f4"), ("timestamp", "f8"),
])

class MemoryData:
    """Loads numpy memories and offers convenient per-cycle slices."""

    def __init__(self):
        self.steps: np.ndarray | None = None
        self.cycles: np.ndarray | None = None
        self.cycle_indices: List[slice] = []  # maps cycle-idx -> slice into steps

        self._load_files()
        if self.steps is not None and self.cycles is not None:
            self._build_cycle_slices()

    def _load_files(self):
        if MEMORY_PATH.exists() and MEMORY_PATH.stat().st_size > 0:
            self.steps = np.load(MEMORY_PATH, allow_pickle=False).astype(_STEP_DTYPE, copy=False)
        if CYCLE_PATH.exists() and CYCLE_PATH.stat().st_size > 0:
            self.cycles = np.load(CYCLE_PATH, allow_pickle=False).astype(_CYCLE_DTYPE, copy=False)

    def _build_cycle_slices(self):
        """Computes which rows of *steps* belong to each cycle using timestamps."""
        timestamps = self.steps["timestamp"]
        cycle_ts = self.cycles["timestamp"]

        prev_end = float("-inf")
        for ts in cycle_ts:
            in_cycle = np.nonzero((timestamps > prev_end) & (timestamps <= ts))[0]
            if in_cycle.size:
                start, end = in_cycle[0], in_cycle[-1] + 1
                self.cycle_indices.append(slice(start, end))
            else:
                # Cycle with zero steps – rare but possible
                self.cycle_indices.append(slice(0, 0))
            prev_end = ts


    def steps_for_cycle(self, idx: int) -> np.ndarray:
        if not self.cycle_indices:
            return np.empty((0,), _STEP_DTYPE)
        return self.steps[self.cycle_indices[idx]]

class CycleTable(QtWidgets.QTableWidget):
    headers = ["#", "Reward", "Cost", "Surprise", "Energy left"]

    def __init__(self, parent=None):
        super().__init__(0, len(self.headers), parent)
        self.setHorizontalHeaderLabels(self.headers)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setStretchLastSection(True)


    def populate(self, cycles: np.ndarray):
        self.setRowCount(len(cycles))
        for row, cyc in enumerate(cycles):
            items = [
                str(row + 1),
                f"{cyc['reward']:.2f}",
                f"{cyc['cost']:.2f}",
                f"{cyc['surprise']:.2f}",
                f"{cyc['energy_left']:.1f}",
            ]
            for col, text in enumerate(items):
                itm = QtWidgets.QTableWidgetItem(text)
                self.setItem(row, col, itm)

class StepTable(QtWidgets.QTableWidget):
    headers = [
        "Idx", "x", "y", "shade", "exp_cost", "act_cost",
        "surprise", "E_before", "E_after"
    ]

    def __init__(self, parent=None):
        super().__init__(0, len(self.headers), parent)
        self.setHorizontalHeaderLabels(self.headers)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setStretchLastSection(True)

    def populate(self, steps: np.ndarray):
        self.setRowCount(len(steps))
        for i, rec in enumerate(steps):
            items = [
                str(i),
                str(rec["x"]), str(rec["y"]), str(rec["shade"]),
                f"{rec['expected_cost']:.2f}", f"{rec['actual_cost']:.2f}",
                f"{rec['surprise']:.2f}",
                f"{rec['energy_before']:.1f}", f"{rec['energy_after']:.1f}",
            ]
            for col, text in enumerate(items):
                itm = QtWidgets.QTableWidgetItem(text)
                self.setItem(i, col, itm)

class AnalyticsWindow(QtWidgets.QMainWindow):
    def __init__(self, data: MemoryData):
        super().__init__()
        self.setWindowTitle("Agent Memory Analytics Viewer")
        self.resize(1400, 800)
        self.data = data

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        hbox = QtWidgets.QHBoxLayout(central)

        self.cycle_table = CycleTable()
        hbox.addWidget(self.cycle_table, 1)

        right_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.step_table = StepTable()
        right_splitter.addWidget(self.step_table)

        # Determine memory dimensions
        if self.data.steps is not None:
            self.step_dims = list(self.data.steps.dtype.names)
        else:
            self.step_dims = [
                "x", "y", "shade", "expected_cost", "actual_cost",
                "surprise", "energy_before", "energy_after", "timestamp"
            ]

        # Create dynamic plots for every memory dimension
        fig = matplotlib.figure.Figure(figsize=(5, max(4, len(self.step_dims) * 1.5)))
        self.canvas = FigureCanvas(fig)
        self.axes: List[matplotlib.axes.Axes] = []
        for i, dim in enumerate(self.step_dims):
            ax = fig.add_subplot(len(self.step_dims), 1, i + 1)
            ax.set_title(dim.replace("_", " ").title())
            self.axes.append(ax)
        fig.tight_layout()
        right_splitter.addWidget(self.canvas)

        hbox.addWidget(right_splitter, 3)

        if self.data.cycles is not None and self.data.steps is not None:
            self.cycle_table.populate(self.data.cycles)
            if self.data.cycles.size:
                self.select_cycle(0)
            self.cycle_table.itemSelectionChanged.connect(self._on_cycle_select)
        else:
            self._show_warning("No memory records found in the save directory.")

    def _show_warning(self, text: str):
        banner = QtWidgets.QLabel(text)
        banner.setAlignment(QtCore.Qt.AlignCenter)
        banner.setStyleSheet("QLabel { color: red; font-size: 16px; }")
        self.setCentralWidget(banner)

    def _on_cycle_select(self):
        rows = self.cycle_table.selectionModel().selectedRows()
        if rows:
            self.select_cycle(rows[0].row())

    def select_cycle(self, idx: int):
        steps = self.data.steps_for_cycle(idx)
        self.step_table.populate(steps)
        self._update_plots(steps)

    def _update_plots(self, steps: np.ndarray):
        for ax in self.axes:
            ax.clear()

        if not steps.size:
            self.canvas.draw()
            return

        xs = np.arange(len(steps))
        for ax, dim in zip(self.axes, self.step_dims):
            data = steps[dim]
            ax.plot(xs, data, label=dim)
            ax.legend()

        self.canvas.draw()


def main():
    app = QtWidgets.QApplication(sys.argv)
    data = MemoryData()
    win = AnalyticsWindow(data)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
