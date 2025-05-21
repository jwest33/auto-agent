from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class HyperParamEvolutionWidget(QtWidgets.QWidget):
    """Live line-plot of ALPHA, BACKTRACK_PENALTY, RECENT_VISIT_PENALTY, CURIOSITY_WEIGHT."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self._records = [] # list of (cycle#, params-dict)

        layout = QtWidgets.QVBoxLayout(self)
        self.figure = plt.Figure(figsize=(5, 4), dpi=100, facecolor="white")
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.axes = self.figure.add_subplot(111)
        self.axes.set_title("Hyper-parameter Evolution")
        self.axes.set_xlabel("Cycle")
        self.axes.set_ylabel("Value")
        self.axes.grid(True, linestyle="--", alpha=0.6)

        self.axes.set_xlim(0, 10) # Default X-axis range
        self.axes.set_ylim(0, 20) # Default Y-axis range
        self.axes.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        self.canvas.draw_idle()

        self.lines = {name: self.axes.plot([], [], label=name)[0]
                      for name in ("ALPHA",
                                   "BACKTRACK_PENALTY",
                                   "RECENT_VISIT_PENALTY",
                                   "CURIOSITY_WEIGHT")}

        self.axes.legend(loc="upper right", fontsize=8)
        self.figure.tight_layout()

    def add_cycle(self, cycle_no: int, params: dict):
        """Append one point per parameter and refresh the plot."""
        self._records.append((cycle_no, params.copy()))
        self._refresh()

    def _refresh(self):
        if not self._records:
            return

        xs = [c for c, _ in self._records]
        for name, line in self.lines.items():
            ys = [p[name] for _, p in self._records]
            line.set_data(xs, ys)
            line.set_marker("o")
            line.set_markersize(4)

        if xs:
            x0, x1 = min(xs), max(xs)
            if x0 == x1:
                pad = 1
                self.axes.set_xlim(x0 - pad, x1 + pad)
            else:
                self.axes.set_xlim(x0, x1)
        self.axes.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        self.axes.relim()
        self.axes.autoscale_view()
        self.canvas.draw_idle()
        
    def clear(self):
        """Erase all data and restore the empty axes."""
        self._records.clear()
        for line in self.lines.values():
            line.set_data([], [])
        # restore default limits so the next point is visible
        self.axes.set_xlim(0, 10)
        self.axes.set_ylim(0, 20)
        self.canvas.draw_idle()
