import json
import os
import random
import sys
from typing import List, Optional
import platform
import argparse

import colorsys
from PyQt5 import QtCore, QtGui, QtWidgets

from module_agent import Agent, Memory, euclidean, MEMORY_PATH, CYCLE_PATH, StepExperience
from module_world import GridWorld, WORLD_PATH, BASE_CELL_COST, Coord

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib as mpl

# Constants
DEFAULT_SIZE = 150
DEFAULT_SEED = None

# Parse command line arguments
parser = argparse.ArgumentParser(description='Hopfield Memory Pathfinder')
parser.add_argument('--reset', action='store_true', help='Reset all memory files at startup')
args, unknown_args = parser.parse_known_args()

# Reset memory files if requested
if args.reset:
    print("Resetting all memory files...")
    paths = [MEMORY_PATH, WORLD_PATH, CYCLE_PATH]
    for path in paths:
        if os.path.exists(path):
            os.remove(path)
            print(f"Removed {path}")

# Ensure OS-compatible paths
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "save")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
CYCLE_HISTORY_PATH = os.path.join(save_dir, "cycle_history.json")

# Modern Professional UI Style
APP_STYLES = """
QWidget { 
    font-family: 'Segoe UI', 'SF Pro Display', Arial, sans-serif; 
    background-color: #f7f9fc; 
    color: #2c3e50; 
    font-size: 10pt;
}
QPushButton { 
    padding: 8px 16px; 
    border-radius: 4px; 
    background-color: #3498db; 
    color: white; 
    font-weight: 500;
    border: none;
}
QPushButton:hover { 
    background-color: #2980b9; 
}
QPushButton:pressed {
    background-color: #1c6ea4;
}
QPushButton:disabled {
    background-color: #bdc3c7;
}
QGroupBox { 
    border: 1px solid #e0e6ed; 
    border-radius: 6px; 
    margin-top: 16px; 
    background-color: white; 
    padding: 10px;
    font-weight: 500;
}
QGroupBox::title { 
    subcontrol-origin: margin; 
    subcontrol-position: top center; 
    padding: 0 10px; 
    color: #34495e; 
    font-weight: bold;
    background-color: white;
}
QTableWidget { 
    background-color: white; 
    alternate-background-color: #f5f8fa;
    border: 1px solid #e0e6ed; 
    gridline-color: #ecf0f1;
    selection-background-color: #d6eaf8;
    selection-color: #2c3e50;
}
QTableWidget::item { 
    padding: 4px;
    border-bottom: 1px solid #ecf0f1;
}
QHeaderView::section { 
    background-color: #f0f4f8; 
    padding: 6px; 
    border: none;
    border-right: 1px solid #e0e6ed;
    border-bottom: 1px solid #e0e6ed;
    color: #34495e;
    font-weight: bold;
}
QSpinBox { 
    border: 1px solid #e0e6ed;
    background-color: white;
    color: #2c3e50;
    padding: 4px;
    border-radius: 4px;
}
QSpinBox::up-button, QSpinBox::down-button {
    border: none;
    background-color: transparent;
    width: 0px;
    height: 0px;
}
QLabel {
    color: #2c3e50;
}
"""

class AgentWindow(QtWidgets.QMainWindow):
    def __init__(self, grid_size: int, seed: Optional[int]):
        super().__init__()
        
        # Set application style
        self.setStyleSheet(APP_STYLES)
        
        # Configure matplotlib for better chart appearance
        self.configure_matplotlib_style()
        
        # Configure main window
        self.setWindowTitle("Hopfield Memory Pathfinder")
        self.resize(1280, 900)
        self.central = QtWidgets.QWidget()
        self.setCentralWidget(self.central)
        
        # Main layout
        main_layout = QtWidgets.QHBoxLayout(self.central)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Left panel containing the grid canvas
        left_panel = QtWidgets.QVBoxLayout()
        left_panel.setSpacing(10)
        
        # Grid canvas
        self.canvas_frame = QtWidgets.QFrame()
        self.canvas_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.canvas_frame.setStyleSheet("background-color: white; border-radius: 6px;")
        canvas_layout = QtWidgets.QVBoxLayout(self.canvas_frame)
        
        grid_title = QtWidgets.QLabel("Grid Environment")
        grid_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 5px;")
        grid_title.setAlignment(QtCore.Qt.AlignCenter)
        canvas_layout.addWidget(grid_title)
        
        self.canvas = QtWidgets.QLabel()
        self.canvas.setFixedSize(grid_size * 5, grid_size * 5)
        self.canvas.setAlignment(QtCore.Qt.AlignCenter)
        self.canvas.setStyleSheet("border: none;")
        canvas_layout.addWidget(self.canvas, alignment=QtCore.Qt.AlignCenter)
        
        left_panel.addWidget(self.canvas_frame)
        
        # Status bar
        self.status_bar = QtWidgets.QFrame()
        self.status_bar.setStyleSheet("background-color: white; border-radius: 6px; padding: 8px;")
        status_layout = QtWidgets.QHBoxLayout(self.status_bar)
        
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setStyleSheet("font-weight: 500;")
        
        self.cycles_run_label = QtWidgets.QLabel("Cycles: 0")
        self.cycles_run_label.setStyleSheet("font-weight: 500;")
        
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.cycles_run_label)
        
        left_panel.addWidget(self.status_bar)
        main_layout.addLayout(left_panel, stretch=5)
        
        # Right panel
        right_panel = QtWidgets.QVBoxLayout()
        right_panel.setSpacing(10)
        main_layout.addLayout(right_panel, stretch=4)
        
        # Controls group
        ctrl_group = QtWidgets.QGroupBox("Controls")
        ctrl_layout = QtWidgets.QVBoxLayout()
        ctrl_group.setLayout(ctrl_layout)
        
        cycles_widget = QtWidgets.QWidget()
        cycles_layout = QtWidgets.QHBoxLayout(cycles_widget)
        cycles_layout.setContentsMargins(0, 0, 0, 0)
        
        # Cycle control with descriptive labels
        cycles_label = QtWidgets.QLabel("Number of Cycles:")
        cycles_label.setStyleSheet("font-weight: 500;")
        cycles_layout.addWidget(cycles_label)
        
        self.spin_cycles = QtWidgets.QSpinBox()
        self.spin_cycles.setMinimum(1)
        self.spin_cycles.setMaximum(100)
        self.spin_cycles.setValue(1)
        self.spin_cycles.setFixedWidth(60)
        self.spin_cycles.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        cycles_layout.addWidget(self.spin_cycles)
        
        minus_btn = QtWidgets.QPushButton("−")
        minus_btn.setFixedSize(40, 30)
        minus_btn.clicked.connect(lambda: self.spin_cycles.setValue(max(1, self.spin_cycles.value() - 1)))
        cycles_layout.addWidget(minus_btn)
        
        plus_btn = QtWidgets.QPushButton("+")
        plus_btn.setFixedSize(40, 30)
        plus_btn.clicked.connect(lambda: self.spin_cycles.setValue(min(100, self.spin_cycles.value() + 1)))
        cycles_layout.addWidget(plus_btn)
        
        cycles_layout.addStretch()
        
        # Button container with better spacing
        button_container = QtWidgets.QWidget()
        button_layout = QtWidgets.QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)
        
        self.run_btn = QtWidgets.QPushButton("Run Cycles")
        self.run_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.run_btn.clicked.connect(self.on_run_cycles)
        
        self.reset_world_btn = QtWidgets.QPushButton("Rebuild World")
        self.reset_world_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload))
        self.reset_world_btn.clicked.connect(self.on_rebuild_world)
        
        self.reset_mem_btn = QtWidgets.QPushButton("Reset Memory")
        self.reset_mem_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogResetButton))
        self.reset_mem_btn.clicked.connect(self.on_reset_memory)
        
        # Add buttons to layout with equal sizing
        for btn in (self.run_btn, self.reset_world_btn, self.reset_mem_btn):
            button_layout.addWidget(btn)
            btn.setMinimumWidth(120)
        
        # Add widgets to control layout
        ctrl_layout.addWidget(cycles_widget)
        ctrl_layout.addWidget(button_container)
        
        right_panel.addWidget(ctrl_group)
        
        # History table with improved styling
        history_group = QtWidgets.QGroupBox("Cycle History")
        hist_layout = QtWidgets.QVBoxLayout()
        history_group.setLayout(hist_layout)
        
        self.table = QtWidgets.QTableWidget(0, 5)
        header_labels = ["Cycle #", "Reward", "Cost", "Surprise", "Energy"]
        self.table.setHorizontalHeaderLabels(header_labels)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        
        hist_layout.addWidget(self.table)
        right_panel.addWidget(history_group)
        
        # Charts with improved styling
        charts = QtWidgets.QGroupBox("Analytics")
        ch_layout = QtWidgets.QVBoxLayout()
        charts.setLayout(ch_layout)
        
        # Use a larger figure with better proportions and white background
        self.figure = plt.Figure(figsize=(6, 9), dpi=100, facecolor='white')
        self.canvas_chart = FigureCanvas(self.figure)
        self.canvas_chart.setStyleSheet("background-color: white;")
        
        # Create subplots with better spacing
        self.ax_surprise = self.figure.add_subplot(3, 1, 1)
        self.ax_energy = self.figure.add_subplot(3, 1, 2)
        self.ax_distance = self.figure.add_subplot(3, 1, 3)
        
        ch_layout.addWidget(self.canvas_chart)
        right_panel.addWidget(charts, stretch=2)
        
        # Initialize world/agent
        self.world = GridWorld(grid_size, grid_size, random.Random(seed))
        self.agent = Agent((0, 0))
        self.cycles_run = 0
        self.current_dummy_plan = None
        self.cycle_history: List[dict] = []
        self.load_cycle_history()
        
        # Visualization helpers
        self.trails: List[List[Coord]] = []
        self.step_history: List[list] = []
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_canvas)
        self.timer.start(200)
        
        self.update_canvas()
        self.update_charts()
        self.update_status_labels()

    def configure_matplotlib_style(self):
        """Configure matplotlib for a modern, professional look"""
        plt.style.use('ggplot')
        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['font.sans-serif'] = ['Segoe UI', 'SF Pro Display', 'Arial']
        mpl.rcParams['axes.labelcolor'] = '#2c3e50'
        mpl.rcParams['axes.edgecolor'] = '#bdc3c7'
        mpl.rcParams['axes.linewidth'] = 1.0
        mpl.rcParams['axes.grid'] = True
        mpl.rcParams['grid.alpha'] = 0.3
        mpl.rcParams['grid.color'] = '#bdc3c7'
        mpl.rcParams['xtick.color'] = '#2c3e50'
        mpl.rcParams['ytick.color'] = '#2c3e50'
        mpl.rcParams['figure.facecolor'] = 'white'

    def update_status_labels(self):
        """Update status bar information"""
        self.cycles_run_label.setText(f"Cycles: {self.cycles_run}")
        
        if self.cycles_run > 0 and self.cycle_history:
            latest = self.cycle_history[-1]
            energy_text = f"{latest['energy']:.1f}"
            self.status_label.setText(f"Energy: {energy_text}")
        else:
            self.status_label.setText("Ready")

    def load_cycle_history(self):
        if os.path.exists(CYCLE_HISTORY_PATH):
            try:
                with open(CYCLE_HISTORY_PATH, "r") as f:
                    self.cycle_history = json.load(f)
                    for s in self.cycle_history:
                        self.add_history_row(s)
                    if self.cycle_history:
                        self.cycles_run = self.cycle_history[-1]["cycle"]
            except (json.JSONDecodeError, IOError, KeyError) as e:
                print(f"Warning: Error loading cycle history: {e}")
                os.remove(CYCLE_HISTORY_PATH)
                print("Removed incompatible cycle history file. Starting fresh.")
                self.cycle_history = []

    def save_cycle_history(self):
        os.makedirs(os.path.dirname(CYCLE_HISTORY_PATH), exist_ok=True)
        with open(CYCLE_HISTORY_PATH, "w") as f:
            json.dump(self.cycle_history, f, indent=2)

    def add_history_row(self, stats):
        """Add a row to the history table with better formatting"""
        r = self.table.rowCount()
        self.table.insertRow(r)
        
        # Define column values with proper formatting
        columns = [
            str(stats["cycle"]),
            f"{stats['reward']:.2f}",
            f"{stats['cost']:.2f}",
            f"{stats['surprise']:.2f}",
            f"{stats['energy']:.2f}"
        ]
        
        # Add each column to the table
        for i, text in enumerate(columns):
            item = QtWidgets.QTableWidgetItem(text)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            
            # Special formatting for energy based on value
            if i == 4:  # Energy column
                energy = stats["energy"]
                if energy < 30:
                    item.setForeground(QtGui.QColor("#e74c3c"))  # Red for low energy
                elif energy < 60:
                    item.setForeground(QtGui.QColor("#f39c12"))  # Orange for medium energy
                else:
                    item.setForeground(QtGui.QColor("#27ae60"))  # Green for high energy
            
            self.table.setItem(r, i, item)
        
        # Scroll to the newest row
        self.table.scrollToBottom()

    def on_run_cycles(self):
        """Start running the specified number of cycles"""
        self.trails.clear()
        self.step_history.clear()
        self.pending = self.spin_cycles.value()
        
        # Disable buttons during run
        self.run_btn.setEnabled(False)
        self.reset_world_btn.setEnabled(False)
        self.reset_mem_btn.setEnabled(False)
        
        self.status_label.setText("Running cycles...")
        self.run_next_cycle()

    def run_next_cycle(self):
        if self.pending <= 0:
            self.save_cycle_history()
            # Re-enable buttons after completion
            self.run_btn.setEnabled(True)
            self.reset_world_btn.setEnabled(True)
            self.reset_mem_btn.setEnabled(True)
            self.status_label.setText("Completed")
            self.update_status_labels()
            return
        
        self.trails.append([])
        self.step_history.append([])
        self.world.reset_cycle()
        self.agent.start_cycle()
        old_explored = self.world.explored.copy()
        self.current_dummy_plan = self.agent.choose_plan(self.world)
        
        # Wrap the agent's experience recording to capture metrics
        orig_add = self.agent.memory.add_experience
        def wrapped(exp: StepExperience):
            orig_add(exp)
            distance_from_start = euclidean(self.agent.origin, exp.position)
            self.step_history[-1].append({
                "expected_cost": exp.expected_cost,
                "actual_cost":   exp.actual_cost,
                "surprise":      exp.surprise,
                "energy":        self.agent.energy,
                "distance":      distance_from_start,
            })
        self.agent.memory.add_experience = wrapped
        
        # Execute the plan with visual updates
        self.agent.execute_plan(
            self.current_dummy_plan,
            self.world,
            repaint_cb=self.repaint_and_update_charts,
        )
        
        # Restore original function
        self.agent.memory.add_experience = orig_add
        
        # Update cycle statistics
        self.cycles_run += 1
        new_cells = int((~old_explored & self.world.explored).sum())
        dist = euclidean(self.agent.origin, self.agent.position)
        reward = dist + new_cells
        total_cost = sum(entry["actual_cost"] for entry in self.step_history[-1]) if self.step_history[-1] else 0
        total_surprise = sum(entry["surprise"] for entry in self.step_history[-1]) if self.step_history[-1] else 0
        
        # Record cycle statistics
        stats = {
            "cycle":    self.cycles_run,
            "reward":   reward,
            "cost":     total_cost,
            "surprise": total_surprise,
            "energy":   self.agent.energy,
        }
        self.cycle_history.append(stats)
        self.add_history_row(stats)
        
        # Update UI
        self.update_charts()
        self.update_status_labels()
        self.pending -= 1
        
        # Continue to next cycle
        QtCore.QTimer.singleShot(100, self.run_next_cycle)
    
    def on_rebuild_world(self):
        """Reset the world to a fresh state"""
        if os.path.exists(WORLD_PATH): 
            os.remove(WORLD_PATH)
        if os.path.exists(CYCLE_HISTORY_PATH): 
            os.remove(CYCLE_HISTORY_PATH)
        
        self.world = GridWorld(self.world.width, self.world.height)
        self.reset_all()
        self.update_canvas()
        self.status_label.setText("World rebuilt")

    def on_reset_memory(self):
        """Reset agent memory"""
        if os.path.exists(MEMORY_PATH): 
            os.remove(MEMORY_PATH)
        if os.path.exists(CYCLE_HISTORY_PATH): 
            os.remove(CYCLE_HISTORY_PATH)
        
        self.agent.memory = Memory()
        self.reset_all()
        self.update_canvas()
        self.status_label.setText("Memory reset")

    def reset_all(self):
        """Reset all cycle data"""
        self.cycles_run = 0
        self.current_dummy_plan = None
        self.cycle_history.clear()
        self.table.setRowCount(0)
        self.update_status_labels()

    def repaint_and_update_charts(self):
        """Update both canvas and charts"""
        self.update_canvas()
        self.update_charts()

    def update_canvas(self):
        """Update the grid visualization with enhanced rendering"""
        sz = self.world.width * 5
        pm = QtGui.QPixmap(sz, sz)
        pm.fill(QtGui.QColor("#ffffff"))
        p = QtGui.QPainter(pm)
        csize = 5
        
        # Add current position to trail if exists
        if self.trails: 
            self.trails[-1].append(self.agent.position)
        
        # Draw grid cells with improved coloring
        for y in range(self.world.height):
            for x in range(self.world.width):
                val = int(self.world.grid[y, x])
                # Create more visually distinct coloring for cells
                if self.world.explored[y, x]:
                    # Use a color gradient from white to blue for explored areas
                    intensity = int(255 - (val / 9.0) * 195)
                    # Blend between white and light blue based on cell value
                    r = intensity
                    g = intensity
                    b = min(255, intensity + 40)  # Blue tint for explored cells
                    col = QtGui.QColor(r, g, b, 200)
                else:
                    # Use grayscale for unexplored areas
                    brightness = int(255 - (val / 9.0) * 195)
                    col = QtGui.QColor(brightness, brightness, brightness, 255)
                
                p.fillRect(x * csize, y * csize, csize, csize, col)
        
        # Draw trails with better visibility
        n = len(self.trails)
        for i, trail in enumerate(self.trails):
            if len(trail) < 2: 
                continue
            
            # Use a more vibrant color palette
            hue = (i * 360 / n) / 360.0
            rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            
            # Draw trail with glow effect
            glow_color = QtGui.QColor(*[int(c * 255) for c in rgb], 60)
            glow_pen = QtGui.QPen(glow_color)
            glow_pen.setWidth(4)
            p.setPen(glow_pen)
            
            for a, b in zip(trail, trail[1:]):
                p.drawLine(
                    int((a[0] + 0.5) * csize), 
                    int((a[1] + 0.5) * csize), 
                    int((b[0] + 0.5) * csize), 
                    int((b[1] + 0.5) * csize)
                )
            
            # Draw main trail line
            main_color = QtGui.QColor(*[int(c * 255) for c in rgb], 180)
            main_pen = QtGui.QPen(main_color)
            main_pen.setWidth(2)
            p.setPen(main_pen)
            
            for a, b in zip(trail, trail[1:]):
                p.drawLine(
                    int((a[0] + 0.5) * csize), 
                    int((a[1] + 0.5) * csize), 
                    int((b[0] + 0.5) * csize), 
                    int((b[1] + 0.5) * csize)
                )
        
        # Draw agent with a more visible marker
        p.setPen(QtCore.Qt.NoPen)
        
        # Draw glow around agent
        p.setBrush(QtGui.QColor(220, 20, 60, 100))
        ax, ay = self.agent.position
        p.drawEllipse(
            int(ax * csize - csize/2), 
            int(ay * csize - csize/2), 
            int(csize * 2), 
            int(csize * 2)
        )
        
        # Draw agent
        p.setBrush(QtGui.QColor(220, 20, 60, 255))
        p.drawEllipse(
            int(ax * csize), 
            int(ay * csize), 
            csize, 
            csize
        )
        
        p.end()
        self.canvas.setPixmap(pm)

    def update_charts(self):
        """Update analytics charts with improved styling"""
        # Clear all axes
        for ax in (self.ax_surprise, self.ax_energy, self.ax_distance):
            ax.clear()
        
        n = len(self.step_history)
        if n == 0:
            # If no data, show placeholders
            for ax in (self.ax_surprise, self.ax_energy, self.ax_distance):
                ax.text(0.5, 0.5, 'No data available', 
                       horizontalalignment='center',
                       verticalalignment='center',
                       transform=ax.transAxes,
                       fontsize=10,
                       color='#95a5a6')
                ax.set_xticks([])
                ax.set_yticks([])
            
            # Set titles anyway
            self.ax_surprise.set_title("Surprise", fontsize=12, pad=10)
            self.ax_energy.set_title("Energy Remaining", fontsize=12, pad=10)
            self.ax_distance.set_title("Distance from Start", fontsize=12, pad=10)
            
            self.figure.tight_layout()
            self.canvas_chart.draw()
            return
            
        # Set fixed colors for metrics to maintain consistency across cycles
        surprise_color = '#3498db'  # Blue for surprise
        energy_color = '#27ae60'    # Green for energy
        distance_color = '#e74c3c'  # Red for distance
        
        max_surprise = 0
        max_distance = 0
        
        # Process each cycle's data
        for idx, hist in enumerate(self.step_history):
            if not hist: 
                continue
                
            # Calculate this cycle's color (for consistent coloring)
            hue = (idx * 360 / n) / 360.0
            rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
            line_color = [c * 255 for c in rgb]
            
            # Extract data series
            surprise_data = [d["surprise"] for d in hist]
            energy_data = [d["energy"] for d in hist]
            distance_data = [d["distance"] for d in hist]
            xs = list(range(len(hist)))
            
            # Update max values for scaling
            max_surprise = max(max_surprise, max(surprise_data) if surprise_data else 0)
            max_distance = max(max_distance, max(distance_data) if distance_data else 0)
            
            # Plot each series with improved styling
            # Plot surprise data
            self.ax_surprise.plot(
                xs, surprise_data, 
                color=tuple(c/255 for c in line_color),
                linewidth=2,
                alpha=0.8,
                label=f"Cycle {self.cycles_run - n + idx + 1}"
            )
            
            # Plot energy data
            self.ax_energy.plot(
                xs, energy_data, 
                color=tuple(c/255 for c in line_color),
                linewidth=2,
                alpha=0.8
            )
            
            # Plot distance data
            self.ax_distance.plot(
                xs, distance_data, 
                color=tuple(c/255 for c in line_color),
                linewidth=2,
                alpha=0.8
            )
        
        # Set axis limits with padding
        self.ax_surprise.set_ylim(0, max(0.1, max_surprise * 1.1))
        self.ax_energy.set_ylim(0, 100)
        self.ax_distance.set_ylim(0, max(0.1, max_distance * 1.1))
        
        # Add grid lines for better readability
        for ax in (self.ax_surprise, self.ax_energy, self.ax_distance):
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add subtle fill_between for the most recent cycle
            if self.step_history and self.step_history[-1]:
                last_cycle = self.step_history[-1]
                xs = list(range(len(last_cycle)))
                
                if ax == self.ax_surprise:
                    data = [d["surprise"] for d in last_cycle]
                    ax.fill_between(xs, 0, data, alpha=0.1, color=surprise_color)
                elif ax == self.ax_energy:
                    data = [d["energy"] for d in last_cycle]
                    ax.fill_between(xs, 0, data, alpha=0.1, color=energy_color)
                elif ax == self.ax_distance:
                    data = [d["distance"] for d in last_cycle]
                    ax.fill_between(xs, 0, data, alpha=0.1, color=distance_color)
        
        # Set titles and labels
        self.ax_surprise.set_title("Surprise", fontsize=12, pad=10, fontweight='bold')
        self.ax_energy.set_title("Energy Remaining", fontsize=12, pad=10, fontweight='bold')
        self.ax_distance.set_title("Distance from Start", fontsize=12, pad=10, fontweight='bold')
        
        # Add x and y labels
        self.ax_surprise.set_ylabel("Value", fontsize=10)
        self.ax_energy.set_ylabel("Value", fontsize=10)
        self.ax_distance.set_ylabel("Value", fontsize=10)
        self.ax_distance.set_xlabel("Steps", fontsize=10)
        
        # Add legend to first plot
        if len(self.step_history) > 1:
            self.ax_surprise.legend(
                fontsize=8,
                loc='upper right',
                framealpha=0.7
            )
        
        # Improve tick label formatting
        for ax in (self.ax_surprise, self.ax_energy, self.ax_distance):
            ax.tick_params(labelsize=8)
            
            # Format y-axis ticks to avoid clutter
            ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
        
        # Add final values as annotations for the latest cycle
        if self.step_history and self.step_history[-1]:
            last_cycle = self.step_history[-1]
            
            if last_cycle:
                last_step = last_cycle[-1]
                last_x = len(last_cycle) - 1
                
                # Add last value annotations
                self.ax_surprise.annotate(
                    f"{last_step['surprise']:.2f}",
                    xy=(last_x, last_step['surprise']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    fontweight='bold'
                )
                
                self.ax_energy.annotate(
                    f"{last_step['energy']:.2f}",
                    xy=(last_x, last_step['energy']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    fontweight='bold'
                )
                
                self.ax_distance.annotate(
                    f"{last_step['distance']:.2f}",
                    xy=(last_x, last_step['distance']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    fontweight='bold'
                )
        
        # Apply tight layout for better spacing
        self.figure.tight_layout()
        self.canvas_chart.draw()

def main():
    app = QtWidgets.QApplication(sys.argv)
    
    # Enable high DPI support
    app.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    
    # Create and show the window
    win = AgentWindow(grid_size=DEFAULT_SIZE, seed=DEFAULT_SEED)
    win.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
