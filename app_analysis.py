from __future__ import annotations
import math
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import torch

# Add import for HopfieldMemory to analyze memory queries
from module_hopfield_memory import HopfieldMemory

BASE_DIR = Path(__file__).resolve().parent
SAVE_DIR = BASE_DIR / "save"
MEMORY_PATH = SAVE_DIR / "memory.npy"
CYCLE_PATH = SAVE_DIR / "cycles.npy"
ORIGIN: Tuple[int, int] = (0, 0)

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
QTabWidget::pane {
    border: 1px solid #e0e6ed;
    background-color: white;
    border-radius: 6px;
}
QTabBar::tab {
    background-color: #f0f4f8;
    color: #34495e;
    padding: 8px 16px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    border: 1px solid #e0e6ed;
    border-bottom: none;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: white;
    border-bottom-color: white;
}
QTabBar::tab:hover:!selected {
    background-color: #e1e9f0;
}
QSplitter::handle {
    background-color: #e0e6ed;
}
QLabel {
    color: #2c3e50;
}
"""

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
        self.hopfield_memory: Optional[HopfieldMemory] = None
        self.positions: Dict[Tuple[int, int], List[int]] = {}  # maps positions to step indices

        self._load_files()
        if self.steps is not None and self.cycles is not None:
            self._build_cycle_slices()
            self._index_positions()
            self._load_hopfield_memory()

    def _load_files(self):
        """Load memory data from files with improved error handling."""
        try:
            if MEMORY_PATH.exists() and MEMORY_PATH.stat().st_size > 0:
                self.steps = np.load(MEMORY_PATH, allow_pickle=False).astype(_STEP_DTYPE, copy=False)
                print(f"Loaded {len(self.steps)} memory steps")
            else:
                print("No memory steps file found or file is empty")
                
            if CYCLE_PATH.exists() and CYCLE_PATH.stat().st_size > 0:
                self.cycles = np.load(CYCLE_PATH, allow_pickle=False).astype(_CYCLE_DTYPE, copy=False)
                print(f"Loaded {len(self.cycles)} cycles")
            else:
                print("No cycles file found or file is empty")
        except Exception as e:
            print(f"Error loading memory files: {e}")
            # Create empty arrays if loading fails
            self.steps = np.empty((0,), _STEP_DTYPE)
            self.cycles = np.empty((0,), _CYCLE_DTYPE)

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
            
    def _index_positions(self):
        """Build an index mapping grid positions to step indices for quick lookup."""
        if self.steps is None or not len(self.steps):
            return
            
        for i, step in enumerate(self.steps):
            pos = (step["x"], step["y"])
            if pos not in self.positions:
                self.positions[pos] = []
            self.positions[pos].append(i)
            
    def _load_hopfield_memory(self):
        """Load Hopfield memory from module_hopfield_memory if available."""
        try:
            # This is just a stub to demonstrate how we'd connect to the actual memory
            # In a real implementation, we'd load the actual memory state
            self.hopfield_memory = HopfieldMemory(key_dim=12, value_dim=1, capacity=4096)
            
            # If we had access to the actual memory weights, we'd load them here
            # For now, we'll just create a sample memory for demonstration
            if self.steps is not None and len(self.steps) > 0:
                # Create some sample key-value pairs based on the data we have
                sample_count = min(100, len(self.steps))
                for i in range(sample_count):
                    try:
                        key = torch.randn(12)  # Random key for demonstration
                        value = torch.tensor([self.steps[i]["actual_cost"]])
                        self.hopfield_memory.write(key, value)
                    except Exception as e:
                        print(f"Error writing to Hopfield memory: {e}")
                        continue
                    
            print(f"Created sample Hopfield memory with {self.hopfield_memory.item_count} items")
        except Exception as e:
            print(f"Could not load Hopfield memory: {e}")
            self.hopfield_memory = None

    def steps_for_cycle(self, idx: int) -> np.ndarray:
        """Get all steps for a specific cycle."""
        if not self.cycle_indices:
            return np.empty((0,), _STEP_DTYPE)
        return self.steps[self.cycle_indices[idx]]
    
    def get_step_positions_in_cycle(self, cycle_idx: int) -> List[Tuple[int, int]]:
        """Get all unique positions visited in a cycle."""
        steps = self.steps_for_cycle(cycle_idx)
        if not len(steps):
            return []
        return [(step["x"], step["y"]) for step in steps]
    
    def get_memory_metrics(self) -> Dict:
        """Get overall memory metrics."""
        if self.steps is None or not len(self.steps):
            return {
                "total_steps": 0,
                "total_cycles": 0,
                "unique_positions": 0,
                "avg_surprise": 0,
                "avg_cost": 0,
                "max_surprise": 0
            }
            
        return {
            "total_steps": len(self.steps),
            "total_cycles": len(self.cycles) if self.cycles is not None else 0,
            "unique_positions": len(self.positions),
            "avg_surprise": float(np.mean(self.steps["surprise"])),
            "avg_cost": float(np.mean(self.steps["actual_cost"])),
            "max_surprise": float(np.max(self.steps["surprise"]))
        }
    
    def simulate_memory_query(self, position: Tuple[int, int]) -> Dict:
        """Simulate a memory query for a given position.
        In a real implementation, this would query the actual memory."""
        if position not in self.positions:
            return {
                "found": False,
                "predicted_cost": None,
                "similar_positions": [],
                "memory_dimensions": {}
            }
        
        # Get all steps for this position
        indices = self.positions[position]
        steps = self.steps[indices]
        
        # Calculate average cost and surprise
        avg_cost = float(np.mean(steps["actual_cost"]))
        avg_surprise = float(np.mean(steps["surprise"]))
        avg_energy_before = float(np.mean(steps["energy_before"]))
        avg_energy_after = float(np.mean(steps["energy_after"]))
        
        # Calculate all memory dimensions
        x, y = position
        
        # Extract step information
        shade_values = steps["shade"]
        avg_shade = float(np.mean(shade_values))
        
        # Calculate estimated distances
        goal = (149, 149)  # Assuming goal is at bottom-right corner
        distance_to_origin = math.sqrt(x**2 + y**2)
        distance_to_goal = math.sqrt((goal[0] - x)**2 + (goal[1] - y)**2)
        
        # Direction vectors (simplified - in a real implementation we'd use the actual agent code)
        dir_origin = self._normalize_vector((-x, -y))
        dir_goal = self._normalize_vector((goal[0] - x, goal[1] - y))
        
        # Find similar positions (this is simplified - in a real implementation
        # we would use the Hopfield memory to find similar positions)
        similar_positions = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (x + dx, y + dy)
            if neighbor in self.positions:
                neighbor_indices = self.positions[neighbor]
                neighbor_steps = self.steps[neighbor_indices]
                neighbor_avg_cost = float(np.mean(neighbor_steps["actual_cost"]))
                neighbor_avg_surprise = float(np.mean(neighbor_steps["surprise"]))
                similar_positions.append({
                    "position": neighbor,
                    "visits": len(neighbor_indices),
                    "avg_cost": neighbor_avg_cost,
                    "avg_surprise": neighbor_avg_surprise,
                    "distance": 1.0  # Manhattan distance
                })
        
        # Compute all memory dimensions as they would be used in the Hopfield memory
        memory_dimensions = {
            "direction_from_origin": dir_origin,
            "direction_to_goal": dir_goal,
            "normalized_energy": avg_energy_before / 100.0,
            "distance_to_goal": distance_to_goal / 211.0,  # Normalize by max possible distance
            "shade": avg_shade / 9.0,  # Normalize by max shade
            "surrounding_shades": self._estimate_surrounding_shades(position),
            "position_x": x / 149.0,  # Normalize by grid size
            "position_y": y / 149.0,  # Normalize by grid size
            "visit_count": len(indices),
            "surprise_history": avg_surprise,
            "cost_history": avg_cost,
            "energy_before": avg_energy_before,
            "energy_after": avg_energy_after,
            "energy_delta": avg_energy_before - avg_energy_after
        }
        
        return {
            "found": True,
            "position": position,
            "visits": len(indices),
            "predicted_cost": avg_cost,
            "avg_surprise": avg_surprise,
            "steps": indices,
            "similar_positions": similar_positions,
            "memory_dimensions": memory_dimensions
        }
    
    def _normalize_vector(self, vec: Tuple[float, float]) -> Tuple[float, float]:
        """Normalize a 2D vector."""
        x, y = vec
        mag = math.sqrt(x**2 + y**2)
        if mag == 0:
            return (0.0, 0.0)
        return (x / mag, y / mag)
    
    def _estimate_surrounding_shades(self, position: Tuple[int, int]) -> List[float]:
        """Estimate the surrounding shades for a position."""
        x, y = position
        neighbors = [(x+1, y), (x, y+1), (x-1, y), (x, y-1)]
        result = []
        
        for nx, ny in neighbors:
            if (nx, ny) in self.positions:
                neighbor_indices = self.positions[(nx, ny)]
                if neighbor_indices:
                    neighbor_steps = self.steps[neighbor_indices]
                    avg_shade = float(np.mean(neighbor_steps["shade"])) / 9.0  # Normalize
                    result.append(avg_shade)
                else:
                    result.append(0.0)
            else:
                result.append(0.0)
        
        # Ensure we have exactly 4 values (pad if needed)
        result.extend([0.0] * (4 - len(result)))
        return result[:4]

class MemoryMetricsWidget(QtWidgets.QGroupBox):
    """Widget to display overall memory metrics."""
    
    def __init__(self, parent=None):
        super().__init__("Memory Metrics", parent)
        self.setStyleSheet("QGroupBox { font-weight: bold; }")
        
        # Create layout
        layout = QtWidgets.QFormLayout(self)
        layout.setVerticalSpacing(8)
        layout.setHorizontalSpacing(20)
        
        # Create metric labels
        self.total_steps_label = QtWidgets.QLabel("0")
        self.total_cycles_label = QtWidgets.QLabel("0")
        self.unique_positions_label = QtWidgets.QLabel("0")
        self.avg_surprise_label = QtWidgets.QLabel("0.00")
        self.avg_cost_label = QtWidgets.QLabel("0.00")
        self.max_surprise_label = QtWidgets.QLabel("0.00")
        
        # Format labels
        for label in [self.total_steps_label, self.total_cycles_label, 
                     self.unique_positions_label, self.avg_surprise_label,
                     self.avg_cost_label, self.max_surprise_label]:
            label.setStyleSheet("font-weight: bold; color: #2980b9;")
        
        # Add to layout
        layout.addRow("Total Steps:", self.total_steps_label)
        layout.addRow("Total Cycles:", self.total_cycles_label)
        layout.addRow("Unique Positions:", self.unique_positions_label)
        layout.addRow("Avg. Surprise:", self.avg_surprise_label)
        layout.addRow("Avg. Cost:", self.avg_cost_label)
        layout.addRow("Max Surprise:", self.max_surprise_label)
    
    def update_metrics(self, metrics: Dict):
        """Update displayed metrics."""
        self.total_steps_label.setText(str(metrics["total_steps"]))
        self.total_cycles_label.setText(str(metrics["total_cycles"]))
        self.unique_positions_label.setText(str(metrics["unique_positions"]))
        self.avg_surprise_label.setText(f"{metrics['avg_surprise']:.2f}")
        self.avg_cost_label.setText(f"{metrics['avg_cost']:.2f}")
        self.max_surprise_label.setText(f"{metrics['max_surprise']:.2f}")

class CycleTable(QtWidgets.QTableWidget):
    """Enhanced table to display cycle information."""
    
    headers = ["#", "Reward", "Cost", "Surprise", "Energy left", "Steps"]

    def __init__(self, parent=None):
        super().__init__(0, len(self.headers), parent)
        self.setHorizontalHeaderLabels(self.headers)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setStretchLastSection(True)
        
        # Configure sizing
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)
        
        # Add context menu
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)
    
    def _context_menu(self, pos):
        """Create context menu for the table."""
        menu = QtWidgets.QMenu(self)
        
        # Add actions
        export_action = menu.addAction("Export Selected Cycles")
        export_action.triggered.connect(self._export_selected)
        
        # Show the menu
        menu.exec_(self.mapToGlobal(pos))
    
    def _export_selected(self):
        """Export selected cycles to CSV."""
        selected_rows = sorted(set(index.row() for index in self.selectedIndexes()))
        
        if not selected_rows:
            return
            
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Cycles", "", "CSV Files (*.csv)"
        )
        
        if not path:
            return
            
        try:
            with open(path, 'w') as f:
                # Write header
                f.write(",".join(self.headers) + "\n")
                
                # Write rows
                for row in selected_rows:
                    values = []
                    for col in range(self.columnCount()):
                        item = self.item(row, col)
                        values.append(item.text() if item else "")
                    f.write(",".join(values) + "\n")
                    
            QtWidgets.QMessageBox.information(
                self, "Export Successful", f"Exported {len(selected_rows)} cycles to {path}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Export Failed", f"Failed to export cycles: {e}"
            )

    def populate(self, cycles: np.ndarray, indices: List[slice]):
        """Populate the table with cycle data."""
        self.setRowCount(len(cycles))
        self.setSortingEnabled(False)  # Disable sorting while populating
        
        for row, cyc in enumerate(cycles):
            # Get number of steps in this cycle
            step_count = indices[row].stop - indices[row].start if row < len(indices) else 0
            
            items = [
                str(row + 1),
                f"{cyc['reward']:.2f}",
                f"{cyc['cost']:.2f}",
                f"{cyc['surprise']:.2f}",
                f"{cyc['energy_left']:.1f}",
                str(step_count)
            ]
            
            for col, text in enumerate(items):
                itm = QtWidgets.QTableWidgetItem(text)
                itm.setTextAlignment(QtCore.Qt.AlignCenter)
                
                # Add conditional formatting for energy level
                if col == 4:  # Energy column
                    energy = cyc["energy_left"]
                    if energy < 30:
                        itm.setForeground(QtGui.QColor("#e74c3c"))  # Red for low energy
                    elif energy < 60:
                        itm.setForeground(QtGui.QColor("#f39c12"))  # Orange for medium energy
                    else:
                        itm.setForeground(QtGui.QColor("#27ae60"))  # Green for high energy
                
                self.setItem(row, col, itm)
        
        # Re-enable sorting
        self.setSortingEnabled(True)
        
        # Resize columns to content
        self.resizeColumnsToContents()

class StepTable(QtWidgets.QTableWidget):
    """Enhanced table to display step information."""
    
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
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.setAlternatingRowColors(True)
        
        # Add tooltip with explanation
        self.setToolTip(
            "Idx: Step index\n"
            "x, y: Position coordinates\n"
            "shade: Cell value (0-9)\n"
            "exp_cost: Expected energy cost\n"
            "act_cost: Actual energy cost\n"
            "surprise: |exp_cost - act_cost|\n"
            "E_before: Energy before step\n"
            "E_after: Energy after step"
        )

    def populate(self, steps: np.ndarray):
        """Populate the table with step data."""
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
                itm.setTextAlignment(QtCore.Qt.AlignCenter)
                
                # Highlight surprise with color coding
                if col == 6:  # Surprise column
                    surprise = rec["surprise"]
                    if surprise > 3.0:
                        itm.setForeground(QtGui.QColor("#e74c3c"))  # Red for high surprise
                    elif surprise > 1.0:
                        itm.setForeground(QtGui.QColor("#f39c12"))  # Orange for medium surprise
                    else:
                        itm.setForeground(QtGui.QColor("#27ae60"))  # Green for low surprise
                
                # Highlight cost differences
                if col == 4:  # Expected cost
                    exp = rec["expected_cost"]
                    act = rec["actual_cost"]
                    if abs(exp - act) > 2.0:
                        itm.setBackground(QtGui.QColor(255, 235, 235))  # Light red for significant error
                
                self.setItem(i, col, itm)

class MemoryQueryPanel(QtWidgets.QGroupBox):
    """Panel to display memory query results."""
    
    def __init__(self, parent=None):
        super().__init__("Memory Query Analysis", parent)
        
        # Create layout
        layout = QtWidgets.QVBoxLayout(self)
        
        # Create position entry
        pos_layout = QtWidgets.QHBoxLayout()
        pos_layout.addWidget(QtWidgets.QLabel("Position (x, y):"))
        
        self.x_spin = QtWidgets.QSpinBox()
        self.x_spin.setRange(0, 999)
        self.y_spin = QtWidgets.QSpinBox()
        self.y_spin.setRange(0, 999)
        
        pos_layout.addWidget(self.x_spin)
        pos_layout.addWidget(QtWidgets.QLabel(","))
        pos_layout.addWidget(self.y_spin)
        
        self.query_btn = QtWidgets.QPushButton("Query Memory")
        self.query_btn.clicked.connect(self._on_query)
        pos_layout.addWidget(self.query_btn)
        pos_layout.addStretch()
        
        layout.addLayout(pos_layout)
        
        # Results area
        self.results_text = QtWidgets.QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet(
            "background-color: white; border: 1px solid #e0e6ed; padding: 8px;"
        )
        layout.addWidget(self.results_text)
        
        # No data initially
        self.data = None
    
    def set_data(self, data: MemoryData):
        """Set the memory data to query."""
        self.data = data
    
    def update_for_position(self, x: int, y: int):
        """Update display for a specific position."""
        self.x_spin.setValue(x)
        self.y_spin.setValue(y)
        self._on_query()
    
    def _on_query(self):
        """Handle memory query."""
        if not self.data:
            self.results_text.setHtml("<p>No memory data available</p>")
            return
            
        x = self.x_spin.value()
        y = self.y_spin.value()
        pos = (x, y)
        
        # Query the memory
        result = self.data.simulate_memory_query(pos)
        
        # Display results
        if not result["found"]:
            self.results_text.setHtml(
                f"<p style='color: #e74c3c;'><b>Position ({x}, {y}) not found in memory</b></p>"
            )
            return
        
        # Format results as HTML
        html = f"""
        <h3 style='color: #2980b9;'>Position ({x}, {y})</h3>
        <p><b>Visits:</b> {result['visits']}</p>
        <p><b>Predicted Cost:</b> {result['predicted_cost']:.2f}</p>
        <p><b>Average Surprise:</b> {result.get('avg_surprise', 0.0):.2f}</p>
        
        <hr/>
        <h4 style='color: #2980b9;'>Memory Dimensions:</h4>
        <table style='width: 100%; border-collapse: collapse;'>
        <tr style='background-color: #f0f4f8;'>
            <th style='padding: 6px; border: 1px solid #e0e6ed;'>Dimension</th>
            <th style='padding: 6px; border: 1px solid #e0e6ed;'>Value</th>
            <th style='padding: 6px; border: 1px solid #e0e6ed;'>Description</th>
        </tr>
        """
        
        memory_dimensions = result.get('memory_dimensions', {})
        
        # Add dimensions to table with descriptions
        dimension_descriptions = {
            "direction_from_origin": "Normalized vector pointing from this position to the origin",
            "direction_to_goal": "Normalized vector pointing from this position to the goal",
            "normalized_energy": "Current energy level (normalized to 0-1 range)",
            "distance_to_goal": "Distance to goal (normalized by max possible distance)",
            "shade": "Cell shade value (normalized to 0-1 range)",
            "surrounding_shades": "Normalized shade values of the 4 neighboring cells",
            "position_x": "X coordinate (normalized to 0-1 range)",
            "position_y": "Y coordinate (normalized to 0-1 range)",
            "visit_count": "Number of times this position was visited",
            "surprise_history": "Average surprise value from previous visits",
            "cost_history": "Average energy cost from previous visits",
            "energy_before": "Average energy before stepping to this position",
            "energy_after": "Average energy after stepping to this position",
            "energy_delta": "Average energy change from stepping to this position"
        }
        
        for dim, value in memory_dimensions.items():
            description = dimension_descriptions.get(dim, "")
            
            # Format the value based on its type
            formatted_value = value
            if isinstance(value, tuple) and len(value) == 2:
                formatted_value = f"({value[0]:.2f}, {value[1]:.2f})"
            elif isinstance(value, list):
                formatted_value = "[" + ", ".join(f"{v:.2f}" for v in value) + "]"
            elif isinstance(value, (float, int)):
                formatted_value = f"{value:.4f}"
            
            html += f"""
            <tr>
                <td style='padding: 6px; border: 1px solid #e0e6ed;'>{dim}</td>
                <td style='padding: 6px; border: 1px solid #e0e6ed;'>{formatted_value}</td>
                <td style='padding: 6px; border: 1px solid #e0e6ed;'>{description}</td>
            </tr>
            """
        
        html += """
        </table>
        <hr/>
        <h4 style='color: #2980b9;'>Similar Positions:</h4>
        <table style='width: 100%; border-collapse: collapse;'>
        <tr style='background-color: #f0f4f8;'>
            <th style='padding: 6px; border: 1px solid #e0e6ed;'>Position</th>
            <th style='padding: 6px; border: 1px solid #e0e6ed;'>Visits</th>
            <th style='padding: 6px; border: 1px solid #e0e6ed;'>Avg. Cost</th>
            <th style='padding: 6px; border: 1px solid #e0e6ed;'>Avg. Surprise</th>
            <th style='padding: 6px; border: 1px solid #e0e6ed;'>Distance</th>
        </tr>
        """
        
        for sim in result["similar_positions"]:
            html += f"""
            <tr>
                <td style='padding: 6px; border: 1px solid #e0e6ed;'>{sim['position']}</td>
                <td style='padding: 6px; border: 1px solid #e0e6ed;'>{sim['visits']}</td>
                <td style='padding: 6px; border: 1px solid #e0e6ed;'>{sim['avg_cost']:.2f}</td>
                <td style='padding: 6px; border: 1px solid #e0e6ed;'>{sim.get('avg_surprise', 0.0):.2f}</td>
                <td style='padding: 6px; border: 1px solid #e0e6ed;'>{sim.get('distance', 0.0):.2f}</td>
            </tr>
            """
        
        html += """
        </table>
        <hr/>
        <h4 style='color: #2980b9;'>Memory Encoding Process:</h4>
        <p>When the agent encounters a position, it encodes it into a 12-dimensional vector:</p>
        <ol>
            <li><b>Direction vectors</b> (6 dimensions):
                <ul>
                    <li>From origin to the cell: 2 dimensions (x, y)</li>
                    <li>From goal to the cell: 2 dimensions (x, y)</li>
                    <li>From current position to the cell: 2 dimensions (x, y)</li>
                </ul>
            </li>
            <li><b>Energy state</b> (1 dimension): Current normalized energy level</li>
            <li><b>Goal information</b> (1 dimension): Normalized distance to goal</li>
            <li><b>Cell information</b> (4 dimensions): Current cell shade and surrounding cell shades</li>
        </ol>
        <p>This 12-dimensional vector serves as the key for the Hopfield memory lookup.</p>
        <p>When predicting a cell's cost, the agent computes a similar vector and uses it<br/>
        to query the Hopfield network, which returns a weighted average of costs<br/>
        from similar experiences in the memory.</p>
        """
        
        self.results_text.setHtml(html)

class HeatmapWidget(QtWidgets.QWidget):
    """Widget to display heatmaps of memory data."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create layout
        layout = QtWidgets.QVBoxLayout(self)
        
        # Create metric selector
        metrics_layout = QtWidgets.QHBoxLayout()
        metrics_layout.addWidget(QtWidgets.QLabel("Metric:"))
        
        self.metric_combo = QtWidgets.QComboBox()
        self.metric_combo.addItems([
            "Visit Count", "Surprise", "Cost", "Cost Error"
        ])
        self.metric_combo.currentIndexChanged.connect(self._update_heatmap)
        metrics_layout.addWidget(self.metric_combo)
        
        # Add cycle range
        metrics_layout.addWidget(QtWidgets.QLabel("Cycle:"))
        
        self.cycle_combo = QtWidgets.QComboBox()
        self.cycle_combo.addItem("All Cycles")
        self.cycle_combo.currentIndexChanged.connect(self._update_heatmap)
        metrics_layout.addWidget(self.cycle_combo)
        
        metrics_layout.addStretch()
        layout.addLayout(metrics_layout)
        
        # Create figure
        self.figure = Figure(figsize=(6, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas)
        
        # No data initially
        self.data = None
        self._positions = {}
        self._current_metric = "Visit Count"
        self._current_cycle = "All Cycles"
    
    def set_data(self, data: MemoryData):
        """Set the memory data and update the cycle selector."""
        self.data = data
        
        # Update cycle selector
        self.cycle_combo.clear()
        self.cycle_combo.addItem("All Cycles")
        
        if data and data.cycles is not None:
            for i in range(len(data.cycles)):
                self.cycle_combo.addItem(f"Cycle {i+1}")
        
        # Compute position metrics for all cycles
        try:
            self._compute_position_metrics()
        except Exception as e:
            print(f"Error computing position metrics: {e}")
            import traceback
            traceback.print_exc()
        
        # Update the heatmap
        try:
            self._update_heatmap()
        except Exception as e:
            print(f"Error updating heatmap: {e}")
            import traceback
            traceback.print_exc()
    
    def _compute_position_metrics(self):
        """Compute metrics for each visited position."""
        if not self.data or self.data.steps is None or not len(self.data.steps):
            self._positions = {}
            return
        
        # Group by position
        positions = {}
        for i, step in enumerate(self.data.steps):
            pos = (step["x"], step["y"])
            if pos not in positions:
                positions[pos] = {
                    "visits": 0,
                    "surprise": [],
                    "cost": [],
                    "cost_error": []
                }
            
            positions[pos]["visits"] += 1
            positions[pos]["surprise"].append(step["surprise"])
            positions[pos]["cost"].append(step["actual_cost"])
            positions[pos]["cost_error"].append(abs(step["expected_cost"] - step["actual_cost"]))
        
        # Compute statistics
        for pos, stats in positions.items():
            stats["avg_surprise"] = np.mean(stats["surprise"])
            stats["avg_cost"] = np.mean(stats["cost"])
            stats["avg_cost_error"] = np.mean(stats["cost_error"])
        
        self._positions = positions
    
    def _filter_by_cycle(self, cycle_idx: Optional[int]):
        """Filter positions by cycle."""
        if cycle_idx is None:
            return self._positions
        
        if not self.data or cycle_idx >= len(self.data.cycle_indices):
            return {}
        
        steps = self.data.steps_for_cycle(cycle_idx)
        positions = {}
        
        for step in steps:
            pos = (step["x"], step["y"])
            if pos not in positions:
                positions[pos] = {
                    "visits": 0,
                    "surprise": [],
                    "cost": [],
                    "cost_error": []
                }
            
            positions[pos]["visits"] += 1
            positions[pos]["surprise"].append(step["surprise"])
            positions[pos]["cost"].append(step["actual_cost"])
            positions[pos]["cost_error"].append(abs(step["expected_cost"] - step["actual_cost"]))
        
        # Compute statistics
        for pos, stats in positions.items():
            stats["avg_surprise"] = np.mean(stats["surprise"])
            stats["avg_cost"] = np.mean(stats["cost"])
            stats["avg_cost_error"] = np.mean(stats["cost_error"])
        
        return positions
    
    def _update_heatmap(self):
        """Update the heatmap with the selected metric and cycle."""
        self.ax.clear()
        
        metric = self.metric_combo.currentText()
        cycle_text = self.cycle_combo.currentText()
        
        # Get positions based on cycle
        cycle_idx = None
        if cycle_text != "All Cycles":
            try:
                # More robust parsing of cycle index
                index_text = ''.join(filter(str.isdigit, cycle_text))
                if index_text:
                    cycle_idx = int(index_text) - 1
            except (ValueError, IndexError):
                print(f"Could not parse cycle index from '{cycle_text}'")
        
        positions = self._filter_by_cycle(cycle_idx)
        
        if not positions:
            self.ax.text(0.5, 0.5, 'No data available', 
                horizontalalignment='center',
                verticalalignment='center',
                transform=self.ax.transAxes,
                fontsize=12)
            self.canvas.draw()
            return
        
        # Extract data for the selected metric
        x = []
        y = []
        values = []
        
        for pos, stats in positions.items():
            x.append(pos[0])
            y.append(pos[1])
            
            if metric == "Visit Count":
                values.append(stats["visits"])
            elif metric == "Surprise":
                values.append(stats["avg_surprise"])
            elif metric == "Cost":
                values.append(stats["avg_cost"])
            elif metric == "Cost Error":
                values.append(stats["avg_cost_error"])
        
        # Create scatter plot with colorbar
        scatter = self.ax.scatter(
            x, y, c=values, s=100, 
            cmap='viridis', alpha=0.7, edgecolor='k', linewidth=0.5
        )
        
        # Add colorbar and labels
        cbar = self.figure.colorbar(scatter, ax=self.ax)
        cbar.set_label(metric)
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        
        # Set title
        cycle_title = "All Cycles" if cycle_idx is None else f"Cycle {cycle_idx+1}"
        self.ax.set_title(f"{metric} Heatmap ({cycle_title})")
        
        # Add grid
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set equal aspect
        self.ax.set_aspect('equal')
        
        # Update the canvas
        self.figure.tight_layout()
        self.canvas.draw()

class MemoryNetworkWidget(QtWidgets.QWidget):
    """Widget to visualize the memory network and connections."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create layout
        layout = QtWidgets.QVBoxLayout(self)
        
        # Create figure
        self.figure = Figure(figsize=(6, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas)
        
        # No data initially
        self.data = None
    
    def set_data(self, data: MemoryData):
        """Set the memory data and update the visualization."""
        self.data = data
        self._update_visualization()
    
    def _update_visualization(self):
        """Update the network visualization."""
        self.ax.clear()
        
        if not self.data or self.data.steps is None or not len(self.data.steps):
            self.ax.text(0.5, 0.5, 'No data available', 
                horizontalalignment='center',
                verticalalignment='center',
                transform=self.ax.transAxes,
                fontsize=12)
            self.canvas.draw()
            return
        
        # Get all positions
        positions = self.data.positions
        
        if not positions:
            self.ax.text(0.5, 0.5, 'No position data available', 
                horizontalalignment='center',
                verticalalignment='center',
                transform=self.ax.transAxes,
                fontsize=12)
            self.canvas.draw()
            return
        
        # Create a graph representation
        pos_list = list(positions.keys())
        
        # Plot nodes
        x = [p[0] for p in pos_list]
        y = [p[1] for p in pos_list]
        
        # Size nodes by visit count
        sizes = [len(positions[p]) * 5 for p in pos_list]
        
        # Create transitions dictionary
        transitions = {}
        
        for i in range(len(self.data.steps) - 1):
            curr = (self.data.steps[i]["x"], self.data.steps[i]["y"])
            next_pos = (self.data.steps[i+1]["x"], self.data.steps[i+1]["y"])
            
            if curr == next_pos:
                continue
                
            key = (curr, next_pos)
            transitions[key] = transitions.get(key, 0) + 1
        
        # Plot nodes
        scatter = self.ax.scatter(
            x, y, s=sizes, c=sizes,
            cmap='viridis', alpha=0.7, edgecolor='k', linewidth=0.5
        )
        
        # Plot edges
        for (src, dst), weight in transitions.items():
            # Skip weak connections to reduce clutter
            if weight < 2:
                continue
                
            self.ax.plot(
                [src[0], dst[0]], [src[1], dst[1]],
                'k-', alpha=min(0.1 + weight * 0.05, 0.8),
                linewidth=weight * 0.2
            )
        
        # Add colorbar and labels
        cbar = self.figure.colorbar(scatter, ax=self.ax)
        cbar.set_label('Visit Count')
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        
        # Set title
        self.ax.set_title(f"Memory Network (Positions: {len(pos_list)}, Transitions: {len(transitions)})")
        
        # Add grid
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set equal aspect
        self.ax.set_aspect('equal')
        
        # Update the canvas
        self.figure.tight_layout()
        self.canvas.draw()

class AnalyticsPanel(QtWidgets.QWidget):
    def __init__(self, data: MemoryData, parent=None):
        super().__init__(parent)
        self.data = data

        # Apply consistent UI style if desired
        self.setStyleSheet(APP_STYLES)

        # Main layout
        layout = QtWidgets.QVBoxLayout(self)

        # Top panel: memory metrics summary
        self.metrics_widget = MemoryMetricsWidget()
        layout.addWidget(self.metrics_widget)

        # Central tab view for analytics
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)

        # Create tab contents
        self.create_data_explorer_tab()
        self.create_memory_heatmap_tab()
        self.create_memory_network_tab()
        self.create_memory_query_tab()

        # Populate memory metrics and tables if data is available
        if self.data.cycles is not None and self.data.steps is not None:
            self.cycle_table.populate(self.data.cycles, self.data.cycle_indices)

            if self.data.cycles.size:
                self.select_cycle(0)

            metrics = self.data.get_memory_metrics()
            self.metrics_widget.update_metrics(metrics)

            self.heatmap_widget.set_data(self.data)
            self.network_widget.set_data(self.data)
            self.query_panel.set_data(self.data)

            self.cycle_table.itemSelectionChanged.connect(self._on_cycle_select)

    def configure_matplotlib_style(self):
        """Configure matplotlib for a modern, professional look"""
        plt.style.use('ggplot')
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = ['Segoe UI', 'SF Pro Display', 'Arial']
        matplotlib.rcParams['axes.labelcolor'] = '#2c3e50'
        matplotlib.rcParams['axes.edgecolor'] = '#bdc3c7'
        matplotlib.rcParams['axes.linewidth'] = 1.0
        matplotlib.rcParams['axes.grid'] = True
        matplotlib.rcParams['grid.alpha'] = 0.3
        matplotlib.rcParams['grid.color'] = '#bdc3c7'
        matplotlib.rcParams['xtick.color'] = '#2c3e50'
        matplotlib.rcParams['ytick.color'] = '#2c3e50'
        matplotlib.rcParams['figure.facecolor'] = 'white'

    def create_data_explorer_tab(self):
        """Create the data explorer tab with tables and charts."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(tab)
        
        # Left side: tables
        tables_layout = QtWidgets.QVBoxLayout()
        
        # Cycle table
        cycle_group = QtWidgets.QGroupBox("Cycle History")
        cycle_layout = QtWidgets.QVBoxLayout(cycle_group)
        self.cycle_table = CycleTable()
        cycle_layout.addWidget(self.cycle_table)
        tables_layout.addWidget(cycle_group)
        
        # Step table
        step_group = QtWidgets.QGroupBox("Step Details")
        step_layout = QtWidgets.QVBoxLayout(step_group)
        self.step_table = StepTable()
        step_layout.addWidget(self.step_table)
        tables_layout.addWidget(step_group)
        
        layout.addLayout(tables_layout, 1)
        
        # Right side: charts
        charts_layout = QtWidgets.QVBoxLayout()
        charts_group = QtWidgets.QGroupBox("Step Analytics")
        ch_layout = QtWidgets.QVBoxLayout(charts_group)
        
        # Create figure with increased height for better spacing
        self.figure = plt.Figure(figsize=(6, 10), dpi=100, facecolor='white', layout='constrained')  # Increased height from 9 to 10
        self.canvas = FigureCanvas(self.figure)
        
        # Create subplots with better spacing
        self.ax_surprise = self.figure.add_subplot(3, 1, 1)
        self.ax_energy = self.figure.add_subplot(3, 1, 2)
        self.ax_distance = self.figure.add_subplot(3, 1, 3)
        
        ch_layout.addWidget(self.canvas)
        charts_layout.addWidget(charts_group)
        
        layout.addLayout(charts_layout, 1)
        
        # Add to tabs
        self.tabs.addTab(tab, "Data Explorer")

    def create_memory_heatmap_tab(self):
        """Create the memory heatmap tab."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        
        # Create heatmap widget
        self.heatmap_widget = HeatmapWidget()
        layout.addWidget(self.heatmap_widget)
        
        # Add to tabs
        self.tabs.addTab(tab, "Memory Heatmap")

    def create_memory_network_tab(self):
        """Create the memory network tab."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        
        # Create network widget
        self.network_widget = MemoryNetworkWidget()
        layout.addWidget(self.network_widget)
        
        # Add to tabs
        self.tabs.addTab(tab, "Memory Network")

    def create_memory_query_tab(self):
        """Create the memory query tab."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        
        # Create query panel
        self.query_panel = MemoryQueryPanel()
        layout.addWidget(self.query_panel)
        
        # Add to tabs
        self.tabs.addTab(tab, "Memory Query")

    def _show_warning(self, text: str):
        banner = QtWidgets.QLabel(text)
        banner.setAlignment(QtCore.Qt.AlignCenter)
        banner.setStyleSheet("QLabel { color: #e74c3c; font-size: 16px; padding: 20px; }")
        self.setCentralWidget(banner)

    def _on_cycle_select(self):
        rows = self.cycle_table.selectionModel().selectedRows()
        if rows:
            self.select_cycle(rows[0].row())

    def select_cycle(self, idx: int):
        """Display data for the selected cycle."""
        steps = self.data.steps_for_cycle(idx)
        self.step_table.populate(steps)
        self._update_plots(steps)
        
        # Update the status bar with cycle info
        if idx < len(self.data.cycles):
            cycle = self.data.cycles[idx]
            print(
                f"[Analytics] Cycle {idx+1}: Reward={cycle['reward']:.2f}, "
                f"Cost={cycle['cost']:.2f}, Surprise={cycle['surprise']:.2f}, "
                f"Energy Left={cycle['energy_left']:.1f}"
            )

    def _update_plots(self, steps: np.ndarray):
        """Update the plots with step data."""
        for ax in [self.ax_surprise, self.ax_energy, self.ax_distance]:
            ax.clear()

        if not steps.size:
            self.canvas.draw()
            return

        xs = np.arange(len(steps))
        
        # Plot surprise
        surprise_data = steps["surprise"]
        self.ax_surprise.plot(xs, surprise_data, 'o-', color='#3498db', linewidth=2)
        self.ax_surprise.set_title("Surprise (|Expected - Actual Cost|)", fontsize=12, pad=10)
        self.ax_surprise.grid(True, linestyle='--', alpha=0.7)
        self.ax_surprise.set_ylabel("Surprise Value", fontsize=10)
        
        # Add fill between
        self.ax_surprise.fill_between(xs, 0, surprise_data, alpha=0.2, color='#3498db')
        
        # Plot energy
        energy_data = steps["energy_after"]
        self.ax_energy.plot(xs, energy_data, 'o-', color='#27ae60', linewidth=2)
        self.ax_energy.set_title("Energy Remaining", fontsize=12, pad=10)
        self.ax_energy.grid(True, linestyle='--', alpha=0.7)
        self.ax_energy.set_ylabel("Energy Value", fontsize=10)
        self.ax_energy.fill_between(xs, 0, energy_data, alpha=0.2, color='#27ae60')
        
        # Plot distance from origin
        distance_data = []
        for step in steps:
            x, y = step["x"], step["y"]
            distance = math.sqrt((x - ORIGIN[0])**2 + (y - ORIGIN[1])**2)
            distance_data.append(distance)
        
        self.ax_distance.plot(xs, distance_data, 'o-', color='#e74c3c', linewidth=2)
        self.ax_distance.set_title("Distance from Origin", fontsize=12, pad=10)
        self.ax_distance.grid(True, linestyle='--', alpha=0.7)
        self.ax_distance.set_ylabel("Distance", fontsize=10)
        self.ax_distance.set_xlabel("Step Index", fontsize=10)
        self.ax_distance.fill_between(xs, 0, distance_data, alpha=0.2, color='#e74c3c')
        
        # Set xticks to match step indices
        for ax in [self.ax_surprise, self.ax_energy, self.ax_distance]:
            if len(xs) > 20:
                # If too many points, show fewer ticks
                ax.set_xticks(xs[::len(xs)//10])
            else:
                ax.set_xticks(xs)

        self.canvas.draw()

def main():
    app = QtWidgets.QApplication([])
    data = MemoryData()
    win = AnalyticsPanel(data)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    
    main()
