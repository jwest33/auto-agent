import sys
import heapq
from typing import Dict, List, Tuple, Optional
import numpy as np
import os
import time
from PyQt5 import QtCore, QtGui, QtWidgets

from module_world import GridWorld, value_to_cost
from module_agent import Memory, StepExperience, CycleSummary

Coord = Tuple[int, int]

class ManualNavigator(QtWidgets.QWidget):
    """Interactive explorer that mirrors the agent's movement & energy logic.

    Controls
    --------
    •  Mouse RIGHT-click ─ auto-walk to a cell
    •  W A S D           ─ manual movement
    •  + / - / 0         ─ zoom in / out / reset
    •  “Goto x,y”        ─ type a coordinate and press ↵ or Go
    """

    MOVE_KEYS = {
        QtCore.Qt.Key_W: (0, -1),
        QtCore.Qt.Key_S: (0,  1),
        QtCore.Qt.Key_A: (-1, 0),
        QtCore.Qt.Key_D: (1,  0),
    }

    def __init__(self, world: GridWorld, start: Coord = (0, 0)):
        super().__init__()

        # --- Hopfield Memory for the manual plan ---
        # ensure the same 'save' directory exists
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "save")
        os.makedirs(save_dir, exist_ok=True)

        # keep track of origin so that encode_cell works the same as the agent
        self.origin = start
        # instantiate the Memory wrapper (default key_dim=12, capacity=4096)
        self.memory = Memory()
        # load any existing state (or start fresh)
        self.memory._raw_steps.clear()
        self.memory.cycles.clear()
        self._manual_cycle_steps: List[Coord] = []
        
        # where to dump only the Hopfield keys/values for this manual run
        self.plan_hopfield_path = os.path.join(save_dir, "memory_dijkstra.npy")
        self.world = world
        self.position = list(start)
        self.prev_pos: Optional[Coord] = None
        self.energy = 100.0
        self.visited: List[Tuple[int, int, float, float]] = []   # (x, y, cost, energy)
        self.auto_path: List[Coord] = []

        self._cell_size = 4
        self._zoom_factor = 1.0

        self._pix_left = 0
        self._pix_top = 0

        self.setWindowTitle("Manual World Navigator")
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(QtCore.Qt.AlignCenter)

        self.canvas = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.canvas.setMinimumSize(600, 600)  # starting viewport
        self.scroll_area.setWidget(self.canvas)

        zoom_layout = QtWidgets.QHBoxLayout()

        self.zoom_out_btn  = QtWidgets.QPushButton("-"); self.zoom_out_btn.setMaximumWidth(40)
        self.zoom_in_btn   = QtWidgets.QPushButton("+"); self.zoom_in_btn.setMaximumWidth(40)
        self.zoom_reset_btn = QtWidgets.QPushButton("Reset"); self.zoom_reset_btn.setMaximumWidth(40)

        self.zoom_out_btn.clicked.connect(self._zoom_out)
        self.zoom_in_btn.clicked.connect(self._zoom_in)
        self.zoom_reset_btn.clicked.connect(self._zoom_reset)

        zoom_layout.addWidget(self.zoom_out_btn)
        zoom_layout.addWidget(self.zoom_in_btn)
        zoom_layout.addWidget(self.zoom_reset_btn)
        
        zoom_layout.addStretch() # push everything left

        # info + history panes
        self.info_box = QtWidgets.QTextEdit(readOnly=True, minimumHeight=60, maximumHeight=80)
        
        self.reset_btn = QtWidgets.QPushButton("Reset Cycle")
        self.reset_btn.clicked.connect(self._reset_cycle)

        self.history_table = QtWidgets.QTableWidget(0, 4)
        self.history_table.setHorizontalHeaderLabels(["Step", "X", "Y", "Energy"])
        self.history_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        # layout
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.scroll_area, 1)
        lay.addLayout(zoom_layout)
        lay.addWidget(self.info_box)
        row = QtWidgets.QHBoxLayout(); row.addStretch(); row.addWidget(self.reset_btn); row.addStretch()
        lay.addLayout(row)
        lay.addWidget(QtWidgets.QLabel("Move History"))
        lay.addWidget(self.history_table)

        # status bar & timer
        self.status_bar = QtWidgets.QStatusBar()
        self.status_bar.showMessage("Ready")
        lay.addWidget(self.status_bar)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._advance_auto_path)

        self._update_status()

    #  zoom helpers
    def _zoom_in(self):    self._zoom_factor *= 1.2; self.update()
    def _zoom_out(self):   self._zoom_factor /= 1.2; self.update()
    def _zoom_reset(self): self._zoom_factor = 1.0;  self.update()

    #  keyboard    
    def keyPressEvent(self, ev: QtGui.QKeyEvent):
        k = ev.key()
        if k in self.MOVE_KEYS: self._attempt_move(self.MOVE_KEYS[k]); return
        if k == QtCore.Qt.Key_Plus:  self._zoom_in();   return
        if k == QtCore.Qt.Key_Minus: self._zoom_out();  return
        if k == QtCore.Qt.Key_0:     self._zoom_reset();return
        super().keyPressEvent(ev)

    #  mouse -> cell
    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        if ev.button() == QtCore.Qt.RightButton:
            # translate click straight into QLabel-coords, then grid
            canvas_pt = self.canvas.mapFrom(self, ev.pos())
            cell = self._pixel_to_grid(canvas_pt)
            if cell:
                self._start_auto_path(cell)
                self.status_bar.showMessage(f"Autowalking to {cell}")
        super().mousePressEvent(ev)

    def _pixel_to_grid(self, pt: QtCore.QPoint) -> Optional[Coord]:
        """Return (gx, gy) for a point already inside the QLabel."""
        scaled = int(self._cell_size * self._zoom_factor)
        if scaled <= 0:
            return None
        gx, gy = pt.x() // scaled, pt.y() // scaled
        if 0 <= gx < self.world.width and 0 <= gy < self.world.height:
            return int(gx), int(gy)
        return None

    #  cycle & movement helpers
    def _attempt_move(self, delta: Coord):
        nx, ny = self.position[0] + delta[0], self.position[1] + delta[1]
        if not (0 <= nx < self.world.width and 0 <= ny < self.world.height):
            self.status_bar.showMessage(f"Cannot move outside grid boundaries")
            return

        # Estimate energy before modifying state
        energy_before = self.energy
        cost, restore = self.world.transition(self.prev_pos, (nx, ny))
        new_energy = 100.0 if restore else self.energy - cost

        if new_energy < 0.0:
            self.status_bar.showMessage(f"Not enough energy for that move")
            self._finalize_cycle("Out of energy")
            return

        # --- Hopfield memory and step recording ---
        goal = (self.world.width - 1, self.world.height - 1)
        pos_before = tuple(self.position)
        cell = (nx, ny)

        # Store experience in Hopfield memory
        self.memory.store_cell_experience(
            self.world,
            self.origin,
            goal,
            pos_before,
            cell,
            energy_before,
            cost
        )

        # Record full step experience
        shade = int(self.world.grid[ny, nx])
        exp_cost = self.memory.estimate_cell_cost(
            self.world, self.origin, goal, pos_before, cell, energy_before
        )
        surprise = abs(exp_cost - cost) if exp_cost is not None else 0.0

        step_exp = StepExperience(
            position=cell,
            shade=shade,
            expected_cost=exp_cost if exp_cost is not None else cost,
            actual_cost=cost,
            surprise=surprise,
            energy_before=energy_before,
            energy_after=new_energy
        )
        self.memory.add_experience(step_exp)
        
        self._manual_cycle_steps.append(tuple(self.position))
        
        # Persist full memory
        self.save_to_file_dijkstra()

        # Apply state change
        self.prev_pos = tuple(self.position)
        self.position = [nx, ny]
        self.energy = new_energy
        self._record_history(cost)
        
    def _finalize_cycle(self, reason: str = "Cycle ended"):
        if self._manual_cycle_steps:
            total_cost = sum(s.actual_cost for s in self.memory._raw_steps)
            total_surprise = sum(s.surprise for s in self.memory._raw_steps)

            summary = CycleSummary(
                reward=0.0,  # or compute normalized distance to goal
                cost=total_cost,
                surprise=total_surprise,
                energy_left=self.energy,
                steps=self._manual_cycle_steps.copy(),
                timestamp=time.time()
            )
            self.memory.cycles.append(summary)

            print(f"[Dijkstra] Cycle finalized: {reason}")
            self.save_to_file_dijkstra()

        # Clear state for next cycle
        self._manual_cycle_steps.clear()
        self.memory._raw_steps.clear()

    def save_to_file_dijkstra(self):
        """Save step memory and cycles to alternate filenames to avoid collisions."""
        alt_memory_path = os.path.join("save", "memory_dijkstra.npy")
        alt_cycle_path = os.path.join("save", "cycles_dijkstra.npy")

        try:
            #print(f"[Dijkstra] Saving memory data to {alt_memory_path}")
            experiences_array = self.memory._experiences_to_array()
            #print(f"[Dijkstra] Saving {len(experiences_array)} experiences")
            np.save(alt_memory_path, experiences_array)

            cycles_array = self.memory._cycles_to_array()
            #print(f"[Dijkstra] Saving {len(cycles_array)} cycles to {alt_cycle_path}")
            np.save(alt_cycle_path, cycles_array)

            #print("[Dijkstra] Memory data saved successfully")

        except Exception as e:
            #print(f"[Dijkstra] Error saving memory data: {e}")
            import traceback
            traceback.print_exc()
            
    def _experiences_to_array(self) -> np.ndarray:
        rows = [
            (
                e.position[0], e.position[1], e.shade,
                e.expected_cost, e.actual_cost, e.surprise, 
                e.energy_before, e.energy_after, e.timestamp,
            )
            for e in self._raw_steps
        ]
        return np.array(rows, dtype=self._step_dtype) if rows else np.empty((0,), self._step_dtype)
    
    def _cycles_to_array(self) -> np.ndarray:
        rows = [(c.reward, c.cost, c.surprise, c.energy_left, c.timestamp) for c in self.cycles]
        return np.array(rows, dtype=self._cycle_dtype) if rows else np.empty((0,), self._cycle_dtype)

    def _reset_cycle(self):
        self.world.reset_cycle()
        self.position = [0, 0]
        self.prev_pos = None
        self.energy = 100.0
        self.visited.clear()
        self.auto_path.clear()
        self.timer.stop()
        self.history_table.setRowCount(0)
        self._update_status()
        self.status_bar.showMessage("Cycle reset")
        
    def _start_auto_path(self, goal: Coord):
        if goal == tuple(self.position): return
        path = self._dijkstra_path(tuple(self.position), goal)
        if path: 
            self.auto_path = path[1:] 
            self.timer.start(20)
        else:
            self.status_bar.showMessage(f"No valid path to {goal}")

    def _advance_auto_path(self):
        if not self.auto_path: 
            self._finalize_cycle("Auto-path complete")
            return

        nx, ny = self.auto_path.pop(0)
        dx = nx - self.position[0]
        dy = ny - self.position[1]
        self._attempt_move((dx, dy))

        if not self.auto_path:
            self._finalize_cycle("Auto-path complete")

    # Dijkstra (energy‑aware, with restore‑pair rule)
    def _dijkstra_path(self, start: Coord, goal: Coord) -> List[Coord]:
        W, H = self.world.width, self.world.height
        dist: Dict[Coord, float] = {start: 0.0}; prev: Dict[Coord, Optional[Coord]] = {start: None}
        pq = [(0.0, start, None)]; visited_pairs = set()
        while pq:
            d, node, came = heapq.heappop(pq)
            if node == goal: break
            if d != dist[node]: continue
            x, y = node
            for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
                mx, my = x+dx, y+dy
                if not (0<=mx<W and 0<=my<H): continue
                neigh = (mx, my); prev_val = int(self.world.grid[y, x]); curr_val = int(self.world.grid[my, mx])
                step = value_to_cost(curr_val); restore = False
                if came is not None and prev_val!=0 and curr_val%prev_val==0:
                    pair = (prev_val, curr_val)
                    if pair not in visited_pairs: restore=True; visited_pairs.add(pair)
                nd = d if restore else d+step
                if nd < dist.get(neigh, float('inf')):
                    dist[neigh] = nd; prev[neigh] = node; heapq.heappush(pq, (nd, neigh, node))
        else: return []
        # rebuild
        path = []; n = goal
        while n is not None: path.append(n); n = prev[n]
        return list(reversed(path))

    def _update_status(self):
        x, y = self.position; cost = value_to_cost(int(self.world.grid[y, x]))
        self.info_box.setPlainText(
            f"Position: ({x}, {y})\n"
            f"Energy left: {self.energy:.2f}\n"
            f"Current cell cost: {cost:.2f}\n"
            f"Visited: {len(self.visited)}\n"
            f"Zoom: {self._zoom_factor:.1f}x"
        )
        self.update()

    def _record_history(self, cost: float):
        x, y = self.position; self.visited.append((x, y, cost, self.energy))
        row = self.history_table.rowCount(); self.history_table.insertRow(row)
        for col, val in enumerate([row, x, y, f"{self.energy:.1f}"]):
            itm = QtWidgets.QTableWidgetItem(str(val)); itm.setTextAlignment(QtCore.Qt.AlignCenter)
            self.history_table.setItem(row, col, itm)
        self.history_table.scrollToBottom()

    def paintEvent(self, _):
        # Calculate base cell size to ensure grid fits
        base_size = max(2, min(
            self.scroll_area.viewport().width() // self.world.width,
            self.scroll_area.viewport().height() // self.world.height
        ))
        
        # Apply zoom factor
        self._cell_size = base_size
        scaled_cell_size = max(1, int(self._cell_size * self._zoom_factor))
        
        # Calculate pixmap dimensions
        pm_w = self.world.width * scaled_cell_size
        pm_h = self.world.height * scaled_cell_size
        
        # Create pixmap and painter
        pm = QtGui.QPixmap(pm_w, pm_h)
        pm.fill(QtGui.QColor("white"))
        qp = QtGui.QPainter(pm)
        
        # Draw grid cells
        for y in range(self.world.height):
            for x in range(self.world.width):
                shade = int(self.world.grid[y, x])
                # Create more distinct coloring
                base = 255 - int((shade/9.0)*195)
                # Add a light blue tint to explored cells
                if self.world.explored[y, x]:
                    qp.fillRect(
                        x * scaled_cell_size, 
                        y * scaled_cell_size, 
                        scaled_cell_size, 
                        scaled_cell_size, 
                        QtGui.QColor(base, base, 255)
                    )
                else:
                    qp.fillRect(
                        x * scaled_cell_size, 
                        y * scaled_cell_size, 
                        scaled_cell_size, 
                        scaled_cell_size, 
                        QtGui.QColor(base, base, base)
                    )
                
                # Draw grid lines
                qp.setPen(QtGui.QPen(QtGui.QColor(200, 200, 200, 100), 1))
                qp.drawRect(
                    x * scaled_cell_size, 
                    y * scaled_cell_size, 
                    scaled_cell_size, 
                    scaled_cell_size
                )
                
                # Draw cell value text for large enough cells
                if scaled_cell_size >= 15:
                    qp.setPen(QtGui.QColor(0, 0, 0))
                    qp.drawText(
                        x * scaled_cell_size, 
                        y * scaled_cell_size, 
                        scaled_cell_size, 
                        scaled_cell_size, 
                        QtCore.Qt.AlignCenter, 
                        str(shade)
                    )
        
        # Draw trail
        if len(self.visited) > 1:
            pen = QtGui.QPen(QtGui.QColor(255,100,0,180), max(2, scaled_cell_size // 3))
            qp.setPen(pen)
            for i in range(1, len(self.visited)):
                ax, ay, *_ = self.visited[i-1]
                bx, by, *_ = self.visited[i]
                qp.drawLine(
                    int((ax+0.5) * scaled_cell_size), 
                    int((ay+0.5) * scaled_cell_size),
                    int((bx+0.5) * scaled_cell_size), 
                    int((by+0.5) * scaled_cell_size)
                )

        # Goal highlight (if path active)
        if self.auto_path:
            gx, gy = self.auto_path[-1]
            qp.setPen(QtGui.QPen(QtGui.QColor(0, 200, 0), 2, QtCore.Qt.DashLine))
            qp.setBrush(QtCore.Qt.NoBrush)
            qp.drawRect(
                gx * scaled_cell_size, 
                gy * scaled_cell_size,
                scaled_cell_size, 
                scaled_cell_size
            )

        # Draw current position
        px, py = self.position
        qp.setPen(QtCore.Qt.NoPen)
        qp.setBrush(QtGui.QColor(0, 200, 0))
        marker_size = max(scaled_cell_size-2, scaled_cell_size//2)
        qp.drawEllipse(
            px * scaled_cell_size + (scaled_cell_size - marker_size)//2, 
            py * scaled_cell_size + (scaled_cell_size - marker_size)//2,
            marker_size, 
            marker_size
        )
        
        # Mark origin with a special symbol
        if not (self.position[0] == 0 and self.position[1] == 0):
            qp.setBrush(QtGui.QColor(0, 0, 200, 150))
            qp.drawEllipse(
                1 + scaled_cell_size//4, 
                1 + scaled_cell_size//4,
                scaled_cell_size//2, 
                scaled_cell_size//2
            )
        
        qp.end()

        # Set the pixmap to the label
        self.canvas.setPixmap(pm)
        
        # Adjust canvas size to fit the pixmap
        self.canvas.setFixedSize(pm_w, pm_h)
        
        # Calculate pixmap offset inside QLabel for click-to-cell conversion
        self._pix_left = 0
        self._pix_top = 0
        
        # Update status with grid dimensions
        self.status_bar.showMessage(f"Grid: {self.world.width}x{self.world.height} | Cell size: {scaled_cell_size}px | Energy: {self.energy:.1f}")

def main():
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication(sys.argv)
    
    # Apply style sheet for a more modern look
    app.setStyle("Fusion")
    
    # Create world and navigator
    world = GridWorld(150, 150)
    nav = ManualNavigator(world)
    
    # Set window to a larger size
    nav.resize(1200, 900)
    
    # Center on screen
    screen = QtWidgets.QDesktopWidget().screenGeometry()
    nav.move((screen.width() - nav.width()) // 2, (screen.height() - nav.height()) // 2)
    
    nav.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
