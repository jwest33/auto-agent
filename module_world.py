from typing import Callable, List, Optional, Tuple
import os
import random
import platform

import numpy as np

MAX_CELL_COST = 5.0
BASE_CELL_COST = 1.0
Coord = Tuple[int, int]

# Ensure OS-compatible paths
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "save")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
WORLD_PATH = os.path.join(save_dir, "world.npy")


def value_to_cost(value: int) -> float:
    """Convert a cell value to an energy cost."""
    return BASE_CELL_COST + (value / 9.0) * (MAX_CELL_COST - BASE_CELL_COST)


class GridWorld:
    def __init__(self, width: int, height: int, rng: Optional[random.Random] = None, 
                cost_function: Optional[Callable[[int, int], float]] = None):
        self.width = width
        self.height = height
        self.cost_function = cost_function or (lambda prev, curr: value_to_cost(curr))
        
        # Load or generate world grid
        if os.path.exists(WORLD_PATH):
            self.grid = np.load(WORLD_PATH)
        else:
            seed = rng.randint(0, 999_999) if rng else None
            self.rng = np.random.default_rng(seed)
            self.grid = self.rng.integers(0, 10, size=(height, width), dtype=np.uint8)
            np.save(WORLD_PATH, self.grid)
            
        self.explored = np.zeros_like(self.grid, dtype=bool)
        self._restored_pairs = set()
        self.restore_pairs = set()

    def reset_cycle(self):
        """Reset cycle state."""
        self._restored_pairs.clear()
        self.explored.fill(False)

    def get_energy_cost(self, x: int, y: int) -> float:
        """Get the base energy cost for a cell."""
        return value_to_cost(int(self.grid[y, x]))

    def get_neighbors(self, x: int, y: int) -> List[Coord]:
        """Get valid neighboring cells."""
        offsets = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        return [(x + dx, y + dy) for dx, dy in offsets 
                if 0 <= x + dx < self.width and 0 <= y + dy < self.height]

    def transition(self, prev: Optional[Coord], curr: Coord) -> Tuple[float, bool]:
        """Execute a transition to a new cell, returning cost and restore flag."""
        cx, cy = curr
        curr_val = int(self.grid[cy, cx])
        
        if prev is None:
            cost = self.cost_function(0, curr_val)
            restore = False
        else:
            px, py = prev
            prev_val = int(self.grid[py, px])
            cost = self.cost_function(prev_val, curr_val)
            
            # Divisible pair restore rule
            if prev_val != 0 and curr_val % prev_val == 0:
                pair = (prev_val, curr_val)
                if pair not in self._restored_pairs:
                    restore = True
                    self.restore_pairs.add(pair)
                    self._restored_pairs.add(pair)
                else:
                    restore = False
            else:
                restore = False
        
        self.explored[cy, cx] = True
        return cost, restore

    def get_restored_cells(self):
        """Get all restored cell pairs."""
        return self.restore_pairs

    def get_neighbor_values(self, x: int, y: int):
        """Get values of neighboring cells."""
        return [int(self.grid[ny, nx]) for nx, ny in self.get_neighbors(x, y)]
    
    def get_cell_context(self, x: int, y: int, radius: int = 1) -> np.ndarray:
        """Get a subgrid of cells centered at (x,y) with given radius."""
        x_min = max(0, x - radius)
        y_min = max(0, y - radius)
        x_max = min(self.width - 1, x + radius)
        y_max = min(self.height - 1, y + radius)
        
        return self.grid[y_min:y_max+1, x_min:x_max+1].copy()
