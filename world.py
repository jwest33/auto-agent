from typing import Callable, List, Optional, Tuple
import os
import random

import numpy as np

MAX_CELL_COST = 5.0
BASE_CELL_COST = 1.0
Coord = Tuple[int, int]
WORLD_PATH = "save/world.npy"


def value_to_cost(value: int) -> float:
    return BASE_CELL_COST + (value / 9.0) * (MAX_CELL_COST - BASE_CELL_COST)


class GridWorld:
    def __init__(self, width: int, height: int, rng: Optional[random.Random] = None, cost_function: Optional[Callable[[int, int], float]] = None):
        self.width = width
        self.height = height
        self.cost_function = cost_function or (lambda prev, curr: value_to_cost(curr))
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
        self._restored_pairs.clear()

    def get_energy_cost(self, x: int, y: int) -> float:
        return value_to_cost(int(self.grid[y, x]))

    def get_neighbors(self, x: int, y: int) -> List[Coord]:
        offsets = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        return [(x + dx, y + dy) for dx, dy in offsets if 0 <= x + dx < self.width and 0 <= y + dy < self.height]

    def transition(self, prev: Optional[Coord], curr: Coord) -> Tuple[float, bool]:
        cx, cy = curr
        curr_val = int(self.grid[cy, cx])
        if prev is None:
            cost = self.cost_function(0, curr_val)
            restore = False
        else:
            px, py = prev
            prev_val = int(self.grid[py, px])
            cost = self.cost_function(prev_val, curr_val)
            # once‑per‑divisible‑pair restore rule
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
        return self.restore_pairs

    def get_neighbor_values(self, x: int, y: int):
        return [int(self.grid[ny, nx]) for nx, ny in self.get_neighbors(x, y)]
