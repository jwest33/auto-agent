from typing import List, Tuple, Callable, Optional
import random
import os
import numpy as np

MAX_CELL_COST = 5.0
BASE_CELL_COST = 1.0
Coord = Tuple[int, int]
WORLD_PATH = "save/world.npy"


def value_to_cost(value: int) -> float:
    """
    Default mapping from a cell value (0-9) to an energy cost between BASE_CELL_COST and MAX_CELL_COST.
    """
    return BASE_CELL_COST + (value / 9.0) * (MAX_CELL_COST - BASE_CELL_COST)


class GridWorld:
    def __init__(
        self,
        width: int,
        height: int,
        rng: Optional[random.Random] = None,
        cost_function: Optional[Callable[[int, int], float]] = None
    ):
        self.width = width
        self.height = height
        self.cost_function = cost_function or (lambda prev, curr: value_to_cost(curr))

        if os.path.exists(WORLD_PATH):
            # Load existing world values
            self.grid = np.load(WORLD_PATH)
        else:
            # Generate a new grid with integer values from 0 to 9
            seed = rng.randint(0, 999999) if rng else None
            self.rng = np.random.default_rng(seed)
            self.grid = self.rng.integers(0, 10, size=(height, width), dtype=np.uint8)
            np.save(WORLD_PATH, self.grid)

        # Track which cells have been explored
        self.explored = np.zeros_like(self.grid, dtype=bool)
        # Track which divisible-value pairs have triggered a full restore in the current cycle
        self._restored_pairs = set()
        self.restore_pairs = set()

    def reset_cycle(self):
        """
        Clear the restore-tracking so that each cycle starts fresh.
        """
        self._restored_pairs.clear()

    def get_energy_cost(self, x: int, y: int) -> float:
        """
        Return the energy cost to enter cell (x, y) based on its grid value (0-9).
        """
        return value_to_cost(int(self.grid[y, x]))
    
    def get_neighbors(self, x: int, y: int) -> List[Coord]:
        """
        Return valid 4-directional neighbors of (x, y).
        """
        offsets = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        return [
            (x + dx, y + dy)
            for dx, dy in offsets
            if 0 <= x + dx < self.width and 0 <= y + dy < self.height
        ]

    def transition(
        self,
        prev: Optional[Coord],
        curr: Coord
    ) -> Tuple[float, bool]:
        """
        Compute the energy cost to move from prev to curr, and whether energy is fully restored.

        - prev: None for the initial cell (no movement cost beyond the cell's own cost).
        - Returns (cost, restore_flag).
          If restore_flag is True, the agent's energy should be reset to max on this move.
        """
        cx, cy = curr
        curr_val = int(self.grid[cy, cx])

        # Base cost based on custom or default cost function
        if prev is None:
            cost = self.cost_function(0, curr_val)
            restore = False
        else:
            px, py = prev
            prev_val = int(self.grid[py, px])
            cost = self.cost_function(prev_val, curr_val)

            # Check the divisible pair rule: restore once per unique pair per cycle
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

        # Mark explored
        self.explored[cy, cx] = True
        return cost, restore

    def get_restored_cells(self) -> set:
        return self.restore_pairs
