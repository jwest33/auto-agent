from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math
import random
import time
import os
import numpy as np
from PyQt5 import QtWidgets

from world import GridWorld

# Constants & helpers

BASE_CELL_COST = 1.0
Coord = Tuple[int, int]
MEMORY_PATH = "save/memory.npy"

def euclidean(a: Coord, b: Coord) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

@dataclass
class CellExperience:
    position: Coord
    shade: int
    expected_cost: float
    actual_cost: float
    surprise: float
    timestamp: float = field(default_factory=time.time)

@dataclass
class Plan:
    steps: List[Coord]
    expected_energy_cost: float = 0.0
    expected_surprise: float = 0.0
    expected_reward: float = 0.0
    _cached_costs_surprises: List[Tuple[float, float]] = field(default_factory=list, init=False, repr=False)

    def recalc_metrics(self, origin: Coord, memory: 'Memory') -> None:
        # total cost & surprise as before
        self.expected_energy_cost = sum(c for c, _ in self._cached_costs_surprises)
        self.expected_surprise     = sum(s for _, s in self._cached_costs_surprises)

        # count how many of the planned steps are novel
        unseen = sum(
            1
            for coord in self.steps
            if memory.get_expected_cost(coord) is None
        )

        # reward = distance + bonus for new cells
        # if no steps, distance = 0 (origin→origin)
        last = self.steps[-1] if self.steps else origin
        self.expected_reward = euclidean(origin, last) + unseen

    def cache_costs_surprises(self, values: List[Tuple[float, float]]):
        self._cached_costs_surprises = values

class Memory:
    """Stores cell experiences and plans using a NumPy structured array on disk."""

    _dtype = np.dtype([
        ("x", "i4"),
        ("y", "i4"),
        ("shade", "i4"),
        ("expected_cost", "f4"),
        ("actual_cost", "f4"),
        ("surprise", "f4"),
        ("timestamp", "f8"),
    ])

    def __init__(self):
        self.plans: List[Plan] = []
        # Dict["x,y" -> List[CellExperience]] kept for in‑memory access speed
        self.cell_records: Dict[str, List[CellExperience]] = {}

    def _experiences_to_array(self) -> np.ndarray:
        """Flatten ``cell_records`` into a structured NumPy array."""
        rows = []
        for key, experiences in self.cell_records.items():
            x_str, y_str = key.split(",")
            x, y = int(x_str), int(y_str)
            for e in experiences:
                rows.append((
                    x, y, e.shade, e.expected_cost, e.actual_cost, e.surprise, e.timestamp
                ))
        if not rows:
            return np.empty((0,), dtype=self._dtype)
        return np.array(rows, dtype=self._dtype)

    def _array_to_experiences(self, arr: np.ndarray):
        """Populate ``cell_records`` from a structured array."""
        self.cell_records.clear()
        for rec in arr:
            coord = (int(rec["x"]), int(rec["y"]))
            key = f"{coord[0]},{coord[1]}"
            exp = CellExperience(
                position=coord,
                shade=int(rec["shade"]),
                expected_cost=float(rec["expected_cost"]),
                actual_cost=float(rec["actual_cost"]),
                surprise=float(rec["surprise"]),
                timestamp=float(rec["timestamp"]),
            )
            self.cell_records.setdefault(key, []).append(exp)

    def save_to_file(self):
        os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
        arr = self._experiences_to_array()
        np.save(MEMORY_PATH, arr)

    def load_from_file(self):
        if os.path.exists(MEMORY_PATH):
            arr: np.ndarray = np.load(MEMORY_PATH, allow_pickle=False)
            self._array_to_experiences(arr)

    # ────────────────────────────── API ────────────────────────────── #

    def add_experience(self, exp: CellExperience):
        key = f"{exp.position[0]},{exp.position[1]}"
        self.cell_records.setdefault(key, []).append(exp)
        self.save_to_file()

    def get_expected_cost(self, coord: Coord) -> Optional[float]:
        key = f"{coord[0]},{coord[1]}"
        recs = self.cell_records.get(key)
        return sum(r.actual_cost for r in recs) / len(recs) if recs else None

    def get_expected_surprise(self, coord: Coord) -> Optional[float]:
        key = f"{coord[0]},{coord[1]}"
        recs = self.cell_records.get(key)
        return sum(r.surprise for r in recs) / len(recs) if recs else None

    def store_plan(self, plan: Plan):
        self.plans.append(plan)

    def retrieve_candidate_plans(self, current_pos: Coord, energy: float) -> List[Plan]:
        return [p for p in self.plans if p.steps and p.steps[0] == current_pos and p.expected_energy_cost <= energy]

class Agent:
    def __init__(self, origin: Coord, max_energy: float = 100.0):
        self.origin = origin
        self.position = origin
        self._prev_pos: Optional[Coord] = None
        self.max_energy = max_energy
        self.energy = max_energy
        self.memory = Memory()
        self.memory.load_from_file()

    def teleport_home(self):
        self.position = self.origin

    def distance_from_origin(self) -> float:
        return euclidean(self.origin, self.position)

    def choose_plan(self, world: GridWorld) -> Plan:
        candidates = self.memory.retrieve_candidate_plans(self.position, self.energy)
        if candidates:
            return max(candidates, key=lambda p: p.expected_reward)
        return self._generate_plan(world)

    def _generate_plan(self, world: GridWorld) -> Plan:
        steps = []
        costs = []
        x, y = self.position
        for _ in range(int(self.energy // BASE_CELL_COST)):
            neighbors = world.get_neighbors(x, y)
            if not neighbors:
                break
            nx, ny = random.choice(neighbors)
            model = world.get_energy_cost(nx, ny)
            cost = self.memory.get_expected_cost((nx, ny)) or model
            surp = self.memory.get_expected_surprise((nx, ny)) or 0
            if sum(c for c, _ in costs) + cost > self.energy:
                break
            steps.append((nx, ny))
            costs.append((cost, surp))
            x, y = nx, ny
        plan = Plan(steps)
        plan.cache_costs_surprises(costs)
        plan.recalc_metrics(self.origin, self.memory)
        return plan

    def execute_plan(self, plan: Plan, world: GridWorld, repaint_cb=None):
        full_plan_steps = []
        full_costs_surprises = []

        while self.energy >= BASE_CELL_COST:
            if not plan.steps:
                break
            for (x, y), (exp_cost, _) in zip(plan.steps, plan._cached_costs_surprises):
                cost, restore = world.transition(self._prev_pos, (x, y))
                actual_cost = 0.0 if restore else cost
                surprise = abs(exp_cost - actual_cost)

                if restore:
                    self.energy = self.max_energy
                else:
                    self.energy -= cost

                # Record the experience
                exp = CellExperience(
                    position=(x, y),
                    shade=int(world.grid[y, x]),
                    expected_cost=exp_cost,
                    actual_cost=actual_cost,
                    surprise=surprise
                )
                self.memory.add_experience(exp)

                self._prev_pos = self.position
                self.position = (x, y)
                world.explored[y, x] = True

                if repaint_cb:
                    repaint_cb()
                    QtWidgets.QApplication.processEvents()

                if self.energy < BASE_CELL_COST:
                    break
                # Track executed steps
                full_plan_steps.append((x, y))
                full_costs_surprises.append((exp_cost, surprise))

            if self.energy >= BASE_CELL_COST:
                print(f'Energy remaining is {self.energy}, generating new plan')
                plan = self._generate_plan(world)

        # Save the full composite plan
        print(f'Completing final plan')
        final_plan = Plan(full_plan_steps)
        print(f'Caching cost surprises')
        final_plan.cache_costs_surprises(full_costs_surprises)
        print(f'Recalulating metrics')
        final_plan.recalc_metrics(self.origin, self.memory)
        print(f'Storing plan')
        self.memory.store_plan(final_plan)

        # Reset after depletion
        print('Teleporting home')
        self.teleport_home()
        self.energy = self.max_energy
