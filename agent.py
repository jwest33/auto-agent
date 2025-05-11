from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
from PyQt5 import QtWidgets

from hopfield_memory import HopfieldMemory  # local import
from world import BASE_CELL_COST, GridWorld, WORLD_PATH


Coord = Tuple[int, int]
MEMORY_PATH = "save/memory.npy"
CYCLE_PATH = "save/cycles.npy"
ALPHA = 1.0

def euclidean(a: Coord, b: Coord) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

@dataclass
class CellExperience:
    position: Coord
    shade: int
    expected_cost: float
    actual_cost: float
    surprise: float
    
@dataclass
class CycleSummary:
    reward: float
    cost: float
    surprise: float
    energy_left: float
    steps: List[Coord]
    timestamp: float = field(default_factory=time.time)
    
@dataclass
class CellStats:
    visits: int = 0
    exp_cost: float = 0.0
    exp_surprise: float = 0.0
    exp_reward: float = 0.0

@dataclass
class StepExperience:
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

    def recalc_metrics(self, origin: Coord, memory: "Memory") -> None:
        self.expected_energy_cost = sum(c for c, _ in self._cached_costs_surprises)
        self.expected_surprise = sum(s for _, s in self._cached_costs_surprises)
        unseen = sum(1 for coord in self.steps if memory.get_expected_cost(coord) is None)
        last = self.steps[-1] if self.steps else origin
        self.expected_reward = euclidean(origin, last) + unseen

    def cache_costs_surprises(self, values: List[Tuple[float, float]]):
        self._cached_costs_surprises = values


class Memory:
    """Stores per-cell experiences and per-cycle summaries with structure-aware Hopfield keys."""
    def __init__(self, key_dim: int = 10, hop_capacity: int = 1024):
        self.plans: List[Plan] = []
        self.cell_records: Dict[str, List[CellExperience]] = {}
        self.cycles: List[CycleSummary] = []
        self.hopfield = HopfieldMemory(key_dim, value_dim=1, capacity=hop_capacity)
        
        self._cells: Dict[Coord, CellStats] = defaultdict(CellStats)
        self._raw_steps: List[StepExperience] = []
        
    _cycle_dtype = np.dtype([
        ("reward", "f4"), ("cost", "f4"), ("surprise", "f4"),
        ("energy_left", "f4"), ("timestamp", "f8")
    ])
    
    _step_dtype = np.dtype([
        ("x",            "i4"),
        ("y",            "i4"),
        ("shade",        "i4"),
        ("expected_cost","f4"),
        ("actual_cost",  "f4"),
        ("surprise",     "f4"),
        ("timestamp",    "f8"),
    ])

    def _experiences_to_array(self) -> np.ndarray:
        rows = [
            (exp.position[0], exp.position[1], exp.shade,
             exp.expected_cost, exp.actual_cost,
             exp.surprise, exp.timestamp)
            for exp in self._raw_steps
        ]
        return (
            np.array(rows, dtype=self._step_dtype)
            if rows else np.empty((0,), dtype=self._step_dtype)
        )

    def _array_to_experiences(self, arr: np.ndarray) -> None:
        self._raw_steps.clear()
        for rec in arr:
            self._raw_steps.append(
                StepExperience(
                    position       =(int(rec["x"]), int(rec["y"])),
                    shade          = int(rec["shade"]),
                    expected_cost  = float(rec["expected_cost"]),
                    actual_cost    = float(rec["actual_cost"]),
                    surprise       = float(rec["surprise"]),
                    timestamp      = float(rec["timestamp"]),
                )
            )
    
    def add_step(self, coord: Coord, cost: float, surprise: float):
        c = self._cells[coord]
        c.visits += 1
        c.exp_cost     += (cost     - c.exp_cost)     / c.visits
        c.exp_surprise += (surprise - c.exp_surprise) / c.visits
        
    def add_experience(self, exp: StepExperience) -> None:
        self._raw_steps.append(exp)
        # also funnel the data into the running per‑cell stats
        self.add_step(exp.position, exp.actual_cost, exp.surprise)

    def add_cycle_reward(self, path: list[Coord], reward: float):
        for coord in set(path):
            c = self._cells[coord]
            n = c.visits
            c.exp_reward += (reward - c.exp_reward) / n

    def _cycles_to_array(self) -> np.ndarray:
        rows = [(c.reward, c.cost, c.surprise, c.energy_left, c.timestamp) for c in self.cycles]
        return np.array(rows, dtype=self._cycle_dtype) if rows else np.empty((0,), dtype=self._cycle_dtype)

    def _array_to_cycles(self, arr: np.ndarray):
        self.cycles.clear()
        for rec in arr:
            self.cycles.append(CycleSummary(float(rec["reward"]), float(rec["cost"]), float(rec["surprise"]), float(rec["energy_left"]), steps=[], timestamp=float(rec["timestamp"])))

    def save_to_file(self):
        os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
        np.save(MEMORY_PATH, self._experiences_to_array())
        np.save(CYCLE_PATH, self._cycles_to_array())

    def load_from_file(self):
        if os.path.exists(MEMORY_PATH):
            self._array_to_experiences(np.load(MEMORY_PATH, allow_pickle=False))
        if os.path.exists(CYCLE_PATH):
            self._array_to_cycles(np.load(CYCLE_PATH, allow_pickle=False))

    @staticmethod
    def _path_shape_features(steps: List[Coord]) -> Tuple[float, float]:
        if not steps:
            return 0.0, 0.0
        sx, sy = steps[0]
        lx, ly = steps[-1]
        net_disp = math.hypot(lx - sx, ly - sy)
        eff = net_disp / len(steps)
        return net_disp, eff

    @staticmethod
    def _neighbor_stats(world: GridWorld, coord: Coord) -> Tuple[float, float]:
        x, y = coord
        vals = world.get_neighbor_values(x, y)
        if not vals:
            return 0.0, 0.0
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        return mean, var

    @staticmethod
    def _encode_cycle(summary: "CycleSummary", world: GridWorld) -> torch.Tensor:
        steps = summary.steps
        net_disp, eff = Memory._path_shape_features(steps)
        last = steps[-1] if steps else (0, 0)
        neigh_mean, neigh_var = Memory._neighbor_stats(world, last)
        return torch.tensor([
            len(steps), net_disp, eff, neigh_mean, neigh_var, summary.cost, summary.surprise, summary.energy_left, summary.reward, 0.0
        ], dtype=torch.float32)

    @staticmethod
    def _encode_partial(steps: List[Coord], cost: float, surprise: float, energy_left: float, world: GridWorld) -> torch.Tensor:
        net_disp, eff = Memory._path_shape_features(steps)
        last = steps[-1] if steps else (0, 0)
        neigh_mean, neigh_var = Memory._neighbor_stats(world, last)
        return torch.tensor([
            len(steps), net_disp, eff, neigh_mean, neigh_var, cost, surprise, energy_left, 0.0, 0.0
        ], dtype=torch.float32)

    def get_expected_cost(self, coord: Coord) -> Optional[float]:
        return self._cells.get(coord).exp_cost if coord in self._cells else None

    def get_expected_surprise(self, coord: Coord) -> Optional[float]:
        return self._cells.get(coord).exp_surprise if coord in self._cells else None

    def get_expected_reward(self, coord: Coord) -> Optional[float]:
        return self._cells.get(coord).exp_reward if coord in self._cells else None

    def add_cycle(self, summary: CycleSummary, world: GridWorld):
        self.cycles.append(summary)
        self.hopfield.write(self._encode_cycle(summary, world), torch.tensor([summary.reward]))
        self.save_to_file()

    def estimate_reward_from_hopfield(self, steps: List[Coord], cost: float, surprise: float, energy_left: float, world: GridWorld) -> float:
        return self.hopfield(self._encode_partial(steps, cost, surprise, energy_left, world)).item()

    def store_plan(self, plan: Plan):
        self.plans.append(plan)

    def retrieve_candidate_plans(self, current_pos: Coord, energy: float) -> List[Plan]:
        return [p for p in self.plans if p.steps and p.steps[0] == current_pos and p.expected_energy_cost <= energy]

class Agent:
    def __init__(self, origin: Coord, max_energy: float = 100.0):
        self.origin = origin
        self.position: Coord = origin
        self._prev_pos: Optional[Coord] = None
        self.max_energy = max_energy
        self.energy = max_energy
        self.memory = Memory()
        self.memory.load_from_file()
        self.start_cycle()

    def start_cycle(self):
        self._cycle_steps: List[Coord] = []
        self._cycle_cost: float = 0.0
        self._cycle_surprise: float = 0.0

    def record_step(self, coord: Coord, cost: float, surprise: float):
        self._cycle_steps.append(coord)
        self._cycle_cost += cost
        self._cycle_surprise += surprise

    def end_cycle(self, world):
        reward = euclidean(self.origin, self.position) + len(self._cycle_steps)
        self.memory.add_cycle_reward(self._cycle_steps, reward)
        summary = CycleSummary(reward, self._cycle_cost, self._cycle_surprise, self.energy, self._cycle_steps.copy())
        self.memory.add_cycle(summary,world)
        # reset for next cycle
        self.teleport_home()
        self.energy = self.max_energy
        self.start_cycle()

    def teleport_home(self):
        self.position = self.origin

    def distance_from_origin(self) -> float:
        return euclidean(self.origin, self.position)

    def choose_plan(self, world: GridWorld) -> Plan:
        candidates = self.memory.retrieve_candidate_plans(self.position, self.energy)
        if candidates:
            return max(candidates, key=lambda p: p.expected_reward)
        return self._generate_plan(world)
        
    def _choose_neighbour(self, world, x, y, visited):
        neighbours = world.get_neighbors(x, y)
        best_score = float("inf")
        best       = []

        for nx, ny in neighbours:
            if (nx, ny) in visited:
                continue

            # Fall back to immediate model cost if we have no memory yet
            cost     = self.memory.get_expected_cost((nx, ny))
            if cost is None:
                cost = world.get_energy_cost(nx, ny)      # <‑‑ change
            surprise = self.memory.get_expected_surprise((nx, ny)) or 0.0
            reward   = self.memory.get_expected_reward((nx, ny))    or 0.0

            efe   = cost + surprise
            score = efe - ALPHA * reward

            if score < best_score - 1e-9:
                best_score, best = score, [(nx, ny)]
            elif math.isclose(score, best_score, rel_tol=1e-6):
                best.append((nx, ny))

        # If several ties remain, pick one at random
        return random.choice(best) if best else random.choice(neighbours)

    def _generate_plan(self, world: GridWorld) -> Plan:
        steps: List[Coord] = []
        costs: List[Tuple[float, float]] = []  # (cost, surprise)
        x, y = self.position
        visited = {(x, y)}
        while self.energy - sum(c for c, _ in costs) >= BASE_CELL_COST:
            neighbors = world.get_neighbors(x, y)
            if not neighbors:
                break
            nx, ny = self._choose_neighbour(world, x, y, visited)
            model_cost = world.get_energy_cost(nx, ny)
            exp_cost = self.memory.get_expected_cost((nx, ny)) or model_cost
            exp_surp = self.memory.get_expected_surprise((nx, ny)) or 0.0
            if sum(c for c, _ in costs) + exp_cost > self.energy:
                break
            steps.append((nx, ny))
            visited.add((nx, ny))
            costs.append((exp_cost, exp_surp))
            x, y = nx, ny
        plan = Plan(steps)
        plan.cache_costs_surprises(costs)
        plan.recalc_metrics(self.origin, self.memory)
        return plan

    def execute_plan(self, plan: Plan, world: GridWorld, repaint_cb=None):
        while self.energy >= BASE_CELL_COST and plan.steps:
            for (x, y), (exp_cost, _) in zip(plan.steps, plan._cached_costs_surprises):
                cost, restore = world.transition(self._prev_pos, (x, y))
                actual_cost = 0.0 if restore else cost
                surprise = abs(exp_cost - actual_cost)
                if restore:
                    self.energy = self.max_energy
                else:
                    self.energy -= cost
                # cycle accumulators
                self.record_step((x, y), cost, surprise)
                exp = StepExperience(
                    position      = (x, y),
                    shade         = int(world.grid[y, x]),
                    expected_cost = exp_cost,
                    actual_cost   = actual_cost,
                    surprise      = surprise,
                )
                self.memory.add_experience(exp)   # <- this now activates the GUI wrapper
                # move agent
                self._prev_pos = self.position
                self.position = (x, y)
                world.explored[y, x] = True
                # repaint if requested
                if repaint_cb:
                    repaint_cb()
                    QtWidgets.QApplication.processEvents()
                if self.energy < BASE_CELL_COST:
                    break
            # plan exhausted or depleted -> new plan
            if self.energy >= BASE_CELL_COST:
                plan = self._generate_plan(world)
        # cycle complete
        self.end_cycle(world)
