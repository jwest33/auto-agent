from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PyQt5 import QtWidgets

from hopfield_memory import HopfieldMemory  # local import
from world import BASE_CELL_COST, GridWorld, WORLD_PATH


Coord = Tuple[int, int]
MEMORY_PATH = "save/memory.npy"
CYCLE_PATH = "save/cycles.npy"


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
class CycleSummary:
    reward: float
    cost: float
    surprise: float
    energy_left: float
    steps: List[Coord]
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
    """Stores per‑cell experiences and per‑cycle summaries with structure‑aware Hopfield keys."""

    # ─────────────────────────── ctor & basic IO ──────────────────────────── #
    def __init__(self, key_dim: int = 10, hop_capacity: int = 1024):
        self.plans: List[Plan] = []
        self.cell_records: Dict[str, List[CellExperience]] = {}
        self.cycles: List[CycleSummary] = []
        self.hopfield = HopfieldMemory(key_dim, value_dim=1, capacity=hop_capacity)

    # numpy serialization helpers unchanged …
    _cell_dtype = np.dtype([
        ("x", "i4"), ("y", "i4"), ("shade", "i4"),
        ("expected_cost", "f4"), ("actual_cost", "f4"), ("surprise", "f4"), ("timestamp", "f8")
    ])
    _cycle_dtype = np.dtype([
        ("reward", "f4"), ("cost", "f4"), ("surprise", "f4"),
        ("energy_left", "f4"), ("timestamp", "f8")
    ])

    def _experiences_to_array(self) -> np.ndarray:
        rows = []
        for key, exps in self.cell_records.items():
            x_str, y_str = key.split(",")
            x, y = int(x_str), int(y_str)
            for e in exps:
                rows.append((x, y, e.shade, e.expected_cost, e.actual_cost, e.surprise, e.timestamp))
        return np.array(rows, dtype=self._cell_dtype) if rows else np.empty((0,), dtype=self._cell_dtype)

    def _cycles_to_array(self) -> np.ndarray:
        rows = [(c.reward, c.cost, c.surprise, c.energy_left, c.timestamp) for c in self.cycles]
        return np.array(rows, dtype=self._cycle_dtype) if rows else np.empty((0,), dtype=self._cycle_dtype)

    def _array_to_experiences(self, arr: np.ndarray):
        self.cell_records.clear()
        for rec in arr:
            coord = (int(rec["x"]), int(rec["y"]))
            key = f"{coord[0]},{coord[1]}"
            exp = CellExperience(coord, int(rec["shade"]), float(rec["expected_cost"]), float(rec["actual_cost"]), float(rec["surprise"]), float(rec["timestamp"]))
            self.cell_records.setdefault(key, []).append(exp)

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

    # ───────────────────────── path‑structure helpers ──────────────────────── #
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

    # ───────────────────────── key encoders ──────────────────────────────── #
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

    # ───────────────────────── experience helpers ─────────────────────────── #
    def add_experience(self, exp: CellExperience):
        key = f"{exp.position[0]},{exp.position[1]}"
        self.cell_records.setdefault(key, []).append(exp)
        self.save_to_file()

    def get_expected_cost(self, coord: Coord) -> Optional[float]:
        recs = self.cell_records.get(f"{coord[0]},{coord[1]}")
        return sum(r.actual_cost for r in recs) / len(recs) if recs else None

    def get_expected_surprise(self, coord: Coord) -> Optional[float]:
        recs = self.cell_records.get(f"{coord[0]},{coord[1]}")
        return sum(r.surprise for r in recs) / len(recs) if recs else None

    # ───────────────────────── cycle helpers ─────────────────────────────── #
    def add_cycle(self, summary: CycleSummary, world: GridWorld):
        self.cycles.append(summary)
        self.hopfield.write(self._encode_cycle(summary, world), torch.tensor([summary.reward]))
        self.save_to_file()

    def estimate_reward_from_hopfield(self, steps: List[Coord], cost: float, surprise: float, energy_left: float, world: GridWorld) -> float:
        return self.hopfield(self._encode_partial(steps, cost, surprise, energy_left, world)).item()

    # ───────────────────────── plan persistence ──────────────────────────── #
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

    def _generate_plan(self, world: GridWorld) -> Plan:
        steps: List[Coord] = []
        costs: List[Tuple[float, float]] = []  # (cost, surprise)
        x, y = self.position
        while self.energy - sum(c for c, _ in costs) >= BASE_CELL_COST:
            neighbors = world.get_neighbors(x, y)
            if not neighbors:
                break
            nx, ny = random.choice(neighbors)
            model_cost = world.get_energy_cost(nx, ny)
            exp_cost = self.memory.get_expected_cost((nx, ny)) or model_cost
            exp_surp = self.memory.get_expected_surprise((nx, ny)) or 0.0
            if sum(c for c, _ in costs) + exp_cost > self.energy:
                break
            steps.append((nx, ny))
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
                # record experience
                self.memory.add_experience(CellExperience((x, y), int(world.grid[y, x]), exp_cost, actual_cost, surprise))
                # cycle accumulators
                self.record_step((x, y), cost, surprise)
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
