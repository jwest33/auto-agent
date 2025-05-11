from __future__ import annotations

"""Agent module – complete behavioural rewrite (2025‑05‑11).

This version *keeps* the public interface that **main.py** expects (notably
`Memory.add_experience`) while implementing the new memory/decision logic that
uses a Modern‑Hopfield network per the user specification.
"""

import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PyQt5 import QtWidgets

from hopfield_memory import HopfieldMemory
from world import BASE_CELL_COST, GridWorld

Coord = Tuple[int, int]
MEMORY_PATH = "save/memory.npy"
CYCLE_PATH = "save/cycles.npy"
ALPHA = 1.0 # reward‑vs‑cost weight
CELL_KEY_DIM = 10 # length of encoded cell vectors
CELL_CAPACITY = 4096 # hopfield slots for cell memories

def euclidean(a: Coord, b: Coord) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def norm_vec(dx: float, dy: float) -> Tuple[float, float]:
    mag = math.hypot(dx, dy)
    if mag == 0:
        return 0.0, 0.0
    return dx / mag, dy / mag
@dataclass
class StepExperience:
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
class CellStats:
    visits: int = 0
    exp_cost: float = 0.0
    exp_surprise: float = 0.0
    exp_reward: float = 0.0


@dataclass
class Plan:
    steps: List[Coord]
    expected_energy_cost: float = 0.0
    expected_surprise: float = 0.0
    expected_reward: float = 0.0
    backtrack: bool = False
    _cached_costs_surprises: List[Tuple[float, float]] = field(default_factory=list, init=False, repr=False)

    def cache_costs_surprises(self, vals: List[Tuple[float, float]]):
        self._cached_costs_surprises = vals

    def recalc_metrics(self, origin: Coord, memory: "Memory") -> None:
        self.expected_energy_cost = sum(c for c, _ in self._cached_costs_surprises)
        self.expected_surprise = sum(s for _, s in self._cached_costs_surprises)
        unseen = sum(1 for c in self.steps if memory.get_expected_cost(c) is None)
        last = self.steps[-1] if self.steps else origin
        self.expected_reward = euclidean(origin, last) + unseen
        if self.backtrack:
            self.expected_reward *= 0.8  # mildly de‑prioritise back‑tracks

class Memory:
    """Two‑tier memory (per‑cell & per‑cycle) plus planning store."""

    _cycle_dtype = np.dtype([
        ("reward", "f4"), ("cost", "f4"), ("surprise", "f4"),
        ("energy_left", "f4"), ("timestamp", "f8"),
    ])

    _step_dtype = np.dtype([
        ("x", "i4"), ("y", "i4"), ("shade", "i4"),
        ("expected_cost", "f4"), ("actual_cost", "f4"), ("surprise", "f4"),
        ("timestamp", "f8"),
    ])

    def __init__(self, key_dim: int = 10, hop_capacity: int = 1024):
        # cycle‑level hopfield for reward prediction (legacy)
        self.cycle_hopfield = HopfieldMemory(key_dim, value_dim=1, capacity=hop_capacity)
        # per‑cell hopfield for cost prediction
        self.cell_hopfield = HopfieldMemory(CELL_KEY_DIM, value_dim=1, capacity=CELL_CAPACITY)

        self.plans: List[Plan] = []
        self.cycles: List[CycleSummary] = []
        self._raw_steps: List[StepExperience] = []
        self._cells: Dict[Coord, CellStats] = defaultdict(CellStats)

    def _experiences_to_array(self) -> np.ndarray:
        rows = [
            (
                e.position[0], e.position[1], e.shade,
                e.expected_cost, e.actual_cost, e.surprise, e.timestamp,
            )
            for e in self._raw_steps
        ]
        return np.array(rows, dtype=self._step_dtype) if rows else np.empty((0,), self._step_dtype)

    def _array_to_experiences(self, arr: np.ndarray):
        self._raw_steps.clear()
        for rec in arr:
            self._raw_steps.append(
                StepExperience(
                    position=(int(rec["x"]), int(rec["y"])),
                    shade=int(rec["shade"]),
                    expected_cost=float(rec["expected_cost"]),
                    actual_cost=float(rec["actual_cost"]),
                    surprise=float(rec["surprise"]),
                    timestamp=float(rec["timestamp"]),
                )
            )

    def _cycles_to_array(self) -> np.ndarray:
        rows = [(c.reward, c.cost, c.surprise, c.energy_left, c.timestamp) for c in self.cycles]
        return np.array(rows, dtype=self._cycle_dtype) if rows else np.empty((0,), self._cycle_dtype)

    def _array_to_cycles(self, arr: np.ndarray):
        self.cycles.clear()
        for rec in arr:
            self.cycles.append(
                CycleSummary(
                    float(rec["reward"]), float(rec["cost"]), float(rec["surprise"]),
                    float(rec["energy_left"]), steps=[], timestamp=float(rec["timestamp"])
                )
            )

    def save_to_file(self):
        os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
        np.save(MEMORY_PATH, self._experiences_to_array())
        np.save(CYCLE_PATH, self._cycles_to_array())

    def load_from_file(self):
        if os.path.exists(MEMORY_PATH):
            self._array_to_experiences(np.load(MEMORY_PATH, allow_pickle=False))
        if os.path.exists(CYCLE_PATH):
            self._array_to_cycles(np.load(CYCLE_PATH, allow_pickle=False))

    # per‑cell hopfield helpers

    @staticmethod
    def _encode_cell(origin: Coord, goal: Coord, self_pos: Coord, cell: Coord, shade: int) -> torch.Tensor:
        """10‑d feature vector: 3 direction vectors (origin, goal, self) + shade."""
        dox, doy = norm_vec(origin[0] - cell[0], origin[1] - cell[1])
        dgx, dgy = norm_vec(goal[0] - cell[0], goal[1] - cell[1])
        dsx, dsy = norm_vec(cell[0] - self_pos[0], cell[1] - self_pos[1])
        shade_norm = shade / 9.0
        return torch.tensor([dox, doy, dgx, dgy, dsx, dsy, shade_norm, 0.0, 0.0, 0.0], dtype=torch.float32)

    def store_cell_experience(self, origin: Coord, goal: Coord, self_pos: Coord, cell: Coord, shade: int, cost: float):
        key = self._encode_cell(origin, goal, self_pos, cell, shade)
        with torch.no_grad():
            self.cell_hopfield.write(key, torch.tensor([cost]))

    def estimate_cell_cost(self, origin: Coord, goal: Coord, self_pos: Coord, cell: Coord, shade: int) -> Optional[float]:
        if self.cell_hopfield.item_count == 0:
            return None
        key = self._encode_cell(origin, goal, self_pos, cell, shade)
        return float(self.cell_hopfield(key).item())

    def add_step_stats(self, coord: Coord, cost: float, surprise: float):
        cs = self._cells[coord]
        cs.visits += 1
        cs.exp_cost += (cost - cs.exp_cost) / cs.visits
        cs.exp_surprise += (surprise - cs.exp_surprise) / cs.visits

    def add_experience(self, exp: StepExperience):
        """Record a step experience and update running stats."""
        self._raw_steps.append(exp)
        self.add_step_stats(exp.position, exp.actual_cost, exp.surprise)
        
    def get_expected_cost(self, coord: Coord) -> Optional[float]:
        return self._cells.get(coord).exp_cost if coord in self._cells else None

    def get_expected_surprise(self, coord: Coord) -> Optional[float]:
        return self._cells.get(coord).exp_surprise if coord in self._cells else None

    def get_expected_reward(self, coord: Coord) -> Optional[float]:
        return self._cells.get(coord).exp_reward if coord in self._cells else None

    def add_cycle(self, summary: CycleSummary, world: GridWorld):
        self.cycles.append(summary)
        self.save_to_file()

    def store_plan(self, plan: Plan):
        self.plans.append(plan)

    def retrieve_candidate_plans(self, current: Coord, energy: float) -> List[Plan]:
        return [p for p in self.plans if p.steps and p.steps[0] == current and p.expected_energy_cost <= energy]

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
        self._cycle_cost = 0.0
        self._cycle_surprise = 0.0
        self._prev_pos = None

    def record_step(self, coord: Coord, cost: float, surprise: float):
        self._cycle_steps.append(coord)
        self._cycle_cost += cost
        self._cycle_surprise += surprise

    def end_cycle(self, world: GridWorld):
        reward = euclidean(self.origin, self.position) + len(self._cycle_steps)
        summary = CycleSummary(reward, self._cycle_cost, self._cycle_surprise, self.energy, self._cycle_steps.copy())
        self.memory.add_cycle(summary, world)
        self.teleport_home()
        self.energy = self.max_energy
        self.start_cycle()

    def teleport_home(self):
        self.position = self.origin

    def _estimate_cost(self, world: GridWorld, self_pos: Coord, cell: Coord) -> float:
        shade = int(world.grid[cell[1], cell[0]])
        goal = (world.width - 1, world.height - 1)
        est = self.memory.estimate_cell_cost(self.origin, goal, self_pos, cell, shade)
        return est if est is not None else world.get_energy_cost(*cell)

    def _choose_neighbour(self, world: GridWorld, x: int, y: int, visited: set[Coord]) -> Tuple[Coord, List[Coord]]:
        neighbours = world.get_neighbors(x, y)
        goal = (world.width - 1, world.height - 1)
        best_score = float("inf")
        best: List[Coord] = []
        alts: List[Coord] = []
        for nx, ny in neighbours:
            cell = (nx, ny)
            if cell in visited:
                continue
            cost = self._estimate_cost(world, (x, y), cell)
            surprise = self.memory.get_expected_surprise(cell) or 0.0
            reward = self.memory.get_expected_reward(cell) or 0.0
            score = (cost + surprise) - ALPHA * reward - 0.01 * euclidean(cell, goal)
            if score < best_score - 1e-9:
                if best:
                    alts.extend(best)
                best_score = score
                best = [cell]
            elif math.isclose(score, best_score, rel_tol=1e-6):
                best.append(cell)
            else:
                alts.append(cell)
        chosen = random.choice(best) if best else random.choice(neighbours)
        alts = [c for c in alts if c != chosen]
        return chosen, alts

    def _generate_plan(self, world: GridWorld) -> Plan:
        steps: List[Coord] = []
        cached: List[Tuple[float, float]] = []
        visited = {self.position}
        x, y = self.position
        backtrack_opts: List[Tuple[Coord, float]] = []
        while self.energy - sum(c for c, _ in cached) >= BASE_CELL_COST:
            chosen, alts = self._choose_neighbour(world, x, y, visited)
            for alt in alts:
                backtrack_opts.append((alt, self._estimate_cost(world, (x, y), alt)))
            exp_cost = self._estimate_cost(world, (x, y), chosen)
            exp_surprise = self.memory.get_expected_surprise(chosen) or 0.0
            if sum(c for c, _ in cached) + exp_cost > self.energy:
                break
            steps.append(chosen)
            cached.append((exp_cost, exp_surprise))
            visited.add(chosen)
            x, y = chosen
        plan = Plan(steps)
        plan.cache_costs_surprises(cached)
        plan.recalc_metrics(self.origin, self.memory)
        # spawn backtrack probes
        for cell, est in backtrack_opts:
            bt = Plan([cell], backtrack=True)
            bt.cache_costs_surprises([(est, 0.0)])
            bt.recalc_metrics(self.origin, self.memory)
            self.memory.store_plan(bt)
        return plan

    def choose_plan(self, world: GridWorld) -> Plan:
        plans = self.memory.retrieve_candidate_plans(self.position, self.energy)
        return max(plans, key=lambda p: p.expected_reward) if plans else self._generate_plan(world)

    def execute_plan(self, plan: Plan, world: GridWorld, repaint_cb=None):
        """Walk the plan, consuming its steps as we go so we never replay it."""
        goal = (world.width - 1, world.height - 1)

        while self.energy >= BASE_CELL_COST:
            if not plan.steps:
                plan = self.choose_plan(world)
                if not plan.steps:
                    break

            x, y = plan.steps.pop(0)
            exp_cost, _ = plan._cached_costs_surprises.pop(0)

            # world transition
            cost, restore = world.transition(self._prev_pos, (x, y))
            actual_cost = 0.0 if restore else cost
            surprise = abs(exp_cost - actual_cost)
            self.energy = self.max_energy if restore else self.energy - cost

            # bookkeeping
            self.record_step((x, y), cost, surprise)
            shade = int(world.grid[y, x])
            self.memory.store_cell_experience(
                self.origin, goal, self.position, (x, y), shade, actual_cost
            )
            exp = StepExperience(
                position=(x, y), shade=shade,
                expected_cost=exp_cost, actual_cost=actual_cost, surprise=surprise,
            )
            self.memory.add_experience(exp)

            # move
            self._prev_pos = self.position
            self.position = (x, y)
            world.explored[y, x] = True

            if repaint_cb:
                repaint_cb(); QtWidgets.QApplication.processEvents()

            if self.energy < BASE_CELL_COST:
                break

        self.end_cycle(world)
