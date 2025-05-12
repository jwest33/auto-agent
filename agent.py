from __future__ import annotations

"""Agent module – simplified behavioral rewrite with rich step memory.

This version replaces the complex plan generation, storage, and retrieval with a 
rich step memory approach using directional vectors and surrounding cell information.
It maintains the goal-seeking behavior for unknown cells.
"""

import math
import os
import random
import time
import platform
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PyQt5 import QtWidgets

from hopfield_memory import HopfieldMemory
from world import BASE_CELL_COST, GridWorld

Coord = Tuple[int, int]

# Ensure OS-compatible paths
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "save")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
MEMORY_PATH = os.path.join(save_dir, "memory.npy")
CYCLE_PATH = os.path.join(save_dir, "cycles.npy")
ALPHA = 1.0  # reward-vs-cost weight
CELL_KEY_DIM = 16  # increased to store more context around cell
CELL_CAPACITY = 4096  # hopfield slots for cell memories

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
    energy_before: float
    energy_after: float
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

class Memory:
    """Rich step memory using directional vectors and surrounding cell information."""

    _cycle_dtype = np.dtype([
        ("reward", "f4"), ("cost", "f4"), ("surprise", "f4"),
        ("energy_left", "f4"), ("timestamp", "f8"),
    ])

    _step_dtype = np.dtype([
        ("x", "i4"), ("y", "i4"), ("shade", "i4"),
        ("expected_cost", "f4"), ("actual_cost", "f4"), ("surprise", "f4"),
        ("energy_before", "f4"), ("energy_after", "f4"),
        ("timestamp", "f8"),
    ])

    def __init__(self, key_dim: int = CELL_KEY_DIM, hop_capacity: int = CELL_CAPACITY):
        # Create save directory if it doesn't exist
        if not os.path.exists(os.path.dirname(MEMORY_PATH)):
            os.makedirs(os.path.dirname(MEMORY_PATH))
            
        # Store the key dimension for compatibility checks
        self.key_dim = key_dim
            
        # rich cell memory hopfield for cost prediction
        self.cell_hopfield = HopfieldMemory(key_dim, value_dim=1, capacity=hop_capacity)
        self.cycles: List[CycleSummary] = []
        self._raw_steps: List[StepExperience] = []
        self._cells: Dict[Coord, CellStats] = defaultdict(CellStats)

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

    def _array_to_experiences(self, arr: np.ndarray):
        self._raw_steps.clear()
        for rec in arr:
            # Handle backward compatibility with old format
            if 'energy_before' not in rec.dtype.names or 'energy_after' not in rec.dtype.names:
                # Old format - set default values
                energy_before = 100.0
                energy_after = 100.0 - float(rec["actual_cost"])
            else:
                # New format
                energy_before = float(rec["energy_before"])
                energy_after = float(rec["energy_after"])
                
            self._raw_steps.append(
                StepExperience(
                    position=(int(rec["x"]), int(rec["y"])),
                    shade=int(rec["shade"]),
                    expected_cost=float(rec["expected_cost"]),
                    actual_cost=float(rec["actual_cost"]),
                    surprise=float(rec["surprise"]),
                    energy_before=energy_before,
                    energy_after=energy_after,
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
        """Save memory data to files"""
        # Ensure save directory exists
        os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
        
        try:
            # Save data
            print(f"Saving memory data to {MEMORY_PATH}")
            experiences_array = self._experiences_to_array()
            print(f"Saving {len(experiences_array)} experiences")
            np.save(MEMORY_PATH, experiences_array)
            
            cycles_array = self._cycles_to_array()
            print(f"Saving {len(cycles_array)} cycles to {CYCLE_PATH}")
            np.save(CYCLE_PATH, cycles_array)
            
            print("Memory data saved successfully")
        except Exception as e:
            print(f"Error saving memory data: {e}")
            import traceback
            traceback.print_exc()

    def load_from_file(self):
        """Load memory data with backward compatibility"""
        if os.path.exists(MEMORY_PATH):
            try:
                self._array_to_experiences(np.load(MEMORY_PATH, allow_pickle=False))
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not load memory file with new format: {e}")
                # If loading fails, delete the old file - we'll create a new one
                os.remove(MEMORY_PATH)
                print("Removed incompatible memory file. Starting fresh.")
                
        if os.path.exists(CYCLE_PATH):
            try:
                self._array_to_cycles(np.load(CYCLE_PATH, allow_pickle=False))
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not load cycle file: {e}")
                # If loading fails, delete the old file
                os.remove(CYCLE_PATH)
                print("Removed incompatible cycle file. Starting fresh.")
                
        # Check if we need to reset due to dimension changes
        sample_key = self._encode_sample_key()
        if sample_key.shape[-1] != self.cell_hopfield.key_dim:
            print(f"Warning: Key dimension changed from {self.cell_hopfield.key_dim} to {sample_key.shape[-1]}")
            print("Resetting memory to avoid dimension mismatch errors.")
            if os.path.exists(MEMORY_PATH):
                os.remove(MEMORY_PATH)
            if os.path.exists(CYCLE_PATH):
                os.remove(CYCLE_PATH)
            self._raw_steps.clear()
            self.cycles.clear()
            self._cells.clear()
            # Create new Hopfield memory with correct dimensions
            self.cell_hopfield = HopfieldMemory(CELL_KEY_DIM, value_dim=1, capacity=CELL_CAPACITY)
            
    def _encode_sample_key(self):
        """Create a sample key to check dimensions"""
        origin = (0, 0)
        goal = (10, 10)
        position = (0, 0)
        cell = (1, 1)
        energy = 100.0
        return torch.zeros(CELL_KEY_DIM)

    # rich cell memory encoding

    def _encode_cell(self, world: GridWorld, origin: Coord, goal: Coord, 
                    position: Coord, cell: Coord, energy: float) -> torch.Tensor:
        """16-d feature vector: 
        - direction vectors (origin, goal, self)
        - current cell shade
        - surrounding cell shades (normalized)
        - current energy level (normalized)
        - distance to goal (normalized)
        """
        # Direction vectors
        dox, doy = norm_vec(origin[0] - cell[0], origin[1] - cell[1])
        dgx, dgy = norm_vec(goal[0] - cell[0], goal[1] - cell[1])
        dsx, dsy = norm_vec(cell[0] - position[0], cell[1] - position[1])
        
        # Cell shade and normalized energy
        shade = int(world.grid[cell[1], cell[0]])
        shade_norm = shade / 9.0
        energy_norm = energy / 100.0
        
        # Get surrounding cell values (if available)
        neighbors = world.get_neighbors(*position)
        neighbor_vals = [int(world.grid[ny, nx]) / 9.0 for nx, ny in neighbors]
        # Pad or truncate to exactly 4 values
        neighbor_vals = (neighbor_vals + [0.0] * 4)[:4]
        
        # Distance to goal (normalized by max possible distance)
        max_dist = math.hypot(world.width - 1, world.height - 1)
        dist_to_goal = euclidean(cell, goal) / max_dist
        
        # Combine all features
        return torch.tensor(
            [dox, doy, dgx, dgy, dsx, dsy, shade_norm, energy_norm, 
             dist_to_goal] + neighbor_vals,
            dtype=torch.float32
        )

    def store_cell_experience(self, world: GridWorld, origin: Coord, goal: Coord, 
                             position: Coord, cell: Coord, energy: float, cost: float):
        key = self._encode_cell(world, origin, goal, position, cell, energy)
        with torch.no_grad():
            self.cell_hopfield.write(key, torch.tensor([cost]))

    def estimate_cell_cost(self, world: GridWorld, origin: Coord, goal: Coord, 
                          position: Coord, cell: Coord, energy: float) -> Optional[float]:
        if self.cell_hopfield.item_count == 0:
            return None
        key = self._encode_cell(world, origin, goal, position, cell, energy)
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

    def add_cycle(self, summary: CycleSummary, world: GridWorld):
        """Add a cycle summary and save memory data to files."""
        self.cycles.append(summary)
        # Ensure we save after each cycle
        self.save_to_file()

class Agent:
    def __init__(self, origin: Coord, max_energy: float = 100.0):
        # Ensure save directory exists
        save_dir = os.path.dirname(MEMORY_PATH)
        if not os.path.exists(save_dir):
            print(f"Creating save directory: {save_dir}")
            os.makedirs(save_dir, exist_ok=True)
            
        self.origin = origin
        self.position: Coord = origin
        self._prev_pos: Optional[Coord] = None
        self.max_energy = max_energy
        self.energy = max_energy
        self.memory = Memory()
        
        print(f"Loading memory from {MEMORY_PATH}")
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
        """End the current cycle and save memory data."""
        print("Ending cycle...")
        
        # compute straight‑line distance from origin to *final* position
        dist = euclidean(self.origin, self.position)
        max_dist = math.hypot(world.width - 1, world.height - 1)
        norm_dist = dist / max_dist if max_dist else 0.0
        reward = norm_dist

        # Create cycle summary
        summary = CycleSummary(
            reward,
            self._cycle_cost,
            self._cycle_surprise,
            self.energy,
            self._cycle_steps.copy()
        )
        
        # Add to memory and explicitly save
        print(f"Adding cycle summary with {len(self._cycle_steps)} steps")
        self.memory.add_cycle(summary, world)
        
        # Reset agent state
        self.teleport_home()
        self.energy = self.max_energy
        self.start_cycle()
        
        print("Cycle completed and data saved")

    def teleport_home(self):
        self.position = self.origin

    def _estimate_cost(self, world: GridWorld, position: Coord, cell: Coord) -> float:
        """Estimate the cost of moving to a cell using rich memory."""
        goal = (world.width - 1, world.height - 1)
        est = self.memory.estimate_cell_cost(world, self.origin, goal, position, cell, self.energy)
        return est if est is not None else world.get_energy_cost(*cell)

    def _evaluate_move(self, world: GridWorld, position: Coord, cell: Coord, goal: Coord) -> float:
        """Score a potential move using rich context."""
        # Cost estimate from rich memory
        cost = self._estimate_cost(world, position, cell)
        
        # Expected surprise (if available)
        surprise = self.memory.get_expected_surprise(cell) or 0.0
        
        # Goal-seeking heuristic
        goal_dist = euclidean(cell, goal)
        goal_factor = 0.01 * goal_dist
        
        # Combine into a single score (lower is better)
        score = (cost + surprise) - ALPHA * (1.0 / (goal_dist + 1.0))
        
        return score

    def choose_next_move(self, world: GridWorld) -> Coord:
        """Choose the next move based on rich cell memory and goal-seeking behavior."""
        x, y = self.position
        neighbors = world.get_neighbors(x, y)
        goal = (world.width - 1, world.height - 1)
        
        # Evaluate all neighbors
        moves = []
        for nx, ny in neighbors:
            cell = (nx, ny)
            score = self._evaluate_move(world, self.position, cell, goal)
            moves.append((cell, score))
        
        # Sort by score (lower is better)
        moves.sort(key=lambda m: m[1])
        
        # Add some randomization for exploration
        # Choose from the top 3 moves with decreasing probability
        if len(moves) >= 3 and random.random() < 0.2:
            weights = [0.6, 0.3, 0.1]  # 60% best, 30% second best, 10% third best
            choices = [moves[i][0] for i in range(min(3, len(moves)))]
            return random.choices(choices, weights=weights[:len(choices)])[0]
        
        # Default to best move
        return moves[0][0] if moves else self.position

    def execute_step(self, world: GridWorld, repaint_cb=None):
        """Execute a single step based on the current best move."""
        try:
            if self.energy < BASE_CELL_COST:
                return False
            
            # Choose best next move
            next_pos = self.choose_next_move(world)
            x, y = next_pos
            
            # Record energy before the move
            energy_before = self.energy
            
            # Estimate cost
            exp_cost = self._estimate_cost(world, self.position, next_pos)
            
            # Execute the move
            cost, restore = world.transition(self._prev_pos, next_pos)
            actual_cost = 0.0 if restore else cost
            surprise = abs(exp_cost - actual_cost)
            self.energy = self.max_energy if restore else self.energy - cost
            
            # Record energy after the move
            energy_after = self.energy
            
            # Bookkeeping
            self.record_step(next_pos, cost, surprise)
            shade = int(world.grid[y, x])
            goal = (world.width - 1, world.height - 1)
            
            # Store experience in rich memory
            try:
                self.memory.store_cell_experience(
                    world, self.origin, goal, self.position, next_pos, energy_before, actual_cost
                )
            except AssertionError as e:
                print(f"Warning: Failed to store cell experience: {e}")
                # Continue without storing to prevent game from crashing
            
            # Update memory
            exp = StepExperience(
                position=next_pos, 
                shade=shade,
                expected_cost=exp_cost, 
                actual_cost=actual_cost, 
                surprise=surprise,
                energy_before=energy_before,
                energy_after=energy_after
            )
            self.memory.add_experience(exp)
            
            # Move
            self._prev_pos = self.position
            self.position = next_pos
            world.explored[y, x] = True
            
            if repaint_cb:
                repaint_cb()
                QtWidgets.QApplication.processEvents()
            
            return True
            
        except Exception as e:
            print(f"Error during step execution: {e}")
            # Instead of ending the cycle here, just return False
            return False

    def execute_plan(self, plan, world: GridWorld, repaint_cb=None):
        """Compatibility method to maintain interface with main.py
        Instead of executing a plan, we execute a series of single steps."""
        step_count = 0
        
        try:
            while self.energy >= BASE_CELL_COST:
                if not self.execute_step(world, repaint_cb):
                    break
                
                step_count += 1
    
                goal = (world.width - 1, world.height - 1)
                if self.position == goal:
                    print("Goal reached! Ending cycle.")
                    break
                    
            self.end_cycle(world)
                
        except Exception as e:
            print(f"Error in execute_plan: {e}")
            import traceback
            traceback.print_exc()
            # Even on error, we still end the cycle properly
            self.end_cycle(world)
                    
    def choose_plan(self, world: GridWorld):
        """Stub method to maintain interface with main.py.
        Returns a dummy plan object."""
        # We create a minimal placeholder plan to maintain compatibility
        class DummyPlan:
            def __init__(self):
                self.steps = []
                self.expected_energy_cost = 0.0
                self.expected_surprise = 0.0
                self.expected_reward = 0.0
                self._cached_costs_surprises = []
        
        return DummyPlan()
