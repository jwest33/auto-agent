from __future__ import annotations
import argparse
import json
import os
import random
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

# Typing helpers & colours

Coord = Tuple[int, int]
RGB = Tuple[int, int, int]

COLOR_KEY = (255, 215, 0)
COLOR_DOOR_LOCKED = (200, 40, 40)
COLOR_DOOR_OPEN = (40, 200, 40)
COLOR_SWITCH_ON = (70, 130, 255)
COLOR_SWITCH_OFF = (25, 25, 112)
COLOR_WALL = (0, 0, 0)
COLOR_START = (173, 255, 47)
COLOR_GOAL = (255, 105, 180)

# World objects

class WorldObject:
    def __init__(self, oid: str):
        self.oid = oid

    @property
    def meta(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["cls"] = type(self).__name__
        return d

    @classmethod
    def from_meta(cls, meta: Dict[str, Any]) -> "WorldObject":
        cls_name = meta.pop("cls")
        cls_map = {"Key": Key, "Door": Door, "Switch": Switch}
        return cls_map[cls_name](**meta)
class Key(WorldObject):
    def __init__(self, oid: str, ktype: str):
        super().__init__(oid)
        self.ktype = ktype
class Door(WorldObject):
    def __init__(self, oid: str, dtype: str, is_locked: bool = True):
        super().__init__(oid)
        self.dtype = dtype
        self.is_locked = is_locked

    def try_open(self, agent) -> bool:
        if self.is_locked and agent and (self.dtype in getattr(agent, "inventory", [])):
            self.is_locked = False
        return not self.is_locked
class Switch(WorldObject):
    def __init__(self, oid: str, target_ids: List[str]):
        super().__init__(oid)
        self.target_ids = target_ids
        self.state = False

    def toggle(self, world: "SymbolicGridWorld") -> None:
        self.state = not self.state
        for tid in self.target_ids:
            obj = world.objects.get(tid)
            if isinstance(obj, Door):
                obj.is_locked = not obj.is_locked

# SymbolicGridWorld

class SymbolicGridWorld:
    """Gridworld container with fixed start/goal anchors."""

    def __init__(self, width: int, height: int, rng: Optional[Any] = None, save_dir: str = "worlds"):
        self.width, self.height = width, height
        # Each world gets its *own* RNG (so phases vary even with a shared generator).
        if rng is None:
            rng = np.random.default_rng()
        elif isinstance(rng, (int, np.integer)):  # modified line
            rng = np.random.default_rng(rng)
        elif isinstance(rng, random.Random):
            rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
        self.rng = rng  # numpy Generator

        self.wall_mask = np.zeros((height, width), dtype=bool)
        self.object_grid = np.full((height, width), None, dtype=object)
        self.explored_mask = np.zeros((height, width), dtype=bool)
        self.objects: Dict[str, WorldObject] = {}

        self.start: Coord = (0, 0)
        self.goal: Coord = (width - 1, height - 1)

        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

    #------ placement------
    def place_wall(self, x: int, y: int):
        if (x, y) not in (self.start, self.goal):
            self.wall_mask[y, x] = True

    def _place_object(self, x: int, y: int, obj: WorldObject):
        if (x, y) in (self.start, self.goal):
            raise ValueError("Cannot place objects on start/goal tiles.")
        self.object_grid[y, x] = obj
        self.objects[obj.oid] = obj

    def place_key(self, x: int, y: int, oid: str, ktype: str):
        self._place_object(x, y, Key(oid, ktype))

    def place_door(self, x: int, y: int, oid: str, dtype: str):
        self._place_object(x, y, Door(oid, dtype))

    def place_switch(self, x: int, y: int, oid: str, targets: List[str]):
        self._place_object(x, y, Switch(oid, targets))

    #------ queries------
    def is_passable(self, x: int, y: int, agent=None) -> bool:
        if self.wall_mask[y, x]:
            return False
        obj = self.object_grid[y, x]
        if isinstance(obj, Door):
            return obj.try_open(agent)
        return True
    def neighbors(self, x: int, y: int) -> List[Coord]:
        offs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        return [
            (x + dx, y + dy)
            for dx, dy in offs
            if 0 <= x + dx < self.width and 0 <= y + dy < self.height
        ]

    #------ persistence------
    def _object_id_matrix(self) -> np.ndarray:
        mat = np.full((self.height, self.width), "", dtype=object)
        for y in range(self.height):
            for x in range(self.width):
                if (obj := self.object_grid[y, x]) is not None:
                    mat[y, x] = obj.oid
        return mat

    def save(self, name: str):
        np.savez_compressed(
            os.path.join(self.save_dir, f"{name}.npz"),
            wall=self.wall_mask,
            explored=self.explored_mask,
            obj_ids=self._object_id_matrix(),
            start=self.start,
            goal=self.goal,
        )
        meta = {oid: obj.meta for oid, obj in self.objects.items()}
        with open(os.path.join(self.save_dir, f"{name}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, name: str, save_dir: str = "worlds") -> "SymbolicGridWorld":
        npz = np.load(os.path.join(save_dir, f"{name}.npz"), allow_pickle=True)
        with open(os.path.join(save_dir, f"{name}_meta.json")) as f:
            meta = json.load(f)
        h, w = npz["wall"].shape
        world = cls(w, h, rng=None, save_dir=save_dir)
        world.wall_mask = npz["wall"]
        world.explored_mask = npz["explored"]
        world.start = tuple(npz["start"])  # type: ignore
        world.goal = tuple(npz["goal"])  # type: ignore
        id_mat = npz["obj_ids"]
        for y in range(h):
            for x in range(w):
                if (oid := id_mat[y, x]):
                    obj = WorldObject.from_meta(meta[oid])
                    world.object_grid[y, x] = obj
                    world.objects[oid] = obj
        return world
    
    #------ visualisation------
    def to_rgb(self) -> np.ndarray:
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = 200
        img[self.wall_mask] = COLOR_WALL
        sx, sy = self.start
        gx, gy = self.goal
        img[sy, sx] = COLOR_START
        img[gy, gx] = COLOR_GOAL
        for y in range(self.height):
            for x in range(self.width):
                obj = self.object_grid[y, x]
                if obj is None:
                    continue
                if isinstance(obj, Key):
                    img[y, x] = COLOR_KEY
                elif isinstance(obj, Door):
                    img[y, x] = COLOR_DOOR_OPEN if not obj.is_locked else COLOR_DOOR_LOCKED
                elif isinstance(obj, Switch):
                    img[y, x] = COLOR_SWITCH_ON if obj.state else COLOR_SWITCH_OFF
        return img

    def show(self, scale: int = 40, block: bool = False):
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError("pip install matplotlib") from e
        img = self.to_rgb().repeat(scale, 0).repeat(scale, 1)
        plt.figure(figsize=(self.width, self.height))
        plt.imshow(img, interpolation="nearest")
        plt.axis("off")
        plt.show(block=block)

# Level Generator

class LevelGenerator:
    """Creates a 5‑phase curriculum of maze worlds."""

    def __init__(self, width: int, height: int, save_dir: str = "worlds", seed: Optional[int] = None):
        self.width, self.height, self.save_dir = width, height, save_dir
        self.rng = np.random.default_rng(seed)

    # internal helpers
    def _random_passable(self, world: SymbolicGridWorld) -> Coord:
        while True:
            x = int(world.rng.integers(0, self.width))
            y = int(world.rng.integers(0, self.height))
            if not world.wall_mask[y, x] and (x, y) not in (world.start, world.goal):
                return x, y

    def _connect_anchor(self, world: SymbolicGridWorld, anchor: Coord):
        x, y = anchor
        # Search outward in a spiral until we find a passable cell
        from collections import deque
        visited = set()
        queue = deque([(x, y)])
        while queue:
            cx, cy = queue.popleft()
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))

            if (cx, cy) != (x, y) and not world.wall_mask[cy, cx]:
                # Carve a straight line back to the anchor
                path = self._straight_path(x, y, cx, cy)
                for px, py in path:
                    world.wall_mask[py, px] = False
                return

            for nx, ny in world.neighbors(cx, cy):
                if 0 <= nx < world.width and 0 <= ny < world.height:
                    queue.append((nx, ny))
                    
    def _straight_path(self, x1, y1, x2, y2):
        """Return all grid cells along a Manhattan L-shaped path from (x1, y1) to (x2, y2)."""
        path = []
        # First move horizontally, then vertically
        for x in range(min(x1, x2), max(x1, x2) + 1):
            path.append((x, y1))
        for y in range(min(y1, y2), max(y1, y2) + 1):
            path.append((x2, y))
        return path

    def _generate_maze(self, loop_pct: float = 0.0) -> SymbolicGridWorld:
        w, h = self.width, self.height
        world = SymbolicGridWorld(w, h, rng=self.rng.integers(0, 2**32 - 1), save_dir=self.save_dir)

        # Start with all walls
        world.wall_mask[:, :] = True

        # Cells at odd coordinates (actual maze cells)
        cells = [(x, y) for y in range(1, h, 2) for x in range(1, w, 2)]
        if not cells:
            return world

        stack = [cells[world.rng.integers(0, len(cells))]]
        visited = {stack[0]}
        while stack:
            cx, cy = stack[-1]
            # Shuffle directions for more variety
            dirs = world.rng.permutation([(2, 0), (-2, 0), (0, 2), (0, -2)])
            neighbors = [
                (cx + dx, cy + dy)
                for dx, dy in dirs
                if 1 <= cx + dx < w and 1 <= cy + dy < h and (cx + dx, cy + dy) not in visited
            ]
            if neighbors:
                nx, ny = neighbors[0]  # take first after shuffle
                wx, wy = (cx + nx) // 2, (cy + ny) // 2
                world.wall_mask[cy, cx] = False
                world.wall_mask[wy, wx] = False
                world.wall_mask[ny, nx] = False
                stack.append((nx, ny))
                visited.add((nx, ny))
            else:
                stack.pop()

        # Ensure anchors clear and connected
        world.wall_mask[world.start[1], world.start[0]] = False
        world.wall_mask[world.goal[1], world.goal[0]] = False
        self._connect_anchor(world, world.start)
        self._connect_anchor(world, world.goal)

        # Introduce loops
        if loop_pct > 0:
            wall_coords = np.argwhere(world.wall_mask)
            remove_n = int(len(wall_coords) * loop_pct)
            if remove_n:
                idx = world.rng.choice(len(wall_coords), remove_n, replace=False)
                for iy in idx:
                    y, x = wall_coords[iy]
                    world.wall_mask[y, x] = False
        return world

    # phases
    def phase1(self):
        return self._generate_maze(loop_pct=0.0)

    def phase2(self):
        return self._generate_maze(loop_pct=0.05)

    def phase3(self):
        return self._generate_maze(loop_pct=0.15)

    def phase4(self):
        w = self._generate_maze(loop_pct=0.15)
        key_id, door_id, switch_id = "K1", "D1", "S1"
        kx, ky = self._random_passable(w)
        w.place_key(kx, ky, key_id, "A")
        dx, dy = self._random_passable(w)
        w.place_door(dx, dy, door_id, "A")
        sx, sy = self._random_passable(w)
        w.place_switch(sx, sy, switch_id, [door_id])
        return w

    def phase5(self):
        w = self._generate_maze(loop_pct=0.30)
        for i in range(1, 4):
            key_id, door_id, switch_id = f"K{i}", f"D{i}", f"S{i}"
            kx, ky = self._random_passable(w)
            w.place_key(kx, ky, key_id, str(i))
            dx, dy = self._random_passable(w)
            w.place_door(dx, dy, door_id, str(i))
            sx, sy = self._random_passable(w)
            w.place_switch(sx, sy, switch_id, [door_id])
        return w

    # public
    def generate(self, phase: int) -> SymbolicGridWorld:
        if not 1 <= phase <= 5:
            raise ValueError("phase must be 1‑5")
        return getattr(self, f"phase{phase}")()

class CurriculumRunner:
    def __init__(self, generator: LevelGenerator):
        self.generator = generator

    def iter_phases(self) -> Iterator[Tuple[int, SymbolicGridWorld]]:
        for phase in range(1, 6):
            world = self.generator.generate(phase)
            world.save(f"phase_{phase}")
            yield phase, world


def main():
    parser = argparse.ArgumentParser(description="Maze‑based symbolic gridworld generator")
    parser.add_argument("--preview", action="store_true", help="visualise the curriculum")
    parser.add_argument("--generate", action="store_true", help="save worlds to disk")
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None, help="optional global seed for reproducibility")
    args = parser.parse_args()

    gen = LevelGenerator(args.width, args.height, seed=args.seed)
    runner = CurriculumRunner(gen)

    if args.generate:
        for phase, _ in runner.iter_phases():
            print(f"Saved phase {phase}")

    if args.preview:
        for phase, world in runner.iter_phases():
            print(f"Previewing phase {phase}")
            world.show(block=True)


if __name__ == "__main__":
    main()
# python world/module_symbolic_world.py --preview --width 15 --height 15 --seed 123
