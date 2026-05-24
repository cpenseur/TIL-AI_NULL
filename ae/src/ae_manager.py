
"""AE manager: tactical search-rush controller with TorchScript fallback."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

FORWARD = 0
BACKWARD = 1
LEFT = 2
RIGHT = 3
STAY = 4
PLACE_BOMB = 5

# World direction convention from AE docs/logs: 0=RIGHT, 1=DOWN, 2=LEFT, 3=UP.
DIR_TO_DELTA = {
    0: (1, 0),
    1: (0, 1),
    2: (-1, 0),
    3: (0, -1),
}

# Local view action deltas: y+ is forward, x+ is right.
ACTION_TO_LOCAL_DELTA = {
    FORWARD: (1, 0),
    BACKWARD: (-1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
    STAY: (0, 0),
}


class AEManager:
    def __init__(self):
        self.grid_size = 16
        self.last_step = -1

        self.recent_actions: deque[int] = deque(maxlen=30)
        self.recent_locations: deque[tuple[int, int]] = deque(maxlen=24)
        self.visited: dict[tuple[int, int], int] = {}
        self.targets: dict[tuple[int, int], tuple[str, float, int]] = {}

        self.escape_ticks = 0
        self.bomb_cooldown = 0
        self.last_bomb_step = -999
        self.loop_break_turn = 0
        self.policy = None
        self.policy_hidden = None
        self.policy_checked = False

    def reset(self) -> None:
        self.last_step = -1
        self.recent_actions.clear()
        self.recent_locations.clear()
        self.visited.clear()
        self.targets.clear()
        self.escape_ticks = 0
        self.bomb_cooldown = 0
        self.last_bomb_step = -999
        self.loop_break_turn = 0
        self.policy_hidden = None

    def ae(self, observation: dict[str, Any]) -> int:
        # Robustness: some wrappers pass {"instances":[{"observation": obs}]}.
        if "instances" in observation:
            observation = observation["instances"][0]["observation"]

        step = int(observation.get("step", 0))
        if step == 0 or step < self.last_step:
            self.reset()
        self.last_step = step

        mask = np.asarray(observation.get("action_mask", [1, 1, 1, 1, 1, 0]), dtype=np.float32).ravel()
        if mask.size < 6:
            fixed = np.zeros(6, dtype=np.float32)
            fixed[: mask.size] = mask
            mask = fixed
        if mask.sum() <= 0:
            return STAY

        loc = self._location(observation)
        if loc is not None:
            self.recent_locations.append(loc)
            self.visited[loc] = self.visited.get(loc, 0) + 1

        if self.bomb_cooldown > 0:
            self.bomb_cooldown -= 1
        if self.escape_ticks > 0:
            self.escape_ticks -= 1
        if self.loop_break_turn > 0:
            self.loop_break_turn -= 1

        view = np.asarray(observation.get("agent_viewcone", np.zeros((7, 5, 25))), dtype=np.float32)
        self._update_memory(observation, view)
        self._update_base_memory(observation)

        danger = self._danger_map(view)
        origin = self._origin(view)

        # 1) Survival first: if in blast/danger or just bombed, find a safe cell by BFS.
        if self._is_dangerous_cell(danger, origin) or self.escape_ticks > 0 or self._local_danger_score(view) > 0.10:
            action = self._bfs_escape_action(view, danger, mask)
            return self._record(self._anti_loop_adjust(action, mask))

        # 2) Place useful bomb only when escape path exists.
        if self._should_bomb(observation, view, mask):
            self.escape_ticks = 5
            self.bomb_cooldown = 3
            self.last_bomb_step = step
            return self._record(PLACE_BOMB)

        # 3) If a tactical local lookahead has a strong move, use it.
        action = self._beam_search_action(view, mask, depth=4, width=8)
        if action is not None:
            return self._record(self._anti_loop_adjust(action, mask))

        # 4) Pathfind to visible targets instead of straight-line walking into obstacles.
        action = self._visible_bfs_target_action(view, mask)
        if action is not None:
            return self._record(self._anti_loop_adjust(action, mask))

        # 5) Move toward remembered high-value target.
        action = self._remembered_target_action(observation, mask)
        if action is not None:
            return self._record(self._anti_loop_adjust(action, mask))

        # 6) Sweep likely enemy territory instead of only local wandering.
        action = self._strategic_sweep_action(observation, mask)
        if action is not None:
            return self._record(self._anti_loop_adjust(action, mask))

        # 7) Exploration / loop breaker.
        action = self._explore_action(view, observation, mask)
        action = self._anti_loop_adjust(action, mask)
        fallback = self._policy_action(observation, mask)
        if fallback is not None and fallback != STAY and (action == STAY or self._is_stuck()):
            action = fallback
        return self._record(action)

    # ------------------------------------------------------------------
    # TorchScript policy fallback
    # ------------------------------------------------------------------
    def _load_policy(self) -> bool:
        if self.policy_checked:
            return self.policy is not None
        self.policy_checked = True
        model_path = Path(__file__).with_name("ae_policy_traced.pt")
        if not model_path.exists():
            return False
        try:
            import torch

            torch.set_num_threads(1)
            self.policy = torch.jit.load(str(model_path), map_location="cpu")
            self.policy.eval()
            self.policy_hidden = torch.zeros(1, 256, dtype=torch.float32)
            return True
        except Exception:
            self.policy = None
            self.policy_hidden = None
            return False

    def _policy_action(self, obs: dict[str, Any], mask: np.ndarray) -> int | None:
        if not self._load_policy():
            return None
        try:
            import torch

            agent_view, base_view, scalars = self._policy_inputs(obs)
            if self.policy_hidden is None:
                self.policy_hidden = torch.zeros(1, 256, dtype=torch.float32)
            with torch.no_grad():
                logits, _value, hidden = self.policy(
                    torch.as_tensor(agent_view, dtype=torch.float32).unsqueeze(0),
                    torch.as_tensor(base_view, dtype=torch.float32).unsqueeze(0),
                    torch.as_tensor(scalars, dtype=torch.float32).unsqueeze(0),
                    self.policy_hidden,
                )
                self.policy_hidden = hidden.detach()
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                if logits.ndim == 1:
                    logits = logits.unsqueeze(0)
                mask_t = torch.as_tensor(mask[:6] > 0, dtype=torch.bool).unsqueeze(0)
                logits = logits.masked_fill(~mask_t, torch.finfo(logits.dtype).min)
                action = int(torch.argmax(logits, dim=-1).item())
            if self._legal(mask, action):
                return action
        except Exception:
            self.policy_hidden = None
            return None
        return None

    def _policy_inputs(self, obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        agent_view_hwc = np.asarray(obs.get("agent_viewcone", np.zeros((7, 5, 25))), dtype=np.float32)
        base_view_hwc = np.asarray(obs.get("base_viewcone", np.zeros((7, 7, 25))), dtype=np.float32)
        agent_view = np.transpose(agent_view_hwc, (2, 0, 1)).astype(np.float32, copy=False)
        base_view = np.transpose(base_view_hwc, (2, 0, 1)).astype(np.float32, copy=False)

        direction = int(obs.get("direction", 0))
        direction_onehot = np.zeros(4, dtype=np.float32)
        if 0 <= direction < 4:
            direction_onehot[direction] = 1.0

        scalars = np.concatenate(
            [
                direction_onehot,
                np.asarray(obs.get("location", [0, 0]), dtype=np.float32).ravel()[:2] / 16.0,
                np.asarray(obs.get("base_location", [0, 0]), dtype=np.float32).ravel()[:2] / 16.0,
                np.asarray(obs.get("health", [60.0]), dtype=np.float32).ravel()[:1] / 60.0,
                np.array([float(obs.get("frozen_ticks", 0)) / 3.0], dtype=np.float32),
                np.asarray(obs.get("base_health", [100.0]), dtype=np.float32).ravel()[:1] / 100.0,
                np.asarray(obs.get("team_resources", [0.0]), dtype=np.float32).ravel()[:1] / 100.0,
                np.array([float(obs.get("team_bombs", 0)) / 50.0], dtype=np.float32),
                np.array([float(obs.get("step", 0)) / 200.0], dtype=np.float32),
                self._policy_local_summary(agent_view_hwc),
            ],
            dtype=np.float32,
        )
        return agent_view, base_view, scalars

    def _policy_local_summary(self, view: np.ndarray) -> np.ndarray:
        if view.ndim != 3:
            return np.zeros(30, dtype=np.float32)
        h, w, c = view.shape
        denom = max(1.0, float(h * w))
        groups = [
            (0,),
            tuple(range(1, min(c, 5))),
            tuple(range(13, min(c, 17))),
            tuple(range(17, min(c, 25))),
            (7,) if c > 7 else (),
            (8,) if c > 8 else (),
            (6,) if c > 6 else (),
            (10,) if c > 10 else (),
            (11, 12) if c > 12 else ((12,) if c > 12 else ()),
        ]
        sums = [self._channel_sum(view, group) / denom for group in groups]
        centroid_features: list[float] = []
        for group in [groups[4], groups[5], groups[6], groups[7], groups[8], groups[3], groups[2]]:
            dx, dy, mass = self._weighted_channel_center(view, group)
            centroid_features.extend([dx / max(1.0, w / 2), dy / max(1.0, h / 2), min(1.0, mass / denom)])
        return np.asarray(sums + centroid_features, dtype=np.float32)

    def _channel_sum(self, view: np.ndarray, channels: tuple[int, ...]) -> float:
        total = 0.0
        c = view.shape[2] if view.ndim == 3 else 0
        for ch in channels:
            if 0 <= int(ch) < c:
                total += float(view[:, :, int(ch)].sum())
        return total

    def _weighted_channel_center(self, view: np.ndarray, channels: tuple[int, ...]) -> tuple[float, float, float]:
        h, w, c = view.shape
        cy, cx = self._origin(view)
        mass = 0.0
        sx = 0.0
        sy = 0.0
        ys, xs = np.mgrid[0:h, 0:w]
        for ch in channels:
            if 0 <= int(ch) < c:
                layer = view[:, :, int(ch)].astype(np.float32)
                m = float(layer.sum())
                if m > 0:
                    sx += float(((xs - cx) * layer).sum())
                    sy += float(((ys - cy) * layer).sum())
                    mass += m
        if mass <= 1e-8:
            return 0.0, 0.0, 0.0
        return sx / mass, sy / mass, mass

    # ------------------------------------------------------------------
    # Observation geometry
    # ------------------------------------------------------------------
    def _location(self, obs: dict[str, Any]) -> tuple[int, int] | None:
        try:
            arr = np.asarray(obs.get("location", []), dtype=np.int64).ravel()
            if arr.size >= 2:
                return int(arr[0]), int(arr[1])
        except Exception:
            pass
        return None

    def _origin(self, view: np.ndarray) -> tuple[int, int]:
        if view.ndim == 3 and view.shape[2] > 9:
            cells = np.argwhere(view[:, :, 9] > 0.5)
            if len(cells) > 0:
                y, x = cells[0]
                return int(y), int(x)
        h, w = view.shape[:2]
        return min(2, h - 1), w // 2

    def _local_to_world(self, obs: dict[str, Any], y: int, x: int, view: np.ndarray) -> tuple[int, int] | None:
        loc = self._location(obs)
        if loc is None:
            return None
        cy, cx = self._origin(view)
        dy = y - cy
        dx = x - cx
        facing = int(obs.get("direction", 0))
        fx, fy = DIR_TO_DELTA.get(facing, (1, 0))
        rx, ry = -fy, fx
        wx = loc[0] + dy * fx + dx * rx
        wy = loc[1] + dy * fy + dx * ry
        if 0 <= wx < self.grid_size and 0 <= wy < self.grid_size:
            return int(wx), int(wy)
        return None

    def _relative_action_towards(self, obs: dict[str, Any], target: tuple[int, int], mask: np.ndarray) -> int | None:
        loc = self._location(obs)
        if loc is None:
            return None
        dx = target[0] - loc[0]
        dy = target[1] - loc[1]
        if dx == 0 and dy == 0:
            return None
        if abs(dx) >= abs(dy):
            desired = 0 if dx > 0 else 2
        else:
            desired = 1 if dy > 0 else 3
        facing = int(obs.get("direction", 0))
        if desired == facing:
            prefs = [FORWARD, RIGHT, LEFT, BACKWARD, STAY]
        elif desired == (facing + 2) % 4:
            prefs = [BACKWARD, RIGHT, LEFT, FORWARD, STAY]
        elif desired == (facing - 1) % 4:
            prefs = [LEFT, FORWARD, BACKWARD, RIGHT, STAY]
        else:
            prefs = [RIGHT, FORWARD, BACKWARD, LEFT, STAY]
        return self._first_legal(mask, prefs)

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------
    def _update_memory(self, obs: dict[str, Any], view: np.ndarray) -> None:
        if view.ndim != 3:
            return
        h, w, c = view.shape
        step = int(obs.get("step", 0))

        # Forget low-value stale targets, keep enemy bases longer.
        if step % 20 == 0 and self.targets:
            new_targets = {}
            for p, (kind, value, last) in self.targets.items():
                max_age = 160 if kind == "enemy_base" else 70
                if step - last < max_age:
                    new_targets[p] = (kind, value, last)
            self.targets = new_targets

        # Channel values from provided logs.
        channels = [
            (12, "enemy_base", 260.0),
            (10, "enemy_agent", 85.0),
            (7, "mission", 62.0),
            (8, "resource", 24.0),
            (6, "recon", 7.0),
        ]
        for y in range(h):
            for x in range(w):
                if c > 0 and view[y, x, 0] <= 0.1:
                    continue
                for ch, kind, value in channels:
                    if ch < c and view[y, x, ch] > 0.5:
                        wp = self._local_to_world(obs, y, x, view)
                        if wp is not None:
                            self.targets[wp] = (kind, value, step)

    def _update_base_memory(self, obs: dict[str, Any]) -> None:
        """Remember close base threats without turning defense into camping."""
        step = int(obs.get("step", 0))
        try:
            base_arr = np.asarray(obs.get("base_location", []), dtype=np.int64).ravel()
            if base_arr.size < 2:
                return
            bx, by = int(base_arr[0]), int(base_arr[1])
        except Exception:
            return

        base_view = np.asarray(obs.get("base_viewcone", np.zeros((5, 5, 25))), dtype=np.float32)
        if base_view.ndim == 3:
            h, w, c = base_view.shape
            cy, cx = h // 2, w // 2
            if c > 10:
                ys, xs = np.where(base_view[:, :, 10] > 0.5)
                for y, x in zip(ys.tolist(), xs.tolist()):
                    wp = (bx + int(x - cx), by + int(y - cy))
                    if 0 <= wp[0] < self.grid_size and 0 <= wp[1] < self.grid_size:
                        self.targets[wp] = ("base_threat", 95.0, step)

        try:
            base_health = float(np.asarray(obs.get("base_health", [100.0]), dtype=np.float32).ravel()[0])
        except Exception:
            base_health = 100.0
        if base_health < 55.0:
            self.targets[(bx, by)] = ("defend_base", 70.0, step)

    # ------------------------------------------------------------------
    # Local grid, danger and BFS
    # ------------------------------------------------------------------
    def _in_view(self, view: np.ndarray, p: tuple[int, int]) -> bool:
        h, w = view.shape[:2]
        return 0 <= p[0] < h and 0 <= p[1] < w

    def _visible(self, view: np.ndarray, p: tuple[int, int]) -> bool:
        if view.ndim != 3 or not self._in_view(view, p):
            return False
        return view[p[0], p[1], 0] > 0.05 if view.shape[2] > 0 else True

    def _blocked(self, view: np.ndarray, p: tuple[int, int]) -> bool:
        if view.ndim != 3 or not self._in_view(view, p):
            return True
        c = view.shape[2]
        if c > 0 and view[p[0], p[1], 0] <= 0.05:
            return True
        # Walls are edge constraints, not blocked tiles. The action mask guards
        # immediate wall collisions, so local search should not treat a cell
        # with wall edges as unreachable.
        bomb = float(view[p[0], p[1], 17:min(c, 19)].sum()) if c > 17 else 0.0
        return bomb > 0.5

    def _neighbors(self, view: np.ndarray, p: tuple[int, int]):
        for a in [FORWARD, BACKWARD, LEFT, RIGHT]:
            dy, dx = ACTION_TO_LOCAL_DELTA[a]
            q = (p[0] + dy, p[1] + dx)
            if self._in_view(view, q) and not self._blocked(view, q):
                yield a, q

    def _danger_map(self, view: np.ndarray, extra_origin_bomb: bool = False) -> np.ndarray:
        if view.ndim != 3:
            return np.zeros((1, 1), dtype=np.float32)
        h, w, c = view.shape
        danger = np.zeros((h, w), dtype=np.float32)
        bomb_cells: list[tuple[int, int, float]] = []
        if c > 17:
            ys, xs = np.where(view[:, :, 17] > 0.5)
            for y, x in zip(ys.tolist(), xs.tolist()):
                bomb_cells.append((int(y), int(x), 1.0))
        if c > 18:
            ys, xs = np.where(view[:, :, 18] > 0.5)
            for y, x in zip(ys.tolist(), xs.tolist()):
                bomb_cells.append((int(y), int(x), 1.0))
        if extra_origin_bomb:
            cy, cx = self._origin(view)
            bomb_cells.append((cy, cx, 1.25))

        r = 2
        for by, bx, weight in bomb_cells:
            for y in range(max(0, by - r), min(h, by + r + 1)):
                for x in range(max(0, bx - r), min(w, bx + r + 1)):
                    d = max(abs(y - by), abs(x - bx))
                    if d <= r:
                        danger[y, x] = max(danger[y, x], weight * (1.0 - 0.12 * d))
        return danger

    def _is_dangerous_cell(self, danger: np.ndarray, p: tuple[int, int]) -> bool:
        if danger.ndim != 2:
            return False
        if 0 <= p[0] < danger.shape[0] and 0 <= p[1] < danger.shape[1]:
            return danger[p[0], p[1]] > 0.20
        return False

    def _local_danger_score(self, view: np.ndarray) -> float:
        danger = self._danger_map(view)
        cy, cx = self._origin(view)
        if danger.ndim == 2 and 0 <= cy < danger.shape[0] and 0 <= cx < danger.shape[1]:
            return float(danger[cy, cx])
        return 0.0

    def _bfs_escape_action(self, view: np.ndarray, danger: np.ndarray, mask: np.ndarray) -> int:
        if view.ndim != 3:
            return self._first_legal(mask, [BACKWARD, FORWARD, RIGHT, LEFT, STAY])
        start = self._origin(view)
        q = deque([(start, [])])
        seen = {start}
        best = None

        while q:
            pos, path = q.popleft()
            depth = len(path)
            safe = not self._is_dangerous_cell(danger, pos)
            if depth > 0 and safe:
                # Prefer not only first safe cell, but useful roomy safe areas.
                room = self._reachable_safe_count(view, danger, pos, max_depth=3)
                score = room - 0.3 * depth + self._cell_value(view, pos) * 0.1
                if best is None or score > best[0]:
                    best = (score, path)
            if depth >= 5:
                continue
            for action, nb in self._neighbors(view, pos):
                if nb in seen:
                    continue
                if depth == 0 and not self._legal(mask, action):
                    continue
                seen.add(nb)
                q.append((nb, path + [action]))

        if best is not None and best[1]:
            return best[1][0]

        # Fallback to action safety scoring.
        scores = []
        for action in [FORWARD, BACKWARD, LEFT, RIGHT, STAY]:
            if self._legal(mask, action):
                scores.append((self._action_safety_score(view, danger, action), action))
        return max(scores)[1] if scores else self._first_legal(mask, [BACKWARD, FORWARD, RIGHT, LEFT, STAY])

    def _reachable_safe_count(self, view: np.ndarray, danger: np.ndarray, start: tuple[int, int], max_depth: int = 3) -> int:
        q = deque([(start, 0)])
        seen = {start}
        count = 0
        while q:
            pos, d = q.popleft()
            if not self._is_dangerous_cell(danger, pos):
                count += 1
            if d >= max_depth:
                continue
            for _a, nb in self._neighbors(view, pos):
                if nb not in seen and not self._is_dangerous_cell(danger, nb):
                    seen.add(nb)
                    q.append((nb, d + 1))
        return count

    # ------------------------------------------------------------------
    # Bombing
    # ------------------------------------------------------------------
    def _blast_target_score(self, view: np.ndarray, origin: tuple[int, int]) -> float:
        if view.ndim != 3:
            return 0.0
        h, w, c = view.shape
        cy, cx = origin
        score = 0.0
        r = 2
        for y in range(max(0, cy - r), min(h, cy + r + 1)):
            for x in range(max(0, cx - r), min(w, cx + r + 1)):
                if max(abs(y - cy), abs(x - cx)) > r:
                    continue
                # Favor bases most, then enemy, then missions/walls.
                if c > 12:
                    score += 260.0 * float(view[y, x, 12])
                if c > 10:
                    score += 75.0 * float(view[y, x, 10])
                if c > 13:
                    score += 5.0 * float(view[y, x, 13:min(c, 17)].sum())
        return score

    def _escape_possible_after_bomb(self, view: np.ndarray, mask: np.ndarray) -> bool:
        danger = self._danger_map(view, extra_origin_bomb=True)
        action = self._bfs_escape_action(view, danger, mask)
        return action != STAY and self._legal(mask, action)

    def _should_bomb(self, obs: dict[str, Any], view: np.ndarray, mask: np.ndarray) -> bool:
        if not self._legal(mask, PLACE_BOMB):
            return False
        if int(obs.get("team_bombs", 0)) <= 0:
            return False
        if self.bomb_cooldown > 0:
            return False
        if self._local_danger_score(view) > 0.05:
            return False
        if self.recent_actions and self.recent_actions[-1] == PLACE_BOMB:
            return False
        if not self._escape_possible_after_bomb(view, mask):
            return False

        score = self._blast_target_score(view, self._origin(view))
        # Aggressive but less wall-spam: require real value.
        if score >= 60.0:
            return True
        return self._stuck_bomb_opportunity(obs, view)

    def _stuck_bomb_opportunity(self, obs: dict[str, Any], view: np.ndarray) -> bool:
        """Use a bomb as a last-resort loop breaker near remembered value."""
        if not self._is_stuck():
            return False
        if int(obs.get("team_bombs", 0)) <= 2:
            return False

        loc = self._location(obs)
        if loc is not None:
            step = int(obs.get("step", 0))
            for target, (kind, value, last_seen) in self.targets.items():
                if kind == "defend_base" or value < 20.0 or step - last_seen > 80:
                    continue
                dist = abs(target[0] - loc[0]) + abs(target[1] - loc[1])
                if dist <= 3:
                    return True

        if view.ndim != 3 or view.shape[2] <= 13:
            return False
        cy, cx = self._origin(view)
        h, w, c = view.shape
        for y in range(max(0, cy - 2), min(h, cy + 3)):
            for x in range(max(0, cx - 2), min(w, cx + 3)):
                if max(abs(y - cy), abs(x - cx)) <= 2 and float(view[y, x, 13:min(c, 17)].sum()) > 0.5:
                    return True
        return False

    # ------------------------------------------------------------------
    # Targeting and search
    # ------------------------------------------------------------------
    def _cell_value(self, view: np.ndarray, p: tuple[int, int]) -> float:
        if view.ndim != 3 or not self._in_view(view, p):
            return -999.0
        y, x = p
        c = view.shape[2]
        if c > 0 and view[y, x, 0] <= 0.05:
            return -20.0
        value = 0.0
        if c > 12:
            value += 240.0 * float(view[y, x, 12])
        if c > 10:
            value += 80.0 * float(view[y, x, 10])
        if c > 7:
            value += 60.0 * float(view[y, x, 7])
        if c > 8:
            value += 22.0 * float(view[y, x, 8])
        if c > 6:
            value += 7.0 * float(view[y, x, 6])
        if c > 13:
            value += 1.5 * float(view[y, x, 13:min(c, 17)].sum())
        # Penalize bombs/walls.
        if c > 17:
            value -= 20.0 * float(view[y, x, 17:min(c, 21)].sum())
        if c > 1:
            value -= 0.8 * float(view[y, x, 1:min(c, 5)].sum())
        return value

    def _visible_bfs_target_action(self, view: np.ndarray, mask: np.ndarray) -> int | None:
        if view.ndim != 3:
            return None
        start = self._origin(view)
        danger = self._danger_map(view)
        q = deque([(start, [])])
        seen = {start}
        best = None

        while q:
            pos, path = q.popleft()
            depth = len(path)
            if depth > 0:
                value = self._cell_value(view, pos)
                if self._is_dangerous_cell(danger, pos):
                    value -= 80.0
                score = value - 2.5 * depth
                if value > 5 and (best is None or score > best[0]):
                    best = (score, path)
            if depth >= 6:
                continue
            for action, nb in self._neighbors(view, pos):
                if nb in seen:
                    continue
                if depth == 0 and not self._legal(mask, action):
                    continue
                seen.add(nb)
                q.append((nb, path + [action]))

        if best is not None and best[1]:
            return best[1][0]
        return None

    def _beam_search_action(self, view: np.ndarray, mask: np.ndarray, depth: int = 4, width: int = 8) -> int | None:
        if view.ndim != 3:
            return None
        start = self._origin(view)
        danger = self._danger_map(view)
        # Only run beam when there is something tactical nearby.
        local_interest = 0.0
        c = view.shape[2]
        if c > 6:
            local_interest += float(view[:, :, 6:min(c, 13)].sum())
        if c > 13:
            local_interest += 0.2 * float(view[:, :, 13:min(c, 17)].sum())
        if c > 17:
            local_interest += float(view[:, :, 17:min(c, 19)].sum())
        if local_interest < 0.5:
            return None

        beams = [(0.0, start, None, [])]  # score, pos, first_action, path
        for d in range(depth):
            new_beams = []
            for score, pos, first, path in beams:
                for action in [FORWARD, BACKWARD, LEFT, RIGHT, STAY]:
                    if d == 0 and not self._legal(mask, action):
                        continue
                    if action == STAY:
                        nb = pos
                    else:
                        dy, dx = ACTION_TO_LOCAL_DELTA[action]
                        nb = (pos[0] + dy, pos[1] + dx)
                        if self._blocked(view, nb):
                            continue
                    first_action = action if first is None else first
                    step_score = self._cell_value(view, nb) * (0.25 if d > 0 else 0.45)
                    if self._is_dangerous_cell(danger, nb):
                        step_score -= 60.0 / (d + 1)
                    if action == STAY:
                        step_score -= 1.5
                    if action in (FORWARD, BACKWARD):
                        step_score += 0.15
                    # Discourage obvious loops.
                    if len(self.recent_actions) >= 2 and action == self.recent_actions[-1] == self.recent_actions[-2]:
                        step_score -= 0.5
                    new_beams.append((score + step_score, nb, first_action, path + [action]))
            if not new_beams:
                break
            new_beams.sort(key=lambda item: item[0], reverse=True)
            beams = new_beams[:width]

        if not beams:
            return None
        best_score, _pos, first, _path = max(beams, key=lambda item: item[0])
        # Only use beam if it is meaningfully confident.
        if first is not None and best_score > 3.0:
            return int(first)
        return None

    def _remembered_target_action(self, obs: dict[str, Any], mask: np.ndarray) -> int | None:
        loc = self._location(obs)
        if loc is None or not self.targets:
            return None
        step = int(obs.get("step", 0))
        best = None
        for p, (kind, value, last_seen) in self.targets.items():
            dist = abs(p[0] - loc[0]) + abs(p[1] - loc[1])
            age_penalty = 0.025 * max(0, step - last_seen) if kind == "enemy_base" else 0.07 * max(0, step - last_seen)
            visit_penalty = 0.5 * self.visited.get(p, 0)
            score = value - 0.16 * dist - age_penalty - visit_penalty
            if best is None or score > best[0]:
                best = (score, p, kind)
        if best is None or best[0] < -3.0:
            return None
        return self._relative_action_towards(obs, best[1], mask)

    def _strategic_sweep_action(self, obs: dict[str, Any], mask: np.ndarray) -> int | None:
        loc = self._location(obs)
        if loc is None:
            return None
        try:
            base = np.asarray(obs.get("base_location", []), dtype=np.int64).ravel()
            bx, by = int(base[0]), int(base[1])
        except Exception:
            bx, by = loc

        step = int(obs.get("step", 0))
        far_corner = (self.grid_size - 1 - bx, self.grid_size - 1 - by)
        center = (self.grid_size // 2, self.grid_size // 2)
        corners = [(1, 1), (self.grid_size - 2, 1), (self.grid_size - 2, self.grid_size - 2), (1, self.grid_size - 2)]
        corners.sort(key=lambda p: abs(p[0] - bx) + abs(p[1] - by), reverse=True)

        route = [center, far_corner, *corners]
        phase = min(len(route) - 1, max(0, step // 35))
        if step >= 120:
            phase = (step // 20) % len(route)

        # If a sweep point has been visited repeatedly, rotate to the next one.
        for offset in range(len(route)):
            target = route[(phase + offset) % len(route)]
            if self.visited.get(target, 0) < 3:
                action = self._relative_action_towards(obs, target, mask)
                if action is not None:
                    return action

        return self._first_legal(mask, [FORWARD, RIGHT, LEFT, BACKWARD, STAY])

    # ------------------------------------------------------------------
    # Exploration / anti-loop
    # ------------------------------------------------------------------
    def _explore_action(self, view: np.ndarray, obs: dict[str, Any], mask: np.ndarray) -> int:
        if self._is_stuck():
            self.loop_break_turn = 3
            return self._first_legal(mask, [RIGHT, LEFT, FORWARD, BACKWARD, STAY])

        if len(self.recent_actions) >= 4:
            pattern = list(self.recent_actions)[-4:]
            if pattern in ([FORWARD, BACKWARD, FORWARD, BACKWARD], [BACKWARD, FORWARD, BACKWARD, FORWARD]):
                return self._first_legal(mask, [RIGHT, LEFT, FORWARD, BACKWARD, STAY])

        # Prefer locally roomy/safe movement.
        danger = self._danger_map(view)
        scores = []
        for action in [FORWARD, RIGHT, LEFT, BACKWARD, STAY]:
            if not self._legal(mask, action):
                continue
            scores.append((self._action_safety_score(view, danger, action), action))
        if scores:
            return max(scores)[1]
        return self._first_legal(mask, [FORWARD, RIGHT, LEFT, BACKWARD, STAY])

    def _action_safety_score(self, view: np.ndarray, danger: np.ndarray, action: int) -> float:
        if view.ndim != 3:
            return 0.0
        cy, cx = self._origin(view)
        dy, dx = ACTION_TO_LOCAL_DELTA.get(action, (0, 0))
        p = (cy + dy, cx + dx)
        if not self._in_view(view, p):
            return -999.0
        if self._blocked(view, p) and action != STAY:
            return -50.0
        score = 1.0
        score += 0.15 * self._reachable_safe_count(view, danger, p, max_depth=3)
        score += 0.08 * self._cell_value(view, p)
        if self._is_dangerous_cell(danger, p):
            score -= 30.0
        if action == STAY:
            score -= 3.0
        if action == FORWARD:
            score += 0.25
        return score

    def _anti_loop_adjust(self, action: int, mask: np.ndarray) -> int:
        if len(self.recent_actions) >= 6:
            last = list(self.recent_actions)[-6:]
            if last.count(LEFT) + last.count(RIGHT) >= 5:
                return self._first_legal(mask, [FORWARD, BACKWARD, RIGHT, LEFT, STAY])
            if last == [FORWARD, BACKWARD, FORWARD, BACKWARD, FORWARD, BACKWARD]:
                return self._first_legal(mask, [RIGHT, LEFT, FORWARD, BACKWARD, STAY])
            if last.count(PLACE_BOMB) >= 2:
                return self._first_legal(mask, [FORWARD, BACKWARD, RIGHT, LEFT, STAY])
        if self._is_stuck() and action in (STAY, LEFT, RIGHT):
            return self._first_legal(mask, [FORWARD, RIGHT, LEFT, BACKWARD, STAY])
        return action

    def _is_stuck(self) -> bool:
        if len(self.recent_locations) < 8:
            return False
        recent = list(self.recent_locations)[-8:]
        return len(set(recent)) <= 2

    def _record(self, action: int) -> int:
        if int(action) == PLACE_BOMB:
            self.escape_ticks = min(self.escape_ticks, 3)
            self.bomb_cooldown = min(self.bomb_cooldown, 2)
        self.recent_actions.append(int(action))
        return int(action)

    def _legal(self, mask: np.ndarray, action: int) -> bool:
        return 0 <= action < len(mask) and mask[action] > 0

    def _first_legal(self, mask: np.ndarray, prefs: list[int]) -> int:
        for a in prefs:
            if self._legal(mask, a):
                return int(a)
        legal = np.flatnonzero(mask > 0)
        return int(legal[0]) if len(legal) else STAY
