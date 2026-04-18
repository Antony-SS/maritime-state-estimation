from __future__ import annotations

import numpy as np

from gridmap import GridMap
import heapq # for astar
from rps.robotarium_abc import ARobotarium

class AStarPathPlanner:
    """
    Path planning over a 2D **cost grid**.  Supports inflation of costs to account for robot size/discretization.

    Cells with value ``-1`` are **hard obstacles**: they are dilated by ``obstacle_inflation_radius_m`` (flat disk,
    no falloff), never used as move targets, and take precedence over cost inflation in the merged map.
    """

    INFLATION_FACTOR = ARobotarium.COLLISION_DIAMETER*1.5 / 2

    def __init__(
        self,
        gridmap: GridMap,
        cost_inflation_radius_m: float | None = None,
        obstacle_inflation_radius_m: float | None = None,
    ):
        self.gridmap = gridmap
        self.data = gridmap.data
        self._resolution = float(gridmap.resolution)
        self._cost_inflation_radius_m = (
            float(cost_inflation_radius_m)
            if cost_inflation_radius_m is not None
            else float(self.INFLATION_FACTOR)
        )
        self._obstacle_inflation_radius_m = (
            float(obstacle_inflation_radius_m)
            if obstacle_inflation_radius_m is not None
            else float(self.INFLATION_FACTOR)
        )
        self.inflated_map = self._create_inflated_map()


    def plan(self, start: np.ndarray, goal: np.ndarray, heuristic_weight: float = 1.0) -> np.ndarray:
        """ Pass in start and goal as (x, y) coordinates (meters in the robotarium, this will handle conversion to grid indices).
        Calculates a path as ``(n, 2)`` int indices ``[i, j], ...`` which are converted back to (x, y) coordinates (meters in the robotarium)."""
        
        assert start.shape == (2,), "Start must be a 2D array"
        assert goal.shape == (2,), "Goal must be a 2D array"

        start_idx = self.gridmap.xy_to_uv(start)
        goal_idx = self.gridmap.xy_to_uv(goal)

        start_idx = tuple(start_idx)
        goal_idx = tuple(goal_idx)

        # ``xy_to_uv`` returns ``(u, v)``: column (east) in ``[0, width)``, row (north) in ``[0, height)``.
        if not (0 <= goal_idx[0] < self.gridmap.width and 0 <= goal_idx[1] < self.gridmap.height):
            print("Goal index out of bounds")
            return np.array([])

        if not (0 <= start_idx[0] < self.gridmap.width and 0 <= start_idx[1] < self.gridmap.height):
            print("Start index out of bounds")
            return np.array([])

        path = self._plan(start_idx, goal_idx, heuristic_weight)

        if len(path) == 0:
            print("No path found")
            return np.array([]) # empty array if no path found
        else:
            path = self.gridmap.uv_to_xy(np.array(path))
            return path

    def _plan(self, start_idx: tuple[int, int], goal_idx: tuple[int, int], heuristic_weight: float = 1.0) -> list[tuple[int, int]]:
        """A* path planning algorithm."""

        tie_breaker_util = 0
        best_costs: dict[tuple[int, int], float] = {}
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        frontier: list[tuple[float, int, tuple[int, int]]] = []
        found_goal = False
        came_from[tuple(start_idx)] = tuple(start_idx)
        heapq.heappush(frontier, (0.0, tie_breaker_util, start_idx))
        best_costs[tuple(start_idx)] = heuristic_weight * self._heuristic(start_idx, goal_idx)

        while len(frontier) > 0:
            _ , _, current_cell = heapq.heappop(frontier)
            current_cost = best_costs[current_cell]
            if current_cell == goal_idx:
                found_goal = True
                break

            for neighbor in self._get_valid_neighbors(tuple(current_cell)):
                u_n, v_n = neighbor
                step_cost = float(self.inflated_map[v_n, u_n])
                neighbor_cost = (
                    current_cost
                    + step_cost
                )
                if neighbor not in best_costs or neighbor_cost < best_costs[neighbor]:
                    best_costs[neighbor] = neighbor_cost
                    came_from[neighbor] = tuple(current_cell)
                    weighted_cost = neighbor_cost + heuristic_weight * self._heuristic(neighbor, goal_idx)
                    heapq.heappush(frontier, (weighted_cost, tie_breaker_util, neighbor))
                    tie_breaker_util += 1

        if not found_goal:
            print("No path found")
            return []
        return self._reconstruct_path(came_from, tuple(start_idx), tuple(goal_idx))

    def _get_valid_neighbors(self, cell: tuple[int, int]) -> list[tuple[int, int]]:
        """8-neighbors; ``cell`` is ``(u, v)`` with ``u`` in ``[0, width)``, ``v`` in ``[0, height)``."""
        u, v = cell
        neighbors: list[tuple[int, int]] = []
        for du in range(-1, 2):
            for dv in range(-1, 2):
                if du == 0 and dv == 0:
                    continue
                nu, nv = u + du, v + dv
                if nu < 0 or nu >= self.gridmap.width or nv < 0 or nv >= self.gridmap.height:
                    continue
                if float(self.inflated_map[nv, nu]) < 0.0:
                    continue
                neighbors.append((nu, nv))
        return neighbors

    def _heuristic(self, cell: tuple[int, int], goal_idx: tuple[int, int]) -> float:
        """ euclidean distance heuristic """
        return np.linalg.norm(np.array(cell) - np.array(goal_idx))

    def _reconstruct_path(self, came_from: dict[tuple[int, int], tuple[int, int]], start_idx: tuple[int, int], goal_idx: tuple[int, int]) -> list[tuple[int, int]]:
        path: list[tuple[int, int]] = []
        current = goal_idx
        while current != start_idx:
            path.append(current)
            current = came_from[current]
        path.append(start_idx)
        path = path[::-1]
        return path

    def _disk_offsets_weights(
        self, radius_m: float, res: float, gaussian: bool
    ) -> list[tuple[int, int, float]]:
        if radius_m <= 0.0 or res <= 0.0:
            return [(0, 0, 1.0)] if not gaussian else []
        r_cells = int(np.ceil(radius_m / res))
        sigma_m = max(radius_m * 0.5, 1e-9)
        offsets: list[tuple[int, int, float]] = []
        for di in range(-r_cells, r_cells + 1):
            for dj in range(-r_cells, r_cells + 1):
                d_m = float(np.hypot(di, dj) * res)
                if d_m > radius_m:
                    continue
                if gaussian:
                    w = float(np.exp(-0.5 * (d_m / sigma_m) ** 2))
                else:
                    w = 1.0
                offsets.append((di, dj, w))
        return offsets

    def _gaussian_inflate(self, sources: np.ndarray, radius_m: float, res: float) -> np.ndarray:
        """Gaussian falloff convolution in metres (same rule as the former single-map inflation)."""
        h, w = sources.shape
        if radius_m <= 0.0 or res <= 0.0:
            return sources.astype(np.float64, copy=True)
        offsets = self._disk_offsets_weights(radius_m, res, gaussian=True)
        out = np.zeros((h, w), dtype=np.float64)
        for di, dj, weight in offsets:
            i_lo = max(0, di)
            i_hi = min(h, h + di)
            j_lo = max(0, dj)
            j_hi = min(w, w + dj)
            if i_lo >= i_hi or j_lo >= j_hi:
                continue
            out[i_lo:i_hi, j_lo:j_hi] += weight * sources[i_lo - di : i_hi - di, j_lo - dj : j_hi - dj]
        return out

    def _dilate_obstacle_mask(self, hard: np.ndarray, radius_m: float, res: float) -> np.ndarray:
        """Flat disk dilation of boolean ``hard`` (obstacle footprint, no falloff)."""
        h, w = hard.shape
        if not np.any(hard):
            return np.zeros((h, w), dtype=bool)
        if radius_m <= 0.0 or res <= 0.0:
            return hard.astype(bool, copy=True)
        offsets = self._disk_offsets_weights(radius_m, res, gaussian=False)
        out = np.zeros((h, w), dtype=bool)
        for di, dj, _w in offsets:
            i_lo = max(0, di)
            i_hi = min(h, h + di)
            j_lo = max(0, dj)
            j_hi = min(w, w + dj)
            if i_lo >= i_hi or j_lo >= j_hi:
                continue
            out[i_lo:i_hi, j_lo:j_hi] |= hard[i_lo - di : i_hi - di, j_lo - dj : j_hi - dj]
        return out

    def _create_inflated_map(self) -> np.ndarray:
        """
        Hard obstacles (``data == -1``) dilate with a flat disk in metres; other costs Gaussian-blur
        in metres using ``_cost_inflation_radius_m``. Dilated obstacle cells are set to ``-1`` last so
        they always override blurred costs.
        """
        raw = np.asarray(self.data, dtype=np.float64)
        h, w = raw.shape
        res = self._resolution
        hard = raw == -1.0
        obstacle_footprint = self._dilate_obstacle_mask(hard, self._obstacle_inflation_radius_m, res)
        sources = np.where(hard, 0.0, raw)
        cost_field = self._gaussian_inflate(sources, self._cost_inflation_radius_m, res)
        out = cost_field
        out[obstacle_footprint] = -1.0
        return out


def _xy_cov_ellipse_width_height_angle_deg(P_xy: np.ndarray, n_sigma: 1.0) -> tuple[float, float, float]:

    """``matplotlib.patches.Ellipse`` diameters (m) and CCW angle (deg) from east for ``P``'s ``(x, y)`` block."""
    a = np.asarray(P_xy, dtype=float)
    if a.ndim == 2 and a.shape == (3, 3):
        a = a[:2, :2]
    elif a.size == 9:
        a = a.reshape(3, 3)[:2, :2]
    elif a.size == 4:
        a = a.reshape(2, 2)
    else:
        raise ValueError(f"expected 3x3 covariance or its (x,y) 2x2 block, got shape {a.shape}")
    P_xy = 0.5 * (a + a.T)
    lam, vecs = np.linalg.eigh(P_xy)
    lam = np.clip(lam, 1e-12, None)
    order = np.argsort(lam)[::-1]
    lam = lam[order]
    vecs = vecs[:, order]
    w = 2.0 * n_sigma * float(np.sqrt(lam[0]))
    h = 2.0 * n_sigma * float(np.sqrt(lam[1]))
    ang_deg = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    return w, h, ang_deg

def _point_xy_at_arclength(path_xy: np.ndarray, s: float) -> np.ndarray:
    """Point on the polyline at arc length ``s`` from ``path_xy[0]`` (clamped to the end)."""
    p = np.asarray(path_xy, dtype=np.float64)
    n = p.shape[0]
    if n == 0:
        return np.zeros(2, dtype=np.float64)
    if n == 1:
        return p[0].copy()
    if s <= 0.0:
        return p[0].copy()
    cum = 0.0
    for i in range(n - 1):
        a = p[i]
        b = p[i + 1]
        L = float(np.linalg.norm(b - a))
        if cum + L >= s - 1e-12:
            t = 0.0 if L < 1e-18 else (s - cum) / L
            return a + t * (b - a)
        cum += L
    return p[-1].copy()

def _closest_point_on_polyline_xy(path_xy: np.ndarray, q_xy: np.ndarray) -> tuple[np.ndarray, float]:
    """Closest point on the polyline to ``q_xy`` and arc length from ``path_xy[0]`` to that point."""
    p = np.asarray(path_xy, dtype=np.float64)
    q = np.asarray(q_xy, dtype=np.float64).reshape(2)
    if p.shape[0] == 0:
        return q.copy(), 0.0
    if p.shape[0] == 1:
        return p[0].copy(), 0.0
    best_d2 = np.inf
    best_xy = p[0].copy()
    best_s = 0.0
    cum = 0.0
    for i in range(p.shape[0] - 1):
        a = p[i]
        b = p[i + 1]
        ab = b - a
        ab2 = float(np.dot(ab, ab))
        t = 0.0 if ab2 < 1e-18 else float(np.dot(q - a, ab) / ab2)
        t = max(0.0, min(1.0, t))
        c = a + t * ab
        d2 = float(np.sum((q - c) ** 2))
        if d2 < best_d2:
            best_d2 = d2
            best_xy = c
            best_s = cum + float(np.linalg.norm(ab)) * t
        cum += float(np.linalg.norm(ab))
    return best_xy, best_s


def _closest_point_on_polyline_xy_forward(
    path_xy: np.ndarray,
    q_xy: np.ndarray,
    s_min: float,
    slack_back_m: float = 0.2,
) -> tuple[np.ndarray, float]:
    """
    Closest point on the polyline to ``q_xy``, but only considering points at arc length
    ``>= max(0, s_min - slack_back_m)``. Use when the path folds back on itself (e.g. lawn wiggle)
    so the tracker does not snap to an earlier branch and stall progress.
    """
    p = np.asarray(path_xy, dtype=np.float64)
    q = np.asarray(q_xy, dtype=np.float64).reshape(2)
    if p.shape[0] == 0:
        return q.copy(), 0.0
    if p.shape[0] == 1:
        return p[0].copy(), 0.0
    seg_lens = np.linalg.norm(np.diff(p, axis=0), axis=1)
    total_len = float(np.sum(seg_lens))
    s_floor = max(0.0, float(s_min) - float(slack_back_m))
    if total_len < 1e-12:
        return p[0].copy(), 0.0
    if s_floor >= total_len - 1e-9:
        return p[-1].copy(), total_len

    best_d2 = np.inf
    best_xy = p[0].copy()
    best_s = 0.0
    cum = 0.0
    for i in range(p.shape[0] - 1):
        a = p[i]
        b = p[i + 1]
        ab = b - a
        L = float(np.linalg.norm(ab))
        if L < 1e-18:
            cum += L
            continue
        s0 = cum
        s1 = cum + L
        cum = s1
        if s1 < s_floor - 1e-12:
            continue
        t_min = 0.0 if s0 + 1e-12 >= s_floor else (s_floor - s0) / L
        t_min = max(0.0, min(1.0, t_min))
        ab2 = float(np.dot(ab, ab))
        t = 0.0 if ab2 < 1e-18 else float(np.dot(q - a, ab) / ab2)
        t = max(t_min, min(1.0, t))
        c = a + t * ab
        d2 = float(np.sum((q - c) ** 2))
        if d2 < best_d2:
            best_d2 = d2
            best_xy = c
            best_s = s0 + L * t
    if not np.isfinite(best_d2):
        return _point_xy_at_arclength(p, s_floor), float(min(s_floor, total_len))
    return best_xy, float(best_s)


def _vertex_index_at_or_after_arclength(path_xy: np.ndarray, s: float) -> int:
    """Smallest vertex index ``k`` such that arc length ``path_xy[0] → path_xy[k]`` is ``>= s``."""
    p = np.asarray(path_xy, dtype=np.float64).reshape(-1, 2)
    m = p.shape[0]
    if m <= 1:
        return max(0, m - 1)
    s = max(0.0, float(s))
    cum = 0.0
    for k in range(1, m):
        cum += float(np.linalg.norm(p[k] - p[k - 1]))
        if cum >= s - 1e-9:
            return k
    return m - 1
