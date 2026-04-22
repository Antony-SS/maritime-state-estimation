from __future__ import annotations

import numpy as np

from gridmap import GridMap
import heapq # for astar
from rps.robotarium_abc import ARobotarium


def flat_disk_index_offsets(radius_m: float, resolution: float) -> list[tuple[int, int]]:
    """Integer (row, col) offsets whose cell-centre distance from origin is ``<= radius_m`` (m)."""
    if radius_m <= 0.0 or resolution <= 0.0:
        return [(0, 0)]
    res = float(resolution)
    r_cells = int(np.ceil(float(radius_m) / res))
    out: list[tuple[int, int]] = []
    for di in range(-r_cells, r_cells + 1):
        for dj in range(-r_cells, r_cells + 1):
            if float(np.hypot(di, dj) * res) <= float(radius_m) + 1e-12:
                out.append((di, dj))
    return out


def max_disk_dilate_nonnegative(
    sources: np.ndarray,
    radius_m: float,
    resolution: float,
) -> np.ndarray:
    """
    Morphological **max** filter with a flat disk of radius ``radius_m`` (greyscale dilation).

    Every cell takes the maximum ``sources`` value over all cells within that disk (isotropic in
    grid steps × ``resolution``). Preserves sharp peaks but **spreads** them into a thick halo—
    used after the anisotropic 2σ belief stamp so robot-radius inflation is visible even for
    near‑degenerate covariances.
    """
    s = np.maximum(0.0, np.asarray(sources, dtype=np.float64))
    h, w = s.shape
    res = float(resolution)
    if radius_m <= 0.0 or res <= 0.0:
        return s.copy()
    offsets = flat_disk_index_offsets(radius_m, res)
    out = s.copy()
    for di, dj in offsets:
        if di == 0 and dj == 0:
            continue
        i0 = max(0, -di)
        i1 = min(h, h - di)
        j0 = max(0, -dj)
        j1 = min(w, w - dj)
        if i0 >= i1 or j0 >= j1:
            continue
        out[i0:i1, j0:j1] = np.maximum(
            out[i0:i1, j0:j1],
            s[i0 + di : i1 + di, j0 + dj : j1 + dj],
        )
    return out


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

    def rebuild_inflated_map(self) -> None:
        """Recompute ``inflated_map`` after ``gridmap.data`` (or ``self.data``) changes."""
        self.data = self.gridmap.data
        self.inflated_map = self._create_inflated_map()

    def _disk_offsets_weights(
        self, radius_m: float, res: float, gaussian: bool
    ) -> list[tuple[int, int, float]]:
        if radius_m <= 0.0 or res <= 0.0:
            return [(0, 0, 1.0)] if not gaussian else []
        r_cells = int(np.ceil(radius_m / res))
        sigma_m = max(radius_m * 0.5, 1e-9) if gaussian else 1.0
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
        Hard obstacles (``data == -1``) dilate with a flat disk (robot footprint). Non‑obstacle
        costs are already **ellipse‑shaped** on ``gridmap.data`` (belief + ``r²I`` inflation in
        :func:`fused_enemy_belief_cost_field`); this step only clears costs on the dilated obstacle
        footprint (no isotropic cost blur / max‑disk, which would circularize hazards).
        """
        raw = np.asarray(self.data, dtype=np.float64)
        res = self._resolution
        hard = raw == -1.0
        obstacle_footprint = self._dilate_obstacle_mask(hard, self._obstacle_inflation_radius_m, res)
        sources = np.where(hard, 0.0, raw)
        sources = np.maximum(0.0, sources)
        cost_field = sources
        if self._cost_inflation_radius_m > 1e-12:
            cost_field = max_disk_dilate_nonnegative(
                sources, self._cost_inflation_radius_m, res
            )
        out = cost_field
        out[obstacle_footprint] = -1.0
        return out


def fused_enemy_belief_cost_field(
    gridmap: GridMap,
    per_landmark: dict,
    *,
    enemy_landmark_prefix: str,
    peak_cost: float = 500.0,
    n_sigma: float = 2.0,
    cov_regularization: float = 1e-8,
    enemy_ground_truth_xy: np.ndarray | None = None,
    fallback_position_variance_m2: float = 0.35**2,
    min_fused_cov_eigen_m2: float | None = None,
    robot_isotropic_inflate_m2: float | None = None,
) -> np.ndarray:
    """
    Per-cell traversal cost (float64): **anisotropic** 2σ Mahalanobis stamp from each enemy.

    Robot-size inflation is **covariance isotropic**: ``P ← P + σ_r² I`` with
    ``σ_r² = robot_isotropic_inflate_m2`` (caller sets e.g. ``(k × robot_collision_radius)²``) before the 2σ
    field is evaluated. That widens the ellipse along its **principal axes** (adds uncertainty
    isotropically) instead of morphological max‑disk dilation, which turns every hazard into a
    circle.

    When ``enemy_ground_truth_xy`` is provided (shape ``(N, 2)``), every enemy index ``0 … N-1``
    gets a layer: fused mean/covariance when scouts fused that track, otherwise ground truth with
    ``fallback_position_variance_m2``. Fused covariances are eigenvalue‑floored at
    ``min_fused_cov_eigen_m2`` (default ~ a few grid cells) so the 2σ footprint is never
    sub‑resolution. Hard obstacles (``gridmap.data < 0``) stay ``-1``.
    """
    base = np.asarray(gridmap.data, dtype=np.float64)
    h, w = base.shape
    obstacle = base < 0.0
    out = np.where(obstacle, -1.0, 0.0)
    x_row, _y_grid = gridmap.cell_centers_xy()
    x_grid = np.broadcast_to(x_row[np.newaxis, :], (h, w))
    y_grid = _y_grid
    n_sig = max(float(n_sigma), 1e-9)
    peak = float(peak_cost)
    reg = float(cov_regularization) * np.eye(2, dtype=np.float64)
    res = float(gridmap.resolution)
    lam_floor = float(min_fused_cov_eigen_m2) if min_fused_cov_eigen_m2 is not None else (2.5 * res) ** 2
    r2 = float(robot_isotropic_inflate_m2) if robot_isotropic_inflate_m2 is not None else 0.0
    if r2 < 0.0:
        r2 = 0.0

    def stamp_layer(mu: np.ndarray, p_mat: np.ndarray) -> None:
        nonlocal out
        p2 = 0.5 * (p_mat + p_mat.T) + reg
        if r2 > 0.0:
            p2 = p2 + r2 * np.eye(2, dtype=np.float64)
        inv_p = np.linalg.inv(p2)
        dx = x_grid - float(mu[0])
        dy = y_grid - float(mu[1])
        d2 = inv_p[0, 0] * dx * dx + 2.0 * inv_p[0, 1] * dx * dy + inv_p[1, 1] * dy * dy
        d2 = np.clip(d2, 0.0, None)
        d = np.sqrt(d2)
        wgt = np.clip(1.0 - (d / n_sig) ** 2, 0.0, 1.0)
        layer = peak * (wgt**2)
        out = np.where(obstacle, -1.0, out + layer)

    if enemy_ground_truth_xy is not None:
        gt = np.asarray(enemy_ground_truth_xy, dtype=np.float64).reshape(-1, 2)
        p_fb = np.eye(2, dtype=np.float64) * float(fallback_position_variance_m2)
        for ei in range(gt.shape[0]):
            key = f"{enemy_landmark_prefix}{ei}"
            stats = per_landmark.get(key, {})
            if int(stats.get("n_sources", 0) or 0) > 0:
                fused_xy = stats.get("fused_xy_m")
                fused_cov = stats.get("fused_cov_m2")
                if fused_xy is None or fused_cov is None:
                    stamp_layer(gt[ei], p_fb)
                    continue
                mu = np.asarray(fused_xy, dtype=np.float64).reshape(2)
                p = np.asarray(fused_cov, dtype=np.float64).reshape(2, 2)
                p = 0.5 * (p + p.T)
                lam, vecs = np.linalg.eigh(p)
                lam = np.maximum(lam, lam_floor)
                p_eff = vecs @ np.diag(lam) @ vecs.T
                stamp_layer(mu, p_eff)
            else:
                stamp_layer(gt[ei], p_fb)
    else:
        for lm_type, stats in per_landmark.items():
            if not str(lm_type).startswith(enemy_landmark_prefix):
                continue
            if int(stats.get("n_sources", 0) or 0) <= 0:
                continue
            fused_xy = stats.get("fused_xy_m")
            fused_cov = stats.get("fused_cov_m2")
            if fused_xy is None or fused_cov is None:
                continue
            mu = np.asarray(fused_xy, dtype=np.float64).reshape(2)
            p = np.asarray(fused_cov, dtype=np.float64).reshape(2, 2)
            p = 0.5 * (p + p.T)
            lam, vecs = np.linalg.eigh(p)
            lam = np.maximum(lam, lam_floor)
            p_eff = vecs @ np.diag(lam) @ vecs.T
            stamp_layer(mu, p_eff)
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
