"""
This file contains the code to generate the map of the maritime environment, including:

- islands; static enemy ships (former buoy sites are included in ``ENEMY_COORDINATES``)
- lighthouses
- gps denied areas
- adversarial things to map

"""

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import rps.robotarium as robotarium
from typing import List, Dict, Optional, Sequence

from gridmap import GridMap
from rps.robotarium_abc import ARobotarium

from utilities import AStarPathPlanner, _point_xy_at_arclength, fused_enemy_belief_cost_field

class Map:
    """
    ENU (East, North, Up) coordinate system.
    x is east (positive to the right)
    y is north (positive up)
    """

    # GRIDMAP PARAMETERS
    BOUNDARIES: np.ndarray = np.array([-1.6, 1.6, -1.0, 1.0])
    RESOLUTION: float = 0.01

    # Buoys removed from the scenario; ``BUOY_COORDINATES`` is empty (``self.buoys`` stays a dict for API).
    BUOY_COORDINATES: list[tuple[float, float]] = []
    BUOY_RADIUS_M: float = 0.01
    BUOY_COLOR = "#ffcc33"
    BUOY_EDGE_COLOR = "none"

    # LIGHTHOUSES
    LIGHTHOUSE_COORDINATES: list[tuple[float, float]] = [(-0.375, -0.2), (0.5, 0.55)]
    LIGHTHOUSE_RADIUS_M = 0.04
    LIGHTHOUSE_COLOR = "#ffffff"
    # Simulated bearing/range noise variances (rad², m²); also used as EKF measurement covariance R.
    LIGHTHOUSE_BEARING_MEAS_VAR_RAD2: float = 0.04**2
    LIGHTHOUSE_RANGE_MEAS_VAR_M2: float = 0.08**2
    # Tiered sighting (m): ``d`` = ground range to beacon; ``d_min`` = ``LIGHTHOUSE_MEASUREMENT_MIN_RANGE_M``.
    # Bearing is usable farther out than range. ``d_min < d < BEARING_AND_RANGE_MAX`` → bearing + range;
    # ``BEARING_AND_RANGE_MAX <= d < BEARING_SIGHTING_MAX`` → bearing only; otherwise no measurement.
    # Require ``BEARING_AND_RANGE_MAX_M <= BEARING_SIGHTING_MAX_M``.
    LIGHTHOUSE_MEASUREMENT_MIN_RANGE_M: float = 0.08
    LIGHTHOUSE_BEARING_AND_RANGE_MAX_M: float = 0.6
    LIGHTHOUSE_BEARING_SIGHTING_MAX_M: float = 0.75

    # Robot spawn poses (ENU m, heading rad): one row per robot; row 0 = mothership. Shape (N, 3);
    # transpose to (3, N) for Robotarium ``initial_conditions``.
    START_POINTS: np.ndarray = np.array(
        [
            [-1.5, 0.90, 0.0],  # Mothership
            [-1.5, 0.30, 0.0],  # Scout 1
            [-.95, 0.325, 0.0],  # Scout 2
            [-1.2, 0.70, 0.0],  # Scout 3
        ],
        dtype=np.float64,
    )

    # ENEMY SHIPS/BASES (ASSUMED TO BE STATIC): original enemies plus former buoy positions.
    ENEMY_COORDINATES: list[tuple[float, float]] = [
        (-1.3, -0.7),
        (-0.1, 0.6),
        (0.0, 0.0),
        (0.6, -0.7),
        (0.1, 0.3),
        (-1.0, 0.1),
        (0.0, -0.325),
        (0.9, 0.25),
        (0.9, -0.55),
    ]
    ENEMY_RADIUS_M: float = 0.03
    ENEMY_COLOR = "#ff4545"  # bright red (projector / small markers)

    # Enemy hazard stamp: ``σ_r = NAV_ENEMY_INFLATION_RADIUS_MULT × (COLLISION_DIAMETER / 2)`` in
    # ``P + σ_r² I`` (see :meth:`apply_fused_enemy_costs_for_nav_planning`). Larger ``σ_r`` vs the
    # belief's eigenvalues makes the 2σ footprint look rounder in world space even when ``P`` is
    # elongated (isotropic inflation dominates anisotropy).
    NAV_ENEMY_INFLATION_RADIUS_MULT: float = 1.75
    
    # LAND MASS PARAMETERS
    LAND_MASS_DATA_PATH: str = "./map_a.npy"
    # Brighter than default for Robotarium projectors (readability in a lit room).
    LAND_MASS_COLOR: str = "#c5fbc8"

    # OCEAN PARAMETERS
    OCEAN_COLOR = "#c4f3ff"
    OCEAN_EDGE_COLOR = "none"

    # GRIDMAP COLOR KEY
    GRIDMAP_COLOR_KEY: dict[int, str] = {
        -1: LAND_MASS_COLOR, # hard obstacle (land mass)
    }

    # Matplotlib z-order: Robotarium robot patches are redrawn at ``zorder=2`` each step, so every
    # decoration added here must stay **strictly below 2** or robots disappear under the map.
    VIZ_Z_ENEMY_COST_HEAT = 0.12
    VIZ_Z_MAINLAND_LABEL = 0.28
    VIZ_Z_ENV_ICON = 0.48  # lighthouses, enemy ship markers
    # ``experiment.py`` overlays (same <2 rule; keep ordering sensible: trails → LM → pose → fused).
    VIZ_Z_GT_KF_TRAIL = 0.92
    VIZ_Z_LM_COV = 1.05
    VIZ_Z_POSE_COV = 1.15
    VIZ_Z_FUSED_LM = 1.32
    VIZ_Z_DEATH_MARKER = 1.70

    def __init__(self, r: robotarium.Robotarium, land_mass_data_path: str = LAND_MASS_DATA_PATH):
        self._r = r
        self.gridmap = GridMap.empty(self.BOUNDARIES, self.RESOLUTION)
        self.buoys = self._generate_buoys()
        self.lighthouses = self._generate_lighthouses()
        self._buoy_patches: List[patches.Circle] = []
        self._lighthouse_patches: List[patches.Circle] = []
        self._enemy_patches: List[patches.RegularPolygon] = []
        self._overlap_im = None
        self._enemy_inflated_risk_im = None
        self._enemy_stamped_positive: np.ndarray | None = None
        self._nav_cost_inflation_radius_m: float | None = None
        self.LAND_MASS_DATA_PATH = land_mass_data_path
        self.land_mass_data = np.load(self.LAND_MASS_DATA_PATH)

        if self._r.show_figure:
            self._draw_ocean_background()
            self._add_land_mass_to_gridmap()
            self._draw_gridmap(self.GRIDMAP_COLOR_KEY)
            if self.buoys:
                self._draw_buoys()
            self._draw_lighthouses()
            self._draw_enemies()
            self._draw_mainland_label()

    def _add_land_mass_to_gridmap(self) -> None:
        """Add the land mass to the gridmap."""
        self.gridmap.data[self.land_mass_data > 0] = -1 # hard obstacle to avoid path planning through land masses

    def _generate_buoys(self) -> Dict[str, np.ndarray]:
        """Place buoys at ``BUOY_COORDINATES`` (ENU, metres); keys ``B1``, ``B2``, … in list order."""
        return {
            f"B{i + 1}": np.asarray(xy, dtype=np.float64).reshape(2)
            for i, xy in enumerate(self.BUOY_COORDINATES)
        }

    def _generate_lighthouses(self) -> Dict[str, np.ndarray]:
        """Known beacon positions at ``LIGHTHOUSE_COORDINATES``; keys ``L1``, ``L2``, … in list order."""
        return {
            f"L{i + 1}": np.asarray(xy, dtype=np.float64).reshape(2)
            for i, xy in enumerate(self.LIGHTHOUSE_COORDINATES)
        }

    def _draw_ocean_background(self) -> None:
        """Fill the arena with a blue rectangle behind everything else."""
        if not self._r.show_figure:
            return
        ax = self._r._axes_handle
        b = self.BOUNDARIES
        bg = patches.Rectangle(
            (b[0], b[2]),
            b[1] - b[0],
            b[3] - b[2],
            facecolor=self.OCEAN_COLOR,
            edgecolor=self.OCEAN_EDGE_COLOR,
            zorder=-1,
        )
        ax.add_patch(bg)

    def _draw_gridmap(self, color_key: dict[int, str]) -> None:
        """Draw the gridmap with the color key. """
        if not self._r.show_figure:
            return
        ax = self._r._axes_handle
        g = self.gridmap.data
        h, w = g.shape
        rgba = np.zeros((h, w, 4), dtype=float)
        rgba[..., :] = (1.0, 1.0, 1.0, 0.0)
        for val, color in color_key.items():
            rgba[g == val] = mcolors.to_rgba(color)
        extent = self.gridmap.extent_xy
        self._gridmap_im = ax.imshow(
            np.flipud(rgba),
            extent=extent,
            origin="lower",
            interpolation="nearest",
            zorder=0,
        )

    def draw_overlap_cells(self) -> None:
        """Occupied cells (count >= 1) tinted; count > 1 drawn black (overlap)."""
        if not self._r.show_figure:
            return
        ax = self._r._axes_handle
        g = self.gridmap.data
        h, w = g.shape
        rgba = np.zeros((h, w, 4), dtype=float)
        rgba[..., :] = (1.0, 1.0, 1.0, 0.0)
        occupied = g > 0
        rgba[occupied] = (0.95, 0.45, 0.15, 0.55)
        rgba[g > 1] = (0.0, 0.0, 0.0, 1.0)
        # ``extent`` must match :class:`GridMap` indexing (cell size ``RESOLUTION``), not only
        # ``BOUNDARIES``, so occupancy lines up with world metres on the Robotarium axes.
        extent = self.gridmap.extent_xy
        # ``data[v=0]`` is the northern row; ``imshow`` with ``origin="lower"`` puts row 0 at
        # ``extent`` bottom, so flip vertically to align RGBA with ENU on the Robotarium axes.
        self._overlap_im = ax.imshow(
            np.flipud(rgba),
            extent=extent,
            origin="lower",
            interpolation="nearest",
            zorder=0,
        )

    def _draw_buoys(self) -> None:
        """Draw buoys as small circles in world coordinates on the Robotarium axes."""
        if not self._r.show_figure:
            return
        ax = self._r._axes_handle
        for buoy in self.buoys.values():
            xy = np.asarray(buoy, dtype=float).reshape(2,)
            c = patches.Circle(
                (xy[0], xy[1]),
                radius=self.BUOY_RADIUS_M,
                facecolor=self.BUOY_COLOR,
                edgecolor=self.BUOY_EDGE_COLOR,
                linewidth=0.0,
                zorder=float(self.VIZ_Z_ENV_ICON),
            )
            ax.add_patch(c)
            self._buoy_patches.append(c)

    def _draw_lighthouses(self) -> None:
        """Draw lighthouses as white circles in world coordinates."""
        if not self._r.show_figure:
            return
        ax = self._r._axes_handle
        r_lh = float(self.LIGHTHOUSE_RADIUS_M)
        for _, pos in self.lighthouses.items():
            p = np.asarray(pos, dtype=float).reshape(2)
            c = patches.Circle(
                (float(p[0]), float(p[1])),
                radius=r_lh,
                facecolor=self.LIGHTHOUSE_COLOR,
                edgecolor="none",
                linewidth=0.0,
                zorder=float(self.VIZ_Z_ENV_ICON),
            )
            ax.add_patch(c)
            self._lighthouse_patches.append(c)

    def _draw_mainland_label(self) -> None:
        """Annotate the land region for viewers (axes upper-right, over the landmass)."""
        if not self._r.show_figure:
            return
        ax = self._r._axes_handle
        ax.text(
            0.82,
            0.78,
            "Mainland",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=15,
            fontweight="semibold",
            color="#0d3d0d",
            zorder=float(self.VIZ_Z_MAINLAND_LABEL),
        )

    def _draw_enemies(self) -> None:
        """Draw static enemy bases/ships as red triangles at ``ENEMY_COORDINATES`` (apex toward +y)."""
        if not self._r.show_figure:
            return
        ax = self._r._axes_handle
        r = float(self.ENEMY_RADIUS_M)
        for xy in self.ENEMY_COORDINATES:
            p = np.asarray(xy, dtype=float).reshape(2)
            tri = patches.RegularPolygon(
                (float(p[0]), float(p[1])),
                numVertices=3,
                radius=r,
                orientation=np.pi / 2.0,
                facecolor=self.ENEMY_COLOR,
                edgecolor="none",
                linewidth=0.0,
                zorder=float(self.VIZ_Z_ENV_ICON),
            )
            ax.add_patch(tri)
            self._enemy_patches.append(tri)

    def apply_fused_enemy_costs_for_nav_planning(
        self,
        fusion_report: dict,
        router: "Router",
        *,
        enemy_landmark_prefix: str,
        enemy_ground_truth_xy: np.ndarray,
        peak_cost: float = 500.0,
        n_sigma: float = 2.0,
        nav_robot_radius_inflation_m: float | None = None,
    ) -> None:
        """
        Stamp **ellipse‑shaped** 2σ enemy costs using ``P + σ_r² I`` with
        ``σ_r = Map.NAV_ENEMY_INFLATION_RADIUS_MULT × (COLLISION_DIAMETER / 2)``. A* dilates **hard
        land** with the same ``σ_r``; cost heat in :meth:`refresh_planning_cost_visual` is masked
        off land.
        """
        per = fusion_report.get("per_landmark", {})
        gt = np.asarray(enemy_ground_truth_xy, dtype=np.float64).reshape(-1, 2)
        r_inf = (
            float(nav_robot_radius_inflation_m)
            if nav_robot_radius_inflation_m is not None
            else float(ARobotarium.COLLISION_DIAMETER)
            * 0.5
            * float(self.NAV_ENEMY_INFLATION_RADIUS_MULT)
        )
        inflate_m2 = float(r_inf) ** 2
        cost = fused_enemy_belief_cost_field(
            self.gridmap,
            per,
            enemy_landmark_prefix=enemy_landmark_prefix,
            peak_cost=peak_cost,
            n_sigma=n_sigma,
            enemy_ground_truth_xy=gt,
            robot_isotropic_inflate_m2=inflate_m2,
        )
        stamped_pos = np.where(cost < 0.0, 0.0, np.maximum(0.0, cost))
        self._enemy_stamped_positive = stamped_pos.astype(np.float64, copy=True)
        self.gridmap.data = cost
        self._nav_cost_inflation_radius_m = r_inf
        router.astar_planner = AStarPathPlanner(
            self.gridmap,
            cost_inflation_radius_m=0.0,
            obstacle_inflation_radius_m=r_inf,
        )

    def refresh_planning_cost_visual(self) -> None:
        """
        Discrete **land** underlay, then **enemy‑only** stamped costs (same ellipse field as A*,
        no isotropic cost dilation) with heat **masked off land** and drawn above the land layer.
        """
        if not self._r.show_figure:
            return
        ax = self._r._axes_handle
        h, w = int(self.land_mass_data.shape[0]), int(self.land_mass_data.shape[1])
        extent = self.gridmap.extent_xy
        land = self.land_mass_data > 0
        rgba_base = np.zeros((h, w, 4), dtype=float)
        rgba_base[land] = mcolors.to_rgba(self.LAND_MASS_COLOR)
        rgba_base[~land] = (1.0, 1.0, 1.0, 0.0)

        if getattr(self, "_gridmap_im", None) is not None:
            self._gridmap_im.remove()
        self._gridmap_im = ax.imshow(
            np.flipud(rgba_base),
            extent=extent,
            origin="lower",
            interpolation="nearest",
            zorder=0,
        )

        if getattr(self, "_enemy_inflated_risk_im", None) is not None:
            self._enemy_inflated_risk_im.remove()
            self._enemy_inflated_risk_im = None

        stamped = self._enemy_stamped_positive
        if stamped is None:
            return
        heat = stamped.astype(np.float64, copy=True)
        heat[land] = 0.0
        cmax = float(np.max(heat))
        if cmax < 1e-18:
            return
        norm = np.clip(heat / cmax, 0.0, 1.0)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "enemy_risk",
            [(0.62, 0.91, 1.0, 1.0), (1.0, 0.75, 0.2, 1.0), (0.75, 0.1, 0.1, 1.0)],
        )
        rgba_h = cmap(norm)[..., :4]
        show = (~land) & (heat > 1e-9 * cmax)
        alpha = np.where(show, 0.88, 0.0).astype(np.float64)
        rgba_h[..., 3] = alpha
        self._enemy_inflated_risk_im = ax.imshow(
            np.flipud(rgba_h),
            extent=extent,
            origin="lower",
            interpolation="nearest",
            zorder=float(self.VIZ_Z_ENEMY_COST_HEAT),
        )


class Router:
    """Build stitched world polylines by running :class:`AStarPathPlanner` between ordered waypoints."""

    # Nominal routes can be offset into a lateral zigzag; A* stitches between offset samples.
    DEFAULT_LAWN_HALF_WIDTH_M = 0.15  # lateral offset from nominal (± this, alternating)
    DEFAULT_LAWN_STEP_M = 0.3  # resample nominal every this many metres along arc length (smaller = finer)

    def __init__(self, map: Map):
        self.map = map
        self.astar_planner = AStarPathPlanner(self.map.gridmap)

    @staticmethod
    def _resample_polyline_arclength(xy: np.ndarray, step_m: float) -> np.ndarray:
        """Polyline ``(L, 2)`` resampled at roughly uniform arc-length spacing ``step_m`` (includes ends)."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        if xy.shape[0] <= 1 or step_m <= 0.0:
            return xy.copy()
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        total = float(np.sum(seg))
        if total < 1e-9:
            return xy[:1].copy()
        step_m = max(float(step_m), 1e-4)
        s_vals = [0.0]
        s = step_m
        while s < total - 1e-9:
            s_vals.append(s)
            s += step_m
        if s_vals[-1] < total - 1e-6:
            s_vals.append(total)
        return np.stack([_point_xy_at_arclength(xy, float(s)) for s in s_vals], axis=0)

    @staticmethod
    def _normals_along_polyline(centers: np.ndarray) -> np.ndarray:
        """Unit left normals (ENU) at each vertex, from chord directions."""
        c = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
        m = c.shape[0]
        nrm = np.zeros_like(c)
        for i in range(m):
            if i == 0 and m > 1:
                d = c[1] - c[0]
            elif i == m - 1 and m > 1:
                d = c[-1] - c[-2]
            else:
                d = c[i + 1] - c[i - 1]
            ln = float(np.linalg.norm(d))
            if ln < 1e-12:
                nrm[i] = np.array([0.0, 1.0], dtype=np.float64)
            else:
                t = d / ln
                nrm[i] = np.array([-t[1], t[0]], dtype=np.float64)
        return nrm

    def _xy_inflated_navigable(self, xy: np.ndarray) -> bool:
        """True iff the grid cell under ``xy`` is traversable in A* (``inflated_map[v, u] >= 0``)."""
        p = np.asarray(xy, dtype=np.float64).reshape(2)
        gm = self.astar_planner.gridmap
        nav = self.astar_planner.inflated_map
        uv = np.asarray(gm.xy_to_uv(p), dtype=np.int64).reshape(2)
        u, v = int(uv[0]), int(uv[1])
        if u < 0 or u >= gm.width or v < 0 or v >= gm.height:
            return False
        return float(nav[v, u]) >= 0.0

    def _snap_xy_to_inflated_navigable(self, xy: np.ndarray, max_ring: int = 100) -> np.ndarray:
        """
        If ``xy`` lies in a blocked **inflated** cell, return the centre of the closest cell with
        ``inflated_map >= 0`` (Chebyshev search up to ``max_ring`` cells). Otherwise return ``xy``.
        """
        p = np.asarray(xy, dtype=np.float64).reshape(2)
        if self._xy_inflated_navigable(p):
            return p.copy()
        gm = self.astar_planner.gridmap
        nav = self.astar_planner.inflated_map
        uv0 = np.asarray(gm.xy_to_uv(p), dtype=np.int64).reshape(2)
        u0, v0 = int(uv0[0]), int(uv0[1])
        best: tuple[int, int, int] | None = None  # (dist2, u, v)
        for du in range(-max_ring, max_ring + 1):
            for dv in range(-max_ring, max_ring + 1):
                u, v = u0 + du, v0 + dv
                if u < 0 or u >= gm.width or v < 0 or v >= gm.height:
                    continue
                if float(nav[v, u]) < 0.0:
                    continue
                d2 = du * du + dv * dv
                if best is None or d2 < best[0]:
                    best = (d2, u, v)
        if best is None:
            return p.copy()
        return np.asarray(
            gm.uv_to_xy(np.array([float(best[1]), float(best[2])], dtype=np.float64)),
            dtype=np.float64,
        ).reshape(2)

    def _lawn_keypoints_from_nominal(
        self,
        nominal_xy: np.ndarray,
        lawn_half_width_m: float,
        lawn_step_m: float,
    ) -> np.ndarray:
        """
        Resample nominal, then push interior samples ± ``lawn_half_width_m`` along left normal
        (alternating). Endpoints stay on the nominal polyline. A lateral sample that falls in an
        **inflated** blocked cell (same mask as A*) is replaced by the on-nominal center so A* can
        still connect segments.
        """
        nominal_xy = np.asarray(nominal_xy, dtype=np.float64).reshape(-1, 2)
        if nominal_xy.shape[0] <= 1 or lawn_half_width_m <= 0.0 or lawn_step_m <= 0.0:
            return nominal_xy.copy()
        centers = self._resample_polyline_arclength(nominal_xy, lawn_step_m)
        m = centers.shape[0]
        if m <= 2:
            return centers.copy()
        normals = self._normals_along_polyline(centers)
        out = centers.copy()
        for i in range(1, m - 1):
            side = 1.0 if (i % 2 == 0) else -1.0
            cand = centers[i] + side * lawn_half_width_m * normals[i]
            out[i] = cand if self._xy_inflated_navigable(cand) else centers[i]
        return out

    def _apply_lawn_around_nominal(
        self,
        nominal_xy: np.ndarray,
        lawn_half_width_m: float,
        lawn_step_m: float,
        *,
        heuristic_weight: float = 1.0,
    ) -> np.ndarray:
        """Stitched A* through lateral zigzag keypoints; single-point routes unchanged."""
        nominal_xy = np.asarray(nominal_xy, dtype=np.float64).reshape(-1, 2)
        if nominal_xy.shape[0] <= 1:
            return nominal_xy.copy()
        if lawn_half_width_m <= 0.0 or lawn_step_m <= 0.0:
            return nominal_xy.copy()
        keys = self._lawn_keypoints_from_nominal(nominal_xy, lawn_half_width_m, lawn_step_m)
        return self.chain_astar(keys, heuristic_weight=heuristic_weight)

    def generate_scouting_phase_routes(
        self,
        initial_conditions: np.ndarray,
        *,
        lawn_half_width_m: float | None = None,
        lawn_step_m: float | None = None,
    ) -> List[np.ndarray]:
        """
        ``initial_conditions`` matches Robotarium: shape ``(3, N)`` with rows ``[x; y; θ]`` and
        column ``i`` = robot ``i`` (same as ``Map.START_POINTS.T``).

        After nominal A* routes are built, each multi-point route is replaced by a **lateral lawn**
        (zigzag about the nominal), then A*-stitched again so motion respects obstacles.

        Tunables (``None`` → class defaults):

        - ``lawn_half_width_m``: lateral offset magnitude; path alternates ``+/-`` this each sample.
        - ``lawn_step_m``: arc-length spacing along nominal between samples (smaller = finer / more A* segments).
        """
        half_w = (
            float(lawn_half_width_m)
            if lawn_half_width_m is not None
            else float(self.DEFAULT_LAWN_HALF_WIDTH_M)
        )
        step = (
            float(lawn_step_m) if lawn_step_m is not None else float(self.DEFAULT_LAWN_STEP_M)
        )
        ic = np.asarray(initial_conditions, dtype=np.float64)
        if ic.ndim != 2 or ic.shape[0] < 2:
            raise ValueError("initial_conditions must be 2D with at least 2 rows (x, y, …).")
        n = ic.shape[1]
        xy = lambda i: ic[:2, i].reshape(2)

        routes: List[np.ndarray] = [xy(0).reshape(1, 2)]
        if n <= 1:
            return routes

        end_goal = np.array([1.5, -0.35], dtype=np.float64)

        end_goal_s1 = np.array([1.4, -0.75], dtype=np.float64)

        ordered_s1 = np.vstack(
            (
                xy(1).reshape(1, 2),
                np.array(
                    [[-1.2, -0.7], [-0.7, -0.45], [-0.25, -0.7], [0.00, -0.45], [1.0, -0.7], end_goal_s1],
                    dtype=np.float64,
                ),
            )
        )

        end_goal_s2 = np.array([1.5, -0.45], dtype=np.float64)

        ordered_s2 = np.vstack(
            (
                xy(2).reshape(1, 2),
                np.array(
                    [[-0.75, -.2], [-0.35, 0.175], [0.15, -0.3], [0.4, -0.2], [0.8, -0.75], end_goal_s2],
                    dtype=np.float64,
                ),
            )
        )

        end_goal_s3 = np.array([1.4, 0.1], dtype=np.float64)

        ordered_s3 = np.vstack(
            (
                xy(3).reshape(1, 2),
                np.array(
                    [[-0.6, 0.6], [0.5, 0.25], [1.1, 0.1], end_goal_s3],
                    dtype=np.float64,
                ),
            )
        )

        routes.append(self.chain_astar(ordered_s1, heuristic_weight=1.0))
        routes.append(self.chain_astar(ordered_s2, heuristic_weight=1.0))
        routes.append(self.chain_astar(ordered_s3, heuristic_weight=1.0))

        return [
            self._apply_lawn_around_nominal(r, half_w, step, heuristic_weight=1.0) for r in routes
        ]

    @staticmethod
    def _dedupe_consecutive_xy(keys: np.ndarray, atol: float = 1e-7) -> np.ndarray:
        keys = np.asarray(keys, dtype=np.float64).reshape(-1, 2)
        if keys.shape[0] <= 1:
            return keys.copy()
        out = [keys[0]]
        for i in range(1, keys.shape[0]):
            if not np.allclose(keys[i], out[-1], atol=atol, rtol=0.0):
                out.append(keys[i])
        return np.stack(out, axis=0)

    def chain_astar(
        self,
        ordered_xy: np.ndarray,
        *,
        heuristic_weight: float = 1.0,
    ) -> np.ndarray:
        """
        ``ordered_xy`` is ``(K, 2)`` world points in visit order. Runs A* from row ``j`` to ``j+1``
        for each segment and concatenates (dropping duplicate junction vertices).
        """
        chain = self._dedupe_consecutive_xy(ordered_xy)
        k = chain.shape[0]
        if k <= 1:
            return chain.copy()
        chunks: List[np.ndarray] = []
        for j in range(k - 1):
            a_raw = chain[j].reshape(2)
            b_raw = chain[j + 1].reshape(2)
            a = self._snap_xy_to_inflated_navigable(a_raw)
            b = self._snap_xy_to_inflated_navigable(b_raw)
            seg = self.astar_planner.plan(a, b, heuristic_weight=heuristic_weight)
            if seg.size == 0:
                raise RuntimeError(
                    f"A* failed on segment {j}→{j + 1}: "
                    f"({a_raw[0]:.4f}, {a_raw[1]:.4f}) → ({b_raw[0]:.4f}, {b_raw[1]:.4f}) "
                    f"(snapped to inflated-free cell centres: "
                    f"({a[0]:.4f}, {a[1]:.4f}) → ({b[0]:.4f}, {b[1]:.4f}))."
                )
            chunks.append(np.asarray(seg, dtype=np.float64).reshape(-1, 2))
        out = chunks[0]
        for seg in chunks[1:]:
            if out.size and seg.size and np.allclose(out[-1], seg[0], atol=1e-8, rtol=0.0):
                out = np.vstack((out, seg[1:]))
            else:
                out = np.vstack((out, seg))
        return out

    def routes_through_waypoints(
        self,
        starts_xy: np.ndarray,
        waypoints_per_robot: Sequence[np.ndarray],
        *,
        heuristic_weight: float = 1.0,
    ) -> List[np.ndarray]:
        """
        For each robot ``i``, build one polyline: **start** = ``starts_xy[:, i]``, then visit that
        robot's rows in ``waypoints_per_robot[i]`` in order. Each consecutive pair is connected with
        A* and segments are concatenated.

        Parameters
        ----------
        starts_xy
            Shape ``(2, N)`` world ENU (m), column ``i`` = robot ``i`` start (e.g. poses from initial conditions).
        waypoints_per_robot
            Length ``N``. Element ``i`` is shape ``(K_i, 2)`` with **additional** goals (not repeating
            the start). Use ``(0, 2)`` if the robot should only hold at its **current** start pose.
        """
        s = np.asarray(starts_xy, dtype=np.float64).reshape(2, -1)
        n = s.shape[1]
        if len(waypoints_per_robot) != n:
            raise ValueError(
                f"waypoints_per_robot has length {len(waypoints_per_robot)}, expected N={n}."
            )
        routes: List[np.ndarray] = []
        for i in range(n):
            start = s[:, i].reshape(1, 2)
            wps = np.asarray(waypoints_per_robot[i], dtype=np.float64).reshape(-1, 2)
            if wps.size == 0:
                routes.append(start.copy())
                continue
            ordered = np.vstack((start, wps))
            try:
                routes.append(self.chain_astar(ordered, heuristic_weight=heuristic_weight))
            except RuntimeError as e:
                raise RuntimeError(f"robot {i}: {e}") from e
        return routes

