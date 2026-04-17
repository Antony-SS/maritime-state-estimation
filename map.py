"""
This file contains the code to generate the map of the maritime environment, including:

- islands/buoys
- lighthouses (tbd)
- gps denied areas
- adversarial things to map

"""

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import rps.robotarium as robotarium
from typing import List, Dict, Optional, Tuple

from gridmap import GridMap

class Map:
    """
    ENU (East, North, Up) coordinate system.
    x is east (positive to the right)
    y is north (positive up)
    """
    BOUNDARIES: np.ndarray = np.array([-1.6, 1.6, -1.0, 1.0])
    BUOY_DIST_FROM_CENTER_M = .5 # meters
    NUMBER_OF_BUOYS = 3
    RESOLUTION: float = 0.01

    # VISUALIZATION PARAMETERS

    # BUOY PARAMETERS
    BUOY_RADIUS_M: float = 0.025
    BUOY_COLOR = "#ffcc33"
    BUOY_EDGE_COLOR = "#cc8800"
    BUOY_LINEWIDTH = 1.0

    # OCEAN PARAMETERS
    OCEAN_COLOR = "#5dade2"
    OCEAN_EDGE_COLOR = "none"

    # LAND MASS PARAMETERS
    LAND_MASS_DATA_PATH: str = "./maritime-state-estimation/assets/map_a.npy"
    LAND_MASS_COLOR: str = "#8be88b"

    # Demo: random Gaussian blob of grid hits for visualization testing (set False to disable)
    DEMO_BLOB_AT_XY: Tuple[float, float] = (0.5, 0.5)
    DEMO_BLOB_N_POINTS: int = 450
    DEMO_BLOB_STD_M: float = 0.035

    def __init__(self, r: robotarium.Robotarium, land_mass_data_path: str = LAND_MASS_DATA_PATH):
        self._r = r
        self.gridmap = GridMap.empty(self.BOUNDARIES, self.RESOLUTION)
        self.buoys = self._generate_buoys()
        self._buoy_patches: List[patches.Circle] = []
        self._overlap_im = None
        self.LAND_MASS_DATA_PATH = land_mass_data_path
        self.land_mass_data = np.load(self.LAND_MASS_DATA_PATH)
        if self._r.show_figure:
            self._draw_ocean_background()
            self._draw_land_masses()
            self.draw_overlap_cells()
            self.draw_buoys()

    def add_random_xy_blob(
        self,
        center_xy: Tuple[float, float] = (0.5, 0.5),
        n_points: int = 400,
        std: float = 0.04,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Scatter `n_points` in a 2D Gaussian around `center_xy`; increment grid cells (clipped to map)."""
        rng = rng or np.random.default_rng()
        cx, cy = center_xy
        pts = rng.normal(loc=[cx, cy], scale=[std, std], size=(n_points, 2))
        uv = self.gridmap.xy_to_uv(pts)
        if uv.ndim == 1:
            uv = uv.reshape(1, -1)
        # xy_to_uv returns columns (u, v); grid rows index v, columns index u
        u = uv[:, 0].astype(np.int64, copy=False)
        v = uv[:, 1].astype(np.int64, copy=False)
        h, w = self.gridmap.data.shape
        mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v = u[mask], v[mask]
        np.add.at(self.gridmap.data, (v, u), 1.0)
    
    def _generate_buoys(self) -> Dict[str, np.ndarray]:
        """Generate the buoys in the environment based on the number of buoys and the radius of the buoys. """
        buoys = {}
        for i in range(self.NUMBER_OF_BUOYS):
            buoy_angle = 2 * np.pi * i / self.NUMBER_OF_BUOYS
            buoy_x = self.BUOY_DIST_FROM_CENTER_M * np.cos(buoy_angle)
            buoy_y = self.BUOY_DIST_FROM_CENTER_M * np.sin(buoy_angle)
            buoys[f"B{i+1}"] = np.array([buoy_x, buoy_y])
        return buoys

    def uv_to_xy(self, uv: np.ndarray) -> np.ndarray:
        return self.gridmap.uv_to_xy(uv)

    def xy_to_uv(self, xy: np.ndarray) -> np.ndarray:
        return self.gridmap.xy_to_uv(xy)

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

    def _draw_land_masses(self) -> None:
        """Draw the land mass in the environment based on the land mass data."""
        if not self._r.show_figure:
            return
        ax = self._r._axes_handle
        g = np.asarray(self.land_mass_data, dtype=np.float64)
        h, w = g.shape
        rgba = np.zeros((h, w, 4), dtype=np.float64)
        rgba[..., :] = (1.0, 1.0, 1.0, 0.0)
        # Hex / name must become float RGBA for imshow (do not assign the string to rgba).
        land_rgba = np.asarray(mcolors.to_rgba(self.LAND_MASS_COLOR), dtype=np.float64)
        rgba[g > 0] = land_rgba
 
        extent = self.gridmap.extent_xy
        self._land_mass_im = ax.imshow(
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

    def draw_buoys(self) -> None:
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
                linewidth=1.0,
                zorder=1,
            )
            ax.add_patch(c)
            self._buoy_patches.append(c)

