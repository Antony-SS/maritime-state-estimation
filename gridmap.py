"""
Discrete grid aligned to ENU world coordinates (Robotarium-style boundaries).

Rows index v (image/top-down); columns index u. Coordinate helpers match map.Map conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _grid_shape(boundaries: np.ndarray, resolution: float) -> tuple[int, int]:
    """
    Number of rows (v / north-up) and columns (u / east) so the grid covers the
    axis-aligned rectangle at ``resolution`` step, using ``round`` so common
    spans (e.g. Robotarium 3.2 m at 1 cm) are not off by one from float noise.
    """
    b = np.asarray(boundaries, dtype=float).reshape(4)
    xmin, xmax, ymin, ymax = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    span_x = xmax - xmin
    span_y = ymax - ymin
    res = float(resolution)
    width = max(1, int(np.round(span_x / res)))
    height = max(1, int(np.round(span_y / res)))
    return height, width


@dataclass
class GridMap:
    """
    Discrete occupancy grid. You may construct with only boundaries and resolution;
    ``data`` is optional and defaults to a zero array of the correct shape.

    Examples: ``GridMap(bounds, 0.01)``, ``GridMap.empty()``, or pass ``data=`` explicitly.
    """

    boundaries: np.ndarray
    resolution: float
    data: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.boundaries = np.asarray(self.boundaries, dtype=np.float64).reshape(4).copy()

        height, width = _grid_shape(self.boundaries, self.resolution)

        if self.data is None:
            self.data = np.zeros((height, width), dtype=np.int32)
        else:
            arr = np.asarray(self.data)
            if np.issubdtype(arr.dtype, np.floating):
                self.data = arr.astype(np.float64, copy=False)
            else:
                self.data = arr.astype(np.int32, copy=False)

        expected = (height, width)
        if self.data.shape != expected:
            raise ValueError(
                f"data shape {self.data.shape} != expected {expected} "
                f"for boundaries {self.boundaries} and resolution {self.resolution}"
            )

    @classmethod
    def empty(
        cls,
        boundaries: np.ndarray | None = None,
        resolution: float = 0.01,
    ) -> GridMap:
        if boundaries is None:
            boundaries = np.array([-1.6, 1.6, -1.0, 1.0], dtype=float)
        b = np.asarray(boundaries, dtype=float).reshape(4)
        return cls(boundaries=b, resolution=resolution)

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    def uv_to_xy(self, uv: np.ndarray) -> np.ndarray:
        """
        Cell **centers** in world ENU (m). ``uv`` rows are ``(u, v)`` with ``u`` easting
        column index and ``v`` row index (``v = 0`` is the northern row, ``y`` near ``y_max``).
        Same column order as :meth:`xy_to_uv` returns.
        """
        uv = np.asarray(uv, dtype=float)
        single = uv.ndim == 1
        if single:
            uv = uv.reshape(1, -1)
        u = uv[:, 0]
        v = uv[:, 1]
        b = np.asarray(self.boundaries, dtype=float).reshape(4)
        xmin, ymax = float(b[0]), float(b[3])
        res = float(self.resolution)
        x = xmin + (u + 0.5) * res
        y = ymax - (v + 0.5) * res
        out = np.stack([x, y], axis=1)
        return out[0] if single else out

    def xy_to_uv(self, xy: np.ndarray) -> np.ndarray:
        """World ``(x, y)`` → grid indices ``(u, v)`` (floored). Out-of-range values are not clipped."""
        xy = np.asarray(xy, dtype=float)
        single = xy.ndim == 1
        if single:
            xy = xy.reshape(1, -1)
        x = xy[:, 0]
        y = xy[:, 1]
        b = np.asarray(self.boundaries, dtype=float).reshape(4)
        xmin, ymax = float(b[0]), float(b[3])
        res = float(self.resolution)
        u = np.floor((x - xmin) / res).astype(np.int64)
        v = np.floor((ymax - y) / res).astype(np.int64)
        out = np.stack([u, v], axis=1)
        return out[0] if single else out

    def to_binary(self) -> np.ndarray:
        """True where the cell has any non-zero occupancy/count."""
        return self.data != 0

    @property
    def extent_xy(self) -> tuple[float, float, float, float]:
        """
        ``extent`` for ``imshow``: ``(left, right, bottom, top)`` = ``(x_min, x_max, y_min, y_max)``
        of the **grid footprint** (``width * resolution`` by ``height * resolution``),
        aligned with :meth:`xy_to_uv` / cell centers so pixels match world metres.
        """
        b = np.asarray(self.boundaries, dtype=float).reshape(4)
        xmin, ymax = float(b[0]), float(b[3])
        h, w = self.data.shape
        res = float(self.resolution)
        x_max = xmin + w * res
        y_min = ymax - h * res
        return (xmin, x_max, y_min, ymax)

    def cell_centers_xy(self) -> tuple[np.ndarray, np.ndarray]:
        """1D ``x`` centers shape ``(width,)`` and 2D ``y`` centers shape ``(height, width)``."""
        b = np.asarray(self.boundaries, dtype=float).reshape(4)
        xmin, ymax = float(b[0]), float(b[3])
        h, w = self.data.shape
        res = float(self.resolution)
        u = np.arange(w, dtype=float)
        v = np.arange(h, dtype=float)
        x_centers = xmin + (u + 0.5) * res
        y_centers = ymax - (v[:, None] + 0.5) * res
        return x_centers, y_centers

    def _cell_mesh_edges_xy(self) -> tuple[np.ndarray, np.ndarray]:
        """
        1D easting edges (length ``width + 1``) and northing edges (length ``height + 1``),
        increasing in ``x`` and ``y``, for ``pcolormesh`` so **columns map to x** and **rows to y**.
        """
        b = np.asarray(self.boundaries, dtype=float).reshape(4)
        xmin, ymax = float(b[0]), float(b[3])
        nrows, ncols = self.data.shape
        res = float(self.resolution)
        y_min = ymax - nrows * res
        xe = xmin + np.arange(ncols + 1, dtype=np.float64) * res
        ye = y_min + np.arange(nrows + 1, dtype=np.float64) * res
        return xe, ye

    def visualize_gridmap(
        self,
        binary: bool = False,
        value_colors: Mapping[Any, str] | None = None,
        *,
        ax: Any | None = None,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
        default_unmapped_color: str = "#e8e8e8",
        vmin: float | None = None,
        vmax: float | None = None,
        interpolation: str = "nearest",
        show: bool = True,
        imshow_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        """
        Matplotlib figure for this grid (independent of Robotarium).

        Modes (first match wins):

        - ``binary=True``: white = free, black = occupied (non-zero ``data``).
        - ``value_colors`` non-empty: each cell value maps to a color (hex/CSS); unmapped
          cells use ``default_unmapped_color``.
        - Otherwise: grayscale uint8 image (occupancy-style), scaling ``data`` to 0–255
          using ``vmin``/``vmax`` (default: min/max of ``data``).

        Binary and grayscale use ``pcolormesh`` so **x is easting (horizontal)** and **y is
        northing (vertical)**, matching Robotarium axes. New figures default to the same
        physical aspect ratio as ``Robotarium._initialize_visualization`` (wider in ``x``).
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgba

        extra = dict(imshow_kwargs or {})
        extent = self.extent_xy
        b = np.asarray(self.boundaries, dtype=np.float64).reshape(4)

        created_fig = ax is None
        if ax is None:
            if figsize is None:
                scale = 2.5
                figsize = (
                    float(b[1] - b[0]) * scale,
                    float(b[3] - b[2]) * scale,
                )
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        nrows, ncols = self.data.shape
        d = self.data
        xe, ye = self._cell_mesh_edges_xy()

        if binary:
            # Row v=0 is north (high y); pcolormesh row 0 is low y → flip vertically.
            z = np.flipud(self.to_binary().astype(np.float32))
            pm_extras = {k: v for k, v in extra.items() if k not in {"cmap", "vmin", "vmax", "interpolation"}}
            im = ax.pcolormesh(
                xe,
                ye,
                z,
                cmap="gray_r",
                vmin=0.0,
                vmax=1.0,
                shading="flat",
                **pm_extras,
            )
        elif value_colors:
            rgba = np.tile(to_rgba(default_unmapped_color), (nrows, ncols, 1))
            for val, hexcol in value_colors.items():
                if np.issubdtype(d.dtype, np.integer):
                    mask = d == np.asarray(val, dtype=d.dtype)
                else:
                    mask = np.isclose(d, float(val), rtol=0.0, atol=1e-12)
                rgba[mask] = to_rgba(hexcol)
            im = ax.imshow(
                np.flipud(rgba),
                extent=extent,
                origin="lower",
                interpolation=interpolation,
                **{k: v for k, v in extra.items() if k not in {"extent", "origin", "interpolation"}},
            )
        else:
            obs = d == -1
            if np.any(obs):
                d_vis = np.where(obs, np.nan, d.astype(np.float64))
                lo = (
                    float(np.nanmin(d_vis))
                    if vmin is None
                    else float(vmin)
                )
                hi = (
                    float(np.nanmax(d_vis))
                    if vmax is None
                    else float(vmax)
                )
                if (
                    hi <= lo
                    or not np.isfinite(lo)
                    or not np.isfinite(hi)
                    or np.all(obs)
                ):
                    scaled = np.zeros_like(d, dtype=np.float64)
                else:
                    scaled = np.clip((d.astype(np.float64) - lo) / (hi - lo) * 255.0, 0.0, 255.0)
                scaled = np.where(obs, np.nan, scaled)
                gray_rgb = np.stack([scaled] * 3, axis=-1) / 255.0
                gray_rgb = np.nan_to_num(gray_rgb, nan=0.0)
                rgba = np.concatenate([gray_rgb, np.ones((*d.shape, 1), dtype=np.float64)], axis=-1)
                rgba[obs] = to_rgba("#c0392b")
                im = ax.imshow(
                    np.flipud(rgba),
                    extent=extent,
                    origin="lower",
                    interpolation=interpolation,
                    **{k: v for k, v in extra.items() if k not in {"extent", "origin", "interpolation"}},
                )
            else:
                lo = float(np.min(d)) if vmin is None else float(vmin)
                hi = float(np.max(d)) if vmax is None else float(vmax)
                if hi <= lo:
                    scaled = np.zeros_like(d, dtype=np.uint8)
                else:
                    scaled = np.clip((d - lo) / (hi - lo) * 255.0, 0.0, 255.0).astype(np.uint8)
                z = np.flipud(scaled.astype(np.float32))
                pm_extras = {k: v for k, v in extra.items() if k not in {"cmap", "vmin", "vmax", "interpolation"}}
                im = ax.pcolormesh(
                    xe,
                    ye,
                    z,
                    cmap="gray",
                    vmin=0,
                    vmax=255,
                    shading="flat",
                    **pm_extras,
                )

        ax.set_xlim(float(b[0]), float(b[1]))
        ax.set_ylim(float(b[2]), float(b[3]))
        ax.set_aspect("equal")
        if title is not None:
            ax.set_title(title)
        ax.set_xlabel("x (east, m)")
        ax.set_ylabel("y (north, m)")
        if show and created_fig:
            plt.show()
        return fig, ax
