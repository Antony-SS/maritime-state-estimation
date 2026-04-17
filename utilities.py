import numpy as np
import matplotlib.patches as mpatches

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