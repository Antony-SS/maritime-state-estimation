"""
Maritime-style experiment: multi-robot scouting polylines, two parallel decentralized UnicycleEKFs
per scout (vanilla pose SLAM on the figure + parameter-augmented SLAM logged offline; robot 0 is a
mothership with assumed perfect GNSS — no filter, SLAM, or RMSE on that agent),
tiered lighthouse updates (bearing + range when close, bearing-only farther out; see ``Map``),
and SLAM-style bearing/range to static enemy ships when nearby (unknown landmarks).

After scouting, a **map fusion** phase fuses landmark tracks across scouts, applies the fused beliefs
back into each EKF, stamps each enemy with a **2σ anisotropic** cost field (ground-truth fallback when
unfused) using ``P + σ_r² I`` with ``σ_r = Map.NAV_ENEMY_INFLATION_RADIUS_MULT ×`` robot collision radius,
dilates **hard land** only for A*, heat‑visualizes enemy costs **masked off land**, then a **Nav phase**
runs: the mothership A* plans on the updated map to ``MOTHERSHIP_NAV_GOAL_XY`` while scouts hold at
their scout endpoints; lighthouse / landmark measurements are paused during Nav.

Use ``--save-offline [DIR]`` to write NPZ bundles (poses, encoders, ``get_orientations()`` headings,
simulated lighthouse / enemy / buoy events with ``sim_step`` and ``wall_time_unix_s``, EKF trajectories,
and metadata) for offline analysis. Each run uses **two** scout EKFs in parallel (same measurements):
a vanilla pose-only filter (shown on the Robotarium figure) and a parameter-augmented filter (logged).
``predicted_paths/`` includes both trajectories, landmark snapshots for both, per-scout kinematic
estimates from the augmented filter, and **pose/parameter variance diagonals only** (no landmark
covariance). A PNG time-series of kinematic estimates is saved right after map fusion. Use ``--skip-nav``
to skip the nav ``while`` loop; scout EKF and
GT/KF/landmark/offline logging stop after fusion (mothership does not run A* nav in that mode).
Optional ``FILTER_EKF_PRIOR_*_M`` constants at top of this file override only the EKF / plot priors
(simulator ``r.*`` still used for encoders and ``v_hat``/``w_hat`` logs).

Optional ``--simulate-death``: during scouting, robot 2 freezes at its pose and a large × is drawn
if it passes within ``SIMULATE_DEATH_ENEMY_PROXIMITY_M`` of a true enemy ship (visualization only).
"""
from __future__ import annotations

import argparse
import json
import secrets
import sys
from pathlib import Path
import time
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import rps.robotarium as robotarium
from filters import UnicycleEKF, wrap_angle
from rps.utilities.barrier_certificates import create_uni_barrier_certificate
from rps.utilities.controllers import create_uni_position_controller
from rps.utilities.misc import create_at_position, determine_font_size
from utilities import (
    _xy_cov_ellipse_width_height_angle_deg,
    _closest_point_on_polyline_xy_forward,
    _point_xy_at_arclength,
)

# Allow `from map import Map` when running from this directory
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from map import Map, Router  # noqa: E402

# Landmark SLAM: pass **one unique string per world object** into ``process_landmark_update`` so each
# robot keeps a single track per enemy (e.g. ``enemy ship:0``). Visualization
# colors use these prefixes.
_LM_PREFIX_BUOY = "buoy:"
_LM_PREFIX_ENEMY_SHIP = "enemy ship:"

# Assumed perfect GNSS: no EKF / SLAM / RMSE for this Robotarium index (mothership).
MOTHERSHIP_INDEX = 0
# Post–map-fusion navigation goal for the mothership (ENU, m).
MOTHERSHIP_NAV_GOAL_XY = np.array([1.35, -0.225], dtype=np.float64)
# Optional viz-only: scout ``2`` “dies” (stops + ×) when this close to any true enemy (scouting only).
SIMULATE_DEATH_SCOUT_INDEX = 2
SIMULATE_DEATH_ENEMY_PROXIMITY_M = 0.05

# Arc-length lookahead (m) from forward projection on the polyline.
PATH_LOOKAHEAD_DISTANCE_M = 0.11
# Forward projection slack for self-overlapping paths (e.g. lawn wiggle).
PATH_TRACK_SLACK_BACK_M = 0.22
# Original experiment cap (~gentle); bump slightly if you want faster transits.
PATH_TRACK_VEL_LIMIT_M_S = 0.09
# If the robot is within this distance of an upcoming vertex, advance arc-length to that vertex
# (helps tight A* corners where projection alone creeps slowly — e.g. scout 1).
PATH_VERTEX_CATCHUP_M = 0.065

# EKF kinematic prior overrides (optional). ``None`` = use ``Robotarium`` values from ``r``.
# Set to a float when the simulator constants differ but the filter should still start from the
# old nominal (e.g. ``FILTER_EKF_PRIOR_BASE_LENGTH_M = 0.11`` while ``ARobotarium.BASE_LENGTH`` is 0.14).
FILTER_EKF_PRIOR_WHEEL_RADIUS_M: float | None = None
FILTER_EKF_PRIOR_BASE_LENGTH_M: float | None = None

# Distinct colors for GT / KF trails and covariance ellipses (Robotarium default N is small).
_ROBOT_COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown")
# Brighter hues for dashed EKF trails (easier on a dim projector).
_ROBOT_KF_COLORS = (
    "#4db8ff",
    "#ffb84d",
    "#5ee85e",
    "#ff7070",
    "#d896ff",
    "#e0a060",
)
# Landmark / fused enemy outline (brighter than dark crimson #922b21).
_VIZ_ENEMY_LM_EDGE = "#ff5050"



def _ekf_landmark_xy_cov(
    ekf: UnicycleEKF, landmark_type: str
) -> tuple[np.ndarray, np.ndarray] | None:
    """Marginal (x, y) and 2×2 covariance for ``landmark_type`` in one EKF, or ``None`` if absent."""
    for idx, typ in ekf.landmarks.items():
        if str(typ) != landmark_type:
            continue
        b0 = int(ekf.base_state_size)
        sl = slice(b0 + 2 * int(idx), b0 + 2 * int(idx) + 2)
        x = np.asarray(ekf.state[sl], dtype=np.float64).reshape(2)
        P_full = np.asarray(ekf.P, dtype=np.float64)
        if P_full.shape[0] < sl.stop or P_full.shape[1] < sl.stop:
            return None
        P_lm = P_full[sl, sl].reshape(2, 2)
        P_lm = 0.5 * (P_lm + P_lm.T)
        return x, P_lm
    return None


def _ekf_pose_and_param_variance_diag(ekf: UnicycleEKF) -> np.ndarray:
    """Diagonal variances for robot block only: (x,y,theta) or (x,y,theta,r_R,r_L,b); excludes landmarks."""
    bs = int(ekf.base_state_size)
    P = np.asarray(ekf.P, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] != P.shape[1] or P.shape[0] < bs:
        return np.full(bs, np.nan, dtype=np.float64)
    return np.maximum(np.diagonal(P)[:bs].astype(np.float64, copy=False), 0.0)


def _fuse_gaussian_beliefs_information_form(
    beliefs: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fuse independent 2D Gaussian estimates using the information vector:
    Ω = Σ P_i^{-1},  ξ = Σ P_i^{-1} x_i,  P_f = Ω^{-1},  x_f = P_f ξ.
    """
    if len(beliefs) == 1:
        x, P = beliefs[0]
        return np.asarray(x, dtype=np.float64).reshape(2).copy(), np.asarray(
            P, dtype=np.float64
        ).reshape(2, 2).copy()
    omega = np.zeros((2, 2), dtype=np.float64)
    xi = np.zeros(2, dtype=np.float64)
    reg = 1e-12 * np.eye(2, dtype=np.float64)
    for x, P in beliefs:
        Pm = np.asarray(P, dtype=np.float64).reshape(2, 2)
        Pm = 0.5 * (Pm + Pm.T)
        inv_p = np.linalg.inv(Pm + reg)
        omega += inv_p
        xi += inv_p @ np.asarray(x, dtype=np.float64).reshape(2)
    p_f = np.linalg.inv(omega + reg)
    x_f = p_f @ xi
    return x_f, p_f


def _apply_map_fusion_to_ekf_landmarks(ekfs: list[UnicycleEKF | None], fusion_report: dict) -> None:
    """Write fused (x,y) and 2×2 marginal covariance into each scout's EKF for landmarks that were fused."""
    per = fusion_report.get("per_landmark", {})
    for i, ekf in enumerate(ekfs):
        if ekf is None or i == MOTHERSHIP_INDEX:
            continue
        for idx, typ in ekf.landmarks.items():
            lm_type = str(typ)
            block = per.get(lm_type)
            if not block or int(block.get("n_sources", 0)) <= 0:
                continue
            fused_xy = block.get("fused_xy_m")
            fused_cov = block.get("fused_cov_m2")
            if fused_xy is None or fused_cov is None:
                continue
            b0 = int(ekf.base_state_size)
            sl = slice(b0 + 2 * int(idx), b0 + 2 * int(idx) + 2)
            xf = np.asarray(fused_xy, dtype=np.float64).reshape(2)
            Pf = np.asarray(fused_cov, dtype=np.float64).reshape(2, 2)
            Pf = 0.5 * (Pf + Pf.T)
            ekf.state[sl] = xf
            P = np.asarray(ekf.P, dtype=np.float64).copy()
            P[sl, sl] = Pf
            ekf.P = P


def _refresh_plot_with_fused_landmarks(
    r_inst: robotarium.Robotarium,
    fusion_report: dict,
    landmark_ellipses: list[list[mpatches.Ellipse]],
    cov_n_sigma: float,
    fused_overlay: list[mpatches.Ellipse],
) -> None:
    """
    Hide per-scout landmark ellipses, remove any prior fused overlay, draw one ellipse per fused object.
    """
    if not r_inst.show_figure:
        return
    for le_list in landmark_ellipses:
        for e in le_list:
            e.set_visible(False)
    for e in fused_overlay:
        e.remove()
    fused_overlay.clear()
    ax = r_inst._axes_handle
    per = fusion_report.get("per_landmark", {})
    for lm_type in sorted(per.keys()):
        stats = per[lm_type]
        if int(stats.get("n_sources", 0)) <= 0:
            continue
        fused_xy = stats.get("fused_xy_m")
        fused_cov = stats.get("fused_cov_m2")
        if fused_xy is None or fused_cov is None:
            continue
        xy = np.asarray(fused_xy, dtype=np.float64).reshape(2)
        Pf = np.asarray(fused_cov, dtype=np.float64).reshape(2, 2)
        Pf = 0.5 * (Pf + Pf.T)
        ew, eh, eang = _xy_cov_ellipse_width_height_angle_deg(Pf, cov_n_sigma)
        if lm_type.startswith(_LM_PREFIX_BUOY):
            edge = "#b8860b"
        elif lm_type.startswith(_LM_PREFIX_ENEMY_SHIP):
            edge = _VIZ_ENEMY_LM_EDGE
        else:
            edge = "0.35"
        lw_fused = 2.65 if lm_type.startswith(_LM_PREFIX_ENEMY_SHIP) else 1.45
        ell = mpatches.Ellipse(
            (float(xy[0]), float(xy[1])),
            width=float(ew),
            height=float(eh),
            angle=float(eang),
            fill=False,
            edgecolor=edge,
            linewidth=lw_fused,
            linestyle="-",
            zorder=float(Map.VIZ_Z_FUSED_LM),
        )
        ax.add_patch(ell)
        fused_overlay.append(ell)
    r_inst._fig.canvas.draw_idle()
    r_inst._fig.canvas.flush_events()


def _build_map_fusion_landmark_report(
    ekfs: list[UnicycleEKF | None],
    env_map: Map,
    enemy_positions: list[np.ndarray],
) -> dict:
    """
    For each world enemy ship id, fuse marginal XY beliefs from all scouts that hold a track.
    RMSE aggregate is RMS of per-landmark position errors vs ground truth (one fused point per object).
    """
    n = len(ekfs)
    scout_indices = [i for i in range(n) if i != MOTHERSHIP_INDEX and ekfs[i] is not None]
    lm_ids: list[str] = [f"{_LM_PREFIX_BUOY}{bk}" for bk in env_map.buoys]
    lm_ids.extend(f"{_LM_PREFIX_ENEMY_SHIP}{ei}" for ei in range(len(enemy_positions)))

    per_landmark: dict[str, dict] = {}
    sum_x2 = 0.0
    sum_y2 = 0.0
    n_with_gt = 0

    for lm_type in lm_ids:
        beliefs: list[tuple[np.ndarray, np.ndarray]] = []
        contributing: list[int] = []
        for i in scout_indices:
            pair = _ekf_landmark_xy_cov(ekfs[i], lm_type)
            if pair is None:
                continue
            beliefs.append(pair)
            contributing.append(i)
        if not beliefs:
            per_landmark[lm_type] = {"status": "no_scout_track", "n_sources": 0}
            continue
        x_f, p_f = _fuse_gaussian_beliefs_information_form(beliefs)
        gxy = _ground_truth_xy_for_landmark(lm_type, env_map, enemy_positions)
        block: dict = {
            "n_sources": int(len(beliefs)),
            "contributing_robot_indices": contributing,
            "fused_xy_m": [float(x_f[0]), float(x_f[1])],
            "fused_cov_m2": [
                [float(p_f[0, 0]), float(p_f[0, 1])],
                [float(p_f[1, 0]), float(p_f[1, 1])],
            ],
        }
        if gxy is not None:
            ex = float(x_f[0] - float(gxy[0]))
            ey = float(x_f[1] - float(gxy[1]))
            block["error_xy_m"] = [ex, ey]
            block["position_error_m"] = float(np.hypot(ex, ey))
            sum_x2 += ex * ex
            sum_y2 += ey * ey
            n_with_gt += 1
        per_landmark[lm_type] = block

    aggregate: dict[str, float | int] = {}
    if n_with_gt > 0:
        aggregate = {
            "n_landmarks_with_ground_truth": int(n_with_gt),
            "rmse_x_m": float(np.sqrt(sum_x2 / n_with_gt)),
            "rmse_y_m": float(np.sqrt(sum_y2 / n_with_gt)),
            "rmse_position_m": float(np.sqrt((sum_x2 + sum_y2) / n_with_gt)),
        }

    return {
        "note": (
            "Map fusion: fused marginal (x,y) and 2×2 covariance per landmark across scouts using "
            "P_f = (Σ P_i^{-1})^{-1}, x_f = P_f Σ P_i^{-1} x_i (independent-Gaussian information fusion)."
        ),
        "per_landmark": per_landmark,
        "aggregate_vs_ground_truth": aggregate,
    }


def _ground_truth_xy_for_landmark(
    landmark_type: str, env_map: Map, enemy_positions: list[np.ndarray]
) -> np.ndarray | None:
    """Return (2,) ground-truth position for a SLAM landmark id, or ``None`` if unknown."""
    if landmark_type.startswith(_LM_PREFIX_BUOY):
        key = landmark_type[len(_LM_PREFIX_BUOY) :]
        b = env_map.buoys.get(key)
        if b is not None:
            return np.asarray(b, dtype=np.float64).reshape(2)
        return None
    if landmark_type.startswith(_LM_PREFIX_ENEMY_SHIP):
        rest = landmark_type[len(_LM_PREFIX_ENEMY_SHIP) :]
        try:
            idx = int(rest)
        except ValueError:
            return None
        if 0 <= idx < len(enemy_positions):
            return np.asarray(enemy_positions[idx], dtype=np.float64).reshape(2)
        return None
    return None


def _pose_rmse(gt: np.ndarray, est: np.ndarray) -> dict[str, float]:
    """RMSE for poses; ``gt`` / ``est`` shaped ``(T, 3)`` with ``x, y, theta`` (rad)."""
    if gt.shape != est.shape or gt.ndim != 2 or gt.shape[1] != 3:
        raise ValueError("gt and est must be (T, 3)")
    ex = est[:, 0] - gt[:, 0]
    ey = est[:, 1] - gt[:, 1]
    et = np.arctan2(np.sin(est[:, 2] - gt[:, 2]), np.cos(est[:, 2] - gt[:, 2]))
    return {
        "rmse_x_m": float(np.sqrt(np.mean(ex**2))),
        "rmse_y_m": float(np.sqrt(np.mean(ey**2))),
        "rmse_position_m": float(np.sqrt(np.mean(ex**2 + ey**2))),
        "rmse_theta_rad": float(np.sqrt(np.mean(et**2))),
        "rmse_theta_deg": float(np.degrees(np.sqrt(np.mean(et**2)))),
        "n_timesteps": int(gt.shape[0]),
    }


def _landmark_rmse_for_robot(
    lm_hist: list[dict[str, list[float]]],
    env_map: Map,
    enemy_positions: list[np.ndarray],
) -> dict[str, dict[str, float | int]]:
    """Per-landmark RMSE for one robot's landmark trajectory (XY only)."""
    sum_x2: dict[str, float] = {}
    sum_y2: dict[str, float] = {}
    counts: dict[str, int] = {}
    for snap in lm_hist:
        for lm_type, xy in snap.items():
            gxy = _ground_truth_xy_for_landmark(str(lm_type), env_map, enemy_positions)
            if gxy is None:
                continue
            ex = float(xy[0]) - float(gxy[0])
            ey = float(xy[1]) - float(gxy[1])
            if lm_type not in counts:
                sum_x2[lm_type] = 0.0
                sum_y2[lm_type] = 0.0
                counts[lm_type] = 0
            sum_x2[lm_type] += ex * ex
            sum_y2[lm_type] += ey * ey
            counts[lm_type] += 1
    out: dict[str, dict[str, float | int]] = {}
    for lm_type, c in counts.items():
        if c <= 0:
            continue
        out[lm_type] = {
            "n_samples": int(c),
            "rmse_x_m": float(np.sqrt(sum_x2[lm_type] / c)),
            "rmse_y_m": float(np.sqrt(sum_y2[lm_type] / c)),
            "rmse_position_m": float(np.sqrt((sum_x2[lm_type] + sum_y2[lm_type]) / c)),
        }
    return out


def _build_rmse_report(
    gt_history: list[list[np.ndarray]],
    kf_history: list[list[np.ndarray]],
    landmark_history: list[list[dict[str, list[float]]]],
    env_map: Map,
    enemy_positions: list[np.ndarray],
) -> dict:
    """Assemble RMSE metrics for ships (pose) and landmarks (XY per robot). Excludes mothership."""
    n = len(gt_history)
    filtered = [i for i in range(n) if i != MOTHERSHIP_INDEX]
    per_robot: list[dict] = []
    gt_mats: list[np.ndarray] = []
    kf_mats: list[np.ndarray] = []
    for i in filtered:
        gt_i = np.asarray(gt_history[i], dtype=np.float64).reshape(-1, 3)
        kf_i = np.asarray(kf_history[i], dtype=np.float64).reshape(-1, 3)
        gt_mats.append(gt_i)
        kf_mats.append(kf_i)
        pose = _pose_rmse(gt_i, kf_i)
        per_robot.append(
            {
                "robot_index": i,
                "pose_vs_ground_truth": pose,
                "landmarks_vs_ground_truth": _landmark_rmse_for_robot(
                    landmark_history[i], env_map, enemy_positions
                ),
            }
        )
    if filtered:
        gt_all = np.vstack(gt_mats)
        kf_all = np.vstack(kf_mats)
        total_pose = _pose_rmse(gt_all, kf_all)
    else:
        total_pose = {
            "rmse_x_m": 0.0,
            "rmse_y_m": 0.0,
            "rmse_position_m": 0.0,
            "rmse_theta_rad": 0.0,
            "rmse_theta_deg": 0.0,
            "n_timesteps": 0,
        }

    # Pool landmark errors across filtered robots and timesteps (each sample is one robot's estimate).
    sum_x2_all = 0.0
    sum_y2_all = 0.0
    n_lm_samples = 0
    for i in filtered:
        for snap in landmark_history[i]:
            for lm_type, xy in snap.items():
                gxy = _ground_truth_xy_for_landmark(str(lm_type), env_map, enemy_positions)
                if gxy is None:
                    continue
                ex = float(xy[0]) - float(gxy[0])
                ey = float(xy[1]) - float(gxy[1])
                sum_x2_all += ex * ex
                sum_y2_all += ey * ey
                n_lm_samples += 1
    landmarks_total: dict[str, float | int] = {
        "n_samples": int(n_lm_samples),
        "rmse_x_m": float(np.sqrt(sum_x2_all / n_lm_samples)) if n_lm_samples else 0.0,
        "rmse_y_m": float(np.sqrt(sum_y2_all / n_lm_samples)) if n_lm_samples else 0.0,
        "rmse_position_m": float(
            np.sqrt((sum_x2_all + sum_y2_all) / n_lm_samples)
        )
        if n_lm_samples
        else 0.0,
    }

    return {
        "note": (
            "RMSE (root mean square error). YAML file uses JSON syntax (valid YAML 1.2). "
            f"Robot {MOTHERSHIP_INDEX} (mothership) is excluded — assumed perfect GNSS."
        ),
        "mothership_index_excluded_from_filter_and_rmse": MOTHERSHIP_INDEX,
        "ships": {
            "per_robot": per_robot,
            "total_filtered_robots_all_timesteps": total_pose,
        },
        "landmarks": {"total_pooled_filtered_robots": landmarks_total},
    }


def _print_rmse_report_section(report: dict, *, heading: str) -> None:
    ships = report["ships"]
    print(f"\n{heading}")
    print("=== RMSE: ship pose vs ground truth ===")
    for block in ships["per_robot"]:
        i = block["robot_index"]
        p = block["pose_vs_ground_truth"]
        print(
            f"  robot {i}:  RMSE xy position = {p['rmse_position_m']:.4f} m  "
            f"(x {p['rmse_x_m']:.4f} m, y {p['rmse_y_m']:.4f} m)  "
            f"RMSE theta = {p['rmse_theta_deg']:.3f} deg ({p['rmse_theta_rad']:.4f} rad)"
        )
    t = ships["total_filtered_robots_all_timesteps"]
    print(
        f"  All filtered robots pooled (excl. mothership {MOTHERSHIP_INDEX}):  "
        f"RMSE position = {t['rmse_position_m']:.4f} m  "
        f"RMSE theta = {t['rmse_theta_deg']:.3f} deg"
    )
    print("\n=== RMSE: landmark (enemy ship) XY vs ground truth (per estimating ship) ===")
    for block in ships["per_robot"]:
        i = block["robot_index"]
        lm = block["landmarks_vs_ground_truth"]
        if not lm:
            print(f"  robot {i}:  (no landmark samples in history)")
            continue
        print(f"  robot {i}:")
        for name, stats in sorted(lm.items()):
            print(
                f"    {name}:  RMSE position = {stats['rmse_position_m']:.4f} m  "
                f"(x {stats['rmse_x_m']:.4f}, y {stats['rmse_y_m']:.4f})  "
                f"n={stats['n_samples']}"
            )
    lt = report["landmarks"]["total_pooled_filtered_robots"]
    print(
        f"\n  Landmarks pooled (filtered robots only, all timesteps):  RMSE position = "
        f"{lt['rmse_position_m']:.4f} m  n={lt['n_samples']}"
    )


def _print_rmse_report_dual(vanilla_block: dict, params_block: dict, fusion_report: dict) -> None:
    _print_rmse_report_section(vanilla_block, heading="=== Vanilla EKF (shown on Robotarium figure) ===")
    _print_rmse_report_section(params_block, heading="=== Parameter-augmented EKF (logged only) ===")
    if fusion_report:
        print("\n=== RMSE: map-fused landmark XY vs ground truth (vanilla scout tracks, end of mission) ===")
        agg = fusion_report.get("aggregate_vs_ground_truth") or {}
        if agg:
            print(
                f"  Fused landmarks (mean over objects with GT):  RMSE position = "
                f"{agg['rmse_position_m']:.4f} m  "
                f"(x {agg['rmse_x_m']:.4f}, y {agg['rmse_y_m']:.4f})  "
                f"n={agg['n_landmarks_with_ground_truth']}"
            )
        else:
            print("  (no fused landmarks with ground truth to aggregate)")
        for name, stats in sorted(fusion_report.get("per_landmark", {}).items()):
            if stats.get("n_sources", 0) == 0:
                print(f"  {name}:  (no scout track)")
                continue
            ns = stats["n_sources"]
            if "position_error_m" in stats:
                print(
                    f"  {name}:  fused error = {stats['position_error_m']:.4f} m  "
                    f"(n_sources={ns}, robots {stats.get('contributing_robot_indices')})"
                )
            else:
                print(f"  {name}:  fused  n_sources={ns}  (no GT for this id)")


def _save_rmse_yaml(report: dict, path: Path) -> None:
    """Write metrics; JSON encoding is a YAML 1.2 subset (no extra dependency)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=False)
        f.write("\n")


def _dual_rmse_metric_paths(base: Path) -> tuple[Path, Path]:
    """
    From a single stem path (e.g. ``.../experiment_rmse_metrics.yaml``), return paths for the
    vanilla (on-figure) filter and the parameter-augmented filter.
    """
    base = Path(base)
    if not base.suffix:
        base = base.with_suffix(".yaml")
    stem, suf = base.stem, base.suffix
    parent = base.parent
    return (parent / f"{stem}_vanilla{suf}", parent / f"{stem}_parameter{suf}")


def _save_scout_kinematics_figure(
    sim_time_s: np.ndarray,
    kin_m: np.ndarray,
    *,
    n_robots: int,
    mothership_index: int,
    wheel_radius_nominal_m: float,
    base_length_nominal_m: float,
    out_path: Path,
) -> None:
    """
    Plot EKF-estimated right wheel radius, left wheel radius, and base length vs time for each scout.

    ``kin_m`` is (T, n_robots, 3): channels [r_R, r_L, b] in meters; mothership row may be NaN.
    """
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    T = int(sim_time_s.shape[0])
    titles = ("Wheel radius right (m)", "Wheel radius left (m)", "Base length (m)")
    fig, axes = plt.subplots(3, 1, figsize=(10.0, 8.0), sharex=True, constrained_layout=True)
    for ax_i, ax in enumerate(axes):
        for ri in range(n_robots):
            if ri == mothership_index:
                continue
            y = kin_m[:, ri, ax_i]
            if not np.any(np.isfinite(y)):
                continue
            ax.plot(
                sim_time_s,
                y,
                color=_ROBOT_KF_COLORS[ri % len(_ROBOT_KF_COLORS)],
                linewidth=1.35,
                label=f"robot {ri}",
            )
        if ax_i in (0, 1):
            ax.axhline(float(wheel_radius_nominal_m), color="0.25", linestyle="--", linewidth=1.0, alpha=0.7)
        else:
            ax.axhline(float(base_length_nominal_m), color="0.25", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_ylabel(titles[ax_i])
        ax.grid(True, alpha=0.35)
    axes[0].legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        "EKF kinematic parameter estimates (scouts only; dashed = FILTER_EKF_PRIOR_* or Robotarium)"
    )
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Scout kinematics figure saved to {out_path}")


def _write_offline_experiment_npz(
    root: Path,
    *,
    dt: float,
    n_robots: int,
    mothership_index: int,
    measurement_rng_seed: int,
    lighthouse_update_interval_s: float,
    R_bearing_var: float,
    R_range_var: float,
    wheel_radius: float,
    base_length: float,
    counts_to_rad: float,
    process_noise: tuple[float, float, float],
    lighthouses: dict[str, np.ndarray],
    enemy_xy: np.ndarray,
    nav_goal_xy: np.ndarray,
    offline_pose: list[np.ndarray],
    offline_enc: list[np.ndarray],
    offline_ori: list[np.ndarray],
    offline_nav: list[int],
    offline_v: list[np.ndarray],
    offline_w: list[np.ndarray],
    offline_lh: list[tuple],
    offline_enemy: list[tuple],
    offline_buoy: list[tuple],
    gt_history: list[list[np.ndarray]],
    kf_history: list[list[np.ndarray]],
    landmark_history: list[list[dict[str, list[float]]]],
    kf_history_params: list[list[np.ndarray]],
    landmark_history_params: list[list[dict[str, list[float]]]],
    routes: list,
    fusion_report: dict,
    kf_kinematics_history: list[list[np.ndarray]],
    cov_var_pose_vanilla: list[list[np.ndarray]],
    cov_var_pose_and_params: list[list[np.ndarray]],
    skip_nav: bool = False,
    ekf_prior_wheel_radius_m: float,
    ekf_prior_base_length_m: float,
) -> None:
    """
    Write NPZ bundles under ``root`` for offline replay / analysis.

    Layout (each name is a directory containing one ``*.npz`` of the same base name):

    - ``poses/`` — ``poses`` (T,3,n), ``sim_time_s`` (T,), ``nav_phase`` (T,) uint8
    - ``encoders/`` — ``encoder_counts`` (T,2,n) int32, ``sim_time_s``, ``v_hat_m_s`` (T,n), ``w_hat_rad_s`` (T,n)
    - ``orientations/`` — ``orientation_deg`` (T,n) simulated IMU-style degrees [0,360), ``sim_time_s``
    - ``lighthouse_measurements/`` — event arrays + ``wall_time_unix_s``
    - ``enemy_measurements/`` — bearing/range SLAM events to enemy ships
    - ``buoy_measurements/`` — same for buoys (may be length 0)
    - ``predicted_paths/`` — GT + vanilla ``ekf_poses_xy_theta`` + param ``ekf_poses_xy_theta_params``,
      ``landmark_snapshots`` / ``landmark_snapshots_params``, ``ekf_kinematics_m`` (T,n,3) from the
      augmented filter, ``ekf_pose_variance_diag_vanilla`` (T,n,3) and ``ekf_pose_and_param_variance_diag``
      (T,n,6) — **robot block diagonal variances only** (no landmark entries). NaN for mothership / padding.
    - ``metadata/`` — ``wheel_radius``/``base_length`` from Robotarium; ``filter_ekf_prior_*`` = EKF init/plot priors.
    """
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    T = len(offline_pose)
    if T == 0:
        return
    sim_time = np.arange(T, dtype=np.float64) * float(dt)
    nav_arr = np.asarray(offline_nav, dtype=np.uint8).reshape(T)
    poses = np.stack(offline_pose, axis=0)
    encoders = np.stack(offline_enc, axis=0)
    orientations = np.stack(offline_ori, axis=0)
    v_hat = np.stack(offline_v, axis=0)
    w_hat = np.stack(offline_w, axis=0)

    d_pose = root / "poses"
    d_pose.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        d_pose / "poses.npz",
        poses=poses,
        sim_time_s=sim_time,
        nav_phase=nav_arr,
        note="poses are Robotarium ground truth at each tracking_step (x,y,theta_rad), same indexing as encoders.",
    )

    d_enc = root / "encoders"
    d_enc.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        d_enc / "encoders.npz",
        encoder_counts=encoders,
        sim_time_s=sim_time,
        v_hat_m_s=v_hat,
        w_hat_rad_s=w_hat,
        note="encoder_counts are cumulative get_encoders() after each step; v_hat/w_hat match experiment unicycle odometry.",
    )

    d_ori = root / "orientations"
    d_ori.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        d_ori / "orientations.npz",
        orientation_deg=orientations,
        sim_time_s=sim_time,
        note="from Robotarium.get_orientations(): heading in degrees with ORIENTATION_NOISE_STD (see rps/robotarium.py).",
    )

    def _events_to_npz(
        rows: list[tuple],
        out_dir: Path,
        fname: str,
        *,
        kind: str,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        m = len(rows)
        if m == 0:
            if kind == "lighthouse":
                np.savez_compressed(
                    out_dir / fname,
                    sim_step=np.zeros(0, dtype=np.int64),
                    wall_time_unix_s=np.zeros(0, dtype=np.float64),
                    robot_index=np.zeros(0, dtype=np.int64),
                    lighthouse_name=np.array([], dtype=object),
                    z_bearing_rad=np.zeros(0, dtype=np.float64),
                    z_range_m=np.zeros(0, dtype=np.float64),
                    has_range=np.zeros(0, dtype=np.uint8),
                    beacon_xy_m=np.zeros((0, 2), dtype=np.float64),
                )
            elif kind == "enemy":
                np.savez_compressed(
                    out_dir / fname,
                    sim_step=np.zeros(0, dtype=np.int64),
                    wall_time_unix_s=np.zeros(0, dtype=np.float64),
                    robot_index=np.zeros(0, dtype=np.int64),
                    enemy_index=np.zeros(0, dtype=np.int64),
                    z_bearing_rad=np.zeros(0, dtype=np.float64),
                    z_range_m=np.zeros(0, dtype=np.float64),
                    landmark_xy_m=np.zeros((0, 2), dtype=np.float64),
                )
            else:
                np.savez_compressed(
                    out_dir / fname,
                    sim_step=np.zeros(0, dtype=np.int64),
                    wall_time_unix_s=np.zeros(0, dtype=np.float64),
                    robot_index=np.zeros(0, dtype=np.int64),
                    buoy_name=np.array([], dtype=object),
                    z_bearing_rad=np.zeros(0, dtype=np.float64),
                    z_range_m=np.zeros(0, dtype=np.float64),
                    beacon_xy_m=np.zeros((0, 2), dtype=np.float64),
                )
            return
        if kind == "lighthouse":
            sim_step, wall_t, ri, name, zb, zr, has_r, bx, by = zip(*rows)
            np.savez_compressed(
                out_dir / fname,
                sim_step=np.asarray(sim_step, dtype=np.int64),
                wall_time_unix_s=np.asarray(wall_t, dtype=np.float64),
                robot_index=np.asarray(ri, dtype=np.int64),
                lighthouse_name=np.asarray(name, dtype=object),
                z_bearing_rad=np.asarray(zb, dtype=np.float64),
                z_range_m=np.asarray(zr, dtype=np.float64),
                has_range=np.asarray(has_r, dtype=np.uint8),
                beacon_xy_m=np.stack([np.asarray(bx, dtype=np.float64), np.asarray(by, dtype=np.float64)], axis=1),
                note="z_bearing_rad is body-frame bearing; z_range_m is NaN when has_range=0.",
            )
        elif kind == "enemy":
            sim_step, wall_t, ri, ei, zb, zr, bx, by = zip(*rows)
            np.savez_compressed(
                out_dir / fname,
                sim_step=np.asarray(sim_step, dtype=np.int64),
                wall_time_unix_s=np.asarray(wall_t, dtype=np.float64),
                robot_index=np.asarray(ri, dtype=np.int64),
                enemy_index=np.asarray(ei, dtype=np.int64),
                z_bearing_rad=np.asarray(zb, dtype=np.float64),
                z_range_m=np.asarray(zr, dtype=np.float64),
                landmark_xy_m=np.stack([np.asarray(bx, dtype=np.float64), np.asarray(by, dtype=np.float64)], axis=1),
            )
        else:
            sim_step, wall_t, ri, bname, zb, zr, bx, by = zip(*rows)
            np.savez_compressed(
                out_dir / fname,
                sim_step=np.asarray(sim_step, dtype=np.int64),
                wall_time_unix_s=np.asarray(wall_t, dtype=np.float64),
                robot_index=np.asarray(ri, dtype=np.int64),
                buoy_name=np.asarray(bname, dtype=object),
                z_bearing_rad=np.asarray(zb, dtype=np.float64),
                z_range_m=np.asarray(zr, dtype=np.float64),
                beacon_xy_m=np.stack([np.asarray(bx, dtype=np.float64), np.asarray(by, dtype=np.float64)], axis=1),
            )

    _events_to_npz(
        offline_lh,
        root / "lighthouse_measurements",
        "lighthouse_measurements.npz",
        kind="lighthouse",
    )
    _events_to_npz(
        offline_enemy,
        root / "enemy_measurements",
        "enemy_measurements.npz",
        kind="enemy",
    )
    _events_to_npz(
        offline_buoy,
        root / "buoy_measurements",
        "buoy_measurements.npz",
        kind="buoy",
    )

    d_pred = root / "predicted_paths"
    d_pred.mkdir(parents=True, exist_ok=True)
    gt_stack = np.stack(
        [np.asarray(gt_history[i], dtype=np.float64).reshape(T, 3) for i in range(n_robots)],
        axis=1,
    )
    kf_nan = np.full((T, n_robots, 3), np.nan, dtype=np.float64)
    kf_params_nan = np.full((T, n_robots, 3), np.nan, dtype=np.float64)
    for i in range(n_robots):
        if i == mothership_index:
            continue
        kf_stack_i = np.asarray(kf_history[i], dtype=np.float64).reshape(T, 3)
        kf_nan[:, i, :] = kf_stack_i
        kfp = np.asarray(kf_history_params[i], dtype=np.float64).reshape(T, 3)
        kf_params_nan[:, i, :] = kfp
    pred_kw: dict = dict(
        gt_poses_xy_theta=gt_stack,
        ekf_poses_xy_theta=kf_nan,
        ekf_poses_xy_theta_params=kf_params_nan,
        sim_time_s=sim_time,
        nav_phase=nav_arr,
        landmark_snapshots=np.asarray(landmark_history, dtype=object),
        landmark_snapshots_params=np.asarray(landmark_history_params, dtype=object),
        note=(
            "landmark_snapshots = vanilla EKF (on-figure); landmark_snapshots_params = parameter-augmented EKF. "
            "ekf_kinematics_m (T,n,3) = [r_R,r_L,b] from augmented filter. "
            "ekf_pose_variance_diag_vanilla (T,n,3) = diag(P) for (x,y,theta) block only. "
            "ekf_pose_and_param_variance_diag (T,n,6) adds (var r_R, var r_L, var b) for augmented filter."
        ),
    )
    kin_nan = np.full((T, n_robots, 3), np.nan, dtype=np.float64)
    for i in range(n_robots):
        if i == mothership_index:
            continue
        arr = np.asarray(kf_kinematics_history[i], dtype=np.float64).reshape(-1, 3)
        if arr.shape[0] == T:
            kin_nan[:, i, :] = arr
    pred_kw["ekf_kinematics_m"] = kin_nan

    var_v = np.full((T, n_robots, 3), np.nan, dtype=np.float64)
    var_p = np.full((T, n_robots, 6), np.nan, dtype=np.float64)
    for i in range(n_robots):
        if i == mothership_index:
            continue
        hv = cov_var_pose_vanilla[i]
        hp = cov_var_pose_and_params[i]
        if len(hv) == T:
            var_v[:, i, :] = np.stack([np.asarray(v, dtype=np.float64).reshape(3) for v in hv], axis=0)
        if len(hp) == T:
            var_p[:, i, :] = np.stack([np.asarray(v, dtype=np.float64).reshape(6) for v in hp], axis=0)
    pred_kw["ekf_pose_variance_diag_vanilla"] = var_v
    pred_kw["ekf_pose_and_param_variance_diag"] = var_p
    np.savez_compressed(d_pred / "predicted_paths.npz", **pred_kw)

    lh_names = np.asarray(list(lighthouses.keys()), dtype=object)
    lh_xy = np.stack([np.asarray(lighthouses[k], dtype=np.float64).reshape(2) for k in lighthouses], axis=0)
    routes_obj = np.empty(n_robots, dtype=object)
    for i in range(n_robots):
        routes_obj[i] = np.asarray(routes[i], dtype=np.float64).reshape(-1, 2)

    d_meta = root / "metadata"
    d_meta.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        d_meta / "metadata.npz",
        dt=dt,
        n_robots=n_robots,
        mothership_index=mothership_index,
        measurement_rng_seed=np.int64(measurement_rng_seed),
        lighthouse_update_interval_s=np.float64(lighthouse_update_interval_s),
        R_bearing_var=np.float64(R_bearing_var),
        R_range_var=np.float64(R_range_var),
        wheel_radius=np.float64(wheel_radius),
        base_length=np.float64(base_length),
        filter_ekf_prior_wheel_radius_m=np.float64(ekf_prior_wheel_radius_m),
        filter_ekf_prior_base_length_m=np.float64(ekf_prior_base_length_m),
        counts_to_rad=np.float64(counts_to_rad),
        process_noise_xyz=np.asarray(process_noise, dtype=np.float64).reshape(3),
        lighthouse_names=lh_names,
        lighthouse_xy_m=lh_xy,
        enemy_xy_m=np.asarray(enemy_xy, dtype=np.float64).reshape(-1, 2),
        mothership_nav_goal_xy_m=np.asarray(nav_goal_xy, dtype=np.float64).reshape(2),
        routes=routes_obj,
        fusion_report=np.asarray(fusion_report, dtype=object),
        dual_scout_ekf=np.bool_(True),
        skip_nav=np.bool_(skip_nav),
        note=(
            "fusion_report is the post-scout map-fusion dict from vanilla scout tracks (allow_pickle=True). "
            "dual_scout_ekf: vanilla + parameter-augmented filters run in parallel each timestep. "
            "sim_step in measurement files indexes poses[*] at the same timestep."
        ),
    )
    print(f"\nOffline experiment NPZ bundles written under {root}")


def _route_vertex_cumlen_list(routes: list) -> list[np.ndarray | None]:
    out: list[np.ndarray | None] = []
    for rt in routes:
        rr = np.asarray(rt, dtype=np.float64).reshape(-1, 2)
        if rr.shape[0] <= 1:
            out.append(None)
        else:
            seg = np.linalg.norm(np.diff(rr, axis=0), axis=1)
            out.append(np.concatenate([[0.0], np.cumsum(seg)]))
    return out


def _simulate_lighthouse_measurements(
    robot_xy: np.ndarray,
    robot_theta: float,
    beacon_xy: np.ndarray,
    rng: np.random.Generator,
    bearing_var_rad2: float,
    range_var_m2: float,
    *,
    include_bearing: bool = True,
    include_range: bool = True,
) -> tuple[float | None, float | None]:
    """
    Bearing in the robot body frame (world azimuth minus heading), matching ``UnicycleEKF``'s
    ``h_hat = wrap_angle(atan2(dy, dx) - theta)``. Range is Euclidean ground-truth range plus noise.
    Omit components with ``include_bearing`` / ``include_range``; omitted values are returned as ``None``.
    """
    dx = float(beacon_xy[0] - float(robot_xy[0]))
    dy = float(beacon_xy[1] - float(robot_xy[1]))
    rho = float(np.hypot(dx, dy))
    rho = max(rho, 1e-9)
    z_bearing: float | None
    if include_bearing:
        bearing_world = float(np.arctan2(dy, dx))
        bearing_body = wrap_angle(bearing_world - float(robot_theta))
        z_bearing = wrap_angle(bearing_body + float(rng.normal(0.0, float(np.sqrt(bearing_var_rad2)))))
    else:
        z_bearing = None
    z_range: float | None
    if include_range:
        z_range = rho + float(rng.normal(0.0, float(np.sqrt(range_var_m2))))
    else:
        z_range = None
    return z_bearing, z_range


def _flash_center_title(
    r_inst: robotarium.Robotarium,
    message: str,
    *,
    duration_s: float = 2.5,
    font_height_m: float = 0.1,
) -> None:
    """
    Full-screen-style banner in arena coordinates, matching ``Robotarium.initialize``'s
    ``...INITIALIZING...`` overlay (position, font scale, bold blue). Uses wall-clock pause only
    (no ``step()``) so the first scouting timestep still aligns with ``encoders_prev``.
    """
    if not r_inst.show_figure:
        return
    ax = r_inst._axes_handle
    font_size = determine_font_size(r_inst, font_height_m)
    banner = ax.text(
        0.0,
        -0.5,
        message,
        fontsize=font_size,
        color="b",
        fontweight="bold",
        horizontalalignment="center",
        verticalalignment="center",
        zorder=999,
    )
    r_inst._fig.canvas.draw_idle()
    r_inst._fig.canvas.flush_events()
    plt.pause(float(duration_s))
    banner.remove()
    r_inst._fig.canvas.draw_idle()
    r_inst._fig.canvas.flush_events()


def run_experiment(
    cycles: int = 2,
    process_noise: tuple[float, float, float] = (0.0001, 0.0001, 0.0001),
    cov_n_sigma: float = 2.0,
    rmse_out: Path | None = None,
    *,
    simulate_death: bool = False,
    offline_log_root: Path | None = None,
    measurement_rng_seed: int | None = None,
    skip_nav: bool = False,
) -> None:
    _ = cycles  # reserved for multi-phase / repeat missions

    n = int(Map.START_POINTS.shape[0])
    initial_conditions = Map.START_POINTS.T

    r = robotarium.Robotarium(
        number_of_robots=n,
        show_figure=True,
        initial_conditions=initial_conditions,
        sim_in_real_time=True,
    )

    env_map = Map(r)
    router = Router(env_map)
    routes = router.generate_scouting_phase_routes(initial_conditions)
    scout_goal_xy = np.stack(
        [np.asarray(r[-1], dtype=np.float64).reshape(2) for r in routes],
        axis=1,
    )
    route_vertex_cumlen = _route_vertex_cumlen_list(routes)

    lighthouses = {k: np.asarray(v, dtype=np.float64).reshape(2) for k, v in env_map.lighthouses.items()}
    enemy_positions = [
        np.asarray(xy, dtype=np.float64).reshape(2) for xy in Map.ENEMY_COORDINATES
    ]

    unicycle_pose_controller = create_uni_position_controller(
        velocity_magnitude_limit=PATH_TRACK_VEL_LIMIT_M_S,
    )
    uni_barrier_cert = create_uni_barrier_certificate()
    at_position = create_at_position()

    x = r.get_poses()
    r.step()

    wheel_radius = r.WHEEL_RADIUS
    base_length = r.BASE_LENGTH
    ekf_wheel_radius = (
        float(FILTER_EKF_PRIOR_WHEEL_RADIUS_M)
        if FILTER_EKF_PRIOR_WHEEL_RADIUS_M is not None
        else float(wheel_radius)
    )
    ekf_base_length = (
        float(FILTER_EKF_PRIOR_BASE_LENGTH_M)
        if FILTER_EKF_PRIOR_BASE_LENGTH_M is not None
        else float(base_length)
    )
    dt = r.TIME_STEP

    counts_to_rad = 2 * np.pi / (r.ENCODER_COUNTS_PER_REVOLUTION * r.MOTOR_GEAR_RATIO)
    encoder_noise_std = r.ENCODER_NOISE_STD
    encoder_ang_vel_var = (encoder_noise_std * counts_to_rad / dt) ** 2
    encoder_noise_matrix = np.eye(2) * encoder_ang_vel_var
    process_noise_matrix = np.eye(3) * np.array(process_noise)

    ekfs_vanilla: list[UnicycleEKF | None] = [None] * n
    ekfs_params: list[UnicycleEKF | None] = [None] * n
    for i in range(MOTHERSHIP_INDEX + 1, n):
        ic = np.asarray(initial_conditions[:, i], dtype=float).reshape(-1)
        ekfs_vanilla[i] = UnicycleEKF(
            initial_state=ic,
            initial_covariance=np.zeros((3, 3)),
            b=ekf_base_length,
            r=ekf_wheel_radius,
            M=encoder_noise_matrix,
            Q=process_noise_matrix,
            estimate_robot_parameters=False,
        )
        ekfs_params[i] = UnicycleEKF(
            initial_state=ic,
            initial_covariance=np.zeros((3, 3)),
            b=ekf_base_length,
            r=ekf_wheel_radius,
            M=encoder_noise_matrix,
            Q=process_noise_matrix,
            estimate_robot_parameters=True,
        )

    R_range_var = float(Map.LIGHTHOUSE_RANGE_MEAS_VAR_M2)
    R_bearing_var = float(Map.LIGHTHOUSE_BEARING_MEAS_VAR_RAD2)
    meas_seed_for_meta: int | None = None
    if measurement_rng_seed is not None:
        meas_seed_for_meta = int(measurement_rng_seed)
        rng_meas = np.random.default_rng(meas_seed_for_meta)
    elif offline_log_root is not None:
        meas_seed_for_meta = int(secrets.randbelow(2**31 - 1)) or 1
        rng_meas = np.random.default_rng(meas_seed_for_meta)
    else:
        rng_meas = np.random.default_rng()

    ax = r._axes_handle
    colors = [_ROBOT_COLORS[i % len(_ROBOT_COLORS)] for i in range(n)]
    kf_colors = [_ROBOT_KF_COLORS[i % len(_ROBOT_KF_COLORS)] for i in range(n)]

    gt_trails: list = []
    kf_trails: list = []
    cov_ellipses: list[mpatches.Ellipse] = []
    # Per-robot unknown-landmark (x,y) covariance ellipses (grows with SLAM map).
    landmark_ellipses: list[list[mpatches.Ellipse]] = [[] for _ in range(n)]
    fused_landmark_overlay: list[mpatches.Ellipse] = []
    for i in range(n):
        c = colors[i]
        ckf = kf_colors[i]
        lw_gt = 1.55 if i == MOTHERSHIP_INDEX else 1.38
        lw_kf = 2.05
        gt_line, = ax.plot([], [], "-", color=c, linewidth=lw_gt, zorder=float(Map.VIZ_Z_GT_KF_TRAIL))
        kf_line, = ax.plot([], [], "--", color=ckf, linewidth=lw_kf, zorder=float(Map.VIZ_Z_GT_KF_TRAIL))
        ell = mpatches.Ellipse(
            (0.0, 0.0),
            width=1e-6,
            height=1e-6,
            angle=0.0,
            fill=False,
            edgecolor=ckf,
            linewidth=1.45,
            linestyle=(0, (3, 2)),
            zorder=float(Map.VIZ_Z_POSE_COV),
        )
        ax.add_patch(ell)
        gt_trails.append(gt_line)
        kf_trails.append(kf_line)
        cov_ellipses.append(ell)

    kf_trails[MOTHERSHIP_INDEX].set_visible(False)
    cov_ellipses[MOTHERSHIP_INDEX].set_visible(False)

    map_legend_handles = [
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="o",
            markersize=9,
            markerfacecolor=Map.LIGHTHOUSE_COLOR,
            markeredgecolor="#888888",
            markeredgewidth=0.6,
            label="Lighthouse",
        ),
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="^",
            markersize=9,
            markerfacecolor=Map.ENEMY_COLOR,
            markeredgecolor="none",
            label="Enemy ship",
        ),
    ]
    ax.legend(
        handles=map_legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.925),
        fontsize=8,
        framealpha=0.9,
    )

    gt_history: list[list[np.ndarray]] = [[] for _ in range(n)]
    kf_history: list[list[np.ndarray]] = [[] for _ in range(n)]
    kf_history_params: list[list[np.ndarray]] = [[] for _ in range(n)]
    kf_kinematics_history: list[list[np.ndarray]] = [[] for _ in range(n)]
    landmark_history: list[list[dict[str, list[float]]]] = [[] for _ in range(n)]
    landmark_history_params: list[list[dict[str, list[float]]]] = [[] for _ in range(n)]
    cov_var_pose_vanilla: list[list[np.ndarray]] = [[] for _ in range(n)]
    cov_var_pose_and_params: list[list[np.ndarray]] = [[] for _ in range(n)]
    encoders_prev = r.get_encoders()

    offline_root = Path(offline_log_root).resolve() if offline_log_root is not None else None
    offline_pose: list[np.ndarray] = []
    offline_enc: list[np.ndarray] = []
    offline_ori: list[np.ndarray] = []
    offline_nav: list[int] = []
    offline_v: list[np.ndarray] = []
    offline_w: list[np.ndarray] = []
    offline_lh_meas: list[tuple] = []
    offline_enemy_meas: list[tuple] = []
    offline_buoy_meas: list[tuple] = []

    lighthouse_last_update: list[dict[str, float]] = [
        dict.fromkeys(lighthouses, 0.0) for _ in range(n)
    ]
    lighthouse_update_interval_s = float(np.random.uniform(0.25, 0.5))

    _landmark_clock_keys: list[str] = [f"buoy:{bk}" for bk in env_map.buoys]
    _landmark_clock_keys.extend(f"enemy:{ei}" for ei in range(len(enemy_positions)))
    landmark_last_update: list[dict[str, float]] = [
        dict.fromkeys(_landmark_clock_keys, 0.0) for _ in range(n)
    ]
    lm_d_min_m = float(Map.LIGHTHOUSE_MEASUREMENT_MIN_RANGE_M)
    lm_d_max_m = float(Map.LIGHTHOUSE_BEARING_SIGHTING_MAX_M)

    if any(np.asarray(rt, dtype=float).size == 0 for rt in routes):
        r.debug()
        return

    s_along_floor = np.zeros(n, dtype=np.float64)
    nav_phase = False
    # Robotarium: at most one ``get_poses()`` per ``step()``; reuse poses when we already fetched
    # for nav route setup immediately before the next ``tracking_step``.
    skip_next_pose_read = False
    scout_robot_dead = False
    death_overlay: list = [None]
    # After map fusion: no scout EKF predict/update, no GT/KF/landmark/offline appends (scouts hold still in nav).
    log_and_filter_scout_and_gt = True

    def _save_kinematics_figures_after_fusion() -> None:
        if not gt_history[MOTHERSHIP_INDEX + 1]:
            return
        T_plot = len(gt_history[MOTHERSHIP_INDEX + 1])
        sim_plot = np.arange(T_plot, dtype=np.float64) * float(dt)
        kin_plot = np.full((T_plot, n, 3), np.nan, dtype=np.float64)
        for ri in range(n):
            if ri == MOTHERSHIP_INDEX:
                continue
            arr = np.asarray(kf_kinematics_history[ri], dtype=np.float64).reshape(-1, 3)
            if arr.shape[0] == T_plot:
                kin_plot[:, ri, :] = arr
        _save_scout_kinematics_figure(
            sim_plot,
            kin_plot,
            n_robots=n,
            mothership_index=MOTHERSHIP_INDEX,
            wheel_radius_nominal_m=float(ekf_wheel_radius),
            base_length_nominal_m=float(ekf_base_length),
            out_path=_ROOT / "ekf_scout_kinematics_estimate.png",
        )
        if offline_root is not None:
            _save_scout_kinematics_figure(
                sim_plot,
                kin_plot,
                n_robots=n,
                mothership_index=MOTHERSHIP_INDEX,
                wheel_radius_nominal_m=float(ekf_wheel_radius),
                base_length_nominal_m=float(ekf_base_length),
                out_path=offline_root / "ekf_scout_kinematics.png",
            )

    def tracking_step() -> None:
        nonlocal x, encoders_prev, s_along_floor, nav_phase, skip_next_pose_read, scout_robot_dead, log_and_filter_scout_and_gt
        offline_step = len(offline_pose) if offline_root is not None else -1
        if skip_next_pose_read:
            skip_next_pose_read = False
        else:
            x = r.get_poses()
        encoders_curr = r.get_encoders()

        if (
            simulate_death
            and not nav_phase
            and not scout_robot_dead
            and SIMULATE_DEATH_SCOUT_INDEX < n
        ):
            p = x[:2, SIMULATE_DEATH_SCOUT_INDEX]
            for exy in enemy_positions:
                if (
                    float(np.linalg.norm(p - np.asarray(exy, dtype=np.float64).reshape(2)))
                    < SIMULATE_DEATH_ENEMY_PROXIMITY_M
                ):
                    scout_robot_dead = True
                    dxy = np.asarray(x[:2, SIMULATE_DEATH_SCOUT_INDEX], dtype=np.float64).reshape(2)
                    scout_goal_xy[:, SIMULATE_DEATH_SCOUT_INDEX] = dxy
                    routes[SIMULATE_DEATH_SCOUT_INDEX] = dxy.reshape(1, 2)
                    route_vertex_cumlen[:] = _route_vertex_cumlen_list(routes)
                    s_along_floor[SIMULATE_DEATH_SCOUT_INDEX] = 0.0
                    if r.show_figure:
                        # ~8 cm tall on the arena — readable but not full-screen (cf. title banner ~0.1 m).
                        fs = float(determine_font_size(r, 0.08))
                        death_overlay[0] = ax.text(
                            float(x[0, SIMULATE_DEATH_SCOUT_INDEX]),
                            float(x[1, SIMULATE_DEATH_SCOUT_INDEX]),
                            "\u00d7",
                            fontsize=fs,
                            color="#5c0000",
                            ha="center",
                            va="center",
                            fontweight="bold",
                            zorder=float(Map.VIZ_Z_DEATH_MARKER),
                            clip_on=False,
                        )
                    break

        waypoint_xy = np.zeros((2, n), dtype=np.float64)
        for i in range(n):
            path_i = np.asarray(routes[i], dtype=np.float64).reshape(-1, 2)
            m = path_i.shape[0]
            if m == 0:
                waypoint_xy[:, i] = x[:2, i]
                continue
            if m == 1:
                waypoint_xy[:, i] = path_i[0]
                continue
            robot_xy = x[:2, i]
            _, s_proj = _closest_point_on_polyline_xy_forward(
                path_i,
                robot_xy,
                s_along_floor[i],
                slack_back_m=PATH_TRACK_SLACK_BACK_M,
            )
            s_along_floor[i] = max(s_along_floor[i], float(s_proj))
            cumv = route_vertex_cumlen[i]
            if cumv is not None:
                s_i = float(s_along_floor[i])
                j = int(np.searchsorted(cumv, s_i, side="left"))
                for k in range(max(0, j - 1), min(m, j + 5)):
                    if float(np.linalg.norm(robot_xy - path_i[k])) < PATH_VERTEX_CATCHUP_M:
                        s_i = max(s_i, float(cumv[k]))
                s_along_floor[i] = max(s_along_floor[i], s_i)
            total_len = float(cumv[-1]) if cumv is not None else 0.0
            s_tgt = min(float(s_along_floor[i]) + PATH_LOOKAHEAD_DISTANCE_M, total_len)
            waypoint_xy[:, i] = _point_xy_at_arclength(path_i, s_tgt)

        dxu = unicycle_pose_controller(x, waypoint_xy)
        dxu = uni_barrier_cert(dxu, x)
        if simulate_death and scout_robot_dead:
            dxu[:, SIMULATE_DEATH_SCOUT_INDEX] = 0.0

        delta_L = encoders_curr[0, :] - encoders_prev[0, :]
        delta_R = encoders_curr[1, :] - encoders_prev[1, :]
        dphi_L = delta_L.astype(np.float64) * counts_to_rad
        dphi_R = delta_R.astype(np.float64) * counts_to_rad
        v_enc_all = (wheel_radius / 2.0) * (dphi_R + dphi_L) / dt
        w_enc_all = (wheel_radius / base_length) * (dphi_R - dphi_L) / dt

        if log_and_filter_scout_and_gt:
            for i in range(MOTHERSHIP_INDEX + 1, n):
                if simulate_death and scout_robot_dead and i == SIMULATE_DEATH_SCOUT_INDEX:
                    continue
                ekfs_vanilla[i].predict(float(dphi_R[i]), float(dphi_L[i]), dt)
                ekfs_params[i].predict(float(dphi_R[i]), float(dphi_L[i]), dt)
        encoders_prev = encoders_curr.copy()

        if log_and_filter_scout_and_gt:
            for i in range(MOTHERSHIP_INDEX + 1, n):
                robot_xy = x[:2, i]
                theta_gt = float(x[2, i])
                if nav_phase:
                    continue
                if simulate_death and scout_robot_dead and i == SIMULATE_DEATH_SCOUT_INDEX:
                    continue
                for lh_name, lh_xy in lighthouses.items():
                    d = float(np.linalg.norm(robot_xy - lh_xy))
                    last_t = lighthouse_last_update[i][lh_name]
                    d_min = float(Map.LIGHTHOUSE_MEASUREMENT_MIN_RANGE_M)
                    d_range_reliable = float(Map.LIGHTHOUSE_BEARING_AND_RANGE_MAX_M)
                    d_bearing_max = float(Map.LIGHTHOUSE_BEARING_SIGHTING_MAX_M)
                    in_sighting = d_min < d < d_bearing_max
                    use_range = d < d_range_reliable
                    if (
                        in_sighting
                        and time.time() - last_t > lighthouse_update_interval_s
                    ):
                        lighthouse_last_update[i][lh_name] = time.time()
                        z_b, z_r = _simulate_lighthouse_measurements(
                            robot_xy,
                            theta_gt,
                            lh_xy,
                            rng_meas,
                            R_bearing_var,
                            R_range_var,
                            include_bearing=True,
                            include_range=use_range,
                        )
                        ekfs_vanilla[i].process_beacon_update(
                            lh_xy,
                            z_b,
                            R_bearing_var,
                            z_r,
                            R_range_var if use_range else None,
                        )
                        ekfs_params[i].process_beacon_update(
                            lh_xy,
                            z_b,
                            R_bearing_var,
                            z_r,
                            R_range_var if use_range else None,
                        )
                        if offline_root is not None:
                            zb_f = float(z_b) if z_b is not None else float("nan")
                            zr_f = float(z_r) if z_r is not None else float("nan")
                            offline_lh_meas.append(
                                (
                                    offline_step,
                                    float(time.time()),
                                    int(i),
                                    str(lh_name),
                                    zb_f,
                                    zr_f,
                                    np.uint8(1 if use_range else 0),
                                    float(lh_xy[0]),
                                    float(lh_xy[1]),
                                )
                            )

                # Unknown landmarks: one EKF track per buoy / per enemy (unique ``landmark_type`` string).
                for bname, bxy in env_map.buoys.items():
                    bxy = np.asarray(bxy, dtype=np.float64).reshape(2)
                    d_lm = float(np.linalg.norm(robot_xy - bxy))
                    lk = f"buoy:{bname}"
                    last_lm = landmark_last_update[i][lk]
                    if (
                        lm_d_min_m < d_lm < lm_d_max_m
                        and time.time() - last_lm > lighthouse_update_interval_s
                    ):
                        landmark_last_update[i][lk] = time.time()
                        zb_lm, zr_lm = _simulate_lighthouse_measurements(
                            robot_xy,
                            theta_gt,
                            bxy,
                            rng_meas,
                            R_bearing_var,
                            R_range_var,
                            include_bearing=True,
                            include_range=True,
                        )
                        ekfs_vanilla[i].process_landmark_update(
                            f"{_LM_PREFIX_BUOY}{bname}",
                            float(zb_lm),
                            R_bearing_var,
                            float(zr_lm),
                            R_range_var,
                        )
                        ekfs_params[i].process_landmark_update(
                            f"{_LM_PREFIX_BUOY}{bname}",
                            float(zb_lm),
                            R_bearing_var,
                            float(zr_lm),
                            R_range_var,
                        )
                        if offline_root is not None:
                            offline_buoy_meas.append(
                                (
                                    offline_step,
                                    float(time.time()),
                                    int(i),
                                    str(bname),
                                    float(zb_lm),
                                    float(zr_lm),
                                    float(bxy[0]),
                                    float(bxy[1]),
                                )
                            )

                for ei, exy in enumerate(enemy_positions):
                    d_lm = float(np.linalg.norm(robot_xy - exy))
                    lk = f"enemy:{ei}"
                    last_lm = landmark_last_update[i][lk]
                    if (
                        lm_d_min_m < d_lm < lm_d_max_m
                        and time.time() - last_lm > lighthouse_update_interval_s
                    ):
                        landmark_last_update[i][lk] = time.time()
                        zb_lm, zr_lm = _simulate_lighthouse_measurements(
                            robot_xy,
                            theta_gt,
                            exy,
                            rng_meas,
                            R_bearing_var,
                            R_range_var,
                            include_bearing=True,
                            include_range=True,
                        )
                        ekfs_vanilla[i].process_landmark_update(
                            f"{_LM_PREFIX_ENEMY_SHIP}{ei}",
                            float(zb_lm),
                            R_bearing_var,
                            float(zr_lm),
                            R_range_var,
                        )
                        ekfs_params[i].process_landmark_update(
                            f"{_LM_PREFIX_ENEMY_SHIP}{ei}",
                            float(zb_lm),
                            R_bearing_var,
                            float(zr_lm),
                            R_range_var,
                        )
                        if offline_root is not None:
                            exy_arr = np.asarray(exy, dtype=np.float64).reshape(2)
                            offline_enemy_meas.append(
                                (
                                    offline_step,
                                    float(time.time()),
                                    int(i),
                                    int(ei),
                                    float(zb_lm),
                                    float(zr_lm),
                                    float(exy_arr[0]),
                                    float(exy_arr[1]),
                                )
                            )

        if offline_root is not None and log_and_filter_scout_and_gt:
            offline_pose.append(np.asarray(x, dtype=np.float64).copy())
            offline_enc.append(encoders_curr.astype(np.int32).copy())
            offline_ori.append(np.asarray(r.get_orientations(), dtype=np.float64).copy())
            offline_nav.append(1 if nav_phase else 0)
            offline_v.append(np.asarray(v_enc_all, dtype=np.float64).copy())
            offline_w.append(np.asarray(w_enc_all, dtype=np.float64).copy())

        if log_and_filter_scout_and_gt:
            for i in range(n):
                gt_history[i].append(
                    np.array(
                        [float(x[0, i]), float(x[1, i]), float(x[2, i])],
                        dtype=np.float64,
                    )
                )
                if i == MOTHERSHIP_INDEX:
                    landmark_history[i].append({})
                    landmark_history_params[i].append({})
                    continue
                st_v = np.asarray(ekfs_vanilla[i].state, dtype=float).reshape(-1)
                st_p = np.asarray(ekfs_params[i].state, dtype=float).reshape(-1)
                kf_history[i].append(
                    np.array([float(st_v[0]), float(st_v[1]), float(st_v[2])], dtype=np.float64)
                )
                kf_history_params[i].append(
                    np.array([float(st_p[0]), float(st_p[1]), float(st_p[2])], dtype=np.float64)
                )
                kf_kinematics_history[i].append(
                    np.asarray(st_p[3:6], dtype=np.float64).reshape(3).copy()
                )
                cov_var_pose_vanilla[i].append(_ekf_pose_and_param_variance_diag(ekfs_vanilla[i]))
                cov_var_pose_and_params[i].append(_ekf_pose_and_param_variance_diag(ekfs_params[i]))
                snap_v: dict[str, list[float]] = {}
                ekf_v = ekfs_vanilla[i]
                b0v = int(ekf_v.base_state_size)
                for lm_idx in sorted(ekf_v.landmarks.keys()):
                    lm_type = str(ekf_v.landmarks[lm_idx])
                    j0 = b0v + 2 * int(lm_idx)
                    snap_v[lm_type] = [float(st_v[j0]), float(st_v[j0 + 1])]
                landmark_history[i].append(snap_v)
                snap_p: dict[str, list[float]] = {}
                ekf_p = ekfs_params[i]
                b0p = int(ekf_p.base_state_size)
                for lm_idx in sorted(ekf_p.landmarks.keys()):
                    lm_type = str(ekf_p.landmarks[lm_idx])
                    j0 = b0p + 2 * int(lm_idx)
                    snap_p[lm_type] = [float(st_p[j0]), float(st_p[j0 + 1])]
                landmark_history_params[i].append(snap_p)

        for i in range(n):
            if not gt_history[i]:
                continue
            Gi = np.asarray(gt_history[i], dtype=np.float64).reshape(-1, 3)
            gt_trails[i].set_data(Gi[:, 0], Gi[:, 1])
            if i != MOTHERSHIP_INDEX and kf_history[i]:
                Ki = np.asarray(kf_history[i], dtype=np.float64).reshape(-1, 3)
                kf_trails[i].set_data(Ki[:, 0], Ki[:, 1])
            elif i != MOTHERSHIP_INDEX:
                kf_trails[i].set_data([], [])

            if i == MOTHERSHIP_INDEX:
                continue

            st = np.asarray(ekfs_vanilla[i].state, dtype=float).reshape(-1)
            P_full = np.asarray(ekfs_vanilla[i].P, dtype=np.float64)
            if P_full.ndim != 2 or P_full.shape[0] != P_full.shape[1]:
                P_full = np.squeeze(P_full)
            Pxy = P_full[:2, :2]
            ew, eh, eang = _xy_cov_ellipse_width_height_angle_deg(Pxy, cov_n_sigma)
            cov_ellipses[i].set_center((float(st[0]), float(st[1])))
            cov_ellipses[i].set_width(ew)
            cov_ellipses[i].set_height(eh)
            cov_ellipses[i].set_angle(eang)

            ekf_i = ekfs_vanilla[i]
            b0 = int(ekf_i.base_state_size)
            # One ellipse per landmark index (each index is a unique object id in ``landmarks``).
            lm_indices = sorted(ekf_i.landmarks.keys())
            le_list = landmark_ellipses[i]
            while len(le_list) < len(lm_indices):
                ell_lm = mpatches.Ellipse(
                    (0.0, 0.0),
                    width=1e-6,
                    height=1e-6,
                    angle=0.0,
                    fill=False,
                    edgecolor="#b8860b",
                    linewidth=1.15,
                    linestyle=":",
                    zorder=float(Map.VIZ_Z_LM_COV),
                )
                ax.add_patch(ell_lm)
                le_list.append(ell_lm)
            while len(le_list) > len(lm_indices):
                old_e = le_list.pop()
                old_e.remove()
            for j, lm_idx in enumerate(lm_indices):
                lm_type = str(ekf_i.landmarks[lm_idx])
                j0 = b0 + 2 * int(lm_idx)
                lx = float(st[j0])
                ly = float(st[j0 + 1])
                sl = slice(j0, j0 + 2)
                P_lm = P_full[sl, sl]
                ew_lm, eh_lm, eang_lm = _xy_cov_ellipse_width_height_angle_deg(P_lm, cov_n_sigma)
                e_lm = le_list[j]
                e_lm.set_center((lx, ly))
                e_lm.set_width(ew_lm)
                e_lm.set_height(eh_lm)
                e_lm.set_angle(eang_lm)
                if lm_type.startswith(_LM_PREFIX_BUOY):
                    e_lm.set_edgecolor("#b8860b")
                    e_lm.set_linewidth(1.25)
                elif lm_type.startswith(_LM_PREFIX_ENEMY_SHIP):
                    e_lm.set_edgecolor(_VIZ_ENEMY_LM_EDGE)
                    e_lm.set_linewidth(2.35)
                else:
                    e_lm.set_edgecolor("0.35")
                    e_lm.set_linewidth(1.15)

        if simulate_death and death_overlay[0] is not None:
            t = death_overlay[0]
            t.set_position((float(x[0, SIMULATE_DEATH_SCOUT_INDEX]), float(x[1, SIMULATE_DEATH_SCOUT_INDEX])))

        r.set_velocities(np.arange(n), dxu)
        r.step()

    _flash_center_title(r, "Scouting Phase")

    # --- Scout phase (all must reach scout route endpoints) ---
    while not at_position(x, scout_goal_xy)[0]:
        tracking_step()

    _flash_center_title(r, "Map Fusion Phase")

    fusion_report = _build_map_fusion_landmark_report(ekfs_vanilla, env_map, enemy_positions)
    _apply_map_fusion_to_ekf_landmarks(ekfs_vanilla, fusion_report)
    _apply_map_fusion_to_ekf_landmarks(ekfs_params, fusion_report)
    # Kinematics plot uses estimates through map fusion only; logging/EKF for scouts stops before nav.
    _save_kinematics_figures_after_fusion()
    log_and_filter_scout_and_gt = False

    # Banner already removed by ``_flash_center_title``; hold the scouting map briefly, then redraw fused.
    if r.show_figure:
        r._fig.canvas.draw_idle()
        r._fig.canvas.flush_events()
        plt.pause(3.0)

    _refresh_plot_with_fused_landmarks(
        r, fusion_report, landmark_ellipses, cov_n_sigma, fused_landmark_overlay
    )
    if r.show_figure:
        r._fig.canvas.draw_idle()
        r._fig.canvas.flush_events()
        plt.pause(4.0)

    # Map fusion (continued): stamp 2σ enemy costs (P + σ_r² I), land dilation for A*, plan nav.
    enemy_gt_xy = np.stack(enemy_positions, axis=0)
    env_map.apply_fused_enemy_costs_for_nav_planning(
        fusion_report,
        router,
        enemy_landmark_prefix=_LM_PREFIX_ENEMY_SHIP,
        enemy_ground_truth_xy=enemy_gt_xy,
        peak_cost=500.0,
        n_sigma=2.0,
    )
    env_map.refresh_planning_cost_visual()
    if r.show_figure:
        r._fig.canvas.draw_idle()
        r._fig.canvas.flush_events()
        plt.pause(3.0)

    if not skip_nav:
        _flash_center_title(r, "Nav Phase")
        x = r.get_poses()
        starts_xy = np.asarray(x[:2, :], dtype=np.float64).reshape(2, n)
        # Scouts must track ``scout_goal_xy`` (same target as ``nav_goal_xy``), not an empty waypoint
        # list (which ``Router`` treats as “hold at nav start only”). Otherwise barrier motion can push
        # them >``create_at_position``'s default 5 cm from ``scout_goal_xy`` while they are never
        # commanded back — ``while not at_position(x, nav_goal_xy)`` never becomes true.
        nav_waypoints: list[np.ndarray] = []
        for i in range(n):
            if i == MOTHERSHIP_INDEX:
                nav_waypoints.append(MOTHERSHIP_NAV_GOAL_XY.reshape(1, 2))
            else:
                nav_waypoints.append(scout_goal_xy[:, i].reshape(1, 2))
        routes[:] = router.routes_through_waypoints(starts_xy, nav_waypoints)
        route_vertex_cumlen[:] = _route_vertex_cumlen_list(routes)
        s_along_floor[:] = 0.0
        nav_phase = True
        skip_next_pose_read = True

        nav_goal_xy = np.zeros((2, n), dtype=np.float64)
        nav_goal_xy[:, MOTHERSHIP_INDEX] = MOTHERSHIP_NAV_GOAL_XY
        for i in range(MOTHERSHIP_INDEX + 1, n):
            nav_goal_xy[:, i] = scout_goal_xy[:, i]

        while not at_position(x, nav_goal_xy)[0]:
            tracking_step()

        nav_phase = False
        skip_next_pose_read = False
    else:
        nav_phase = False
        skip_next_pose_read = False

    rmse_vanilla = _build_rmse_report(
        gt_history, kf_history, landmark_history, env_map, enemy_positions
    )
    rmse_params = _build_rmse_report(
        gt_history, kf_history_params, landmark_history_params, env_map, enemy_positions
    )
    _print_rmse_report_dual(rmse_vanilla, rmse_params, fusion_report)
    base_rmse = rmse_out if rmse_out is not None else (_ROOT / "experiment_rmse_metrics.yaml")
    path_vanilla, path_params = _dual_rmse_metric_paths(base_rmse)
    vanilla_payload = {**rmse_vanilla, "map_fusion_landmarks": fusion_report}
    _save_rmse_yaml(vanilla_payload, path_vanilla)
    _save_rmse_yaml(rmse_params, path_params)
    print(f"\nRMSE metrics (vanilla EKF + map fusion) saved to {path_vanilla.resolve()}")
    print(f"RMSE metrics (parameter-augmented EKF) saved to {path_params.resolve()}")

    if offline_root is not None and meas_seed_for_meta is not None:
        enemy_xy = np.stack(
            [np.asarray(p, dtype=np.float64).reshape(2) for p in enemy_positions],
            axis=0,
        )
        _write_offline_experiment_npz(
            offline_root,
            dt=float(dt),
            n_robots=n,
            mothership_index=MOTHERSHIP_INDEX,
            measurement_rng_seed=int(meas_seed_for_meta),
            lighthouse_update_interval_s=float(lighthouse_update_interval_s),
            R_bearing_var=R_bearing_var,
            R_range_var=R_range_var,
            wheel_radius=float(wheel_radius),
            base_length=float(base_length),
            counts_to_rad=float(counts_to_rad),
            process_noise=process_noise,
            lighthouses=lighthouses,
            enemy_xy=enemy_xy,
            nav_goal_xy=np.asarray(MOTHERSHIP_NAV_GOAL_XY, dtype=np.float64).reshape(2),
            offline_pose=offline_pose,
            offline_enc=offline_enc,
            offline_ori=offline_ori,
            offline_nav=offline_nav,
            offline_v=offline_v,
            offline_w=offline_w,
            offline_lh=offline_lh_meas,
            offline_enemy=offline_enemy_meas,
            offline_buoy=offline_buoy_meas,
            gt_history=gt_history,
            kf_history=kf_history,
            landmark_history=landmark_history,
            kf_history_params=kf_history_params,
            landmark_history_params=landmark_history_params,
            routes=routes,
            fusion_report=fusion_report,
            kf_kinematics_history=kf_kinematics_history,
            cov_var_pose_vanilla=cov_var_pose_vanilla,
            cov_var_pose_and_params=cov_var_pose_and_params,
            skip_nav=skip_nav,
            ekf_prior_wheel_radius_m=float(ekf_wheel_radius),
            ekf_prior_base_length_m=float(ekf_base_length),
        )

    r.debug()


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Decentralized maritime EKF per scout + lighthouse bearing/range (body-frame bearing). "
            f"Robot {MOTHERSHIP_INDEX} is the mothership (no filter / SLAM / RMSE)."
        )
    )
    p.add_argument("--cycles", type=int, default=2)
    p.add_argument("--process_noise", type=float, nargs=3, default=[0.001, 0.001, 0.001])
    p.add_argument(
        "--cov_n_sigma",
        type=float,
        default=2.0,
        help="Half-width of EKF (x,y) covariance ellipse in standard deviations (e.g. 2 for ~95%% contour).",
    )
    p.add_argument(
        "--rmse-out",
        type=Path,
        default=None,
        help=(
            "Base path for RMSE metrics (YAML-compatible JSON); two files are written: "
            "<stem>_vanilla<suffix> (includes map_fusion_landmarks) and <stem>_parameter<suffix>. "
            "Default base: maritime-state-estimation/experiment_rmse_metrics.yaml"
        ),
    )
    p.add_argument(
        "--simulate-death",
        action="store_true",
        help=(
            f"Viz only: robot {SIMULATE_DEATH_SCOUT_INDEX} stops and shows a large × when within "
            f"~{SIMULATE_DEATH_ENEMY_PROXIMITY_M * 100:.0f} cm of an enemy during scouting (default: off)."
        ),
    )
    p.add_argument(
        "--save-offline",
        type=Path,
        nargs="?",
        const=Path("offline_capture"),
        default=None,
        help=(
            "Write NPZ bundles under this directory (subdirs poses/, encoders/, orientations/, "
            "lighthouse_measurements/, enemy_measurements/, buoy_measurements/, predicted_paths/, metadata/). "
            "Default directory if flag is given with no path: ./offline_capture"
        ),
    )
    p.add_argument(
        "--measurement-seed",
        type=int,
        default=None,
        help="Seed for simulated lighthouse / landmark measurements (also used when --save-offline draws RNG).",
    )
    p.add_argument(
        "--skip-nav",
        action="store_true",
        help=(
            "Skip the mothership nav phase (no extra ``tracking_step`` after map fusion). Scout EKF, "
            "GT/KF/landmark logging, and offline bundles end right after map fusion for faster iteration."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    run_experiment(
        cycles=args.cycles,
        process_noise=tuple(args.process_noise),
        cov_n_sigma=args.cov_n_sigma,
        rmse_out=args.rmse_out,
        simulate_death=bool(args.simulate_death),
        offline_log_root=args.save_offline,
        measurement_rng_seed=args.measurement_seed,
        skip_nav=bool(args.skip_nav),
    )


if __name__ == "__main__":
    main()
