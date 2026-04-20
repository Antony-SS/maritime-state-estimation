"""
Maritime-style experiment: multi-robot scouting polylines, one decentralized UnicycleEKF per robot,
tiered lighthouse updates (bearing + range when close, bearing-only farther out; see ``Map``),
and SLAM-style bearing/range to buoys and static enemy ships when nearby (unknown landmarks).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import time
import numpy as np
import matplotlib.patches as mpatches

import rps.robotarium as robotarium
from filters import UnicycleEKF, wrap_angle
from rps.utilities.barrier_certificates import create_uni_barrier_certificate
from rps.utilities.controllers import create_uni_position_controller
from rps.utilities.misc import create_at_position
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
# robot keeps a single track per buoy / enemy (e.g. ``buoy:B2``, ``enemy ship:0``). Visualization
# colors use these prefixes.
_LM_PREFIX_BUOY = "buoy:"
_LM_PREFIX_ENEMY_SHIP = "enemy ship:"

# Arc-length lookahead (m) from forward projection on the polyline.
PATH_LOOKAHEAD_DISTANCE_M = 0.11
# Forward projection slack for self-overlapping paths (e.g. lawn wiggle).
PATH_TRACK_SLACK_BACK_M = 0.22
# Original experiment cap (~gentle); bump slightly if you want faster transits.
PATH_TRACK_VEL_LIMIT_M_S = 0.09
# If the robot is within this distance of an upcoming vertex, advance arc-length to that vertex
# (helps tight A* corners where projection alone creeps slowly — e.g. scout 1).
PATH_VERTEX_CATCHUP_M = 0.065

# Distinct colors for GT / KF trails and covariance ellipses (Robotarium default N is small).
_ROBOT_COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown")



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


def run_experiment(
    cycles: int = 2,
    process_noise: tuple[float, float, float] = (0.0001, 0.0001, 0.0001),
    cov_n_sigma: float = 2.0,
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
    dt = r.TIME_STEP

    counts_to_rad = 2 * np.pi / (r.ENCODER_COUNTS_PER_REVOLUTION * r.MOTOR_GEAR_RATIO)
    encoder_noise_std = r.ENCODER_NOISE_STD
    encoder_ang_vel_var = (encoder_noise_std * counts_to_rad / dt) ** 2
    encoder_noise_matrix = np.eye(2) * encoder_ang_vel_var
    process_noise_matrix = np.eye(3) * np.array(process_noise)

    ekfs: list[UnicycleEKF] = []
    for i in range(n):
        ekfs.append(
            UnicycleEKF(
                initial_state=np.asarray(initial_conditions[:, i], dtype=float).reshape(-1),
                initial_covariance=np.zeros((3, 3)),
                b=base_length,
                r=wheel_radius,
                M=encoder_noise_matrix,
                Q=process_noise_matrix,
            )
        )

    R_range_var = float(Map.LIGHTHOUSE_RANGE_MEAS_VAR_M2)
    R_bearing_var = float(Map.LIGHTHOUSE_BEARING_MEAS_VAR_RAD2)
    rng_meas = np.random.default_rng()

    ax = r._axes_handle
    colors = [_ROBOT_COLORS[i % len(_ROBOT_COLORS)] for i in range(n)]

    gt_trails: list = []
    kf_trails: list = []
    cov_ellipses: list[mpatches.Ellipse] = []
    # Per-robot unknown-landmark (x,y) covariance ellipses (grows with SLAM map).
    landmark_ellipses: list[list[mpatches.Ellipse]] = [[] for _ in range(n)]
    legend_handles: list = []
    for i in range(n):
        c = colors[i]
        gt_line, = ax.plot([], [], "-", color=c, linewidth=1.2, label=f"GT {i}", zorder=4)
        kf_line, = ax.plot([], [], "--", color=c, linewidth=1.2, label=f"EKF {i}", zorder=4)
        ell = mpatches.Ellipse(
            (0.0, 0.0),
            width=1e-6,
            height=1e-6,
            angle=0.0,
            fill=False,
            edgecolor=c,
            linewidth=1.1,
            linestyle=(0, (3, 2)),
            zorder=6,
            label=f"P_xy {i} ({cov_n_sigma:g}σ)",
        )
        ax.add_patch(ell)
        gt_trails.append(gt_line)
        kf_trails.append(kf_line)
        cov_ellipses.append(ell)
        legend_handles.extend([gt_line, kf_line, ell])

    _leg_lm_buoy = mpatches.Ellipse(
        (0.0, 0.0),
        width=1e-6,
        height=1e-6,
        angle=0.0,
        fill=False,
        edgecolor="#b8860b",
        linewidth=1.1,
        linestyle=":",
        label=f"LM buoy P_xy ({cov_n_sigma:g}σ)",
    )
    _leg_lm_enemy = mpatches.Ellipse(
        (0.0, 0.0),
        width=1e-6,
        height=1e-6,
        angle=0.0,
        fill=False,
        edgecolor="#922b21",
        linewidth=1.1,
        linestyle=":",
        label=f"LM enemy P_xy ({cov_n_sigma:g}σ)",
    )
    legend_handles.extend([_leg_lm_buoy, _leg_lm_enemy])

    ax.legend(handles=legend_handles, loc="upper left", fontsize=7, ncol=2)

    gt_history: list[list[np.ndarray]] = [[] for _ in range(n)]
    kf_history: list[list[np.ndarray]] = [[] for _ in range(n)]
    encoders_prev = r.get_encoders()

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
    home_xy = initial_conditions[:2, :].copy()

    def tracking_step(poses: np.ndarray | None = None) -> None:
        nonlocal x, encoders_prev, s_along_floor
        if poses is not None:
            x = poses
        else:
            x = r.get_poses()
        encoders_curr = r.get_encoders()

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

        delta_L = encoders_curr[0, :] - encoders_prev[0, :]
        delta_R = encoders_curr[1, :] - encoders_prev[1, :]
        dphi_L = delta_L.astype(np.float64) * counts_to_rad
        dphi_R = delta_R.astype(np.float64) * counts_to_rad
        v_enc_all = (wheel_radius / 2.0) * (dphi_R + dphi_L) / dt
        w_enc_all = (wheel_radius / base_length) * (dphi_R - dphi_L) / dt

        for i in range(n):
            ekfs[i].predict(float(v_enc_all[i]), float(w_enc_all[i]), dt)
        encoders_prev = encoders_curr.copy()

        for i in range(n):
            robot_xy = x[:2, i]
            theta_gt = float(x[2, i])
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
                    ekfs[i].process_beacon_update(
                        lh_xy,
                        z_b,
                        R_bearing_var,
                        z_r,
                        R_range_var if use_range else None,
                    )

            # Unknown landmarks: one EKF track per buoy / per enemy (unique ``landmark_type`` string).
            # Using a single label for all buoys (or gating association on ``‖m_landmark - p_robot‖``)
            # spawns duplicate tracks and mis-associations; landmark–robot coupling in ``P`` then drags
            # the pose estimate. Landmarks correctly have zero process noise; the bug was data association.
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
                    ekfs[i].process_landmark_update(
                        f"{_LM_PREFIX_BUOY}{bname}",
                        float(zb_lm),
                        R_bearing_var,
                        float(zr_lm),
                        R_range_var,
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
                    ekfs[i].process_landmark_update(
                        f"{_LM_PREFIX_ENEMY_SHIP}{ei}",
                        float(zb_lm),
                        R_bearing_var,
                        float(zr_lm),
                        R_range_var,
                    )

        for i in range(n):
            gt_history[i].append(x[:2, i].copy())
            st = np.asarray(ekfs[i].state, dtype=float).reshape(-1)
            kf_history[i].append(np.array([st[0], st[1]], dtype=float))

        for i in range(n):
            if not gt_history[i]:
                continue
            Gi = np.asarray(gt_history[i])
            Ki = np.asarray(kf_history[i])
            gt_trails[i].set_data(Gi[:, 0], Gi[:, 1])
            kf_trails[i].set_data(Ki[:, 0], Ki[:, 1])

            st = np.asarray(ekfs[i].state, dtype=float).reshape(-1)
            P_full = np.asarray(ekfs[i].P, dtype=np.float64)
            if P_full.ndim != 2 or P_full.shape[0] != P_full.shape[1]:
                P_full = np.squeeze(P_full)
            Pxy = P_full[:2, :2]
            ew, eh, eang = _xy_cov_ellipse_width_height_angle_deg(Pxy, cov_n_sigma)
            cov_ellipses[i].set_center((float(st[0]), float(st[1])))
            cov_ellipses[i].set_width(ew)
            cov_ellipses[i].set_height(eh)
            cov_ellipses[i].set_angle(eang)

            ekf_i = ekfs[i]
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
                    linewidth=0.95,
                    linestyle=":",
                    zorder=5,
                )
                ax.add_patch(ell_lm)
                le_list.append(ell_lm)
            while len(le_list) > len(lm_indices):
                old_e = le_list.pop()
                old_e.remove()
            for j, lm_idx in enumerate(lm_indices):
                lm_type = str(ekf_i.landmarks[lm_idx])
                lx = float(st[3 + 2 * lm_idx])
                ly = float(st[3 + 2 * lm_idx + 1])
                sl = slice(3 + 2 * lm_idx, 3 + 2 * lm_idx + 2)
                P_lm = P_full[sl, sl]
                ew_lm, eh_lm, eang_lm = _xy_cov_ellipse_width_height_angle_deg(P_lm, cov_n_sigma)
                e_lm = le_list[j]
                e_lm.set_center((lx, ly))
                e_lm.set_width(ew_lm)
                e_lm.set_height(eh_lm)
                e_lm.set_angle(eang_lm)
                if lm_type.startswith(_LM_PREFIX_BUOY):
                    e_lm.set_edgecolor("#b8860b")
                elif lm_type.startswith(_LM_PREFIX_ENEMY_SHIP):
                    e_lm.set_edgecolor("#922b21")
                else:
                    e_lm.set_edgecolor("0.35")

        r.set_velocities(np.arange(n), dxu)
        r.step()

    # --- Scout phase (all must reach scout route endpoints) ---
    while not at_position(x, scout_goal_xy)[0]:
        tracking_step()

    # --- Return to home: A* only from current pose to each robot's initial (x, y); no lawn ---
    # Robotarium allows at most one ``get_poses()`` between ``step()`` calls — plan return routes
    # on the first loop iteration using that same pose (no extra ``get_poses`` before ``tracking_step``).
    s_along_floor[:] = 0.0
    need_return_routes = True
    while True:
        x = r.get_poses()
        if need_return_routes:
            routes = router.generate_return_to_home_routes(x[:2, :], initial_conditions)
            if any(np.asarray(rt, dtype=float).size == 0 for rt in routes):
                r.debug()
                return
            route_vertex_cumlen = _route_vertex_cumlen_list(routes)
            need_return_routes = False
        if at_position(x, home_xy)[0]:
            break
        tracking_step(poses=x)

    r.debug()


def parse_args():
    p = argparse.ArgumentParser(
        description="Decentralized maritime EKF per robot + lighthouse bearing/range (body-frame bearing)."
    )
    p.add_argument("--cycles", type=int, default=2)
    p.add_argument("--process_noise", type=float, nargs=3, default=[0.001, 0.001, 0.001])
    p.add_argument(
        "--cov_n_sigma",
        type=float,
        default=2.0,
        help="Half-width of EKF (x,y) covariance ellipse in standard deviations (e.g. 2 for ~95%% contour).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    run_experiment(
        cycles=args.cycles,
        process_noise=tuple(args.process_noise),
        cov_n_sigma=args.cov_n_sigma,
    )


if __name__ == "__main__":
    main()
