"""
Maritime-style experiment: multi-robot scouting polylines, encoder-only EKF on the mothership only,
range updates to map buoys when the mothership true pose is within range gate.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import time
import numpy as np
import matplotlib.patches as mpatches

import rps.robotarium as robotarium
from rps.examples.state_estimation.extended_kalman_filter.uni_ekf import UnicycleEKF
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

# One row per robot: index 0 = mothership. Shape (N, 3) → transpose to (3, N) for Robotarium.
START_POINTS = np.array(
    [
        [-1.5, 0.90, 0.0],  # Mothership
        [-1.5, 0.30, 0.0],  # Scout 1
        [-.95, 0.325, 0.0],  # Scout 2
        [-1.2, 0.70, 0.0],  # Scout 3
    ],
    dtype=np.float64,
)
# Arc-length lookahead (m) from forward projection on the polyline.
PATH_LOOKAHEAD_DISTANCE_M = 0.11
# Forward projection slack for self-overlapping paths (e.g. lawn wiggle).
PATH_TRACK_SLACK_BACK_M = 0.22
# Original experiment cap (~gentle); bump slightly if you want faster transits.
PATH_TRACK_VEL_LIMIT_M_S = 0.09
# If the robot is within this distance of an upcoming vertex, advance arc-length to that vertex
# (helps tight A* corners where projection alone creeps slowly — e.g. scout 1).
PATH_VERTEX_CATCHUP_M = 0.065

MOTHERSHIP_INDEX = 0

def run_experiment(
    cycles: int = 2,
    process_noise: tuple[float, float, float] = (0.0001, 0.0001, 0.0001),
    range_gate_m: float = 0.5,
    range_meas_std_m: float = 0.025,
    cov_n_sigma: float = 2.0,
) -> None:
    _ = cycles  # reserved for multi-phase / repeat missions

    n = int(START_POINTS.shape[0])
    initial_conditions = START_POINTS.T

    r = robotarium.Robotarium(
        number_of_robots=n,
        show_figure=True,
        initial_conditions=initial_conditions,
        sim_in_real_time=True,
    )

    env_map = Map(r)
    router = Router(env_map)
    routes = router.generate_scouting_phase_routes(initial_conditions)
    final_goal_xy = np.stack(
        [np.asarray(r[-1], dtype=np.float64).reshape(2) for r in routes],
        axis=1,
    )

    route_vertex_cumlen: list[np.ndarray | None] = []
    for rt in routes:
        rr = np.asarray(rt, dtype=np.float64).reshape(-1, 2)
        if rr.shape[0] <= 1:
            route_vertex_cumlen.append(None)
        else:
            seg = np.linalg.norm(np.diff(rr, axis=0), axis=1)
            route_vertex_cumlen.append(np.concatenate([[0.0], np.cumsum(seg)]))

    beacons = env_map.buoys

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

    ekf = UnicycleEKF(
        initial_state=np.asarray(initial_conditions[:, MOTHERSHIP_INDEX], dtype=float).reshape(-1),
        initial_covariance=np.zeros((3, 3)),
        b=base_length,
        r=wheel_radius,
        M=encoder_noise_matrix,
        Q=process_noise_matrix,
    )

    R_range = np.array([[range_meas_std_m**2]], dtype=float)
    rng_meas = np.random.default_rng()

    ax = r._axes_handle
    gt_trail, = ax.plot([], [], "b-", linewidth=1.5, label="GT mothership", zorder=4)
    kf_trail, = ax.plot([], [], "r--", linewidth=1.5, label="EKF mothership", zorder=4)
    cov_ellipse = mpatches.Ellipse(
        (0.0, 0.0),
        width=1e-6,
        height=1e-6,
        angle=0.0,
        fill=False,
        edgecolor="tab:purple",
        linewidth=1.4,
        linestyle=(0, (4, 3)),
        zorder=6,
        label=f"EKF P_xy ({cov_n_sigma:g}σ)",
    )
    ax.add_patch(cov_ellipse)
    ax.legend(loc="upper left", fontsize=9)

    gt_history: list[np.ndarray] = []
    kf_history: list[np.ndarray] = []
    encoders_prev = r.get_encoders()

    beacon_last_update_time = dict.fromkeys(beacons, 0.0)
    beacon_update_interval_s = float(np.random.uniform(0.25, 0.5))

    if any(np.asarray(rt, dtype=float).size == 0 for rt in routes):
        r.debug()
        return

    s_along_floor = np.zeros(n, dtype=np.float64)

    while not at_position(x, final_goal_xy)[0]:
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

        ekf.predict(float(v_enc_all[MOTHERSHIP_INDEX]), float(w_enc_all[MOTHERSHIP_INDEX]), dt)
        encoders_prev = encoders_curr.copy()

        mothership_xy = x[:2, MOTHERSHIP_INDEX]
        for beacon_name, beacon in beacons.items():
            d = float(np.linalg.norm(mothership_xy - beacon))
            last_update_time = beacon_last_update_time[beacon_name]
            if d <= range_gate_m and d > 0.1 and time.time() - last_update_time > beacon_update_interval_s:
                beacon_last_update_time[beacon_name] = time.time()
                z = d + float(rng_meas.normal(0.0, range_meas_std_m))
                ekf.update_range(beacon, z, R_range=R_range)

        gt_history.append(mothership_xy.copy())
        kf_history.append(np.array([ekf.state[0], ekf.state[1]], dtype=float))

        if gt_history:
            G = np.asarray(gt_history)
            K = np.asarray(kf_history)
            gt_trail.set_data(G[:, 0], G[:, 1])
            kf_trail.set_data(K[:, 0], K[:, 1])

        st = np.asarray(ekf.state, dtype=float).reshape(-1)
        P_full = np.squeeze(np.asarray(ekf.P, dtype=float)).reshape(3, 3)
        Pxy = P_full[:2, :2]
        ew, eh, eang = _xy_cov_ellipse_width_height_angle_deg(Pxy, cov_n_sigma)
        cov_ellipse.set_center((float(st[0]), float(st[1])))
        cov_ellipse.set_width(ew)
        cov_ellipse.set_height(eh)
        cov_ellipse.set_angle(eang)

        r.set_velocities(np.arange(n), dxu)
        r.step()

    r.debug()


def parse_args():
    p = argparse.ArgumentParser(description="Maritime encoder EKF (mothership) + beacon range updates")
    p.add_argument("--cycles", type=int, default=2)
    p.add_argument("--process_noise", type=float, nargs=3, default=[0.001, 0.001, 0.001])
    p.add_argument("--range_gate_m", type=float, default=0.85)
    p.add_argument("--range_meas_std_m", type=float, default=0.006)
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
        range_gate_m=args.range_gate_m,
        range_meas_std_m=args.range_meas_std_m,
        cov_n_sigma=args.cov_n_sigma,
    )


if __name__ == "__main__":
    main()
