"""
Maritime-style experiment: rectangle patrol, encoder-only EKF prediction,
range updates to map buoys when the true pose is within range gate.
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
from rps.utilities.controllers import create_pose_controller_hybrid
from rps.utilities.misc import create_at_pose
from utilities import _xy_cov_ellipse_width_height_angle_deg

# Allow `from map import Map` when running from this directory
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from map import Map  # noqa: E402

RECTANGLE_WAYPOINTS = [
    [-1.25, -0.75, 0.0],
    [1.25, -0.75, 0.0],
    [1.25, 0.75, 0.0],
    [-1.25, 0.75, 0.0],
]


def run_experiment(
    cycles: int = 2,
    process_noise: tuple[float, float, float] = (0.0001, 0.0001, 0.0001),
    range_gate_m: float = 0.5,
    range_meas_std_m: float = 0.025,
    cov_n_sigma: float = 2.0,
) -> None:
    goal_points = np.array(RECTANGLE_WAYPOINTS * cycles).reshape(-1, 3)
    N = 1
    initial_conditions = goal_points[0].reshape(3, 1)

    r = robotarium.Robotarium(
        number_of_robots=N,
        show_figure=True,
        initial_conditions=initial_conditions,
        sim_in_real_time=True,
    )
    map = Map(r)

    beacons = map.buoys

    unicycle_pose_controller = create_pose_controller_hybrid()
    uni_barrier_cert = create_uni_barrier_certificate()
    at_pose = create_at_pose()

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
        initial_state=initial_conditions.flatten(),
        initial_covariance=np.zeros((3, 3)),
        b=base_length,
        r=wheel_radius,
        M=encoder_noise_matrix,
        Q=process_noise_matrix,
    )

    R_range = np.array([[range_meas_std_m**2]], dtype=float)
    rng_meas = np.random.default_rng()

    ax = r._axes_handle
    gt_trail, = ax.plot([], [], "b-", linewidth=1.5, label="Ground truth", zorder=4)
    kf_trail, = ax.plot([], [], "r--", linewidth=1.5, label="EKF (enc + range)", zorder=4)
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
    beacon_update_interval_rng = np.random.uniform(0.25, 0.5) # seconds

    for waypoint in goal_points:
        while not at_pose(x, waypoint.reshape(3, 1))[0]:
            x = r.get_poses()
            encoders_curr = r.get_encoders()

            dxu = unicycle_pose_controller(x, waypoint.reshape(3, 1))
            dxu = uni_barrier_cert(dxu, x)

            delta_L = encoders_curr[0, 0] - encoders_prev[0, 0]
            delta_R = encoders_curr[1, 0] - encoders_prev[1, 0]
            dphi_L = delta_L * counts_to_rad
            dphi_R = delta_R * counts_to_rad
            v_enc = (wheel_radius / 2) * (dphi_R + dphi_L) / dt
            w_enc = (wheel_radius / base_length) * (dphi_R - dphi_L) / dt

            ekf.predict(v_enc, w_enc, dt)
            encoders_prev = encoders_curr.copy()

            robot_xy = x[:2, 0]
            for beacon_name, beacon in beacons.items():
                d = float(np.linalg.norm(robot_xy - beacon))
                last_update_time = beacon_last_update_time[beacon_name]
                if d <= range_gate_m and d > 0.1 and time.time() - last_update_time > beacon_update_interval_rng:
                    beacon_last_update_time[beacon_name] = time.time()
                    z = d + float(rng_meas.normal(0.0, range_meas_std_m))
                    ekf.update_range(beacon, z, R_range=R_range)

            gt_history.append(robot_xy.copy())
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

            r.set_velocities(np.arange(N), dxu)
            r.step()

    r.debug()


def parse_args():
    p = argparse.ArgumentParser(description="Maritime encoder EKF + beacon range updates")
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
