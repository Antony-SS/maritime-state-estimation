"""
Filter variants to be used for the Kalman Filters final project.
"""


import numpy as np
import time
from typing import Optional


def wrap_angle(angle):
    """Constrain angle to [-pi, pi]"""
    return np.arctan2(np.sin(angle), np.cos(angle))


class UnicycleEKF:
    """ Vanilla EKF For Unicycle model that can grow it's state vector as it sees landmarks. 
    To be clear on naming.  Beacons are things we have aprior knowledge of (not a part of the state vector).  
    Landmarks are things we don't have aprior knowledge of and are estimating (part of the state vector). 

    If estimate_robot_parameters is True, the base state is 6D: (x, y, theta, wheel_radius_right, wheel_radius_left, base_length), then landmarks.

    State vector (pose only): [x, y, theta, landmark_1, landmark_2, ...]
    State vector (with estimated kinematic parameters): [x, y, theta, r_right, r_left, base_length, landmark_1, ...]
    r: float | (wheel radius right, wheel radius left) — scalar uses the same radius for both wheels
    b: float = base length
    """

    def __init__(self, initial_state: np.ndarray, initial_covariance: np.ndarray, b: float, r: float | tuple[float, float], M: np.ndarray, Q: np.ndarray, estimate_robot_parameters: bool = False, parameter_uncertainty: float = 0.0001):
        self.estimate_robot_parameters = estimate_robot_parameters
        if isinstance(r, (int, float, np.floating)):
            r_pair = (float(r), float(r))
        else:
            r_arr = np.asarray(r, dtype=float).reshape(-1)
            r_pair = (float(r_arr[0]), float(r_arr[1]) if r_arr.size > 1 else float(r_arr[0]))

        pose_dim = int(np.asarray(initial_state).reshape(-1).shape[0])
        if self.estimate_robot_parameters:
            self.state = np.concatenate(
                (np.asarray(initial_state, dtype=float).reshape(-1), np.array([r_pair[0], r_pair[1], b], dtype=float))
            )
            n = int(self.state.shape[0])
            template = np.zeros((n, n))
            template[:pose_dim, :pose_dim] = np.asarray(initial_covariance, dtype=float)[:pose_dim, :pose_dim]
            template[pose_dim : pose_dim + 3, pose_dim : pose_dim + 3] = np.eye(3) * float(parameter_uncertainty)
            self.P = template
            self.base_state_size = pose_dim + 3
        else:
            self.state = np.asarray(initial_state, dtype=float).reshape(-1).copy()
            self.P = np.asarray(initial_covariance, dtype=float).copy()
            self.base_state_size = pose_dim

        self.base_length = float(b)
        self.wheel_radius_right = r_pair[0]
        self.wheel_radius_left = r_pair[1]
        self.wheel_encoder_noise_covariance_matrix = M
        self.Q = Q # process noise covariance for the unicycle mode

        self.last_gyro_update_time = time.time()

        self.landmark_counter = 0
        self.landmarks = {}  # index -> type (use a unique string per world object, e.g. ``buoy:B2``)

    def predict(self, delta_phi_right: float, delta_phi_left: float, dt: float) -> None:
        """Predict from one step of wheel encoder integration.

        ``delta_phi_right`` / ``delta_phi_left`` are the right and left wheel angle increments
        over ``dt`` (radians), same convention as ``form_F``'s wheel angular rates.
        """
        dt = float(dt)
        if dt <= 0.0:
            return
        theta_dot_right = float(delta_phi_right) / dt
        theta_dot_left = float(delta_phi_left) / dt
        F = self.form_F(theta_dot_right, theta_dot_left, dt)
        G = self.form_G(dt)
        W = self.form_wheel_encoder_process_noise_matrix(dt)
        Q = self.form_Q(dt)
        self.P = F @ self.P @ F.T + W + Q
        if self.estimate_robot_parameters:
            r_r = float(self.state[3])
            r_l = float(self.state[4])
            b = float(self.state[5])
        else:
            r_r = self.wheel_radius_right
            r_l = self.wheel_radius_left
            b = self.base_length
        v = (r_r * theta_dot_right + r_l * theta_dot_left) * 0.5
        w = (r_r * theta_dot_right - r_l * theta_dot_left) / b
        self.state[:3] = self.state[:3] + G @ np.array([v, w], dtype=float)
        self.state[2] = wrap_angle(self.state[2])
    
    def update_gps(self, gps_measurement: np.ndarray, R_gps: np.ndarray):
        """ Update the state and covariance using the GPS measurement of the robot's position (x, y).  Assumes GPS is at the robot's base link. """
        H = np.zeros((2, self.state.shape[0]))
        H[:,:2] = np.eye(2) # first 2 rows are the identity matrix for the robot's position, everything else 0
        R = R_gps
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + R)
        n = self.state.shape[0]
        I = np.eye(n)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
        self.state = self.state + K @ (gps_measurement - H @ self.state)

    def update_pose_measurement(self, xy_theta: np.ndarray, R_pose: np.ndarray) -> None:
        """Linear-Gaussian update on the robot pose block ``(x, y, theta)``.

        Measurement model: ``z = [x, y, theta]^T + noise``. Heading innovation is wrapped to
        ``[-pi, pi]``. ``R_pose`` must be ``3×3`` (typically tiny diagonal for oracle / pseudo-GNSS).
        """
        z = np.asarray(xy_theta, dtype=np.float64).reshape(3)
        R = np.asarray(R_pose, dtype=np.float64).reshape(3, 3)
        n = int(self.state.shape[0])
        H = np.zeros((3, n), dtype=np.float64)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        hx = float(self.state[0])
        hy = float(self.state[1])
        ht = wrap_angle(float(self.state[2]))
        zt = wrap_angle(float(z[2]))
        innov = np.array(
            [float(z[0]) - hx, float(z[1]) - hy, wrap_angle(zt - ht)],
            dtype=np.float64,
        )
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ innov
        self.state[2] = wrap_angle(float(self.state[2]))
        I = np.eye(n, dtype=np.float64)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T

    def _get_landmark_estimate(self, landmark_index: int) -> np.ndarray:
        """ Get the estimate of a landmark from the state vector. """
        i0 = self.base_state_size + landmark_index * 2
        return self.state[i0 : i0 + 2]

    def process_landmark_update(self, landmark_type: str, bearing_measurement: float, R_bearing: float, range_measurement: float, R_range: float) -> None:
        """Landmark SLAM: always bearing + range. Use a **unique** ``landmark_type`` per world object
        (e.g. ``buoy:B3``, ``enemy ship:1``) so each track is unique. Do **not** gate association on
        distance from the robot to the landmark estimate: that breaks re-observation when the robot
        has moved far from a landmark that is still far away in the state.
        """
        candidates = [index for index in self.landmarks.keys() if self.landmarks[index] == landmark_type]

        if not candidates:
            self.add_landmark(landmark_type, bearing_measurement, R_bearing, range_measurement, R_range)
            return

        if len(candidates) == 1:
            index_to_use = candidates[0]
        else:
            # Rare: duplicate type labels — triangulate from this pose + measurement, pick nearest track.
            p_meas = np.array(
                [
                    self.state[0] + range_measurement * np.cos(bearing_measurement + self.state[2]),
                    self.state[1] + range_measurement * np.sin(bearing_measurement + self.state[2]),
                ],
                dtype=float,
            )

            def _score(idx: int) -> float:
                return float(np.linalg.norm(self._get_landmark_estimate(idx) - p_meas))

            index_to_use = min(candidates, key=_score)

        self.landmark_bearing_update(bearing_measurement, R_bearing, index_to_use)
        self.landmark_range_update(range_measurement, R_range, index_to_use)

    def add_landmark(self, landmark_type: str, bearing_measurement: float, bearing_R: float, range_measurement: float, range_R: float):
        """ Add a landmark to the state vector. """
        
        # first update the state vector with new landmark, this is the easy part
        old_state_shape = self.state.shape[0]
        lx = self.state[0] + range_measurement * np.cos(bearing_measurement + self.state[2])
        ly = self.state[1] + range_measurement * np.sin(bearing_measurement + self.state[2])
        position = np.array([lx, ly])
        self.state = np.concatenate((self.state, position))

        # now find new covariance matrix, this is the messy part

        # Jacobian of new landmark position wrt robot state estimate
        G_r = np.array([[1, 0, -range_measurement*np.sin(bearing_measurement + self.state[2])],
                        [0, 1, range_measurement*np.cos(bearing_measurement + self.state[2])]])

        # Jacobian of new landmark position wrt the new measurement
        G_z = np.array([[np.cos(bearing_measurement + self.state[2]), -range_measurement*np.sin(bearing_measurement + self.state[2])],
                        [np.sin(bearing_measurement + self.state[2]), range_measurement*np.cos(bearing_measurement + self.state[2])]])

        # Intialize new covariance matrix to populate    
        new_P = np.zeros((self.state.shape[0], self.state.shape[0]))
        new_P[:old_state_shape, :old_state_shape] = self.P # top left block stays the same, it's the variance of the error wrt to the existing state estimate, which is what we already have
        
        R = np.array([[range_R, 0], [0, bearing_R]])
        J = np.zeros((2, old_state_shape))
        J[:, :3] = G_r
        Plnewnew = J @ self.P @ J.T + G_z @ R @ G_z.T
        new_P[old_state_shape:, old_state_shape:] = Plnewnew

        new_P[:old_state_shape, old_state_shape:] = self.P @ J.T
        new_P[old_state_shape:, :old_state_shape] = J @ self.P

        self.P = new_P
        self.landmark_counter += 1
        index = len(self.landmarks)
        self.landmarks[index] = landmark_type

    def landmark_range_update(self, range_measurement: float, R_range: float, landmark_index: int):
        """ Update the state and covariance using the range measurement to a landmark. """
        position = self._get_landmark_estimate(landmark_index)
        dx = position[0] - self.state[0]
        dy = position[1] - self.state[1]
        rho = np.hypot(dx, dy)
        rho = max(rho, 1e-6) # to avoid division by 0
        H = np.zeros((1, self.state.shape[0]))
        H[:, :3] = np.array([[-dx / rho, -dy / rho, 0.0]])
        lm0 = self.base_state_size + landmark_index * 2
        H[:, lm0 : lm0 + 2] = np.array([[dx / rho, dy / rho]])
        R = np.array([[R_range]])
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + R)
        innovation = range_measurement - rho
        self.state = self.state + (K * innovation).ravel()
        n = self.state.shape[0]
        I = np.eye(n)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T

    def landmark_bearing_update(self, bearing_measurement: float, R_bearing: float, landmark_index: int):
        """ Update the state and covariance using the bearing measurement to a landmark. """
        
        position = self._get_landmark_estimate(landmark_index)
        lx = position[0]
        ly = position[1]
        q = (lx - self.state[0])**2 + (ly - self.state[1])**2
        q = max(q, 1e-6) # to avoid division by 0
        dhdx = (ly - self.state[1]) / q
        dhdy = - (lx - self.state[0]) / q
        H = np.zeros((1, self.state.shape[0]))
        H[:, :3] = np.array([[dhdx, dhdy, -1]])
        lm0 = self.base_state_size + landmark_index * 2
        H[:, lm0 : lm0 + 2] = np.array([[-dhdx, -dhdy]])
        R = np.array([[R_bearing]])
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + R)
        h_hat = wrap_angle(np.arctan2(ly - self.state[1], lx - self.state[0]) - self.state[2])
        innovation = wrap_angle(bearing_measurement - h_hat)
        self.state = self.state + (K * float(innovation)).ravel()
        self.state[2] = wrap_angle(self.state[2])
        n = self.state.shape[0]
        I = np.eye(n)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T


    def process_beacon_update(
        self,
        beacon_position: np.ndarray,
        bearing_measurement: Optional[float],
        R_bearing: Optional[float],
        range_measurement: Optional[float] = None,
        R_range: Optional[float] = None,
    ) -> None:
        """Process a beacon update. Omit bearing (and ``R_bearing``) for a range-only update."""
        if bearing_measurement is not None and R_bearing is not None:
            self.beacon_bearing_update(beacon_position, bearing_measurement, R_bearing)
        if range_measurement is not None and R_range is not None:
            self.beacon_range_update(beacon_position, range_measurement, R_range)

    def beacon_bearing_update(self, beacon_position: np.ndarray, bearing_measurement: float, R_bearing: float):
        """EKF update for a bearing measurement to a known beacon position. """
        lx = beacon_position[0]
        ly = beacon_position[1]
        q = (lx - self.state[0])**2 + (ly - self.state[1])**2
        q = max(q, 1e-6) # to avoid division by 0
        dhdx = (ly - self.state[1]) / q
        dhdy = - (lx - self.state[0]) / q
        H = np.zeros((1, self.state.shape[0]))
        H[:, :3] = np.array([[dhdx, dhdy, -1]])
        R = np.array([[R_bearing]])
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + R)
        h_hat = wrap_angle(np.arctan2(ly - self.state[1], lx - self.state[0]) - self.state[2])
        innovation = wrap_angle(bearing_measurement - h_hat)
        self.state = self.state + (K * float(innovation)).ravel()
        self.state[2] = wrap_angle(self.state[2])
        n = self.state.shape[0]
        I = np.eye(n)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T

    def beacon_range_update(self, beacon_position: np.ndarray, range_measurement: float, R_range: float):
        """EKF update for a scalar range measurement to a known beacon position. """
        
        bx = float(np.asarray(beacon_position).reshape(-1)[0])
        by = float(np.asarray(beacon_position).reshape(-1)[1])
        x = float(self.state[0])
        y = float(self.state[1])
        dx = bx - x
        dy = by - y
        rho = float(np.hypot(dx, dy))
        rho = max(rho, 1e-6)
        H = np.zeros((1, self.state.shape[0])) # everything 0s except for first 3 cols
        H[:, :3] = np.array([-dx / rho, -dy / rho, 0.0]) # jacobian of the range measurement function wrt the state
        h_hat = rho
        R = np.array([[R_range]], dtype=float)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        innovation = float(range_measurement) - h_hat
        n = self.state.shape[0]
        I = np.eye(n)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
        self.state = self.state + (K * innovation).ravel()

    def orientation_update(self, orientation: float, R_orientation: float = None):
        """ Update the state and covariance using direct measurement of the robot's orientation.  This is from the magnetometer sensor. """
        
        H = np.zeros((1, self.state.shape[0])) # everything 0s except for first 3 cols
        H[:, :3] = np.array([0, 0, 1]) # jacobian of the orientation measurement function wrt the state
        R = np.array([[R_orientation]])
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + R)
        innovation = wrap_angle(orientation - H @ self.state)
        self.state = self.state + K @ innovation
        self.state[2] = wrap_angle(self.state[2])
        n = self.state.shape[0]
        I = np.eye(n)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T

    def form_F(self, theta_dot_right, theta_dot_left, dt):
        """ For discretezed state transition matrix F """
        F_state_x_state = None
        if not self.estimate_robot_parameters:
            v = (theta_dot_right*self.wheel_radius_right + theta_dot_left*self.wheel_radius_left) / 2
            F_state_x_state = np.array([[1, 0, -v*dt*np.sin(self.state[2])],
                         [0, 1, v*dt*np.cos(self.state[2])],
                         [0, 0, 1]])
        else:
            theta = self.state[2]
            s = np.sin(theta)
            c = np.cos(theta)
            v = (theta_dot_right*self.state[3] + theta_dot_left*self.state[4]) / 2
            b = self.state[5]
            r_right = self.state[3]
            r_left = self.state[4]

            A_state_x_state = np.array([[0, 0, -v*s, (theta_dot_right/2)*c, (theta_dot_left/2)*c, 0],
                                        [0, 0, v*c, (theta_dot_right/2)*s, (theta_dot_left/2)*s, 0],
                                        [0, 0, 0, theta_dot_right/b, -theta_dot_left/b, (theta_dot_left*r_left - theta_dot_right*r_right)/b**2],
                                        [0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0]])
            I = np.eye(A_state_x_state.shape[0])
            A2 = A_state_x_state@A_state_x_state
            F_state_x_state = I + A_state_x_state*dt + A2*dt**2/2 # 2nd order is exact b/c nilpotent matrix

        F_nxn = np.eye(self.state.shape[0]) # initalize all landmarks to identity matrix for the state transition matrix (they don't move)
        F_nxn[:self.base_state_size, :self.base_state_size] = F_state_x_state
        return F_nxn 
    
    def form_G(self, dt):
        """ For discretezed process noise matrix G """
        return np.array([[np.cos(self.state[2])*dt, 0],
                         [np.sin(self.state[2])*dt, 0],
                         [0, dt]])
    
    def form_Q(self, dt):
        """ For discretezed process noise covariance Q """
        Q_3x3 = self.Q*dt
        Q_nxn = np.zeros((self.state.shape[0], self.state.shape[0])) # set process noise for all landmarks to 0
        Q_nxn[:3, :3] = Q_3x3
        if self.estimate_robot_parameters:
            Q_nxn[3:6, 3:6] = np.eye(3) * self.Q[0,0]* 0.0001 * dt # much smaller covariance for the robot parameters
        return Q_nxn

    def form_wheel_encoder_process_noise_matrix(self, dt):
        """ For discretezed wheel encoder process noise matrix W """
        if self.estimate_robot_parameters:
            r_bar = (float(self.state[3]) + float(self.state[4])) * 0.5
            b_use = float(self.state[5])
        else:
            r_bar = (self.wheel_radius_right + self.wheel_radius_left) * 0.5
            b_use = float(self.base_length)
        tmp = (r_bar / 2) * dt
        tmp2 = (r_bar / b_use) * dt
        L = np.array([[tmp*np.cos(self.state[2]), tmp*np.cos(self.state[2])], [tmp*np.sin(self.state[2]), tmp*np.sin(self.state[2])], [tmp2, -tmp2]])
        W_3x3 = L @ self.wheel_encoder_noise_covariance_matrix @ L.T
        W_nxn = np.zeros((self.state.shape[0], self.state.shape[0])) # set process noise for all landmarks to 0
        W_nxn[:3, :3] = W_3x3
        return W_nxn


