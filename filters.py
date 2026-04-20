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
    Landmarks are things we don't have aprior knowledge of and are estimating (part of the state vector). """

    def __init__(self, initial_state: np.ndarray, initial_covariance: np.ndarray, b: float, r: float, M: np.ndarray, Q: np.ndarray):
        self.state = initial_state
        self.covariance = initial_covariance
        self.base_length = b
        self.wheel_radius = r
        self.wheel_encoder_noise_covariance_matrix = M
        self.Q = Q # process noise covariance for the unicycle mode

        self.landmark_merge_threshold = 0.1 # meters
        
        self.P = initial_covariance

        self.last_gyro_update_time = time.time()
        self.previous_theta = self.state[2] # we need this for the IMU update

        self.landmark_counter = 0
        self.landmarks = {}  # index -> type (use a unique string per world object, e.g. ``buoy:B2``)

    def predict(self, v, w, dt):
        """ Predict the state and covariance using the unicycle model using wheel encoder measurements. 
        Only updates the first 3 dimensions of the state and covariance, the rest are left unchanged (if they exist)."""
        F = self.form_F(v, dt)
        G = self.form_G(dt) # this is really our nonlinear state transition function f_x(x, u, dt) evaluated at the current state and control input, but it's really only G, but in the case of the unicycle model, it's only G
        W = self.form_wheel_encoder_process_noise_matrix(dt)
        Q = self.form_Q(dt)
        self.P = F @ self.P @ F.T + W + Q # update covariance using the linearized state transition function F, encoder noise matrix W, and process noise covariance Q
        self.state[:3] = self.state[:3] + G @ np.array([v, w]) # update state using discretized nonlinear state transition function f_x(x, u, dt)
        self.state[2] = wrap_angle(self.state[2]) # 
    
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

    def _get_landmark_estimate(self, landmark_index: int) -> np.ndarray:
        """ Get the estimate of a landmark from the state vector. """
        return self.state[3 + landmark_index*2: 3 + landmark_index*2 + 2]

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
        Plnewnew = G_r @ self.P[:3, :3] @ G_r.T + G_z @ R @ G_z.T # covariance of the new measurement
        new_P[old_state_shape:, old_state_shape:] = Plnewnew # update bottom right block with new measurement covariance
        
        # Update covariance of error wrt to the robot state estimate
        Plnew_robot_state = G_r @ self.P[:3, :3] # this going to be 2 x 3 matrix
        new_P[:3, old_state_shape:] = Plnew_robot_state.T # update top right alley
        new_P[old_state_shape:, :3] = Plnew_robot_state # update bottom left block with new covariance of error wrt to the robot state estimate

        # Update covariance of error wrt to the map estimate
        if self.landmark_counter != 0:
            Px_map = self.P[:3, 3:]
            Plnew_map = G_r @ Px_map
            new_P[3:old_state_shape, old_state_shape:] = Plnew_map.T
            new_P[old_state_shape:, 3:old_state_shape] = Plnew_map

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
        H[:, 3+landmark_index*2: 3+landmark_index*2 + 2] = np.array([[dx / rho, dy / rho]]) # this is the Jacobian of the range measurement function wrt the landmark position
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
        H[:, :3] = np.array([[dhdx, dhdy, -1]]) # this is the Jacobian of the bearing measurement function wrt the robot state estimate
        H[:, 3+landmark_index*2: 3+landmark_index*2 + 2] = np.array([[-dhdx, -dhdy]]) # this is the Jacobian of the bearing measurement function wrt the landmark position
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

    def form_fx(self, v, w, dt):
        """ For nonlinear state transition function f_x """
        return np.array([[dt*v*np.cos(self.state[2])],
                         [dt*v*np.sin(self.state[2])],
                         [dt*w]])
        

    def form_F(self, v, dt):
        """ For discretezed state transition matrix F """
        F_3x3 = np.array([[1, 0, -v*dt*np.sin(self.state[2])],
                         [0, 1, v*dt*np.cos(self.state[2])],
                         [0, 0, 1]])
        F_nxn = np.eye(self.state.shape[0]) # initalize all landmarks to identity matrix for the state transition matrix (they don't move)
        F_nxn[:3, :3] = F_3x3
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
        return Q_nxn

    def form_wheel_encoder_process_noise_matrix(self, dt):
        """ For discretezed wheel encoder process noise matrix W """
        tmp = (self.wheel_radius/2) * dt
        tmp2 = (self.wheel_radius/self.base_length) * dt
        L = np.array([[tmp*np.cos(self.state[2]), tmp*np.cos(self.state[2])], [tmp*np.sin(self.state[2]), tmp*np.sin(self.state[2])], [tmp2, -tmp2]])
        W_3x3 = L @ self.wheel_encoder_noise_covariance_matrix @ L.T
        W_nxn = np.zeros((self.state.shape[0], self.state.shape[0])) # set process noise for all landmarks to 0
        W_nxn[:3, :3] = W_3x3
        return W_nxn


