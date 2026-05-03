# Maritime EKF-SLAM Implementation for the Robotarium

This repository holds the holds the code for EKF SLAM and helper code to run experiments around it in the Robotarium.  See [this paper](https://antony-blog.notion.site/maritime-inpsired-ekf-slam-in-the-robotarium?source=copy_link) for more details on the math, simulation, and implementation.

# Overview

<img width="356" height="172" alt="Screenshot 2026-05-03 at 2 18 47 AM" src="https://github.com/user-attachments/assets/46599267-97b0-41b2-b826-dc460dff33d3" />

This repository implements a maritime-inspired Extended Kalman Filter Simultaneous Localization and Mapping (EKF-SLAM) architecture on differential-drive Robotarium unicycles. Relying on range and bearing measurements to simulated lighthouses, the discrete-time EKF simultaneously estimates the robot state, a localized map, and internal kinematic parameters (wheel radii & wheelbase length).

Can be run in conjunction with the Robotarium simulator and submitted to the [Robotarium](https://www.robotarium.gatech.edu/) to run on actual hardware.

# Interesting Findings

- The Mathematical Sponge: On physical hardware, the EKF acts as a "mathematical sponge." Rather than converging to true geometric dimensions, the filter violently distorts its wheel radii and wheelbase estimates to absorb unmodeled physical residuals like wheel slip and floor imperfections.

- Active Estimation Can Degrade SLAM: Because the filter rigidly overfits to environmental noise, actively estimating kinematic parameters online can actually degrade overall localization and mapping performance compared to simply relying on static, nominal factory values.

- The Sim-to-Real Discrepancy: While the EKF successfully isolates geometric parameters in sterile, idealized simulation environments (limited primarily by the Errors-in-Variables bias), physical deployment proves that unless a kinematic model comprehensively accounts for dynamic real-world physics, asking an EKF to estimate geometric truth will only result in overfitted slop parameters.
