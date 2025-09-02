import mujoco
import mujoco.viewer
import numpy as np
import time
import csv
from typing import List, Tuple
from scipy.spatial.transform import Rotation as R

# Simulation timestep in seconds.
dt: float = 0.002


def log_trajectories(filename: str,
                     mocap_traj: List[Tuple[float, ...]],
                     site_traj: List[Tuple[float, ...]]) -> None:
    """
    Log the trajectories of mocap and site positions, quaternions,
    and rotation matrices to a CSV file.
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = [
            "time",
            "mocap_x", "mocap_y", "mocap_z",
            "mocap_quat_w", "mocap_quat_x", "mocap_quat_y", "mocap_quat_z",
            "mocap_rot_00", "mocap_rot_01", "mocap_rot_02",
            "mocap_rot_10", "mocap_rot_11", "mocap_rot_12",
            "mocap_rot_20", "mocap_rot_21", "mocap_rot_22",
            "site_x", "site_y", "site_z",
            "site_quat_w", "site_quat_x", "site_quat_y", "site_quat_z",
            "site_rot_00", "site_rot_01", "site_rot_02",
            "site_rot_10", "site_rot_11", "site_rot_12",
            "site_rot_20", "site_rot_21", "site_rot_22"
        ]
        writer.writerow(header)
        for mocap, site in zip(mocap_traj, site_traj):
            writer.writerow(list(mocap) + list(site))


def parabolic_pose(time, amplitude=0.3, frequency=0.5, phase=0.0):
    """
    Generate parabolic motion trajectory.
    Creates a parabolic arc in 3D space.
    """
    # Parabolic trajectory: y = -a*x^2 + h (maximum at x=0)
    x = 0.2 * np.sin(frequency * time + phase)
    y = -amplitude * x**2 + amplitude
    z = 0.1 + 0.05 * np.sin(2 * frequency * time)

    # Orientation: rotate around z-axis based on position
    angle = 2 * np.pi * x  # Full rotation as we move
    quat = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])
    quat /= np.linalg.norm(quat)

    return (x, y, z), quat


def main() -> None:
    assert mujoco.__version__ >= "3.1.0", \
        "Please upgrade to mujoco 3.1.0 or later."

    # Load the box model and data.
    model = mujoco.MjModel.from_xml_path("box/scene.xml")
    data = mujoco.MjData(model)

    # Simulation timestep
    model.opt.timestep = dt

    # For box tracking, we'll use the box body as our reference point
    box_body_id = model.body("box").id

    # Trajectory lists for logging
    mocap_traj: List[Tuple[float, ...]] = []
    site_traj: List[Tuple[float, ...]] = []

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        # Reset the simulation.
        mujoco.mj_resetDataKeyframe(model, data, 0)

        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Enable body frame visualization.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY

        while viewer.is_running() and data.time < 6.0:
            step_start = time.time()

            # Generate parabolic trajectory (for reference/logging only)
            mocap_pos, mocap_quat = parabolic_pose(data.time,
                                                   amplitude=0.4,
                                                   frequency=0.3,
                                                   phase=np.pi/4)

            # Step the simulation (box falls naturally under gravity)
            mujoco.mj_step(model, data)

            # Log mocap reference trajectory and actual box positions
            mocap_pos_log = np.array(mocap_pos)
            site_pos = data.xpos[box_body_id]

            # Calculate site quaternion and rotation matrix
            site_quat = np.zeros(4)
            mujoco.mju_mat2Quat(site_quat,
                                data.xmat[box_body_id].reshape(9))
            site_rotmat = data.xmat[box_body_id].reshape(9)

            # Convert mocap quaternion to rotation matrix
            mocap_rot = R.from_quat(mocap_quat).as_matrix()
            mocap_rotmat = mocap_rot.flatten()

            mocap_traj.append((data.time,
                              float(mocap_pos_log[0]),
                              float(mocap_pos_log[1]),
                              float(mocap_pos_log[2]),
                              float(mocap_quat[0]),
                              float(mocap_quat[1]),
                              float(mocap_quat[2]),
                              float(mocap_quat[3]),
                              float(mocap_rotmat[0]),
                              float(mocap_rotmat[1]),
                              float(mocap_rotmat[2]),
                              float(mocap_rotmat[3]),
                              float(mocap_rotmat[4]),
                              float(mocap_rotmat[5]),
                              float(mocap_rotmat[6]),
                              float(mocap_rotmat[7]),
                              float(mocap_rotmat[8])))
            site_traj.append((data.time,
                             float(site_pos[0]),
                             float(site_pos[1]),
                             float(site_pos[2]),
                             float(site_quat[0]),
                             float(site_quat[1]),
                             float(site_quat[2]),
                             float(site_quat[3]),
                             float(site_rotmat[0]),
                             float(site_rotmat[1]),
                             float(site_rotmat[2]),
                             float(site_rotmat[3]),
                             float(site_rotmat[4]),
                             float(site_rotmat[5]),
                             float(site_rotmat[6]),
                             float(site_rotmat[7]),
                             float(site_rotmat[8])))

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # After simulation, save trajectories
    log_trajectories("box_parabolic_trajectories.csv", mocap_traj, site_traj)


if __name__ == "__main__":
    main()
