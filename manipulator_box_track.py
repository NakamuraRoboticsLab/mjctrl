import mujoco
import mujoco.viewer
import numpy as np
import time
import csv
from typing import List, Tuple

"""End-effector trajectory tracking using DLS IK + joint-space PD torque.

This mirrors diffik_nullspace.py's target pose generator, but here we:
1) Compute a task-space twist to track the desired site pose (world frame).
2) Solve for joint velocity dq via damped least squares (J in world frame).
3) Integrate to get a one-step-ahead joint target q_des.
4) Apply torque-mode joint-space PD: tau = Kp*(q_des - q) + Kd*(qd_des - qd).
5) (Optional) Add gravity/Coriolis compensation using data.qfrc_bias.
We also log desired (as mocap_*) and actual site pose to CSV.
"""

# Simulation and IK integration settings
DT_SIM: float = 0.002         # Must match XML timestep
T_END: float = 20.0           # Total simulation time [s]
integration_dt: float = 0.1   # IK integration horizon [s]
damping: float = 1e-4         # DLS damping (lambda)
Kpos: float = 0.95            # Task-space position gain in [0, 1]
Kori: float = 0.95            # Task-space orientation gain in [0, 1]
PRINT_PERIOD: float = 0.05    # Console print period [s]

# Nullspace bias towards initial joint configuration (small for stability)
Kn = np.asarray([5.0, 5.0, 5.0, 5.0, 3.0, 3.0])

# PD torque gains per joint
KP = np.array([2000, 1000, 500, 500, 200, 1], dtype=float)
KD = np.array([50, 20, 20, 20, 20, 0.1], dtype=float)

# Gravity/Coriolis compensation
USE_BIAS_COMP: bool = False
MAX_ANGVEL: float = 3.0  # rad/s clamp for dq magnitude

# Joint order to control (matches manipulator_box/scene.xml)
JOINT_ORDER = [
    "j1_yaw", "j2_pitch", "j3_pitch",
    "j4_pitch", "j5_yaw", "j6_roll"
]


def pose(t: float) -> Tuple[np.ndarray, np.ndarray]:
    """Desired end-effector pose: sinusoidal X, constant Y/Z and
    quaternion (wxyz)."""
    # Sinusoid parameters
    x0 = 0.100  # center X [m]
    A = 0.1   # amplitude [m]
    f = 0.5    # frequency [Hz]
    omega = 2.0 * np.pi * f

    x = 1.000
    y = x0 + A * np.sin(omega * t)
    z = 0.350
    pos = np.array([x, y, z], dtype=float)

    # Constant orientation (identity quaternion, wxyz)
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return pos, quat  # wxyz


def log_trajectories(filename: str,
                     mocap_traj: List[Tuple[float, ...]],
                     site_traj: List[Tuple[float, ...]]) -> None:
    """Log desired (mocap_*) and actual site pose to CSV (same schema)."""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = [
            "time",
            "mocap_x", "mocap_y", "mocap_z",
            "mocap_quat_w", "mocap_quat_x", "mocap_quat_y", "mocap_quat_z",
            "mocap_rot_00", "mocap_rot_01", "mocap_rot_02",
            "mocap_rot_10", "mocap_rot_11", "mocap_rot_12",
            "mocap_rot_20", "mocap_rot_21", "mocap_rot_22",
            "time",
            "site_x", "site_y", "site_z",
            "site_quat_w", "site_quat_x", "site_quat_y", "site_quat_z",
            "site_rot_00", "site_rot_01", "site_rot_02",
            "site_rot_10", "site_rot_11", "site_rot_12",
            "site_rot_20", "site_rot_21", "site_rot_22"
        ]
        writer.writerow(header)
        for mocap, site in zip(mocap_traj, site_traj):
            writer.writerow(list(mocap) + list(site))


def run() -> None:
    assert mujoco.__version__ >= "3.1.0", "MuJoCo 3.1.0+ required"

    # Load model and data
    model = mujoco.MjModel.from_xml_path("manipulator_box/scene.xml")
    data = mujoco.MjData(model)

    # End-effector site
    site_name = "ee"
    site_id = model.site(site_name).id

    # Controlled joints: indices and actuator mapping
    joint_ids = [model.joint(n).id for n in JOINT_ORDER]
    dof_indices = np.array(
        [model.jnt_dofadr[jid] for jid in joint_ids], dtype=int
    )
    actuator_ids = [model.actuator(i).id for i in range(model.nu)]

    # Buffers
    jac = np.zeros((6, model.nv))
    diag = damping * np.eye(6)
    eye_nv = np.eye(model.nv)
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)

    # Trajectories for logging
    mocap_traj: List[Tuple[float, ...]] = []
    site_traj: List[Tuple[float, ...]] = []

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        # Visualize site frame for debugging
        model.vis.scale.framelength = 0.4
        model.vis.scale.framewidth = 0.02
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        # Save initial joint configuration for nullspace bias
        q0 = data.qpos.copy()
        last_print_time = -1e9

        while viewer.is_running() and data.time < T_END:
            t0 = time.time()

            # Desired pose at current sim time
            des_pos, des_quat = pose(data.time)

            # Task-space twist in world frame
            dx = des_pos - data.site(site_id).xpos
            twist[:3] = Kpos * dx / integration_dt

            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, des_quat, site_quat_conj)
            mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
            twist[3:] *= Kori / integration_dt

            # Jacobian in world frame
            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

            # DLS IK: dq_task
            dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, twist)

            # Nullspace bias towards q0 (projector N = I - J^+ J)
            J_pinv = np.linalg.pinv(jac)
            dq += (eye_nv - J_pinv @ jac) @ (Kn * (q0 - data.qpos))

            # Clamp maximum joint velocity magnitude for stability
            dq_abs_max = np.abs(dq).max()
            if dq_abs_max > MAX_ANGVEL:
                dq *= MAX_ANGVEL / dq_abs_max

            # Integrate to get one-step-ahead joint target
            q_tmp = data.qpos.copy()
            mujoco.mj_integratePos(model, q_tmp, dq, integration_dt)
            np.clip(q_tmp, *model.jnt_range.T, out=q_tmp)
            q_des = q_tmp[dof_indices]
            qd_des = dq[dof_indices]

            # Current joint state
            q = data.qpos[dof_indices]
            qd = data.qvel[dof_indices]

            # Joint-space PD torque
            tau = KP * (q_des - q) + KD * (qd_des - qd)

            # Add gravity/Coriolis/centrifugal compensation if enabled
            if USE_BIAS_COMP:
                tau += data.qfrc_bias[dof_indices]

            # Apply torques to actuators corresponding to joints
            # (clip to actuator ctrlrange)
            for i, ai in enumerate(actuator_ids):
                ctrl_min, ctrl_max = model.actuator_ctrlrange[ai]
                data.ctrl[ai] = float(np.clip(tau[i], ctrl_min, ctrl_max))

            # Log desired (as mocap_*) and site pose
            mocap_rot = np.zeros(9, dtype=float)
            mujoco.mju_quat2Mat(mocap_rot, des_quat)
            site_rot = data.site(site_id).xmat.reshape(9)
            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)

            mocap_traj.append((
                data.time,
                float(des_pos[0]), float(des_pos[1]), float(des_pos[2]),
                float(des_quat[0]), float(des_quat[1]),
                float(des_quat[2]), float(des_quat[3]),
                float(mocap_rot[0]), float(mocap_rot[1]), float(mocap_rot[2]),
                float(mocap_rot[3]), float(mocap_rot[4]), float(mocap_rot[5]),
                float(mocap_rot[6]), float(mocap_rot[7]), float(mocap_rot[8])
            ))
            site_traj.append((
                data.time,
                float(data.site(site_id).xpos[0]),
                float(data.site(site_id).xpos[1]),
                float(data.site(site_id).xpos[2]),
                float(site_quat[0]), float(site_quat[1]),
                float(site_quat[2]), float(site_quat[3]),
                float(site_rot[0]), float(site_rot[1]), float(site_rot[2]),
                float(site_rot[3]), float(site_rot[4]), float(site_rot[5]),
                float(site_rot[6]), float(site_rot[7]), float(site_rot[8])
            ))

            # Periodic print of desired and actual EE position
            if data.time - last_print_time >= PRINT_PERIOD:
                px, py, pz = data.site(site_id).xpos
                print(
                    f"t={data.time:.3f}  des=[{des_pos[0]:.3f} "
                    f"{des_pos[1]:.3f} {des_pos[2]:.3f}] act=[{px:.3f} "
                    f"{py:.3f} {pz:.3f}] quat=[{site_quat[0]:.3f} "
                    f"{site_quat[1]:.3f} {site_quat[2]:.3f} "
                    f"{site_quat[3]:.3f}]"
                )
                last_print_time = data.time

            # Step simulation
            mujoco.mj_step(model, data)
            viewer.sync()

            # Real-time pacing
            rem = DT_SIM - (time.time() - t0)
            if rem > 0:
                time.sleep(rem)

    # Save logs
    out = "manipulator_box_track_trajectories.csv"
    log_trajectories(out, mocap_traj, site_traj)
    print(f"Saved trajectory log to {out}")


if __name__ == "__main__":
    run()
