import mujoco
import mujoco.viewer
import numpy as np
import time
import csv
from typing import List, Tuple

"""Joint-space PD control demo for manipulator/scene.xml.

Each joint follows an independent sinusoidal reference around the initial
configuration. Torque command: tau = Kp*(q_des - q) + Kd*(qd_des - qd).
Results are logged to CSV for analysis.
"""

DT_SIM = 0.002        # Matches XML timestep
T_END = 30.0          # Total simulation time [s]

# Constant desired posture configuration
# Provide target joint angles in degrees (readable). Set RELATIVE_TO_INITIAL
# = True to interpret these as offsets added to the initial configuration.
JOINT_TARGET_DEG = np.array([50, -70, -40, 50, 60, 15], dtype=float)
RELATIVE_TO_INITIAL = False  # False: absolute targets; True: offsets from q0

# Joint-space PD gains (element-wise)
KP = np.array([2000, 500, 400, 400, 50, 10], dtype=float)
KD = np.array([10, 5, 5, 5, 5, 0.1], dtype=float)

# Torque saturation fraction of actuator ctrlrange
SAT_FRACTION = 1

JOINT_ORDER = [
    "j1_yaw", "j2_pitch", "j3_pitch",
    "j4_pitch", "j5_yaw", "j6_roll"
]


def build_constant_target(q0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    target_rad = np.deg2rad(JOINT_TARGET_DEG)
    if RELATIVE_TO_INITIAL:
        q_des = q0 + target_rad
    else:
        q_des = target_rad
    qd_des = np.zeros_like(q_des)
    return q_des, qd_des


def run():
    assert mujoco.__version__ >= "3.1.0", "MuJoCo 3.1.0+ required"
    model = mujoco.MjModel.from_xml_path("manipulator_box/scene.xml")
    data = mujoco.MjData(model)

    joint_ids = [model.joint(n).id for n in JOINT_ORDER]
    dof_indices = []
    for jid in joint_ids:
        vadr = model.jnt_dofadr[jid]
        dof_indices.append(vadr)
    dof_indices = np.array(dof_indices, dtype=int)

    actuator_ids = [model.actuator(i).id for i in range(model.nu)]

    rows: List[List[float]] = []

    with mujoco.viewer.launch_passive(model=model, data=data,
                                      show_left_ui=False,
                                      show_right_ui=False) as viewer:
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        q0 = data.qpos[dof_indices].copy()
        q_des_const, qd_des_const = build_constant_target(q0)

        tau = np.zeros_like(q0)

        while viewer.is_running() and data.time < T_END:
            step_start = time.time()

            # Constant desired posture
            q_des = q_des_const
            qd_des = qd_des_const
            q = data.qpos[dof_indices]
            qd = data.qvel[dof_indices]

            # PD torque
            tau = KP * (q_des - q) + KD * (qd_des - qd)

            # print("3 angle, velocity, torque:", q[2], qd[2], tau[2])
            # print("4 angle, velocity, torque:", q[3], qd[3], tau[3])
            # print("5 angle, velocity, torque:", q[4], qd[4], tau[4])
            # print("6 angle, velocity, torque:", q[5], qd[5], tau[5])

            # Direct torque assignment (no saturation)
            for ai, joint_dof in zip(actuator_ids, dof_indices):
                data.ctrl[ai] = float(tau[dof_indices == joint_dof][0])

            # Log (time, q_des[6], q[6], qd_des[6], qd[6])
            rows.append([
                data.time,
                *q_des.tolist(),
                *q.tolist(),
                *qd_des.tolist(),
                *qd.tolist(),
            ])

            mujoco.mj_step(model, data)
            viewer.sync()

            rem = DT_SIM - (time.time() - step_start)
            if rem > 0:
                time.sleep(rem)

    # Write CSV
    header = [
        "time",
        *(f"q_des_{i+1}" for i in range(6)),
        *(f"q_{i+1}" for i in range(6)),
        *(f"qd_des_{i+1}" for i in range(6)),
        *(f"qd_{i+1}" for i in range(6)),
    ]
    out = "manipulator_joint_pd_trajectories.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"Saved joint-space trajectory log to {out}")


if __name__ == "__main__":
    run()
