import numpy as np

def quat_mul(q, r):
    # q and r are arrays [w, x, y, z]
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = r
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1
    ])

def quat_conjugate(q):
    w,x,y,z = q
    return np.array([w, -x, -y, -z])

def quat_normalize(q):
    return q / np.linalg.norm(q)

def quat_error_angle_axis(q_current, q_target, eps=1e-12):
    # both quaternions as [w,x,y,z], assumed normalized (if not, normalize)
    q1 = quat_normalize(q_current)
    q2 = quat_normalize(q_target)

    # relative rotation that maps q1 -> q2
    q_err = quat_mul(q2, quat_conjugate(q1))
    q_err = quat_normalize(q_err)

    # ensure shortest rotation (w >= 0)
    if q_err[0] < 0:
        q_err = -q_err

    w = q_err[0]
    v = q_err[1:]
    v_norm = np.linalg.norm(v)

    # angle (robust)
    angle = 2.0 * np.arctan2(v_norm, w)

    if v_norm > eps:
        axis = v / v_norm
    else:
        # angle ~ 0 -> axis arbitrary
        axis = np.array([1.0, 0.0, 0.0])

    return angle, axis, q_err

# Example:
q_curr = np.array([0.9659258263, 0.0, 0.2588190451, 0.0])   # 30 degrees about Y
q_targ = np.array([1.0, 0.0, 0.0, 0.0])                    # identity
angle, axis, q_err = quat_error_angle_axis(q_curr, q_targ)
print("angle (rad):", angle)
print("angle (deg):", np.degrees(angle))
print("axis:", axis)
print("q_err:", q_err)
