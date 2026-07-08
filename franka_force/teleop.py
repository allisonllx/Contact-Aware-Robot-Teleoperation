import threading

import mujoco
import numpy as np

from .config import DEFAULT_TELEOP_NUDGE_STEP, DEFAULT_TELEOP_SPEED

try:
    from pynput import keyboard as pynput_keyboard
except ImportError:
    pynput_keyboard = None


def initialize_state(env, teleop_speed=DEFAULT_TELEOP_SPEED, teleop_nudge_step=DEFAULT_TELEOP_NUDGE_STEP):
    env.target_pos = np.zeros(3)
    env.target_rot = np.eye(3)
    env.target_roll = 0.0
    env.teleop_speed = teleop_speed
    env.teleop_nudge_step = teleop_nudge_step
    env.orientation_speed = 0.8
    env.roll_speed = 0.8
    env.gripper_closed = False
    env._teleop_lock = threading.Lock()
    env._move_cmd = np.zeros(3)
    env._orientation_cmd = np.zeros(3)
    env._roll_cmd = 0.0
    env._keyboard_listener = None
    env.ik_target_body_id = -1
    env.ik_target_mocap_id = -1


def add_target_marker(spec):
    ik_target = spec.worldbody.add_body(name="ik_target", mocap=True)
    ik_target.add_geom(
        name="ik_target_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.015],
        rgba=[0.1, 0.9, 0.2, 0.55],
        contype=0,
        conaffinity=0,
    )


def add_floor_compass(spec, origin):
    """World-frame N/E/S/W arrows on the floor (teleop uses world +X/+Y, not camera axes)."""
    base = spec.worldbody.add_body(name="floor_compass", pos=list(origin))
    z = 0.002
    arm = 0.11
    thick = 0.007
    decal = dict(contype=0, conaffinity=0)

    base.add_geom(
        name="compass_e",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[arm / 2, thick, 0.001],
        pos=[arm / 2, 0, z],
        rgba=[0.95, 0.25, 0.2, 0.95],
        **decal,
    )
    base.add_geom(
        name="compass_e_tip",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[thick * 1.4, thick * 2.2, 0.001],
        pos=[arm - thick, 0, z],
        rgba=[0.95, 0.25, 0.2, 0.95],
        **decal,
    )
    base.add_geom(
        name="compass_w",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[arm / 2, thick, 0.001],
        pos=[-arm / 2, 0, z],
        rgba=[0.55, 0.15, 0.12, 0.85],
        **decal,
    )
    base.add_geom(
        name="compass_n",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[thick, arm / 2, 0.001],
        pos=[0, arm / 2, z],
        rgba=[0.2, 0.35, 0.95, 0.95],
        **decal,
    )
    base.add_geom(
        name="compass_n_tip",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[thick * 2.2, thick * 1.4, 0.001],
        pos=[0, arm - thick, z],
        rgba=[0.2, 0.35, 0.95, 0.95],
        **decal,
    )
    base.add_geom(
        name="compass_s",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[thick, arm / 2, 0.001],
        pos=[0, -arm / 2, z],
        rgba=[0.15, 0.55, 0.85, 0.85],
        **decal,
    )
    base.add_geom(
        name="compass_hub",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.014],
        pos=[0, 0, z + 0.001],
        rgba=[0.95, 0.95, 0.95, 0.9],
        **decal,
    )
    label_z = 0.004
    label_size = 0.018
    for name, pos, rgba in (
        ("compass_label_e", [arm + 0.03, 0, label_z], [0.95, 0.25, 0.2, 1]),
        ("compass_label_w", [-arm - 0.03, 0, label_z], [0.55, 0.15, 0.12, 1]),
        ("compass_label_n", [0, arm + 0.03, label_z], [0.2, 0.35, 0.95, 1]),
        ("compass_label_s", [0, -arm - 0.03, label_z], [0.15, 0.55, 0.85, 1]),
    ):
        base.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[label_size, label_size, 0.001],
            pos=pos,
            rgba=rgba,
            **decal,
        )


def resolve_ids(env):
    env.ik_target_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "ik_target")
    if env.ik_target_body_id >= 0:
        env.ik_target_mocap_id = env.model.body_mocapid[env.ik_target_body_id]


def boost_arm_actuators(env, scale=10.0):
    for i in range(7):
        env.model.actuator_gainprm[i, 0] *= scale
        env.model.actuator_biasprm[i, 1] *= scale
        env.model.actuator_biasprm[i, 2] *= scale


def init_home_pose(env, home_q, gripper_closed=False):
    env.data.qpos[:7] = home_q
    mujoco.mj_forward(env.model, env.data)
    for i in range(7):
        env.data.ctrl[i] = home_q[i]
    with env._teleop_lock:
        env.target_pos = env.data.xpos[env.hand_body_id].copy()
        env.target_rot = env.data.xmat[env.hand_body_id].reshape(3, 3).copy()
        env.gripper_closed = gripper_closed
    env.data.ctrl[7] = 255.0 if gripper_closed else 0.0
    sync_target_marker(env)


def print_controls(
    env,
    title,
    goal,
    roll_controls=False,
    free_orientation_controls=False,
    orientation_note=None,
):
    print(f"\n=== INTERACTIVE {title} TASK ===")
    print(goal)
    print()
    print("Click the MuJoCo window, then use:")
    print("  Arrow keys   : Up/Down = North/South, Left/Right = West/East")
    print("  9 / 8        : Raise / lower target (Z)")
    print("  Page Up/Down : Also raise / lower (Z), if your keyboard has them")
    print(f"  Step size    : {env.teleop_nudge_step * 1000:.0f} mm per discrete key press")
    if roll_controls:
        print("  6 / 7        : Roll tool left / right (spin about vertical)")
    if free_orientation_controls:
        print("  [ / ]        : Pitch tool")
        print("  - / =        : Yaw tool")
        print("  6 / 7        : Roll tool")
    print("  , / .        : Open / close gripper")
    print()
    print("DO NOT press I, J, K, or U - those are MuJoCo debug toggles")
    print("(red collision boxes, joint axes, etc.), not robot controls.")
    print("Hold arrow keys for smooth motion if pynput is installed.")
    if orientation_note:
        print(orientation_note)


def viewer_key_callback(env, keycode, roll_enabled=False, orientation_enabled=False):
    """Nudge the IK target from the MuJoCo window (one step per key press)."""
    if not env.interactive:
        return

    # GLFW key codes - avoid I/J/K/U; those toggle MuJoCo debug overlays.
    if keycode == 265:      # Up arrow -> North (+Y)
        nudge_target(env, dy=1.0)
    elif keycode == 264:    # Down arrow -> South (-Y)
        nudge_target(env, dy=-1.0)
    elif keycode == 262:    # Right arrow -> East (+X)
        nudge_target(env, dx=1.0)
    elif keycode == 263:    # Left arrow -> West (-X)
        nudge_target(env, dx=-1.0)
    elif keycode == 266:    # Page Up -> +Z
        nudge_target(env, dz=1.0)
    elif keycode == 267:    # Page Down -> -Z
        nudge_target(env, dz=-1.0)
    elif keycode in (57,):  # 9 raise (Z+)
        nudge_target(env, dz=1.0)
    elif keycode in (56,):  # 8 lower (Z-)
        nudge_target(env, dz=-1.0)
    elif keycode in (44,):  # , open gripper
        set_gripper(env, False)
    elif keycode in (46,):  # . close gripper
        set_gripper(env, True)
    elif roll_enabled and keycode in (54,):  # 6 roll CCW
        nudge_roll(env, -1.0)
    elif roll_enabled and keycode in (55,):  # 7 roll CW
        nudge_roll(env, 1.0)
    elif orientation_enabled and keycode in (91,):  # [ pitch negative
        nudge_orientation(env, 1, -1.0)
    elif orientation_enabled and keycode in (93,):  # ] pitch positive
        nudge_orientation(env, 1, 1.0)
    elif orientation_enabled and keycode in (45,):  # - yaw negative
        nudge_orientation(env, 2, -1.0)
    elif orientation_enabled and keycode in (61,):  # = yaw positive
        nudge_orientation(env, 2, 1.0)
    elif orientation_enabled and keycode in (54,):  # 6 roll negative
        nudge_orientation(env, 0, -1.0)
    elif orientation_enabled and keycode in (55,):  # 7 roll positive
        nudge_orientation(env, 0, 1.0)


def before_step(env, dt, roll_enabled=False, orientation_enabled=False):
    apply_motion(env, dt, roll_enabled=roll_enabled, orientation_enabled=orientation_enabled)
    sync_target_marker(env)


def start(env, roll_enabled=False, orientation_enabled=False):
    if pynput_keyboard is None:
        print("Note: install pynput for smoother hold-to-move teleop (pip install pynput).")
        return

    def on_press(key):
        try:
            if key == pynput_keyboard.Key.up:
                adjust_move_cmd(env, 1, 1.0)
            elif key == pynput_keyboard.Key.down:
                adjust_move_cmd(env, 1, -1.0)
            elif key == pynput_keyboard.Key.right:
                adjust_move_cmd(env, 0, 1.0)
            elif key == pynput_keyboard.Key.left:
                adjust_move_cmd(env, 0, -1.0)
            elif key == pynput_keyboard.Key.page_up:
                adjust_move_cmd(env, 2, 1.0)
            elif key == pynput_keyboard.Key.page_down:
                adjust_move_cmd(env, 2, -1.0)
            elif hasattr(key, "char") and key.char == "9":
                adjust_move_cmd(env, 2, 1.0)
            elif hasattr(key, "char") and key.char == "8":
                adjust_move_cmd(env, 2, -1.0)
            elif hasattr(key, "char") and key.char == ",":
                set_gripper(env, False)
            elif hasattr(key, "char") and key.char == ".":
                set_gripper(env, True)
            elif roll_enabled and hasattr(key, "char") and key.char == "6":
                adjust_roll_cmd(env, -1.0)
            elif roll_enabled and hasattr(key, "char") and key.char == "7":
                adjust_roll_cmd(env, 1.0)
            elif orientation_enabled and hasattr(key, "char") and key.char == "[":
                adjust_orientation_cmd(env, 1, -1.0)
            elif orientation_enabled and hasattr(key, "char") and key.char == "]":
                adjust_orientation_cmd(env, 1, 1.0)
            elif orientation_enabled and hasattr(key, "char") and key.char == "-":
                adjust_orientation_cmd(env, 2, -1.0)
            elif orientation_enabled and hasattr(key, "char") and key.char == "=":
                adjust_orientation_cmd(env, 2, 1.0)
            elif orientation_enabled and hasattr(key, "char") and key.char == "6":
                adjust_orientation_cmd(env, 0, -1.0)
            elif orientation_enabled and hasattr(key, "char") and key.char == "7":
                adjust_orientation_cmd(env, 0, 1.0)
        except Exception:
            pass

    def on_release(key):
        try:
            if key == pynput_keyboard.Key.up:
                adjust_move_cmd(env, 1, -1.0)
            elif key == pynput_keyboard.Key.down:
                adjust_move_cmd(env, 1, 1.0)
            elif key == pynput_keyboard.Key.right:
                adjust_move_cmd(env, 0, -1.0)
            elif key == pynput_keyboard.Key.left:
                adjust_move_cmd(env, 0, 1.0)
            elif key == pynput_keyboard.Key.page_up:
                adjust_move_cmd(env, 2, -1.0)
            elif key == pynput_keyboard.Key.page_down:
                adjust_move_cmd(env, 2, 1.0)
            elif hasattr(key, "char") and key.char == "9":
                adjust_move_cmd(env, 2, -1.0)
            elif hasattr(key, "char") and key.char == "8":
                adjust_move_cmd(env, 2, 1.0)
            elif roll_enabled and hasattr(key, "char") and key.char == "6":
                adjust_roll_cmd(env, 1.0)
            elif roll_enabled and hasattr(key, "char") and key.char == "7":
                adjust_roll_cmd(env, -1.0)
            elif orientation_enabled and hasattr(key, "char") and key.char == "[":
                adjust_orientation_cmd(env, 1, 1.0)
            elif orientation_enabled and hasattr(key, "char") and key.char == "]":
                adjust_orientation_cmd(env, 1, -1.0)
            elif orientation_enabled and hasattr(key, "char") and key.char == "-":
                adjust_orientation_cmd(env, 2, 1.0)
            elif orientation_enabled and hasattr(key, "char") and key.char == "=":
                adjust_orientation_cmd(env, 2, -1.0)
            elif orientation_enabled and hasattr(key, "char") and key.char == "6":
                adjust_orientation_cmd(env, 0, 1.0)
            elif orientation_enabled and hasattr(key, "char") and key.char == "7":
                adjust_orientation_cmd(env, 0, -1.0)
        except Exception:
            pass

    env._keyboard_listener = pynput_keyboard.Listener(
        on_press=on_press,
        on_release=on_release,
    )
    env._keyboard_listener.start()


def stop(env):
    if env._keyboard_listener is not None:
        env._keyboard_listener.stop()
        env._keyboard_listener = None
    with env._teleop_lock:
        env._move_cmd[:] = 0.0
        env._orientation_cmd[:] = 0.0
        env._roll_cmd = 0.0


def sync_target_marker(env):
    if env.ik_target_mocap_id < 0:
        return
    with env._teleop_lock:
        target = env.target_pos.copy()
    env.data.mocap_pos[env.ik_target_mocap_id] = target
    env.data.mocap_quat[env.ik_target_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])


def nudge_target(env, dx=0.0, dy=0.0, dz=0.0):
    with env._teleop_lock:
        env.target_pos[0] += dx * env.teleop_nudge_step
        env.target_pos[1] += dy * env.teleop_nudge_step
        env.target_pos[2] += dz * env.teleop_nudge_step


def set_gripper(env, closed):
    with env._teleop_lock:
        env.gripper_closed = closed


def nudge_roll(env, delta):
    step = 0.08
    with env._teleop_lock:
        env.target_roll += delta * step


def nudge_orientation(env, axis, delta):
    step = 0.08
    with env._teleop_lock:
        env.target_rot = _axis_rotation(axis, delta * step) @ env.target_rot


def adjust_roll_cmd(env, delta):
    with env._teleop_lock:
        env._roll_cmd = np.clip(env._roll_cmd + delta, -1.0, 1.0)


def adjust_orientation_cmd(env, axis, delta):
    with env._teleop_lock:
        env._orientation_cmd[axis] = np.clip(env._orientation_cmd[axis] + delta, -1.0, 1.0)


def adjust_move_cmd(env, axis, delta):
    with env._teleop_lock:
        env._move_cmd[axis] = np.clip(env._move_cmd[axis] + delta, -1.0, 1.0)


def apply_motion(env, dt, roll_enabled=False, orientation_enabled=False):
    with env._teleop_lock:
        move_cmd = env._move_cmd.copy()
        orientation_cmd = env._orientation_cmd.copy()
        roll_cmd = env._roll_cmd
    orienting = orientation_enabled and np.any(orientation_cmd != 0)
    if np.any(move_cmd != 0) or (roll_enabled and roll_cmd != 0) or orienting:
        with env._teleop_lock:
            if np.any(move_cmd != 0):
                env.target_pos += move_cmd * env.teleop_speed * dt
            if orienting:
                delta = orientation_cmd * env.orientation_speed * dt
                env.target_rot = _axis_rotation(2, delta[2]) @ env.target_rot
                env.target_rot = _axis_rotation(1, delta[1]) @ env.target_rot
                env.target_rot = _axis_rotation(0, delta[0]) @ env.target_rot
            if roll_enabled and roll_cmd != 0:
                env.target_roll += roll_cmd * env.roll_speed * dt


def _axis_rotation(axis, angle):
    c = np.cos(angle)
    s = np.sin(angle)
    if axis == 0:
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == 1:
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def orientation_error(current_rot, target_rot):
    rot_err = target_rot @ current_rot.T
    return 0.5 * np.array([
        rot_err[2, 1] - rot_err[1, 2],
        rot_err[0, 2] - rot_err[2, 0],
        rot_err[1, 0] - rot_err[0, 1],
    ])


def solve_position_ik(env, target_pos):
    saved_qpos = env.data.qpos.copy()
    q_cmd = saved_qpos[:7].copy()
    dls_lambda = 0.025
    max_dq = 0.10
    cart_step_cap = 0.18

    try:
        for _ in range(20):
            env.data.qpos[:7] = q_cmd
            mujoco.mj_kinematics(env.model, env.data)
            ee = env.data.xpos[env.hand_body_id].copy()
            error = target_pos - ee
            error_norm = np.linalg.norm(error)
            if error_norm < 3e-3:
                break

            step_error = error / max(error_norm, 1e-6) * min(error_norm, cart_step_cap)
            jac_p = np.zeros((3, env.model.nv))
            jac_r = np.zeros((3, env.model.nv))
            mujoco.mj_jac(
                env.model, env.data, jac_p, jac_r, ee, env.hand_body_id,
            )
            j_arm = jac_p[:, :7]
            dq = j_arm.T @ np.linalg.solve(
                j_arm @ j_arm.T + dls_lambda ** 2 * np.eye(3),
                step_error,
            )
            q_cmd += np.clip(dq, -max_dq, max_dq)
    finally:
        env.data.qpos[:] = saved_qpos
        mujoco.mj_kinematics(env.model, env.data)

    return q_cmd


def solve_pose_ik(env, target_pos, target_rot):
    saved_qpos = env.data.qpos.copy()
    q_cmd = saved_qpos[:7].copy()
    dls_lambda = 0.025
    max_dq = 0.10
    pos_step_cap = 0.18
    ori_step_cap = 0.35
    pos_weight = 1.0
    ori_weight = 1.8

    try:
        for _ in range(24):
            env.data.qpos[:7] = q_cmd
            mujoco.mj_kinematics(env.model, env.data)
            ee = env.data.xpos[env.hand_body_id].copy()
            current_rot = env.data.xmat[env.hand_body_id].reshape(3, 3).copy()

            pos_err = target_pos - ee
            ori_err = orientation_error(current_rot, target_rot)
            pos_norm = np.linalg.norm(pos_err)
            ori_norm = np.linalg.norm(ori_err)
            if pos_norm < 3e-3 and ori_norm < 0.04:
                break

            if pos_norm > 1e-6:
                pos_step = pos_err / pos_norm * min(pos_norm, pos_step_cap)
            else:
                pos_step = np.zeros(3)
            if ori_norm > 1e-6:
                ori_step = ori_err / ori_norm * min(ori_norm, ori_step_cap)
            else:
                ori_step = np.zeros(3)

            task_err = np.concatenate([pos_weight * pos_step, ori_weight * ori_step])
            jac_p = np.zeros((3, env.model.nv))
            jac_r = np.zeros((3, env.model.nv))
            mujoco.mj_jac(
                env.model, env.data, jac_p, jac_r, ee, env.hand_body_id,
            )
            j_arm = np.vstack([pos_weight * jac_p[:, :7], ori_weight * jac_r[:, :7]])
            dq = j_arm.T @ np.linalg.solve(
                j_arm @ j_arm.T + dls_lambda ** 2 * np.eye(6),
                task_err,
            )
            q_cmd += np.clip(dq, -max_dq, max_dq)
    finally:
        env.data.qpos[:] = saved_qpos
        mujoco.mj_kinematics(env.model, env.data)

    return q_cmd


def apply_position_ik_control(env):
    with env._teleop_lock:
        target_pos = env.target_pos.copy()
        gripper_closed = env.gripper_closed

    env.data.qfrc_applied[:7] = 0.0
    env.impedance_tau_norm = 0.0
    q_des = solve_position_ik(env, target_pos)
    for i in range(7):
        env.data.ctrl[i] = q_des[i]
    env.data.ctrl[7] = 255.0 if gripper_closed else 0.0


def apply_free_pose_ik_control(env):
    with env._teleop_lock:
        target_pos = env.target_pos.copy()
        target_rot = env.target_rot.copy()
        gripper_closed = env.gripper_closed

    env.data.qfrc_applied[:7] = 0.0
    env.impedance_tau_norm = 0.0
    q_des = solve_pose_ik(env, target_pos, target_rot)
    for i in range(7):
        env.data.ctrl[i] = q_des[i]
    env.data.ctrl[7] = 255.0 if gripper_closed else 0.0


def hud_text(env, force_line="", show_roll=False, show_free_orientation=False, suffix=""):
    with env._teleop_lock:
        target = env.target_pos.copy()
        gripper = "closed" if env.gripper_closed else "open"
        moving = (
            np.any(env._move_cmd != 0)
            or (show_roll and env._roll_cmd != 0)
            or (show_free_orientation and np.any(env._orientation_cmd != 0))
        )
        roll_deg = np.degrees(env.target_roll)
    title = "Arrows=N/S/E/W | 9/8=Z | ,/.=gripper"
    if show_roll:
        title = "Arrows=N/S/E/W | 9/8=Z | 6/7=roll | ,/.=gripper"
    if show_free_orientation:
        title = "Arrows=N/S/E/W | 9/8=Z | [/]=pitch | -/= yaw | 6/7=roll | ,/.=gripper"
    body = f"target ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})"
    if show_roll:
        body += f"  roll {roll_deg:.0f} deg"
    if show_free_orientation:
        body += "  free orientation"
    body += f"  {gripper}"
    body += "  MOVING" if moving else ""
    body += f"  |  {force_line}" if force_line else ""
    body += suffix
    return (
        title,
        body,
    )
