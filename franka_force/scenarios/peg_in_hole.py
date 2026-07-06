import mujoco
import numpy as np

from .. import force_visuals, teleop
from .base import Scenario


class PegInHoleScenario(Scenario):
    name = "peg_in_hole"
    supports_interactive = True
    force_visual_min = 10.0
    force_visual_max = 1000.0
    peg_radius = 0.012
    socket_origin = np.array([0.50, 0.0, 0.0])
    socket_wall_thick = 0.015
    socket_wall_len = 0.05
    socket_wall_height = 0.04
    occluder_thick = 0.006
    occluder_gap = 0.015
    occluder_extra_gap = 0.020
    occluder_extra_height = 0.04
    occluder_width_scale = 1.35
    occluder_x_offset = 0.025
    occluded_socket_offset = np.array([0.0, 0.025, 0.0])
    occluded_feedback_front_margin = 0.008
    occluded_feedback_lift = 0.025
    success_pad_thickness = 0.001
    success_hold_required = 0.15

    def initialize_state(self, env):
        env._peg_home_q = np.array([0.0, 0.229, 0.0, -1.80, 0.0, 2.25, 0.80])
        env._peg_down = np.array([0.0, 0.0, -1.0])
        env.cushion_release_threshold = env.cushion_threshold * 0.60
        env._occluded_success_announced = False
        env._occluded_recording_camera = None
        if env.occluded_task:
            env.occluded_hole_world_pos = self._socket_origin(env).copy()

    def augment_model_spec(self, env, spec):
        hand_body = spec.body("hand")
        hand_body.add_geom(
            name="peg_geom",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[self.peg_radius, 0.05],
            pos=[0, 0, 0.10],
            rgba=[0.8, 0.8, 0.8, env.peg_alpha],
            mass=0.2,
            condim=3,
        )

        socket_base = spec.worldbody.add_body(name="socket", pos=list(self._socket_origin(env)))
        wall_thick = self.socket_wall_thick
        wall_len = self.socket_wall_len
        wall_height = self.socket_wall_height
        hole_gap = self._hole_gap(env)
        socket_rgba = [0.4, 0.4, 0.4, env.socket_alpha]

        socket_base.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[wall_thick, wall_len, wall_height],
            pos=[-hole_gap - wall_thick, 0, wall_height],
            rgba=socket_rgba,
        )
        socket_base.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[wall_thick, wall_len, wall_height],
            pos=[hole_gap + wall_thick, 0, wall_height],
            rgba=socket_rgba,
        )
        socket_base.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[wall_len, wall_thick, wall_height],
            pos=[0, -hole_gap - wall_thick, wall_height],
            rgba=socket_rgba,
        )
        socket_base.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[wall_len, wall_thick, wall_height],
            pos=[0, hole_gap + wall_thick, wall_height],
            rgba=socket_rgba,
        )

        if env.occluded_task:
            self._add_occluded_task_geoms(env, spec, socket_base)

        teleop.add_target_marker(spec)
        teleop.add_floor_compass(spec, origin=[0.38, 0.0, 0.0])

    def _add_occluded_task_geoms(self, env, spec, socket_base):
        occluder_height = self._occluder_height()
        hole_gap = self._hole_gap(env)
        spec.worldbody.add_geom(
            name="occlusion_wall_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[
                self.socket_wall_len * self.occluder_width_scale,
                self.occluder_thick,
                occluder_height / 2.0,
            ],
            pos=list(self._occluder_world_pos()),
            rgba=[0.08, 0.08, 0.08, 1.0],
            contype=0,
            conaffinity=0,
        )
        socket_base.add_geom(
            name="peg_success_pad_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[
                hole_gap * 0.75,
                hole_gap * 0.75,
                self.success_pad_thickness,
            ],
            pos=[0.0, 0.0, self.success_pad_thickness],
            rgba=[0.05, 0.85, 0.20, 0.0],
            contype=1,
            conaffinity=1,
            condim=3,
        )

    def resolve_ids(self, env):
        env.peg_geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "peg_geom")
        teleop.resolve_ids(env)
        env.occlusion_wall_geom_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_GEOM, "occlusion_wall_geom",
        )
        env.success_pad_geom_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_GEOM, "peg_success_pad_geom",
        )

    def after_model_init(self, env):
        if env.interactive:
            teleop.boost_arm_actuators(env)
        self._init_peg_home_pose(env)

    def apply_control(self, env):
        self._apply_peg_ik_control(env)

    def sample_forces(self, env):
        (
            in_contact,
            f_true,
            contact_pos,
            contact_frame,
            contact_force,
            contact_force_vector,
        ) = self._peg_contact_summary(env)
        env.latest_contact_pos = contact_pos
        env.latest_contact_frame = contact_frame
        env.latest_contact_force = contact_force
        env.latest_contact_force_vector = contact_force_vector
        force_visuals.update_contact_visual(
            env,
            in_contact,
            contact_pos,
            contact_force,
            contact_force_vector,
        )
        f_est = env._estimate_virtual_force() if in_contact else 0.0
        return in_contact, f_true, f_est

    def print_controls(self, env):
        print("\n=== INTERACTIVE PEG-IN-HOLE TASK ===")
        print("Goal: Align the peg over the socket and insert safely.")
        print()
        print("Click the MuJoCo window, then use:")
        print("  Arrow keys   : Up/Down = North/South, Left/Right = West/East")
        print("  9 / 8        : Raise / lower target (Z)")
        print("  6 / 7        : Roll peg left / right (spin about vertical)")
        print("  Page Up/Down : Also raise / lower (Z), if your keyboard has them")
        print("  , / .        : Open / close gripper")
        print()
        print("DO NOT press I, J, K, or U — those are MuJoCo debug toggles")
        print("(red collision boxes, joint axes, etc.), not robot controls.")
        print("Hold arrow keys for smooth motion if pynput is installed.")
        print("Peg orientation is locked pointing down; use 6/7 to spin it.")
        print(f"Hole clearance: {env.hole_clearance_mm:.2f} mm total.")
        if env._force_feedback_overlay_enabled():
            print("Force feedback overlay: ON")
            print(f"  Visual mode: {env.force_visual}")
            print("  Green sphere above hand = waiting for contact.")
            print("  Red/orange arrow shows the raw contact-force direction and magnitude.")
            print("  Red/orange contact ring is centered on the strongest contact surface.")
            if env.occluded_task:
                print("  Occluded mode projects feedback in front of the wall so it stays visible.")
            if env.force_feedback:
                print("  Live viewer overlay enabled.")
            if env.record_force_feedback:
                print("  Recording overlay enabled.")
        else:
            print("Force feedback overlay: OFF")
        if env.contact_cushion:
            print("Experimental impedance cushion: ON")
            print(f"  Activates above {env.cushion_threshold:.1f} N, releases below {env.cushion_release_threshold:.1f} N")
        else:
            print("Experimental impedance cushion: OFF")
        if env.occluded_task:
            print("Occluded task: ON")
            print("  Wider opaque front wall hides an off-center socket and hidden success pad.")
            if env.randomize_occluded_hole:
                print(
                    "  Hidden socket randomization: ON "
                    f"(X {env.occluded_hole_x_range[0]:+.3f} to {env.occluded_hole_x_range[1]:+.3f} m, "
                    f"Y {env.occluded_hole_y_range[0]:+.3f} to {env.occluded_hole_y_range[1]:+.3f} m)."
                )
            print("  Live camera starts wide front-on; recordings use a side observer camera.")
            print(f"  Success requires {self.success_hold_required:.2f}s of sustained peg-pad contact.")
        if env.audio_feedback:
            print("Audio feedback: ON")
            print(f"  Mode: {env.audio_mode}")
            print(f"  Contact click above {env.audio_contact_threshold:.1f} N Jacobian estimate.")
            print(
                f"  Geiger ticks above {env.audio_lateral_threshold:.1f} N lateral force, "
                f"maxing near {env.audio_lateral_max:.1f} N."
            )
        else:
            print("Audio feedback: OFF")
        print()

    def start_interactive(self, env):
        teleop.start(env, roll_enabled=True)

    def stop_interactive(self, env):
        teleop.stop(env)

    def configure_viewer_camera(self, env, camera):
        if env.occluded_task:
            self._set_free_camera(
                camera,
                lookat=self._nominal_occluded_socket_origin() + np.array([0.055, -0.02, 0.18]),
                distance=0.56,
                azimuth=90.0,
                elevation=-3.0,
            )

    def recording_camera(self, env, viewer_camera):
        if not env.occluded_task:
            return viewer_camera
        if env._occluded_recording_camera is None:
            env._occluded_recording_camera = mujoco.MjvCamera()
            self._set_free_camera(
                env._occluded_recording_camera,
                lookat=self._nominal_occluded_socket_origin() + np.array([0.0, -0.005, 0.07]),
                distance=0.52,
                azimuth=135.0,
                elevation=-25.0,
            )
        return env._occluded_recording_camera

    def viewer_key_callback(self, env, keycode):
        teleop.viewer_key_callback(env, keycode, roll_enabled=True)

    def before_interactive_step(self, env, dt):
        teleop.before_step(env, dt, roll_enabled=True)

    def after_step(self, env, dt):
        self._update_occluded_success(env, dt)

    def update_interactive_viewer(self, env, viewer):
        force_visuals.draw_force_feedback(env, viewer.user_scn, clear=True)
        self._update_peg_hud(env, viewer)

    def update_recording_scene(self, env, scene):
        force_visuals.draw_force_feedback(env, scene, clear=False)

    def _socket_origin(self, env):
        if env.occluded_task:
            return self._nominal_occluded_socket_origin() + env.occluded_hole_offset
        return self.socket_origin

    def _nominal_occluded_socket_origin(self):
        return self.socket_origin + self.occluded_socket_offset

    def _occluder_world_pos(self):
        origin = self._nominal_occluded_socket_origin()
        return origin + np.array([
            self.occluder_x_offset,
            self._occluder_local_y(),
            self._occluder_height() / 2.0,
        ])

    def _hole_gap(self, env):
        return self._hole_gap_from_clearance(env.hole_clearance_mm)

    def _hole_gap_from_clearance(self, clearance_mm):
        clearance_m = clearance_mm / 1000.0
        return self.peg_radius + clearance_m / 2.0

    def _occluder_height(self):
        return self.socket_wall_height * 2.0 + self.occluder_extra_height

    def _occluder_local_y(self):
        return (
            -self.socket_wall_len
            - self.occluder_gap
            - self.occluder_extra_gap
            - self.occluder_thick
        )

    def _occluded_feedback_y(self, env):
        socket_origin = self._nominal_occluded_socket_origin()
        return (
            socket_origin[1]
            + self._occluder_local_y()
            - self.occluder_thick
            - self.occluded_feedback_front_margin
        )

    def force_feedback_display_pos(self, env, contact_pos):
        pos = np.asarray(contact_pos, dtype=np.float64).copy()
        if not env.occluded_task:
            return pos

        pos[1] = self._occluded_feedback_y(env)
        pos[2] = max(pos[2], self._occluder_height() + self.occluded_feedback_lift)
        return pos

    def force_feedback_ring_frame(self, env):
        if not env.occluded_task:
            return super().force_feedback_ring_frame(env)

        normal = np.array([0.0, 0.0, 1.0])
        tangent_a = np.array([1.0, 0.0, 0.0])
        tangent_b = np.array([0.0, 1.0, 0.0])
        center = self.force_feedback_display_pos(env, env.latest_contact_pos)
        return center, normal, tangent_a, tangent_b

    def _force_feedback_display_pos(self, env, contact_pos):
        return self.force_feedback_display_pos(env, contact_pos)

    def _set_free_camera(self, camera, lookat, distance, azimuth, elevation):
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.fixedcamid = -1
        camera.lookat[:] = lookat
        camera.distance = distance
        camera.azimuth = azimuth
        camera.elevation = elevation

    def _init_peg_home_pose(self, env):
        """Place the arm over the socket and sync the IK target to the current hand pose."""
        env.data.qpos[:7] = env._peg_home_q
        mujoco.mj_forward(env.model, env.data)
        for i in range(7):
            env.data.ctrl[i] = env._peg_home_q[i]
        env.data.ctrl[7] = 0.0
        with env._teleop_lock:
            env.target_pos = env.data.xpos[env.hand_body_id].copy()
        self._sync_target_marker(env)

    def _target_hand_rotmat(self, env, roll):
        """Hand frame with peg axis (+Z) pointing down; roll spins peg about vertical."""
        z_des = env._peg_down
        x_des = np.array([np.cos(roll), np.sin(roll), 0.0])
        y_des = np.cross(z_des, x_des)
        return np.column_stack([x_des, y_des, z_des])

    def _orientation_error(self, current_rot, target_rot):
        rot_err = target_rot @ current_rot.T
        return 0.5 * np.array([
            rot_err[2, 1] - rot_err[1, 2],
            rot_err[0, 2] - rot_err[2, 0],
            rot_err[1, 0] - rot_err[0, 1],
        ])

    def _solve_peg_ik(self, env, target_pos, target_roll):
        """6-DOF iterative IK: reach target_pos with peg axis pointing down."""
        saved_qpos = env.data.qpos.copy()
        q_cmd = saved_qpos[:7].copy()
        target_rot = self._target_hand_rotmat(env, target_roll)
        dls_lambda = 0.025
        max_dq = 0.10
        pos_step_cap = 0.18
        ori_step_cap = 0.35
        pos_weight = 1.0
        ori_weight = 2.5

        try:
            for _ in range(24):
                env.data.qpos[:7] = q_cmd
                mujoco.mj_kinematics(env.model, env.data)
                ee = env.data.xpos[env.hand_body_id].copy()
                current_rot = env.data.xmat[env.hand_body_id].reshape(3, 3).copy()

                pos_err = target_pos - ee
                ori_err = self._orientation_error(current_rot, target_rot)
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

    def _solve_peg_ik_pos_only(self, env, target_pos):
        """3-DOF fallback for non-interactive peg mode."""
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

    def _apply_peg_ik_control(self, env):
        """IK toward target pose (interactive: 6-DOF peg-down + roll)."""
        with env._teleop_lock:
            target_pos = env.target_pos.copy()
            target_roll = env.target_roll
            gripper_closed = env.gripper_closed

        env.data.qfrc_applied[:7] = 0.0
        if env.contact_cushion and env.interactive and self._update_cushion_state(env):
            self._apply_impedance_control(env, target_pos, target_roll, gripper_closed)
            return

        env.impedance_tau_norm = 0.0
        if env.interactive:
            q_des = self._solve_peg_ik(env, target_pos, target_roll)
        else:
            q_des = self._solve_peg_ik_pos_only(env, target_pos)

        for i in range(7):
            env.data.ctrl[i] = q_des[i]
        env.data.ctrl[7] = 255.0 if gripper_closed else 0.0

    def _update_cushion_state(self, env):
        force_magnitude = max(env.latest_contact_force, env.latest_f_true, env.latest_f_est)
        if env.cushion_active:
            env.cushion_active = force_magnitude >= env.cushion_release_threshold
        else:
            env.cushion_active = force_magnitude >= env.cushion_threshold

        if env.cushion_active:
            env.cushion_scale = max(0.05, self._force_visual_intensity(force_magnitude))
        else:
            env.cushion_scale = 0.0
            env.impedance_tau_norm = 0.0

        return env.cushion_active

    def _apply_impedance_control(self, env, target_pos, target_roll, gripper_closed):
        """Experimental contact cushion: soft Cartesian spring/damper around the teleop target."""
        for i in range(7):
            env.data.ctrl[i] = env.data.qpos[i]
        env.data.ctrl[7] = 255.0 if gripper_closed else 0.0

        hand_pos = env.data.xpos[env.hand_body_id].copy()
        current_rot = env.data.xmat[env.hand_body_id].reshape(3, 3).copy()
        target_rot = self._target_hand_rotmat(env, target_roll)

        jac_p = np.zeros((3, env.model.nv))
        jac_r = np.zeros((3, env.model.nv))
        mujoco.mj_jac(env.model, env.data, jac_p, jac_r, hand_pos, env.hand_body_id)
        j_arm = np.vstack([jac_p[:, :7], jac_r[:, :7]])
        qvel_arm = env.data.qvel[:7]

        pos_err = target_pos - hand_pos
        ori_err = self._orientation_error(current_rot, target_rot)
        lin_vel = jac_p[:, :7] @ qvel_arm
        ang_vel = jac_r[:, :7] @ qvel_arm

        linear_force = env.impedance_kp * pos_err - env.impedance_dp * lin_vel
        angular_torque = env.impedance_kr * ori_err - env.impedance_dr * ang_vel
        wrench = np.concatenate([linear_force, angular_torque])

        tau = j_arm.T @ wrench
        tau = np.nan_to_num(
            tau,
            nan=0.0,
            posinf=env.impedance_torque_limit,
            neginf=-env.impedance_torque_limit,
        )
        tau = np.clip(tau, -env.impedance_torque_limit, env.impedance_torque_limit)
        env.data.qfrc_applied[:7] = tau
        env.impedance_tau_norm = float(np.linalg.norm(tau))

    def _peg_contact_summary(self, env):
        force_world = np.zeros(3)
        strongest_force = 0.0
        strongest_pos = None
        strongest_frame = None
        strongest_force_vector = np.zeros(3)

        for i in range(env.data.ncon):
            contact = env.data.contact[i]
            if contact.geom1 != env.peg_geom_id and contact.geom2 != env.peg_geom_id:
                continue

            contact_force = self._contact_force_on_peg(env, contact, i)
            contact_force_mag = float(np.linalg.norm(contact_force))
            force_world += contact_force

            if contact_force_mag > strongest_force:
                strongest_force = contact_force_mag
                strongest_pos = contact.pos.copy()
                strongest_frame = contact.frame.reshape(3, 3).copy()
                strongest_force_vector = contact_force.copy()

        return (
            strongest_pos is not None,
            float(np.linalg.norm(force_world)),
            strongest_pos,
            strongest_frame,
            strongest_force,
            strongest_force_vector,
        )

    def _contact_force_on_peg(self, env, contact, contact_idx):
        """Return raw contact force in world frame with a consistent peg sign."""
        c_forces = np.zeros(6)
        mujoco.mj_contactForce(env.model, env.data, contact_idx, c_forces)
        frame = contact.frame.reshape(3, 3)
        full_force = frame.T @ c_forces[:3]

        if contact.geom2 == env.peg_geom_id:
            return full_force
        return -full_force

    def _update_occluded_success(self, env, dt):
        if not env.occluded_task or env.task_success:
            return

        success_contact = self._has_success_pad_contact(env)
        self._update_success_state(env, success_contact, dt)

        if env.task_success and not env._occluded_success_announced:
            env._occluded_success_announced = True
            print(
                f"Occluded peg-in-hole SUCCESS at t={env.data.time:.3f}s "
                f"after {env.success_hold_time:.2f}s of hidden pad contact."
            )

    def _has_success_pad_contact(self, env):
        if env.success_pad_geom_id < 0:
            return False

        for i in range(env.data.ncon):
            contact = env.data.contact[i]
            geoms = {contact.geom1, contact.geom2}
            if env.peg_geom_id in geoms and env.success_pad_geom_id in geoms:
                return True
        return False

    def _update_success_state(self, env, success_contact, dt):
        env.success_contact = success_contact
        if success_contact:
            env.success_hold_time += dt
        else:
            env.success_hold_time = 0.0

        if env.success_hold_time >= self.success_hold_required:
            env.task_success = True
            env.task_stop_requested = True

    def _sync_target_marker(self, env):
        teleop.sync_target_marker(env)

    def _nudge_target(self, env, dx=0.0, dy=0.0, dz=0.0):
        teleop.nudge_target(env, dx=dx, dy=dy, dz=dz)

    def _set_gripper(self, env, closed):
        teleop.set_gripper(env, closed)

    def _nudge_roll(self, env, delta):
        teleop.nudge_roll(env, delta)

    def _adjust_roll_cmd(self, env, delta):
        teleop.adjust_roll_cmd(env, delta)

    def _apply_teleop_motion(self, env, dt):
        teleop.apply_motion(env, dt, roll_enabled=True)

    def _adjust_move_cmd(self, env, axis, delta):
        teleop.adjust_move_cmd(env, axis, delta)

    def _start_pynput_teleop(self, env):
        teleop.start(env, roll_enabled=True)

    def _stop_pynput_teleop(self, env):
        teleop.stop(env)

    def _update_peg_hud(self, env, viewer):
        with env._teleop_lock:
            target = env.target_pos.copy()
            gripper = "closed" if env.gripper_closed else "open"
            moving = np.any(env._move_cmd != 0) or env._roll_cmd != 0
            roll_deg = np.degrees(env.target_roll)
        force_line = ""
        if env._force_feedback_overlay_enabled():
            f_display = env._force_feedback_magnitude()
            force_line = (
                f"force {f_display:.1f} N"
                + (" (contact)" if env.latest_in_contact else " (no contact yet)")
                + f" | visual {env.force_visual}"
            )
        if env.contact_cushion:
            cushion_state = "ON" if env.cushion_active else "idle"
            cushion_line = (
                f"cushion {cushion_state}"
                f" @{env.cushion_threshold:.0f}N"
                f" scale {env.cushion_scale:.2f}"
            )
            force_line = f"{force_line} | {cushion_line}" if force_line else cushion_line
        if env.occluded_task:
            if env.task_success:
                success_line = "occluded SUCCESS"
            else:
                contact_state = "contact" if env.success_contact else "seeking"
                success_line = (
                    f"occluded {contact_state}"
                    f" {env.success_hold_time:.2f}/{self.success_hold_required:.2f}s"
                )
            force_line = f"{force_line} | {success_line}" if force_line else success_line
        viewer.set_texts([
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                "Arrows=N/S/E/W | 9/8=Z | 6/7=roll | ,/.=gripper | peg locked down",
                f"target ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})  roll {roll_deg:.0f}°  {gripper}"
                + ("  MOVING" if moving else "")
                + (f"  |  {force_line}" if force_line else ""),
            ),
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                "Compass on floor: blue=N (Up arrow)  red=E (Right arrow)",
                "",
            ),
        ])

    def _draw_force_feedback(self, env, scene, clear):
        force_visuals.draw_force_feedback(env, scene, clear)

    def _force_visual_intensity(self, force_magnitude):
        if force_magnitude <= self.force_visual_min:
            return 0.0
        log_min = np.log(self.force_visual_min)
        log_max = np.log(self.force_visual_max)
        log_force = np.log(min(force_magnitude, self.force_visual_max))
        return float(np.clip((log_force - log_min) / (log_max - log_min), 0.0, 1.0))

    def _force_feedback_magnitude(self, env):
        return env._force_feedback_magnitude()
