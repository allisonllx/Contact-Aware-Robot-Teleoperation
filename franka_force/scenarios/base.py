import mujoco
import numpy as np

from .. import force_visuals, teleop


class Scenario:
    name = ""
    supports_interactive = False

    def initialize_state(self, env):
        pass

    def augment_model_spec(self, env, spec):
        pass

    def resolve_ids(self, env):
        pass

    def after_model_init(self, env):
        pass

    def apply_control(self, env):
        raise NotImplementedError

    def sample_forces(self, env):
        raise NotImplementedError

    def print_controls(self, env):
        teleop.print_controls(
            env,
            self.name.upper(),
            "Goal: Move the robot and observe contact forces.",
            free_orientation_controls=env.free_orientation,
        )

    def start_interactive(self, env):
        teleop.start(env, orientation_enabled=env.free_orientation)

    def stop_interactive(self, env):
        teleop.stop(env)

    def configure_viewer_camera(self, env, camera):
        pass

    def recording_camera(self, env, viewer_camera):
        return viewer_camera

    def viewer_key_callback(self, env, keycode):
        teleop.viewer_key_callback(env, keycode, orientation_enabled=env.free_orientation)

    def before_interactive_step(self, env, dt):
        teleop.before_step(env, dt, orientation_enabled=env.free_orientation)

    def after_step(self, env, dt):
        pass

    def update_interactive_viewer(self, env, viewer):
        force_visuals.draw_force_feedback(env, viewer.user_scn, clear=True)
        self._update_generic_hud(env, viewer)

    def update_recording_scene(self, env, scene):
        force_visuals.draw_force_feedback(env, scene, clear=False)

    def force_feedback_display_pos(self, env, contact_pos):
        return np.asarray(contact_pos, dtype=np.float64).copy()

    def force_feedback_ring_frame(self, env):
        frame = env.latest_contact_frame
        normal = force_visuals.unit_vector(frame[0], np.array([0.0, 0.0, 1.0]))
        tangent_a = force_visuals.unit_vector(frame[1], np.array([1.0, 0.0, 0.0]))
        tangent_b = np.cross(normal, tangent_a)
        if np.linalg.norm(tangent_b) < 1e-9:
            tangent_b = force_visuals.unit_vector(frame[2], np.array([0.0, 1.0, 0.0]))
        else:
            tangent_b = force_visuals.unit_vector(tangent_b, np.array([0.0, 1.0, 0.0]))
        tangent_a = force_visuals.unit_vector(np.cross(tangent_b, normal), tangent_a)
        center = env.latest_contact_pos + normal * 0.002
        return center, normal, tangent_a, tangent_b

    def _update_generic_hud(self, env, viewer):
        force_line = ""
        if env._force_feedback_overlay_enabled():
            force_line = (
                f"force {env._force_feedback_magnitude():.1f} N"
                + (" (contact)" if env.latest_in_contact else " (no contact yet)")
                + f" | visual {env.force_visual}"
            )
        title, body = teleop.hud_text(
            env,
            force_line,
            show_free_orientation=env.free_orientation,
        )
        viewer.set_texts([
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                title,
                body,
            ),
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                "Compass on floor: blue=N (Up arrow)  red=E (Right arrow)",
                "",
            ),
        ])


class TargetContactScenario(Scenario):
    supports_interactive = True

    def is_target_contact(self, env, contact, gripper_ids):
        raise NotImplementedError

    def sample_forces(self, env):
        gripper_ids = env._get_active_gripper_body_ids()
        (
            in_contact,
            f_true,
            contact_pos,
            contact_frame,
            contact_force,
            contact_force_vector,
        ) = self._target_contact_summary(env, gripper_ids)
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

    def _target_contact_summary(self, env, gripper_ids):
        force_world = np.zeros(3)
        strongest_force = 0.0
        strongest_pos = None
        strongest_frame = None
        strongest_force_vector = np.zeros(3)

        for i in range(env.data.ncon):
            contact = env.data.contact[i]
            if not self.is_target_contact(env, contact, gripper_ids):
                continue

            contact_force = self._contact_force_on_gripper(env, contact, i, gripper_ids)
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

    def _contact_force_on_gripper(self, env, contact, contact_idx, gripper_ids):
        force_world = env._contact_force_in_world(contact_idx)
        body1 = env.model.geom_bodyid[contact.geom1]
        body2 = env.model.geom_bodyid[contact.geom2]
        if body2 in gripper_ids:
            return force_world
        if body1 in gripper_ids:
            return -force_world
        return force_world
