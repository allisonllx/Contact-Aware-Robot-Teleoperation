import mujoco
import numpy as np

from .. import teleop
from .base import TargetContactScenario


class HitFloorScenario(TargetContactScenario):
    name = "hit_floor"

    def initialize_state(self, env):
        env._hit_floor_home_q = np.array([0.0, 0.229, 0.0, -2.20, 0.0, 2.30, 0.80])

    def augment_model_spec(self, env, spec):
        if env.interactive:
            teleop.add_target_marker(spec)
            teleop.add_floor_compass(spec, origin=[0.38, 0.0, 0.0])

    def resolve_ids(self, env):
        env.floor_geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if env.interactive:
            teleop.resolve_ids(env)

    def after_model_init(self, env):
        if env.interactive:
            teleop.boost_arm_actuators(env)
            teleop.init_home_pose(env, env._hit_floor_home_q, gripper_closed=True)

    def apply_control(self, env):
        if env.interactive:
            if env.free_orientation:
                teleop.apply_free_pose_ik_control(env)
                return
            teleop.apply_position_ik_control(env)
            return

        env.data.ctrl[1] = 0.229
        env.data.ctrl[3] = -2.20
        env.data.ctrl[5] = 2.30
        env.data.ctrl[6] = 0.80
        env.data.ctrl[7] = 255

    def print_controls(self, env):
        teleop.print_controls(
            env,
            "HIT-FLOOR",
            "Goal: Move the gripper into the floor and observe the raw contact-force arrow.",
            free_orientation_controls=env.free_orientation,
            orientation_note=(
                "Free orientation is ON; bracket, minus/equal, and 6/7 keys rotate the end effector."
                if env.free_orientation
                else "Position-only teleop is ON; orientation is left unconstrained by the target."
            ),
        )
        self._print_force_feedback_status(env)
        self._print_audio_feedback_status(env)
        print()

    def _print_force_feedback_status(self, env):
        if env._force_feedback_overlay_enabled():
            print("Force feedback overlay: ON")
            print(f"  Visual mode: {env.force_visual}")
            print("  Red/orange arrow shows the raw contact-force direction and magnitude.")
            print("  Red/orange contact ring is centered on the strongest target contact.")
        else:
            print("Force feedback overlay: OFF")

    def _print_audio_feedback_status(self, env):
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

    def is_target_contact(self, env, contact, gripper_ids):
        body1 = env.model.geom_bodyid[contact.geom1]
        body2 = env.model.geom_bodyid[contact.geom2]
        gripper_on_1 = body1 in gripper_ids
        gripper_on_2 = body2 in gripper_ids
        floor_on_1 = contact.geom1 == env.floor_geom_id
        floor_on_2 = contact.geom2 == env.floor_geom_id
        return (gripper_on_1 and floor_on_2) or (gripper_on_2 and floor_on_1)
