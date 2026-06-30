import mujoco

from .base import TargetContactScenario


class HitFloorScenario(TargetContactScenario):
    name = "hit_floor"

    def resolve_ids(self, env):
        env.floor_geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    def apply_control(self, env):
        env.data.ctrl[1] = 0.229
        env.data.ctrl[3] = -2.20
        env.data.ctrl[5] = 2.30
        env.data.ctrl[6] = 0.80
        env.data.ctrl[7] = 255

    def is_target_contact(self, env, contact, gripper_ids):
        body1 = env.model.geom_bodyid[contact.geom1]
        body2 = env.model.geom_bodyid[contact.geom2]
        gripper_on_1 = body1 in gripper_ids
        gripper_on_2 = body2 in gripper_ids
        floor_on_1 = contact.geom1 == env.floor_geom_id
        floor_on_2 = contact.geom2 == env.floor_geom_id
        return (gripper_on_1 and floor_on_2) or (gripper_on_2 and floor_on_1)
