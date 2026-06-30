import mujoco

from .base import TargetContactScenario


class PushBlockScenario(TargetContactScenario):
    name = "push_block"

    def augment_model_spec(self, env, spec):
        body = spec.worldbody.add_body(name="target_block", pos=[0.55, 0.0, 0.03])
        body.add_freejoint()
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.08, 0.08, 0.06],
            mass=5.0,
            rgba=[1, 0, 0, 1],
            condim=3,
            friction=[1, 0.005, 0.0001],
        )

    def resolve_ids(self, env):
        env.block_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "target_block")

    def apply_control(self, env):
        if env.data.time < 1.5:
            env.data.ctrl[1] = 0.229
            env.data.ctrl[3] = -1.80
            env.data.ctrl[5] = 1.87
            env.data.ctrl[6] = 0.80
            env.data.ctrl[7] = 255
        elif env.data.time < 3.0:
            progress = (env.data.time - 1.5) / 1.5
            env.data.ctrl[1] = 0.229
            env.data.ctrl[3] = -1.80 + progress * (-2.37 - (-1.80))
            env.data.ctrl[5] = 1.87 + progress * (2.25 - 1.87)
            env.data.ctrl[6] = 0.80
            env.data.ctrl[7] = 255
        elif env.data.time < 6.0:
            progress = (env.data.time - 3.0) / 3.0
            env.data.ctrl[3] = -2.37 + progress * (-2.05 - (-2.37))
            env.data.ctrl[1] = 0.229 + progress * (0.420 - 0.229)
            env.data.ctrl[5] = 2.25
            env.data.ctrl[6] = 0.80
            env.data.ctrl[7] = 255
        else:
            env.data.ctrl[1] = 0.420
            env.data.ctrl[3] = -2.05
            env.data.ctrl[5] = 2.25
            env.data.ctrl[6] = 0.80
            env.data.ctrl[7] = 255

    def is_target_contact(self, env, contact, gripper_ids):
        body1 = env.model.geom_bodyid[contact.geom1]
        body2 = env.model.geom_bodyid[contact.geom2]
        gripper_on_1 = body1 in gripper_ids
        gripper_on_2 = body2 in gripper_ids
        block_on_1 = body1 == env.block_id
        block_on_2 = body2 == env.block_id
        return (gripper_on_1 and block_on_2) or (gripper_on_2 and block_on_1)
