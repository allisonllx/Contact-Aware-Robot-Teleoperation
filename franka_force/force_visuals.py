import mujoco
import numpy as np

FORCE_VISUAL_MIN = 10.0
FORCE_VISUAL_THRESHOLD = FORCE_VISUAL_MIN
FORCE_VISUAL_MAX = 1000.0
FORCE_ARROW_MIN_LENGTH = 0.045
FORCE_ARROW_LENGTH_RANGE = 0.215
FORCE_ARROW_MIN_WIDTH = 0.0026
FORCE_ARROW_WIDTH_RANGE = 0.0064
FORCE_RING_MIN_RADIUS = 0.022
FORCE_RING_RADIUS_RANGE = 0.066
FORCE_RING_MIN_WIDTH = 0.0020
FORCE_RING_WIDTH_RANGE = 0.0042
IDLE_MARKER_OFFSET = np.array([0.0, -0.12, 0.14])


def initialize_state(env):
    env._smoothed_contact_arrow_pos = None
    env._smoothed_contact_arrow_vector = np.zeros(3)
    env._smoothed_contact_arrow_force = 0.0
    env._contact_arrow_smoothing = 0.28
    env._contact_arrow_reset_distance = 0.018


def update_contact_visual(env, in_contact, contact_pos, contact_force, force_vector):
    if not in_contact or contact_pos is None or contact_force <= FORCE_VISUAL_THRESHOLD:
        env._smoothed_contact_arrow_pos = None
        env._smoothed_contact_arrow_vector = np.zeros(3)
        env._smoothed_contact_arrow_force = 0.0
        env.latest_contact_arrow_pos = None
        env.latest_contact_arrow_vector = np.zeros(3)
        env.latest_contact_arrow_force = 0.0
        return

    contact_pos = np.asarray(contact_pos, dtype=np.float64)
    force_vector = np.asarray(force_vector, dtype=np.float64)
    if np.linalg.norm(force_vector) < 1e-9:
        force_vector = np.asarray(env.latest_contact_force_vector, dtype=np.float64)

    reset = env._smoothed_contact_arrow_pos is None
    if not reset:
        jump = np.linalg.norm(contact_pos - env._smoothed_contact_arrow_pos)
        reset = jump > env._contact_arrow_reset_distance

    if reset:
        env._smoothed_contact_arrow_pos = contact_pos.copy()
        env._smoothed_contact_arrow_vector = force_vector.copy()
        env._smoothed_contact_arrow_force = contact_force
    else:
        alpha = env._contact_arrow_smoothing
        prev_vector = env._smoothed_contact_arrow_vector
        env._smoothed_contact_arrow_pos = (
            (1.0 - alpha) * env._smoothed_contact_arrow_pos + alpha * contact_pos
        )
        env._smoothed_contact_arrow_vector = (
            (1.0 - alpha) * prev_vector + alpha * force_vector
        )
        env._smoothed_contact_arrow_force = (
            (1.0 - alpha) * env._smoothed_contact_arrow_force + alpha * contact_force
        )

    env.latest_contact_arrow_pos = env._smoothed_contact_arrow_pos.copy()
    env.latest_contact_arrow_vector = env._smoothed_contact_arrow_vector.copy()
    env.latest_contact_arrow_force = float(env._smoothed_contact_arrow_force)


def draw_force_feedback(env, scene, clear):
    if not env._force_feedback_overlay_enabled() or scene is None:
        return

    f_display = env._force_feedback_magnitude()
    hand_pos = env.data.xpos[env.hand_body_id].astype(np.float64)
    identity = np.eye(3, dtype=np.float64).reshape(9, 1)

    if clear:
        scene.ngeom = 0
    idx = scene.ngeom

    if f_display <= FORCE_VISUAL_THRESHOLD:
        idx = _draw_idle_marker(env, scene, idx, hand_pos, identity)

    if f_display > FORCE_VISUAL_THRESHOLD:
        if env.force_visual in ("arrow", "both"):
            idx = _draw_force_arrow(env, scene, idx, identity)
        if env.force_visual in ("ring", "both"):
            idx = _draw_contact_ring(env, scene, idx, identity)

    scene.ngeom = idx


def _draw_idle_marker(env, scene, idx, hand_pos, identity):
    if not _has_scene_geom_slot(scene, idx):
        return idx

    base_pos = (hand_pos + IDLE_MARKER_OFFSET).reshape(3, 1)
    base_rgba = np.array([0.25, 0.85, 0.35, 0.9], dtype=np.float32).reshape(4, 1)

    _set_user_geom(
        scene.geoms[idx],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.020, 0.0, 0.0], dtype=np.float64).reshape(3, 1),
        base_pos,
        identity,
        base_rgba,
    )
    return idx + 1


def _draw_force_arrow(env, scene, idx, identity):
    if env.latest_contact_arrow_pos is None or not _has_scene_geom_slot(scene, idx):
        return idx

    force_vector = np.asarray(env.latest_contact_arrow_vector, dtype=np.float64)
    force_magnitude = max(float(np.linalg.norm(force_vector)), env.latest_contact_arrow_force)
    if force_magnitude <= FORCE_VISUAL_THRESHOLD:
        return idx

    intensity = _force_visual_intensity(force_magnitude)
    arrow_len = FORCE_ARROW_MIN_LENGTH + intensity * FORCE_ARROW_LENGTH_RANGE
    shaft_width = FORCE_ARROW_MIN_WIDTH + intensity * FORCE_ARROW_WIDTH_RANGE
    fallback = np.array([0.0, 0.0, 1.0])
    if env.latest_contact_frame is not None:
        fallback = env.latest_contact_frame[0]
    direction = _unit_vector(force_vector, fallback)

    contact_pos = env.scenario_impl.force_feedback_display_pos(env, env.latest_contact_arrow_pos)
    p1 = contact_pos
    p2 = contact_pos + direction * arrow_len
    color = _force_color(intensity)
    zero3 = np.zeros((3, 1), dtype=np.float64)

    arrow_geom = scene.geoms[idx]
    _set_user_geom(arrow_geom, mujoco.mjtGeom.mjGEOM_ARROW, zero3, zero3, identity, color)
    mujoco.mjv_connector(arrow_geom, mujoco.mjtGeom.mjGEOM_ARROW, shaft_width, p1, p2)
    return idx + 1


def _draw_contact_ring(env, scene, idx, identity):
    if env.latest_contact_pos is None or env.latest_contact_frame is None:
        return idx

    segments = 24
    if not _has_scene_geom_slot(scene, idx + segments - 1):
        return idx

    force_magnitude = max(env.latest_contact_force, env._force_feedback_magnitude())
    intensity = _force_visual_intensity(force_magnitude)
    radius = FORCE_RING_MIN_RADIUS + intensity * FORCE_RING_RADIUS_RANGE
    ring_width = FORCE_RING_MIN_WIDTH + intensity * FORCE_RING_WIDTH_RANGE
    color = _force_color(intensity)
    center, normal, tangent_a, tangent_b = env.scenario_impl.force_feedback_ring_frame(env)
    zero3 = np.zeros((3, 1), dtype=np.float64)

    for segment in range(segments):
        theta1 = 2.0 * np.pi * segment / segments
        theta2 = 2.0 * np.pi * (segment + 1) / segments
        p1 = center + radius * (np.cos(theta1) * tangent_a + np.sin(theta1) * tangent_b)
        p2 = center + radius * (np.cos(theta2) * tangent_a + np.sin(theta2) * tangent_b)
        ring_geom = scene.geoms[idx]
        _set_user_geom(ring_geom, mujoco.mjtGeom.mjGEOM_CAPSULE, zero3, zero3, identity, color)
        mujoco.mjv_connector(ring_geom, mujoco.mjtGeom.mjGEOM_CAPSULE, ring_width, p1, p2)
        idx += 1

    return idx


def _force_visual_intensity(force_magnitude):
    if force_magnitude <= FORCE_VISUAL_MIN:
        return 0.0
    log_min = np.log(FORCE_VISUAL_MIN)
    log_max = np.log(FORCE_VISUAL_MAX)
    log_force = np.log(min(force_magnitude, FORCE_VISUAL_MAX))
    return float(np.clip((log_force - log_min) / (log_max - log_min), 0.0, 1.0))


def _force_color(intensity):
    green = 0.35 * (1.0 - intensity)
    return np.array([1.0, green, 0.02, 0.95], dtype=np.float32).reshape(4, 1)


def unit_vector(value, fallback):
    return _unit_vector(value, fallback)


def _unit_vector(value, fallback):
    value = np.asarray(value, dtype=np.float64)
    norm = np.linalg.norm(value)
    if norm < 1e-9:
        return fallback.astype(np.float64)
    return value / norm


def _set_user_geom(geom, gtype, size, pos, mat, rgba):
    mujoco.mjv_initGeom(geom, gtype, size, pos, mat, rgba)
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR


def _has_scene_geom_slot(scene, idx):
    maxgeom = getattr(scene, "maxgeom", None)
    if maxgeom is None:
        maxgeom = len(scene.geoms)
    return idx < maxgeom
