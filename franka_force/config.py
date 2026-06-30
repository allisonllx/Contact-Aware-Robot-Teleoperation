from pathlib import Path

MODEL_PATH = Path("mujoco_menagerie/franka_emika_panda/scene.xml")
RESULTS_DIR = Path("results")
SCENARIOS = ("hit_floor", "push_block", "peg_in_hole")
FORCE_VISUAL_MODES = ("arrow", "ring", "both")

DEFAULT_CUSHION_THRESHOLD = 100.0
DEFAULT_IMPEDANCE_KP = 300.0
DEFAULT_IMPEDANCE_DP = 35.0
DEFAULT_IMPEDANCE_KR = 20.0
DEFAULT_IMPEDANCE_DR = 4.0
DEFAULT_IMPEDANCE_TORQUE_LIMIT = 30.0

VIDEO_FPS = 30
VIDEO_WIDTH = 960
VIDEO_HEIGHT = 540
VIDEO_CAPTURE_EVERY = 2
