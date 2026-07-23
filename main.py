import argparse
from pathlib import Path

from franka_force import AUDIO_MODES, FORCE_VISUAL_MODES, SCENARIOS
from franka_force.config import (
    DEFAULT_ACTUATOR_BOOST,
    DEFAULT_AUDIO_CONTACT_THRESHOLD,
    DEFAULT_AUDIO_LATERAL_MAX,
    DEFAULT_AUDIO_LATERAL_THRESHOLD,
    DEFAULT_AUDIO_VOLUME,
    DEFAULT_CUSHION_THRESHOLD,
    DEFAULT_HOLE_CLEARANCE_MM,
    DEFAULT_HOLD_TELEOP,
    DEFAULT_IMPEDANCE_DP,
    DEFAULT_IMPEDANCE_DR,
    DEFAULT_IMPEDANCE_KP,
    DEFAULT_IMPEDANCE_KR,
    DEFAULT_IMPEDANCE_TORQUE_LIMIT,
    DEFAULT_OCCLUDER_ALPHA,
    DEFAULT_OCCLUDER_STYLE,
    DEFAULT_OCCLUDED_HOLE_X_RANGE,
    DEFAULT_OCCLUDED_HOLE_Y_RANGE,
    DEFAULT_PEG_ALPHA,
    DEFAULT_SOCKET_ALPHA,
    DEFAULT_TELEOP_NUDGE_STEP,
    DEFAULT_TELEOP_SPEED,
    OCCLUDER_STYLES,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Franka force verification scenarios")
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS,
        default="push_block",
        help="Simulation scenario to run",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable keyboard control (requires mjpython on macOS)",
    )
    parser.add_argument(
        "--force-feedback",
        action="store_true",
        help="Enable live force visual overlay (--interactive only)",
    )
    parser.add_argument(
        "--force-visual",
        choices=FORCE_VISUAL_MODES,
        default="arrow",
        help="Force feedback visual to show when live or recorded feedback is enabled",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Save run_recording.mp4 in results/<scenario>/ (uses passive viewer; mjpython on macOS)",
    )
    parser.add_argument(
        "--record-force-feedback",
        action="store_true",
        help="Include force feedback overlay geoms in --record-video output",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Override the output folder for CSVs, plots, and optional video.",
    )
    parser.add_argument(
        "--disable-policy",
        action="store_true",
        help="Disable scripted scenario motion in non-interactive runs",
    )
    parser.add_argument(
        "--free-orientation",
        action="store_true",
        help="Allow interactive side-task teleop to control end-effector pitch/yaw/roll",
    )
    parser.add_argument(
        "--occluded-task",
        action="store_true",
        help="Enable peg_in_hole visual-occlusion experiment with hidden success pad",
    )
    parser.add_argument(
        "--randomize-occluded-hole",
        action="store_true",
        help="Randomize the hidden socket position for --occluded-task while keeping the obstacle fixed",
    )
    parser.add_argument(
        "--occluded-hole-x-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=DEFAULT_OCCLUDED_HOLE_X_RANGE,
        help="Hidden socket X offset range in meters around the default occluded socket center",
    )
    parser.add_argument(
        "--occluded-hole-y-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=DEFAULT_OCCLUDED_HOLE_Y_RANGE,
        help="Hidden socket Y offset range in meters around the default occluded socket center",
    )
    parser.add_argument(
        "--occluded-hole-seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible hidden socket placement",
    )
    parser.add_argument(
        "--hole-clearance-mm",
        type=float,
        default=DEFAULT_HOLE_CLEARANCE_MM,
        help="Total peg/hole clearance in millimeters for peg_in_hole",
    )
    parser.add_argument(
        "--audio-feedback",
        action="store_true",
        help="Enable live audio force cues (--interactive only)",
    )
    parser.add_argument(
        "--audio-mode",
        choices=AUDIO_MODES,
        default="both",
        help="Audio cue mode for --audio-feedback",
    )
    parser.add_argument(
        "--audio-contact-threshold",
        type=float,
        default=DEFAULT_AUDIO_CONTACT_THRESHOLD,
        help="Jacobian-estimate force threshold in newtons for contact click",
    )
    parser.add_argument(
        "--audio-lateral-threshold",
        type=float,
        default=DEFAULT_AUDIO_LATERAL_THRESHOLD,
        help="Lateral contact-force threshold in newtons for Geiger ticks",
    )
    parser.add_argument(
        "--audio-lateral-max",
        type=float,
        default=DEFAULT_AUDIO_LATERAL_MAX,
        help="Lateral contact force in newtons that maps to max Geiger tick rate",
    )
    parser.add_argument(
        "--audio-volume",
        type=float,
        default=DEFAULT_AUDIO_VOLUME,
        help="Audio cue volume from 0.0 to 1.0",
    )
    parser.add_argument(
        "--contact-cushion",
        action="store_true",
        help="Enable experimental impedance cushion (peg_in_hole + --interactive only)",
    )
    parser.add_argument(
        "--cushion-threshold",
        type=float,
        default=DEFAULT_CUSHION_THRESHOLD,
        help="Contact force threshold in newtons that activates --contact-cushion",
    )
    parser.add_argument(
        "--impedance-kp",
        type=float,
        default=DEFAULT_IMPEDANCE_KP,
        help="Cartesian translational stiffness for --contact-cushion",
    )
    parser.add_argument(
        "--impedance-dp",
        type=float,
        default=DEFAULT_IMPEDANCE_DP,
        help="Cartesian translational damping for --contact-cushion",
    )
    parser.add_argument(
        "--impedance-kr",
        type=float,
        default=DEFAULT_IMPEDANCE_KR,
        help="Cartesian rotational stiffness for --contact-cushion",
    )
    parser.add_argument(
        "--impedance-dr",
        type=float,
        default=DEFAULT_IMPEDANCE_DR,
        help="Cartesian rotational damping for --contact-cushion",
    )
    parser.add_argument(
        "--impedance-torque-limit",
        type=float,
        default=DEFAULT_IMPEDANCE_TORQUE_LIMIT,
        help="Per-joint torque clamp for --contact-cushion",
    )
    parser.add_argument(
        "--peg-alpha",
        type=float,
        default=DEFAULT_PEG_ALPHA,
        help="Peg opacity for peg_in_hole, from 0.0 transparent to 1.0 opaque",
    )
    parser.add_argument(
        "--socket-alpha",
        type=float,
        default=DEFAULT_SOCKET_ALPHA,
        help="Socket wall opacity for peg_in_hole, from 0.0 transparent to 1.0 opaque",
    )
    parser.add_argument(
        "--occluder-alpha",
        type=float,
        default=DEFAULT_OCCLUDER_ALPHA,
        help="Occlusion obstacle opacity for --occluded-task, from 0.0 transparent to 1.0 opaque",
    )
    parser.add_argument(
        "--occluder-style",
        choices=OCCLUDER_STYLES,
        default=DEFAULT_OCCLUDER_STYLE,
        help="Occlusion obstacle visual style for --occluded-task",
    )
    parser.add_argument(
        "--teleop-nudge-step",
        type=float,
        default=DEFAULT_TELEOP_NUDGE_STEP,
        help="Keyboard nudge distance in meters for each discrete teleop key press",
    )
    parser.add_argument(
        "--teleop-speed",
        type=float,
        default=DEFAULT_TELEOP_SPEED,
        help="Keyboard hold-to-move speed in meters per second when pynput is installed",
    )
    parser.add_argument(
        "--hold-teleop",
        action="store_true",
        default=DEFAULT_HOLD_TELEOP,
        help="Enable continuous hold-to-move keyboard teleop via pynput",
    )
    parser.add_argument(
        "--actuator-boost",
        type=float,
        default=DEFAULT_ACTUATOR_BOOST,
        help="Interactive arm actuator gain scale; lower values reduce lurching but feel softer",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    from franka_force import FrankaForceEnv

    env = FrankaForceEnv(
        scenario=args.scenario,
        interactive=args.interactive,
        force_feedback=args.force_feedback,
        force_visual=args.force_visual,
        record_video=args.record_video,
        record_force_feedback=args.record_force_feedback,
        results_dir=args.results_dir,
        disable_policy=args.disable_policy,
        free_orientation=args.free_orientation,
        occluded_task=args.occluded_task,
        randomize_occluded_hole=args.randomize_occluded_hole,
        occluded_hole_x_range=args.occluded_hole_x_range,
        occluded_hole_y_range=args.occluded_hole_y_range,
        occluded_hole_seed=args.occluded_hole_seed,
        hole_clearance_mm=args.hole_clearance_mm,
        audio_feedback=args.audio_feedback,
        audio_mode=args.audio_mode,
        audio_contact_threshold=args.audio_contact_threshold,
        audio_lateral_threshold=args.audio_lateral_threshold,
        audio_lateral_max=args.audio_lateral_max,
        audio_volume=args.audio_volume,
        contact_cushion=args.contact_cushion,
        cushion_threshold=args.cushion_threshold,
        impedance_kp=args.impedance_kp,
        impedance_dp=args.impedance_dp,
        impedance_kr=args.impedance_kr,
        impedance_dr=args.impedance_dr,
        impedance_torque_limit=args.impedance_torque_limit,
        peg_alpha=args.peg_alpha,
        socket_alpha=args.socket_alpha,
        occluder_alpha=args.occluder_alpha,
        occluder_style=args.occluder_style,
        teleop_nudge_step=args.teleop_nudge_step,
        teleop_speed=args.teleop_speed,
        hold_teleop=args.hold_teleop,
        actuator_boost=args.actuator_boost,
    )
    env.run()
