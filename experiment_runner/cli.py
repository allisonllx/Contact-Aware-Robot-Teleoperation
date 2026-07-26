"""Command-line parsing and validation for the experiment runner."""

import argparse
from pathlib import Path

from franka_force.config import (
    DEFAULT_ACTUATOR_BOOST,
    DEFAULT_FORCE_THRESHOLD_N,
    DEFAULT_HOLE_CLEARANCE_MM,
    DEFAULT_HOLD_TELEOP,
    DEFAULT_JAMMING_THRESHOLD_N,
    DEFAULT_MAX_TRIAL_DURATION_S,
    DEFAULT_OCCLUDED_HOLE_X_RANGE,
    DEFAULT_OCCLUDED_HOLE_Y_RANGE,
    DEFAULT_PEG_ALPHA,
    DEFAULT_SOCKET_ALPHA,
    DEFAULT_TELEOP_NUDGE_STEP,
    DEFAULT_TELEOP_SPEED,
    OCCLUDER_STYLES,
)

from .planning import CONDITIONS


EXPERIMENT_ROOT = Path("experiment_results")
EXPERIMENT_OCCLUDER_ALPHA = 0.75
EXPERIMENT_OCCLUDER_STYLE = "frosted"


def parse_args():
    """Parse the legacy ``experiment.py`` command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run the occluded peg-in-hole feedback experiment."
    )
    parser.add_argument("--tester", help="Tester name. Prompts if omitted.")
    parser.add_argument(
        "--experiment-root", type=Path, default=EXPERIMENT_ROOT,
        help="Root folder for tester study outputs.",
    )
    parser.add_argument(
        "--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS),
        help="Feedback conditions to run.",
    )
    parser.add_argument(
        "--order", nargs="+", choices=CONDITIONS,
        help="Manual condition order. Must contain the same conditions selected by --conditions.",
    )
    parser.add_argument("--familiarization-trials", type=int, default=1,
                        help="One-time no-feedback familiarization trials before the measured conditions.")
    parser.add_argument("--practice-trials", type=int, default=0,
                        help="Optional practice trials per condition (excluded from the main summary).")
    parser.add_argument("--recorded-trials", type=int, default=3, help="Measured trials per condition.")
    parser.add_argument(
        "--max-trial-duration", type=float, default=DEFAULT_MAX_TRIAL_DURATION_S,
        help="Wall-clock seconds before a trial auto-closes and advances. Use 0 to disable.",
    )
    parser.add_argument("--base-seed", type=int, default=None,
                        help="Optional base seed for deterministic hidden-hole seeds.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Create/reuse the tester plan and print trial commands without launching MuJoCo.")
    parser.add_argument(
        "--trial-python", default=None,
        help="Python launcher for each MuJoCo trial. Defaults to mjpython on macOS when available, otherwise the current Python executable.",
    )
    parser.add_argument("--record-video", action=argparse.BooleanOptionalAction, default=True,
                        help="Record MP4 video for each trial (default: on). Use --no-record-video to disable.")
    parser.add_argument("--record-force-feedback", action="store_true",
                        help="Include force overlay in recorded MP4s for visual-feedback conditions.")
    parser.add_argument("--rerun-existing", action="store_true",
                        help="Run even if a trial folder already contains telemetry.")
    parser.add_argument("--force-threshold", type=float, default=DEFAULT_FORCE_THRESHOLD_N,
                        help="Ground-truth force threshold for experiment analysis.")
    parser.add_argument("--jamming-threshold", type=float, default=DEFAULT_JAMMING_THRESHOLD_N,
                        help="Lateral-force threshold in newtons used to count jamming episodes.")
    parser.add_argument("--hole-clearance-mm", type=float, default=DEFAULT_HOLE_CLEARANCE_MM,
                        help="Total peg/hole clearance in millimeters.")
    parser.add_argument("--occluded-hole-x-range", type=float, nargs=2, metavar=("MIN", "MAX"),
                        default=DEFAULT_OCCLUDED_HOLE_X_RANGE,
                        help="Hidden socket X offset range in meters around the default occluded socket center.")
    parser.add_argument("--occluded-hole-y-range", type=float, nargs=2, metavar=("MIN", "MAX"),
                        default=DEFAULT_OCCLUDED_HOLE_Y_RANGE,
                        help="Hidden socket Y offset range in meters around the default occluded socket center.")
    parser.add_argument("--peg-alpha", type=float, default=DEFAULT_PEG_ALPHA, help="Peg opacity.")
    parser.add_argument("--socket-alpha", type=float, default=DEFAULT_SOCKET_ALPHA, help="Socket wall opacity.")
    parser.add_argument("--occluder-alpha", type=float, default=EXPERIMENT_OCCLUDER_ALPHA,
                        help="Occlusion obstacle opacity, from 0.0 transparent to 1.0 opaque.")
    parser.add_argument("--occluder-style", choices=OCCLUDER_STYLES, default=EXPERIMENT_OCCLUDER_STYLE,
                        help="Occlusion obstacle visual style.")
    parser.add_argument("--teleop-nudge-step", type=float, default=DEFAULT_TELEOP_NUDGE_STEP,
                        help="Keyboard nudge distance in meters for each discrete teleop key press.")
    parser.add_argument("--teleop-speed", type=float, default=DEFAULT_TELEOP_SPEED,
                        help="Keyboard hold-to-move speed in meters per second when pynput is installed.")
    parser.add_argument("--hold-teleop", action="store_true", default=DEFAULT_HOLD_TELEOP,
                        help="Enable continuous hold-to-move keyboard teleop via pynput.")
    parser.add_argument("--actuator-boost", type=float, default=DEFAULT_ACTUATOR_BOOST,
                        help="Interactive arm actuator gain scale; lower values reduce lurching but feel softer.")
    return parser.parse_args()


def validate_args(args):
    """Raise ``ValueError`` when CLI option combinations are invalid."""
    for name in ("familiarization_trials", "practice_trials", "recorded_trials"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if not any((args.familiarization_trials, args.practice_trials, args.recorded_trials)):
        raise ValueError("At least one familiarization, practice, or recorded trial is required")
    if args.max_trial_duration < 0.0:
        raise ValueError("--max-trial-duration must be non-negative (0 disables the limit)")
    if args.record_force_feedback and not args.record_video:
        raise ValueError("--record-force-feedback requires --record-video")
    if len(set(args.conditions)) != len(args.conditions):
        raise ValueError("--conditions cannot contain duplicates")
    if args.order:
        if len(set(args.order)) != len(args.order):
            raise ValueError("--order cannot contain duplicates")
        if set(args.order) != set(args.conditions):
            raise ValueError("--order must contain exactly the same conditions as --conditions")
    validate_range("--occluded-hole-x-range", args.occluded_hole_x_range)
    validate_range("--occluded-hole-y-range", args.occluded_hole_y_range)
    for name in ("force_threshold", "jamming_threshold", "teleop_nudge_step", "teleop_speed", "actuator_boost"):
        if getattr(args, name) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 <= args.occluder_alpha <= 1.0:
        raise ValueError("--occluder-alpha must be between 0.0 and 1.0")


def validate_range(name, values):
    """Validate a two-value inclusive range."""
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    if values[0] > values[1]:
        raise ValueError(f"{name} minimum must be <= maximum")
