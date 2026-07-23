import argparse
import csv
import hashlib
import json
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from analysis import SUMMARY_COLUMNS, analyze_result_dir, csv_value
from franka_force.config import (
    DEFAULT_ACTUATOR_BOOST,
    DEFAULT_HOLE_CLEARANCE_MM,
    DEFAULT_HOLD_TELEOP,
    DEFAULT_MAX_TRIAL_DURATION_S,
    DEFAULT_OCCLUDED_HOLE_X_RANGE,
    DEFAULT_OCCLUDED_HOLE_Y_RANGE,
    DEFAULT_PEG_ALPHA,
    DEFAULT_SOCKET_ALPHA,
    DEFAULT_TELEOP_NUDGE_STEP,
    DEFAULT_TELEOP_SPEED,
    OCCLUDER_STYLES,
)


EXPERIMENT_ROOT = Path("experiment_results")
REPO_ROOT = Path(__file__).resolve().parent
SCENARIO = "peg_in_hole"
CONDITIONS = ("no_feedback", "visual_feedback", "audio_feedback", "both_feedback")
EXPERIMENT_OCCLUDER_ALPHA = 0.8
EXPERIMENT_OCCLUDER_STYLE = "frosted"
TRIAL_METADATA_NAME = "trial_metadata.json"
TRIAL_OUTCOME_NAME = "trial_outcome.json"
RAW_LOG_NAME = "force_verification_log.csv"
FILTERED_LOG_NAME = "force_verification_log_filtered.csv"
WILLIAMS_ORDERS = (
    ("no_feedback", "visual_feedback", "both_feedback", "audio_feedback"),
    ("visual_feedback", "audio_feedback", "no_feedback", "both_feedback"),
    ("audio_feedback", "both_feedback", "visual_feedback", "no_feedback"),
    ("both_feedback", "no_feedback", "audio_feedback", "visual_feedback"),
)

EXPERIMENT_COLUMNS = [
    "tester",
    "tester_id",
    "condition",
    "condition_order",
    "condition_position",
    "trial_type",
    "trial_index",
    "trial_dir",
    "occluded_hole_seed",
    "visual_feedback",
    "audio_feedback",
]
EXPERIMENT_SUMMARY_COLUMNS = EXPERIMENT_COLUMNS + SUMMARY_COLUMNS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the occluded peg-in-hole feedback experiment."
    )
    parser.add_argument("--tester", help="Tester name. Prompts if omitted.")
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=EXPERIMENT_ROOT,
        help="Root folder for tester study outputs.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=list(CONDITIONS),
        help="Feedback conditions to run.",
    )
    parser.add_argument(
        "--order",
        nargs="+",
        choices=CONDITIONS,
        help="Manual condition order. Must contain the same conditions selected by --conditions.",
    )
    parser.add_argument(
        "--familiarization-trials",
        type=int,
        default=1,
        help="One-time no-feedback familiarization trials before the measured conditions.",
    )
    parser.add_argument(
        "--practice-trials",
        type=int,
        default=0,
        help="Optional practice trials per condition (excluded from the main summary).",
    )
    parser.add_argument(
        "--recorded-trials",
        type=int,
        default=3,
        help="Measured trials per condition.",
    )
    parser.add_argument(
        "--max-trial-duration",
        type=float,
        default=DEFAULT_MAX_TRIAL_DURATION_S,
        help="Wall-clock seconds before a trial auto-closes and advances. Use 0 to disable.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=None,
        help="Optional base seed for deterministic hidden-hole seeds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create/reuse the tester plan and print trial commands without launching MuJoCo.",
    )
    parser.add_argument(
        "--trial-python",
        default=None,
        help=(
            "Python launcher for each MuJoCo trial. Defaults to mjpython on macOS "
            "when available, otherwise the current Python executable."
        ),
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record MP4 video for each trial.",
    )
    parser.add_argument(
        "--record-force-feedback",
        action="store_true",
        help="Include force overlay in recorded MP4s for visual-feedback conditions.",
    )
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Run even if a trial folder already contains telemetry.",
    )
    parser.add_argument(
        "--force-threshold",
        type=float,
        default=100.0,
        help="Ground-truth force threshold for experiment analysis.",
    )
    parser.add_argument(
        "--hole-clearance-mm",
        type=float,
        default=DEFAULT_HOLE_CLEARANCE_MM,
        help="Total peg/hole clearance in millimeters.",
    )
    parser.add_argument(
        "--occluded-hole-x-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=DEFAULT_OCCLUDED_HOLE_X_RANGE,
        help="Hidden socket X offset range in meters around the default occluded socket center.",
    )
    parser.add_argument(
        "--occluded-hole-y-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=DEFAULT_OCCLUDED_HOLE_Y_RANGE,
        help="Hidden socket Y offset range in meters around the default occluded socket center.",
    )
    parser.add_argument(
        "--peg-alpha",
        type=float,
        default=DEFAULT_PEG_ALPHA,
        help="Peg opacity.",
    )
    parser.add_argument(
        "--socket-alpha",
        type=float,
        default=DEFAULT_SOCKET_ALPHA,
        help="Socket wall opacity.",
    )
    parser.add_argument(
        "--occluder-alpha",
        type=float,
        default=EXPERIMENT_OCCLUDER_ALPHA,
        help="Occlusion obstacle opacity, from 0.0 transparent to 1.0 opaque.",
    )
    parser.add_argument(
        "--occluder-style",
        choices=OCCLUDER_STYLES,
        default=EXPERIMENT_OCCLUDER_STYLE,
        help="Occlusion obstacle visual style.",
    )
    parser.add_argument(
        "--teleop-nudge-step",
        type=float,
        default=DEFAULT_TELEOP_NUDGE_STEP,
        help="Keyboard nudge distance in meters for each discrete teleop key press.",
    )
    parser.add_argument(
        "--teleop-speed",
        type=float,
        default=DEFAULT_TELEOP_SPEED,
        help="Keyboard hold-to-move speed in meters per second when pynput is installed.",
    )
    parser.add_argument(
        "--hold-teleop",
        action="store_true",
        default=DEFAULT_HOLD_TELEOP,
        help="Enable continuous hold-to-move keyboard teleop via pynput.",
    )
    parser.add_argument(
        "--actuator-boost",
        type=float,
        default=DEFAULT_ACTUATOR_BOOST,
        help="Interactive arm actuator gain scale; lower values reduce lurching but feel softer.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    validate_args(args)

    tester_name = args.tester.strip() if args.tester else input("Tester name: ").strip()
    tester_id = sanitize_tester_name(tester_name)
    tester_dir = args.experiment_root / tester_id
    tester_dir.mkdir(parents=True, exist_ok=True)

    selected_conditions = tuple(dict.fromkeys(args.conditions))
    plan = load_or_create_plan(
        tester_name=tester_name,
        tester_id=tester_id,
        tester_dir=tester_dir,
        experiment_root=args.experiment_root,
        selected_conditions=selected_conditions,
        manual_order=tuple(args.order) if args.order else None,
        familiarization_trials=args.familiarization_trials,
        practice_trials=args.practice_trials,
        recorded_trials=args.recorded_trials,
        max_trial_duration=args.max_trial_duration,
        base_seed=args.base_seed,
    )
    ensure_selected_conditions_in_plan(selected_conditions, plan)

    trial_specs = build_trial_specs(args, tester_name, tester_id, tester_dir, plan, selected_conditions)
    print_experiment_overview(tester_name, tester_dir, plan, trial_specs, args.dry_run, args)

    if args.dry_run:
        print("\nDry run only; no MuJoCo windows launched.")
        return

    interrupted = False
    try:
        for trial in trial_specs:
            state = trial_state(trial["trial_dir"])
            if state["complete"] and not args.rerun_existing:
                print(f"\nSkipping completed trial: {trial['trial_dir']}")
                continue
            print_resume_note(trial, state, args.rerun_existing)
            wait_for_trial(trial, args)
            run_trial(args, plan, trial)
    except KeyboardInterrupt:
        interrupted = True
        print("\nExperiment interrupted. Rerun the same tester command to resume at the first incomplete trial.")
    finally:
        write_experiment_summaries(
            tester_dir=tester_dir,
            tester_name=tester_name,
            tester_id=tester_id,
            plan=plan,
            trial_specs=trial_specs,
            force_threshold=args.force_threshold,
        )

    if interrupted:
        raise SystemExit(130)

    if all(trial_completed(trial["trial_dir"]) for trial in trial_specs):
        zip_path = zip_tester_results(tester_dir)
        print(f"\nExperiment complete. Send this zip file:")
        print(f"  {zip_path}")
    else:
        print("\nExperiment not fully complete yet. Rerun the same tester command to resume.")


def validate_args(args):
    if args.familiarization_trials < 0:
        raise ValueError("--familiarization-trials must be non-negative")
    if args.practice_trials < 0:
        raise ValueError("--practice-trials must be non-negative")
    if args.recorded_trials < 0:
        raise ValueError("--recorded-trials must be non-negative")
    if args.familiarization_trials == 0 and args.practice_trials == 0 and args.recorded_trials == 0:
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
    if args.force_threshold <= 0.0:
        raise ValueError("--force-threshold must be positive")
    if not 0.0 <= args.occluder_alpha <= 1.0:
        raise ValueError("--occluder-alpha must be between 0.0 and 1.0")
    if args.teleop_nudge_step <= 0.0:
        raise ValueError("--teleop-nudge-step must be positive")
    if args.teleop_speed <= 0.0:
        raise ValueError("--teleop-speed must be positive")
    if args.actuator_boost <= 0.0:
        raise ValueError("--actuator-boost must be positive")


def validate_range(name, values):
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    if values[0] > values[1]:
        raise ValueError(f"{name} minimum must be <= maximum")


def ensure_selected_conditions_in_plan(selected_conditions, plan):
    assigned = set(plan.get("condition_order", []))
    missing = [condition for condition in selected_conditions if condition not in assigned]
    if missing:
        raise ValueError(
            "Selected conditions are not present in this tester's saved plan: "
            + ", ".join(missing)
        )


def sanitize_tester_name(name):
    if not name:
        raise ValueError("Tester name cannot be empty")
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("._-").lower()
    if not cleaned:
        raise ValueError("Tester name must contain at least one letter or number")
    return cleaned


def load_or_create_plan(
    tester_name,
    tester_id,
    tester_dir,
    experiment_root,
    selected_conditions,
    manual_order,
    familiarization_trials,
    practice_trials,
    recorded_trials,
    max_trial_duration,
    base_seed,
):
    plan_path = tester_dir / "experiment_plan.json"
    if plan_path.exists() and manual_order is None:
        with plan_path.open() as f:
            plan = json.load(f)
        if plan.get("base_seed") is None and plan.get("generated_seed_base") is None:
            plan["generated_seed_base"] = secrets.randbits(32)
            plan["updated_at"] = utc_now()
            write_json(plan_path, plan)
        plan["loaded_existing_plan"] = True
        return plan

    if manual_order is not None:
        condition_order = list(manual_order)
        order_index = "manual"
    else:
        condition_order, order_index = assign_counterbalanced_order(
            experiment_root,
            tester_id,
            selected_conditions,
        )

    now = utc_now()
    generated_seed_base = secrets.randbits(32) if base_seed is None else None
    plan = {
        "tester": tester_name,
        "tester_id": tester_id,
        "scenario": SCENARIO,
        "conditions": list(selected_conditions),
        "condition_order": condition_order,
        "order_index": order_index,
        "familiarization_trials": familiarization_trials,
        "practice_trials": practice_trials,
        "recorded_trials": recorded_trials,
        "max_trial_duration": max_trial_duration,
        "base_seed": base_seed,
        "generated_seed_base": generated_seed_base,
        "created_at": now,
        "updated_at": now,
        "loaded_existing_plan": False,
    }
    write_json(plan_path, plan)
    return plan


def assign_counterbalanced_order(experiment_root, tester_id, selected_conditions):
    orders = candidate_orders(selected_conditions)
    counts = [0 for _ in orders]
    if experiment_root.exists():
        for plan_path in experiment_root.glob("*/experiment_plan.json"):
            try:
                with plan_path.open() as f:
                    plan = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            existing_order = tuple(plan.get("condition_order", []))
            for i, order in enumerate(orders):
                if existing_order == tuple(order):
                    counts[i] += 1

    min_count = min(counts)
    candidates = [i for i, count in enumerate(counts) if count == min_count]
    choice = candidates[stable_int(tester_id) % len(candidates)]
    return list(orders[choice]), choice


def candidate_orders(selected_conditions):
    selected_conditions = tuple(selected_conditions)
    if selected_conditions == CONDITIONS:
        return WILLIAMS_ORDERS
    return tuple(
        selected_conditions[i:] + selected_conditions[:i]
        for i in range(len(selected_conditions))
    )


def stable_int(text):
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def build_trial_specs(args, tester_name, tester_id, tester_dir, plan, selected_conditions):
    run_order = [condition for condition in plan["condition_order"] if condition in selected_conditions]
    if not run_order:
        raise ValueError("No selected conditions appear in the assigned experiment order")
    seed_base = plan["base_seed"] if plan.get("base_seed") is not None else plan["generated_seed_base"]
    specs = []

    for trial_index in range(1, args.familiarization_trials + 1):
        trial_name = f"familiarization_{trial_index:02d}"
        trial_dir = tester_dir / "familiarization" / trial_name
        seed = trial_seed(seed_base, tester_id, "familiarization", "familiarization", trial_index)
        specs.append({
            "tester": tester_name,
            "tester_id": tester_id,
            "condition": "no_feedback",
            "condition_position": 0,
            "trial_type": "familiarization",
            "trial_index": trial_index,
            "trial_name": trial_name,
            "trial_dir": trial_dir,
            "seed": seed,
            "visual_feedback": False,
            "audio_feedback": False,
        })

    for condition_position, condition in enumerate(run_order, start=1):
        for trial_type, trial_count in (
            ("practice", args.practice_trials),
            ("recorded", args.recorded_trials),
        ):
            for trial_index in range(1, trial_count + 1):
                trial_name = f"{trial_type}_{trial_index:02d}"
                trial_dir = tester_dir / condition / trial_name
                seed = trial_seed(seed_base, tester_id, condition, trial_type, trial_index)
                specs.append({
                    "tester": tester_name,
                    "tester_id": tester_id,
                    "condition": condition,
                    "condition_position": condition_position,
                    "trial_type": trial_type,
                    "trial_index": trial_index,
                    "trial_name": trial_name,
                    "trial_dir": trial_dir,
                    "seed": seed,
                    "visual_feedback": condition in ("visual_feedback", "both_feedback"),
                    "audio_feedback": condition in ("audio_feedback", "both_feedback"),
                })
    return specs


def trial_seed(seed_base, tester_id, condition, trial_type, trial_index):
    seed_key = f"{seed_base}:{tester_id}:{condition}:{trial_type}:{trial_index}"
    return stable_int(seed_key) % (2 ** 32)


def print_experiment_overview(tester_name, tester_dir, plan, trial_specs, dry_run, args):
    mode = "DRY RUN" if dry_run else "LIVE RUN"
    print(f"\n=== {mode}: OCCLUDED PEG-IN-HOLE EXPERIMENT ===")
    print(f"Tester: {tester_name}")
    print(f"Output folder: {tester_dir.resolve()}")
    print(f"Condition order: {' -> '.join(plan['condition_order'])}")
    print(
        f"Structure: {args.familiarization_trials} familiarization "
        f"(no feedback), then {args.recorded_trials} recorded trials per condition"
        + (f", plus {args.practice_trials} practice per condition" if args.practice_trials else "")
        + "."
    )
    if args.max_trial_duration > 0.0:
        print(f"Trial time limit: {args.max_trial_duration:.0f}s wall clock.")
    else:
        print("Trial time limit: disabled.")
    if plan.get("loaded_existing_plan"):
        print("Using existing experiment_plan.json.")
    print_progress_summary(trial_specs)
    print()
    for trial in trial_specs:
        flags = condition_flags_for_display(trial)
        state = trial_state(trial["trial_dir"])
        label = (
            f"familiarization / {trial['trial_name']}"
            if trial["trial_type"] == "familiarization"
            else f"{trial['condition_position']}. {trial['condition']} / {trial['trial_name']}"
        )
        print(
            f"{label} / seed={trial['seed']} / {flags} / "
            f"{trial_status_for_display(state)}"
        )
        print(f"   {trial['trial_dir']}")
        if dry_run:
            print(f"   {shlex.join(build_trial_command(args, trial))}")


def condition_flags_for_display(trial):
    if trial["trial_type"] == "familiarization":
        return "no feedback (familiarization)"
    flags = []
    if trial["visual_feedback"]:
        flags.append("visual")
    if trial["audio_feedback"]:
        flags.append("audio")
    return "+".join(flags) if flags else "no feedback"


def print_progress_summary(trial_specs):
    states = [trial_state(trial["trial_dir"]) for trial in trial_specs]
    completed = sum(1 for state in states if state["complete"])
    interrupted = sum(1 for state in states if state["status"] == "interrupted")
    failed = sum(1 for state in states if state["status"] == "failed")
    started = sum(1 for state in states if state["status"] == "started")
    total = len(states)
    print(f"Progress: {completed}/{total} completed.")
    if interrupted or failed or started:
        details = []
        if interrupted:
            details.append(f"{interrupted} interrupted")
        if failed:
            details.append(f"{failed} failed")
        if started:
            details.append(f"{started} started/incomplete")
        print("Resume state: " + ", ".join(details) + ".")


def print_resume_note(trial, state, rerun_existing):
    if state["complete"] and rerun_existing:
        print(f"\nRerunning completed trial because --rerun-existing is set: {trial['trial_dir']}")
    elif state["status"] != "not_started" or state["has_telemetry"]:
        print(
            f"\nResuming at incomplete trial: {trial['trial_dir']} "
            f"({trial_status_for_display(state)})."
        )
        print("Existing partial outputs in this trial folder will be overwritten by the rerun.")


def trial_completed(trial_dir):
    return trial_state(trial_dir)["complete"]


def trial_state(trial_dir):
    trial_dir = Path(trial_dir)
    metadata = read_json_if_exists(trial_dir / TRIAL_METADATA_NAME)
    status = metadata.get("status") if isinstance(metadata, dict) else None
    has_raw_csv = (trial_dir / RAW_LOG_NAME).exists()
    has_filtered_csv = (trial_dir / FILTERED_LOG_NAME).exists()
    has_telemetry = has_raw_csv or has_filtered_csv
    if not status:
        status = "not_started"
    return {
        "status": status,
        "has_metadata": metadata is not None,
        "has_raw_csv": has_raw_csv,
        "has_filtered_csv": has_filtered_csv,
        "has_telemetry": has_telemetry,
        "complete": status == "completed" and has_telemetry,
    }


def trial_status_for_display(state):
    if state["complete"]:
        return "status=completed"
    status = state["status"]
    if status == "not_started" and state["has_telemetry"]:
        return "status=partial/no metadata"
    if status == "completed":
        return "status=completed/missing telemetry"
    if state["has_telemetry"]:
        return f"status={status}/partial telemetry"
    return f"status={status}"


def wait_for_trial(trial, args):
    print("\n" + "-" * 72)
    if trial["trial_type"] == "familiarization":
        print(f"Next trial: familiarization / {trial['trial_name']} (no feedback)")
        print("This is a one-time practice run to learn the controls.")
    else:
        print(f"Next trial: {trial['condition']} / {trial['trial_name']}")
    print(f"Output: {trial['trial_dir']}")
    print(f"Feedback: {condition_flags_for_display(trial)}")
    if args.max_trial_duration > 0.0:
        print(
            f"Time limit: {args.max_trial_duration:.0f}s. "
            "The MuJoCo window closes automatically on success or timeout."
        )
    else:
        print("No time limit. The MuJoCo window closes automatically on success.")
    print("Press Enter when the tester is ready.")
    input()


def run_trial(args, plan, trial):
    trial_dir = trial["trial_dir"]
    trial_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = trial_dir / "trial_metadata.json"
    metadata = trial_metadata(args, plan, trial, status="started")
    metadata["command"] = build_trial_command(args, trial)
    write_json(metadata_path, metadata)

    try:
        subprocess.run(metadata["command"], cwd=REPO_ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        interrupted = is_interrupt_returncode(exc.returncode)
        metadata["status"] = "interrupted" if interrupted else "failed"
        metadata["ended_at"] = utc_now()
        metadata["returncode"] = exc.returncode
        metadata["error"] = (
            "Trial interrupted"
            if interrupted
            else f"Trial process exited with status {exc.returncode}"
        )
        write_json(metadata_path, metadata)
        if interrupted:
            raise KeyboardInterrupt
        raise
    except KeyboardInterrupt:
        metadata["status"] = "interrupted"
        metadata["ended_at"] = utc_now()
        metadata["error"] = "Trial interrupted"
        write_json(metadata_path, metadata)
        raise
    except OSError as exc:
        metadata["status"] = "failed"
        metadata["ended_at"] = utc_now()
        metadata["error"] = repr(exc)
        write_json(metadata_path, metadata)
        raise

    metadata["status"] = "completed"
    metadata["ended_at"] = utc_now()
    metadata.update(read_trial_outcome(trial_dir))
    write_json(metadata_path, metadata)


def is_interrupt_returncode(returncode):
    return returncode in (-signal.SIGINT, 128 + signal.SIGINT)


def build_trial_command(args, trial):
    command = [
        trial_python(args),
        str(REPO_ROOT / "main.py"),
        "--scenario",
        SCENARIO,
        "--interactive",
        "--occluded-task",
        "--randomize-occluded-hole",
        "--occluded-hole-seed",
        str(trial["seed"]),
        "--occluded-hole-x-range",
        str(args.occluded_hole_x_range[0]),
        str(args.occluded_hole_x_range[1]),
        "--occluded-hole-y-range",
        str(args.occluded_hole_y_range[0]),
        str(args.occluded_hole_y_range[1]),
        "--hole-clearance-mm",
        str(args.hole_clearance_mm),
        "--peg-alpha",
        str(args.peg_alpha),
        "--socket-alpha",
        str(args.socket_alpha),
        "--occluder-alpha",
        str(args.occluder_alpha),
        "--occluder-style",
        args.occluder_style,
        "--teleop-nudge-step",
        str(args.teleop_nudge_step),
        "--teleop-speed",
        str(args.teleop_speed),
        "--actuator-boost",
        str(args.actuator_boost),
        "--results-dir",
        str(trial["trial_dir"].resolve()),
    ]
    if args.max_trial_duration > 0.0:
        command.extend(["--max-trial-duration", str(args.max_trial_duration)])
    if args.hold_teleop:
        command.append("--hold-teleop")
    if trial["visual_feedback"]:
        command.extend(["--force-feedback", "--force-visual", "both"])
    if trial["audio_feedback"]:
        command.extend(["--audio-feedback", "--audio-mode", "both"])
    if args.record_video:
        command.append("--record-video")
    if args.record_force_feedback and trial["visual_feedback"]:
        command.append("--record-force-feedback")
    return command


def trial_python(args):
    if args.trial_python:
        return args.trial_python
    if sys.platform == "darwin":
        return shutil.which("mjpython") or sys.executable
    return sys.executable


def read_trial_outcome(trial_dir):
    outcome = {}
    outcome_path = trial_dir / TRIAL_OUTCOME_NAME
    if outcome_path.exists():
        try:
            with outcome_path.open() as f:
                saved = json.load(f)
            if isinstance(saved, dict):
                outcome.update({
                    "task_success": bool(saved.get("task_success", False)),
                    "timed_out": bool(saved.get("timed_out", False)),
                    "wall_time_elapsed_s": saved.get("wall_time_elapsed_s"),
                    "sim_time_s": saved.get("sim_time_s"),
                    "success_hold_time": saved.get("success_hold_time"),
                })
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    log_path = trial_dir / FILTERED_LOG_NAME
    if not log_path.exists():
        log_path = trial_dir / RAW_LOG_NAME
    if not log_path.exists():
        return outcome

    with log_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return outcome

    outcome.setdefault("task_success", any_csv_bool(rows, "Task Success"))
    outcome.setdefault(
        "success_hold_time",
        max_csv_float(rows, "Success Hold Time", default=0.0),
    )
    outcome["occluded_hole_world_pos"] = [
        first_csv_float(rows, "Occluded Hole X (m)"),
        first_csv_float(rows, "Occluded Hole Y (m)"),
    ]
    outcome["occluded_hole_offset"] = [
        first_csv_float(rows, "Occluded Hole Offset X (m)"),
        first_csv_float(rows, "Occluded Hole Offset Y (m)"),
    ]
    return outcome


def any_csv_bool(rows, column):
    return any(str(row.get(column, "")).strip().lower() in {"1", "1.0", "true", "yes"} for row in rows)


def max_csv_float(rows, column, default=None):
    values = [
        value
        for value in (parse_float(row.get(column, "")) for row in rows)
        if value is not None
    ]
    return max(values) if values else default


def first_csv_float(rows, column):
    for row in rows:
        value = parse_float(row.get(column, ""))
        if value is not None:
            return value
    return None


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def trial_metadata(args, plan, trial, status):
    return {
        "status": status,
        "started_at": utc_now(),
        "tester": trial["tester"],
        "tester_id": trial["tester_id"],
        "scenario": SCENARIO,
        "condition": trial["condition"],
        "condition_order": plan["condition_order"],
        "condition_position": trial["condition_position"],
        "trial_type": trial["trial_type"],
        "trial_index": trial["trial_index"],
        "occluded_hole_seed": trial["seed"],
        "visual_feedback": trial["visual_feedback"],
        "audio_feedback": trial["audio_feedback"],
        "hole_clearance_mm": args.hole_clearance_mm,
        "occluded_hole_x_range": list(args.occluded_hole_x_range),
        "occluded_hole_y_range": list(args.occluded_hole_y_range),
        "occluder_alpha": args.occluder_alpha,
        "occluder_style": args.occluder_style,
        "teleop_nudge_step": args.teleop_nudge_step,
        "teleop_speed": args.teleop_speed,
        "hold_teleop": args.hold_teleop,
        "actuator_boost": args.actuator_boost,
        "max_trial_duration": args.max_trial_duration,
        "record_video": args.record_video,
        "record_force_feedback": args.record_force_feedback and trial["visual_feedback"],
    }


def write_experiment_summaries(tester_dir, tester_name, tester_id, plan, trial_specs, force_threshold):
    recorded_rows = []
    practice_rows = []
    familiarization_rows = []
    for trial in trial_specs:
        if not trial_completed(trial["trial_dir"]):
            continue
        row = experiment_analysis_row(tester_name, tester_id, plan, trial, force_threshold)
        if trial["trial_type"] == "recorded":
            recorded_rows.append(row)
        elif trial["trial_type"] == "familiarization":
            familiarization_rows.append(row)
        else:
            practice_rows.append(row)

    recorded_path = tester_dir / "experiment_analysis_summary.csv"
    practice_path = tester_dir / "practice_analysis_summary.csv"
    familiarization_path = tester_dir / "familiarization_analysis_summary.csv"
    write_experiment_summary(recorded_path, recorded_rows)
    write_experiment_summary(practice_path, practice_rows)
    write_experiment_summary(familiarization_path, familiarization_rows)
    print(f"\nSaved recorded-trial analysis to {recorded_path.resolve()}")
    print(f"Saved practice-trial analysis to {practice_path.resolve()}")
    print(f"Saved familiarization-trial analysis to {familiarization_path.resolve()}")


def experiment_analysis_row(tester_name, tester_id, plan, trial, force_threshold):
    metrics = analyze_result_dir(
        trial["trial_dir"],
        scenario=SCENARIO,
        source="auto",
        force_threshold=force_threshold,
        include_anomalies=False,
    )
    row = {
        "tester": tester_name,
        "tester_id": tester_id,
        "condition": trial["condition"],
        "condition_order": " -> ".join(plan["condition_order"]),
        "condition_position": trial["condition_position"],
        "trial_type": trial["trial_type"],
        "trial_index": trial["trial_index"],
        "trial_dir": str(trial["trial_dir"]),
        "occluded_hole_seed": trial["seed"],
        "visual_feedback": int(trial["visual_feedback"]),
        "audio_feedback": int(trial["audio_feedback"]),
    }
    row.update(metrics)
    return row


def write_experiment_summary(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENT_SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: csv_value(row.get(key, ""))
                for key in EXPERIMENT_SUMMARY_COLUMNS
            })


def zip_tester_results(tester_dir):
    tester_dir = Path(tester_dir).resolve()
    archive_base = tester_dir.parent / tester_dir.name
    zip_path = Path(
        shutil.make_archive(str(archive_base), "zip", root_dir=tester_dir.parent, base_dir=tester_dir.name)
    )
    return zip_path.resolve()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(data)
    serializable.pop("loaded_existing_plan", None)
    with path.open("w") as f:
        json.dump(serializable, f, indent=2, sort_keys=True)
        f.write("\n")


def read_json_if_exists(path):
    if not path.exists():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"status": "metadata_unreadable"}


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    main()
