"""Launch one experiment trial and persist its execution metadata."""

import csv
import json
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIO = "peg_in_hole"
RAW_LOG_NAME = "force_verification_log.csv"
FILTERED_LOG_NAME = "force_verification_log_filtered.csv"
TRIAL_OUTCOME_NAME = "trial_outcome.json"


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
        metadata["error"] = "Trial interrupted" if interrupted else f"Trial process exited with status {exc.returncode}"
        write_json(metadata_path, metadata)
        if interrupted:
            raise KeyboardInterrupt from exc
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


def build_trial_command(args, trial):
    command = [
        trial_python(args), str(REPO_ROOT / "main.py"), "--scenario", SCENARIO,
        "--interactive", "--occluded-task", "--randomize-occluded-hole",
        "--occluded-hole-seed", str(trial["seed"]),
        "--occluded-hole-x-range", str(args.occluded_hole_x_range[0]), str(args.occluded_hole_x_range[1]),
        "--occluded-hole-y-range", str(args.occluded_hole_y_range[0]), str(args.occluded_hole_y_range[1]),
        "--hole-clearance-mm", str(args.hole_clearance_mm), "--peg-alpha", str(args.peg_alpha),
        "--socket-alpha", str(args.socket_alpha), "--occluder-alpha", str(args.occluder_alpha),
        "--occluder-style", args.occluder_style, "--teleop-nudge-step", str(args.teleop_nudge_step),
        "--teleop-speed", str(args.teleop_speed), "--actuator-boost", str(args.actuator_boost),
        "--results-dir", str(trial["trial_dir"].resolve()),
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


def is_interrupt_returncode(returncode):
    return returncode in (-signal.SIGINT, 128 + signal.SIGINT)


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
    outcome.setdefault("success_hold_time", max_csv_float(rows, "Success Hold Time", default=0.0))
    outcome["occluded_hole_world_pos"] = [first_csv_float(rows, "Occluded Hole X (m)"), first_csv_float(rows, "Occluded Hole Y (m)")]
    outcome["occluded_hole_offset"] = [first_csv_float(rows, "Occluded Hole Offset X (m)"), first_csv_float(rows, "Occluded Hole Offset Y (m)")]
    return outcome


def any_csv_bool(rows, column):
    return any(str(row.get(column, "")).strip().lower() in {"1", "1.0", "true", "yes"} for row in rows)


def max_csv_float(rows, column, default=None):
    values = [value for value in (parse_float(row.get(column, "")) for row in rows) if value is not None]
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
        "status": status, "started_at": utc_now(), "tester": trial["tester"],
        "tester_id": trial["tester_id"], "scenario": SCENARIO,
        "condition": trial["condition"], "condition_order": plan["condition_order"],
        "condition_position": trial["condition_position"], "trial_type": trial["trial_type"],
        "trial_index": trial["trial_index"], "occluded_hole_seed": trial["seed"],
        "visual_feedback": trial["visual_feedback"], "audio_feedback": trial["audio_feedback"],
        "hole_clearance_mm": args.hole_clearance_mm,
        "occluded_hole_x_range": list(args.occluded_hole_x_range),
        "occluded_hole_y_range": list(args.occluded_hole_y_range),
        "occluder_alpha": args.occluder_alpha, "occluder_style": args.occluder_style,
        "teleop_nudge_step": args.teleop_nudge_step, "teleop_speed": args.teleop_speed,
        "hold_teleop": args.hold_teleop, "actuator_boost": args.actuator_boost,
        "max_trial_duration": args.max_trial_duration, "record_video": args.record_video,
        "record_force_feedback": args.record_force_feedback and trial["visual_feedback"],
    }


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(data)
    serializable.pop("loaded_existing_plan", None)
    with path.open("w") as f:
        json.dump(serializable, f, indent=2, sort_keys=True)
        f.write("\n")


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
