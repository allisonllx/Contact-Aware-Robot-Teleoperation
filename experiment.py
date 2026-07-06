import argparse
import csv
import hashlib
import json
import re
import secrets
from datetime import datetime
from pathlib import Path

from analysis import SUMMARY_COLUMNS, analyze_result_dir, csv_value
from franka_force.config import (
    DEFAULT_HOLE_CLEARANCE_MM,
    DEFAULT_OCCLUDED_HOLE_X_RANGE,
    DEFAULT_OCCLUDED_HOLE_Y_RANGE,
    DEFAULT_PEG_ALPHA,
    DEFAULT_SOCKET_ALPHA,
)


EXPERIMENT_ROOT = Path("experiment_results")
SCENARIO = "peg_in_hole"
CONDITIONS = ("no_feedback", "visual_feedback", "audio_feedback", "both_feedback")
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
        "--practice-trials",
        type=int,
        default=2,
        help="Practice trials per condition.",
    )
    parser.add_argument(
        "--recorded-trials",
        type=int,
        default=1,
        help="Recorded trials per condition.",
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
        practice_trials=args.practice_trials,
        recorded_trials=args.recorded_trials,
        base_seed=args.base_seed,
    )
    ensure_selected_conditions_in_plan(selected_conditions, plan)

    trial_specs = build_trial_specs(args, tester_name, tester_id, tester_dir, plan, selected_conditions)
    print_experiment_overview(tester_name, tester_dir, plan, trial_specs, args.dry_run)

    if args.dry_run:
        print("\nDry run only; no MuJoCo windows launched.")
        return

    for trial in trial_specs:
        if trial_completed(trial["trial_dir"]) and not args.rerun_existing:
            print(f"\nSkipping completed trial: {trial['trial_dir']}")
            continue
        wait_for_trial(trial)
        run_trial(args, plan, trial)

    write_experiment_summaries(
        tester_dir=tester_dir,
        tester_name=tester_name,
        tester_id=tester_id,
        plan=plan,
        trial_specs=trial_specs,
        force_threshold=args.force_threshold,
    )


def validate_args(args):
    if args.practice_trials < 0:
        raise ValueError("--practice-trials must be non-negative")
    if args.recorded_trials < 0:
        raise ValueError("--recorded-trials must be non-negative")
    if args.practice_trials == 0 and args.recorded_trials == 0:
        raise ValueError("At least one practice or recorded trial is required")
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
    practice_trials,
    recorded_trials,
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
        "practice_trials": practice_trials,
        "recorded_trials": recorded_trials,
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


def print_experiment_overview(tester_name, tester_dir, plan, trial_specs, dry_run):
    mode = "DRY RUN" if dry_run else "LIVE RUN"
    print(f"\n=== {mode}: OCCLUDED PEG-IN-HOLE EXPERIMENT ===")
    print(f"Tester: {tester_name}")
    print(f"Output folder: {tester_dir.resolve()}")
    print(f"Condition order: {' -> '.join(plan['condition_order'])}")
    if plan.get("loaded_existing_plan"):
        print("Using existing experiment_plan.json.")
    print()
    for trial in trial_specs:
        flags = condition_flags_for_display(trial)
        print(
            f"{trial['condition_position']}. {trial['condition']} / "
            f"{trial['trial_name']} / seed={trial['seed']} / {flags}"
        )
        print(f"   {trial['trial_dir']}")


def condition_flags_for_display(trial):
    flags = []
    if trial["visual_feedback"]:
        flags.append("visual")
    if trial["audio_feedback"]:
        flags.append("audio")
    return "+".join(flags) if flags else "no feedback"


def trial_completed(trial_dir):
    return (trial_dir / "force_verification_log.csv").exists()


def wait_for_trial(trial):
    print("\n" + "-" * 72)
    print(f"Next trial: {trial['condition']} / {trial['trial_name']}")
    print(f"Output: {trial['trial_dir']}")
    print(f"Feedback: {condition_flags_for_display(trial)}")
    print("Press Enter when the tester is ready. Close the MuJoCo window after the trial.")
    input()


def run_trial(args, plan, trial):
    from franka_force import FrankaForceEnv

    trial_dir = trial["trial_dir"]
    trial_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = trial_dir / "trial_metadata.json"
    metadata = trial_metadata(args, plan, trial, status="started")
    write_json(metadata_path, metadata)

    try:
        env = FrankaForceEnv(
            scenario=SCENARIO,
            interactive=True,
            occluded_task=True,
            randomize_occluded_hole=True,
            occluded_hole_seed=trial["seed"],
            occluded_hole_x_range=args.occluded_hole_x_range,
            occluded_hole_y_range=args.occluded_hole_y_range,
            hole_clearance_mm=args.hole_clearance_mm,
            force_feedback=trial["visual_feedback"],
            force_visual="both",
            audio_feedback=trial["audio_feedback"],
            audio_mode="both",
            record_video=args.record_video,
            record_force_feedback=args.record_force_feedback and trial["visual_feedback"],
            peg_alpha=args.peg_alpha,
            socket_alpha=args.socket_alpha,
            results_dir=trial_dir,
        )
        env.run()
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["ended_at"] = utc_now()
        metadata["error"] = repr(exc)
        write_json(metadata_path, metadata)
        raise

    metadata["status"] = "completed"
    metadata["ended_at"] = utc_now()
    metadata["task_success"] = bool(getattr(env, "task_success", False))
    metadata["success_hold_time"] = float(getattr(env, "success_hold_time", 0.0))
    metadata["occluded_hole_world_pos"] = list(map(float, env.occluded_hole_world_pos[:2]))
    metadata["occluded_hole_offset"] = list(map(float, env.occluded_hole_offset[:2]))
    write_json(metadata_path, metadata)


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
        "record_video": args.record_video,
        "record_force_feedback": args.record_force_feedback and trial["visual_feedback"],
    }


def write_experiment_summaries(tester_dir, tester_name, tester_id, plan, trial_specs, force_threshold):
    recorded_rows = []
    practice_rows = []
    for trial in trial_specs:
        if not trial_completed(trial["trial_dir"]):
            continue
        row = experiment_analysis_row(tester_name, tester_id, plan, trial, force_threshold)
        if trial["trial_type"] == "recorded":
            recorded_rows.append(row)
        else:
            practice_rows.append(row)

    recorded_path = tester_dir / "experiment_analysis_summary.csv"
    practice_path = tester_dir / "practice_analysis_summary.csv"
    write_experiment_summary(recorded_path, recorded_rows)
    write_experiment_summary(practice_path, practice_rows)
    print(f"\nSaved recorded-trial analysis to {recorded_path.resolve()}")
    print(f"Saved practice-trial analysis to {practice_path.resolve()}")


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


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(data)
    serializable.pop("loaded_existing_plan", None)
    with path.open("w") as f:
        json.dump(serializable, f, indent=2, sort_keys=True)
        f.write("\n")


def utc_now():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


if __name__ == "__main__":
    main()
