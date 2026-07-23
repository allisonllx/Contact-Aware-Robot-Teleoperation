import argparse
import csv
import json
import math
from pathlib import Path

from franka_force.config import (
    DEFAULT_FORCE_THRESHOLD_N,
    DEFAULT_JAMMING_THRESHOLD_N,
    RESULTS_DIR,
    SCENARIOS,
)


RAW_LOG_NAME = "force_verification_log.csv"
FILTERED_LOG_NAME = "force_verification_log_filtered.csv"
TRIAL_OUTCOME_NAME = "trial_outcome.json"
TRIAL_METADATA_NAME = "trial_metadata.json"
EPS = 1e-9

SUMMARY_COLUMNS = [
    "scenario",
    "status",
    "source_csv",
    "used_filtered_csv",
    "force_threshold_n",
    "jamming_threshold_n",
    "samples_total",
    "samples_clean",
    "samples_contact",
    "samples_contact_clean",
    "duration_s",
    "wall_time_elapsed_s",
    "timed_out",
    "contact_sample_fraction",
    "first_contact_time_s",
    "contact_duration_s",
    "task_success",
    "completion_time_s",
    "completion_time_wall_s",
    "hole_clearance_mm",
    "occluded_hole_randomized",
    "occluded_hole_x_m",
    "occluded_hole_y_m",
    "occluded_hole_offset_x_m",
    "occluded_hole_offset_y_m",
    "occluder_alpha",
    "occluder_style",
    "audio_feedback_enabled",
    "cushion_used",
    "mean_ground_truth_contact_n",
    "mean_estimate_contact_n",
    "peak_ground_truth_contact_n",
    "peak_estimate_contact_n",
    "peak_contact_proxy_n",
    "peak_lateral_force_n",
    "time_above_threshold_s",
    "time_above_jamming_s",
    "jamming_count",
    "contact_episode_count",
    "contact_force_impulse_n_s",
    "mean_action_jerk",
    "velocity_reversals",
    "retraction_count",
    "mae_contact_n",
    "mse_contact_n2",
    "rmse_contact_n",
    "bias_contact_n",
    "median_abs_error_contact_n",
    "p95_abs_error_contact_n",
    "max_abs_error_contact_n",
    "nmae_contact_mean_gt",
    "nrmse_contact_mean_gt",
    "mae_all_clean_n",
    "mse_all_clean_n2",
    "rmse_all_clean_n",
    "bias_all_clean_n",
    "p95_abs_error_all_clean_n",
]

CONDITION_SUMMARY_COLUMNS = [
    "tester",
    "tester_id",
    "condition",
    "n_trials",
    "success_rate",
    "mean_completion_time_wall_s",
    "mean_completion_time_sim_s",
    "mean_peak_contact_proxy_n",
    "mean_peak_ground_truth_contact_n",
    "mean_peak_lateral_force_n",
    "mean_time_above_threshold_s",
    "mean_jamming_count",
    "mean_contact_episode_count",
    "mean_action_jerk",
    "mean_velocity_reversals",
    "mean_retraction_count",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze force-estimation accuracy and task safety metrics."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory containing results/<scenario>/ telemetry folders.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="Analyze a tester folder under experiment_results/ and write condition summaries.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=SCENARIOS,
        help="Scenario names to analyze.",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "filtered", "raw"),
        default="auto",
        help="Use filtered logs, raw logs, or prefer filtered logs when available.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Summary CSV path. Defaults to <results-dir>/force_analysis_summary.csv.",
    )
    parser.add_argument(
        "--force-threshold",
        type=float,
        default=DEFAULT_FORCE_THRESHOLD_N,
        help="Ground-truth force threshold for safety-duration metrics.",
    )
    parser.add_argument(
        "--jamming-threshold",
        type=float,
        default=DEFAULT_JAMMING_THRESHOLD_N,
        help="Lateral-force threshold in newtons used to count jamming episodes.",
    )
    parser.add_argument(
        "--include-anomalies",
        action="store_true",
        help="Include rows flagged as anomalies when reading raw logs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.force_threshold <= 0.0:
        raise ValueError("--force-threshold must be positive")
    if args.jamming_threshold <= 0.0:
        raise ValueError("--jamming-threshold must be positive")

    if args.experiment_dir is not None:
        summarize_experiment_dir(
            args.experiment_dir,
            source=args.source,
            force_threshold=args.force_threshold,
            jamming_threshold=args.jamming_threshold,
            include_anomalies=args.include_anomalies,
        )
        return

    output_path = args.output or args.results_dir / "force_analysis_summary.csv"
    rows = [
        analyze_scenario(
            scenario=scenario,
            results_dir=args.results_dir,
            source=args.source,
            force_threshold=args.force_threshold,
            jamming_threshold=args.jamming_threshold,
            include_anomalies=args.include_anomalies,
        )
        for scenario in args.scenarios
    ]

    write_summary(output_path, rows)
    print_summary(rows, output_path)


def analyze_scenario(
    scenario,
    results_dir,
    source,
    force_threshold,
    include_anomalies,
    jamming_threshold=DEFAULT_JAMMING_THRESHOLD_N,
):
    log_path = select_log_path(results_dir, scenario, source)
    if log_path is None:
        return empty_summary(
            scenario,
            status="missing_csv",
            force_threshold=force_threshold,
            jamming_threshold=jamming_threshold,
        )

    return analyze_log_file(
        log_path=log_path,
        scenario=scenario,
        force_threshold=force_threshold,
        jamming_threshold=jamming_threshold,
        include_anomalies=include_anomalies,
    )


def analyze_result_dir(
    result_dir,
    scenario,
    source="auto",
    force_threshold=DEFAULT_FORCE_THRESHOLD_N,
    jamming_threshold=DEFAULT_JAMMING_THRESHOLD_N,
    include_anomalies=False,
):
    result_dir = Path(result_dir)
    log_path = select_log_path_from_dir(result_dir, source)
    if log_path is None:
        summary = empty_summary(
            scenario,
            status="missing_csv",
            force_threshold=force_threshold,
            jamming_threshold=jamming_threshold,
        )
    else:
        summary = analyze_log_file(
            log_path=log_path,
            scenario=scenario,
            force_threshold=force_threshold,
            jamming_threshold=jamming_threshold,
            include_anomalies=include_anomalies,
        )
    summary.update(read_trial_outcome_metrics(result_dir))
    if summary.get("completion_time_wall_s") in ("", None):
        summary["completion_time_wall_s"] = summary.get("wall_time_elapsed_s", "")
    return summary


def analyze_log_file(
    log_path,
    scenario,
    force_threshold,
    include_anomalies,
    jamming_threshold=DEFAULT_JAMMING_THRESHOLD_N,
):
    with log_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    summary = empty_summary(
        scenario,
        status="ok",
        force_threshold=force_threshold,
        jamming_threshold=jamming_threshold,
        source_csv=log_path,
        used_filtered_csv=log_path.name == FILTERED_LOG_NAME,
    )

    if not rows:
        summary["status"] = "no_samples"
        return summary

    columns = set(rows[0].keys())
    times = float_column(rows, "Time (s)")
    f_true = float_column(rows, "Ground Truth (N)")
    f_est = float_column(rows, "Jacobian Estimate (N)")
    if "In Contact" in columns:
        in_contact = bool_column(rows, "In Contact")
    else:
        in_contact = [value > EPS for value in f_true]

    is_anomaly = [False] * len(rows)
    if not include_anomalies and "Is Anomaly" in columns:
        is_anomaly = bool_column(rows, "Is Anomaly")
    is_clean = [not value for value in is_anomaly]
    contact_clean = and_masks(in_contact, is_clean)

    dt = sample_widths(times)
    task_success = optional_bool_column(rows, columns, "Task Success")
    audio_feedback = optional_bool_column(rows, columns, "Audio Feedback")
    cushion_active = optional_bool_column(rows, columns, "Cushion Active")
    hole_clearance = optional_float_column(rows, columns, "Hole Clearance (mm)", math.nan)
    occluded_hole_randomized = optional_bool_column(rows, columns, "Occluded Hole Randomized")
    occluded_hole_x = optional_float_column(rows, columns, "Occluded Hole X (m)", math.nan)
    occluded_hole_y = optional_float_column(rows, columns, "Occluded Hole Y (m)", math.nan)
    occluded_hole_offset_x = optional_float_column(rows, columns, "Occluded Hole Offset X (m)", math.nan)
    occluded_hole_offset_y = optional_float_column(rows, columns, "Occluded Hole Offset Y (m)", math.nan)
    occluder_alpha = optional_float_column(rows, columns, "Occluder Alpha", math.nan)
    occluder_style = optional_text_column(rows, columns, "Occluder Style")

    fx = optional_float_column(rows, columns, "Contact Force X (N)", 0.0)
    fy = optional_float_column(rows, columns, "Contact Force Y (N)", 0.0)
    # Always derive lateral from contact Fx/Fy. Audio Lateral Force is 0 when
    # audio feedback is off, so it must not be preferred for study metrics.
    lateral = [math.hypot(x, y) for x, y in zip(fx, fy)]
    jamming_mask = [
        clean and value >= jamming_threshold
        for clean, value in zip(is_clean, lateral)
    ]
    smoothness = smoothness_metrics(
        times=times,
        is_clean=is_clean,
        in_contact=in_contact,
        target_x=optional_float_column(rows, columns, "Target X (m)", math.nan),
        target_y=optional_float_column(rows, columns, "Target Y (m)", math.nan),
        target_z=optional_float_column(rows, columns, "Target Z (m)", math.nan),
    )

    peak_proxy_values = select(f_est, is_clean) or select(f_true, is_clean)
    summary.update({
        "samples_total": len(rows),
        "samples_clean": count_true(is_clean),
        "samples_contact": count_true(in_contact),
        "samples_contact_clean": count_true(contact_clean),
        "duration_s": duration(times),
        "contact_sample_fraction": safe_divide(count_true(contact_clean), count_true(is_clean)),
        "first_contact_time_s": first_time(times, contact_clean),
        "contact_duration_s": weighted_duration(dt, contact_clean),
        "task_success": int(any(task_success)),
        "completion_time_s": first_time(times, task_success),
        "hole_clearance_mm": finite_median(hole_clearance),
        "occluded_hole_randomized": int(any(occluded_hole_randomized)),
        "occluded_hole_x_m": finite_median(occluded_hole_x),
        "occluded_hole_y_m": finite_median(occluded_hole_y),
        "occluded_hole_offset_x_m": finite_median(occluded_hole_offset_x),
        "occluded_hole_offset_y_m": finite_median(occluded_hole_offset_y),
        "occluder_alpha": finite_median(occluder_alpha),
        "occluder_style": first_nonempty(occluder_style),
        "audio_feedback_enabled": int(any(audio_feedback)),
        "cushion_used": int(any(cushion_active)),
        "mean_ground_truth_contact_n": mean_or_blank(select(f_true, contact_clean)),
        "mean_estimate_contact_n": mean_or_blank(select(f_est, contact_clean)),
        "peak_ground_truth_contact_n": max_or_blank(select(f_true, contact_clean)),
        "peak_estimate_contact_n": max_or_blank(select(f_est, contact_clean)),
        "peak_contact_proxy_n": max_or_blank(peak_proxy_values),
        "peak_lateral_force_n": max_or_blank(select(lateral, is_clean)),
        "time_above_threshold_s": weighted_duration(
            dt,
            [clean and true >= force_threshold for clean, true in zip(is_clean, f_true)],
        ),
        "time_above_jamming_s": weighted_duration(dt, jamming_mask),
        "jamming_count": count_episodes(jamming_mask),
        "contact_episode_count": count_episodes(contact_clean),
        "contact_force_impulse_n_s": weighted_sum(f_true, dt, contact_clean),
    })
    summary.update(smoothness)
    summary.update(error_metrics(f_true, f_est, contact_clean, "contact"))
    summary.update(error_metrics(f_true, f_est, is_clean, "all_clean"))
    return summary


def read_trial_outcome_metrics(result_dir):
    outcome_path = Path(result_dir) / TRIAL_OUTCOME_NAME
    if not outcome_path.exists():
        return {
            "wall_time_elapsed_s": "",
            "timed_out": "",
            "completion_time_wall_s": "",
        }
    try:
        with outcome_path.open() as f:
            outcome = json.load(f)
    except (OSError, ValueError):
        return {
            "wall_time_elapsed_s": "",
            "timed_out": "",
            "completion_time_wall_s": "",
        }
    wall = outcome.get("wall_time_elapsed_s", "")
    return {
        "wall_time_elapsed_s": wall,
        "timed_out": int(bool(outcome.get("timed_out", False))),
        "completion_time_wall_s": wall,
    }


def smoothness_metrics(times, is_clean, in_contact, target_x, target_y, target_z):
    blank = {
        "mean_action_jerk": "",
        "velocity_reversals": "",
        "retraction_count": "",
    }
    points = [
        (t, x, y, z, contact)
        for t, clean, x, y, z, contact in zip(
            times, is_clean, target_x, target_y, target_z, in_contact
        )
        if clean and math.isfinite(x) and math.isfinite(y) and math.isfinite(z)
    ]
    if len(points) < 4:
        return blank

    ts = [p[0] for p in points]
    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    zs = [p[3] for p in points]
    contacts = [p[4] for p in points]

    vx = finite_differences(xs, ts)
    vy = finite_differences(ys, ts)
    vz = finite_differences(zs, ts)
    ax = finite_differences(vx, ts[1:])
    ay = finite_differences(vy, ts[1:])
    az = finite_differences(vz, ts[1:])
    jx = finite_differences(ax, ts[2:])
    jy = finite_differences(ay, ts[2:])
    jz = finite_differences(az, ts[2:])
    jerks = [math.sqrt(x * x + y * y + z * z) for x, y, z in zip(jx, jy, jz)]

    reversals = (
        count_sign_changes(vx)
        + count_sign_changes(vy)
        + count_sign_changes(vz)
    )
    retractions = count_retractions(vz, contacts[1:] if len(contacts) > 1 else contacts)

    return {
        "mean_action_jerk": mean_or_blank(jerks),
        "velocity_reversals": reversals,
        "retraction_count": retractions,
    }


def finite_differences(values, times):
    if len(values) < 2 or len(times) < 2:
        return []
    diffs = []
    count = min(len(values) - 1, len(times) - 1)
    for i in range(count):
        dt = times[i + 1] - times[i]
        if dt <= EPS:
            diffs.append(0.0)
        else:
            diffs.append((values[i + 1] - values[i]) / dt)
    return diffs


def count_sign_changes(values, deadband=1e-4):
    filtered = [0.0 if abs(value) < deadband else value for value in values]
    changes = 0
    prev = 0.0
    for value in filtered:
        if value == 0.0:
            continue
        if prev != 0.0 and value * prev < 0.0:
            changes += 1
        prev = value
    return changes


def count_retractions(vz, contacts, lift_speed=1e-4):
    """Count upward target-move episodes after contact has begun."""
    return _count_retraction_episodes(vz, contacts, lift_speed)


def _count_retraction_episodes(vz, contacts, lift_speed):
    seen_contact = False
    lifting = []
    for i, speed in enumerate(vz):
        contact = contacts[i] if i < len(contacts) else False
        seen_contact = seen_contact or bool(contact)
        lifting.append(seen_contact and speed > lift_speed)
    return count_episodes(lifting)


def count_episodes(mask):
    episodes = 0
    prev = False
    for value in mask:
        current = bool(value)
        if current and not prev:
            episodes += 1
        prev = current
    return episodes


def summarize_experiment_dir(
    experiment_dir,
    source="auto",
    force_threshold=DEFAULT_FORCE_THRESHOLD_N,
    jamming_threshold=DEFAULT_JAMMING_THRESHOLD_N,
    include_anomalies=False,
):
    experiment_dir = Path(experiment_dir)
    trial_rows = []
    for metadata_path in sorted(experiment_dir.rglob(TRIAL_METADATA_NAME)):
        trial_dir = metadata_path.parent
        metadata = read_json(metadata_path)
        if not isinstance(metadata, dict):
            continue
        if metadata.get("trial_type") != "recorded":
            continue
        if metadata.get("status") != "completed":
            continue
        metrics = analyze_result_dir(
            trial_dir,
            scenario=metadata.get("scenario", "peg_in_hole"),
            source=source,
            force_threshold=force_threshold,
            jamming_threshold=jamming_threshold,
            include_anomalies=include_anomalies,
        )
        row = {
            "tester": metadata.get("tester", experiment_dir.name),
            "tester_id": metadata.get("tester_id", experiment_dir.name),
            "condition": metadata.get("condition", ""),
            "condition_order": " -> ".join(metadata.get("condition_order", [])),
            "condition_position": metadata.get("condition_position", ""),
            "trial_type": metadata.get("trial_type", ""),
            "trial_index": metadata.get("trial_index", ""),
            "trial_dir": str(trial_dir),
            "occluded_hole_seed": metadata.get("occluded_hole_seed", ""),
            "visual_feedback": int(bool(metadata.get("visual_feedback", False))),
            "audio_feedback": int(bool(metadata.get("audio_feedback", False))),
        }
        row.update(metrics)
        trial_rows.append(row)

    trial_path = experiment_dir / "experiment_analysis_summary.csv"
    condition_rows = aggregate_condition_rows(trial_rows)
    condition_path = experiment_dir / "condition_comparison_summary.csv"
    write_rows(trial_path, trial_rows, fieldnames=None)
    write_condition_summary(condition_path, condition_rows)
    print_condition_summary(condition_rows, condition_path)
    return condition_rows


def aggregate_condition_rows(trial_rows):
    grouped = {}
    for row in trial_rows:
        condition = row.get("condition", "")
        grouped.setdefault(condition, []).append(row)

    summaries = []
    for condition, rows in grouped.items():
        summaries.append({
            "tester": rows[0].get("tester", ""),
            "tester_id": rows[0].get("tester_id", ""),
            "condition": condition,
            "n_trials": len(rows),
            "success_rate": mean([float(row.get("task_success") or 0) for row in rows]),
            "mean_completion_time_wall_s": mean_numeric(rows, "completion_time_wall_s"),
            "mean_completion_time_sim_s": mean_numeric(rows, "completion_time_s"),
            "mean_peak_contact_proxy_n": mean_numeric(rows, "peak_contact_proxy_n"),
            "mean_peak_ground_truth_contact_n": mean_numeric(rows, "peak_ground_truth_contact_n"),
            "mean_peak_lateral_force_n": mean_numeric(rows, "peak_lateral_force_n"),
            "mean_time_above_threshold_s": mean_numeric(rows, "time_above_threshold_s"),
            "mean_jamming_count": mean_numeric(rows, "jamming_count"),
            "mean_contact_episode_count": mean_numeric(rows, "contact_episode_count"),
            "mean_action_jerk": mean_numeric(rows, "mean_action_jerk"),
            "mean_velocity_reversals": mean_numeric(rows, "velocity_reversals"),
            "mean_retraction_count": mean_numeric(rows, "retraction_count"),
        })
    summaries.sort(key=lambda row: row["condition"])
    return summaries


def mean_numeric(rows, key):
    values = []
    for row in rows:
        value = row.get(key, "")
        if value in ("", None):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return mean_or_blank(values)


def write_condition_summary(path, rows):
    write_rows(path, rows, fieldnames=CONDITION_SUMMARY_COLUMNS)


def write_rows(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            fieldnames = SUMMARY_COLUMNS
        else:
            fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in fieldnames})


def print_condition_summary(rows, output_path):
    print(f"\nSaved condition comparison to {Path(output_path).resolve()}")
    if not rows:
        print("No completed recorded trials found.")
        return
    print()
    print(
        "condition          n  success  wall_s  peak_proxy  peak_GT  t>thr  jams  episodes"
    )
    print(
        "----------------- -- -------- ------- ----------- -------- ------ ----- --------"
    )
    for row in rows:
        print(
            f"{str(row['condition'])[:17]:17} "
            f"{display(row['n_trials'], width=2)} "
            f"{display(row['success_rate'], width=8)} "
            f"{display(row['mean_completion_time_wall_s'], width=7)} "
            f"{display(row['mean_peak_contact_proxy_n'], width=11)} "
            f"{display(row['mean_peak_ground_truth_contact_n'], width=8)} "
            f"{display(row['mean_time_above_threshold_s'], width=6)} "
            f"{display(row['mean_jamming_count'], width=5)} "
            f"{display(row['mean_contact_episode_count'], width=8)}"
        )


def read_json(path):
    try:
        with Path(path).open() as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def select_log_path(results_dir, scenario, source):
    return select_log_path_from_dir(results_dir / scenario, source)


def select_log_path_from_dir(result_dir, source):
    filtered_path = result_dir / FILTERED_LOG_NAME
    raw_path = result_dir / RAW_LOG_NAME

    if source == "filtered":
        return filtered_path if filtered_path.exists() else None
    if source == "raw":
        return raw_path if raw_path.exists() else None
    if filtered_path.exists():
        return filtered_path
    if raw_path.exists():
        return raw_path
    return None


def empty_summary(
    scenario,
    status,
    force_threshold,
    jamming_threshold=DEFAULT_JAMMING_THRESHOLD_N,
    source_csv="",
    used_filtered_csv=False,
):
    row = {column: "" for column in SUMMARY_COLUMNS}
    row.update({
        "scenario": scenario,
        "status": status,
        "source_csv": str(source_csv),
        "used_filtered_csv": int(used_filtered_csv),
        "force_threshold_n": force_threshold,
        "jamming_threshold_n": jamming_threshold,
    })
    return row


def optional_float_column(rows, columns, name, default=0.0):
    if name not in columns:
        return [default] * len(rows)
    return float_column(rows, name, default)


def optional_bool_column(rows, columns, name):
    if name not in columns:
        return [False] * len(rows)
    return bool_column(rows, name)


def optional_text_column(rows, columns, name):
    if name not in columns:
        return [""] * len(rows)
    return [row.get(name, "") for row in rows]


def float_column(rows, name, default=0.0):
    values = []
    for row in rows:
        raw_value = row.get(name, "")
        if raw_value in ("", None):
            values.append(default)
            continue
        try:
            values.append(float(raw_value))
        except ValueError:
            values.append(default)
    return values


def bool_column(rows, name):
    return [bool(value) for value in float_column(rows, name)]


def and_masks(*masks):
    return [all(values) for values in zip(*masks)]


def count_true(mask):
    return sum(1 for value in mask if value)


def select(values, mask):
    return [value for value, selected in zip(values, mask) if selected]


def sample_widths(times):
    if len(times) == 0:
        return []
    if len(times) == 1:
        return [0.0]

    diffs = [max(b - a, 0.0) for a, b in zip(times, times[1:])]
    positive_diffs = [value for value in diffs if value > 0.0]
    final_width = median(positive_diffs) if positive_diffs else 0.0
    return diffs + [final_width]


def duration(times):
    if len(times) < 2:
        return 0.0
    return max(times[-1] - times[0], 0.0)


def first_time(times, mask):
    for time, selected in zip(times, mask):
        if selected:
            return time
    return ""


def weighted_duration(widths, mask):
    return sum(width for width, selected in zip(widths, mask) if selected)


def weighted_sum(values, widths, mask):
    return sum(
        value * width
        for value, width, selected in zip(values, widths, mask)
        if selected
    )


def error_metrics(f_true, f_est, mask, scope):
    if scope == "contact":
        keys = {
            "mae": "mae_contact_n",
            "mse": "mse_contact_n2",
            "rmse": "rmse_contact_n",
            "bias": "bias_contact_n",
            "median_abs": "median_abs_error_contact_n",
            "p95_abs": "p95_abs_error_contact_n",
            "max_abs": "max_abs_error_contact_n",
            "nmae": "nmae_contact_mean_gt",
            "nrmse": "nrmse_contact_mean_gt",
        }
    else:
        keys = {
            "mae": "mae_all_clean_n",
            "mse": "mse_all_clean_n2",
            "rmse": "rmse_all_clean_n",
            "bias": "bias_all_clean_n",
            "p95_abs": "p95_abs_error_all_clean_n",
        }

    selected_true = select(f_true, mask)
    selected_est = select(f_est, mask)
    if not selected_true:
        return {key: "" for key in keys.values()}

    error = [estimate - truth for truth, estimate in zip(selected_true, selected_est)]
    abs_error = [abs(value) for value in error]
    mse = mean([value ** 2 for value in error])
    metrics = {
        keys["mae"]: mean(abs_error),
        keys["mse"]: mse,
        keys["rmse"]: math.sqrt(mse),
        keys["bias"]: mean(error),
        keys["p95_abs"]: percentile(abs_error, 95),
    }

    if scope == "contact":
        mean_gt = mean([abs(value) for value in selected_true])
        metrics.update({
            keys["median_abs"]: median(abs_error),
            keys["max_abs"]: max(abs_error),
            keys["nmae"]: safe_divide(metrics[keys["mae"]], mean_gt),
            keys["nrmse"]: safe_divide(metrics[keys["rmse"]], mean_gt),
        })

    return metrics


def safe_divide(numerator, denominator):
    denominator = float(denominator)
    if abs(denominator) <= EPS:
        return ""
    return float(numerator) / denominator


def mean_or_blank(values):
    if not values:
        return ""
    return mean(values)


def max_or_blank(values):
    if not values:
        return ""
    return max(values)


def finite_median(values):
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return ""
    return median(finite_values)


def first_nonempty(values):
    for value in values:
        if value:
            return value
    return ""


def mean(values):
    return sum(values) / len(values)


def median(values):
    if not values:
        return ""
    return percentile(values, 50)


def percentile(values, pct):
    if not values:
        return ""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def write_summary(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in SUMMARY_COLUMNS})


def csv_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.10g}"
    return value


def print_summary(rows, output_path):
    print(f"Saved analysis summary to {output_path.resolve()}")
    print()
    print("scenario        status       contact n   MAE (N)   RMSE (N)   bias (N)   peak GT (N)")
    print("--------------- ------------ ----------- --------- ---------- ---------- ------------")
    for row in rows:
        print(
            f"{str(row['scenario'])[:15]:15} "
            f"{str(row['status'])[:12]:12} "
            f"{display(row['samples_contact_clean'], width=11)} "
            f"{display(row['mae_contact_n'], width=9)} "
            f"{display(row['rmse_contact_n'], width=10)} "
            f"{display(row['bias_contact_n'], width=10)} "
            f"{display(row['peak_ground_truth_contact_n'], width=12)}"
        )


def display(value, width):
    if value == "":
        return " " * width
    if isinstance(value, float):
        text = f"{value:.3g}"
    else:
        text = str(value)
    return text[:width].rjust(width)


if __name__ == "__main__":
    main()
