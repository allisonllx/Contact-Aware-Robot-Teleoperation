import argparse
import csv
import json
import math
import statistics
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

FORCE_ESTIMATION_DIR = Path("force_estimation_runs")
EXPERIMENT_RESULTS_DIR = Path("experiment_results")
FORCE_ESTIMATION_SCENARIOS = ("hit_floor", "push_block", "peg_in_hole")
FORCE_COMPARISON_CANDIDATES = (
    "force_comparison_contact_only_filtered.png",
    "force_comparison_filtered.png",
    "force_comparison_contact_only_raw.png",
    "force_comparison_raw.png",
    "force_comparison_contact_only.png",
    "force_comparison.png",
)

FORCE_EST_PER_RUN_COLUMNS = [
    "scenario",
    "run_id",
    "source",
    "run_dir",
    "status",
    "source_csv",
    "force_comparison_png",
    "samples_contact_clean",
    "mae_contact_n",
    "mse_contact_n2",
    "rmse_contact_n",
    "bias_contact_n",
    "median_abs_error_contact_n",
    "p95_abs_error_contact_n",
    "max_abs_error_contact_n",
]

FORCE_EST_BY_SCENARIO_COLUMNS = [
    "scenario",
    "source",
    "n_runs",
    "n_ok",
    "mean_mae_contact_n",
    "std_mae_contact_n",
    "mean_mse_contact_n2",
    "std_mse_contact_n2",
    "mean_rmse_contact_n",
    "std_rmse_contact_n",
    "mean_bias_contact_n",
    "std_bias_contact_n",
]

FORCE_EST_ERROR_KEYS = (
    "mae_contact_n",
    "mse_contact_n2",
    "rmse_contact_n",
    "bias_contact_n",
)

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
    parser.add_argument(
        "--force-estimation-report",
        action="store_true",
        help=(
            "Aggregate multi-run force-estimation accuracy under "
            "force_estimation_runs/ (MAE/MSE tables + plots)."
        ),
    )
    parser.add_argument(
        "--force-estimation-root",
        type=Path,
        default=FORCE_ESTIMATION_DIR,
        help="Root folder of scripted repeats: <root>/<scenario>/run_XX/.",
    )
    parser.add_argument(
        "--include-tester-pool",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also ingest experiment_results/** peg_in_hole trial logs as "
            "source=tester (default: on)."
        ),
    )
    parser.add_argument(
        "--experiment-results-dir",
        type=Path,
        default=EXPERIMENT_RESULTS_DIR,
        help="Tester pool root used with --force-estimation-report.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.force_threshold <= 0.0:
        raise ValueError("--force-threshold must be positive")
    if args.jamming_threshold <= 0.0:
        raise ValueError("--jamming-threshold must be positive")

    if args.force_estimation_report:
        write_force_estimation_report(
            root=args.force_estimation_root,
            source=args.source,
            force_threshold=args.force_threshold,
            jamming_threshold=args.jamming_threshold,
            include_anomalies=args.include_anomalies,
            include_tester_pool=args.include_tester_pool,
            experiment_results_dir=args.experiment_results_dir,
        )
        return

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


def write_force_estimation_report(
    root,
    source="auto",
    force_threshold=DEFAULT_FORCE_THRESHOLD_N,
    jamming_threshold=DEFAULT_JAMMING_THRESHOLD_N,
    include_anomalies=False,
    include_tester_pool=True,
    experiment_results_dir=EXPERIMENT_RESULTS_DIR,
):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    discovered = discover_force_estimation_runs(root)
    if include_tester_pool:
        discovered.extend(discover_tester_pool_runs(Path(experiment_results_dir)))

    per_run_rows = []
    for item in discovered:
        summary = analyze_result_dir(
            result_dir=item["run_dir"],
            scenario=item["scenario"],
            source=source,
            force_threshold=force_threshold,
            jamming_threshold=jamming_threshold,
            include_anomalies=include_anomalies,
        )
        per_run_rows.append(
            {
                "scenario": item["scenario"],
                "run_id": item["run_id"],
                "source": item["source"],
                "run_dir": str(item["run_dir"]),
                "status": summary.get("status", ""),
                "source_csv": summary.get("source_csv", ""),
                "force_comparison_png": find_force_comparison_png(item["run_dir"]),
                "samples_contact_clean": summary.get("samples_contact_clean", ""),
                "mae_contact_n": summary.get("mae_contact_n", ""),
                "mse_contact_n2": summary.get("mse_contact_n2", ""),
                "rmse_contact_n": summary.get("rmse_contact_n", ""),
                "bias_contact_n": summary.get("bias_contact_n", ""),
                "median_abs_error_contact_n": summary.get(
                    "median_abs_error_contact_n", ""
                ),
                "p95_abs_error_contact_n": summary.get("p95_abs_error_contact_n", ""),
                "max_abs_error_contact_n": summary.get("max_abs_error_contact_n", ""),
            }
        )

    by_scenario_rows = aggregate_force_estimation_rows(per_run_rows)

    per_run_path = root / "force_estimation_per_run.csv"
    by_scenario_path = root / "force_estimation_by_scenario.csv"
    write_rows(per_run_path, per_run_rows, FORCE_EST_PER_RUN_COLUMNS)
    write_rows(by_scenario_path, by_scenario_rows, FORCE_EST_BY_SCENARIO_COLUMNS)

    plots_dir = root / "plots"
    plot_force_estimation_bars(by_scenario_rows, per_run_rows, plots_dir)
    write_force_estimation_exemplars(per_run_rows, plots_dir / "exemplar_overlays.txt")

    print_force_estimation_report(
        per_run_rows,
        by_scenario_rows,
        per_run_path,
        by_scenario_path,
        plots_dir,
    )


def discover_force_estimation_runs(root):
    """Discover scripted repeats under force_estimation_runs/<scenario>/run_XX/."""
    root = Path(root)
    discovered = []
    if not root.exists():
        return discovered

    scenario_names = list(FORCE_ESTIMATION_SCENARIOS)
    for child in sorted(root.iterdir()):
        if child.is_dir() and child.name not in scenario_names and child.name != "plots":
            if child.name in SCENARIOS:
                scenario_names.append(child.name)

    for scenario in scenario_names:
        scenario_dir = root / scenario
        if not scenario_dir.is_dir():
            continue
        run_dirs = sorted(
            path
            for path in scenario_dir.iterdir()
            if path.is_dir() and path.name.startswith("run_")
        )
        if not run_dirs and select_log_path_from_dir(scenario_dir, "auto") is not None:
            # Allow a single flat folder as run_01 for convenience.
            run_dirs = [scenario_dir]
        for run_dir in run_dirs:
            run_id = run_dir.name if run_dir != scenario_dir else "run_01"
            discovered.append(
                {
                    "scenario": scenario,
                    "run_id": run_id,
                    "source": "scripted",
                    "run_dir": run_dir.resolve(),
                }
            )
    return discovered


def discover_tester_pool_runs(experiment_results_dir):
    """Discover occluded peg_in_hole trial logs under experiment_results/."""
    experiment_results_dir = Path(experiment_results_dir)
    discovered = []
    if not experiment_results_dir.exists():
        return discovered

    seen = set()
    for pattern in (FILTERED_LOG_NAME, RAW_LOG_NAME):
        for log_path in sorted(experiment_results_dir.rglob(pattern)):
            run_dir = log_path.parent.resolve()
            if run_dir in seen:
                continue
            if select_log_path_from_dir(run_dir, "auto") is None:
                continue
            metadata = read_json(run_dir / TRIAL_METADATA_NAME)
            if metadata is not None and metadata.get("status") != "completed":
                continue
            seen.add(run_dir)
            try:
                rel = run_dir.relative_to(experiment_results_dir.resolve())
                run_id = str(rel).replace("\\", "/")
            except ValueError:
                run_id = run_dir.name
            discovered.append(
                {
                    "scenario": "peg_in_hole",
                    "run_id": run_id,
                    "source": "tester",
                    "run_dir": run_dir,
                }
            )
    return discovered


def find_force_comparison_png(run_dir):
    run_dir = Path(run_dir)
    for name in FORCE_COMPARISON_CANDIDATES:
        path = run_dir / name
        if path.exists():
            return str(path.resolve())
    return ""


def aggregate_force_estimation_rows(per_run_rows):
    groups = {}
    for row in per_run_rows:
        key = (row["scenario"], row["source"])
        groups.setdefault(key, []).append(row)

    aggregated = []
    for (scenario, source), rows in sorted(groups.items()):
        ok_rows = [
            row
            for row in rows
            if row.get("status") not in ("missing_csv", "")
            and is_finite_number(row.get("mae_contact_n"))
        ]
        aggregate = {
            "scenario": scenario,
            "source": source,
            "n_runs": len(rows),
            "n_ok": len(ok_rows),
        }
        for key in FORCE_EST_ERROR_KEYS:
            values = [float(row[key]) for row in ok_rows if is_finite_number(row.get(key))]
            aggregate[f"mean_{key}"] = mean_or_blank(values)
            aggregate[f"std_{key}"] = std_or_blank(values)
        aggregated.append(aggregate)
    return aggregated


def plot_force_estimation_bars(by_scenario_rows, per_run_rows, plots_dir):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping force-estimation plots")
        return

    labels = []
    mae_means = []
    mae_stds = []
    mse_means = []
    mse_stds = []
    for row in by_scenario_rows:
        if row["n_ok"] == 0:
            continue
        labels.append(f"{row['scenario']}\n({row['source']})")
        mae_means.append(float(row["mean_mae_contact_n"]))
        mae_stds.append(
            float(row["std_mae_contact_n"])
            if is_finite_number(row["std_mae_contact_n"])
            else 0.0
        )
        mse_means.append(float(row["mean_mse_contact_n2"]))
        mse_stds.append(
            float(row["std_mse_contact_n2"])
            if is_finite_number(row["std_mse_contact_n2"])
            else 0.0
        )

    if labels:
        _save_error_bar_chart(
            plt,
            labels,
            mae_means,
            mae_stds,
            ylabel="MAE (N)",
            title="Contact-force MAE by scenario",
            path=plots_dir / "mae_by_scenario.png",
        )
        _save_error_bar_chart(
            plt,
            labels,
            mse_means,
            mse_stds,
            ylabel="MSE (N²)",
            title="Contact-force MSE by scenario",
            path=plots_dir / "mse_by_scenario.png",
        )

    _save_error_box_plot(
        plt,
        per_run_rows,
        metric_key="mae_contact_n",
        ylabel="MAE (N)",
        title="Contact-force MAE distribution",
        path=plots_dir / "mae_box_by_scenario.png",
    )
    _save_error_box_plot(
        plt,
        per_run_rows,
        metric_key="mse_contact_n2",
        ylabel="MSE (N²)",
        title="Contact-force MSE distribution",
        path=plots_dir / "mse_box_by_scenario.png",
    )


def _save_error_bar_chart(plt, labels, means, stds, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(max(6.0, 1.4 * len(labels)), 4.5))
    x = list(range(len(labels)))
    ax.bar(x, means, yerr=stds, capsize=4, color="#4C78A8", ecolor="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_error_box_plot(plt, per_run_rows, metric_key, ylabel, title, path):
    groups = {}
    for row in per_run_rows:
        if not is_finite_number(row.get(metric_key)):
            continue
        if row.get("status") in ("missing_csv", ""):
            continue
        label = f"{row['scenario']}\n({row['source']})"
        groups.setdefault(label, []).append(float(row[metric_key]))
    if not groups:
        return

    labels = sorted(groups)
    data = [groups[label] for label in labels]
    fig, ax = plt.subplots(figsize=(max(6.0, 1.4 * len(labels)), 4.5))
    try:
        ax.boxplot(data, tick_labels=labels, showmeans=True)
    except TypeError:
        ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_force_estimation_exemplars(per_run_rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Representative per-run GT vs estimate overlays",
        "# Prefer contact-only filtered force_comparison_*.png from each group.",
        "",
    ]
    best_by_group = {}
    for row in per_run_rows:
        if not row.get("force_comparison_png"):
            continue
        if not is_finite_number(row.get("mae_contact_n")):
            continue
        key = (row["scenario"], row["source"])
        current = best_by_group.get(key)
        if current is None or float(row["mae_contact_n"]) < float(current["mae_contact_n"]):
            best_by_group[key] = row

    for (scenario, source), row in sorted(best_by_group.items()):
        lines.append(
            f"{scenario} / {source} / {row['run_id']}: {row['force_comparison_png']}"
        )
    if len(lines) == 3:
        lines.append("(no force_comparison PNGs found in discovered runs)")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_force_estimation_report(
    per_run_rows,
    by_scenario_rows,
    per_run_path,
    by_scenario_path,
    plots_dir,
):
    print(f"Saved per-run force-estimation table to {per_run_path.resolve()}")
    print(f"Saved by-scenario force-estimation table to {by_scenario_path.resolve()}")
    print(f"Plots directory: {plots_dir.resolve()}")
    print()
    print(
        "scenario          source     n_ok/n   MAE mean±std (N)      MSE mean±std (N²)"
    )
    print(
        "----------------- ---------- -------- --------------------- ---------------------"
    )
    for row in by_scenario_rows:
        n_text = f"{row['n_ok']}/{row['n_runs']}"
        mae_text = format_mean_std(row["mean_mae_contact_n"], row["std_mae_contact_n"])
        mse_text = format_mean_std(
            row["mean_mse_contact_n2"], row["std_mse_contact_n2"]
        )
        print(
            f"{str(row['scenario'])[:17]:17} "
            f"{str(row['source'])[:10]:10} "
            f"{n_text:8} "
            f"{mae_text:21} "
            f"{mse_text:21}"
        )
    if not by_scenario_rows:
        print("(no runs found)")
        print(
            "Collect repeats with ./scripts/run_force_estimation_repeats.sh "
            "or copy logs into force_estimation_runs/<scenario>/run_XX/"
        )
    print()
    print(f"Per-run rows analyzed: {len(per_run_rows)}")


def format_mean_std(mean_value, std_value):
    if not is_finite_number(mean_value):
        return ""
    mean_text = f"{float(mean_value):.3g}"
    if not is_finite_number(std_value):
        return mean_text
    return f"{mean_text}±{float(std_value):.3g}"


def is_finite_number(value):
    if value is None or value == "":
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def std_or_blank(values):
    if len(values) < 2:
        return ""
    return statistics.stdev(values)


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
