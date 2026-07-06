import argparse
import csv
import math
from pathlib import Path

from franka_force.config import RESULTS_DIR, SCENARIOS


RAW_LOG_NAME = "force_verification_log.csv"
FILTERED_LOG_NAME = "force_verification_log_filtered.csv"
DEFAULT_FORCE_THRESHOLD_N = 100.0
EPS = 1e-9

SUMMARY_COLUMNS = [
    "scenario",
    "status",
    "source_csv",
    "used_filtered_csv",
    "force_threshold_n",
    "samples_total",
    "samples_clean",
    "samples_contact",
    "samples_contact_clean",
    "duration_s",
    "contact_sample_fraction",
    "first_contact_time_s",
    "contact_duration_s",
    "task_success",
    "completion_time_s",
    "hole_clearance_mm",
    "occluded_hole_randomized",
    "occluded_hole_x_m",
    "occluded_hole_y_m",
    "occluded_hole_offset_x_m",
    "occluded_hole_offset_y_m",
    "audio_feedback_enabled",
    "cushion_used",
    "mean_ground_truth_contact_n",
    "mean_estimate_contact_n",
    "peak_ground_truth_contact_n",
    "peak_estimate_contact_n",
    "time_above_threshold_s",
    "contact_force_impulse_n_s",
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
        "--include-anomalies",
        action="store_true",
        help="Include rows flagged as anomalies when reading raw logs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.force_threshold <= 0.0:
        raise ValueError("--force-threshold must be positive")

    output_path = args.output or args.results_dir / "force_analysis_summary.csv"
    rows = [
        analyze_scenario(
            scenario=scenario,
            results_dir=args.results_dir,
            source=args.source,
            force_threshold=args.force_threshold,
            include_anomalies=args.include_anomalies,
        )
        for scenario in args.scenarios
    ]

    write_summary(output_path, rows)
    print_summary(rows, output_path)


def analyze_scenario(scenario, results_dir, source, force_threshold, include_anomalies):
    log_path = select_log_path(results_dir, scenario, source)
    if log_path is None:
        return empty_summary(
            scenario,
            status="missing_csv",
            force_threshold=force_threshold,
        )

    with log_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    summary = empty_summary(
        scenario,
        status="ok",
        force_threshold=force_threshold,
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
        "audio_feedback_enabled": int(any(audio_feedback)),
        "cushion_used": int(any(cushion_active)),
        "mean_ground_truth_contact_n": mean_or_blank(select(f_true, contact_clean)),
        "mean_estimate_contact_n": mean_or_blank(select(f_est, contact_clean)),
        "peak_ground_truth_contact_n": max_or_blank(select(f_true, contact_clean)),
        "peak_estimate_contact_n": max_or_blank(select(f_est, contact_clean)),
        "time_above_threshold_s": weighted_duration(
            dt,
            [clean and true >= force_threshold for clean, true in zip(is_clean, f_true)],
        ),
        "contact_force_impulse_n_s": weighted_sum(f_true, dt, contact_clean),
    })
    summary.update(error_metrics(f_true, f_est, contact_clean, "contact"))
    summary.update(error_metrics(f_true, f_est, is_clean, "all_clean"))
    return summary


def select_log_path(results_dir, scenario, source):
    scenario_dir = results_dir / scenario
    filtered_path = scenario_dir / FILTERED_LOG_NAME
    raw_path = scenario_dir / RAW_LOG_NAME

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
