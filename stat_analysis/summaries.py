
import csv
import math
from pathlib import Path

from .io import csv_value, display, mean, mean_or_blank, read_json
from .schemas import (
    CONDITION_SUMMARY_COLUMNS,
    DEFAULT_FORCE_THRESHOLD_N,
    DEFAULT_JAMMING_THRESHOLD_N,
    SUMMARY_COLUMNS,
    TRIAL_METADATA_NAME,
)
from .telemetry import analyze_result_dir


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
