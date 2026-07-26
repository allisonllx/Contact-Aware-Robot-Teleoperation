"""Persist and summarize completed experiment trials."""

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from analysis import (
    SUMMARY_COLUMNS,
    aggregate_condition_rows,
    analyze_result_dir,
    csv_value,
    print_condition_summary,
    write_condition_summary,
)
from franka_force.config import DEFAULT_JAMMING_THRESHOLD_N

from .planning import trial_completed as default_trial_completed


SCENARIO = "peg_in_hole"
EXPERIMENT_COLUMNS = [
    "tester", "tester_id", "condition", "condition_order", "condition_position",
    "trial_type", "trial_index", "trial_dir", "occluded_hole_seed",
    "visual_feedback", "audio_feedback",
]
EXPERIMENT_SUMMARY_COLUMNS = EXPERIMENT_COLUMNS + SUMMARY_COLUMNS


def write_experiment_summaries(
    tester_dir,
    tester_name,
    tester_id,
    plan,
    trial_specs,
    force_threshold,
    *,
    jamming_threshold=DEFAULT_JAMMING_THRESHOLD_N,
    trial_completed=None,
):
    """Write summaries, defaulting to the standard completion predicate."""
    if trial_completed is None:
        trial_completed = default_trial_completed
    grouped = {"recorded": [], "practice": [], "familiarization": []}
    for trial in trial_specs:
        if trial_completed(trial["trial_dir"]):
            grouped[trial["trial_type"]].append(
                experiment_analysis_row(
                    tester_name,
                    tester_id,
                    plan,
                    trial,
                    force_threshold,
                    jamming_threshold,
                )
            )
    paths = {
        "recorded": tester_dir / "experiment_analysis_summary.csv",
        "practice": tester_dir / "practice_analysis_summary.csv",
        "familiarization": tester_dir / "familiarization_analysis_summary.csv",
    }
    for trial_type, path in paths.items():
        write_experiment_summary(path, grouped[trial_type])
    condition_path = tester_dir / "condition_comparison_summary.csv"
    condition_rows = aggregate_condition_rows(grouped["recorded"])
    write_condition_summary(condition_path, condition_rows)
    print(f"\nSaved recorded-trial analysis to {paths['recorded'].resolve()}")
    print(f"Saved practice-trial analysis to {paths['practice'].resolve()}")
    print(f"Saved familiarization-trial analysis to {paths['familiarization'].resolve()}")
    print_condition_summary(condition_rows, condition_path)


def experiment_analysis_row(
    tester_name,
    tester_id,
    plan,
    trial,
    force_threshold,
    jamming_threshold=DEFAULT_JAMMING_THRESHOLD_N,
):
    metrics = analyze_result_dir(
        trial["trial_dir"],
        scenario=SCENARIO,
        source="auto",
        force_threshold=force_threshold,
        jamming_threshold=jamming_threshold,
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
    extra_columns = (
        [key for key in rows[0] if key not in EXPERIMENT_SUMMARY_COLUMNS]
        if rows
        else []
    )
    fieldnames = EXPERIMENT_SUMMARY_COLUMNS + extra_columns
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in fieldnames})


def zip_tester_results(tester_dir):
    tester_dir = Path(tester_dir).resolve()
    archive = shutil.make_archive(
        str(tester_dir.parent / tester_dir.name),
        "zip",
        root_dir=tester_dir.parent,
        base_dir=tester_dir.name,
    )
    return Path(archive).resolve()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(data)
    serializable.pop("loaded_existing_plan", None)
    with path.open("w") as file:
        json.dump(serializable, file, indent=2, sort_keys=True)
        file.write("\n")


def read_json_if_exists(path):
    if not path.exists():
        return None
    try:
        with path.open() as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {"status": "metadata_unreadable"}


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
