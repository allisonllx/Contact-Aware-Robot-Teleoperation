
import csv
import json
import math

from ..schemas import CONDITIONS, PRIMARY_METRIC, SECONDARY_METRICS


class ValidationError(ValueError):
    """Raised when active experiment data are not safe for inference."""

def _read_active_trial_rows(root):
    rows = []
    issues = []
    tester_count = 0
    threshold_values = {
        "force_threshold_n": set(),
        "jamming_threshold_n": set(),
    }
    for tester_dir in sorted(root.iterdir()):
        if not tester_dir.is_dir() or not (tester_dir / "experiment_plan.json").is_file():
            continue
        tester_count += 1
        with (tester_dir / "experiment_plan.json").open() as f:
            plan = json.load(f)
        tester_id = str(plan.get("tester_id") or tester_dir.name)
        planned_conditions = tuple(plan.get("conditions", ()))
        recorded_trials = int(plan.get("recorded_trials", 0))
        if set(planned_conditions) != set(CONDITIONS) or len(planned_conditions) != len(CONDITIONS):
            issues.append(_validation_issue(
                "error",
                "unexpected_conditions",
                tester_id,
                message=(
                    f"Expected exactly {', '.join(CONDITIONS)}; found "
                    f"{', '.join(planned_conditions) or 'none'}"
                ),
            ))
        if recorded_trials != 3:
            issues.append(_validation_issue(
                "error",
                "unexpected_recorded_trial_count",
                tester_id,
                message=f"Expected 3 recorded trials per condition; found {recorded_trials}",
            ))
        summary_path = tester_dir / "experiment_analysis_summary.csv"
        if not summary_path.is_file():
            issues.append(_validation_issue(
                "error",
                "missing_summary",
                tester_id,
                message=f"Missing {summary_path.name}",
            ))
            continue
        with summary_path.open(newline="") as f:
            source_rows = list(csv.DictReader(f))
        indexed = {}
        for row in source_rows:
            if row.get("trial_type") != "recorded":
                continue
            key = (row.get("condition", ""), _int_or_zero(row.get("trial_index")))
            indexed.setdefault(key, []).append(row)

        for condition in planned_conditions:
            for trial_index in range(1, recorded_trials + 1):
                metadata_path = (
                    tester_dir
                    / condition
                    / f"recorded_{trial_index:02d}"
                    / "trial_metadata.json"
                )
                if not metadata_path.is_file():
                    issues.append(_validation_issue(
                        "error",
                        "missing_trial_metadata",
                        tester_id,
                        condition,
                        trial_index,
                        f"Missing planned trial metadata: {metadata_path}",
                    ))
                else:
                    with metadata_path.open() as f:
                        metadata = json.load(f)
                    if metadata.get("status") != "completed":
                        issues.append(_validation_issue(
                            "error",
                            "incomplete_trial",
                            tester_id,
                            condition,
                            trial_index,
                            f"Trial status is {metadata.get('status', 'missing')!r}",
                        ))
                    expected_metadata = {
                        "tester_id": tester_id,
                        "condition": condition,
                        "trial_type": "recorded",
                        "trial_index": trial_index,
                    }
                    mismatches = [
                        f"{key}={metadata.get(key)!r}"
                        for key, expected in expected_metadata.items()
                        if metadata.get(key) != expected
                    ]
                    if mismatches:
                        issues.append(_validation_issue(
                            "error",
                            "metadata_mismatch",
                            tester_id,
                            condition,
                            trial_index,
                            "Unexpected metadata: " + ", ".join(mismatches),
                        ))

                matches = indexed.get((condition, trial_index), [])
                if not matches:
                    issues.append(_validation_issue(
                        "error",
                        "missing_summary_row",
                        tester_id,
                        condition,
                        trial_index,
                        "No experiment summary row for planned trial",
                    ))
                    continue
                if len(matches) > 1:
                    issues.append(_validation_issue(
                        "error",
                        "duplicate_trial",
                        tester_id,
                        condition,
                        trial_index,
                        f"Found {len(matches)} summary rows for one planned trial",
                    ))
                    continue

                row = matches[0]
                if row.get("status") != "ok":
                    issues.append(_validation_issue(
                        "error",
                        "invalid_summary_status",
                        tester_id,
                        condition,
                        trial_index,
                        f"Summary status is {row.get('status', 'missing')!r}",
                    ))
                primary_value = _float_or_blank(row.get(PRIMARY_METRIC, ""))
                if not math.isfinite(primary_value):
                    issues.append(_validation_issue(
                        "error",
                        "missing_primary_metric",
                        tester_id,
                        condition,
                        trial_index,
                        f"{PRIMARY_METRIC} is missing or non-finite",
                    ))
                for threshold_key, values in threshold_values.items():
                    value = _float_or_blank(row.get(threshold_key, ""))
                    if math.isfinite(value):
                        values.add(value)
                converted = dict(row)
                for key in (
                    PRIMARY_METRIC,
                    "peak_contact_proxy_n",
                    "task_success",
                    "completion_time_wall_s",
                    "timed_out",
                    *SECONDARY_METRICS,
                ):
                    converted[key] = _float_or_blank(row.get(key, ""))
                converted["trial_index"] = int(row["trial_index"])
                rows.append(converted)
    if tester_count == 0:
        issues.append(_validation_issue(
            "error",
            "no_active_testers",
            message=f"No tester directories with experiment_plan.json found under {root}",
        ))
    threshold_codes = {
        "force_threshold_n": "inconsistent_force_threshold",
        "jamming_threshold_n": "inconsistent_jamming_threshold",
    }
    for threshold_key, values in threshold_values.items():
        if len(values) > 1:
            issues.append(_validation_issue(
                "error",
                threshold_codes[threshold_key],
                message=(
                    f"Active trials contain multiple {threshold_key} values: "
                    + ", ".join(str(value) for value in sorted(values))
                ),
            ))
    if not issues:
        issues.append(_validation_issue(
            "info",
            "validation_passed",
            message=f"Validated {len(rows)} recorded trial rows",
        ))
    return rows, issues

def _validation_issue(
    severity,
    code,
    tester_id="",
    condition="",
    trial_index="",
    message="",
):
    return {
        "severity": severity,
        "code": code,
        "tester_id": tester_id,
        "condition": condition,
        "trial_index": trial_index,
        "message": message,
    }

def _float_or_blank(value):
    if value in ("", None):
        return math.nan
    return float(value)

def _int_or_zero(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
