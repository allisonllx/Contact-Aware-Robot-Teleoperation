
import math
import statistics

from ..schemas import CONDITIONS, PRIMARY_METRIC, SECONDARY_METRICS


def _aggregate_participant_conditions(trial_rows):
    grouped = {}
    for row in trial_rows:
        key = (row["tester_id"], row["condition"])
        grouped.setdefault(key, []).append(row)

    aggregated = []
    for (tester_id, condition), rows in sorted(grouped.items()):
        aggregate = {
            "tester_id": tester_id,
            "condition": condition,
            "n_trials": len(rows),
        }
        for metric in (PRIMARY_METRIC, "peak_contact_proxy_n", *SECONDARY_METRICS):
            aggregate[metric] = _median_numeric(row.get(metric) for row in rows)
        aggregate["success_rate"] = statistics.mean(
            float(row["task_success"]) for row in rows
        )
        aggregate["timeout_count"] = sum(int(row["timed_out"]) for row in rows)
        aggregate["median_successful_completion_time_wall_s"] = _median_numeric(
            row["completion_time_wall_s"]
            for row in rows
            if row["task_success"] == 1
        )
        aggregated.append(aggregate)
    return aggregated

def _participant_value(rows, tester_id, condition, metric):
    matches = [
        row[metric]
        for row in rows
        if row["tester_id"] == tester_id and row["condition"] == condition
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one aggregate row for {tester_id}/{condition}; found {len(matches)}"
        )
    return matches[0]

def _completion_survival_rows(trial_rows):
    return [
        {
            "tester_id": row["tester_id"],
            "condition": row["condition"],
            "trial_index": row["trial_index"],
            "duration_s": row["completion_time_wall_s"],
            "event": int(row["task_success"] == 1),
            "censored": int(row["task_success"] != 1),
        }
        for row in trial_rows
    ]

def _median_numeric(values):
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.median(finite) if finite else math.nan
