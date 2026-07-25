"""Participant-level statistical analysis across experiment tester runs."""

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path


CONDITIONS = (
    "no_feedback",
    "visual_feedback",
    "audio_feedback",
    "both_feedback",
)
WILLIAMS_ORDERS = (
    ("no_feedback", "visual_feedback", "both_feedback", "audio_feedback"),
    ("visual_feedback", "audio_feedback", "no_feedback", "both_feedback"),
    ("audio_feedback", "both_feedback", "visual_feedback", "no_feedback"),
    ("both_feedback", "no_feedback", "audio_feedback", "visual_feedback"),
)
PRIMARY_METRIC = "peak_ground_truth_contact_n"
SECONDARY_METRICS = (
    "jamming_count",
    "time_above_threshold_s",
    "contact_force_impulse_n_s",
    "mean_action_jerk",
    "velocity_reversals",
    "retraction_count",
)


class ValidationError(ValueError):
    """Raised when active experiment data are not safe for inference."""


def run_analysis(
    experiment_results_dir,
    output_dir,
    *,
    permutations=100_000,
    bootstrap_resamples=10_000,
    seed=20260725,
    alpha=0.05,
):
    """Analyze all complete active tester runs and write a reproducible report."""
    root = Path(experiment_results_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    trial_rows, validation = _read_active_trial_rows(root)
    _write_csv(
        output / "validation_report.csv",
        validation,
        fieldnames=("severity", "code", "tester_id", "condition", "trial_index", "message"),
    )
    _write_json(output / "validation_report.json", {"issues": validation})
    errors = [issue for issue in validation if issue["severity"] == "error"]
    if errors:
        raise ValidationError(
            f"Cross-tester analysis stopped: {len(errors)} validation error(s); "
            f"see {output / 'validation_report.csv'}"
        )
    participant_rows = _aggregate_participant_conditions(trial_rows)
    participants = sorted({row["tester_id"] for row in participant_rows})
    matrix = [
        [
            _participant_value(participant_rows, tester_id, condition, PRIMARY_METRIC)
            for condition in CONDITIONS
        ]
        for tester_id in participants
    ]
    statistic, p_value, kendalls_w = _permutation_friedman(
        matrix,
        permutations=permutations,
        rng=random.Random(seed),
    )
    primary = {
        "metric": PRIMARY_METRIC,
        "aggregation": "median_across_three_recorded_trials",
        "participants": len(participants),
        "conditions": list(CONDITIONS),
        "statistic": statistic,
        "p_value": p_value,
        "kendalls_w": kendalls_w,
        "permutations": permutations,
        "seed": seed,
        "alpha": alpha,
        "significant": p_value < alpha,
    }
    pairwise = []
    if primary["significant"]:
        raw_comparisons = []
        bootstrap_rng = random.Random(seed + 1)
        for comparison_index, condition in enumerate(CONDITIONS[1:]):
            feedback = [
                _participant_value(
                    participant_rows, tester_id, condition, PRIMARY_METRIC
                )
                for tester_id in participants
            ]
            control = [
                _participant_value(
                    participant_rows, tester_id, "no_feedback", PRIMARY_METRIC
                )
                for tester_id in participants
            ]
            differences = [
                feedback_value - control_value
                for feedback_value, control_value in zip(feedback, control)
            ]
            wilcoxon = _wilcoxon_signed_rank(
                differences,
                permutations=permutations,
                rng=random.Random(seed + 3 + comparison_index),
            )
            ci_low, ci_high = _bootstrap_median_ci(
                differences,
                resamples=bootstrap_resamples,
                rng=bootstrap_rng,
            )
            raw_comparisons.append({
                "comparison": f"{condition} vs no_feedback",
                "condition": condition,
                "control": "no_feedback",
                "n_pairs": len(differences),
                "n_nonzero_pairs": wilcoxon["n_nonzero_pairs"],
                "wilcoxon_method": wilcoxon["method"],
                "wilcoxon_statistic": wilcoxon["statistic"],
                "raw_p_value": wilcoxon["p_value"],
                "rank_biserial_correlation": wilcoxon[
                    "rank_biserial_correlation"
                ],
                "paired_median_difference_n": statistics.median(differences),
                "bootstrap_ci_low_n": ci_low,
                "bootstrap_ci_high_n": ci_high,
                "difference_asymmetry": _pearson_asymmetry(differences),
                "sign_test_p_value": _exact_sign_test(differences),
            })
        adjusted = _holm_adjust(
            [comparison["raw_p_value"] for comparison in raw_comparisons]
        )
        for comparison, adjusted_p in zip(raw_comparisons, adjusted):
            comparison["holm_adjusted_p_value"] = adjusted_p
            comparison["significant"] = adjusted_p < alpha
        pairwise = raw_comparisons

    _write_csv(output / "participant_condition_summary.csv", participant_rows)
    primary_conditions = _write_primary_condition_summary(output, participant_rows)
    _write_secondary_condition_summary(
        output,
        participant_rows,
        bootstrap_resamples=bootstrap_resamples,
        rng=random.Random(seed + 2),
    )
    survival_rows = _completion_survival_rows(trial_rows)
    _write_csv(output / "completion_time_survival.csv", survival_rows)
    _write_completion_condition_summary(output, survival_rows)
    _write_completion_survival_plot(
        output / "completion_time_survival_plot.png",
        survival_rows,
    )
    design = _write_design_diagnostics(output, trial_rows, len(participants))
    _write_json(output / "primary_statistics.json", primary)
    _write_csv(
        output / "primary_pairwise_comparisons.csv",
        pairwise,
        fieldnames=(
            "comparison",
            "condition",
            "control",
            "n_pairs",
            "n_nonzero_pairs",
            "wilcoxon_method",
            "wilcoxon_statistic",
            "raw_p_value",
            "holm_adjusted_p_value",
            "significant",
            "rank_biserial_correlation",
            "paired_median_difference_n",
            "bootstrap_ci_low_n",
            "bootstrap_ci_high_n",
            "difference_asymmetry",
            "sign_test_p_value",
        ),
    )
    _write_connected_dot_plot(output / "primary_connected_dot_plot.png", participant_rows)
    _write_secondary_participant_plot(
        output / "secondary_participant_plots.png",
        participant_rows,
    )
    _write_report(
        output / "report.md",
        participants=len(participants),
        primary=primary,
        primary_conditions=primary_conditions,
        pairwise=pairwise,
        design=design,
    )
    return {
        "participants": len(participants),
        "primary": primary,
        "pairwise": pairwise,
        "design": design,
    }


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
                for threshold_key in threshold_values:
                    value = _float_or_blank(row.get(threshold_key, ""))
                    if math.isfinite(value):
                        threshold_values[threshold_key].add(value)
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


def _permutation_friedman(matrix, *, permutations, rng):
    observed = _friedman_statistic(matrix)
    n = len(matrix)
    k = len(matrix[0]) if matrix else 0
    if not matrix or observed == 0.0:
        return observed, 1.0, 0.0

    extreme = 0
    for _ in range(permutations):
        permuted = []
        for row in matrix:
            shuffled = list(row)
            rng.shuffle(shuffled)
            permuted.append(shuffled)
        if _friedman_statistic(permuted) >= observed - 1e-12:
            extreme += 1
    p_value = (extreme + 1) / (permutations + 1)
    kendalls_w = observed / (n * (k - 1))
    return observed, p_value, kendalls_w


def _friedman_statistic(matrix):
    if not matrix:
        return 0.0
    n = len(matrix)
    k = len(matrix[0])
    ranked = [_average_ranks(row) for row in matrix]
    rank_sums = [sum(row[column] for row in ranked) for column in range(k)]
    statistic = (
        12.0 * sum(value * value for value in rank_sums) / (n * k * (k + 1))
        - 3.0 * n * (k + 1)
    )
    tie_sum = 0
    for row in matrix:
        counts = {}
        for value in row:
            counts[value] = counts.get(value, 0) + 1
        tie_sum += sum(count ** 3 - count for count in counts.values())
    correction = 1.0 - tie_sum / (n * (k ** 3 - k))
    return 0.0 if correction == 0 else statistic / correction


def _average_ranks(values):
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[ordered[position][0]] = average
        start = end
    return ranks


def _wilcoxon_signed_rank(differences, *, permutations, rng):
    rounded = [round(float(value), 12) for value in differences]
    nonzero = [value for value in rounded if value != 0.0]
    if not nonzero:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "rank_biserial_correlation": 0.0,
            "n_nonzero_pairs": 0,
            "method": "all_zero",
        }
    ranks = _average_ranks([abs(value) for value in nonzero])
    positive = sum(rank for rank, value in zip(ranks, nonzero) if value > 0)
    negative = sum(rank for rank, value in zip(ranks, nonzero) if value < 0)
    observed_distance = abs(positive - (positive + negative) / 2.0)
    if len(ranks) <= 20:
        extreme = 0
        total = 1 << len(ranks)
        for mask in range(total):
            permuted_positive = sum(
                rank for index, rank in enumerate(ranks) if mask & (1 << index)
            )
            distance = abs(permuted_positive - (positive + negative) / 2.0)
            if distance >= observed_distance - 1e-12:
                extreme += 1
        p_value = extreme / total
        method = "exact_sign_flip"
    else:
        extreme = 0
        rank_total = positive + negative
        for _ in range(permutations):
            permuted_positive = sum(rank for rank in ranks if rng.random() < 0.5)
            distance = abs(permuted_positive - rank_total / 2.0)
            if distance >= observed_distance - 1e-12:
                extreme += 1
        p_value = (extreme + 1) / (permutations + 1)
        method = "monte_carlo_sign_flip"
    return {
        "statistic": min(positive, negative),
        "p_value": p_value,
        "rank_biserial_correlation": (positive - negative) / (positive + negative),
        "n_nonzero_pairs": len(nonzero),
        "method": method,
    }


def _holm_adjust(p_values):
    count = len(p_values)
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * count
    running_max = 0.0
    for rank, (original_index, p_value) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * p_value)
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
    return adjusted


def _bootstrap_median_ci(differences, *, resamples, rng):
    if not differences:
        return math.nan, math.nan
    estimates = []
    for _ in range(resamples):
        sample = [rng.choice(differences) for _ in differences]
        estimates.append(statistics.median(sample))
    return _quantile(estimates, 0.025), _quantile(estimates, 0.975)


def _bootstrap_statistic_ci(values, *, statistic, resamples, rng):
    if not values:
        return "", ""
    estimates = []
    for _ in range(resamples):
        sample = [rng.choice(values) for _ in values]
        estimates.append(statistic(sample))
    return _quantile(estimates, 0.025), _quantile(estimates, 0.975)


def _pearson_asymmetry(values):
    if len(values) < 3:
        return 0.0
    spread = statistics.pstdev(values)
    if spread == 0.0:
        return 0.0
    return 3.0 * (statistics.mean(values) - statistics.median(values)) / spread


def _exact_sign_test(differences):
    nonzero = [value for value in differences if value != 0]
    count = len(nonzero)
    if count == 0:
        return 1.0
    positive = sum(value > 0 for value in nonzero)
    tail = min(positive, count - positive)
    probability = sum(math.comb(count, value) for value in range(tail + 1))
    return min(1.0, 2.0 * probability / (2 ** count))


def _write_primary_condition_summary(output, participant_rows):
    rows = []
    for condition in CONDITIONS:
        values = [
            row[PRIMARY_METRIC]
            for row in participant_rows
            if row["condition"] == condition
        ]
        rows.append({
            "condition": condition,
            "n_participants": len(values),
            "median": statistics.median(values),
            "q1": _quantile(values, 0.25),
            "q3": _quantile(values, 0.75),
        })
    _write_csv(output / "primary_condition_summary.csv", rows)
    return rows


def _write_secondary_condition_summary(
    output,
    participant_rows,
    *,
    bootstrap_resamples,
    rng,
):
    rows = []
    for metric in SECONDARY_METRICS:
        for condition in CONDITIONS:
            values = [
                row[metric]
                for row in participant_rows
                if row["condition"] == condition and math.isfinite(row[metric])
            ]
            ci_low, ci_high = _bootstrap_statistic_ci(
                values,
                statistic=statistics.median,
                resamples=bootstrap_resamples,
                rng=rng,
            )
            rows.append({
                "analysis_role": "exploratory",
                "metric": metric,
                "condition": condition,
                "n_participants": len(values),
                "median": statistics.median(values) if values else "",
                "q1": _quantile(values, 0.25) if values else "",
                "q3": _quantile(values, 0.75) if values else "",
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
            })
    _write_csv(output / "secondary_condition_summary.csv", rows)


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


def _write_completion_condition_summary(output, survival_rows):
    rows = []
    for condition in CONDITIONS:
        condition_rows = [
            row for row in survival_rows if row["condition"] == condition
        ]
        successful_times = [
            row["duration_s"] for row in condition_rows if row["event"] == 1
        ]
        points = _kaplan_meier_curve(condition_rows)
        km_median = next(
            (time for time, probability in points if probability <= 0.5),
            "",
        )
        successes = sum(row["event"] for row in condition_rows)
        rows.append({
            "condition": condition,
            "n_trials": len(condition_rows),
            "successes": successes,
            "timeouts_or_failures": len(condition_rows) - successes,
            "success_rate": successes / len(condition_rows) if condition_rows else "",
            "median_successful_completion_time_s": (
                statistics.median(successful_times) if successful_times else ""
            ),
            "kaplan_meier_median_completion_time_s": km_median,
        })
    _write_csv(output / "completion_time_condition_summary.csv", rows)


def _kaplan_meier_curve(rows):
    at_risk = len(rows)
    probability = 1.0
    points = [(0.0, probability)]
    grouped = {}
    for row in rows:
        grouped.setdefault(float(row["duration_s"]), []).append(row)
    for time in sorted(grouped):
        tied = grouped[time]
        events = sum(int(row["event"]) for row in tied)
        if at_risk and events:
            probability *= 1.0 - events / at_risk
        points.append((time, probability))
        at_risk -= len(tied)
    return points


def _write_completion_survival_plot(path, survival_rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    for condition in CONDITIONS:
        rows = [row for row in survival_rows if row["condition"] == condition]
        points = _kaplan_meier_curve(rows)
        axis.step(
            [point[0] for point in points],
            [point[1] for point in points],
            where="post",
            label=condition.replace("_", " "),
        )
    axis.set_xlabel("Wall-clock time (s)")
    axis.set_ylabel("Probability task remains incomplete")
    axis.set_ylim(-0.02, 1.02)
    axis.set_title("Completion-time distributions with failures censored")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_design_diagnostics(output, trial_rows, participant_count):
    participant_orders = {}
    for row in trial_rows:
        participant_orders.setdefault(
            row["tester_id"],
            tuple(part.strip() for part in row["condition_order"].split("->")),
        )
    counts = {order: 0 for order in WILLIAMS_ORDERS}
    unexpected = {}
    for order in participant_orders.values():
        if order in counts:
            counts[order] += 1
        else:
            unexpected[order] = unexpected.get(order, 0) + 1
    order_rows = [
        {
            "order_index": index,
            "condition_order": " -> ".join(order),
            "participant_count": counts[order],
            "is_williams_order": 1,
        }
        for index, order in enumerate(WILLIAMS_ORDERS)
    ]
    order_rows.extend({
        "order_index": "unexpected",
        "condition_order": " -> ".join(order),
        "participant_count": count,
        "is_williams_order": 0,
    } for order, count in sorted(unexpected.items()))
    _write_csv(output / "condition_order_balance.csv", order_rows)

    pattern_rows = []
    for grouping_name, grouping_key, grouping_values in (
        ("trial_index", "trial_index", (1, 2, 3)),
        ("condition_position", "condition_position", (1, 2, 3, 4)),
    ):
        for condition in CONDITIONS:
            for grouping_value in grouping_values:
                values = [
                    row[PRIMARY_METRIC]
                    for row in trial_rows
                    if row["condition"] == condition
                    and _int_or_zero(row.get(grouping_key)) == grouping_value
                    and math.isfinite(row[PRIMARY_METRIC])
                ]
                pattern_rows.append({
                    "analysis_role": "descriptive",
                    "grouping": grouping_name,
                    "group_value": grouping_value,
                    "condition": condition,
                    "metric": PRIMARY_METRIC,
                    "n_trials": len(values),
                    "median": statistics.median(values) if values else "",
                    "q1": _quantile(values, 0.25) if values else "",
                    "q3": _quantile(values, 0.75) if values else "",
                })
    _write_csv(output / "trial_and_order_patterns.csv", pattern_rows)

    williams_counts = list(counts.values())
    next_checkpoint = max(12, math.ceil(participant_count / 4) * 4)
    if participant_count >= 12 and participant_count % 4 == 0:
        next_checkpoint = participant_count
    design = {
        "participant_count": participant_count,
        "williams_order_counts": williams_counts,
        "unexpected_order_count": sum(unexpected.values()),
        "balanced_as_possible": (
            not unexpected and max(williams_counts) - min(williams_counts) <= 1
        ),
        "balanced_at_three_per_order": (
            not unexpected and participant_count == 12 and williams_counts == [3] * 4
        ),
        "next_balanced_recruitment_checkpoint": next_checkpoint,
    }
    _write_json(output / "design_diagnostics.json", design)
    return design


def _write_secondary_participant_plot(path, participant_rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    x_values = list(range(len(CONDITIONS)))
    for axis, metric in zip(axes.flat, SECONDARY_METRICS):
        by_participant = {}
        for row in participant_rows:
            by_participant.setdefault(row["tester_id"], {})[row["condition"]] = row[
                metric
            ]
        for values in by_participant.values():
            axis.plot(
                x_values,
                [values[condition] for condition in CONDITIONS],
                marker="o",
                linewidth=0.9,
                alpha=0.55,
            )
        axis.set_xticks(
            x_values,
            ["none", "visual", "audio", "both"],
            rotation=20,
        )
        axis.set_title(metric.replace("_", " "))
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Exploratory participant-level secondary outcomes")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_connected_dot_plot(path, participant_rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_participant = {}
    for row in participant_rows:
        by_participant.setdefault(row["tester_id"], {})[row["condition"]] = row[
            PRIMARY_METRIC
        ]
    figure, axis = plt.subplots(figsize=(9, 5.5))
    x_values = list(range(len(CONDITIONS)))
    for tester_id, values in sorted(by_participant.items()):
        axis.plot(
            x_values,
            [values[condition] for condition in CONDITIONS],
            marker="o",
            linewidth=1.2,
            alpha=0.7,
            label=tester_id,
        )
    axis.set_xticks(x_values, [condition.replace("_", " ") for condition in CONDITIONS])
    axis.set_ylabel("Participant median peak ground-truth contact force (N)")
    axis.set_title("Primary safety outcome by participant and condition")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_report(
    path,
    *,
    participants,
    primary,
    primary_conditions,
    pairwise,
    design,
):
    lines = [
        "# Cross-Tester Statistical Analysis",
        "",
        f"Participants included: **{participants}**",
        "",
        "## Confirmatory primary outcome",
        "",
        f"- Metric: `{primary['metric']}` (participant median across three trials)",
        f"- Permutation Friedman statistic: {primary['statistic']:.6g}",
        f"- Permutation p-value: {primary['p_value']:.6g}",
        f"- Kendall's W: {primary['kendalls_w']:.6g}",
        f"- Monte Carlo permutations: {primary['permutations']}",
        f"- Reproducibility seed: {primary['seed']}",
        f"- Decision at α={primary['alpha']}: "
        + ("significant" if primary["significant"] else "not significant"),
        "",
        "| Condition | n | Median (N) | IQR (N) |",
        "|---|---:|---:|---:|",
    ]
    for condition in primary_conditions:
        lines.append(
            f"| {condition['condition']} "
            f"| {condition['n_participants']} "
            f"| {condition['median']:.4g} "
            f"| [{condition['q1']:.4g}, {condition['q3']:.4g}] |"
        )
    lines.extend([
        "",
        "Negative paired differences and rank-biserial correlations mean the "
        "feedback condition produced lower peak force than no feedback.",
        "",
    ])
    if not pairwise:
        lines.extend([
            "The omnibus test was not significant, so no confirmatory post-hoc "
            "comparisons were run.",
            "",
        ])
    else:
        lines.extend([
            "## Pre-specified post-hoc comparisons",
            "",
            "| Comparison | Raw p | Holm p | Rank-biserial r | Median difference (N) | 95% bootstrap CI (N) |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for comparison in pairwise:
            lines.append(
                f"| {comparison['comparison']} "
                f"| {comparison['raw_p_value']:.6g} "
                f"| {comparison['holm_adjusted_p_value']:.6g} "
                f"| {comparison['rank_biserial_correlation']:.4g} "
                f"| {comparison['paired_median_difference_n']:.4g} "
                f"| [{comparison['bootstrap_ci_low_n']:.4g}, "
                f"{comparison['bootstrap_ci_high_n']:.4g}] |"
            )
        lines.append("")
        asymmetric = [
            comparison
            for comparison in pairwise
            if abs(comparison["difference_asymmetry"]) > 1.0
        ]
        if asymmetric:
            lines.extend([
                "Paired differences were highly asymmetric (absolute Pearson "
                "asymmetry > 1) for: "
                + ", ".join(item["comparison"] for item in asymmetric)
                + ". Use the reported exact sign-test p-values as sensitivity analyses.",
                "",
            ])
    lines.extend([
        "## Design and exploratory outputs",
        "",
        "- Williams-order participant counts: "
        + ", ".join(str(value) for value in design["williams_order_counts"]),
        f"- Balanced as far as the current sample permits: "
        f"{'yes' if design['balanced_as_possible'] else 'no'}",
        f"- Next balanced recruitment checkpoint: "
        f"{design['next_balanced_recruitment_checkpoint']} participants",
        "- Completion failures are represented as censored observations; the "
        "successful-completion median excludes them.",
        "- Secondary metrics and trial/order patterns are exploratory and must not "
        "be promoted to confirmatory findings based on their observed values.",
        "",
        "See `secondary_condition_summary.csv`, `completion_time_condition_summary.csv`, "
        "`condition_order_balance.csv`, and `trial_and_order_patterns.csv` for details.",
        "",
    ])
    path.write_text("\n".join(lines))


def _write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else ()
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _float_or_blank(value):
    if value in ("", None):
        return math.nan
    return float(value)


def _int_or_zero(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _median_numeric(values):
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.median(finite) if finite else math.nan


def _quantile(values, probability):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run participant-level confirmatory and exploratory analyses across "
            "active tester experiment results."
        )
    )
    parser.add_argument(
        "--experiment-results-dir",
        type=Path,
        default=Path("experiment_results"),
        help="Root containing active per-tester experiment folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Report directory. Defaults to "
            "<experiment-results-dir>/cross_tester_analysis."
        ),
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=100_000,
        help="Monte Carlo permutations for the Friedman omnibus test.",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=10_000,
        help="Participant-level bootstrap resamples for confidence intervals.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260725,
        help="Random seed for reproducible permutation and bootstrap results.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Confirmatory family-wise significance level.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.permutations <= 0:
        raise SystemExit("--permutations must be positive")
    if args.bootstrap_resamples <= 0:
        raise SystemExit("--bootstrap-resamples must be positive")
    if not 0.0 < args.alpha < 1.0:
        raise SystemExit("--alpha must be between 0 and 1")
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else args.experiment_results_dir / "cross_tester_analysis"
    )
    try:
        result = run_analysis(
            args.experiment_results_dir,
            output_dir,
            permutations=args.permutations,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed,
            alpha=args.alpha,
        )
    except ValidationError as error:
        raise SystemExit(str(error)) from error
    print(f"Analyzed {result['participants']} participants.")
    print(f"Saved cross-tester report to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
