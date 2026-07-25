
import csv
import json
import math
import statistics
from pathlib import Path

from ..schemas import (
    CONDITIONS,
    PRIMARY_METRIC,
    SECONDARY_METRICS,
    WILLIAMS_ORDERS,
)
from .statistics import _bootstrap_statistic_ci, _quantile
from .validation import _int_or_zero


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
