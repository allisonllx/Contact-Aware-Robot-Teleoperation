
import random
import statistics
from pathlib import Path

from ..schemas import CONDITIONS, PRIMARY_METRIC
from .aggregation import (
    _aggregate_participant_conditions,
    _completion_survival_rows,
    _participant_value,
)
from .reporting import (
    _write_completion_condition_summary,
    _write_completion_survival_plot,
    _write_connected_dot_plot,
    _write_csv,
    _write_design_diagnostics,
    _write_json,
    _write_primary_condition_summary,
    _write_report,
    _write_secondary_condition_summary,
    _write_secondary_participant_plot,
)
from .statistics import (
    _bootstrap_median_ci,
    _exact_sign_test,
    _holm_adjust,
    _pearson_asymmetry,
    _permutation_friedman,
    _wilcoxon_signed_rank,
)
from .validation import ValidationError, _read_active_trial_rows


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
