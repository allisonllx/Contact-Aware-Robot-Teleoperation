"""Stable statistical analysis APIs for Odyssey experiments."""

from .cli import main, parse_args
from .force_estimation import (
    aggregate_force_estimation_rows,
    discover_force_estimation_runs,
    discover_tester_pool_runs,
    find_force_comparison_png,
    format_mean_std,
    is_finite_number,
    plot_force_estimation_bars,
    print_force_estimation_report,
    std_or_blank,
    write_force_estimation_exemplars,
    write_force_estimation_report,
)
from .io import (
    and_masks,
    bool_column,
    count_true,
    csv_value,
    display,
    duration,
    empty_summary,
    finite_median,
    first_nonempty,
    first_time,
    float_column,
    max_or_blank,
    mean,
    mean_or_blank,
    median,
    optional_bool_column,
    optional_float_column,
    optional_text_column,
    percentile,
    print_summary,
    read_json,
    safe_divide,
    sample_widths,
    select,
    select_log_path,
    select_log_path_from_dir,
    weighted_duration,
    weighted_sum,
    write_summary,
)
from .metrics import (
    count_episodes,
    count_retractions,
    count_sign_changes,
    error_metrics,
    finite_differences,
    smoothness_metrics,
)
from .schemas import (
    CONDITION_SUMMARY_COLUMNS,
    EPS,
    EXPERIMENT_RESULTS_DIR,
    FILTERED_LOG_NAME,
    FORCE_COMPARISON_CANDIDATES,
    FORCE_ESTIMATION_DIR,
    FORCE_ESTIMATION_SCENARIOS,
    FORCE_EST_BY_SCENARIO_COLUMNS,
    FORCE_EST_ERROR_KEYS,
    FORCE_EST_PER_RUN_COLUMNS,
    RAW_LOG_NAME,
    SUMMARY_COLUMNS,
    TRIAL_METADATA_NAME,
    TRIAL_OUTCOME_NAME,
)
from .summaries import (
    aggregate_condition_rows,
    mean_numeric,
    print_condition_summary,
    summarize_experiment_dir,
    write_condition_summary,
    write_rows,
)
from .telemetry import (
    analyze_log_file,
    analyze_result_dir,
    analyze_scenario,
    read_trial_outcome_metrics,
)

__all__ = tuple(
    name
    for name in globals()
    if not name.startswith("_") and name not in {"annotations"}
)
