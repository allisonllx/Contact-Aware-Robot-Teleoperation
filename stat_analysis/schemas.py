
from pathlib import Path

from franka_force.config import (  # pylint: disable=unused-import
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
