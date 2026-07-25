
import csv
import json
import math
from pathlib import Path

from .io import (
    and_masks,
    bool_column,
    count_true,
    duration,
    empty_summary,
    finite_median,
    first_nonempty,
    first_time,
    float_column,
    max_or_blank,
    mean_or_blank,
    optional_bool_column,
    optional_float_column,
    optional_text_column,
    safe_divide,
    sample_widths,
    select,
    select_log_path,
    select_log_path_from_dir,
    weighted_duration,
    weighted_sum,
)
from .metrics import count_episodes, error_metrics, smoothness_metrics
from .schemas import (
    DEFAULT_FORCE_THRESHOLD_N,
    DEFAULT_JAMMING_THRESHOLD_N,
    EPS,
    FILTERED_LOG_NAME,
    TRIAL_OUTCOME_NAME,
)


def analyze_scenario(
    scenario,
    results_dir,
    source,
    force_threshold,
    include_anomalies,
    jamming_threshold=DEFAULT_JAMMING_THRESHOLD_N,
):
    log_path = select_log_path(results_dir, scenario, source)
    if log_path is None:
        return empty_summary(
            scenario,
            status="missing_csv",
            force_threshold=force_threshold,
            jamming_threshold=jamming_threshold,
        )

    return analyze_log_file(
        log_path=log_path,
        scenario=scenario,
        force_threshold=force_threshold,
        jamming_threshold=jamming_threshold,
        include_anomalies=include_anomalies,
    )

def analyze_result_dir(
    result_dir,
    scenario,
    source="auto",
    force_threshold=DEFAULT_FORCE_THRESHOLD_N,
    jamming_threshold=DEFAULT_JAMMING_THRESHOLD_N,
    include_anomalies=False,
):
    result_dir = Path(result_dir)
    log_path = select_log_path_from_dir(result_dir, source)
    if log_path is None:
        summary = empty_summary(
            scenario,
            status="missing_csv",
            force_threshold=force_threshold,
            jamming_threshold=jamming_threshold,
        )
    else:
        summary = analyze_log_file(
            log_path=log_path,
            scenario=scenario,
            force_threshold=force_threshold,
            jamming_threshold=jamming_threshold,
            include_anomalies=include_anomalies,
        )
    summary.update(read_trial_outcome_metrics(result_dir))
    if summary.get("completion_time_wall_s") in ("", None):
        summary["completion_time_wall_s"] = summary.get("wall_time_elapsed_s", "")
    return summary

def analyze_log_file(
    log_path,
    scenario,
    force_threshold,
    include_anomalies,
    jamming_threshold=DEFAULT_JAMMING_THRESHOLD_N,
):
    with log_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    summary = empty_summary(
        scenario,
        status="ok",
        force_threshold=force_threshold,
        jamming_threshold=jamming_threshold,
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
    occluder_alpha = optional_float_column(rows, columns, "Occluder Alpha", math.nan)
    occluder_style = optional_text_column(rows, columns, "Occluder Style")

    fx = optional_float_column(rows, columns, "Contact Force X (N)", 0.0)
    fy = optional_float_column(rows, columns, "Contact Force Y (N)", 0.0)
    # Always derive lateral from contact Fx/Fy. Audio Lateral Force is 0 when
    # audio feedback is off, so it must not be preferred for study metrics.
    lateral = [math.hypot(x, y) for x, y in zip(fx, fy)]
    jamming_mask = [
        clean and value >= jamming_threshold
        for clean, value in zip(is_clean, lateral)
    ]
    smoothness = smoothness_metrics(
        times=times,
        is_clean=is_clean,
        in_contact=in_contact,
        target_x=optional_float_column(rows, columns, "Target X (m)", math.nan),
        target_y=optional_float_column(rows, columns, "Target Y (m)", math.nan),
        target_z=optional_float_column(rows, columns, "Target Z (m)", math.nan),
    )

    peak_proxy_values = select(f_est, is_clean) or select(f_true, is_clean)
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
        "occluder_alpha": finite_median(occluder_alpha),
        "occluder_style": first_nonempty(occluder_style),
        "audio_feedback_enabled": int(any(audio_feedback)),
        "cushion_used": int(any(cushion_active)),
        "mean_ground_truth_contact_n": mean_or_blank(select(f_true, contact_clean)),
        "mean_estimate_contact_n": mean_or_blank(select(f_est, contact_clean)),
        "peak_ground_truth_contact_n": max_or_blank(select(f_true, contact_clean)),
        "peak_estimate_contact_n": max_or_blank(select(f_est, contact_clean)),
        "peak_contact_proxy_n": max_or_blank(peak_proxy_values),
        "peak_lateral_force_n": max_or_blank(select(lateral, is_clean)),
        "time_above_threshold_s": weighted_duration(
            dt,
            [clean and true >= force_threshold for clean, true in zip(is_clean, f_true)],
        ),
        "time_above_jamming_s": weighted_duration(dt, jamming_mask),
        "jamming_count": count_episodes(jamming_mask),
        "contact_episode_count": count_episodes(contact_clean),
        "contact_force_impulse_n_s": weighted_sum(f_true, dt, contact_clean),
    })
    summary.update(smoothness)
    summary.update(error_metrics(f_true, f_est, contact_clean, "contact"))
    summary.update(error_metrics(f_true, f_est, is_clean, "all_clean"))
    return summary

def read_trial_outcome_metrics(result_dir):
    outcome_path = Path(result_dir) / TRIAL_OUTCOME_NAME
    if not outcome_path.exists():
        return {
            "wall_time_elapsed_s": "",
            "timed_out": "",
            "completion_time_wall_s": "",
        }
    try:
        with outcome_path.open() as f:
            outcome = json.load(f)
    except (OSError, ValueError):
        return {
            "wall_time_elapsed_s": "",
            "timed_out": "",
            "completion_time_wall_s": "",
        }
    wall = outcome.get("wall_time_elapsed_s", "")
    return {
        "wall_time_elapsed_s": wall,
        "timed_out": int(bool(outcome.get("timed_out", False))),
        "completion_time_wall_s": wall,
    }
