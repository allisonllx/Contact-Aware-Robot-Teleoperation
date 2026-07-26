"""Compatibility entry point for the packaged experiment runner."""

from experiment_runner.briefing import (  # pylint: disable=unused-import
    bold_terminal_text,
    confirm_tester_briefing,
    run_audio_calibration,
)
from experiment_runner.cli import (  # pylint: disable=unused-import
    parse_args,
    validate_args,
    validate_range,
)
from experiment_runner.execution import (  # pylint: disable=unused-import
    build_trial_command,
    read_trial_outcome,
    run_trial,
    trial_metadata,
    trial_python,
)
from experiment_runner.planning import (  # pylint: disable=unused-import
    CONDITIONS,
    assign_counterbalanced_order,
    build_trial_specs,
    candidate_orders,
    ensure_selected_conditions_in_plan,
    load_or_create_plan,
    sanitize_tester_name,
    stable_int,
    trial_completed,
    trial_seed,
    trial_state,
    trial_status_for_display,
)
from experiment_runner.results import (  # pylint: disable=unused-import
    experiment_analysis_row,
    read_json_if_exists,
    utc_now,
    write_experiment_summary,
    write_experiment_summaries,
    write_json,
    zip_tester_results,
)
from experiment_runner.workflow import (  # pylint: disable=unused-import
    condition_flags_for_display,
    main,
    print_experiment_overview,
    print_progress_summary,
    print_resume_note,
    wait_for_trial,
)


if __name__ == "__main__":
    main()
