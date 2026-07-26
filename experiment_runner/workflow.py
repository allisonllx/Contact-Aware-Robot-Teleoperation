"""Interactive experiment-session orchestration and terminal presentation."""

import shlex

from .briefing import bold_terminal_text, confirm_tester_briefing, run_audio_calibration
from .cli import parse_args, validate_args
from .execution import build_trial_command, run_trial
from .planning import (
    build_trial_specs,
    ensure_selected_conditions_in_plan,
    load_or_create_plan,
    sanitize_tester_name,
    trial_completed,
    trial_state,
    trial_status_for_display,
)
from .results import write_experiment_summaries, zip_tester_results


def main():
    """Run one tester's experiment session using the established CLI."""
    args = parse_args()
    validate_args(args)
    tester_name = args.tester.strip() if args.tester else input("Tester name: ").strip()
    tester_id = sanitize_tester_name(tester_name)
    tester_dir = args.experiment_root / tester_id
    tester_dir.mkdir(parents=True, exist_ok=True)
    selected_conditions = tuple(dict.fromkeys(args.conditions))
    plan = load_or_create_plan(
        tester_name=tester_name,
        tester_id=tester_id,
        tester_dir=tester_dir,
        experiment_root=args.experiment_root,
        selected_conditions=selected_conditions,
        manual_order=tuple(args.order) if args.order else None,
        familiarization_trials=args.familiarization_trials,
        practice_trials=args.practice_trials,
        recorded_trials=args.recorded_trials,
        max_trial_duration=args.max_trial_duration,
        base_seed=args.base_seed,
    )
    ensure_selected_conditions_in_plan(selected_conditions, plan)
    trial_specs = build_trial_specs(
        args, tester_name, tester_id, tester_dir, plan, selected_conditions
    )
    print_experiment_overview(tester_name, tester_dir, plan, trial_specs, args.dry_run, args)
    if args.dry_run:
        print("\nDry run only; no MuJoCo windows launched.")
        return
    confirm_tester_briefing()
    run_audio_calibration()
    interrupted = False
    try:
        for trial in trial_specs:
            state = trial_state(trial["trial_dir"])
            if state["complete"] and not args.rerun_existing:
                print(f"\nSkipping completed trial: {trial['trial_dir']}")
                continue
            print_resume_note(trial, state, args.rerun_existing)
            wait_for_trial(trial, args)
            run_trial(args, plan, trial)
    except KeyboardInterrupt:
        interrupted = True
        print("\nExperiment interrupted. Rerun the same tester command to resume at the first incomplete trial.")
    finally:
        write_experiment_summaries(
            tester_dir,
            tester_name,
            tester_id,
            plan,
            trial_specs,
            args.force_threshold,
            jamming_threshold=args.jamming_threshold,
            trial_completed=trial_completed,
        )
    if interrupted:
        raise SystemExit(130)
    if all(trial_completed(trial["trial_dir"]) for trial in trial_specs):
        zip_path = zip_tester_results(tester_dir)
        print("\nExperiment complete. Send this zip file:")
        print(f"  {zip_path}")
    else:
        print("\nExperiment not fully complete yet. Rerun the same tester command to resume.")


def print_experiment_overview(tester_name, tester_dir, plan, trial_specs, dry_run, args):
    """Print the planned session and each trial's resume status."""
    mode = "DRY RUN" if dry_run else "LIVE RUN"
    print(f"\n=== {mode}: OCCLUDED PEG-IN-HOLE EXPERIMENT ===")
    print(f"Tester: {tester_name}")
    print(f"Output folder: {tester_dir.resolve()}")
    print(f"Condition order: {' -> '.join(plan['condition_order'])}")
    practice_note = f", plus {args.practice_trials} practice per condition" if args.practice_trials else ""
    print(
        f"Structure: {args.familiarization_trials} familiarization (no feedback), then "
        f"{args.recorded_trials} recorded trials per condition{practice_note}."
    )
    print(
        f"Trial time limit: {args.max_trial_duration:.0f}s wall clock."
        if args.max_trial_duration > 0.0
        else "Trial time limit: disabled."
    )
    print(f"Video recording: {'ON' if args.record_video else 'OFF'}.")
    if plan.get("loaded_existing_plan"):
        print("Using existing experiment_plan.json.")
    print_progress_summary(trial_specs)
    for trial in trial_specs:
        flags = condition_flags_for_display(trial)
        state = trial_state(trial["trial_dir"])
        label = (
            f"familiarization / {trial['trial_name']}"
            if trial["trial_type"] == "familiarization"
            else f"{trial['condition_position']}. {trial['condition']} / {trial['trial_name']}"
        )
        print(f"{label} / seed={trial['seed']} / {flags} / {trial_status_for_display(state)}")
        print(f"   {trial['trial_dir']}")
        if dry_run:
            print(f"   {shlex.join(build_trial_command(args, trial))}")


def condition_flags_for_display(trial):
    """Format a trial's enabled feedback modalities for the terminal."""
    if trial["trial_type"] == "familiarization":
        return "no feedback (familiarization)"
    flags = []
    if trial["visual_feedback"]:
        flags.append("visual")
    if trial["audio_feedback"]:
        flags.append("audio")
    return "+".join(flags) if flags else "no feedback"


def print_progress_summary(trial_specs):
    """Print aggregate completion and incomplete-trial counts."""
    states = [trial_state(trial["trial_dir"]) for trial in trial_specs]
    completed = sum(1 for state in states if state["complete"])
    print(f"Progress: {completed}/{len(states)} completed.")
    labels = {
        "interrupted": "interrupted",
        "failed": "failed",
        "started": "started/incomplete",
    }
    details = [
        f"{sum(1 for state in states if state['status'] == status)} {label}"
        for status, label in labels.items()
        if any(state["status"] == status for state in states)
    ]
    if details:
        print("Resume state: " + ", ".join(details) + ".")


def print_resume_note(trial, state, rerun_existing):
    """Explain why a trial will be repeated or resumed."""
    if state["complete"] and rerun_existing:
        print(f"\nRerunning completed trial because --rerun-existing is set: {trial['trial_dir']}")
    elif state["status"] != "not_started" or state["has_telemetry"]:
        print(
            f"\nResuming at incomplete trial: {trial['trial_dir']} "
            f"({trial_status_for_display(state)})."
        )
        print("Existing partial outputs in this trial folder will be overwritten by the rerun.")


def wait_for_trial(trial, args):
    """Show one trial's instructions and wait for the facilitator's confirmation."""
    print("\n" + "-" * 72)
    if trial["trial_type"] == "familiarization":
        print(bold_terminal_text(f"Next trial: familiarization / {trial['trial_name']} (no feedback)"))
        print("This is a one-time practice run to learn the controls.")
    else:
        print(bold_terminal_text(f"Next trial: {trial['condition']} / {trial['trial_name']}"))
    print(f"Output: {trial['trial_dir']}")
    print(f"Feedback: {condition_flags_for_display(trial)}")
    if args.max_trial_duration > 0.0:
        print(f"Time limit: {args.max_trial_duration:.0f}s. The MuJoCo window closes automatically on success or timeout.")
    else:
        print("No time limit. The MuJoCo window closes automatically on success.")
    print("Press Enter when the tester is ready.")
    input()
