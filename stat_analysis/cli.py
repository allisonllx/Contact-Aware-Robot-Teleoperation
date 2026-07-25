
import argparse
from pathlib import Path

from franka_force.config import (
    DEFAULT_FORCE_THRESHOLD_N,
    DEFAULT_JAMMING_THRESHOLD_N,
    RESULTS_DIR,
    SCENARIOS,
)

from .force_estimation import write_force_estimation_report
from .io import print_summary, write_summary
from .schemas import EXPERIMENT_RESULTS_DIR, FORCE_ESTIMATION_DIR
from .summaries import summarize_experiment_dir
from .telemetry import analyze_scenario


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze force-estimation accuracy and task safety metrics."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory containing results/<scenario>/ telemetry folders.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="Analyze a tester folder under experiment_results/ and write condition summaries.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=SCENARIOS,
        help="Scenario names to analyze.",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "filtered", "raw"),
        default="auto",
        help="Use filtered logs, raw logs, or prefer filtered logs when available.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Summary CSV path. Defaults to <results-dir>/force_analysis_summary.csv.",
    )
    parser.add_argument(
        "--force-threshold",
        type=float,
        default=DEFAULT_FORCE_THRESHOLD_N,
        help="Ground-truth force threshold for safety-duration metrics.",
    )
    parser.add_argument(
        "--jamming-threshold",
        type=float,
        default=DEFAULT_JAMMING_THRESHOLD_N,
        help="Lateral-force threshold in newtons used to count jamming episodes.",
    )
    parser.add_argument(
        "--include-anomalies",
        action="store_true",
        help="Include rows flagged as anomalies when reading raw logs.",
    )
    parser.add_argument(
        "--force-estimation-report",
        action="store_true",
        help=(
            "Aggregate multi-run force-estimation accuracy under "
            "force_estimation_runs/ (MAE/MSE tables + plots)."
        ),
    )
    parser.add_argument(
        "--force-estimation-root",
        type=Path,
        default=FORCE_ESTIMATION_DIR,
        help="Root folder of scripted repeats: <root>/<scenario>/run_XX/.",
    )
    parser.add_argument(
        "--include-tester-pool",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also ingest experiment_results/** peg_in_hole trial logs as "
            "source=tester (default: on)."
        ),
    )
    parser.add_argument(
        "--experiment-results-dir",
        type=Path,
        default=EXPERIMENT_RESULTS_DIR,
        help="Tester pool root used with --force-estimation-report.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    if args.force_threshold <= 0.0:
        raise ValueError("--force-threshold must be positive")
    if args.jamming_threshold <= 0.0:
        raise ValueError("--jamming-threshold must be positive")

    if args.force_estimation_report:
        write_force_estimation_report(
            root=args.force_estimation_root,
            source=args.source,
            force_threshold=args.force_threshold,
            jamming_threshold=args.jamming_threshold,
            include_anomalies=args.include_anomalies,
            include_tester_pool=args.include_tester_pool,
            experiment_results_dir=args.experiment_results_dir,
        )
        return

    if args.experiment_dir is not None:
        summarize_experiment_dir(
            args.experiment_dir,
            source=args.source,
            force_threshold=args.force_threshold,
            jamming_threshold=args.jamming_threshold,
            include_anomalies=args.include_anomalies,
        )
        return

    output_path = args.output or args.results_dir / "force_analysis_summary.csv"
    rows = [
        analyze_scenario(
            scenario=scenario,
            results_dir=args.results_dir,
            source=args.source,
            force_threshold=args.force_threshold,
            jamming_threshold=args.jamming_threshold,
            include_anomalies=args.include_anomalies,
        )
        for scenario in args.scenarios
    ]

    write_summary(output_path, rows)
    print_summary(rows, output_path)
