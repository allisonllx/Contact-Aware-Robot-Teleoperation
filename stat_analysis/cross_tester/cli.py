
import argparse
from pathlib import Path

from .validation import ValidationError
from .workflow import run_analysis


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
