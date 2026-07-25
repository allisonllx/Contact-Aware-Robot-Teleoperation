"""Compatibility wrapper for the packaged cross-tester analysis."""

from stat_analysis.cross_tester import ValidationError, main, parse_args, run_analysis

__all__ = ("ValidationError", "main", "parse_args", "run_analysis")


if __name__ == "__main__":
    main()
