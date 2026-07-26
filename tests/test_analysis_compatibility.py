import csv
import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import analysis
import cross_tester_analysis
import stat_analysis
from stat_analysis.cross_tester import run_analysis as packaged_run_analysis
from stat_analysis.force_estimation import write_force_estimation_report


class AnalysisCompatibilityTests(unittest.TestCase):
    def test_sample_peg_in_hole_metrics_remain_stable(self):
        summary = analysis.analyze_result_dir(
            Path("sample_results/peg_in_hole"),
            scenario="peg_in_hole",
        )

        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["samples_total"], 172)
        self.assertEqual(summary["samples_contact_clean"], 101)
        self.assertAlmostEqual(summary["duration_s"], 10.320000000000118)
        self.assertAlmostEqual(
            summary["peak_ground_truth_contact_n"],
            957.3517000237696,
        )
        self.assertAlmostEqual(summary["peak_contact_proxy_n"], 709.2628834370877)
        self.assertAlmostEqual(summary["contact_duration_s"], 6.119999999999859)
        self.assertAlmostEqual(summary["mae_contact_n"], 24.329188141197033)

    def test_legacy_modules_expose_repository_public_interfaces(self):
        for name in (
            "SUMMARY_COLUMNS",
            "aggregate_condition_rows",
            "analyze_result_dir",
            "csv_value",
            "print_condition_summary",
            "write_condition_summary",
        ):
            self.assertTrue(hasattr(analysis, name), name)
        self.assertIsNotNone(cross_tester_analysis.ValidationError)
        self.assertTrue(callable(cross_tester_analysis.run_analysis))
        self.assertIs(analysis.analyze_result_dir, stat_analysis.analyze_result_dir)
        self.assertIs(cross_tester_analysis.run_analysis, packaged_run_analysis)

    def test_legacy_cli_help_commands_remain_available(self):
        for script, expected in (
            ("analysis.py", "--force-estimation-report"),
            ("cross_tester_analysis.py", "--bootstrap-resamples"),
        ):
            completed = subprocess.run(
                [sys.executable, script, "--help"],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(expected, completed.stdout)

    def test_per_tester_summary_still_uses_recorded_completed_trials(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp) / "tester_a"
            trial_dir = experiment_dir / "visual_feedback" / "recorded_01"
            trial_dir.mkdir(parents=True)
            for name in (
                "force_verification_log.csv",
                "force_verification_log_filtered.csv",
            ):
                shutil.copy(Path("sample_results/peg_in_hole") / name, trial_dir / name)
            (trial_dir / "trial_metadata.json").write_text(json.dumps({
                "tester": "tester_a",
                "tester_id": "tester_a",
                "condition": "visual_feedback",
                "condition_order": [
                    "no_feedback",
                    "visual_feedback",
                    "audio_feedback",
                    "both_feedback",
                ],
                "condition_position": 2,
                "trial_type": "recorded",
                "trial_index": 1,
                "occluded_hole_seed": 123,
                "visual_feedback": True,
                "audio_feedback": False,
                "scenario": "peg_in_hole",
                "status": "completed",
            }))

            with contextlib.redirect_stdout(io.StringIO()):
                conditions = analysis.summarize_experiment_dir(experiment_dir)

            self.assertEqual(len(conditions), 1)
            self.assertEqual(conditions[0]["condition"], "visual_feedback")
            self.assertEqual(conditions[0]["n_trials"], 1)
            with (experiment_dir / "experiment_analysis_summary.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["tester_id"], "tester_a")

    def test_force_estimation_report_excludes_archived_tester_trials(self):
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "experiment_results"
            for tester, condition in (
                ("active_tester", "visual_feedback"),
                ("_archive/previous_tester", "audio_feedback"),
            ):
                trial_dir = results_dir / tester / condition / "recorded_01"
                trial_dir.mkdir(parents=True)
                for name in (
                    "force_verification_log.csv",
                    "force_verification_log_filtered.csv",
                ):
                    shutil.copy(
                        Path("sample_results/peg_in_hole") / name,
                        trial_dir / name,
                    )
                (trial_dir / "trial_metadata.json").write_text(json.dumps({
                    "status": "completed",
                }))

            output_dir = Path(tmp) / "force_estimation_report"
            with contextlib.redirect_stdout(io.StringIO()):
                write_force_estimation_report(
                    output_dir,
                    experiment_results_dir=results_dir,
                )

            with (output_dir / "force_estimation_per_run.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertIn("active_tester", rows[0]["run_id"])
            self.assertNotIn("_archive", rows[0]["run_id"])
            self.assertTrue(
                (output_dir / "plots" / "estimate_vs_ground_truth.png").is_file()
            )
            self.assertTrue(
                (output_dir / "plots" / "estimate_vs_ground_truth_0_150n.png").is_file()
            )


if __name__ == "__main__":
    unittest.main()
