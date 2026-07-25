import csv
import json
import tempfile
import unittest
from pathlib import Path

from cross_tester_analysis import ValidationError, run_analysis


CONDITIONS = (
    "no_feedback",
    "visual_feedback",
    "audio_feedback",
    "both_feedback",
)


def write_tester(root, tester_id, values_by_condition):
    tester_dir = root / tester_id
    tester_dir.mkdir(parents=True)
    (tester_dir / "experiment_plan.json").write_text(json.dumps({
        "tester": tester_id,
        "tester_id": tester_id,
        "conditions": list(CONDITIONS),
        "condition_order": list(CONDITIONS),
        "recorded_trials": 3,
    }))

    rows = []
    for condition in CONDITIONS:
        for trial_index, value in enumerate(values_by_condition[condition], start=1):
            trial_dir = tester_dir / condition / f"recorded_{trial_index:02d}"
            trial_dir.mkdir(parents=True)
            (trial_dir / "trial_metadata.json").write_text(json.dumps({
                "tester": tester_id,
                "tester_id": tester_id,
                "condition": condition,
                "condition_order": list(CONDITIONS),
                "condition_position": CONDITIONS.index(condition) + 1,
                "trial_type": "recorded",
                "trial_index": trial_index,
                "status": "completed",
                "task_success": True,
                "timed_out": False,
                "wall_time_elapsed_s": 20.0 + trial_index,
            }))
            rows.append({
                "tester": tester_id,
                "tester_id": tester_id,
                "condition": condition,
                "condition_order": " -> ".join(CONDITIONS),
                "condition_position": CONDITIONS.index(condition) + 1,
                "trial_type": "recorded",
                "trial_index": trial_index,
                "trial_dir": str(trial_dir),
                "status": "ok",
                "force_threshold_n": 100,
                "jamming_threshold_n": 50,
                "peak_ground_truth_contact_n": value,
                "peak_contact_proxy_n": value + 5,
                "task_success": 1,
                "completion_time_wall_s": 20.0 + trial_index,
                "timed_out": 0,
                "jamming_count": 1,
                "time_above_threshold_s": 0.5,
                "contact_force_impulse_n_s": 10,
                "mean_action_jerk": 2,
                "velocity_reversals": 1,
                "retraction_count": 0,
            })

    with (tester_dir / "experiment_analysis_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class CrossTesterAnalysisTests(unittest.TestCase):
    def test_equal_conditions_generate_participant_level_report_without_post_hoc_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiment_results"
            output = Path(tmp) / "report"
            for participant in range(4):
                values = {
                    condition: [100 + participant] * 3
                    for condition in CONDITIONS
                }
                write_tester(root, f"tester_{participant}", values)

            result = run_analysis(
                root,
                output,
                permutations=200,
                bootstrap_resamples=200,
                seed=7,
            )

            self.assertEqual(result["participants"], 4)
            self.assertEqual(result["primary"]["p_value"], 1.0)
            self.assertEqual(result["pairwise"], [])
            with (output / "participant_condition_summary.csv").open(newline="") as f:
                self.assertEqual(len(list(csv.DictReader(f))), 16)
            self.assertTrue((output / "primary_connected_dot_plot.png").is_file())
            self.assertTrue((output / "report.md").is_file())

    def test_known_condition_effect_runs_three_holm_corrected_control_comparisons(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiment_results"
            output = Path(tmp) / "report"
            offsets = {
                "no_feedback": 200,
                "visual_feedback": 160,
                "audio_feedback": 150,
                "both_feedback": 120,
            }
            for participant in range(8):
                values = {
                    condition: [base + participant - 1, base + participant, base + participant + 1]
                    for condition, base in offsets.items()
                }
                write_tester(root, f"tester_{participant}", values)

            result = run_analysis(
                root,
                output,
                permutations=999,
                bootstrap_resamples=200,
                seed=11,
            )

            self.assertTrue(result["primary"]["significant"])
            self.assertEqual(len(result["pairwise"]), 3)
            self.assertEqual(
                {row["comparison"] for row in result["pairwise"]},
                {
                    "visual_feedback vs no_feedback",
                    "audio_feedback vs no_feedback",
                    "both_feedback vs no_feedback",
                },
            )
            self.assertTrue(all(row["holm_adjusted_p_value"] <= 0.05 for row in result["pairwise"]))
            self.assertTrue(all(row["rank_biserial_correlation"] == -1.0 for row in result["pairwise"]))

    def test_missing_planned_trial_is_reported_and_stops_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiment_results"
            output = Path(tmp) / "report"
            values = {condition: [100, 101, 102] for condition in CONDITIONS}
            write_tester(root, "incomplete_tester", values)
            metadata = (
                root
                / "incomplete_tester"
                / "visual_feedback"
                / "recorded_03"
                / "trial_metadata.json"
            )
            metadata.unlink()

            with self.assertRaises(ValidationError):
                run_analysis(root, output, permutations=20, bootstrap_resamples=20)

            with (output / "validation_report.csv").open(newline="") as f:
                issues = list(csv.DictReader(f))
            self.assertIn("missing_trial_metadata", {row["code"] for row in issues})
            self.assertFalse((output / "primary_statistics.json").exists())

    def test_inconsistent_force_thresholds_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiment_results"
            output = Path(tmp) / "report"
            values = {condition: [100, 101, 102] for condition in CONDITIONS}
            write_tester(root, "tester_a", values)
            summary_path = root / "tester_a" / "experiment_analysis_summary.csv"
            with summary_path.open(newline="") as f:
                rows = list(csv.DictReader(f))
                fieldnames = list(rows[0])
            rows[-1]["force_threshold_n"] = "90"
            with summary_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaises(ValidationError):
                run_analysis(root, output, permutations=20, bootstrap_resamples=20)

            with (output / "validation_report.csv").open(newline="") as f:
                codes = {row["code"] for row in csv.DictReader(f)}
            self.assertIn("inconsistent_force_threshold", codes)

    def test_timeout_is_censored_and_excluded_from_successful_completion_median(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiment_results"
            output = Path(tmp) / "report"
            values = {condition: [100, 101, 102] for condition in CONDITIONS}
            write_tester(root, "tester_a", values)
            summary_path = root / "tester_a" / "experiment_analysis_summary.csv"
            with summary_path.open(newline="") as f:
                rows = list(csv.DictReader(f))
                fieldnames = list(rows[0])
            timed_out_row = next(
                row
                for row in rows
                if row["condition"] == "audio_feedback" and row["trial_index"] == "2"
            )
            timed_out_row["task_success"] = "0"
            timed_out_row["timed_out"] = "1"
            timed_out_row["completion_time_wall_s"] = "150"
            with summary_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            metadata_path = (
                root / "tester_a" / "audio_feedback" / "recorded_02" / "trial_metadata.json"
            )
            metadata = json.loads(metadata_path.read_text())
            metadata.update(task_success=False, timed_out=True, wall_time_elapsed_s=150)
            metadata_path.write_text(json.dumps(metadata))

            run_analysis(root, output, permutations=20, bootstrap_resamples=20)

            with (output / "completion_time_survival.csv").open(newline="") as f:
                survival = list(csv.DictReader(f))
            censored = next(
                row
                for row in survival
                if row["condition"] == "audio_feedback" and row["trial_index"] == "2"
            )
            self.assertEqual(censored["event"], "0")
            self.assertEqual(float(censored["duration_s"]), 150.0)
            with (output / "completion_time_condition_summary.csv").open(newline="") as f:
                condition_rows = list(csv.DictReader(f))
            audio = next(row for row in condition_rows if row["condition"] == "audio_feedback")
            self.assertAlmostEqual(float(audio["success_rate"]), 2 / 3)
            self.assertEqual(float(audio["median_successful_completion_time_s"]), 22.0)

    def test_wilcoxon_reports_ties_and_zero_differences_without_approximation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiment_results"
            output = Path(tmp) / "report"
            for participant in range(12):
                control = 200 + participant
                visual = control if participant < 2 else control - (10 if participant < 7 else 20)
                values = {
                    "no_feedback": [control] * 3,
                    "visual_feedback": [visual] * 3,
                    "audio_feedback": [control - 30] * 3,
                    "both_feedback": [control - 40] * 3,
                }
                write_tester(root, f"tester_{participant}", values)

            result = run_analysis(
                root,
                output,
                permutations=999,
                bootstrap_resamples=100,
                seed=19,
            )

            visual = next(
                row
                for row in result["pairwise"]
                if row["condition"] == "visual_feedback"
            )
            self.assertEqual(visual["n_nonzero_pairs"], 10)
            self.assertEqual(visual["wilcoxon_method"], "exact_sign_flip")
            self.assertGreaterEqual(visual["raw_p_value"], 0.0)
            self.assertLessEqual(visual["raw_p_value"], 1.0)

    def test_duplicate_summary_trial_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiment_results"
            output = Path(tmp) / "report"
            values = {condition: [100, 101, 102] for condition in CONDITIONS}
            write_tester(root, "tester_a", values)
            summary_path = root / "tester_a" / "experiment_analysis_summary.csv"
            with summary_path.open(newline="") as f:
                rows = list(csv.DictReader(f))
                fieldnames = list(rows[0])
            rows.append(dict(rows[0]))
            with summary_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaises(ValidationError):
                run_analysis(root, output, permutations=20, bootstrap_resamples=20)

            with (output / "validation_report.csv").open(newline="") as f:
                codes = {row["code"] for row in csv.DictReader(f)}
            self.assertIn("duplicate_trial", codes)

    def test_non_completed_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiment_results"
            output = Path(tmp) / "report"
            values = {condition: [100, 101, 102] for condition in CONDITIONS}
            write_tester(root, "tester_a", values)
            metadata_path = (
                root / "tester_a" / "both_feedback" / "recorded_03" / "trial_metadata.json"
            )
            metadata = json.loads(metadata_path.read_text())
            metadata["status"] = "interrupted"
            metadata_path.write_text(json.dumps(metadata))

            with self.assertRaises(ValidationError):
                run_analysis(root, output, permutations=20, bootstrap_resamples=20)

            with (output / "validation_report.csv").open(newline="") as f:
                codes = {row["code"] for row in csv.DictReader(f)}
            self.assertIn("incomplete_trial", codes)

    def test_archived_tester_directories_are_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiment_results"
            output = Path(tmp) / "report"
            values = {condition: [100, 101, 102] for condition in CONDITIONS}
            write_tester(root, "active_tester", values)
            write_tester(root / "archive", "archived_tester", values)
            archived_metadata = (
                root
                / "archive"
                / "archived_tester"
                / "visual_feedback"
                / "recorded_03"
                / "trial_metadata.json"
            )
            archived_metadata.unlink()

            result = run_analysis(
                root,
                output,
                permutations=20,
                bootstrap_resamples=20,
            )

            self.assertEqual(result["participants"], 1)
            with (output / "participant_condition_summary.csv").open(newline="") as f:
                testers = {row["tester_id"] for row in csv.DictReader(f)}
            self.assertEqual(testers, {"active_tester"})


if __name__ == "__main__":
    unittest.main()
