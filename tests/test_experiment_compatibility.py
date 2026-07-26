import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import experiment


class ExperimentCompatibilityTests(unittest.TestCase):
    def test_dry_run_keeps_the_public_command_and_plan_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiment_results"
            completed = subprocess.run(
                [
                    sys.executable,
                    "experiment.py",
                    "--tester",
                    "compatibility_tester",
                    "--experiment-root",
                    str(root),
                    "--dry-run",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("DRY RUN: OCCLUDED PEG-IN-HOLE EXPERIMENT", completed.stdout)
            self.assertTrue((root / "compatibility_tester" / "experiment_plan.json").is_file())

    def test_familiarization_seed_keeps_its_legacy_seed_namespace(self):
        args = SimpleNamespace(familiarization_trials=1, practice_trials=0, recorded_trials=1)
        plan = {
            "base_seed": 123,
            "generated_seed_base": None,
            "condition_order": ["no_feedback"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            trials = experiment.build_trial_specs(
                args,
                "Compatibility Tester",
                "compatibility_tester",
                Path(tmp),
                plan,
                ["no_feedback"],
            )

        self.assertEqual(
            trials[0]["seed"],
            experiment.trial_seed(
                123,
                "compatibility_tester",
                "familiarization",
                "familiarization",
                1,
            ),
        )


if __name__ == "__main__":
    unittest.main()
