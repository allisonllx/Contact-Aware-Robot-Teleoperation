# Odyssey: Occluded Peg-in-Hole Teleoperation

Odyssey is a MuJoCo user study of a simulated Franka Emika Panda robot. A tester
uses the keyboard to insert a peg into a hole whose position is hidden by an
occluder. Trials compare whether visual force guidance, audio force guidance,
both, or no guidance helps the tester complete the task.

The main study randomizes the hidden hole position and records task, timing, and
contact-force data. The repository also contains smaller force-verification
scenarios used during development.

## Project summary

| Topic | Summary |
| --- | --- |
| Task | Insert a downward-facing peg into an occluded hole |
| Control | Discrete keyboard movement in X, Y, and Z |
| Guidance modes | No feedback, visual, audio, and visual + audio |
| Study structure | 1 familiarization trial, then 3 measured trials per mode (2.5 min each) |
| Main command | `python experiment.py --tester firstname_lastname` |
| Output | Per-tester results + trial videos under `experiment_results/`, zipped when the session finishes |

## Documentation

| Guide | Use it for |
| --- | --- |
| [Tester guide](docs/tester-guide.md) | Study background, the complete tester workflow, controls, and the one experiment command |
| [Setup guide](docs/setup.md) | Installing Python dependencies and checking the MuJoCo environment |
| [Technical reference](docs/technical-reference.md) | Scenario variants, CLI options, feedback details, recording, analysis, outputs, and implementation notes |

## Quick start

Set up the environment by following the [setup guide](docs/setup.md), then run:

```bash
python experiment.py --tester firstname_lastname
```

Replace `firstname_lastname` with your ID (for example `jane_doe`). On each
prompt, press Enter when the tester is ready, click the MuJoCo window, and use
the controls in the [tester guide](docs/tester-guide.md).

The experiment runner creates a counterbalanced plan the first time an ID is
used. Running the same command again resumes that tester and skips completed
trials. When every trial is finished, it writes
`experiment_results/<tester_id>.zip` and prints the path so you can send the
results back. On macOS, the runner automatically uses `mjpython` for the
interactive MuJoCo trial processes when it is available.
