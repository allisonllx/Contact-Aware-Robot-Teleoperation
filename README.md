# Odyssey Project: Robot Haptic Feedback

MuJoCo simulations for checking contact-force estimates on a Franka Emika Panda arm. The project runs a small set of scenarios, logs ground-truth contact forces against Jacobian-based estimates, and writes comparison plots for each run.

## Scenarios

- `push_block`: moves the gripper into a free block and compares block contact force against the virtual force estimate.
- `hit_floor`: lowers the gripper toward the floor and compares floor contact force against the virtual force estimate.
- `peg_in_hole`: adds a peg, socket, IK target, and optional keyboard teleoperation for insertion practice.

## Setup

Create and activate a Python environment, then install the requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The Franka model is loaded from:

```text
mujoco_menagerie/franka_emika_panda/scene.xml
```

On macOS, MuJoCo viewer workflows that use the passive viewer, including interactive peg control and video recording, may need `mjpython` instead of regular `python3`.

## Usage

Run the default scenario:

```bash
python3 main.py
```

Run a specific scenario:

```bash
python3 main.py --scenario push_block
python3 main.py --scenario hit_floor
python3 main.py --scenario peg_in_hole
```

Run the interactive peg-in-hole task:

```bash
mjpython main.py --scenario peg_in_hole --interactive
```

Enable live force feedback during interactive peg insertion:

```bash
mjpython main.py --scenario peg_in_hole --interactive --force-feedback
```

Record a video of a run:

```bash
mjpython main.py --scenario push_block --record-video
mjpython main.py --scenario peg_in_hole --interactive --record-video
```

Show CLI options:

```bash
python3 main.py --help
```

## Outputs

Each run writes artifacts under `results/<scenario>/`:

- `force_verification_log.csv`: raw force samples.
- `force_verification_log_filtered.csv`: samples after anomaly filtering.
- `force_comparison_raw.png`: all measured and estimated force samples.
- `force_comparison_filtered.png`: filtered measured and estimated force samples.
- `force_comparison_contact_only_raw.png`: contact-only raw comparison.
- `force_comparison_contact_only_filtered.png`: contact-only filtered comparison.
- `run_recording.mp4`: video output when `--record-video` is used.

Reference outputs are checked in under `sample_results/`.

## Project Layout

```text
main.py                         CLI entrypoint
franka_force/config.py          Paths, scenario names, video defaults
franka_force/env.py             Shared MuJoCo environment and viewer orchestration
franka_force/recording.py       Offscreen MP4 recording helper
franka_force/plotting.py        CSV filtering and plot generation
franka_force/scenarios/         Scenario-specific model, control, and contact logic
```

`FrankaForceEnv` delegates scenario-specific behavior through the scenario registry in `franka_force/scenarios/__init__.py`, so adding a new scenario should usually mean adding one scenario module and registering it there.

## Development Checks

Compile the Python files:

```bash
python3 -m compileall main.py franka_force
```

Check the command-line interface without launching MuJoCo:

```bash
python3 main.py --help
```
