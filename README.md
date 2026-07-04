# Odyssey Project: Contact-Aware Robot Teleoperation

MuJoCo simulations for contact-aware teleoperation on a Franka Emika Panda arm. The project mainly focuses on a `peg_in_hole` insertion task, with `push_block` and `hit_floor` as smaller side tasks for checking contact behavior.

The goal is to investigate whether Jacobian-based force estimation is accurate against MuJoCo ground-truth contact forces, and whether live force feedback in the MuJoCo environment helps users carry out contact-rich tasks more safely and faster. Feedback can be visual, audio, or both. Each run logs measured contact forces against estimated forces and writes comparison plots for later analysis.

## Scenarios

- `peg_in_hole`: the primary task. Adds a peg, socket, IK target, and keyboard teleoperation for insertion practice with visual and audio force-feedback experiments.
- `push_block`: a side task that moves the gripper into a free block and compares block contact force against the virtual force estimate. It can also be run with keyboard teleoperation to freely explore contact, with optional free end-effector orientation.
- `hit_floor`: a side task that lowers the gripper toward the floor and compares floor contact force against the virtual force estimate. It can also be run with keyboard teleoperation to freely explore contact, with optional free end-effector orientation.

## Setup

Activate your `odyssey` environment, where `python` and `mjpython` should resolve to the MuJoCo-capable environment. If you are setting up from scratch instead:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The Franka model is loaded from:

```text
mujoco_menagerie/franka_emika_panda/scene.xml
```

On macOS, MuJoCo viewer workflows that use the passive viewer, including interactive keyboard control and video recording, may need `mjpython` instead of regular `python3`.

## Usage

### Quick Summary

Use `python3` for plain scripted runs. Use `mjpython` on macOS for commands that open the MuJoCo viewer, use keyboard teleoperation, or record video.

| Goal | Example command |
| --- | --- |
| Run the default scripted task | `python3 main.py` |
| Run a side-task force check | `python3 main.py --scenario push_block` |
| Practice peg insertion manually | `mjpython main.py --scenario peg_in_hole --interactive` |
| Practice with visual feedback | `mjpython main.py --scenario peg_in_hole --interactive --force-feedback --force-visual both` |
| Add audio feedback | `mjpython main.py --scenario peg_in_hole --interactive --audio-feedback --audio-mode both` |
| Record a run | `mjpython main.py --scenario peg_in_hole --interactive --record-video` |

### Common Workflows

#### 1. Scripted force verification

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

Disable the scripted policy if you want a non-interactive scenario to load without carrying out its default motion:

```bash
python3 main.py --scenario push_block --disable-policy
python3 main.py --scenario hit_floor --disable-policy
```

#### 2. Interactive keyboard teleoperation

Run the main peg-in-hole task:

```bash
mjpython main.py --scenario peg_in_hole --interactive
```

Run side tasks with keyboard control:

```bash
mjpython main.py --scenario push_block --interactive
mjpython main.py --scenario hit_floor --interactive
```

Side-task teleoperation is position-only by default. Add free orientation when you want to rotate the end effector during contact exploration:

```bash
mjpython main.py --scenario push_block --interactive --free-orientation
mjpython main.py --scenario hit_floor --interactive --free-orientation
```

`peg_in_hole` intentionally does not use free orientation: the peg stays constrained to face downward, and `6`/`7` only spin the peg about the vertical insertion axis.

#### 3. Visual force feedback

```bash
mjpython main.py --scenario peg_in_hole --interactive --force-feedback --force-visual both
mjpython main.py --scenario push_block --interactive --force-feedback --force-visual both
mjpython main.py --scenario hit_floor --interactive --force-feedback --force-visual both
```

`--force-visual` can be `arrow`, `ring`, or `both`. `arrow` draws a red/orange raw force vector at the strongest target contact point. Its direction is the measured MuJoCo contact-force direction in world coordinates, with a consistent sign on the robot/tool side of the contact. `ring` draws a red/orange ring at the selected contact surface. The overlay size uses a log scale from roughly `10 N` to `1000 N`, so mid-range forces remain visible without huge spikes dominating the view.

In the simulator, these visual overlays are contact-data based rather than Jacobian-estimate based. MuJoCo provides the contact point, contact frame, and contact force, so the ring can show where contact occurs and the arrow can show the raw contact-force vector. The Jacobian estimate is still logged and plotted for comparison, but by itself it gives an end-effector wrench estimate rather than a ground-truth contact point on the contacted surface.

For future physical robot runs, MuJoCo contact data will not be available. A real-system version should rely on the Jacobian/torque-based wrench estimate, force-torque sensing, or another contact-localization signal. Without tactile sensing, vision, proximity checks, or geometry-based inference, the real robot can show an estimated force vector at the end effector, but it cannot know the exact surface contact point in the same way the simulator can.

#### 4. Audio force feedback

```bash
mjpython main.py --scenario peg_in_hole --interactive --hole-clearance-mm 1.0 --audio-feedback --audio-mode both
mjpython main.py --scenario push_block --interactive --audio-feedback --audio-mode both
mjpython main.py --scenario hit_floor --interactive --audio-feedback --audio-mode both
```

`--audio-feedback` is live-only and works with all interactive scenarios. It uses dependency-free click/tick cues. `contact` mode plays a short click when the Jacobian estimate crosses `--audio-contact-threshold` (`2 N` by default). `geiger` mode maps lateral contact force to sparse ticking: silence below `--audio-lateral-threshold`, faster ticks as lateral resistance approaches `--audio-lateral-max`. `both` combines the contact click with Geiger ticks. This v1 does not generate continuous pitch or embed audio into `run_recording.mp4`; if macOS `afplay` is unavailable, the run continues silently with a warning.

#### 5. Peg-in-hole experiment variants

Run a tight-clearance peg-in-hole trial:

```bash
mjpython main.py --scenario peg_in_hole --interactive --hole-clearance-mm 0.5 --force-feedback
```

`--hole-clearance-mm` controls total peg/hole clearance. The default is `8.0 mm`, matching the original scene. Useful experiment values are `8.0`, `2.0`, `1.0`, and `0.5`; at sub-millimeter clearance, small lateral offsets become hard to diagnose visually, so force patterns become more informative.

Make the peg and socket walls semi-transparent when inspecting internal contacts:

```bash
mjpython main.py --scenario peg_in_hole --interactive --force-feedback --force-visual both --peg-alpha 0.45 --socket-alpha 0.45
```

Run the occluded peg-in-hole experiment:

```bash
mjpython main.py --scenario peg_in_hole --interactive --occluded-task --force-feedback --force-visual both --peg-alpha 0.45 --socket-alpha 0.45
```

`--occluded-task` adds an opaque visual-only wall in front of a slightly deeper, off-center socket, plus a hidden collision pad at the bottom of the hole. The wall blocks the user's line of sight without affecting physics, and the selected arrow/ring feedback is projected to a visible proxy position just in front of and above the wall. After sustained peg-pad contact, the run prints a success message, updates the HUD, records a final frame when video recording is enabled, and exits cleanly.

Enable the experimental impedance cushion during interactive peg insertion:

```bash
mjpython main.py --scenario peg_in_hole --interactive --force-feedback --force-visual both --contact-cushion
```

The cushion activates after contact force crosses `--cushion-threshold` (`100 N` by default). While active, the arm position servos are commanded to the current joint positions to cancel the servo spring, and a torque-limited Cartesian spring/damper is applied through `J.T @ wrench`. You can tune it with `--impedance-kp`, `--impedance-dp`, `--impedance-kr`, `--impedance-dr`, and `--impedance-torque-limit`.

#### 6. Video recording

```bash
mjpython main.py --scenario push_block --record-video
mjpython main.py --scenario push_block --record-video --record-force-feedback --force-visual both
mjpython main.py --scenario hit_floor --record-video --record-force-feedback --force-visual both
mjpython main.py --scenario peg_in_hole --interactive --record-video
mjpython main.py --scenario peg_in_hole --interactive --record-video --record-force-feedback --force-visual both
```

`--record-force-feedback` includes the same visual feedback geoms in the saved video: the green idle marker before contact, plus the selected red/orange arrow, ring, or both during contact. Videos are encoded at `30 fps` against simulation time, so the saved MP4 duration should track the simulation timeline rather than how fast the viewer/render loop happened to run.

When the occluded task starts, the live viewer camera is initialized to a wider front-on, near eye-level view that includes the starting peg and obstacle. Video recording uses a separate side/three-quarter observer camera for occluded trials, so the saved MP4 shows the obstacle, peg, and socket area rather than only the participant's blocked view.

Show CLI options:

```bash
python3 main.py --help
```

### Keyboard Controls

| Control | Meaning |
| --- | --- |
| Arrow keys | Move target in X/Y: north, south, east, west |
| `9` / `8` | Raise / lower target in Z |
| Page Up / Page Down | Also raise / lower target in Z, if your keyboard has them |
| `,` / `.` | Open / close gripper |
| `6` / `7` in `peg_in_hole` | Spin the downward-facing peg about the vertical insertion axis |
| `[` / `]` with `--free-orientation` | Pitch the end effector in side-task teleop |
| `-` / `=` with `--free-orientation` | Yaw the end effector in side-task teleop |
| `6` / `7` with `--free-orientation` | Roll the end effector in side-task teleop |

Avoid `I`, `J`, `K`, and `U` in the MuJoCo viewer because they toggle debug visualizations, not robot controls.

### Flag Reference

| Flag | Applies to | Purpose |
| --- | --- | --- |
| `--scenario` | All runs | Choose `peg_in_hole`, `push_block`, or `hit_floor`. |
| `--interactive` | All scenarios | Enable keyboard teleoperation and live viewer control. |
| `--free-orientation` | `push_block` / `hit_floor` with `--interactive` | Add pitch, yaw, and roll controls for side-task exploration. |
| `--disable-policy` | Non-interactive runs | Load the scenario without applying its scripted motion policy. |
| `--force-feedback` | Interactive runs | Show live visual force feedback. |
| `--force-visual` | Visual feedback and recorded feedback | Choose `arrow`, `ring`, or `both`. |
| `--audio-feedback` | Interactive runs | Enable live contact audio cues across all scenarios. |
| `--audio-mode` | Audio feedback | Choose `contact`, `geiger`, or `both`. |
| `--audio-contact-threshold` | Audio feedback | Set the Jacobian-estimate threshold for the contact click. |
| `--audio-lateral-threshold`, `--audio-lateral-max`, `--audio-volume` | Audio feedback | Tune Geiger ticking thresholds and cue volume. |
| `--record-video` | All scenarios | Save `run_recording.mp4` under `results/<scenario>/`. |
| `--record-force-feedback` | Video recording | Include force-feedback overlay geoms in the MP4. |
| `--hole-clearance-mm` | `peg_in_hole` | Set total peg/hole clearance in millimeters. |
| `--peg-alpha`, `--socket-alpha` | `peg_in_hole` | Adjust peg and socket opacity for inspection. |
| `--occluded-task` | `peg_in_hole` with `--interactive` | Hide the socket behind a visual wall for occlusion experiments. |
| `--contact-cushion` | `peg_in_hole` with `--interactive` | Enable the experimental reactive impedance cushion. |
| `--cushion-threshold`, `--impedance-*` | Contact cushion | Tune when the cushion activates and how stiff/damped it feels. |

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
franka_force/teleop.py          Shared keyboard teleoperation and IK target helpers
franka_force/force_visuals.py   Shared raw-force overlay drawing
franka_force/recording.py       Offscreen MP4 recording helper
franka_force/plotting.py        CSV filtering and plot generation
franka_force/scenarios/         Scenario-specific model, control, and contact logic
```

`FrankaForceEnv` delegates scenario-specific behavior through the scenario registry in `franka_force/scenarios/__init__.py`, so adding a new scenario should usually mean adding one scenario module and registering it there.

The CSV files also include cushion state, cushion scale, impedance torque norm, strongest contact-force vector components, occluded-task success state, hole clearance, and audio feedback state when enabled.

## Control Experiments

The `--contact-cushion` mode is reactive, not predictive: it only engages after MuJoCo reports contact force. Without a proximity sensor or a simulation-only distance check, the controller cannot slow before first contact.

The experimental cushion uses the impedance idea:

```text
tau_impedance = J.T @ (K * (X_target - X_current) - D * Xdot_current)
```

This first version keeps normal IK as the default and switches to torque-level impedance only after the force threshold is exceeded. It is useful for comparing force graphs with and without cushioning, but the gains should be treated as tuning parameters rather than final control values.

## Development Checks

Compile the Python files:

```bash
python3 -m compileall main.py franka_force
```

Check the command-line interface without launching MuJoCo:

```bash
python3 main.py --help
```
