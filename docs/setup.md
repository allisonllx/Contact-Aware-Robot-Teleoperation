# Setup Guide

## Requirements

- macOS, Linux, or Windows with Python 3
- A keyboard with arrow keys and number keys
- Working speakers or headphones for audio-feedback trials
- Git installed (`git --version` should print a version)
- A terminal

The experiment itself is launched with `python` on every operating system. On
macOS, the experiment runner automatically uses `mjpython` for the interactive
MuJoCo trial processes. `mjpython` is installed with the `mujoco` Python
package.

## Clone the repository

In a terminal, clone the project and enter its folder:

```bash
git clone https://github.com/allisonllx/Contact-Aware-Robot-Teleoperation.git
cd Contact-Aware-Robot-Teleoperation
```

All later setup commands assume your terminal is in this repository root.

## Create the environment

Follow the instructions for your operating system.

### macOS or Linux

1. Create a virtual environment:

   ```bash
   python3 -m venv .venv
   ```

2. Activate the environment:

   ```bash
   source .venv/bin/activate
   ```

3. Install the project dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Windows PowerShell

1. Create a virtual environment:

   ```powershell
   python -m venv .venv
   ```

   This command creates the `.venv` folder and its activation script locally.
   The activation script is generated on the tester's computer; it is not a
   file included in the repository.

2. After the first command finishes, activate the environment:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the project dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

When activation succeeds, PowerShell or the terminal will usually show
`(.venv)` at the start of the command prompt.

The dependencies include MuJoCo, NumPy, Matplotlib, keyboard input support, and
video utilities.

## Fetch the Franka model

The Franka Emika Panda assets may not bundled with this repository. From the
repository root, clone only the Panda folder from MuJoCo Menagerie:

```bash
git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git
cd mujoco_menagerie
git sparse-checkout set franka_emika_panda
cd ..
```

When this succeeds, the scene file should exist at
`mujoco_menagerie/franka_emika_panda/scene.xml`. Skip this step if that path is
already present.

## Pre-study checklist

Before starting a tester session:

1. Confirm you cloned the repository and your terminal is in its root folder.
2. Activate the virtual environment.
3. Confirm that the Franka scene file exists at
   `mujoco_menagerie/franka_emika_panda/scene.xml`.
4. Confirm that the activated environment's Python command works:

   ```bash
   python --version
   ```

5. Make sure the computer's audio is audible.
6. Give the MuJoCo window keyboard focus after it opens.

The experiment command is the same on macOS, Linux, and Windows. On macOS, the
runner selects `mjpython` automatically when it launches each interactive trial.

Continue with the [tester guide](tester-guide.md) to run the study.

## Common setup problems

### Missing `mujoco_menagerie/franka_emika_panda/scene.xml`

Run the [Fetch the Franka model](#fetch-the-franka-model) steps from the
repository root. A full Menagerie clone is not required; the sparse checkout
above downloads only the Panda assets.

If `git clone` fails because `mujoco_menagerie` already exists, remove the empty
folder and retry:

```bash
rm -rf mujoco_menagerie
```

Then run the fetch steps again from the repository root. On Windows PowerShell,
use `Remove-Item -Recurse -Force mujoco_menagerie` instead of `rm -rf`.

### A macOS trial reports `mjpython: command not found`

Activate the virtual environment and confirm that `pip install -r
requirements.txt` completed successfully. Linux and Windows trials use
`python`, so they do not require the `mjpython` command.

### The robot does not respond to keys

Click once inside the MuJoCo viewer. The study uses discrete movement, so press
and release a key for each 5 mm movement rather than holding it down.

### No audio is heard

Some conditions intentionally have no audio. In an audio or combined condition,
check the system volume and output device. Audio cues appear when contact force
crosses their thresholds, so silence away from contact is expected.

### Red boxes or axes appear

The `I`, `J`, `K`, and `U` keys toggle MuJoCo debug views. Avoid these keys
during the study. If one is pressed accidentally, tell the study operator before
continuing.
