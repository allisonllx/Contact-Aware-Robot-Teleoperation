"""Participant-facing terminal briefing and audio calibration."""

import sys

from franka_force.audio import AudioFeedback
from franka_force.config import (
    DEFAULT_AUDIO_CONTACT_THRESHOLD,
    DEFAULT_AUDIO_LATERAL_MAX,
    DEFAULT_AUDIO_LATERAL_THRESHOLD,
    DEFAULT_AUDIO_VOLUME,
)


def confirm_tester_briefing():
    print("\n" + "=" * 72)
    print("TESTER BRIEFING")
    print("=" * 72)
    print("Task: Find the hidden hole and insert the peg using gentle contact.")
    print()
    print("Aims:")
    print("  1. Keep contact forces low. If the peg jams, back off and adjust.")
    print("  2. Complete the insertion quickly without sacrificing gentle contact.")
    print()
    print("The four feedback modes:")
    print("  - No feedback     : no visual or audio force guidance")
    print("  - Visual feedback : force arrow and contact ring")
    print("  - Audio feedback  : contact click and force-sensitive ticking")
    print("  - Visual + audio  : both forms of guidance")
    print("Turn on the computer volume or wear earphones for the audio cues.")
    print()
    print("The six main control keys (one press moves 5 mm):")
    print("  - Up arrow        : move north")
    print("  - Down arrow      : move south")
    print("  - Left arrow      : move west")
    print("  - Right arrow     : move east")
    print("  - 9               : raise the peg")
    print("  - 8               : lower the peg")
    print()
    print("Keep the default front-on camera view; do not move behind the wall.")
    print("If anything above is unclear, read docs/tester-guide.md before continuing.")
    print("Ask the study organizer any questions before acknowledging.")
    while True:
        acknowledgement = input("\nType YES to confirm that you understand the task: ").strip()
        if acknowledgement.casefold() == "yes":
            print("Acknowledged. The experiment will now begin.")
            return
        print("Acknowledgement not received. Read the tester guide, then type YES.")


def run_audio_calibration():
    """Preview the standard audio cues before the non-recorded familiarization."""
    print("\nAudio calibration: listen for one contact click, then ticks that speed up.")
    print("Faster ticks mean higher lateral force; back off and readjust when they are fast.")
    audio = AudioFeedback(
        mode="both",
        contact_threshold=DEFAULT_AUDIO_CONTACT_THRESHOLD,
        lateral_threshold=DEFAULT_AUDIO_LATERAL_THRESHOLD,
        lateral_max=DEFAULT_AUDIO_LATERAL_MAX,
        volume=DEFAULT_AUDIO_VOLUME,
    )
    try:
        if not audio.play_calibration_preview():
            print("Audio preview unavailable; check that system audio is configured.")
    finally:
        audio.close()
    print("Audio calibration complete. The familiarization trial is not recorded in the main results.")


def bold_terminal_text(text):
    if hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
        return f"\033[1m{text}\033[0m"
    return text
