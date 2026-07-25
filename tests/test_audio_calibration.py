import contextlib
import io
import unittest
from unittest.mock import Mock, patch

import experiment
from franka_force.audio import AudioFeedback
from franka_force.config import (
    DEFAULT_AUDIO_CONTACT_THRESHOLD,
    DEFAULT_AUDIO_LATERAL_MAX,
    DEFAULT_AUDIO_LATERAL_THRESHOLD,
    DEFAULT_AUDIO_VOLUME,
)


class AudioCalibrationTests(unittest.TestCase):
    def test_preview_plays_contact_then_ticks_from_slow_to_fast(self):
        feedback = AudioFeedback(
            mode="both",
            contact_threshold=2.0,
            lateral_threshold=5.0,
            lateral_max=200.0,
            volume=0.35,
            player_path="preview-player",
        )
        played = []
        try:
            with patch.object(feedback, "_play", side_effect=played.append), patch(
                "franka_force.audio.time.sleep"
            ) as sleep:
                available = feedback.play_calibration_preview()

            self.assertTrue(available)
            self.assertEqual(played[0], feedback._contact_path)
            self.assertTrue(all(path == feedback._tick_path for path in played[1:]))
            self.assertGreaterEqual(len(played), 4)
            intervals = [call.args[0] for call in sleep.call_args_list]
            self.assertEqual(intervals, sorted(intervals, reverse=True))
        finally:
            feedback.close()

    def test_experiment_calibration_uses_trial_audio_settings_without_recording_data(self):
        preview = Mock(return_value=True)
        controller = Mock(play_calibration_preview=preview)
        with patch("experiment.AudioFeedback", return_value=controller) as audio_class, contextlib.redirect_stdout(io.StringIO()) as output:
            experiment.run_audio_calibration()

        audio_class.assert_called_once_with(
            mode="both",
            contact_threshold=DEFAULT_AUDIO_CONTACT_THRESHOLD,
            lateral_threshold=DEFAULT_AUDIO_LATERAL_THRESHOLD,
            lateral_max=DEFAULT_AUDIO_LATERAL_MAX,
            volume=DEFAULT_AUDIO_VOLUME,
        )
        preview.assert_called_once_with()
        controller.close.assert_called_once_with()
        self.assertIn("Audio calibration", output.getvalue())


if __name__ == "__main__":
    unittest.main()
