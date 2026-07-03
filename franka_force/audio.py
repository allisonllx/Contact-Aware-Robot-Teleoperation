import math
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AudioFeedbackState:
    contact_event: bool = False
    tick_rate: float = 0.0
    lateral_force: float = 0.0


class AudioFeedback:
    """Small dependency-free click/tick player for interactive force feedback."""

    geiger_min_rate = 1.5
    geiger_max_rate = 14.0
    sample_rate = 44100

    def __init__(
        self,
        mode,
        contact_threshold,
        lateral_threshold,
        lateral_max,
        volume,
        player_path=None,
    ):
        self.mode = mode
        self.contact_threshold = contact_threshold
        self.contact_rearm_threshold = contact_threshold * 0.5
        self.lateral_threshold = lateral_threshold
        self.lateral_max = lateral_max
        self.volume = volume
        self.player_path = player_path if player_path is not None else shutil.which("afplay")
        self._contact_armed = True
        self._next_tick_time = None
        self._processes = []
        self._tmpdir = None
        self._contact_path = None
        self._tick_path = None

        if self.player_path is None:
            print("Audio feedback requested, but `afplay` was not found; running silently.")
            return

        self._tmpdir = tempfile.TemporaryDirectory(prefix="franka_force_audio_")
        root = Path(self._tmpdir.name)
        self._contact_path = root / "contact_click.wav"
        self._tick_path = root / "geiger_tick.wav"
        self._write_tone(self._contact_path, frequency=1700.0, duration=0.035)
        self._write_tone(self._tick_path, frequency=2300.0, duration=0.022)

    def update(self, sim_time, f_est, contact_force_vector):
        lateral_force = math.hypot(float(contact_force_vector[0]), float(contact_force_vector[1]))
        contact_event = self._update_contact_click(f_est)
        tick_rate = self._update_geiger_ticks(sim_time, lateral_force)
        return AudioFeedbackState(
            contact_event=contact_event,
            tick_rate=tick_rate,
            lateral_force=lateral_force,
        )

    def close(self):
        for process in self._processes:
            if process.poll() is None:
                process.terminate()
        self._processes = []
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    def _update_contact_click(self, f_est):
        contact_event = False
        if self.mode in ("contact", "both"):
            if self._contact_armed and f_est > self.contact_threshold:
                contact_event = True
                self._contact_armed = False
                self._play(self._contact_path)
            elif not self._contact_armed and f_est < self.contact_rearm_threshold:
                self._contact_armed = True
        return contact_event

    def _update_geiger_ticks(self, sim_time, lateral_force):
        if self.mode not in ("geiger", "both"):
            return 0.0
        if lateral_force < self.lateral_threshold:
            self._next_tick_time = None
            return 0.0

        span = max(self.lateral_max - self.lateral_threshold, 1e-9)
        intensity = min(max((lateral_force - self.lateral_threshold) / span, 0.0), 1.0)
        tick_rate = self.geiger_min_rate + intensity * (self.geiger_max_rate - self.geiger_min_rate)
        if self._next_tick_time is None or sim_time >= self._next_tick_time:
            self._play(self._tick_path)
            self._next_tick_time = sim_time + 1.0 / tick_rate
        return tick_rate

    def _play(self, path):
        if self.player_path is None or path is None:
            return
        self._reap_processes()
        if len(self._processes) >= 8:
            return
        try:
            process = subprocess.Popen(
                [self.player_path, str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            return
        self._processes.append(process)

    def _reap_processes(self):
        self._processes = [process for process in self._processes if process.poll() is None]

    def _write_tone(self, path, frequency, duration):
        n_samples = max(1, int(self.sample_rate * duration))
        amplitude = int(max(0.0, min(self.volume, 1.0)) * 32767)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            frames = bytearray()
            for i in range(n_samples):
                t = i / self.sample_rate
                envelope = math.exp(-7.0 * i / n_samples)
                sample = int(amplitude * envelope * math.sin(2.0 * math.pi * frequency * t))
                frames.extend(sample.to_bytes(2, byteorder="little", signed=True))
            wav.writeframes(bytes(frames))
