from pathlib import Path

import mujoco

from .config import VIDEO_FPS, VIDEO_HEIGHT, VIDEO_WIDTH


class VideoRecorder:
    """Stream offscreen frames to mp4 with the camera supplied by the environment."""

    def __init__(self, model, path, fps=VIDEO_FPS, width=VIDEO_WIDTH, height=VIDEO_HEIGHT):
        self.path = Path(path)
        self.fps = fps
        self.frame_interval = 1.0 / fps
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        self.renderer = mujoco.Renderer(model, height, width)
        self._writer = None
        self._next_frame_time = None
        self._saved_frames = 0

    def start(self):
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise RuntimeError(
                "Video recording requires imageio. Install with: pip install imageio imageio-ffmpeg"
            ) from exc

        try:
            import imageio_ffmpeg  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Video recording requires imageio-ffmpeg (imageio fell back to a non-video writer). "
                "Install with: pip install imageio-ffmpeg"
            ) from exc

        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Force FFMPEG so imageio does not pick an unrelated plugin (e.g. tifffile).
        self._writer = imageio.get_writer(
            str(self.path),
            format="FFMPEG",
            mode="I",
            fps=self.fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=1,
        )

    def capture(self, data, camera, overlay_callback=None, force=False):
        if self._writer is None:
            return

        current_time = float(data.time)
        if self._next_frame_time is None:
            self._next_frame_time = current_time
        if not force and current_time + 1e-9 < self._next_frame_time:
            return

        self.renderer.update_scene(data, camera=camera)
        if overlay_callback is not None:
            overlay_callback(self.renderer.scene)
        frame = self.renderer.render()
        frames_written = 0
        while current_time + 1e-9 >= self._next_frame_time:
            self._writer.append_data(frame)
            self._saved_frames += 1
            frames_written += 1
            self._next_frame_time += self.frame_interval

        if force and frames_written == 0:
            self._writer.append_data(frame)
            self._saved_frames += 1

    def close(self):
        if self._writer is None:
            return

        self._writer.close()
        self._writer = None
        if self._saved_frames == 0:
            print("No video frames captured; skipping video save.")
            if self.path.exists():
                self.path.unlink()
            return

        duration = self._saved_frames / self.fps
        print(
            f"Saved run video ({self._saved_frames} frames, {duration:.2f}s @ {self.fps} fps) "
            f"to {self.path.resolve()}"
        )
