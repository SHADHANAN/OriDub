"""
VoiceFusion AI – VideoMerger
==============================
Replaces the original audio track of a video with the generated dubbed audio.
Uses MoviePy so no subprocess calls are needed.
"""
from __future__ import annotations

from pathlib import Path

from .config import PipelineConfig, PipelineError


class VideoMerger:
    """Merges a dubbed audio track back into the original video.

    Parameters
    ----------
    config:
        Shared pipeline configuration object.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def merge(self, video_path: Path, audio_path: Path) -> Path:
        """Replace the audio track of *video_path* with *audio_path*.

        Parameters
        ----------
        video_path:
            Source video file.
        audio_path:
            Dubbed audio WAV file.

        Returns
        -------
        Path
            Path to the output video with the new audio track.
        """
        for p, label in [(video_path, "input video"), (audio_path, "dubbed audio")]:
            if not Path(p).exists():
                raise PipelineError(f"VideoMerger: {label} not found: {p}")

        out_path = self._config.final_video
        print(f"[VideoMerger] Merging audio into video → {out_path}")

        # 1. Try fast FFmpeg stream copy (instant, lossless, no re-encoding required)
        try:
            from .ffmpeg_utils import get_ffmpeg_path
            import subprocess
            ffmpeg_bin = get_ffmpeg_path()
            cmd = [
                ffmpeg_bin, "-y",
                "-i", str(video_path),
                "-i", str(audio_path),
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                str(out_path),
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
                print(f"[VideoMerger] FFmpeg merge succeeded → {out_path}")
                return out_path
        except Exception as exc:
            print(f"[VideoMerger] FFmpeg direct merge failed ({exc}), falling back to MoviePy...")

        # 2. MoviePy fallback (supporting both MoviePy 1.x and 2.x)
        try:
            try:
                from moviepy.editor import AudioFileClip, VideoFileClip
            except ImportError:
                from moviepy import AudioFileClip, VideoFileClip
        except ImportError as exc:
            raise PipelineError(
                "Neither FFmpeg nor MoviePy could merge the audio into the video."
            ) from exc

        video = VideoFileClip(str(video_path))
        audio = AudioFileClip(str(audio_path))
        dur = getattr(video, "duration", None)
        if hasattr(audio, "with_duration"):
            audio = audio.with_duration(dur)
        elif hasattr(audio, "set_duration"):
            audio = audio.set_duration(dur)

        if hasattr(video, "with_audio"):
            final = video.with_audio(audio)
        else:
            final = video.set_audio(audio)

        final.write_videofile(
            str(out_path), codec="libx264", audio_codec="aac", logger=None
        )
        video.close()
        audio.close()

        if not out_path.exists():
            raise PipelineError(f"Merged video was not created: {out_path}")

        print(f"[VideoMerger] Done → {out_path}")
        return out_path
