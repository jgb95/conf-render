"""Progress reporting for long-running FFmpeg processes."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from datetime import datetime
from typing import TextIO


def format_duration(seconds: float) -> str:
    """Format a wall-clock duration compactly for completion messages."""
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remainder:04.1f}s"
    hours, remainder_minutes = divmod(int(minutes), 60)
    return f"{hours}h {remainder_minutes:02d}m {remainder:04.1f}s"


def format_realtime_factor(media_duration_ms: int, elapsed_seconds: float) -> str:
    """Format processed media duration divided by elapsed wall time."""
    factor = media_duration_ms / 1000 / max(elapsed_seconds, 0.001)
    return f"{factor:.2f}x realtime"


class TimestampLogger:
    """Write atomic, clock-timestamped messages from concurrent workers."""

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        self._stream = stream or sys.stderr
        self._clock = clock
        self._lock = threading.Lock()

    def log(self, message: str, *, job_id: str | None = None) -> None:
        scope = f" [{job_id}]" if job_id is not None else ""
        line = f"[{self._clock():%H:%M:%S}]{scope} {message}"
        with self._lock:
            print(line, file=self._stream, flush=True)

    def reporter(self, job_id: str | None = None) -> Callable[[str], None]:
        """Return a callback suitable for FFmpeg and transcription progress."""
        return lambda message: self.log(message, job_id=job_id)


class FFmpegError(RuntimeError):
    """Raised when FFmpeg exits unsuccessfully."""


def run_ffmpeg(
    command: list[str],
    duration_ms: int,
    *,
    report: Callable[[str], None] = lambda message: print(message, file=sys.stderr),
) -> None:
    """Execute FFmpeg and report percentages from its machine-readable output."""
    progress_command = command[:1] + ["-progress", "pipe:1", "-nostats"] + command[1:]
    try:
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as error_log:
            process = subprocess.Popen(
                progress_command,
                stdout=subprocess.PIPE,
                stderr=error_log,
                text=True,
            )
            assert process.stdout is not None
            last_percent = -1
            for raw_line in process.stdout:
                key, separator, value = raw_line.strip().partition("=")
                if not separator or key not in {"out_time_us", "out_time_ms"}:
                    continue
                try:
                    # FFmpeg currently reports microseconds for both historical keys.
                    elapsed_ms = int(value) // 1000
                except ValueError:
                    continue
                percent = min(100, elapsed_ms * 100 // max(1, duration_ms))
                if percent >= last_percent + 5:
                    report(f"Rendering: {percent}%")
                    last_percent = percent
            return_code = process.wait()
            error_log.seek(0)
            stderr = error_log.read()
    except FileNotFoundError as exc:
        raise FFmpegError("ffmpeg was not found on PATH") from exc
    if return_code:
        lines = [line for line in stderr.strip().splitlines() if line]
        detail = lines[-1] if lines else "unknown FFmpeg error"
        raise FFmpegError(f"ffmpeg failed (exit {return_code}): {detail}")
    if last_percent < 100:
        report("Rendering: 100%")
