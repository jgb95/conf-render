"""ffprobe subprocess integration and media metadata parsing."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Callable


class ProbeError(RuntimeError):
    """Raised when media metadata cannot be obtained."""


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    duration_ms: int
    has_video: bool
    has_audio: bool
    width: int | None
    height: int | None
    frame_rate: str | None
    chunks: tuple[Path, ...] = ()

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["path"] = str(self.path)
        value["chunks"] = [str(path) for path in self.chunks]
        return value


def _duration_ms(value: object) -> int:
    try:
        return int((Decimal(str(value)) * 1000).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    except (InvalidOperation, ValueError) as exc:
        raise ProbeError(f"invalid ffprobe duration: {value!r}") from exc


def parse_probe(payload: dict[str, Any], path: Path) -> MediaInfo:
    """Convert ffprobe JSON output into stable typed metadata."""
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise ProbeError(f"ffprobe returned no stream list for {path}")
    videos = [item for item in streams if isinstance(item, dict) and item.get("codec_type") == "video"]
    audios = [item for item in streams if isinstance(item, dict) and item.get("codec_type") == "audio"]
    video = videos[0] if videos else None
    duration = payload.get("format", {}).get("duration") if isinstance(payload.get("format"), dict) else None
    if duration in (None, "N/A"):
        candidates = [item.get("duration") for item in streams if isinstance(item, dict)]
        duration = next((item for item in candidates if item not in (None, "N/A")), None)
    if duration is None:
        # Still-image demuxers commonly report no intrinsic duration. Its
        # manifest duration controls the generated input instead.
        duration = 0
    return MediaInfo(
        path=path,
        duration_ms=_duration_ms(duration),
        has_video=bool(videos),
        has_audio=bool(audios),
        width=int(video["width"]) if video and video.get("width") is not None else None,
        height=int(video["height"]) if video and video.get("height") is not None else None,
        frame_rate=(
            str(video.get("r_frame_rate") or video.get("avg_frame_rate"))
            if video and (video.get("r_frame_rate") or video.get("avg_frame_rate"))
            else None
        ),
    )


def probe_media(
    path: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> MediaInfo:
    """Run ffprobe for one source file."""
    command = [
        "ffprobe", "-v", "error", "-show_streams", "-show_format",
        "-of", "json", str(path),
    ]
    try:
        result = runner(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise ProbeError("ffprobe was not found on PATH") from exc
    if result.returncode:
        detail = result.stderr.strip() or "unknown ffprobe error"
        raise ProbeError(f"could not probe {path}: {detail}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ProbeError(f"ffprobe returned invalid JSON for {path}") from exc
    return parse_probe(payload, path)


def probe_chunked_media(
    first: Path,
    chunks: tuple[Path, ...],
    *,
    probe: Callable[[Path], MediaInfo] = probe_media,
) -> MediaInfo:
    """Probe and aggregate one compatible sequence as a logical source."""
    infos = [probe(path) for path in chunks]
    reference = infos[0]
    if not reference.has_video:
        raise ProbeError(f"chunk has no video stream: {reference.path}")
    for info in infos[1:]:
        attributes = (info.has_video, info.has_audio, info.width, info.height, info.frame_rate)
        expected = (
            reference.has_video, reference.has_audio, reference.width,
            reference.height, reference.frame_rate,
        )
        if attributes != expected:
            raise ProbeError(f"chunk stream layout differs from first chunk: {info.path}")
    return MediaInfo(
        path=first,
        duration_ms=sum(info.duration_ms for info in infos),
        has_video=reference.has_video,
        has_audio=reference.has_audio,
        width=reference.width,
        height=reference.height,
        frame_rate=reference.frame_rate,
        chunks=chunks,
    )
