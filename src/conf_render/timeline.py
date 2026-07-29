"""Resolve validated segments onto the final output timeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from conf_render.models import ImageSegment, RenderJob, RenderSettings
from conf_render.probe import MediaInfo


@dataclass(frozen=True)
class PlannedAudio:
    src: Path
    mode: str
    source_in_ms: int
    gain_db: float
    source_gain_db: float

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["src"] = str(self.src)
        return data


@dataclass(frozen=True)
class PlannedSegment:
    index: int
    type: str
    src: Path
    duration_ms: int
    source_in_ms: int
    source_out_ms: int
    timeline_start_ms: int
    timeline_end_ms: int
    has_audio: bool
    transcribe: bool
    chunks: tuple[Path, ...] = ()
    overlay: Path | None = None
    audio: PlannedAudio | None = None

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["src"] = str(self.src)
        data["chunks"] = [str(path) for path in self.chunks]
        if self.overlay is None:
            data.pop("overlay")
        else:
            data["overlay"] = str(self.overlay)
        if self.audio is None:
            data.pop("audio")
        else:
            data["audio"] = self.audio.to_dict()
        return data


@dataclass(frozen=True)
class RenderPlan:
    id: str
    width: int
    height: int
    fps: int
    transition_ms: int
    audio_sample_rate: int
    video_encoder: str
    video_bitrate: str | None
    nvenc_cq: int
    software_preset: str
    software_crf: int
    duration_ms: int
    segments: tuple[PlannedSegment, ...]

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["segments"] = [segment.to_dict() for segment in self.segments]
        return data


def build_plan(job: RenderJob, settings: RenderSettings, probes: dict[Path, MediaInfo]) -> RenderPlan:
    """Calculate trim durations and transition-overlapped output positions."""
    planned: list[PlannedSegment] = []
    cursor_ms = 0
    for index, segment in enumerate(job.segments):
        info = probes[segment.src]
        if not info.has_video:
            raise ValueError(f"segments[{index}] source has no video stream: {segment.src}")
        if isinstance(segment, ImageSegment):
            source_in_ms = 0
            source_out_ms = segment.duration_ms
            duration_ms = segment.duration_ms
            has_audio = False
            transcribe = False
        else:
            source_in_ms = segment.in_ms or 0
            source_out_ms = segment.out_ms if segment.out_ms is not None else info.duration_ms
            if source_in_ms >= info.duration_ms:
                raise ValueError(f"segments[{index}] in timestamp is outside source duration")
            if source_out_ms > info.duration_ms:
                raise ValueError(f"segments[{index}] out timestamp exceeds source duration")
            duration_ms = source_out_ms - source_in_ms
            has_audio = info.has_audio
            transcribe = segment.transcribe
        if len(job.segments) > 1 and duration_ms < settings.transition_ms:
            raise ValueError(
                f"segments[{index}] duration ({duration_ms}ms) must be at least transitionMs "
                f"({settings.transition_ms}ms)"
            )
        start_ms = cursor_ms
        end_ms = start_ms + duration_ms
        planned_audio = None
        if segment.audio is not None:
            audio_info = probes[segment.audio.src]
            if not audio_info.has_audio:
                raise ValueError(f"segments[{index}] audio source has no audio stream: {segment.audio.src}")
            if segment.audio.in_ms >= audio_info.duration_ms:
                raise ValueError(f"segments[{index}] audio in timestamp is outside source duration")
            planned_audio = PlannedAudio(
                src=segment.audio.src,
                mode=segment.audio.mode,
                source_in_ms=segment.audio.in_ms,
                gain_db=segment.audio.gain_db,
                source_gain_db=segment.audio.source_gain_db,
            )
        planned.append(PlannedSegment(
            index=index,
            type=segment.type,
            src=segment.src,
            duration_ms=duration_ms,
            source_in_ms=source_in_ms,
            source_out_ms=source_out_ms,
            timeline_start_ms=start_ms,
            timeline_end_ms=end_ms,
            has_audio=has_audio,
            transcribe=transcribe,
            chunks=info.chunks,
            overlay=segment.overlay,
            audio=planned_audio,
        ))
        cursor_ms = end_ms - settings.transition_ms if index < len(job.segments) - 1 else end_ms
    return RenderPlan(
        id=job.id,
        width=settings.width,
        height=settings.height,
        fps=settings.fps,
        transition_ms=settings.transition_ms,
        audio_sample_rate=settings.audio_sample_rate,
        video_encoder=settings.video_encoder,
        video_bitrate=settings.video_bitrate,
        nvenc_cq=settings.nvenc_cq,
        software_preset=settings.software_preset,
        software_crf=settings.software_crf,
        duration_ms=cursor_ms,
        segments=tuple(planned),
    )