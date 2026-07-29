"""Strict Pydantic models for manifest version 1."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator, model_validator

from conf_render.chunks import discover_chunks
from conf_render.timecodes import parse_timecode

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class RenderSettings(StrictModel):
    width: PositiveInt = 1920
    height: PositiveInt = 1080
    fps: PositiveInt = 30
    transition_ms: PositiveInt = Field(1000, alias="transitionMs")
    image_ms: PositiveInt = Field(4000, alias="imageMs")
    audio_sample_rate: PositiveInt = Field(48000, alias="audioSampleRate")
    video_encoder: Literal["auto", "software", "videotoolbox", "nvenc"] = Field("auto", alias="videoEncoder")
    video_bitrate: str | None = Field(None, alias="videoBitrate", pattern=r"^[1-9][0-9]*(?:[KMG])?$")
    nvenc_cq: int = Field(23, alias="nvencCq", ge=0, le=51)
    software_preset: str = Field("medium", alias="softwarePreset", min_length=1)
    software_crf: int = Field(23, alias="softwareCrf", ge=0, le=51)


class SegmentAudio(StrictModel):
    src: Path
    mode: Literal["replace", "mix"] = "replace"
    in_ms: int = Field(0, alias="in", ge=0)
    gain_db: float = Field(0, alias="gainDb")
    source_gain_db: float = Field(0, alias="sourceGainDb")

    @field_validator("in_ms", mode="before")
    @classmethod
    def parse_timestamp(cls, value: object) -> object:
        if not isinstance(value, str):
            raise ValueError("timestamp must be a string")
        return parse_timecode(value)

    @model_validator(mode="after")
    def check_source_gain_mode(self) -> "SegmentAudio":
        if self.mode != "mix" and self.source_gain_db != 0:
            raise ValueError("sourceGainDb is only valid when mode is 'mix'")
        return self


class SegmentBase(StrictModel):
    overlay: Path | None = None
    audio: SegmentAudio | None = None


class ImageSegment(SegmentBase):
    type: Literal["image"]
    src: Path
    duration_ms: PositiveInt = Field(4000, alias="durationMs")


class VideoBase(SegmentBase):
    src: Path
    in_ms: int | None = Field(None, alias="in", ge=0)
    out_ms: int | None = Field(None, alias="out", ge=0)
    transcribe: bool = False

    @field_validator("in_ms", "out_ms", mode="before")
    @classmethod
    def parse_timestamp(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("timestamp must be a string")
        return parse_timecode(value)

    @model_validator(mode="after")
    def check_window(self) -> "VideoBase":
        if self.in_ms is not None and self.out_ms is not None and self.out_ms <= self.in_ms:
            raise ValueError("out must be later than in")
        return self


class VideoSegment(VideoBase):
    type: Literal["video"]


class ChunkedVideoSegment(VideoBase):
    type: Literal["chunkedVideo"]


Segment = Annotated[ImageSegment | VideoSegment | ChunkedVideoSegment, Field(discriminator="type")]


class RenderJob(StrictModel):
    id: str
    segments: list[Segment] = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def filename_safe_id(cls, value: str) -> str:
        if not _SAFE_ID.fullmatch(value):
            raise ValueError("id must be nonempty and contain only letters, numbers, '.', '_' or '-'")
        return value


class Manifest(StrictModel):
    version: Literal[1]
    settings: RenderSettings = Field(default_factory=RenderSettings)
    jobs: list[RenderJob] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def apply_image_duration_default(cls, value: object) -> object:
        """Apply the manifest image default before segment validation."""
        if not isinstance(value, dict):
            return value

        settings = value.get("settings")
        if isinstance(settings, RenderSettings):
            image_ms: object = settings.image_ms
        elif isinstance(settings, dict):
            image_ms = settings.get("imageMs", settings.get("image_ms", 4000))
        else:
            image_ms = 4000

        jobs = value.get("jobs")
        if not isinstance(jobs, list):
            return value

        result = value.copy()
        result_jobs: list[object] = []
        for job in jobs:
            if not isinstance(job, dict) or not isinstance(job.get("segments"), list):
                result_jobs.append(job)
                continue
            result_job = job.copy()
            result_segments: list[object] = []
            for segment in job["segments"]:
                if (
                    isinstance(segment, dict)
                    and segment.get("type") == "image"
                    and "durationMs" not in segment
                    and "duration_ms" not in segment
                ):
                    segment = {**segment, "durationMs": image_ms}
                result_segments.append(segment)
            result_job["segments"] = result_segments
            result_jobs.append(result_job)
        result["jobs"] = result_jobs
        return result

    @model_validator(mode="after")
    def check_jobs(self) -> "Manifest":
        ids = [job.id for job in self.jobs]
        if len(ids) != len(set(ids)):
            raise ValueError("job ids must be unique")
        for job_index, job in enumerate(self.jobs):
            if len(job.segments) > 1:
                for segment_index, segment in enumerate(job.segments):
                    if isinstance(segment, ImageSegment) and segment.duration_ms < self.settings.transition_ms:
                        raise ValueError(
                            f"jobs[{job_index}].segments[{segment_index}] durationMs must be at least "
                            f"transitionMs ({self.settings.transition_ms})"
                        )
        return self


def load_manifest(path: Path, *, require_sources: bool = True) -> Manifest:
    """Read a JSON manifest and resolve source paths against its directory."""
    try:
        manifest = Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read manifest {path}: {exc}") from exc
    base = path.resolve().parent
    missing: list[Path] = []
    for job in manifest.jobs:
        for segment in job.segments:
            segment.src = (base / segment.src).resolve() if not segment.src.is_absolute() else segment.src.resolve()
            if require_sources and not segment.src.is_file():
                missing.append(segment.src)
            elif require_sources and isinstance(segment, ChunkedVideoSegment):
                discover_chunks(segment.src)
            if segment.overlay is not None:
                segment.overlay = (
                    (base / segment.overlay).resolve()
                    if not segment.overlay.is_absolute()
                    else segment.overlay.resolve()
                )
                if require_sources and not segment.overlay.is_file():
                    missing.append(segment.overlay)
            if segment.audio is not None:
                segment.audio.src = (
                    (base / segment.audio.src).resolve()
                    if not segment.audio.src.is_absolute()
                    else segment.audio.src.resolve()
                )
                if require_sources and not segment.audio.src.is_file():
                    missing.append(segment.audio.src)
    if missing:
        raise ValueError("source file(s) not found: " + ", ".join(str(item) for item in missing))
    return manifest
