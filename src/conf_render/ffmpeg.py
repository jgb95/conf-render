"""FFmpeg input, filter graph, command, and diagnostic generation."""

from __future__ import annotations

import json
import platform
import re
import shlex
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from conf_render.progress import run_ffmpeg
from conf_render.probe import MediaInfo, probe_media
from conf_render.timecodes import ffmpeg_seconds
from conf_render.timeline import RenderPlan


@dataclass(frozen=True)
class RenderArtifacts:
    command: tuple[str, ...]
    filter_script: Path
    command_file: Path
    plan_file: Path
    concat_files: dict[int, Path]


def available_video_encoders() -> set[str]:
    """Return video encoder names advertised by the local FFmpeg build."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot inspect FFmpeg encoders: {exc}") from exc
    encoders: set[str] = set()
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0].startswith("V"):
            encoders.add(fields[1])
    return encoders


def resolve_video_encoder(requested: str, available: set[str] | None = None) -> str:
    """Resolve a portable encoder setting to a concrete FFmpeg encoder."""
    available = available if available is not None else available_video_encoders()
    choices = {
        "software": "libx264",
        "videotoolbox": "h264_videotoolbox",
        "nvenc": "h264_nvenc",
    }
    if requested == "auto":
        preferred = (
            ("h264_videotoolbox", "h264_nvenc", "libx264")
            if platform.system() == "Darwin"
            else ("h264_nvenc", "libx264")
        )
        for encoder in preferred:
            if encoder in available:
                return encoder
        raise RuntimeError("FFmpeg has no supported H.264 encoder")
    encoder = choices[requested]
    if encoder not in available:
        raise RuntimeError(f"requested video encoder is unavailable: {encoder}")
    return encoder


def _video_encoder_args(plan: RenderPlan) -> list[str]:
    encoder = resolve_video_encoder(plan.video_encoder)
    rate_control: list[str]
    if plan.video_bitrate is None:
        rate_control = []
    else:
        buffer_size = _double_bitrate(plan.video_bitrate)
        rate_control = [
            "-b:v", plan.video_bitrate,
            "-minrate", plan.video_bitrate,
            "-maxrate", plan.video_bitrate,
            "-bufsize", buffer_size,
        ]
    if encoder == "libx264":
        quality = ["-crf", str(plan.software_crf)] if plan.video_bitrate is None else rate_control
        return ["-c:v", encoder, "-preset", plan.software_preset, *quality,
                "-profile:v", "high", "-pix_fmt", "yuv420p"]
    if encoder == "h264_videotoolbox":
        quality = ["-q:v", "65"] if plan.video_bitrate is None else rate_control
        return ["-c:v", encoder, *quality, "-profile:v", "high", "-pix_fmt", "yuv420p"]
    if plan.video_bitrate is None:
        rate_control = ["-rc", "vbr", "-cq", str(plan.nvenc_cq), "-b:v", "0"]
    return [
        "-c:v", encoder, "-preset", "p4", *rate_control,
        "-profile:v", "high", "-pix_fmt", "yuv420p",
    ]


def _double_bitrate(value: str) -> str:
    match = re.fullmatch(r"([1-9][0-9]*)([kKmMgG]?)", value)
    if match is None:
        raise ValueError(f"invalid video bitrate: {value}")
    return f"{int(match.group(1)) * 2}{match.group(2)}"


def _normalized_output_args(plan: RenderPlan) -> list[str]:
    gop = plan.fps * plan.keyframe_interval
    return ["-r", str(plan.fps), "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0"]


def verify_render_output(plan: RenderPlan, info: MediaInfo) -> None:
    """Reject an output that does not satisfy conf-render's normalized contract."""
    try:
        frame_rate_matches = info.frame_rate is not None and Fraction(info.frame_rate) == plan.fps
    except (ValueError, ZeroDivisionError):
        frame_rate_matches = False
    checks = [
        (info.has_video, "video stream"),
        (info.has_audio, "audio stream"),
        (info.video_codec == "h264", "H.264 video"),
        (info.video_profile == "High", "H.264 High profile"),
        (info.pixel_format == "yuv420p", "yuv420p pixel format"),
        (info.width == plan.width and info.height == plan.height, f"{plan.width}x{plan.height} resolution"),
        (frame_rate_matches, f"{plan.fps} fps"),
        (info.audio_codec == "aac", "AAC audio"),
        (info.audio_sample_rate == str(plan.audio_sample_rate), f"{plan.audio_sample_rate} Hz audio"),
        (info.audio_channels == 2, "stereo audio"),
    ]
    missing = [description for valid, description in checks if not valid]
    if missing:
        raise RuntimeError(f"rendered output failed verification ({', '.join(missing)}): {info.path}")


def _write_concat_file(chunks: tuple[Path, ...], path: Path) -> None:
    """Write an FFconcat list using shell-style quoting accepted by FFmpeg."""
    lines = ["ffconcat version 1.0", *(f"file {shlex.quote(str(chunk))}" for chunk in chunks)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_filter_graph(plan: RenderPlan) -> str:
    """Build a normalized A/V graph with transitions and endpoint fades."""
    filters: list[str] = []
    input_index = 0
    for segment in plan.segments:
        index = segment.index
        duration = ffmpeg_seconds(segment.duration_ms)
        source_input = input_index
        input_index += 1
        normalized_label = f"vn{index}" if segment.overlay is not None else f"v{index}"
        video = (
            f"[{source_input}:v]trim=duration={duration},setpts=PTS-STARTPTS,"
            f"scale={plan.width}:{plan.height}:force_original_aspect_ratio=decrease,"
            f"pad={plan.width}:{plan.height}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"fps={plan.fps},format=yuv420p,settb=AVTB[{normalized_label}]"
        )
        filters.append(video)
        source_audio_label = f"asource{index}" if segment.audio is not None else f"a{index}"
        if segment.has_audio and (segment.audio is None or segment.audio.mode == "mix"):
            # Materialize timestamp gaps (notably AAC gaps at concat boundaries) as
            # samples before acrossfade, which otherwise shortens audio cumulatively.
            filters.append(
                f"[{source_input}:a]atrim=duration={duration},asetpts=PTS-STARTPTS,"
                f"aresample={plan.audio_sample_rate}:async=1:first_pts=0,aformat=sample_fmts=fltp:"
                f"sample_rates={plan.audio_sample_rate}:channel_layouts=stereo,"
                f"apad,atrim=duration={duration},asetpts=PTS-STARTPTS[{source_audio_label}]"
            )
        elif segment.audio is None or segment.audio.mode == "mix":
            filters.append(
                f"anullsrc=r={plan.audio_sample_rate}:cl=stereo,"
                f"atrim=duration={duration},asetpts=PTS-STARTPTS[{source_audio_label}]"
            )
        if segment.overlay is not None:
            overlay_input = input_index
            input_index += 1
            filters.append(
                f"[{overlay_input}:v]trim=duration={duration},setpts=PTS-STARTPTS,"
                f"scale={plan.width}:{plan.height}:force_original_aspect_ratio=decrease,"
                f"pad={plan.width}:{plan.height}:(ow-iw)/2:(oh-ih)/2:color=black@0,"
                f"fps={plan.fps},format=rgba,settb=AVTB[overlay{index}]"
            )
            filters.append(
                f"[vn{index}][overlay{index}]overlay=0:0:shortest=1:format=auto,"
                f"format=yuv420p[v{index}]"
            )
        if segment.audio is not None:
            audio_input = input_index
            input_index += 1
            external_label = f"aexternal{index}"
            filters.append(
                f"[{audio_input}:a]asetpts=PTS-STARTPTS,"
                f"aresample={plan.audio_sample_rate}:async=1:first_pts=0,"
                f"aformat=sample_fmts=fltp:sample_rates={plan.audio_sample_rate}:channel_layouts=stereo,"
                f"volume={segment.audio.gain_db:g}dB,apad,atrim=duration={duration}[{external_label}]"
            )
            if segment.audio.mode == "replace":
                filters.append(f"[{external_label}]anull[a{index}]")
            else:
                filters.append(
                    f"[{source_audio_label}]volume={segment.audio.source_gain_db:g}dB[amixsource{index}]"
                )
                filters.append(
                    f"[amixsource{index}][{external_label}]"
                    f"amix=inputs=2:duration=first:normalize=0[a{index}]"
                )

    video_label = "v0"
    audio_label = "a0"
    for index in range(1, len(plan.segments)):
        offset_ms = plan.segments[index].timeline_start_ms
        next_video = f"vx{index}"
        next_audio = f"ax{index}"
        filters.append(
            f"[{video_label}][v{index}]xfade=transition=fade:"
            f"duration={ffmpeg_seconds(plan.transition_ms)}:offset={ffmpeg_seconds(offset_ms)}[{next_video}]"
        )
        filters.append(
            f"[{audio_label}][a{index}]acrossfade=d={ffmpeg_seconds(plan.transition_ms)}:"
            f"c1=tri:c2=tri[{next_audio}]"
        )
        video_label = next_video
        audio_label = next_audio

    fade_ms = min(1000, plan.duration_ms)
    fade_out_ms = max(0, plan.duration_ms - fade_ms)
    filters.append(
        f"[{video_label}]fade=t=in:st=0:d={ffmpeg_seconds(fade_ms)}:color=black,"
        f"fade=t=out:st={ffmpeg_seconds(fade_out_ms)}:d={ffmpeg_seconds(fade_ms)}:color=black[vout]"
    )
    filters.append(
        f"[{audio_label}]afade=t=in:st=0:d={ffmpeg_seconds(fade_ms)},"
        f"afade=t=out:st={ffmpeg_seconds(fade_out_ms)}:d={ffmpeg_seconds(fade_ms)}[aout]"
    )
    return ";\n".join(filters) + ";\n"


def build_command(
    plan: RenderPlan,
    output: Path,
    overwrite: bool,
    concat_files: dict[int, Path] | None = None,
) -> list[str]:
    """Construct the final FFmpeg command for a resolved plan."""
    command = ["ffmpeg", "-y" if overwrite else "-n", "-hide_banner"]
    for segment in plan.segments:
        if segment.type == "image":
            command.extend([
                "-loop", "1", "-framerate", str(plan.fps),
                "-t", ffmpeg_seconds(segment.duration_ms), "-i", str(segment.src),
            ])
        else:
            command.extend(["-ss", ffmpeg_seconds(segment.source_in_ms), "-t", ffmpeg_seconds(segment.duration_ms)])
            if segment.chunks:
                if concat_files is None or segment.index not in concat_files:
                    raise ValueError(f"missing concat list for segment {segment.index}")
                command.extend(["-f", "concat", "-safe", "0", "-i", str(concat_files[segment.index])])
            else:
                command.extend(["-i", str(segment.src)])
        if segment.overlay is not None:
            command.extend([
                "-loop", "1", "-framerate", str(plan.fps),
                "-t", ffmpeg_seconds(segment.duration_ms), "-i", str(segment.overlay),
            ])
        if segment.audio is not None:
            command.extend([
                "-ss", ffmpeg_seconds(segment.audio.source_in_ms),
                "-t", ffmpeg_seconds(segment.duration_ms), "-i", str(segment.audio.src),
            ])
    command.extend([
        "-filter_complex", build_filter_graph(plan),
        "-map", "[vout]", "-map", "[aout]",
    ])
    command.extend(_video_encoder_args(plan))
    command.extend([
        *_normalized_output_args(plan),
        "-c:a", "aac", "-b:a", plan.audio_bitrate, "-ar", str(plan.audio_sample_rate), "-ac", "2",
        "-movflags", "+faststart", str(output),
    ])
    return command


def write_artifacts(plan: RenderPlan, output: Path, work_dir: Path, overwrite: bool) -> RenderArtifacts:
    """Write reproducible planning and FFmpeg diagnostics."""
    work_dir.mkdir(parents=True, exist_ok=True)
    filter_script = work_dir / "filter-complex.txt"
    plan_file = work_dir / "plan.json"
    command_file = work_dir / "ffmpeg-command.txt"
    concat_files: dict[int, Path] = {}
    for segment in plan.segments:
        if segment.chunks:
            concat_file = work_dir / f"segment-{segment.index:03d}.ffconcat"
            _write_concat_file(segment.chunks, concat_file)
            concat_files[segment.index] = concat_file
    filter_script.write_text(build_filter_graph(plan), encoding="utf-8")
    plan_file.write_text(json.dumps(plan.to_dict(), indent=2) + "\n", encoding="utf-8")
    command = build_command(plan, output, overwrite, concat_files)
    command_file.write_text(shlex.join(command) + "\n", encoding="utf-8")
    return RenderArtifacts(tuple(command), filter_script, command_file, plan_file, concat_files)


def render(
    artifacts: RenderArtifacts,
    plan: RenderPlan,
    *,
    report: Callable[[str], None] = lambda message: print(message, file=sys.stderr),
) -> None:
    """Execute a previously materialized render command."""
    run_ffmpeg(list(artifacts.command), plan.duration_ms, report=report)
    report(f"Verifying {artifacts.command[-1]}")
    info = probe_media(Path(artifacts.command[-1]))
    verify_render_output(plan, info)
