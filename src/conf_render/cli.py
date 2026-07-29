"""Command-line interface for validation, planning, and rendering."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from conf_render.chunks import discover_chunks
from conf_render.ffmpeg import RenderArtifacts, render as run_render
from conf_render.ffmpeg import write_artifacts
from conf_render.models import ChunkedVideoSegment, Manifest, load_manifest
from conf_render.probe import MediaInfo, ProbeError, probe_chunked_media, probe_media
from conf_render.progress import TimestampLogger, format_duration, format_realtime_factor
from conf_render.timeline import RenderPlan, build_plan
from conf_render.transcription import WhisperRuntime, load_whisper_runtime, transcribe_plan


@dataclass(frozen=True)
class JobWork:
    plan: RenderPlan
    output: Path
    work_dir: Path
    artifacts: RenderArtifacts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="conf-render", description="Render conference videos from JSON manifests")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "plan"):
        command = commands.add_parser(name)
        command.add_argument("manifest", type=Path)
    render = commands.add_parser("render")
    render.add_argument("manifest", type=Path)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--work-dir", type=Path)
    render.add_argument("--overwrite", action="store_true")
    render.add_argument("--dry-run", action="store_true")
    render.add_argument(
        "--only", nargs="+", metavar="JOB_ID",
        help="render only the listed job IDs",
    )
    render.add_argument("--whisper-model", default="distil-large-v3")
    render.add_argument("--whisper-language", default="en")
    render.add_argument(
        "--sequential", action="store_true",
        help="run transcription and rendering sequentially instead of in parallel lanes",
    )
    return parser


def _probe_all(manifest: Manifest, logger: TimestampLogger | None = None) -> dict[Path, MediaInfo]:
    probes: dict[Path, MediaInfo] = {}
    report = logger.reporter() if logger is not None else lambda message: print(message, file=sys.stderr)
    for segment in (segment for job in manifest.jobs for segment in job.segments):
        if segment.src not in probes:
            if isinstance(segment, ChunkedVideoSegment):
                chunks = discover_chunks(segment.src)
                report(f"Probing {len(chunks)} chunks beginning at {segment.src}")
                probes[segment.src] = probe_chunked_media(segment.src, chunks)
            else:
                report(f"Probing {segment.src}")
                probes[segment.src] = probe_media(segment.src)
        if segment.audio is not None and segment.audio.src not in probes:
            report(f"Probing {segment.audio.src}")
            probes[segment.audio.src] = probe_media(segment.audio.src)
    return probes


def _resolve_plans(
    manifest_path: Path,
    logger: TimestampLogger | None = None,
    only: list[str] | None = None,
) -> tuple[Manifest, dict[Path, MediaInfo], list[RenderPlan]]:
    manifest = load_manifest(manifest_path)
    if only:
        requested = set(only)
        available = {job.id for job in manifest.jobs}
        unknown = sorted(requested - available)
        if unknown:
            raise ValueError("unknown job id(s): " + ", ".join(unknown))
        manifest.jobs = [job for job in manifest.jobs if job.id in requested]
    probes = _probe_all(manifest, logger)
    return manifest, probes, [build_plan(job, manifest.settings, probes) for job in manifest.jobs]


def _write_probes(probes: dict[Path, MediaInfo], work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / "probes.json"
    path.write_text(json.dumps([item.to_dict() for item in probes.values()], indent=2) + "\n", encoding="utf-8")
    return path


def _transcription_duration_ms(plan: RenderPlan) -> int:
    return sum(segment.duration_ms for segment in plan.segments if segment.transcribe)


def _render_lane(jobs: list[JobWork], logger: TimestampLogger) -> None:
    for job in jobs:
        report = logger.reporter(job.plan.id)
        report(f"Rendering {job.output}")
        started = time.monotonic()
        run_render(job.artifacts, job.plan, report=report)
        elapsed = time.monotonic() - started
        report(
            f"Render complete in {format_duration(elapsed)} "
            f"({format_realtime_factor(job.plan.duration_ms, elapsed)})"
        )


def _transcription_lane(
    jobs: list[JobWork],
    logger: TimestampLogger,
    *,
    model_name: str,
    language: str,
    runtime: WhisperRuntime | None = None,
) -> WhisperRuntime | None:
    selected = [job for job in jobs if _transcription_duration_ms(job.plan)]
    if not selected:
        return runtime
    shared_runtime = runtime or load_whisper_runtime(model_name, logger.reporter())
    for job in selected:
        report = logger.reporter(job.plan.id)
        report("Transcribing selected segments...")
        started = time.monotonic()
        transcribe_plan(
            job.plan, job.output, job.work_dir,
            model_name=model_name, language=language,
            concat_files=job.artifacts.concat_files,
            progress=report, runtime=shared_runtime,
        )
        elapsed = time.monotonic() - started
        report(
            f"Transcription complete in {format_duration(elapsed)} "
            f"({format_realtime_factor(_transcription_duration_ms(job.plan), elapsed)})"
        )
    return shared_runtime


def _run_sequential(
    jobs: list[JobWork],
    logger: TimestampLogger,
    *,
    model_name: str,
    language: str,
) -> None:
    runtime: WhisperRuntime | None = None
    for job in jobs:
        runtime = _transcription_lane(
            [job], logger, model_name=model_name, language=language, runtime=runtime,
        )
        _render_lane([job], logger)


def _run_parallel(
    jobs: list[JobWork],
    logger: TimestampLogger,
    *,
    model_name: str,
    language: str,
) -> None:
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="conf-render") as executor:
        render_future = executor.submit(_render_lane, jobs, logger)
        transcription_future = executor.submit(
            _transcription_lane, jobs, logger,
            model_name=model_name, language=language,
        )
        render_future.result()
        transcription_future.result()


def execute(args: argparse.Namespace) -> int:
    if args.command == "validate":
        manifest = load_manifest(args.manifest)
        segment_count = sum(len(job.segments) for job in manifest.jobs)
        print(f"Valid manifest: {len(manifest.jobs)} jobs ({segment_count} segments)")
        return 0
    logger = TimestampLogger()
    command_started = time.monotonic()
    _, probes, plans = _resolve_plans(
        args.manifest,
        logger if args.command == "render" else None,
        args.only if args.command == "render" else None,
    )
    if args.command == "plan":
        print(json.dumps([plan.to_dict() for plan in plans], indent=2))
        return 0

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_root = (args.work_dir or output_dir / ".work").resolve()
    existing = [output_dir / f"{plan.id}.mp4" for plan in plans if (output_dir / f"{plan.id}.mp4").exists()]
    if existing and not args.overwrite:
        raise ValueError(f"output already exists: {existing[0]} (use --overwrite)")
    _write_probes(probes, work_root)
    jobs: list[JobWork] = []
    for plan in plans:
        output = output_dir / f"{plan.id}.mp4"
        work_dir = work_root / plan.id
        artifacts = write_artifacts(plan, output, work_dir, args.overwrite)
        if args.dry_run:
            logger.log(f"Dry run complete. Diagnostics: {work_dir}", job_id=plan.id)
            continue
        jobs.append(JobWork(plan, output, work_dir, artifacts))
    if args.dry_run:
        logger.log(f"Dry run complete in {format_duration(time.monotonic() - command_started)}")
        return 0
    runner = _run_sequential if args.sequential else _run_parallel
    runner(
        jobs, logger,
        model_name=args.whisper_model,
        language=args.whisper_language,
    )
    logger.log(f"All jobs complete in {format_duration(time.monotonic() - command_started)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return execute(_parser().parse_args(argv))
    except (ValidationError, ValueError, ProbeError, RuntimeError) as exc:
        TimestampLogger().log(f"error: {exc}")
        return 1