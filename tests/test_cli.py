from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from conf_render.cli import JobWork, _parser, _resolve_plans, _run_parallel, _transcription_lane, execute
from conf_render.models import Manifest
from conf_render.probe import MediaInfo
from conf_render.ffmpeg import RenderArtifacts
from conf_render.progress import TimestampLogger
from conf_render.timeline import PlannedSegment, RenderPlan
from conf_render.transcription import WhisperRuntime


def make_job(tmp_path: Path, job_id: str) -> JobWork:
    segment = PlannedSegment(
        index=0,
        type="video",
        src=tmp_path / f"{job_id}.mp4",
        duration_ms=5000,
        source_in_ms=0,
        source_out_ms=5000,
        timeline_start_ms=0,
        timeline_end_ms=5000,
        has_audio=True,
        transcribe=True,
    )
    plan = RenderPlan(
        job_id, 1920, 1080, 30, 1000, 48000,
        "software", None, 23, "medium", 23, 5000, (segment,),
    )
    work_dir = tmp_path / ".work" / job_id
    artifacts = RenderArtifacts((), work_dir / "filter", work_dir / "command", work_dir / "plan", {})
    return JobWork(plan, tmp_path / f"{job_id}.output.mp4", work_dir, artifacts)


def test_render_only_accepts_multiple_job_ids() -> None:
    args = _parser().parse_args([
        "render", "manifest.json", "--output", "output", "--only", "two", "one",
    ])
    assert args.only == ["two", "one"]


def test_render_modes_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args([
            "render", "manifest.json", "--output", "output",
            "--render-only", "--transcribe-only",
        ])


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("--render-only", ["render"]),
        ("--transcribe-only", ["transcribe"]),
        (None, ["both"]),
    ],
)
def test_render_mode_selects_requested_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str | None,
    expected: list[str],
) -> None:
    job = make_job(tmp_path, "talk")
    calls: list[str] = []
    monkeypatch.setattr("conf_render.cli._resolve_plans", lambda *_args, **_kwargs: (None, {}, [job.plan]))
    monkeypatch.setattr("conf_render.cli._write_probes", lambda *_args: tmp_path / "probes.json")
    monkeypatch.setattr("conf_render.cli.write_artifacts", lambda *_args: job.artifacts)
    monkeypatch.setattr("conf_render.cli._render_lane", lambda *_args: calls.append("render"))
    monkeypatch.setattr("conf_render.cli._transcription_lane", lambda *_args, **_kwargs: calls.append("transcribe"))
    monkeypatch.setattr("conf_render.cli._run_parallel", lambda *_args, **_kwargs: calls.append("both"))
    argv = ["render", "manifest.json", "--output", str(tmp_path / "output")]
    if mode is not None:
        argv.append(mode)

    assert execute(_parser().parse_args(argv)) == 0
    assert calls == expected


def test_transcribe_only_allows_existing_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = make_job(tmp_path, "talk")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "talk.mp4").touch()
    calls: list[str] = []
    monkeypatch.setattr("conf_render.cli._resolve_plans", lambda *_args, **_kwargs: (None, {}, [job.plan]))
    monkeypatch.setattr("conf_render.cli._write_probes", lambda *_args: tmp_path / "probes.json")
    monkeypatch.setattr("conf_render.cli.write_artifacts", lambda *_args: job.artifacts)
    monkeypatch.setattr(
        "conf_render.cli._transcription_lane",
        lambda *_args, **_kwargs: calls.append("transcribe"),
    )

    args = _parser().parse_args([
        "render", "manifest.json", "--output", str(output_dir), "--transcribe-only",
    ])
    assert execute(args) == 0
    assert calls == ["transcribe"]


def test_render_only_rejects_existing_video_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = make_job(tmp_path, "talk")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "talk.mp4").touch()
    monkeypatch.setattr("conf_render.cli._resolve_plans", lambda *_args, **_kwargs: (None, {}, [job.plan]))
    args = _parser().parse_args([
        "render", "manifest.json", "--output", str(output_dir), "--render-only",
    ])

    with pytest.raises(ValueError, match="output already exists"):
        execute(args)


def test_resolve_plans_filters_before_probing_and_preserves_manifest_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = Manifest.model_validate({
        "version": 1,
        "jobs": [
            {"id": "one", "segments": [{"type": "video", "src": "one.mp4"}]},
            {"id": "two", "segments": [{"type": "video", "src": "two.mp4"}]},
            {"id": "three", "segments": [{"type": "video", "src": "three.mp4"}]},
        ],
    })
    monkeypatch.setattr("conf_render.cli.load_manifest", lambda _path: manifest)
    probed: list[str] = []

    def fake_probe(selected: Manifest, _logger: object) -> dict[Path, MediaInfo]:
        probed.extend(job.id for job in selected.jobs)
        return {
            segment.src: MediaInfo(segment.src, 5000, True, True, 1920, 1080, "30/1")
            for job in selected.jobs for segment in job.segments
        }

    monkeypatch.setattr("conf_render.cli._probe_all", fake_probe)
    _, _, plans = _resolve_plans(tmp_path / "manifest.json", only=["three", "one"])
    assert probed == ["one", "three"]
    assert [plan.id for plan in plans] == ["one", "three"]


def test_resolve_plans_rejects_unknown_only_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = Manifest.model_validate({
        "version": 1,
        "jobs": [{"id": "known", "segments": [{"type": "video", "src": "known.mp4"}]}],
    })
    monkeypatch.setattr("conf_render.cli.load_manifest", lambda _path: manifest)
    with pytest.raises(ValueError, match=r"unknown job id\(s\): missing"):
        _resolve_plans(tmp_path / "manifest.json", only=["missing"])


def test_transcription_lane_loads_runtime_once_for_all_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs = [make_job(tmp_path, "one"), make_job(tmp_path, "two")]
    runtime = WhisperRuntime(object(), "tiny", "cpu", "int8")  # type: ignore[arg-type]
    loads = 0
    received: list[WhisperRuntime] = []

    def fake_load(_model: str, _progress: object) -> WhisperRuntime:
        nonlocal loads
        loads += 1
        return runtime

    def fake_transcribe(*_args: object, runtime: WhisperRuntime, **_kwargs: object) -> None:
        received.append(runtime)

    monkeypatch.setattr("conf_render.cli.load_whisper_runtime", fake_load)
    monkeypatch.setattr("conf_render.cli.transcribe_plan", fake_transcribe)

    _transcription_lane(jobs, TimestampLogger(), model_name="tiny", language="en")

    assert loads == 1
    assert received == [runtime, runtime]


def test_parallel_runner_overlaps_render_and_transcription(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = make_job(tmp_path, "talk")
    barrier = threading.Barrier(2, timeout=1)

    def synchronized_work(*_args: object, **_kwargs: object) -> None:
        barrier.wait()
        time.sleep(0.01)

    runtime = WhisperRuntime(object(), "tiny", "cpu", "int8")  # type: ignore[arg-type]
    monkeypatch.setattr("conf_render.cli.run_render", synchronized_work)
    monkeypatch.setattr("conf_render.cli.load_whisper_runtime", lambda *_args: runtime)
    monkeypatch.setattr("conf_render.cli.transcribe_plan", synchronized_work)

    _run_parallel(
        [job], TimestampLogger(), model_name="tiny", language="en",
    )