from pathlib import Path

import pytest

from conf_render.models import Manifest
from conf_render.probe import MediaInfo
from conf_render.timeline import build_plan


def test_plan_overlaps_segments_by_transition() -> None:
    manifest = Manifest.model_validate({
        "version": 1, "settings": {"transitionMs": 1000}, "jobs": [{
            "id": "talk", "segments": [
                {"type": "image", "src": "a.png", "durationMs": 3000},
                {"type": "video", "src": "b.mp4", "in": "00:00:01", "out": "00:00:05"},
            ],
        }],
    })
    a, b = Path("a.png"), Path("b.mp4")
    probes = {
        a: MediaInfo(a, 0, True, False, 10, 10, None),
        b: MediaInfo(b, 10_000, True, True, 10, 10, "30/1"),
    }
    plan = build_plan(manifest.jobs[0], manifest.settings, probes)
    assert plan.segments[1].timeline_start_ms == 2000
    assert plan.duration_ms == 6000
    assert plan.segments[1].source_in_ms == 1000


def test_plan_carries_segment_overlay() -> None:
    manifest = Manifest.model_validate({
        "version": 1,
        "jobs": [{"id": "talk", "segments": [{
            "type": "video", "src": "talk.mp4", "overlay": "bug.png",
        }]}],
    })
    source = Path("talk.mp4")
    plan = build_plan(
        manifest.jobs[0], manifest.settings,
        {source: MediaInfo(source, 5000, True, True, 1920, 1080, "30/1")},
    )
    assert plan.segments[0].overlay == Path("bug.png")
    assert plan.segments[0].to_dict()["overlay"] == "bug.png"


def test_plan_validates_and_carries_external_audio() -> None:
    manifest = Manifest.model_validate({
        "version": 1,
        "jobs": [{"id": "talk", "segments": [{
            "type": "video", "src": "talk.mp4",
            "audio": {"src": "clean.wav", "in": "00:00:01", "gainDb": -2},
        }]}],
    })
    source, audio = Path("talk.mp4"), Path("clean.wav")
    plan = build_plan(
        manifest.jobs[0], manifest.settings,
        {
            source: MediaInfo(source, 5000, True, True, 1920, 1080, "30/1"),
            audio: MediaInfo(audio, 3000, False, True, None, None, None),
        },
    )
    planned = plan.segments[0].audio
    assert planned is not None
    assert planned.source_in_ms == 1000
    assert planned.gain_db == -2
    assert plan.segments[0].to_dict()["audio"]["src"] == "clean.wav"


def test_plan_rejects_audio_source_without_audio_stream() -> None:
    manifest = Manifest.model_validate({
        "version": 1,
        "jobs": [{"id": "talk", "segments": [{
            "type": "image", "src": "card.png", "audio": {"src": "silent.mp4"},
        }]}],
    })
    image, audio = Path("card.png"), Path("silent.mp4")
    probes = {
        image: MediaInfo(image, 0, True, False, 10, 10, None),
        audio: MediaInfo(audio, 5000, True, False, 10, 10, "30/1"),
    }
    with pytest.raises(ValueError, match="has no audio stream"):
        build_plan(manifest.jobs[0], manifest.settings, probes)
