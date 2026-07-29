from pathlib import Path

import pytest
from pydantic import ValidationError

from conf_render.models import ChunkedVideoSegment, ImageSegment, Manifest, RenderSettings, VideoSegment, load_manifest


def test_discriminated_segments_and_timestamp_conversion() -> None:
    manifest = Manifest.model_validate({
        "version": 1,
        "jobs": [{"id": "safe-id", "segments": [
            {"type": "video", "src": "talk.mp4", "in": "00:00:01.500"}
        ]}],
    })
    segment = manifest.jobs[0].segments[0]
    assert isinstance(segment, VideoSegment)
    assert segment.in_ms == 1500


def test_nvenc_defaults_to_constant_quality_with_optional_bitrate() -> None:
    settings = RenderSettings()
    assert settings.nvenc_cq == 23
    assert settings.video_bitrate is None
    assert RenderSettings.model_validate({"videoBitrate": "6M"}).video_bitrate == "6M"


def test_image_duration_uses_settings_default_and_allows_override() -> None:
    manifest = Manifest.model_validate({
        "version": 1,
        "settings": {"imageMs": 5000},
        "jobs": [{"id": "images", "segments": [
            {"type": "image", "src": "default.png"},
            {"type": "image", "src": "override.png", "durationMs": 7000},
        ]}],
    })
    default, override = manifest.jobs[0].segments
    assert isinstance(default, ImageSegment)
    assert isinstance(override, ImageSegment)
    assert default.duration_ms == 5000
    assert override.duration_ms == 7000


def test_image_duration_has_builtin_default() -> None:
    manifest = Manifest.model_validate({
        "version": 1,
        "jobs": [{"id": "image", "segments": [{"type": "image", "src": "default.png"}]}],
    })
    segment = manifest.jobs[0].segments[0]
    assert isinstance(segment, ImageSegment)
    assert segment.duration_ms == 4000


@pytest.mark.parametrize("image", [
    {"type": "image", "src": "inherited.png"},
    {"type": "image", "src": "override.png", "durationMs": 500},
])
def test_image_duration_must_cover_transition(image: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="durationMs must be at least transitionMs"):
        Manifest.model_validate({
            "version": 1,
            "settings": {"imageMs": 500, "transitionMs": 1000},
            "jobs": [{"id": "images", "segments": [image, {"type": "video", "src": "video.mp4"}]}],
        })


def test_chunked_video_is_discriminated_and_strict() -> None:
    manifest = Manifest.model_validate({
        "version": 1,
        "jobs": [{"id": "chunks", "segments": [
            {"type": "chunkedVideo", "src": "talk_0000.mp4", "out": "00:20:00"}
        ]}],
    })
    assert isinstance(manifest.jobs[0].segments[0], ChunkedVideoSegment)
    assert manifest.jobs[0].segments[0].out_ms == 1_200_000


@pytest.mark.parametrize("change", [
    {"version": 2},
    {"jobs": [{"id": "bad/id", "segments": [{"type": "image", "src": "x.png", "durationMs": 1000}]}]},
    {"unknown": True},
])
def test_manifest_rejects_invalid_values(change: dict[str, object]) -> None:
    data: dict[str, object] = {
        "version": 1, "jobs": [{
            "id": "valid", "segments": [{"type": "image", "src": "x.png", "durationMs": 1000}]
        }],
    }
    data.update(change)
    with pytest.raises(ValidationError):
        Manifest.model_validate(data)


def test_load_manifest_resolves_sources(tmp_path: Path) -> None:
    (tmp_path / "image.png").touch()
    path = tmp_path / "manifest.json"
    path.write_text('{"version":1,"jobs":[{"id":"x","segments":[{"type":"image","src":"image.png","durationMs":1000}]}]}')
    assert load_manifest(path).jobs[0].segments[0].src == (tmp_path / "image.png").resolve()


def test_load_manifest_resolves_overlay_and_requires_it_to_exist(tmp_path: Path) -> None:
    (tmp_path / "video.mp4").touch()
    (tmp_path / "bug.png").touch()
    path = tmp_path / "manifest.json"
    path.write_text(
        '{"version":1,"jobs":[{"id":"x","segments":['
        '{"type":"video","src":"video.mp4","overlay":"bug.png"}]}]}'
    )
    segment = load_manifest(path).jobs[0].segments[0]
    assert segment.overlay == (tmp_path / "bug.png").resolve()

    (tmp_path / "bug.png").unlink()
    with pytest.raises(ValueError, match="bug.png"):
        load_manifest(path)


def test_segment_audio_parses_options_and_resolves_source(tmp_path: Path) -> None:
    (tmp_path / "image.png").touch()
    (tmp_path / "music.wav").touch()
    path = tmp_path / "manifest.json"
    path.write_text(
        '{"version":1,"jobs":[{"id":"x","segments":['
        '{"type":"image","src":"image.png","audio":{"src":"music.wav",'
        '"mode":"mix","in":"00:00:01.250","gainDb":-12,"sourceGainDb":-3}}]}]}'
    )
    audio = load_manifest(path).jobs[0].segments[0].audio
    assert audio is not None
    assert audio.src == (tmp_path / "music.wav").resolve()
    assert audio.in_ms == 1250
    assert audio.mode == "mix"
    assert audio.gain_db == -12
    assert audio.source_gain_db == -3


def test_replacement_audio_rejects_source_gain() -> None:
    with pytest.raises(ValidationError, match="sourceGainDb is only valid"):
        Manifest.model_validate({
            "version": 1,
            "jobs": [{"id": "x", "segments": [{
                "type": "video", "src": "video.mp4",
                "audio": {"src": "clean.wav", "sourceGainDb": -3},
            }]}],
        })


def test_load_manifest_validates_complete_chunk_sequence(tmp_path: Path) -> None:
    (tmp_path / "talk_0000.mp4").touch()
    (tmp_path / "talk_0002.mp4").touch()
    path = tmp_path / "manifest.json"
    path.write_text(
        '{"version":1,"jobs":[{"id":"x","segments":['
        '{"type":"chunkedVideo","src":"talk_0000.mp4"}]}]}'
    )
    with pytest.raises(ValueError, match="talk_0001.mp4"):
        load_manifest(path)


def test_manifest_rejects_duplicate_job_ids() -> None:
    job = {"id": "same", "segments": [{"type": "image", "src": "x.png", "durationMs": 1000}]}
    with pytest.raises(ValidationError, match="job ids must be unique"):
        Manifest.model_validate({"version": 1, "jobs": [job, job]})
