import platform
from pathlib import Path

import pytest

from conf_render.ffmpeg import _video_encoder_args, build_command, build_filter_graph, resolve_video_encoder
from conf_render.timeline import PlannedAudio, PlannedSegment, RenderPlan


def make_overlay_plan() -> RenderPlan:
    segments = (
        PlannedSegment(
            0, "video", Path("first.mp4"), 3000, 0, 3000, 0, 3000,
            True, False, overlay=Path("bug.png"),
        ),
        PlannedSegment(
            1, "image", Path("card.png"), 2000, 0, 2000, 2000, 4000,
            False, False,
        ),
    )
    return RenderPlan(
        "test", 1920, 1080, 30, 1000, 48000, "software", None, 23,
        "medium", 23, 4000, segments,
    )


def make_audio_plan(mode: str = "replace", *, has_audio: bool = True) -> RenderPlan:
    segment = PlannedSegment(
        0, "video", Path("first.mp4"), 3000, 0, 3000, 0, 3000,
        has_audio, False, overlay=Path("bug.png"),
        audio=PlannedAudio(Path("clean.wav"), mode, 500, -6, -3 if mode == "mix" else 0),
    )
    return RenderPlan(
        "test", 1920, 1080, 30, 1000, 48000, "software", None, 23,
        "medium", 23, 3000, (segment,),
    )


def test_overlay_filter_uses_extra_input_before_following_segment() -> None:
    graph = build_filter_graph(make_overlay_plan())
    assert "[0:v]trim=duration=3" in graph
    assert "[1:v]trim=duration=3" in graph
    assert "[vn0][overlay0]overlay=0:0:shortest=1:format=auto" in graph
    assert "[2:v]trim=duration=2" in graph


def test_overlay_image_is_added_as_looped_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("conf_render.ffmpeg.resolve_video_encoder", lambda requested: "libx264")
    command = build_command(
        make_overlay_plan(), Path("output.mp4"), Path("filter.txt"), True,
    )
    overlay_index = command.index("bug.png")
    assert command[overlay_index - 7:overlay_index + 1] == [
        "-loop", "1", "-framerate", "30", "-t", "3.000", "-i", "bug.png",
    ]


def test_replacement_audio_uses_input_after_visual_overlay() -> None:
    graph = build_filter_graph(make_audio_plan())
    assert "[2:a]asetpts=PTS-STARTPTS" in graph
    assert "volume=-6dB,apad,atrim=duration=3.000[aexternal0]" in graph
    assert "[aexternal0]anull[a0]" in graph
    assert "[0:a]" not in graph


@pytest.mark.parametrize("has_audio", [True, False])
def test_mixed_audio_combines_source_or_silence(has_audio: bool) -> None:
    graph = build_filter_graph(make_audio_plan("mix", has_audio=has_audio))
    expected_source = "[0:a]atrim" if has_audio else "anullsrc=r=48000:cl=stereo"
    assert expected_source in graph
    assert "[asource0]volume=-3dB[amixsource0]" in graph
    assert "[amixsource0][aexternal0]amix=inputs=2:duration=first:normalize=0[a0]" in graph


def test_external_audio_input_uses_configured_trim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("conf_render.ffmpeg.resolve_video_encoder", lambda requested: "libx264")
    command = build_command(make_audio_plan(), Path("output.mp4"), Path("filter.txt"), True)
    audio_index = command.index("clean.wav")
    assert command[audio_index - 5:audio_index + 1] == [
        "-ss", "0.500", "-t", "3.000", "-i", "clean.wav",
    ]


def test_explicit_encoder_resolution() -> None:
    assert resolve_video_encoder("software", {"libx264"}) == "libx264"
    assert resolve_video_encoder("videotoolbox", {"h264_videotoolbox"}) == "h264_videotoolbox"
    assert resolve_video_encoder("nvenc", {"h264_nvenc"}) == "h264_nvenc"


def test_unavailable_explicit_encoder_fails() -> None:
    with pytest.raises(RuntimeError, match="unavailable"):
        resolve_video_encoder("nvenc", {"libx264"})


def test_auto_encoder_prefers_platform_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    available = {"libx264", "h264_nvenc", "h264_videotoolbox"}
    assert resolve_video_encoder("auto", available) == "h264_videotoolbox"
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    assert resolve_video_encoder("auto", available) == "h264_nvenc"


def test_videotoolbox_uses_legacy_quality_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "conf_render.ffmpeg.resolve_video_encoder",
        lambda requested: "h264_videotoolbox",
    )
    plan = RenderPlan(
        id="test",
        width=1920,
        height=1080,
        fps=30,
        transition_ms=1000,
        audio_sample_rate=48000,
        video_encoder="videotoolbox",
        video_bitrate=None,
        nvenc_cq=23,
        software_preset="medium",
        software_crf=23,
        duration_ms=1000,
        segments=(),
    )

    assert _video_encoder_args(plan) == [
        "-c:v", "h264_videotoolbox", "-q:v", "65",
    ]


@pytest.mark.parametrize(
    ("bitrate", "rate_control"),
    [
        (None, ["-rc", "vbr", "-cq", "23", "-b:v", "0"]),
        ("6M", ["-b:v", "6M"]),
    ],
)
def test_nvenc_rate_control(
    monkeypatch: pytest.MonkeyPatch,
    bitrate: str | None,
    rate_control: list[str],
) -> None:
    monkeypatch.setattr(
        "conf_render.ffmpeg.resolve_video_encoder",
        lambda requested: "h264_nvenc",
    )
    plan = RenderPlan(
        id="test",
        width=1920,
        height=1080,
        fps=30,
        transition_ms=1000,
        audio_sample_rate=48000,
        video_encoder="nvenc",
        video_bitrate=bitrate,
        nvenc_cq=23,
        software_preset="medium",
        software_crf=23,
        duration_ms=1000,
        segments=(),
    )

    assert _video_encoder_args(plan) == [
        "-c:v", "h264_nvenc", "-preset", "p4", *rate_control,
        "-profile:v", "high", "-pix_fmt", "yuv420p",
    ]