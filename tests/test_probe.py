from pathlib import Path

from conf_render.probe import MediaInfo, probe_chunked_media, parse_probe


def test_parse_probe_metadata() -> None:
    info = parse_probe({
        "streams": [
            {
                "codec_type": "video", "width": 1920, "height": 1080,
                "r_frame_rate": "30000/1001", "avg_frame_rate": "2997/100",
            },
            {"codec_type": "audio"},
        ],
        "format": {"duration": "12.3456"},
    }, Path("talk.mp4"))
    assert info.duration_ms == 12_346
    assert info.has_video and info.has_audio
    assert info.frame_rate == "30000/1001"


def test_still_image_can_have_no_duration() -> None:
    info = parse_probe({"streams": [{"codec_type": "video", "width": 10, "height": 10}]}, Path("x.png"))
    assert info.duration_ms == 0


def test_chunked_probe_aggregates_duration_and_paths() -> None:
    paths = (Path("talk_0000.mp4"), Path("talk_0001.mp4"))

    def fake_probe(path: Path) -> MediaInfo:
        return MediaInfo(path, 600_000, True, True, 1920, 1080, "30000/1001")

    info = probe_chunked_media(paths[0], paths, probe=fake_probe)
    assert info.duration_ms == 1_200_000
    assert info.chunks == paths
