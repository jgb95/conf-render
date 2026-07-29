import pytest

from conf_render.timecodes import ffmpeg_seconds, format_timecode, parse_timecode


def test_parse_and_format_timecode() -> None:
    assert parse_timecode("01:02:03.500") == 3_723_500
    assert parse_timecode("00:00:03") == 3_000
    assert format_timecode(3_723_500) == "01:02:03.500"
    assert ffmpeg_seconds(3_005) == "3.005"


@pytest.mark.parametrize("value", ["1:02:03", "00:60:00", "00:00:01.2", "nope"])
def test_invalid_timecodes(value: str) -> None:
    with pytest.raises(ValueError):
        parse_timecode(value)
