from pathlib import Path

import pytest

from conf_render.chunks import discover_chunks


def _touch(path: Path) -> Path:
    path.touch()
    return path


def test_discovers_zero_based_consecutive_chunks(tmp_path: Path) -> None:
    expected = tuple(_touch(tmp_path / f"talk_{index:04d}.mp4").resolve() for index in range(3))
    _touch(tmp_path / "other_0003.mp4")
    assert discover_chunks(expected[0]) == expected


def test_discovers_from_declared_nonzero_first_chunk(tmp_path: Path) -> None:
    _touch(tmp_path / "talk_0000.mp4")
    expected = tuple(_touch(tmp_path / f"talk_{index:04d}.mp4").resolve() for index in range(1, 3))
    assert discover_chunks(expected[0]) == expected


def test_rejects_gap_before_later_matching_file(tmp_path: Path) -> None:
    first = _touch(tmp_path / "talk_0000.mp4")
    _touch(tmp_path / "talk_0002.mp4")
    with pytest.raises(ValueError, match="talk_0001.mp4"):
        discover_chunks(first)