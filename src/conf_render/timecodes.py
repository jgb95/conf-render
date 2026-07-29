"""Exact parsing and formatting for timeline timestamps."""

from __future__ import annotations

import re

_TIMECODE = re.compile(r"^(?P<hours>\d{2,}):(?P<minutes>\d{2}):(?P<seconds>\d{2})(?:\.(?P<millis>\d{3}))?$")


def parse_timecode(value: str) -> int:
    """Parse ``HH:MM:SS[.mmm]`` into integer milliseconds."""
    match = _TIMECODE.fullmatch(value)
    if not match:
        raise ValueError("timecode must use HH:MM:SS or HH:MM:SS.mmm")
    hours = int(match.group("hours"))
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    if minutes >= 60 or seconds >= 60:
        raise ValueError("timecode minutes and seconds must be between 00 and 59")
    millis = int(match.group("millis") or 0)
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


def format_timecode(milliseconds: int) -> str:
    """Format non-negative integer milliseconds as ``HH:MM:SS.mmm``."""
    if isinstance(milliseconds, bool) or not isinstance(milliseconds, int) or milliseconds < 0:
        raise ValueError("milliseconds must be a non-negative integer")
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def ffmpeg_seconds(milliseconds: int) -> str:
    """Return exact decimal seconds suitable for an FFmpeg argument."""
    if milliseconds < 0:
        raise ValueError("milliseconds must be non-negative")
    seconds, millis = divmod(milliseconds, 1000)
    return f"{seconds}.{millis:03d}"
