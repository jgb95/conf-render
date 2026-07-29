"""Discovery and validation of consecutively numbered media chunks."""

from __future__ import annotations

import re
from pathlib import Path

_CHUNK_NAME = re.compile(r"^(?P<prefix>.+)_(?P<number>\d{2,})(?P<extension>\.[^.]+)$")


def discover_chunks(first: Path) -> tuple[Path, ...]:
    """Return one complete consecutive sequence beginning with ``first``."""
    match = _CHUNK_NAME.fullmatch(first.name)
    if match is None:
        raise ValueError(
            f"chunked video source must match NAME_NUM.EXT with a zero-padded number: {first}"
        )
    digits = match.group("number")
    if len(digits) < 2 or not digits.startswith("0"):
        raise ValueError(f"chunk number must be zero-padded: {first.name}")
    prefix = match.group("prefix")
    extension = match.group("extension")
    width = len(digits)
    start = int(digits)
    sibling_pattern = re.compile(
        rf"^{re.escape(prefix)}_(?P<number>\d{{{width}}}){re.escape(extension)}$"
    )
    numbered: dict[int, Path] = {}
    for sibling in first.parent.iterdir():
        sibling_match = sibling_pattern.fullmatch(sibling.name)
        if sibling_match is not None and sibling.is_file():
            numbered[int(sibling_match.group("number"))] = sibling.resolve()
    if start not in numbered:
        raise ValueError(f"first chunk does not exist: {first}")
    later = sorted(number for number in numbered if number >= start)
    expected = list(range(start, later[-1] + 1))
    missing = [number for number in expected if number not in numbered]
    if missing:
        names = ", ".join(f"{prefix}_{number:0{width}d}{extension}" for number in missing)
        raise ValueError(f"chunk sequence has missing file(s): {names}")
    return tuple(numbered[number] for number in expected)