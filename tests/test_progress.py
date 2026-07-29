from datetime import datetime
from io import StringIO

from conf_render.progress import TimestampLogger, format_duration, format_realtime_factor


def test_completion_formatters() -> None:
    assert format_duration(12.34) == "12.3s"
    assert format_duration(74.25) == "1m 14.2s"
    assert format_duration(3674.25) == "1h 01m 14.2s"
    assert format_realtime_factor(120_000, 30) == "4.00x realtime"


def test_timestamp_logger_adds_clock_and_optional_job_scope() -> None:
    output = StringIO()
    logger = TimestampLogger(
        stream=output,
        clock=lambda: datetime(2026, 7, 28, 23, 41, 8),
    )

    logger.log("Starting")
    logger.reporter("keynote")("Rendering: 35%")

    assert output.getvalue() == (
        "[23:41:08] Starting\n"
        "[23:41:08] [keynote] Rendering: 35%\n"
    )