"""Faster-whisper transcription and timeline-aligned SRT artifacts."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Protocol

from conf_render.timecodes import ffmpeg_seconds
from conf_render.timeline import PlannedSegment, RenderPlan

MAX_CHARS = 42
MIN_DURATION_MS = 833
MAX_DURATION_MS = 7000
MIN_GAP_MS = 80
PAUSE_BREAK_MS = 300
HEARTBEAT_SECONDS = 30

Progress = Callable[[str], None]


class WhisperPipeline(Protocol):
    def transcribe(self, audio: str, **kwargs: object) -> tuple[Iterable[Any], Any]: ...


@dataclass(frozen=True)
class WhisperRuntime:
    pipeline: WhisperPipeline
    model_name: str
    device: str
    compute_type: str


@dataclass(frozen=True)
class Word:
    text: str
    start_ms: int
    end_ms: int
    segment_index: int | None = None


@dataclass(frozen=True)
class Cue:
    start_ms: int
    end_ms: int
    text: str


def seconds_to_milliseconds(value: object) -> int:
    """Convert external timestamp seconds to integer milliseconds once."""
    return int((Decimal(str(value)) * 1000).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def srt_timestamp(milliseconds: int) -> str:
    """Format integer milliseconds as an SRT timestamp."""
    hours, remainder = divmod(max(0, milliseconds), 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def words_to_subtitles(words: list[Word]) -> list[Cue]:
    """Group word timestamps into readable two-line subtitle cues."""
    cues: list[Cue] = []
    lines: list[list[str]] = [[]]
    start_ms: int | None = None
    end_ms = 0
    previous_word_end: int | None = None

    def flush() -> None:
        nonlocal lines, start_ms, end_ms
        if start_ms is None or not lines[0]:
            return
        start = start_ms
        if cues:
            start = max(start, cues[-1].end_ms + MIN_GAP_MS)
        end = max(end_ms, start + MIN_DURATION_MS)
        text = "\n".join(" ".join(line) for line in lines if line)
        cues.append(Cue(start, end, text))
        lines = [[]]
        start_ms = None
        end_ms = 0

    for word in words:
        text = word.text.strip()
        if not text:
            continue
        if previous_word_end is not None and word.start_ms - previous_word_end >= PAUSE_BREAK_MS:
            flush()
        if start_ms is None:
            start_ms = word.start_ms
        target = lines[-1]
        candidate = " ".join([*target, text])
        if len(candidate) <= MAX_CHARS:
            target.append(text)
        elif len(lines) == 1 and len(text) <= MAX_CHARS:
            lines.append([text])
        else:
            flush()
            start_ms = word.start_ms
            lines = [[text]]
        end_ms = word.end_ms
        previous_word_end = word.end_ms
        if end_ms - start_ms >= MAX_DURATION_MS or text.endswith((".", "!", "?")):
            flush()
    flush()
    return cues


def write_srt(cues: Iterable[Cue], path: Path) -> None:
    """Write readable subtitle cues in SubRip format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = [
        f"{index}\n{srt_timestamp(cue.start_ms)} --> {srt_timestamp(cue.end_ms)}\n{cue.text}"
        for index, cue in enumerate(cues, start=1)
        if cue.text.strip()
    ]
    body = "\n\n".join(blocks)
    path.write_text(body + ("\n" if body else ""), encoding="utf-8")


def write_word_srt(words: Iterable[Word], path: Path) -> None:
    """Write one raw, timeline-aligned Whisper word per SubRip cue."""
    write_srt(
        (
            Cue(word.start_ms, max(word.end_ms, word.start_ms + 1), word.text.strip())
            for word in words
            if word.text.strip()
        ),
        path,
    )


def load_whisper_runtime(model_name: str, progress: Progress) -> WhisperRuntime:
    """Load a batched model using CUDA float16 or efficient CPU int8."""
    try:
        import ctranslate2
        from faster_whisper import BatchedInferencePipeline, WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is unavailable; reinstall conf-render to restore "
            "its required transcription dependencies"
        ) from exc

    cuda_devices = ctranslate2.get_cuda_device_count()
    device = "cuda" if cuda_devices else "cpu"
    compute_type = "float16" if cuda_devices else "int8"
    cpu_threads = 0 if cuda_devices else (os.cpu_count() or 4)
    thread_note = f", {cpu_threads} threads" if cpu_threads else ""
    progress(f"Loading Whisper model {model_name} on {device} ({compute_type}{thread_note})...")
    started = time.monotonic()
    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
    )
    progress(f"Whisper model ready in {time.monotonic() - started:.1f}s")
    return WhisperRuntime(BatchedInferencePipeline(model=model), model_name, device, compute_type)


def _extract_audio(segment: PlannedSegment, target: Path, concat_file: Path | None = None) -> None:
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", ffmpeg_seconds(segment.source_in_ms), "-t", ffmpeg_seconds(segment.duration_ms),
    ]
    if segment.chunks:
        if concat_file is None:
            raise RuntimeError(f"missing concat list for transcription segment {segment.index}")
        command.extend(["-f", "concat", "-safe", "0", "-i", str(concat_file)])
    else:
        command.extend(["-i", str(segment.src)])
    command.extend(["-vn", "-ar", "16000", "-ac", "1", str(target)])
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg was not found on PATH") from exc
    if result.returncode:
        raise RuntimeError(f"audio extraction failed for {segment.src}: {result.stderr.strip()}")


def _consume_with_heartbeat(
    segments: Iterable[Any],
    label: str,
    progress: Progress,
    interval: float = HEARTBEAT_SECONDS,
) -> list[Any]:
    done = threading.Event()

    def heartbeat() -> None:
        while not done.wait(interval):
            progress(f"{label}: still transcribing...")

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        return list(segments)
    finally:
        done.set()
        thread.join()


def transcribe_plan(
    plan: RenderPlan,
    output: Path,
    work_dir: Path,
    *,
    model_name: str = "distil-large-v3",
    language: str = "en",
    concat_files: dict[int, Path] | None = None,
    progress: Progress | None = None,
    runtime: WhisperRuntime | None = None,
) -> tuple[Path, Path] | None:
    """Transcribe selected segments and write word-level and readable SRT files."""
    selected = [segment for segment in plan.segments if segment.transcribe]
    if not selected:
        return None
    report = progress or (lambda _message: None)
    whisper = runtime or load_whisper_runtime(model_name, report)
    words: list[Word] = []
    audio_dir = work_dir / "transcription"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for segment in selected:
        if not segment.has_audio:
            raise RuntimeError(f"cannot transcribe segment {segment.index}: source has no audio stream")
        label = f"Segment {segment.index}"
        audio_path = audio_dir / f"segment-{segment.index:03d}.wav"
        report(f"{label}: extracting {segment.duration_ms / 1000:.1f}s of audio...")
        _extract_audio(segment, audio_path, (concat_files or {}).get(segment.index))
        report(f"{label}: transcribing...")
        segments, _ = whisper.pipeline.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"speech_pad_ms": 200, "min_silence_duration_ms": 300},
            batch_size=16,
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=True,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
        recognized = 0
        for transcript_segment in _consume_with_heartbeat(segments, label, report):
            for item in transcript_segment.words or []:
                text = item.word.strip()
                if text:
                    words.append(Word(
                        text=text,
                        start_ms=segment.timeline_start_ms + seconds_to_milliseconds(item.start),
                        end_ms=segment.timeline_start_ms + seconds_to_milliseconds(item.end),
                        segment_index=segment.index,
                    ))
                    recognized += 1
        report(f"{label}: recognized {recognized} words")
    words.sort(key=lambda word: (word.start_ms, word.end_ms))
    cues = words_to_subtitles(words)
    words_path = output.with_name(f"{output.stem}.words.srt")
    subs_path = output.with_name(f"{output.stem}.subs.srt")
    write_word_srt(words, words_path)
    write_srt(cues, subs_path)
    report(f"Wrote word-level SRT: {words_path}")
    report(f"Wrote readable SRT subtitles: {subs_path}")
    return words_path, subs_path