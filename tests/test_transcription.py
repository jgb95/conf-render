import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from conf_render.timeline import PlannedSegment, RenderPlan
from conf_render.transcription import (
    Cue,
    WhisperRuntime,
    Word,
    load_whisper_runtime,
    srt_timestamp,
    transcribe_plan,
    words_to_subtitles,
    write_srt,
    write_word_srt,
)


def make_plan(source: Path, *, timeline_start_ms: int = 2000) -> RenderPlan:
    segment = PlannedSegment(
        index=1,
        type="video",
        src=source,
        duration_ms=5000,
        source_in_ms=1500,
        source_out_ms=6500,
        timeline_start_ms=timeline_start_ms,
        timeline_end_ms=timeline_start_ms + 5000,
        has_audio=True,
        transcribe=True,
    )
    return RenderPlan(
        "talk", 1920, 1080, 30, 1000, 48000,
        "software", None, 23, "medium", 23,
        timeline_start_ms + 5000, (segment,),
    )


def test_subtitle_cards_obey_line_and_duration_limits() -> None:
    words = [Word("hello", 0, 100), Word("world.", 120, 300), Word("again", 700, 900)]
    cues = words_to_subtitles(words)
    assert cues[0].text == "hello world."
    assert cues[0].end_ms - cues[0].start_ms >= 833
    assert cues[1].start_ms >= cues[0].end_ms + 80
    assert all(len(line) <= 42 for cue in cues for line in cue.text.splitlines())


def test_srt_writers_produce_word_and_readable_artifacts(tmp_path: Path) -> None:
    assert srt_timestamp(3_723_500) == "01:02:03,500"
    words_path = tmp_path / "talk.words.srt"
    subs_path = tmp_path / "talk.subs.srt"
    write_word_srt(
        [Word("Hello", 2125, 2500), Word("brief", 2500, 2500), Word("world.", 2550, 2900)],
        words_path,
    )
    write_srt([Cue(2125, 2958, "Hello world.")], subs_path)
    assert words_path.read_text() == (
        "1\n00:00:02,125 --> 00:00:02,500\nHello\n\n"
        "2\n00:00:02,500 --> 00:00:02,501\nbrief\n\n"
        "3\n00:00:02,550 --> 00:00:02,900\nworld.\n"
    )
    assert subs_path.read_text() == "1\n00:00:02,125 --> 00:00:02,958\nHello world.\n"


def test_cpu_runtime_uses_int8_and_batched_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    model_calls: list[dict[str, object]] = []
    model = object()
    pipeline = object()

    class FakeModel:
        def __new__(cls, name: str, **kwargs: object) -> object:
            model_calls.append({"name": name, **kwargs})
            return model

    class FakePipeline:
        def __new__(cls, *, model: object) -> object:
            assert model is not None
            return pipeline

    monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(get_cuda_device_count=lambda: 0))
    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        SimpleNamespace(WhisperModel=FakeModel, BatchedInferencePipeline=FakePipeline),
    )
    monkeypatch.setattr("conf_render.transcription.os.cpu_count", lambda: 8)

    runtime = load_whisper_runtime("tiny", lambda _message: None)

    assert runtime == WhisperRuntime(pipeline, "tiny", "cpu", "int8")
    assert model_calls == [{"name": "tiny", "device": "cpu", "compute_type": "int8", "cpu_threads": 8}]


def test_transcribe_plan_aligns_words_and_writes_required_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcribe_kwargs: dict[str, object] = {}

    class FakePipeline:
        def transcribe(self, audio: str, **kwargs: object) -> tuple[list[object], object]:
            assert audio.endswith("segment-001.wav")
            transcribe_kwargs.update(kwargs)
            words = [
                SimpleNamespace(word=" Hello", start=0.125, end=0.5),
                SimpleNamespace(word="world.", start=0.55, end=0.9),
            ]
            return [SimpleNamespace(words=words)], object()

    monkeypatch.setattr("conf_render.transcription._extract_audio", lambda *_args, **_kwargs: None)
    plan = make_plan(tmp_path / "source.mp4")
    runtime = WhisperRuntime(FakePipeline(), "tiny", "cpu", "int8")
    output = tmp_path / "talk.mp4"

    paths = transcribe_plan(plan, output, tmp_path / "work", runtime=runtime)

    assert paths == (
        tmp_path / "talk.words.srt",
        tmp_path / "talk.subs.srt",
    )
    assert transcribe_kwargs["batch_size"] == 16
    assert transcribe_kwargs["beam_size"] == 1
    assert transcribe_kwargs["vad_parameters"] == {
        "speech_pad_ms": 200,
        "min_silence_duration_ms": 300,
    }
    assert "00:00:02,125 --> 00:00:02,500\nHello" in paths[0].read_text()
    assert "00:00:02,125 --> 00:00:02,958\nHello world." in paths[1].read_text()
