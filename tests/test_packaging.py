"""Rendering-only installs should not require a transcription runtime."""
import sys
import tomllib
from pathlib import Path

import pytest

from conf_render.transcription import load_whisper_runtime


def test_transcription_dependencies_are_an_explicit_extra():
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())["project"]
    assert not any(value.startswith("faster-whisper") for value in project["dependencies"])
    assert "faster-whisper>=1.1" in project["optional-dependencies"]["speech"]


def test_missing_transcription_runtime_explains_how_to_install_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "ctranslate2", None)
    with pytest.raises(RuntimeError, match=r"conf-render\[speech\]"):
        load_whisper_runtime("base", lambda message: None)
