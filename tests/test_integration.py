from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from conf_render.cli import main
from conf_render.probe import probe_media


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg unavailable")
def test_real_render_with_video_image_and_silence(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    image = tmp_path / "card.png"
    overlay = tmp_path / "overlay.png"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=24",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "2", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-c:a", "aac", str(video),
    ], check=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=blue:size=120x120",
        "-frames:v", "1", "-update", "1", str(image),
    ], check=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
        "color=red@0.5:size=160x90,format=rgba", "-frames:v", "1", "-update", "1", str(overlay),
    ], check=True)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "version": 1,
        "settings": {"width": 160, "height": 90, "fps": 24, "transitionMs": 500,
                     "audioSampleRate": 48000, "videoEncoder": "software"},
        "jobs": [{"id": "smoke", "segments": [
            {"type": "image", "src": image.name, "durationMs": 1500},
            {"type": "video", "src": video.name, "overlay": overlay.name},
        ]}],
    }))
    output_dir = tmp_path / "output"
    assert main(["render", str(manifest), "--output", str(output_dir), "--overwrite"]) == 0
    output = output_dir / "smoke.mp4"
    info = probe_media(output)
    assert info.has_video and info.has_audio
    assert 2800 <= info.duration_ms <= 3200


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg unavailable")
@pytest.mark.parametrize("mode", ["replace", "mix"])
def test_real_render_with_external_segment_audio(tmp_path: Path, mode: str) -> None:
    video = tmp_path / "video.mp4"
    audio = tmp_path / "audio.wav"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
        "testsrc2=size=96x54:rate=12", "-f", "lavfi", "-i",
        "sine=frequency=440:sample_rate=48000", "-t", "1.5",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(video),
    ], check=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
        "sine=frequency=880:sample_rate=48000", "-t", "1", str(audio),
    ], check=True)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "version": 1,
        "settings": {"width": 96, "height": 54, "fps": 12, "transitionMs": 500,
                     "audioSampleRate": 48000, "videoEncoder": "software"},
        "jobs": [{"id": mode, "segments": [{
            "type": "video", "src": video.name,
            "audio": {"src": audio.name, "mode": mode, "gainDb": -6},
        }]}],
    }))
    output_dir = tmp_path / "output"
    assert main(["render", str(manifest), "--output", str(output_dir), "--overwrite"]) == 0
    info = probe_media(output_dir / f"{mode}.mp4")
    assert info.has_video and info.has_audio
    assert 1400 <= info.duration_ms <= 1600


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg unavailable")
def test_real_chunked_render_crosses_file_boundary(tmp_path: Path) -> None:
    for index, color in enumerate(("red", "green")):
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
            f"color={color}:size=96x54:rate=12", "-f", "lavfi", "-i",
            f"sine=frequency={440 + index * 220}:sample_rate=48000", "-t", "1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(tmp_path / f"talk_{index:04d}.mp4"),
        ], check=True)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "version": 1,
        "settings": {"width": 96, "height": 54, "fps": 12, "transitionMs": 500,
                     "audioSampleRate": 48000, "videoEncoder": "software"},
        "jobs": [{"id": "chunks", "segments": [{
            "type": "chunkedVideo", "src": "talk_0000.mp4",
            "in": "00:00:00.500", "out": "00:00:01.500",
        }]}],
    }))
    output_dir = tmp_path / "output"
    assert main(["render", str(manifest), "--output", str(output_dir), "--overwrite"]) == 0
    output = output_dir / "chunks.mp4"
    info = probe_media(output)
    assert info.has_video and info.has_audio
    assert 900 <= info.duration_ms <= 1100
    concat = output_dir / ".work" / "chunks" / "segment-000.ffconcat"
    assert "talk_0000.mp4" in concat.read_text()
    assert "talk_0001.mp4" in concat.read_text()