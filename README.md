# conf-render

`conf-render` is a Python 3.12+ CLI for turning images, audio, and video into
consistent conference videos from a declarative JSON manifest. It trims and
joins segments, normalizes media, adds crossfades and overlays, and can generate
subtitles with faster-whisper.

FFmpeg and ffprobe must be available on `PATH`.

## Install

```bash
uv sync
uv run conf-render --help
```

## How manifests work

A manifest defines shared settings and one or more output jobs. These examples
cover every segment type and the most useful options.

### Image and video segments

This job displays a five-second title card, then trims a talk from a longer
recording:

```json
{
  "version": 1,
  "settings": {
    "width": 1920,
    "height": 1080,
    "fps": 30,
    "transitionMs": 1000,
    "imageMs": 4000
  },
  "jobs": [
    {
      "id": "opening-talk",
      "segments": [
        { "type": "image", "src": "title-card.png", "durationMs": 5000 },
        {
          "type": "video",
          "src": "recording.mp4",
          "in": "00:10:00.000",
          "out": "00:45:00.000"
        }
      ]
    }
  ]
}
```

Save it as `conference.json`, then validate, inspect, and render it:

```bash
uv run conf-render validate conference.json
uv run conf-render plan conference.json
uv run conf-render render conference.json --output build/
```

The output is `build/opening-talk.mp4`. Each job ID becomes its output filename.

### Overlay, audio, and transcription

This job replaces source audio, adds an overlay, generates subtitles, and mixes
music into a closing image:

```json
{
  "version": 1,
  "jobs": [
    {
      "id": "keynote",
      "segments": [
        {
          "type": "video",
          "src": "keynote.mp4",
          "overlay": "sponsor-overlay.png",
          "transcribe": true,
          "audio": {
            "src": "clean-mic.wav",
            "mode": "replace",
            "in": "00:00:12.500"
          }
        },
        {
          "type": "image",
          "src": "closing-card.png",
          "durationMs": 10000,
          "audio": {
            "src": "music.mp3",
            "mode": "mix",
            "gainDb": -14
          }
        }
      ]
    }
  ]
}
```

`gainDb` controls external audio; `sourceGainDb` controls source audio in `mix`
mode. Audio is fitted to the segment, while transcription uses the original.

### Chunked recordings

For consecutively numbered files, point `src` to the first chunk. Trimming uses
the joined timeline:

```json
{
  "version": 1,
  "jobs": [
    {
      "id": "afternoon-panel",
      "segments": [
        {
          "type": "chunkedVideo",
          "src": "panel_0000.mp4",
          "in": "00:14:22.500",
          "out": "00:51:08.250",
          "transcribe": true
        }
      ]
    }
  ]
}
```

Chunks must be zero-padded, consecutive, and media-compatible.

### Notes

- Source, overlay, and audio paths are relative to the manifest file.
- Images use `durationMs`, or `settings.imageMs` when omitted.
- Segments crossfade for `transitionMs`; outputs fade from and to black.
- Inputs are normalized to shared settings; missing audio becomes silence.
- Unknown fields, unsafe or duplicate job IDs, invalid timestamps, and missing
  sources are rejected.

See [`examples/talks.json`](examples/talks.json) for a larger real-world manifest.

## Commands and options

- `validate` checks the manifest and source files without running FFmpeg.
- `plan` probes media and prints the resolved millisecond timeline as JSON.
- `render` creates the videos and writes diagnostic plans, probes, filter
  scripts, and FFmpeg commands under the work directory.

```bash
uv run conf-render render manifest.json --output build/ --dry-run
uv run conf-render render manifest.json --output build/ --only talk-1 talk-2
uv run conf-render render manifest.json --output build/ \
  --work-dir build/work --overwrite
```

By default, rendering and transcription run in parallel serial lanes. Pass
`--sequential` to process them one job at a time.

## Transcription

Set `transcribe: true` on a video segment to create timeline-aligned subtitle
files next to the rendered video:

- `<output-stem>.words.srt` — one raw Whisper word per cue
- `<output-stem>.subs.srt` — readable two-line subtitle cards

The default model is `distil-large-v3`. Override it with `--whisper-model`; use
`--whisper-language` to change the default language (`en`). Transcription uses
the original video audio, not replacement or mixed audio.

## Encoding

`settings.videoEncoder` accepts `auto` (default), `software`, `videotoolbox`, or
`nvenc`. Auto prefers VideoToolbox on macOS, NVENC where available elsewhere,
and otherwise libx264. Adjust quality with `softwareCrf`, `softwarePreset`,
`nvencCq`, or `videoBitrate`.

## Tests

```bash
uv sync --extra dev
uv run pytest
```
