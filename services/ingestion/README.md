# Ingestion Service

The Ingestion Service is responsible for **offline preprocessing of raw video files**
and converting them into **structured, timestamped transcripts**.

It acts as the **entry point of the multimedia search pipeline**.

---

## What this service does

For each video placed in the `/videos` directory, the service performs:

1. **Video - Audio extraction**
   - Uses FFmpeg to extract mono 16kHz WAV audio

2. **Audio - Text transcription**
   - Uses `whisper.cpp` (offline, no Python)
   - Generates `.srt`, `.txt`, and `.vtt` files

3. **Transcript normalization**
   - Parses Whisper-generated `.srt`
   - Produces a unified `transcripts.json` file

All steps run **inside a Docker container**.

---

## Responsibilities

- Scan `/videos` for input video files
- Extract audio using FFmpeg
- Transcribe speech using Whisper (C++ implementation)
- Generate structured transcript metadata
- Write outputs to `/output`

This service does **not** perform:
- embeddings
- vector indexing
- search or retrieval

---

## Directory Contract

The container expects the following mounted volumes:

- `/videos` - input directory containing raw video files
- `/output` - directory for generated artifacts

These paths are **hard contracts** and must not be changed.

---

## Output Artifacts

For each input video:

- `<video>.wav` - extracted audio
- `<video>.srt` - timestamped subtitles
- `<video>.txt` - plain text transcript
- `<video>.vtt` - WebVTT transcript

Additionally:

- `transcripts.json` - aggregated, schema-compliant transcript metadata

---

## Dependencies

- FFmpeg
- `whisper.cpp` (added as a git submodule)

Whisper models are downloaded at runtime and **must not be committed**.

---

## How to Build

From repository root:

```bash
docker build -t ingestion services/ingestion

```
> If you encounter build issues due to cached layers, rebuild with `--no-cache`.

---

## How to Run
From repository root:
```bash
docker run --rm \
  -v "$(pwd)/videos:/videos" \
  -v "$(pwd)/output:/output" \
  ingestion

```

---