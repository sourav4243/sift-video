# Sift Video Query Engine

[![Build Status](https://img.shields.io/github/actions/workflow/status/sourav4243/sift-video/docker.yml?branch=main&style=flat-square)](https://github.com/sourav4243/sift-video/actions)
[![Docker Image Version](https://img.shields.io/docker/v/meledo/sift-video-query-engine?sort=semver&style=flat-square)](https://hub.docker.com/r/meledo/sift-video-query-engine)

A multimodal search orchestrator built in Rust with Axum. This service handles YouTube downloads, triggers ingestion pipelines, dynamically loads ONNX text embeddings, and executes parallel vector searches against Qdrant.

## Features

- **Text Embeddings**: Lazy-loads `MobileCLIP2-S3-OpenCLIP-ONNX` into memory for text queries, evicting after 120 seconds of idle time.
- **Multimodal Search**: Queries `audio_segments` and `visual_frames` simultaneously. Merges temporal results within a 2.0s window.
- **Custom Weighting**: Audio score weight is `0.3`, visual score weight is `0.7`.
- **Ingestion Pipeline**: Integrated `yt-dlp` for downloads and filesystem-based trigger orchestration for the C++ ingestion and embedding workers.
- **Filtering**: Search by specific modality (`audio`, `visual`, `all`) or restrict searches to a specific video ID.

## Installation & Setup

### 1. Run with Docker Compose
*Note: Run these commands from the root directory of the repository.*

#### Standard (CPU):

```bash
docker-compose up -d --build
```

#### GPU Offload (NVIDIA):
Requires the `nvidia-container-toolkit` to be installed on the host machine.

```bash
docker-compose -f docker-compose.gpu.yml up -d --build
```

### 2. Local Development
Run Qdrant via Docker and start the Rust server natively:

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
cargo run
```

#### Environment Variables
If running locally via `cargo run`, you may need to specify the following environment variables (Docker handles these automatically):
- `QDRANT_URL` (default: `http://localhost:6334`)
- `OUTPUT_DIR` (default: `/output` - used for ingestion triggers and embeddings)

## API Endpoints

### `POST /search`  
Executes a multimodal vector search.

**Request:**

```json
{
    "query": "iron man flying",
    "video_id": null,
    "match_type": "all"
}
```

**Response:**

```json
{
    "results": [
        {
            "video_id": "vid_001",
            "video_name": "iron_man.mp4",
            "timestamp": 45.2,
            "score": 0.89,
            "match_type": "audio+visual",
            "match_context": "I am Iron Man"
        }
    ]
}
```

### `POST /download`  
Downloads a video via `yt-dlp` and triggers the ingestion pipeline.

**Request:**

```json
{
    "url": "https://youtube.com/watch?v=..."
}
```

### `GET /videos`  
Returns a list of all indexed videos and their respective audio/visual segment counts.

### `DELETE /videos/:video_id`  
Removes a video from the file system and deletes all associated embeddings from the Qdrant collections.