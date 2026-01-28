# SIFT-Video

## Semantic Indexing for Timelines in Video

SIFT-Video is a multimodal semantic video search engine that enables natural-language search inside videos and returns the most relevant timestamps based on audio and visual content.

The system processes videos offline, converts audio and visual information into semantic embeddings using pre-trained inference models, and indexes them in a vector database for similarity search at query time.

### Quick Start

> WIP

Clone the repository and run the setup script:

```bash
git clone https://github.com/sourav4243/sift-video.git
cd sift-video

chmod +x setup.sh
./setup.sh
```

### Installation (Manual Setup)

**Prerequisites**

- [docker](https://docs.docker.com/get-docker/) and docker-compose
- git

**1. Clone the repository and setup environment**

Use `--recurse-submodules` for external dependencies (ingestion service depends on `whisper.cpp`) 

```bash 
git clone --recurse-submodules https://github.com/sourav4243/sift-video.git
cd sift-video
mkdir -p videos output
```

> Note: If you cloned without submodules, you can fix it by running:

```bash
git submodule update --init --recursive
```

#### Configuration

> The following environment variables can be changed in `docker-compose.yml`

| Variable | Default | Description |
| :--- | :--- | :--- |
| `QDRANT_URL` | `http://localhost:6334` | URL of the Qdrant gRPC interface. Use `http://qdrant:6334` inside Docker. |
| `RUST_LOG` | `info` | Logging level. |

---

> Note: You can access Qdrant web dashboard at http://localhost:6333/dashboard

### Usage

1. **Prepare your videos:**
Place the video files you want to index into the videos/ directory at project root.

2. **Start the services:**
Run the following command to build and start the indexing pipeline and search API:

```bash
docker-compose up --build
```

This spins up three containers:

- `sift_qdrant`: The vector database (Ports: 6333, 6334).
- `sift_ingestion`: Processes videos from the `videos/` folder and saves transcripts to `output/`.
- `sift_query_engine`: The search API (Port: 8080).

3. **Search via API:**
Once the system is running, you can search your indexed videos using HTTP API:

```bash
curl -X POST http://localhost:8080/search \
    -H "Content-Type: application/json" \
    -d '{"query": "what is the meaning of life"}'
```

> Note: A dedicated CLI tool for easier searching is planned.

### Planned Features

- [ ] Offline video ingestion and indexing pipeline
- [ ] Audio extraction from video files
- [ ] Speech-to-text transcription with timestamps
- [ ] Periodic video frame extraction
- [ ] Text and image embedding generation
- [ ] Multimodal semantic search (audio + visual)
- [ ] Timestamp resolution and retrieval
- [ ] Fully containerized services
- [ ] CLI tool for natural language search

### Technology Stack

#### Languages

- C++ - multimedia processing and model inference
- Rust - query engine, API layer, and vector database interaction

#### Models

- Whisper (whisper.cpp) - speech-to-text with timestamps
- CLIP (via ONNX Runtime) - text and image embeddings

#### Infrastructure & Tooling

- FFmpeg
- Vector database (Qdrant)
- Docker

---

## Contributing

Suggestions, fixes and improvements are welcome. Feel free to open an issue or a PR.

## License

This project is licensed under [GNU GPLv3](LICENSE)
