# SIFT-Video

## Semantic Indexing for Timelines in Video

SIFT-Video is a multimodal semantic video search engine that enables natural-language search inside videos and returns the most relevant timestamps based on audio and visual content.

The system processes videos offline, converts audio and visual information into semantic embeddings using pre-trained inference models, and indexes them in a vector database for similarity search at query time.

### Installation

>WIP

### Submodules

This repository uses git submodules for external dependencies.

In particular, the ingestion service depends on `whisper.cpp` for speech-to-text.

After cloning the repository, initialize submodules using:

```bash
git submodule update --init --recursive
```

Or clone the repository with submodules included:
```bash 
git clone --recurse-submodules https://github.com/sourav4243/sift-video.git
```

### Usage

>WIP

### Planned Features

- Offline video ingestion and indexing pipeline
- Audio extraction from video files
- Speech-to-text transcription with timestamps
- Periodic video frame extraction
- Text and image embedding generation
- Multimodal semantic search (audio + visual)
- Timestamp resolution and retrieval
- Fully containerized services

### Technology Stack

#### Languages

- C++ - multimedia processing and model inference
- Rust - query engine, API layer, and vector database interaction

#### Models

- Whisper (whisper.cpp) - speech-to-text with timestamps
- CLIP (via ONNX Runtime) - text and image embeddings

#### Infrastructure & Tooling

- FFmpeg
- Vector database (probably Qdrant)
- Docker

---

## Contributing

Suggestions, fixes and improvments are welcome. Feel free to open an issue or a PR.

## License

This project is licensed under [GNU GPLv3](LICENSE)
