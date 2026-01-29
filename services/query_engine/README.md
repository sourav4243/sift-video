# Query Engine

This service accepts natural language queries, converts them into vector embeddings (WIP) and performs parallel searches across Audio and Visual collections in Qdrant.

- Queries `audio_segments` and `visual_frames` simultaneously.
- Merges results with custom weighting (Audio: 0.6, Visual: 0.4).

## How to Run

### Option 1: With Docker

From the root of the repo:

```bash
docker-compose up --build
```

### Option 2: Local Development (only running qdrant in docker)

1. Start Qdrant:

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

2. Run the Service:

```bash
cargo run
```

## API Usage

**Endpoint:** ```POST /search```

**Request:**

```json
{
    "query": "iron man flying"
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
            "match_type": "audio",
            "match_context": "I am Iron Man"
        }
    ]
}
```