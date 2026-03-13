# Embedding Service

The Embedding Service is responsible for converting extracted video frames into
**vector embeddings** using the **CLIP vision encoder**.

It transforms raw image frames into **512-dimensional semantic vectors**
that can later be indexed inside a vector database for similarity search.

This service runs **fully offline in C++** using **OpenCV + ONNX Runtime**.

---

## What this service does

For each video frame extracted by the ingestion service:

1. **Frame Loading**
   - Reads `frame_XXXX.jpg` files generated during ingestion

2. **Image Preprocessing**
   - Resize to `224 × 224`
   - Convert BGR → RGB
   - Normalize pixel values

3. **Embedding Generation**
   - Runs the **CLIP Vision Transformer (ViT-B/32)** via ONNX Runtime
   - Produces a **512-dimensional embedding vector**

4. **Embedding Storage**
   - Saves embeddings as binary files (`.bin`)
   - Each file contains **512 float32 values**

---

## Responsibilities

- Scan `/output/frames` for extracted frames
- Preprocess images using OpenCV
- Run CLIP inference using ONNX Runtime
- Generate visual embeddings
- Store embeddings in `/output/embeddings`

This service does **not** perform:

- video decoding
- transcription
- vector database insertion
- search queries

---

## Directory Contract

The service expects the following directory layout:

```
output/
├── frames/
│ └── <video_name>/
│ ├── frame_0001.jpg
│ ├── frame_0002.jpg
│ └── frames.json
│
└── embeddings/
└── <video_name>/
```
These paths must remain consistent across services.

---
