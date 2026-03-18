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

## Output Artifacts

For each processed frame:
`output/embeddings/<video_name>/frame_XXXX.bin
`

Each `.bin` file contains:
512 float32 values
≈ 2048 bytes


These represent the **CLIP image embedding vector**.

---

## Dependencies

- OpenCV (image processing)
- ONNX Runtime (model inference)
- CLIP Vision Model (`clip_image.onnx`)

The model file is **not committed to the repository** due to size.

---

## Model Setup

Download the CLIP ONNX model manually:

```bash
wget https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/model.onnx
```

Move it to:
```bash
services/embedding/models/clip_image.onnx
```

---

## How to Build

From repository root:
```bash
g++ services/embedding/src/main.cpp -o embed \
-I/usr/include/onnxruntime \
`pkg-config --cflags --libs opencv4` \
-lonnxruntime
```

---

## How to Run

From repository root:
```bash
./embed
```

The service will automatically:

- read frames from output/frames
- generate embeddings
- store results in output/embeddings
