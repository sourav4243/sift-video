ARG RUNTIME_BASE=ubuntu:22.04
ARG BUILDER_BASE=ubuntu:22.04

# Build
FROM ${BUILDER_BASE} AS builder
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

ARG USE_CUDA=0

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    wget \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

RUN if [ "$USE_CUDA" = "1" ]; then \
        wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-gpu-1.20.1.tgz && \
        tar -xzf onnxruntime-linux-x64-gpu-1.20.1.tgz && \
        mv onnxruntime-linux-x64-gpu-1.20.1 /onnxruntime && \
        rm onnxruntime-linux-x64-gpu-1.20.1.tgz; \
    else \
        wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz && \
        tar -xzf onnxruntime-linux-x64-1.20.1.tgz && \
        mv onnxruntime-linux-x64-1.20.1 /onnxruntime && \
        rm onnxruntime-linux-x64-1.20.1.tgz; \
    fi

RUN mkdir -p /app/models && \
    wget -qO /app/models/visual.onnx "https://huggingface.co/RuteNL/MobileCLIP2-S3-OpenCLIP-ONNX/resolve/main/visual.onnx" && \
    wget -qO /app/models/visual.onnx.data "https://huggingface.co/RuteNL/MobileCLIP2-S3-OpenCLIP-ONNX/resolve/main/visual.onnx.data"

COPY src ./src
RUN if [ "$USE_CUDA" = "1" ]; then \
        g++ src/main.cpp -o embed \
            -DUSE_CUDA=1 \
            -I/onnxruntime/include \
            -L/onnxruntime/lib \
            -lonnxruntime \
            `pkg-config --cflags --libs opencv4`; \
    else \
        g++ src/main.cpp -o embed \
            -I/onnxruntime/include \
            -L/onnxruntime/lib \
            -lonnxruntime \
            `pkg-config --cflags --libs opencv4`; \
    fi

# Runtime
FROM ${RUNTIME_BASE}
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-core4.5d \
    libopencv-imgproc4.5d \
    libopencv-imgcodecs4.5d \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/models ./models
COPY --from=builder /app/embed .
COPY --from=builder /onnxruntime/lib /onnxruntime/lib

ENV LD_LIBRARY_PATH=/onnxruntime/lib
CMD ["./embed"]