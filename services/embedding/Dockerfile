ARG RUNTIME_BASE=ubuntu:24.04
ARG BUILDER_BASE=ubuntu:24.04

FROM ${BUILDER_BASE} AS builder

ARG USE_CUDA=0

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    wget \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

RUN if [ "$USE_CUDA" = "1" ]; then \
        wget https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/onnxruntime-linux-x64-gpu_cuda13-1.24.4.tgz && \
        tar -xzf onnxruntime-linux-x64-gpu_cuda13-1.24.4.tgz && \
        mv onnxruntime-linux-x64-gpu-1.24.4 /onnxruntime && \
        rm onnxruntime-linux-x64-gpu_cuda13-1.24.4.tgz; \
    else \
        wget https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/onnxruntime-linux-x64-1.24.4.tgz && \
        tar -xzf onnxruntime-linux-x64-1.24.4.tgz && \
        mv onnxruntime-linux-x64-1.24.4 /onnxruntime && \
        rm onnxruntime-linux-x64-1.24.4.tgz; \
    fi

WORKDIR /app

RUN --mount=type=cache,target=/model-cache \
    mkdir -p /app/models && \
    if [ ! -f /model-cache/visual.onnx ]; then \
        wget -qO /model-cache/visual.onnx \
            "https://huggingface.co/RuteNL/MobileCLIP2-S3-OpenCLIP-ONNX/resolve/main/visual.onnx" && \
        wget -qO /model-cache/visual.onnx.data \
            "https://huggingface.co/RuteNL/MobileCLIP2-S3-OpenCLIP-ONNX/resolve/main/visual.onnx.data"; \
    fi && \
    cp /model-cache/visual.onnx /app/models/visual.onnx && \
    cp /model-cache/visual.onnx.data /app/models/visual.onnx.data

COPY src ./src
RUN if [ "$USE_CUDA" = "1" ]; then \
        g++ src/main.cpp -o embed \
            -DUSE_CUDA=1 \
            -I/onnxruntime/include \
            -L/onnxruntime/lib \
            -lonnxruntime \
            `pkg-config --cflags --libs opencv4` \
            -O2; \
    else \
        g++ src/main.cpp -o embed \
            -I/onnxruntime/include \
            -L/onnxruntime/lib \
            -lonnxruntime \
            `pkg-config --cflags --libs opencv4` \
            -O2; \
    fi

# Runtime
FROM ${RUNTIME_BASE}
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-core406t64 \
    libopencv-imgproc406t64 \
    libopencv-imgcodecs406t64 \
    libopencv-dnn406t64 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/models ./models
COPY --from=builder /app/embed .
COPY --from=builder /onnxruntime/lib /onnxruntime/lib

ENV LD_LIBRARY_PATH=/onnxruntime/lib
CMD ["./embed"]