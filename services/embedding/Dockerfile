# Build
FROM ubuntu:22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    wget \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz && \
    tar -xzf onnxruntime-linux-x64-1.20.1.tgz && \
    mv onnxruntime-linux-x64-1.20.1 /onnxruntime && \
    rm onnxruntime-linux-x64-1.20.1.tgz

COPY src ./src
RUN g++ src/main.cpp -o embed \
    -I/onnxruntime/include \
    -L/onnxruntime/lib \
    -lonnxruntime \
    `pkg-config --cflags --libs opencv4`

# Runtime
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libopencv-core4.5d \
    libopencv-imgproc4.5d \
    libopencv-imgcodecs4.5d \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/embed .
COPY --from=builder /onnxruntime/lib /onnxruntime/lib

ENV LD_LIBRARY_PATH=/onnxruntime/lib
CMD ["./embed"]
