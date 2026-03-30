ARG RUNTIME_BASE=ubuntu:24.04
ARG BUILDER_BASE=ubuntu:24.04

FROM ${BUILDER_BASE} AS builder

ARG USE_CUDA=0

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update && \
    apt-get install -y --fix-missing \
        g++ \
        cmake \
        make \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY external/whisper ./external/whisper

RUN --mount=type=cache,target=/model-cache \
    mkdir -p external/whisper/models && \
    if [ ! -f /model-cache/ggml-small.en.bin ]; then \
        curl -L -f -o /model-cache/ggml-small.en.bin \
        https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin; \
    fi && \
    cp /model-cache/ggml-small.en.bin external/whisper/models/ggml-small.en.bin

RUN cd external/whisper && \
    if [ "$USE_CUDA" = "1" ]; then \
        cmake -B build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON; \
    else \
        cmake -B build -DBUILD_SHARED_LIBS=OFF; \
    fi && \
    cmake --build build --config Release --parallel 2

COPY src ./src
RUN if [ "$USE_CUDA" = "1" ]; then \
        g++ src/main.cpp -o ingest -DUSE_CUDA=1 -static-libstdc++ -static-libgcc; \
    else \
        g++ src/main.cpp -o ingest -static-libstdc++ -static-libgcc; \
    fi

# Runtime
FROM ${RUNTIME_BASE}
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/external/whisper/models/ggml-small.en.bin \
                    ./external/whisper/models/ggml-small.en.bin
COPY --from=builder /app/ingest .
COPY --from=builder /app/external/whisper/build/bin/whisper-cli \
                    ./external/whisper/build/bin/whisper-cli

CMD ["./ingest"]