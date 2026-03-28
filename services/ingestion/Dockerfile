ARG RUNTIME_BASE=ubuntu:22.04
ARG BUILDER_BASE=ubuntu:22.04
# Build
ARG USE_CUDA=0
FROM ${BUILDER_BASE} AS builder

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

RUN mkdir -p external/whisper/models && \
    curl -L -f -o external/whisper/models/ggml-small.en.bin \
    https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin


ARG USE_CUDA=0
RUN cd external/whisper && \
    if [ "$USE_CUDA" = "1" ]; then \
        cmake -B build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON; \
    else \
        cmake -B build -DBUILD_SHARED_LIBS=OFF; \
    fi && \
    cmake --build build --config Release --parallel $(nproc)

COPY src ./src
RUN g++ src/main.cpp -o ingest -static-libstdc++ -static-libgcc

# Runtime
FROM ${RUNTIME_BASE}
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/external/whisper/models/ggml-small.en.bin ./external/whisper/models/ggml-small.en.bin
COPY --from=builder /app/ingest .
COPY --from=builder /app/external/whisper/build/bin/whisper-cli \
                    ./external/whisper/build/bin/whisper-cli

CMD ["./ingest"]