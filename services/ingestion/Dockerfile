# Build
FROM ubuntu:22.04 AS builder

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

RUN cd external/whisper && \
    cmake -B build -DBUILD_SHARED_LIBS=OFF && \
    cmake --build build --config Release

COPY src ./src
RUN g++ src/main.cpp -o ingest -static-libstdc++ -static-libgcc

# Runtime
FROM ubuntu:22.04
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/external/whisper/models/ggml-small.en.bin ./external/whisper/models/ggml-small.en.bin
COPY --from=builder /app/ingest .
COPY --from=builder /app/external/whisper/build/bin/whisper-cli \
                    ./external/whisper/build/bin/whisper-cli

CMD ["./ingest"]