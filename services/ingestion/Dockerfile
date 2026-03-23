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
RUN cd external/whisper && make
COPY src ./src
RUN g++ src/main.cpp -o ingest

# Runtime
FROM ubuntu:22.04
WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libstdc++6 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p external/whisper/models && \
    curl -L -o external/whisper/models/ggml-small.en.bin \
    https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin

COPY --from=builder /app/ingest .
COPY --from=builder /app/external/whisper/build/bin/whisper-cli \
                    ./external/whisper/build/bin/whisper-cli

CMD ["./ingest"]