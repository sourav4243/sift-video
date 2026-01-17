FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    ffmpeg \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src ./src
RUN g++ src/main.cpp -o ingest

CMD ["./ingest"]