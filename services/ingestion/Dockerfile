FROM ubuntu:22.04

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update && \
    apt-get install -y --fix-missing \
        ffmpeg \
        g++ \
        cmake \
        make \
        curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY external/whisper ./external/whisper
RUN cd external/whisper && make

COPY src ./src
RUN g++ src/main.cpp -o ingest

CMD ["./ingest"]