FROM ubuntu:22.04

WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    wget \
    libopencv-dev \
    libonnxruntime-dev

COPY src ./src
COPY models ./models

RUN g++ src/main.cpp -o embed \
-I/usr/include/onnxruntime \
`pkg-config --cflags --libs opencv4` \
-lonnxruntime

CMD ["./embed"]