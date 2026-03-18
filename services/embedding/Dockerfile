FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    wget \
    libopencv-dev

RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/onnxruntime-linux-x64-1.17.3.tgz && \
    tar -xzf onnxruntime-linux-x64-1.17.3.tgz && \
    mv onnxruntime-linux-x64-1.17.3 /onnxruntime

COPY src ./src
COPY models ./models

RUN g++ src/main.cpp -o embed \
-I/onnxruntime/include \
-L/onnxruntime/lib \
-lonnxruntime \
`pkg-config --cflags --libs opencv4`

ENV LD_LIBRARY_PATH=/onnxruntime/lib

CMD ["./embed"]