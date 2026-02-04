#!/bin/sh
set -e

echo "Initializing submodules..."
git submodule update --init --recursive

echo "Creating 'videos' and 'output' directories..."
mkdir -p videos output

mkdir -p services/ingestion/external/whisper/models

MODEL_PATH="services/ingestion/external/whisper/models/ggml-small.en.bin"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading Whisper model (ggml-small.en.bin)..."
    curl -L -o "$MODEL_PATH" \
      https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin
else
    echo "Whisper model already present. Skipping download."
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "Warning: Docker is not installed. You'll need it to run the project."
    exit 0
fi

if command -v docker-compose >/dev/null 2>&1; then
    COMPOSE="docker-compose"
else
    COMPOSE="docker compose"
fi

printf "\nDo you want to build and start the containers now? (Y/n)"
read -r answer

case "$answer" in
    ""|y|Y)
        $COMPOSE up --build
        ;;
    *)
        echo "Skipping container startup."
        echo "You can run it later with: $COMPOSE up --build"
        ;;
esac