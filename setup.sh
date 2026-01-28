#!/bin/sh
set -e

echo "Initializing submodules..."
git submodule update --init --recursive

echo "Creating 'videos' and 'output' directories..."
mkdir -p videos output

if ! command -v docker &> /dev/null; then
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
        echo "You can run it later with: docker-compose up --build"
        ;;
esac