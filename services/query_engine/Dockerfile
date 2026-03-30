FROM rust:1-slim-bookworm AS rust-base

FROM debian:trixie-slim AS builder
WORKDIR /app

COPY --from=rust-base /usr/local/cargo /usr/local/cargo
COPY --from=rust-base /usr/local/rustup /usr/local/rustup

ENV PATH="/usr/local/cargo/bin:${PATH}"
ENV RUSTUP_HOME="/usr/local/rustup"
ENV CARGO_HOME="/usr/local/cargo"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    gcc \
    g++ \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -L -f https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp \
    -o /app/yt-dlp && chmod +x /app/yt-dlp

COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/target \
    cargo build --release && \
    rm -rf src

COPY src ./src
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/target \
    touch src/main.rs && \
    cargo build --release && \
    cp /app/target/release/query_engine /app/query_engine

# Runtime
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3t64 \
    libstdc++6 \
    python3 \
    ffmpeg \
    nodejs \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/yt-dlp /usr/local/bin/yt-dlp
COPY --from=builder /app/query_engine .

EXPOSE 8080
CMD ["./query_engine"]