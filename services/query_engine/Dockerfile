# Build
FROM rust:1-slim-bookworm AS rust-base

FROM debian:trixie-slim AS builder
WORKDIR /app

COPY --from=rust-base /usr/local/cargo /usr/local/cargo
COPY --from=rust-base /usr/local/rustup /usr/local/rustup

ENV PATH="/usr/local/cargo/bin:${PATH}"
ENV RUSTUP_HOME="/usr/local/rustup"
ENV CARGO_HOME="/usr/local/cargo"

RUN apt-get update && apt-get install -y \
    ca-certificates \
    gcc \
    g++ \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
RUN mkdir src && \
    echo "fn main() {}" > src/main.rs && \
    cargo build --release && \
    rm -rf src
COPY src ./src
RUN touch src/main.rs && cargo build --release

# Runtime
FROM debian:trixie-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/query_engine .
EXPOSE 8080
CMD ["./query_engine"]
