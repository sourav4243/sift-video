# Build
FROM rust:1-bullseye AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    protobuf-compiler \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
RUN mkdir src && \
    echo "fn main() {println!(\"if you see this, the build failed\");}" > src/main.rs && \
    cargo build --release && \
    rm -rf src
COPY src ./src
RUN touch src/main.rs && cargo build --release

# Runtime
FROM debian:bullseye-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl1.1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/query_engine .

EXPOSE 8080
CMD ["./query_engine"]