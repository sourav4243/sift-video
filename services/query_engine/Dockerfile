# Build
FROM rust:1-bookworm AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
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
FROM debian:bookworm-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libonnxruntime \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/query_engine .

# Model and tokenizer must be present at build time in services/query_engine/models/ 
COPY models/model.onnx ./models/model.onnx
COPY models/tokenizer.json ./models/tokenizer.json

EXPOSE 8080

CMD ["./query_engine"]