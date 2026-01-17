# Ingestion Service

This service is responsible for **offline preprocessing of raw video files**.

Its job is to:
- discover video files
- prepare them for downstream indexing
- act as the **entry point of the pipeline**

At this stage, the service only verifies video discovery inside a containerized environment.

---

## Responsibilities

- Scan the `/videos` directory for input files
- Run inside a Docker container
- Ensure correct volume mounting and filesystem access
- Log discovered video filenames

This service **does not perform search, embeddings, or ML inference**.
It only handles **raw video ingestion**.

---

## Directory Contract

The container expects the following volumes:

- `/videos` — input directory containing raw video files
- `/output` — directory reserved for generated artifacts (used in later stages)

These paths are **hard contracts** and must not be changed.

---

## How to Build

From repository root:

```bash
docker build -t ingestion services/ingestion
```

---

## How to Run
From repository root:
```bash
docker run --rm \
  -v "$(pwd)/videos:/videos" \
  -v "$(pwd)/output:/output" \
  ingestion

```

---

## Expected Output:
Example log:
```text
Ingestion service started
Found files: example.mp4
```
This confirms:
* Docker image built correctly
* Volumes mounted correctly
* Video files are visible inside the container