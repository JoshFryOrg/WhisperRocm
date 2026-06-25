# Whisper ROCm

Docker image for running `whisper.cpp`'s OpenAI-compatible transcription server with AMD ROCm/HIP acceleration.

The image builds `whisper.cpp` from source inside AMD's `rocm/dev-ubuntu-24.04:latest` development image, enables `GGML_HIP`, compiles for explicit AMD GPU targets, downloads the `large-v3-q5_0` model, and exposes the server on port `8080`.

## Docker Compose

AMD ROCm example:

```yaml
services:
  whisper-rocm:
    image: 'joshfryup/whisper-rocm:latest'
    container_name: WhisperRocm
    restart: unless-stopped
    ports:
      - '8080:8080'
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    group_add:
      - video
    ipc: host
    security_opt:
      - seccomp=unconfined
```

## Docker Run

```bash
docker run --rm \
  --name WhisperRocm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --security-opt seccomp=unconfined \
  -p 8080:8080 \
  joshfryup/whisper-rocm:latest
```

## Build Locally

The default build targets common recent AMD GPU architectures: `gfx1100`, `gfx1101`, `gfx1102`, `gfx1200`, and `gfx1201`.

```bash
docker build -t whisper-rocm .
```

To build for a different GPU, pass `AMDGPU_TARGETS`. You can usually find your architecture with `rocminfo | grep gfx`.

```bash
docker build --build-arg AMDGPU_TARGETS=gfx1030 -t whisper-rocm .
```

## API

The server exposes the OpenAI-compatible transcription endpoint at:

```text
http://localhost:8080/v1/audio/transcriptions
```

Example request:

```bash
curl http://localhost:8080/v1/audio/transcriptions \
  -F file=@audio.mp3 \
  -F model=whisper-1
```
