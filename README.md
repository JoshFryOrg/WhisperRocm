# Whisper ROCm

Docker image for running `whisper.cpp`'s OpenAI-compatible transcription server with AMD ROCm/HIP acceleration.

The image builds `whisper.cpp` from source inside AMD's `rocm/dev-ubuntu-24.04:latest` development image, enables `GGML_HIP`, compiles for explicit AMD GPU targets, and exposes the server on port `8080`. Voice-activity detection is enabled by default so the model doesn't hallucinate filler over silence/music; its parameters mirror `faster-whisper`'s VAD defaults (min-silence 2000 ms, speech-pad 400 ms, threshold 0.5).

The models are **not** baked into the image. On first startup the supervisor downloads the full `large-v3` model plus a Silero VAD model into `/models` (using `whisper.cpp`'s own download scripts) if they aren't already there. Mount a volume at `/models` so the ~3 GB download happens only once and persists across rebuilds and restarts. See [Models volume](#models-volume).

## Models volume

The container expects its models in `/models` and downloads them there on startup if missing. Mount a host directory or named volume so they survive container recreation and can be shared between the ROCm and Vulkan images:

```bash
-v whisper-models:/models      # named volume
# or
-v /path/on/host/models:/models  # host directory
```

The download is to disk only — the model is loaded into VRAM lazily on the first transcription request (see below), so this doesn't change the on-demand behaviour. If the volume already contains `ggml-large-v3.bin` and `ggml-silero-v5.1.2.bin`, startup skips the download entirely. The model files default to `large-v3` and `silero-v5.1.2`; override with the `WHISPER_MODEL` / `VAD_MODEL` / `MODELS_DIR` environment variables (the entrypoint paths must match `MODELS_DIR`).

## On-demand model loading

`whisper.cpp`'s server normally loads the model into VRAM at startup and keeps it there for as long as the container runs, pinning several GB even when nothing is being transcribed. To free the GPU for other workloads in between, the image runs a small supervisor in front of the server: the model is loaded on the first transcription request and unloaded again after a configurable idle period, then reloaded transparently on the next request.

The trade-off is a cold-start delay (a few seconds to reload `large-v3` and the VAD model) on the first request after an idle gap; subsequent requests are served immediately while the model stays resident.

Tune it with environment variables (all optional):

| Variable | Default | Purpose |
| --- | --- | --- |
| `WHISPER_IDLE_TTL` | `300` | Seconds with no requests before the model is unloaded and its VRAM freed. |
| `WHISPER_START_TIMEOUT` | `180` | Seconds to wait for the model to load before failing a request. |
| `INTERNAL_PORT` | `8081` | Private port the underlying server binds; the public port stays `8080`. |

The `/health` endpoint is answered by the supervisor directly, so health probes neither load the model nor keep it warm.

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
    # Models download here on first start and persist across restarts/rebuilds.
    volumes:
      - whisper-models:/models
    # Optional: seconds idle before the model is unloaded to free VRAM (default 300).
    environment:
      - WHISPER_IDLE_TTL=300
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    group_add:
      - video
    ipc: host
    security_opt:
      - seccomp=unconfined

volumes:
  whisper-models:
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
  -v whisper-models:/models \
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

## Vulkan variant (alternative backend)

`Dockerfile.vulkan` builds the same server with the **Vulkan** backend (via Mesa RADV) instead of HIP/ROCm.

On AMD RDNA cards the ROCm/HIP runtime pins the GPU to 100% utilisation (and high clocks/power) for the whole life of the process, even when idle — a ROCm bug, not `whisper.cpp`. The Vulkan backend does not, so the GPU idles properly even while the model is resident, not just after the on-demand unload above. It also drops the ROCm stack entirely: it needs only `/dev/dri` at runtime, not `/dev/kfd`, and there is no per-architecture build target to set.

The trade-off is transcription speed: HIP/`rocBLAS` is usually AMD's fastest path, while Vulkan is often comparable but can be slower for `large-v3`. **Benchmark it against the HIP image on your own audio before adopting it as the default.**

```bash
docker build -f Dockerfile.vulkan -t whisper-vulkan .
```

```bash
docker run --rm \
  --name WhisperVulkan \
  --device=/dev/dri \
  --group-add video \
  -p 8080:8080 \
  -v whisper-models:/models \
  whisper-vulkan
```

## API

The server exposes the OpenAI-compatible transcription endpoint at:

```text
http://localhost:8080/v1/audio/transcriptions
```

`whisper.cpp`'s server serves transcription at `/inference` by default; the image starts it with
`--inference-path /v1/audio/transcriptions` so OpenAI-compatible clients reach it without any client-side
path configuration. A health check is available at `/health`.

Example request:

```bash
curl http://localhost:8080/v1/audio/transcriptions \
  -F file=@audio.mp3 \
  -F model=whisper-1 \
  -F response_format=verbose_json
```

Clients that take an OpenAI base URL should point at the `/v1` base; they append `/audio/transcriptions` themselves:

```text
http://<host>:8080/v1
```
