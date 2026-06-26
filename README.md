# Whisper ROCm

Docker image for running `whisper.cpp`'s OpenAI-compatible transcription server with AMD ROCm/HIP acceleration.

The image builds `whisper.cpp` from source inside AMD's `rocm/dev-ubuntu-24.04:latest` development image, enables `GGML_HIP`, compiles for explicit AMD GPU targets, downloads the full `large-v3` model plus a Silero VAD model, and exposes the server on port `8080`. Voice-activity detection is enabled by default so the model doesn't hallucinate filler over silence/music; its parameters mirror `faster-whisper`'s VAD defaults (min-silence 2000 ms, speech-pad 400 ms, threshold 0.5).

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
