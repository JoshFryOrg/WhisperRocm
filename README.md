# Whisper ROCm

Docker image for running `whisper.cpp`'s OpenAI-compatible transcription server with AMD ROCm/HIP acceleration.

The image builds `whisper.cpp` from source inside AMD's `rocm/dev-ubuntu-24.04:latest` development image, enables `GGML_HIP`, compiles for explicit AMD GPU targets, and exposes the server on port `8080`. Voice-activity detection is enabled by default so the model doesn't hallucinate filler over silence/music; its parameters mirror `faster-whisper`'s VAD defaults (min-silence 2000 ms, speech-pad 400 ms, threshold 0.5). Decoding is further hardened against the large models' tendency to emit a stock end-card phrase ("subscribe to my channel", "Transcription by …") over quiet stretches: `--max-context 0` carries no previously-decoded text between the internal 30-second chunks (so a hallucination in one chunk can't chain into the next), `--suppress-nst` suppresses non-speech tokens, and `--beam-size 5` uses beam search instead of the more loop-prone greedy decode (slower, but accuracy is preferred over speed).

The models are **not** baked into the image. On first startup the supervisor downloads the full `large-v2` model plus a Silero VAD model into `/models` (using `whisper.cpp`'s own download scripts) if they aren't already there. Mount a volume at `/models` so the ~3 GB download happens only once and persists across rebuilds and restarts. See [Models volume](#models-volume).

## Models volume

The container expects its models in `/models` and downloads them there on startup if missing. Mount a host directory or named volume so they survive container recreation and can be shared between the ROCm and Vulkan images:

```bash
-v whisper-models:/models      # named volume
# or
-v /path/on/host/models:/models  # host directory
```

The download is to disk only — the model is loaded into VRAM lazily on the first transcription request (see below), so this doesn't change the on-demand behaviour. If the volume already contains `ggml-large-v2.bin` and `ggml-silero-v5.1.2.bin`, startup skips the download entirely. The model files default to `large-v2` and `silero-v5.1.2`; override with the `WHISPER_MODEL` / `VAD_MODEL` / `MODELS_DIR` environment variables. The supervisor derives both the download target and the server's load path (`-m` / `--vad-model`) from these, so the model name has a single source of truth and the two cannot drift.

## On-demand model loading

`whisper.cpp`'s server normally loads the model into VRAM at startup and keeps it there for as long as the container runs, pinning several GB even when nothing is being transcribed. To free the GPU for other workloads in between, the image runs a small supervisor in front of the server: the model is loaded on the first transcription request and unloaded again after a configurable idle period, then reloaded transparently on the next request.

The trade-off is a cold-start delay (a few seconds to reload `large-v2` and the VAD model) on the first request after an idle gap; subsequent requests are served immediately while the model stays resident.

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

The trade-off is transcription speed: HIP/`rocBLAS` is usually AMD's fastest path, while Vulkan is often comparable but can be slower for the large models. **Benchmark it against the HIP image on your own audio before adopting it as the default.**

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

### faster-whisper / OpenAI parameter compatibility

A few transcription parameters have different names (or senses) between `faster-whisper` (and OpenAI-style
servers built on it) and `whisper.cpp`. So a client written for those APIs works against this server unchanged,
the supervisor renames them on the way through before handing the request to `whisper.cpp`:

| Client sends | Forwarded to `whisper.cpp` as | Notes |
| --- | --- | --- |
| `vad_filter` | `vad` | Same boolean; enables voice-activity detection. (VAD is also on by default via the entrypoint.) |
| `condition_on_previous_text` | `no_context` (value inverted) | `no_context` is the negation; `condition_on_previous_text=false` becomes `no_context=true`. |

The rewrite is strictly defensive: it only touches those small text fields and forwards the request unchanged
on anything unexpected, so it never corrupts the audio part.

Parameters `whisper.cpp` has no equivalent for are passed through and harmlessly ignored, because its own
defaults already cover them:

- `compression_ratio_threshold` — `whisper.cpp` uses `entropy_thold` (default `2.4`) for the same job: it
  flags degenerate/over-repetitive output and triggers temperature fallback.
- `no_repeat_ngram_size` — no `whisper.cpp` equivalent (its anti-repetition relies on the above plus VAD).
- `timestamp_granularities[]` — `whisper.cpp` returns per-segment **and** per-word timestamps in
  `verbose_json` by default (`token_timestamps`), so word-level times are present without it.

The net effect is an anti-repetition posture equivalent to faster-whisper's recommended one, on by default:
no cross-segment context carry-over (`no_context`), temperature fallback on degenerate output
(`entropy_thold`), and VAD to drop non-speech.
