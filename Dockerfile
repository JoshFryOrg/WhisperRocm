# Use the official AMD ROCm development base image
FROM rocm/dev-ubuntu-24.04:latest

# Set non-interactive timezone to prevent tzdata from blocking the build
ENV DEBIAN_FRONTEND=noninteractive

# ROCm/HIP needs explicit GPU architecture targets when building in CI.
ARG AMDGPU_TARGETS=gfx1100;gfx1101;gfx1102;gfx1200;gfx1201

# Install build dependencies and ffmpeg (essential for audio processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    make \
    g++ \
    python3 \
    ca-certificates \
    ffmpeg \
    curl \
    hipblas-dev \
    rocblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone the whisper.cpp repository
RUN git clone --depth=1 https://github.com/ggerganov/whisper.cpp.git .

# Configure and build with AMD HIP (ROCm) support.
# The GGML_HIP=ON flag is what triggers the AMD GPU compilation.
RUN cmake -B build \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS="${AMDGPU_TARGETS}" \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DCMAKE_BUILD_TYPE=Release

# Build the binaries using all available CPU cores
RUN cmake --build build -j $(nproc)

# Models are NOT baked into the image. The supervisor downloads them on first
# startup (via whisper.cpp's own download scripts, still present in this clone)
# into MODELS_DIR if missing. Mount a volume at /models to persist them across
# rebuilds/restarts and share them between containers.
#   large-v3   (~3 GB) the most accurate model; the targeted AMD cards have the VRAM.
#   silero VAD lets VAD drop non-speech so the model doesn't hallucinate filler
#              (e.g. "Thank you.") over silence/music.
ENV MODELS_DIR=/models \
    WHISPER_MODEL=large-v3 \
    VAD_MODEL=silero-v5.1.2 \
    WHISPER_CPP_DIR=/app
VOLUME /models

# On-demand supervisor: fronts the server on the public port and only loads the
# model into VRAM while transcriptions are in flight, unloading it after an idle
# period so the GPU is free in between.
COPY supervisor.py /app/supervisor.py

# Expose the API port
EXPOSE 8080

# The supervisor binds the public 8080 and starts whisper-server on a private port
# on demand; pass the real server command (without --host/--port, which the
# supervisor appends) plus its flags.
# --inference-path serves the OpenAI route (default /inference); --vad + --vad-* enable VAD tuned to
# faster-whisper's default parameters (min-silence 2000, speech-pad 400, threshold 0.5).
ENTRYPOINT ["python3", "supervisor.py", "./build/bin/whisper-server", "-m", "/models/ggml-large-v3.bin", "--inference-path", "/v1/audio/transcriptions", "--vad", "--vad-model", "/models/ggml-silero-v5.1.2.bin", "--vad-threshold", "0.5", "--vad-min-speech-duration-ms", "0", "--vad-min-silence-duration-ms", "2000", "--vad-speech-pad-ms", "400"]
