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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone the whisper.cpp repository
RUN git clone --depth=1 https://github.com/ggerganov/whisper.cpp.git .

# Configure and build with AMD HIP (ROCm) support.
# The GGML_HIP=ON flag is what triggers the AMD GPU compilation.
RUN cmake -B build \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS="${AMDGPU_TARGETS}" \
    -DCMAKE_BUILD_TYPE=Release

# Build the binaries using all available CPU cores
RUN cmake --build build -j $(nproc)

# Download a model.
# We use large-v3-q5_0 here: it offers the accuracy of the large-v3 model
# but is quantized to fit comfortably inside most AMD GPUs' VRAM (~500MB + model size).
RUN bash ./models/download-ggml-model.sh large-v3-q5_0

# Expose the API port
EXPOSE 8080

# Start the whisper-server.
# This exposes the standard OpenAI compatible API: /v1/audio/transcriptions
ENTRYPOINT ["./build/bin/whisper-server", "-m", "models/ggml-large-v3-q5_0.bin", "--host", "0.0.0.0", "--port", "8080"]
