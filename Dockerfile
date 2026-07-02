# PyTorch + CUDA runtime base (torch & torchvision preinstalled, GPU-ready)
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    NODE_ENV=production \
    PORT=3000 \
    PYTHON_BIN=python \
    EXIFTOOL=exiftool \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1

# System deps:
#  - libimage-exiftool-perl : exiftool CLI (replaces the bundled Windows .exe)
#  - libgl1 / libglib2.0-0  : runtime libs for OpenCV
#  - nodejs                 : Node 20 (via NodeSource) to run the Express server
RUN apt-get update && apt-get install -y --no-install-recommends \
        libimage-exiftool-perl \
        libgl1 libglib2.0-0 \
        curl ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python deps (torch/torchvision already in the base image) ---
COPY python/requirements.txt ./python/requirements.txt
RUN pip install --no-cache-dir -r python/requirements.txt

# --- Node deps (production only) ---
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

# --- Application source (see .dockerignore for what is excluded) ---
COPY . .

EXPOSE 3000
CMD ["node", "index.js"]
