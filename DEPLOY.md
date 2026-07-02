# Acacia Inference & Geolocation — Deployment

Dockerized deployment of the Node/Express web app + Python (PyTorch/CUDA) inference backend.

## What's in the image
- **Node 20** runs the Express server (`index.js`) on port `3000`.
- **PyTorch + CUDA 12.1** base image — GPU-accelerated inference.
- **exiftool** (installed via apt) for drone-image metadata.
- Python deps from `python/requirements.txt` (torch/torchvision come from the base image).

## Prerequisites (host)
- Docker + Docker Compose
- NVIDIA GPU with recent drivers
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  (so the container can access the GPU)

Verify GPU access works:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## Required files NOT in git
These are excluded from version control and must be present in the build context:
- **`python/model/checkpoint_epoch_20_nogamma.pth`** — the trained model weights (~351 MB). Copy it in before building.

## Build & run
```bash
docker compose up --build -d
```
The app is then available at **http://localhost:3000**.

View logs (Python inference output streams here):
```bash
docker compose logs -f
```

Stop:
```bash
docker compose down
```

## Configuration (environment variables)
| Var          | Default     | Purpose                                            |
|--------------|-------------|----------------------------------------------------|
| `PORT`       | `3000`      | Server port                                        |
| `PYTHON_BIN` | `python`    | Python interpreter used to spawn inference         |
| `EXIFTOOL`   | `exiftool`  | exiftool binary (apt-installed in the container)   |
| `MPLBACKEND` | `Agg`       | Headless matplotlib backend (no display needed)    |

## Data persistence
`./uploads` on the host is mounted into the container at `/app/uploads`, so uploaded
batches, visuals, KML, CSV, and `status.json` survive container restarts/rebuilds.

## Notes
- The model checkpoint is baked into the image at build time (`COPY . .`). For a 351 MB
  file this makes the image large; alternatively mount it as a volume and keep it out of
  the image.
- CPU-only fallback: the code uses CUDA if available and falls back to CPU automatically.
  To build without GPU, use a non-CUDA base image and install the CPU torch wheels.
- The bundled Windows `exiftool-13.59_64/exiftool.exe` is ignored in the container (the
  Linux `exiftool` package is used via the `EXIFTOOL` env var); it remains for local
  Windows development.
