## Chess Coach — Adaptive AI with Real‑time Elo Estimation

This project is an adaptive chess training system. It uses a multi‑head CNN (policy/value/elo) plus a Kalman filter to estimate your playing strength from just a few moves, then adjusts the AI difficulty on the fly. The AI plays with human‑like mistakes, sets tactical traps, and provides a post‑game error book to help you learn.

## ⚠️ Important Note for Developers

Due to GitHub’s file size limits, **the following large binary files have been removed from this repository**:

- `engines/cublas64_10.dll` (CUDA runtime)
- `engines/cudnn64_7.dll` (cuDNN)
- `stockfish/stockfish-windows-x86-64-avx2.exe` (Stockfish engine)

If you clone this repository directly, **the project will NOT run out‑of‑the‑box**. You need to manually obtain and place the missing dependencies.

## 📦 Required Dependencies

### 1. Stockfish Chess Engine
- Download the appropriate version from the [official Stockfish website](https://stockfishchess.org/download/) (Windows/Linux/macOS).
- Place the executable inside the `stockfish/` folder.
- Make sure the file name matches what the code expects (e.g., `stockfish-windows-x86-64-avx2.exe`). You can also update the path in the configuration file.

### 2. CUDA & cuDNN (if you want GPU acceleration when continuously training the CNN models)
- Install [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (version 12.x or newer, depending on your setup&GPU).
- Download [cuDNN](https://developer.nvidia.com/cudnn) from NVIDIA (requires free registration).
- Copy `cublas64_10.dll` and `cudnn64_7.dll` into the `engines/` folder (or set the system PATH accordingly).

### 3. Python Environment

- Python 3.8+ is required.
- It is strongly recommended to use a virtual environment.

Install the required packages:

```bash
pip install python-chess torch numpy tqdm matplotlib chess
