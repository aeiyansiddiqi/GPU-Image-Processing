# GPU Image Processing with NVIDIA Warp

GPU-accelerated image sharpening and noise removal using the NVIDIA Warp API. Supports both greyscale and RGB images with reflective border handling.

## Features
- **Noise Removal** — Gaussian blur filter
- **Sharpening** — Unsharp masking using Gaussian blur
- **Border Handling** — Reflective padding for full-image processing
- **GPU Acceleration** — Parallelized kernels via NVIDIA Warp

## Requirements
```
pip install warp-lang Pillow numpy
```

## Usage
```bash
python3 a3.py <algType> <kernSize> <param> <inFileName> <outFileName>
```

### Arguments
| Argument | Description |
|---|---|
| `algType` | `-s` for sharpening, `-n` for noise removal |
| `kernSize` | Kernel size (must be positive and odd, e.g. 3, 5, 7) |
| `param` | Sigma for Gaussian / scaling factor k for sharpening |
| `inFileName` | Input image path |
| `outFileName` | Output image path |

## Examples

**Noise Removal** (Gaussian blur, 5x5 kernel, sigma=1.5):
```bash
python3 a3.py -n 5 1.5 input.jpg output.jpg
```

**Sharpening** (Unsharp masking, 3x3 kernel, k=1.5):
```bash
python3 a3.py -s 3 1.5 input.jpg output.jpg
```

## How It Works

### Noise Removal
Applies a Gaussian blur kernel where each pixel is weighted by:
```
w(i,j) = e^(-(i² + j²) / 2σ²)
```

### Sharpening
Uses unsharp masking on top of the Gaussian blur:
```
output = input + k * (input - blurred)
```

### Border Handling
Uses reflective padding to compute values for border pixels, ensuring the full image is processed without edge artifacts.

## Docker
```bash
docker build -t gpu-image-processing .
docker run gpu-image-processing python3 a3.py -s 3 1.5 input.jpg output.jpg
```
