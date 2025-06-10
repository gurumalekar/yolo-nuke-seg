# WSI Nuclei Segmentation Pipeline

A high-performance inference pipeline for nuclei segmentation in whole slide images (WSI) utilizing YOLO-based detection followed by UNet++ segmentation. The models are specifically trained and optimized for WSI data captured at 40x magnification (~0.25 microns per pixel).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python infer.py \
    --input_dir /path/to/images \
    --output_dir predictions \
    --yolo_weights /path/to/yolo/weights.pt \
    --seg_weights /path/to/segmentation/dims64_depth4.pt \
    --device cuda:0 \
    --conf 0.35 \
    --max_det 9999
```

## Dataset Specifications

- **Magnification**: 40x objective magnification
- **Resolution**: ~0.25 microns per pixel (mpp)
- **Image Format**: TIFF files from whole slide imaging systems
- **Target**: Individual nuclei detection and instance segmentation

## Pipeline Overview

The inference pipeline employs a two-stage approach:
1. **Detection Stage**: YOLO model identifies potential nuclei regions
2. **Segmentation Stage**: UNet++ performs precise instance segmentation on detected patches

This architecture ensures both computational efficiency and high segmentation accuracy for large-scale WSI analysis.

## Parameters

- `--input_dir`: Directory containing WSI images (supports .tif format)
- `--output_dir`: Directory for prediction outputs (default: predictions)
- `--yolo_weights`: Path to trained YOLO detection model weights
- `--seg_weights`: Path to trained UNet++ segmentation model weights (e.g., dims64_depth4.pt)
- `--device`: Computation device (default: cuda:0, fallback: cpu)
- `--conf`: Detection confidence threshold (default: 0.35)
- `--max_det`: Maximum nuclei detections per image (default: 9999)

## Output

The pipeline generates instance segmentation maps as numpy arrays (.npy files), where each nucleus is assigned a unique integer label. Output files are saved in the specified directory with corresponding image filenames.

## Performance Notes

- Optimized for NVIDIA GPUs with CUDA support
- Memory usage scales with image size and detection density
- Processing time varies based on nuclei count per image