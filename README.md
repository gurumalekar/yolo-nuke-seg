# Segmentation Inference Pipeline

A modular inference pipeline for medical image segmentation using YOLO detection and UNet++ segmentation.

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
    --seg_weights /path/to/segmentation/weights.pt \
    --device cuda:0 \
    --dims 64 \
    --depth 4 \
    --conf 0.35 \
    --max_det 9999
```

## Parameters

- `--input_dir`: Directory containing input images (supports .tif files)
- `--output_dir`: Directory to save predictions (default: predictions)
- `--yolo_weights`: Path to YOLO model weights
- `--seg_weights`: Path to segmentation model weights
- `--device`: Device to use (default: cuda:0)
- `--dims`: Patch dimensions for segmentation (default: 64)
- `--depth`: Encoder depth for UNet++ (default: 4)
- `--conf`: Confidence threshold for YOLO (default: 0.35)
- `--max_det`: Maximum detections per image (default: 9999)

## Output

The pipeline saves segmentation maps as numpy arrays (.npy files) in the specified output directory.