# WSI Nuclei Segmentation Pipeline

A high-performance inference pipeline for nuclei segmentation in whole slide images (WSI) utilizing YOLO-based detection followed by UNet++ segmentation. The models are specifically trained and optimized for WSI data captured at 40x magnification (~0.25 microns per pixel).

## Installation

```bash
pip install -r requirements.txt
```

## Model Checkpoints

### Download Pre-trained Models

**Google Drive**: [Download All Models]([https://drive.google.com/file/d/1tlwUdb-3BIS0P1nM6wRHnYVPt01oV4p-/view?usp=sharing])
The download contains two folders:
- `yolo_models/` - YOLO detection model weights
- `seg_models/` - UNet++ segmentation model weights

#### Available Segmentation Models
After downloading and extracting, you'll find these segmentation model configurations:
- `dims32_depth3.pt` - 32x32 patches, depth 3
- `dims32_depth4.pt` - 32x32 patches, depth 4
- `dims48_depth3.pt` - 48x48 patches, depth 3
- `dims48_depth4.pt` - 48x48 patches, depth 4
- `dims64_depth3.pt` - 64x64 patches, depth 3 *(recommended)*
- `dims64_depth4.pt` - 64x64 patches, depth 4 
- `dims96_depth3.pt` - 96x96 patches, depth 3
- `dims96_depth4.pt` - 96x96 patches, depth 4
- `dims128_depth3.pt` - 128x128 patches, depth 3
- `dims128_depth4.pt` - 128x128 patches, depth 4

#### Model Selection Guidelines
- **dims64_depth3.pt**: Best balance of accuracy and speed (recommended for most use cases)
- **dims32_depth3.pt**: Fastest inference, lower memory usage



## Quick Start

1. **Download model checkpoints** from Google Drive (see Model Checkpoints section above)
2. **Extract and organize your data structure**:

3. **Run inference**:
   ```bash
   python infer.py \
       --input_dir ./input_images \
       --output_dir ./predictions \
       --yolo_weights ./yolo_models/last.pt \
       --seg_weights ./seg_models/dims64_depth4.pt
   ```

## Usage

```bash
python infer.py \
    --input_dir /path/to/images \
    --output_dir predictions \
    --yolo_weights ./yolo_models/last.pt \
    --seg_weights ./seg_models/dims64_depth4.pt \
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
![image](infer_diagram.png) 

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
- **Benchmark**: <1 minute per 2048x2048 WSI patch on RTX 4070

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@misc{wsi_nuclei_segmentation,
  title={WSI Nuclei Segmentation Pipeline},
  author={}, // Add your name/organization
  year={2025},
  url={} // Add your repository URL
}
```
