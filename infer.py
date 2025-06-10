import argparse
from pathlib import Path
import torch
from models import load_segmentation_model
from yolo_detector import YOLODetector
from image_processor import ImageProcessor
from utils import create_output_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='predictions')
    parser.add_argument('--yolo_weights', type=str, required=True)
    parser.add_argument('--seg_weights', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--conf', type=float, default=0.35)
    parser.add_argument('--max_det', type=int, default=9999)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    dims = load_segmentation_model(args.seg_weights, device)
    seg_model = dims['model']
    dims_value = dims['dims']
    depth_value = dims['depth']
    yolo_detector = YOLODetector(args.yolo_weights, device)
    processor = ImageProcessor(seg_model, yolo_detector, dims_value, device)
    
    output_dir = create_output_dir(args.output_dir, dims_value, depth_value)
    
    images = list(Path(args.input_dir).rglob("*.tif"))
    
    for image_path in images:
        result = processor.process_image(image_path, args.conf, args.max_det)
        output_path = output_dir / f"{image_path.stem}.npy"
        result.save(output_path)

if __name__ == "__main__":
    main()