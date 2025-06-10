import torch
import segmentation_models_pytorch as smp
from pathlib import Path
import re

def load_segmentation_model(weights_path, device):
    filename = Path(weights_path).stem
    
    dims_match = re.search(r'dims(\d+)', filename)
    depth_match = re.search(r'depth(\d+)', filename)
    
    if not dims_match or not depth_match:
        raise ValueError(f"Cannot parse dims and depth from filename: {filename}")
    
    dims = int(dims_match.group(1))
    depth = int(depth_match.group(1))
    
    decoder_channels = [128, 64, 32, 16][::-1][:depth][::-1]
    
    model = smp.UnetPlusPlus(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation="sigmoid",
        encoder_depth=depth,
        decoder_channels=decoder_channels,
    ).to(device)
    
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    return {'model': model, 'dims': dims, 'depth': depth}