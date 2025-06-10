import torch
import segmentation_models_pytorch as smp

def load_segmentation_model(weights_path, dims, depth, device):
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
    
    return model