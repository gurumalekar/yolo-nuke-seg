import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
import tqdm

class ImageProcessor:
    def __init__(self, seg_model, yolo_detector, dims, device):
        self.seg_model = seg_model
        self.yolo_detector = yolo_detector
        self.dims = dims
        self.device = device
        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def process_image(self, image_source, conf, max_det, verbose=True):
        boxes = self.yolo_detector.detect(image_source, conf, max_det)
        if isinstance(image_source, (str, Path)):
             im = cv2.imread(str(image_source))
        elif isinstance(image_source, np.ndarray):
             im = image_source
        else:
             raise ValueError("Unsupported image source type")

        boxes = np.array(boxes)
        boxes[boxes < 0] = 0
        boxes = boxes.astype(int)
        
        h, w = im.shape[:2]
        segmentation_map = np.zeros((h, w), dtype=np.int32)
        
        counter = 1
        iterator = boxes[:, 1:-1]
        if verbose:
            iterator = tqdm.tqdm(iterator, ncols=50)
            
        for box in iterator:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = x1 - 4, y1 - 4, x2 + 4, y2 + 4
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, w - 1), min(y2, h - 1)
            
            patch = cv2.resize(im[y1:y2, x1:x2], (self.dims, self.dims))
            patch_tensor = self.patch_transform(Image.fromarray(patch)).to(self.device)
            patch_tensor = torch.unsqueeze(patch_tensor, 0)
            
            with torch.no_grad():
                pred_mask = self.seg_model(patch_tensor).squeeze().cpu().numpy()
            
            pred_mask = cv2.resize(pred_mask, (x2 - x1, y2 - y1))
            pred_mask = cv2.dilate(pred_mask, (3, 3))
            
            segmentation_map[y1:y2, x1:x2][pred_mask > 0.5] = counter
            counter += 1
        
        return SegmentationResult(segmentation_map)
    
    def process_image_batch(self, image_list, conf, max_det, verbose=False):
        batch_boxes = self.yolo_detector.detect_batch(image_list, conf, max_det)
        
        results = []
        for idx, (im, boxes) in enumerate(zip(image_list, batch_boxes)):
            boxes = np.array(boxes)
            boxes[boxes < 0] = 0
            boxes = boxes.astype(int)
            
            h, w = im.shape[:2]
            segmentation_map = np.zeros((h, w), dtype=np.int32)
            
            counter = 1
            iterator = boxes[:, 1:-1]
            
            for box in iterator:
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = x1 - 4, y1 - 4, x2 + 4, y2 + 4
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, w - 1), min(y2, h - 1)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                patch = cv2.resize(im[y1:y2, x1:x2], (self.dims, self.dims))
                patch_tensor = self.patch_transform(Image.fromarray(patch)).to(self.device)
                patch_tensor = torch.unsqueeze(patch_tensor, 0)
                
                with torch.no_grad():
                    pred_mask = self.seg_model(patch_tensor).squeeze().cpu().numpy()
                
                pred_mask = cv2.resize(pred_mask, (x2 - x1, y2 - y1))
                pred_mask = cv2.dilate(pred_mask, (3, 3))
                
                segmentation_map[y1:y2, x1:x2][pred_mask > 0.5] = counter
                counter += 1
            
            results.append(SegmentationResult(segmentation_map))
        
        return results

class SegmentationResult:
    def __init__(self, segmentation_map):
        self.segmentation_map = segmentation_map
    
    def save(self, path):
        np.save(path, self.segmentation_map)