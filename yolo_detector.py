import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, weights_path, device):
        self.model = YOLO(weights_path).to(device)
    
    def detect(self, image_source, conf=0.35, max_det=2000):
        if isinstance(image_source, (str, Path)):
            im = cv2.imread(str(image_source), cv2.IMREAD_COLOR)
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            im = Image.fromarray(im)
            true_w, true_h = Image.open(image_source).size
        elif isinstance(image_source, np.ndarray):
            im = Image.fromarray(image_source)
            true_w, true_h = im.size
        elif isinstance(image_source, Image.Image):
            im = image_source
            true_w, true_h = im.size
        else:
            raise ValueError("image_source must be a path, numpy array, or PIL Image")
        
        w, h = im.size
        w = w + (32 - w%32)
        h = h + (32 - h%32)
        results = self.model.predict(im, conf=conf, max_det=max_det, imgsz=w, verbose=False)[0]
        boxes = results.boxes.xyxyn.cpu().numpy()
        boxes = boxes * true_w
        boxes = boxes.astype(int)
        
        if boxes.shape[0] > 0:
            boxes = np.hstack([np.ones((boxes.shape[0], 1)), boxes, np.ones((boxes.shape[0], 1))])
        
        return boxes
    
    def detect_batch(self, image_list, conf=0.35, max_det=2000):
        if len(image_list) == 0:
            return []
        
        pil_images = []
        true_sizes = []
        
        for img_np in image_list:
            im = Image.fromarray(img_np)
            pil_images.append(im)
            true_sizes.append(im.size)
        
        max_w = max(size[0] for size in true_sizes)
        max_h = max(size[1] for size in true_sizes)
        max_w = max_w + (32 - max_w % 32) if max_w % 32 != 0 else max_w
        max_h = max_h + (32 - max_h % 32) if max_h % 32 != 0 else max_h
        
        results = self.model.predict(
            pil_images, 
            conf=conf, 
            max_det=max_det, 
            imgsz=max(max_w, max_h),
            verbose=False
        )
        
        batch_boxes = []
        for idx, result in enumerate(results):
            boxes = result.boxes.xyxyn.cpu().numpy()
            true_w, true_h = true_sizes[idx]
            
            boxes = boxes * true_w
            boxes = boxes.astype(int)
            
            if boxes.shape[0] > 0:
                boxes = np.hstack([np.ones((boxes.shape[0], 1)), boxes, np.ones((boxes.shape[0], 1))])
            
            batch_boxes.append(boxes)
        
        return batch_boxes