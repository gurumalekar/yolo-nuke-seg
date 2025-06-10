import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, weights_path, device):
        self.model = YOLO(weights_path).to(device)
    
    def detect(self, image_path, conf=0.35, max_det=2000):
        im = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        im = Image.fromarray(im)
        
        true_w, true_h = Image.open(image_path).size
        w, h = im.size
        
        results = self.model.predict(im, conf=conf, max_det=max_det, imgsz=w)[0]
        boxes = results.boxes.xyxyn.cpu().numpy()
        boxes = boxes * true_w
        boxes = boxes.astype(int)
        
        if boxes.shape[0] > 0:
            boxes = np.hstack([np.ones((boxes.shape[0], 1)), boxes, np.ones((boxes.shape[0], 1))])
        
        return boxes