import os
import argparse
import json
import uuid
import gc
import numpy as np
import openslide
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
from shapely.geometry import Polygon as shape_Polygon, box
from shapely.ops import orient
from shapely.strtree import STRtree
from multiprocessing import Pool, cpu_count
from models import load_segmentation_model
from yolo_detector import YOLODetector
from image_processor import ImageProcessor

def _process_chunk_for_deduplication(args):
    chunk_start, chunk_end, polygons, tree, iou_threshold, distance_threshold = args
    duplicates = []
    
    for i in range(chunk_start, chunk_end):
        if i in duplicates:
            continue
        
        poly_i = polygons[i]
        
        search_area = poly_i.buffer(distance_threshold)
        candidates = tree.query(search_area)
        
        for poly_j in candidates:
            try:
                j = polygons.index(poly_j)
            except ValueError:
                continue
            
            if j <= i or j in duplicates:
                continue
            
            try:
                intersection = poly_i.intersection(poly_j).area
                union = poly_i.union(poly_j).area
                
                if union > 0:
                    iou = intersection / union
                    
                    if iou > iou_threshold:
                        if poly_i.area >= poly_j.area:
                            duplicates.append(j)
                        else:
                            duplicates.append(i)
                            break
            except Exception:
                continue
    
    return duplicates

class WSIInference:
    def __init__(self, wsi_path, yolo_weights, seg_weights, device='cuda:0', conf=0.35, max_det=9999):
        self.wsi_path = Path(wsi_path)
        self.slide = openslide.OpenSlide(str(wsi_path))
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading segmentation model from {seg_weights}...")
        dims_data = load_segmentation_model(seg_weights, self.device)
        self.seg_model = dims_data['model']
        self.patch_size = dims_data['dims']
        self.depth = dims_data['depth']
        
        print(f"Loading YOLO model from {yolo_weights}...")
        self.yolo_detector = YOLODetector(yolo_weights, self.device)
        self.processor = ImageProcessor(self.seg_model, self.yolo_detector, self.patch_size, self.device)
        
        self.conf = conf
        self.max_det = max_det
        
        self.target_mpp = 0.25
        self.mpp = self.get_mpp()
        
        if self.mpp:
            raw_scale = self.mpp / self.target_mpp
            allowed_scales = [0.5, 1.0, 2.0, 4.0]
            self.scale_factor = min(allowed_scales, key=lambda x: abs(x - raw_scale))
            print(f"Raw scale: {raw_scale:.2f}, Snapped scale: {self.scale_factor}")
        else:
            self.scale_factor = 1.0
        
        print(f"Slide MPP: {self.mpp}, Target MPP: {self.target_mpp}, Scale Factor: {self.scale_factor}")

    def get_mpp(self):
        try:
            mpp_x = float(self.slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0))
            if mpp_x > 0:
                return mpp_x
            return 0.25 
        except:
            return 0.25

    def get_tiles(self, tile_size=2048, overlap=0):
        w, h = self.slide.dimensions
        step = tile_size - overlap
        
        target_w = int(w * self.scale_factor)
        target_h = int(h * self.scale_factor)
        
        for y in range(0, target_h, step):
            for x in range(0, target_w, step):
                src_x = int(x / self.scale_factor)
                src_y = int(y / self.scale_factor)
                req_w_target = min(tile_size, target_w - x)
                req_h_target = min(tile_size, target_h - y)
                req_w_src = int(req_w_target / self.scale_factor)
                req_h_src = int(req_h_target / self.scale_factor)
                
                if req_w_src <= 0 or req_h_src <= 0:
                    continue
                    
                yield (x, y), (src_x, src_y, req_w_src, req_h_src), (req_w_target, req_h_target)

    def process_wsi(self, output_path, output_type='polygon', batch_size=16):
        proc_size = 2048 
        overlap = 256 
        tile_batch = []
        coords_batch = []
        all_features = []
        
        w, h = self.slide.dimensions
        target_w = int(w * self.scale_factor)
        target_h = int(h * self.scale_factor)
        step = proc_size - overlap
        nx = (target_w + step - 1) // step
        ny = (target_h + step - 1) // step
        total_tiles = nx * ny
        tile_gen = self.get_tiles(proc_size, overlap)
        
        print(f"Processing tiles with {overlap}px overlap for deduplication...")
        pbar = tqdm(total=total_tiles, desc="Processing Tiles", ncols=80)
        batch_count = 0
        
        for (tx, ty), (sx, sy, sw, sh), (tw, th) in tile_gen:
            try:
                region = self.slide.read_region((sx, sy), 0, (sw, sh)).convert('RGB')
                region_np = np.array(region)
                
                if self.scale_factor != 1.0:
                     region_np = cv2.resize(region_np, (tw, th))
                tile_batch.append(region_np)
                coords_batch.append(((tx, ty), (tw, th)))
                
                if len(tile_batch) >= batch_size:
                    new_features = self.process_batch(tile_batch, coords_batch, output_type, (target_w, target_h))
                    all_features.extend(new_features)
                    pbar.update(len(tile_batch))
                    tile_batch = []
                    coords_batch = []
                    batch_count += 1
                    
                    if batch_count % 50 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error reading tile at {tx},{ty}: {e}")
                continue

        if tile_batch:
            new_features = self.process_batch(tile_batch, coords_batch, output_type, (target_w, target_h))
            all_features.extend(new_features)
            pbar.update(len(tile_batch))
        
        pbar.close()
        
        print(f"Total features before deduplication: {len(all_features)}")
        deduplicated_features = self.deduplicate_polygons(all_features)
        print(f"Total features after deduplication: {len(deduplicated_features)}")
        
        print(f"Writing output to {output_path}...")
        with open(output_path, 'w') as f:
            f.write('{"type":"FeatureCollection","features":[')
            for idx, feat in enumerate(deduplicated_features):
                if idx > 0:
                    f.write(',')
                json.dump(feat, f)
            f.write(']}')
        
        return None


    def process_batch(self, tiles, coords, output_type, global_size):
        target_w, target_h = global_size
        batch_features = []
        
        try:
            results = self.processor.process_image_batch(tiles, self.conf, self.max_det, verbose=False)
            
            for i, result in enumerate(results):
                (tx, ty), (tw, th) = coords[i]
                seg_map = result.segmentation_map
                features = self.contours_from_mask(seg_map, offset=(tx, ty), tile_size=(tw, th), global_size=(target_w, target_h), output_type=output_type)
                batch_features.extend(features)
        except Exception as e:
            print(f"Error processing batch: {e}")
            
        return batch_features

    def contours_from_mask(self, mask, offset, tile_size, global_size, output_type='polygon'):
        features = []
        ids = np.unique(mask)
        off_x, off_y = offset
        tw, th = tile_size
        target_w, target_h = global_size
        
        for i in ids:
            if i == 0: continue 
            
            binary_mask = (mask == i).astype(np.uint8)
            
            if output_type == 'both' or output_type == 'box':
                y_indices, x_indices = np.where(binary_mask)
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                gx_min, gx_max = x_min + off_x, x_max + off_x
                gy_min, gy_max = y_min + off_y, y_max + off_y
                gx_min, gx_max = gx_min / self.scale_factor, gx_max / self.scale_factor
                gy_min, gy_max = gy_min / self.scale_factor, gy_max / self.scale_factor
                
                box_coords = [[
                    [int(gx_min), int(gy_min)],
                    [int(gx_max), int(gy_min)],
                    [int(gx_max), int(gy_max)],
                    [int(gx_min), int(gy_max)],
                    [int(gx_min), int(gy_min)]
                ]]
                
                features.append({
                    "type": "Feature",
                    "id": str(uuid.uuid4()),
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": box_coords
                    },
                    "properties": {
                        "objectType": "annotation",
                        "classification": {"name": "Nucleus", "color": [0, 255, 0]}
                    }
                })

            if output_type == 'both' or output_type == 'polygon':
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if len(contour) < 3: continue
                    epsilon = 0.001 * cv2.arcLength(contour, True)
                    contour = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(contour) < 3: continue
                    contour_global = contour.copy().astype(float)
                    contour_global[:, 0, 0] += off_x
                    contour_global[:, 0, 1] += off_y
                    contour_global = contour_global / self.scale_factor
                    coords = contour_global.reshape(-1, 2).tolist()
                    try:
                        poly = shape_Polygon(coords)
                        
                        if not poly.is_valid:
                            poly = poly.buffer(0)
                        
                        if not poly.is_valid or poly.is_empty:
                            continue
                            
                        poly = orient(poly, sign=1.0)
                        if poly.exterior:
                            cleaned_coords = [[round(x, 1), round(y, 1)] for x, y in poly.exterior.coords]
                            if len(cleaned_coords) < 3: continue
                            features.append({
                                "type": "Feature",
                                "id": str(uuid.uuid4()),
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": [cleaned_coords]
                                },
                                "properties": {
                                    "objectType": "annotation",
                                    "classification": {"name": "Nucleus", "color": [0, 255, 0]}
                                }
                            })
                    except Exception:
                        continue


        return features
    
    def deduplicate_polygons(self, features, iou_threshold=0.5, distance_threshold=20):
        if len(features) == 0:
            return features
        print(f"Deduplicating {len(features)} features using spatial indexing...")
        polygons = []
        for feat in features:
            coords = feat['geometry']['coordinates'][0]
            poly = shape_Polygon(coords)
            polygons.append(poly)
        
        print("Building spatial index...")
        tree = STRtree(polygons)
        
        to_keep = [True] * len(features)
        
        num_processes = max(1, cpu_count() // 2)
        print(f"Using {num_processes} processes for deduplication...")
        
        chunk_size = max(100, len(features) // (num_processes * 4))
        chunks = []
        for i in range(0, len(features), chunk_size):
            chunks.append((i, min(i + chunk_size, len(features))))
        
        with Pool(num_processes) as pool:
            args = [(chunk_start, chunk_end, polygons, tree, iou_threshold, distance_threshold) 
                    for chunk_start, chunk_end in chunks]
            
            results = list(tqdm(
                pool.imap(_process_chunk_for_deduplication, args),
                total=len(chunks),
                desc="Deduplication chunks"
            ))
        
        for chunk_duplicates in results:
            for idx in chunk_duplicates:
                to_keep[idx] = False
        
        deduplicated = [feat for idx, feat in enumerate(features) if to_keep[idx]]
        
        removed_count = len(features) - len(deduplicated)
        if removed_count > 0:
            print(f"Removed {removed_count} duplicate polygons ({removed_count/len(features)*100:.1f}%)")
        
        return deduplicated
    
    def create_geojson(self, features):
        return features


def main():
    parser = argparse.ArgumentParser(description="WSI Nuclei Segmentation")
    parser.add_argument('--wsi_path', type=str, required=True, help="Path to WSI file")
    parser.add_argument('--output_path', type=str, default="output.geojson")
    parser.add_argument('--yolo_weights', type=str, required=True)
    parser.add_argument('--seg_weights', type=str, required=True)
    parser.add_argument('--output_type', type=str, choices=['polygon', 'box', 'both'], default='polygon')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for processing tiles (default: 4)')
    
    args = parser.parse_args()
    inferencer = WSIInference(
        args.wsi_path, 
        args.yolo_weights, 
        args.seg_weights, 
        args.device
    )
    
    print("Starting processing...")
    inferencer.process_wsi(args.output_path, args.output_type, args.batch_size)
    
    print("Done!")

if __name__ == "__main__":
    main()
