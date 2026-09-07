import cv2
import json
import numpy as np
from pathlib import Path

def extract_cutouts(coco_json_path: str = "blueObjectDetection.coco/train/_annotations.coco.json", output_dir: str = "cutouts"):
    """
    Extracts transparent PNG cutouts of the blue object using color segmentation
    and morphological refinement, preserving internal apertures.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)
        
    img_map = {img['id']: img for img in coco['images']}
    saved_cutouts = []
    
    for i, ann in enumerate(coco['annotations']):
        img_info = img_map[ann['image_id']]
        img_file = Path(coco_json_path).parent / img_info['file_name']
        if not img_file.exists():
            continue
            
        img = cv2.imread(str(img_file))
        if img is None:
            continue
            
        x, y, w, h = [int(round(v)) for v in ann['bbox']]
        
        margin = 15
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(img.shape[1], x + w + margin)
        y1 = min(img.shape[0], y + h + margin)
        
        crop = img[y0:y1, x0:x1]
        
        # Segment blue object in HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([95, 85, 60]), np.array([130, 255, 255]))
        
        # Morphological smoothing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Keep largest connected component
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
            
        # Antialias alpha edges
        mask_feathered = cv2.GaussianBlur(mask, (3, 3), 0)
        
        # Create RGBA
        b, g, r = cv2.split(crop)
        rgba = cv2.merge([b, g, r, mask_feathered])
        
        # Crop tight to non-zero alpha
        coords = cv2.findNonZero(mask)
        if coords is not None:
            bx, by, bw, bh = cv2.boundingRect(coords)
            rgba_tight = rgba[by:by+bh, bx:bx+bw]
            cutout_filename = f"cutout_{i}.png"
            cutout_filepath = out_path / cutout_filename
            cv2.imwrite(str(cutout_filepath), rgba_tight)
            saved_cutouts.append(str(cutout_filepath))
            print(f"Extracted {cutout_filename} (size: {bw}x{bh}) from {img_info['file_name']}")
            
    return saved_cutouts

if __name__ == "__main__":
    extract_cutouts()
