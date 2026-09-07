#!/usr/bin/env python3
"""
generate_dataset.py
Automated Synthetic Dataset Generator & Annotator for YOLOv11n Object Detection.

Pasts transparent object cutouts onto diverse natural backgrounds (COCO + procedural scenes)
with realistic physical augmentations (rotation, perspective, scale, shadows, lighting,
motion blur, noise, and hard blue distractors).

Outputs standard YOLO bounding box format dataset:
  dataset/
    images/train/, images/val/
    labels/train/, labels/val/
    data.yaml
    previews/
"""

import os
import sys
import glob
import random
import argparse
import urllib.request
from pathlib import Path
import numpy as np
import cv2

# Known diverse image IDs from COCO val2017 (indoor, outdoor, living rooms, kitchens, workshops, streets)
COCO_VAL_IDS = [
    139, 285, 632, 724, 776, 785, 802, 872, 885, 1000,
    1268, 1296, 1353, 1422, 1442, 1490, 1503, 1532, 1584, 1678,
    1760, 1926, 1993, 2006, 2149, 2153, 2157, 2299, 2378, 2404,
    2426, 2465, 2529, 2587, 2603, 2677, 2769, 2781, 2840, 2848,
    2987, 3037, 3088, 3144, 3154, 3217, 3256, 3267, 3350, 3374,
    3540, 3550, 3577, 3584, 3605, 3632, 3647, 3757, 3778, 3816,
    3879, 3931, 3962, 3971, 4014, 4038, 4125, 4141, 4287, 4310
]

def download_coco_backgrounds(output_dir="backgrounds", count=50):
    """Download diverse real-world background images from COCO val2017."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    existing = list(out_path.glob("coco_*.jpg"))
    if len(existing) >= count:
        return [str(p) for p in existing]
        
    print(f"Downloading {count} background images from COCO val2017...")
    downloaded = []
    for img_id in COCO_VAL_IDS[:count]:
        filename = f"coco_{img_id:012d}.jpg"
        target_path = out_path / filename
        if not target_path.exists():
            url = f"http://images.cocodataset.org/val2017/{img_id:012d}.jpg"
            try:
                urllib.request.urlretrieve(url, str(target_path))
                downloaded.append(str(target_path))
            except Exception as e:
                print(f"Skipping {url}: {e}")
        else:
            downloaded.append(str(target_path))
            
    print(f"Backgrounds available: {len(downloaded)}")
    return downloaded

def generate_procedural_background(size=(640, 640)):
    """
    Generates synthetic backgrounds with varied colors, textures, gradients,
    and adversarial blue patches to make the model robust against blue backgrounds/HSV failure.
    """
    w, h = size
    bg_type = random.choice(["table", "gradient", "noise", "blue_distractor", "room"])
    
    if bg_type == "table":
        # Wood / desk / floor texture
        base_color = np.array([random.randint(60, 160), random.randint(70, 180), random.randint(100, 210)], dtype=np.uint8)
        img = np.full((h, w, 3), base_color, dtype=np.uint8)
        # Add grain/planks
        noise = np.random.normal(0, 12, (h, w, 3)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # Add wood grain lines
        for _ in range(random.randint(5, 15)):
            y = random.randint(0, h)
            cv2.line(img, (0, y), (w, y + random.randint(-10, 10)), (int(base_color[0]*0.8), int(base_color[1]*0.8), int(base_color[2]*0.8)), random.randint(1, 3))
            
    elif bg_type == "gradient":
        c1 = np.random.randint(40, 220, 3)
        c2 = np.random.randint(40, 220, 3)
        grad = np.tile(np.linspace(0, 1, h)[:, None, None], (1, w, 3))
        img = (c1 * (1 - grad) + c2 * grad).astype(np.uint8)
        
    elif bg_type == "blue_distractor":
        # Adversarial scene with blue objects / blue fabric / blue table mat
        base_color = np.random.randint(50, 180, 3)
        img = np.full((h, w, 3), base_color, dtype=np.uint8)
        # Add blue patches/shapes that are NOT the target object
        for _ in range(random.randint(2, 6)):
            bx = random.randint(0, w - 80)
            by = random.randint(0, h - 80)
            bw = random.randint(40, 200)
            bh = random.randint(40, 200)
            # Blue HSV range
            blue_bgr = (random.randint(160, 255), random.randint(80, 180), random.randint(10, 60))
            if random.random() > 0.5:
                cv2.rectangle(img, (bx, by), (bx + bw, by + bh), blue_bgr, -1)
            else:
                cv2.circle(img, (bx + bw//2, by + bh//2), bw//2, blue_bgr, -1)
                
    else:  # room / texture
        c = np.random.randint(50, 200, 3)
        img = np.full((h, w, 3), c, dtype=np.uint8)
        noise = np.random.normal(0, 25, (h, w, 3)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
    return img

def apply_perspective_and_rotation(cutout, max_tilt=25, max_angle=360):
    """Applies 3D perspective warp, rotation, and scaling to the RGBA cutout."""
    h, w = cutout.shape[:2]
    
    # 1. Random 2D rotation
    angle = random.uniform(0, max_angle)
    center = (w / 2, h / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Compute new bounding dimensions for full rotation
    cos = np.abs(rot_mat[0, 0])
    sin = np.abs(rot_mat[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    rot_mat[0, 2] += (new_w / 2) - center[0]
    rot_mat[1, 2] += (new_h / 2) - center[1]
    
    rotated = cv2.warpAffine(cutout, rot_mat, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    
    # 2. Random 3D perspective tilt
    tilt_x = random.uniform(-max_tilt, max_tilt)
    tilt_y = random.uniform(-max_tilt, max_tilt)
    rh, rw = rotated.shape[:2]
    
    dx = int(abs(tilt_x) * rw / 100)
    dy = int(abs(tilt_y) * rh / 100)
    
    src_pts = np.float32([[0, 0], [rw, 0], [rw, rh], [0, rh]])
    dst_pts = np.float32([
        [random.randint(0, max(1, dx)), random.randint(0, max(1, dy))],
        [rw - random.randint(0, max(1, dx)), random.randint(0, max(1, dy))],
        [rw - random.randint(0, max(1, dx)), rh - random.randint(0, max(1, dy))],
        [random.randint(0, max(1, dx)), rh - random.randint(0, max(1, dy))]
    ])
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(rotated, M, (rw, rh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    
    # Crop tight to non-zero alpha
    alpha = warped[:, :, 3]
    coords = cv2.findNonZero(alpha)
    if coords is not None:
        bx, by, bw, bh = cv2.boundingRect(coords)
        return warped[by:by+bh, bx:bx+bw]
    return warped

def augment_cutout_lighting(cutout):
    """Augments brightness, contrast, shadows, and color temperature on cutout."""
    b, g, r, a = cv2.split(cutout)
    bgr = cv2.merge([b, g, r])
    
    # Random brightness and contrast
    alpha_contrast = random.uniform(0.7, 1.3)
    beta_brightness = random.uniform(-35, 35)
    bgr_aug = cv2.convertScaleAbs(bgr, alpha=alpha_contrast, beta=beta_brightness)
    
    # Random color temperature jitter
    hsv = cv2.cvtColor(bgr_aug, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] + random.randint(-8, 8), 0, 179)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + random.randint(-20, 20), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + random.randint(-25, 25), 0, 255)
    bgr_aug = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # Random shadow gradient over object
    if random.random() > 0.4:
        ch, cw = cutout.shape[:2]
        shadow_mask = np.linspace(random.uniform(0.4, 0.8), 1.0, cw)[None, :, None]
        if random.random() > 0.5:
            shadow_mask = np.fliplr(shadow_mask)
        bgr_aug = np.clip(bgr_aug.astype(np.float32) * shadow_mask, 0, 255).astype(np.uint8)
        
    return cv2.merge([bgr_aug[:, :, 0], bgr_aug[:, :, 1], bgr_aug[:, :, 2], a])

def paste_cutout(bg, cutout, scale_range=(0.18, 0.70)):
    """
    Pastes the cutout onto the background with alpha blending and returns the
    composite image and the exact bounding box [xmin, ymin, xmax, ymax].
    """
    bg_h, bg_w = bg.shape[:2]
    
    # Desired size relative to background
    target_scale = random.uniform(*scale_range)
    target_max_dim = int(min(bg_w, bg_h) * target_scale)
    
    ch, cw = cutout.shape[:2]
    aspect = cw / max(1, ch)
    if aspect >= 1.0:
        new_w = target_max_dim
        new_h = max(15, int(target_max_dim / aspect))
    else:
        new_h = target_max_dim
        new_w = max(15, int(target_max_dim * aspect))
        
    resized_cutout = cv2.resize(cutout, (new_w, new_h), interpolation=cv2.INTER_AREA if new_w < cw else cv2.INTER_LINEAR)
    
    # Random position on canvas (can be partially off-screen up to 10% to simulate realistic camera edges)
    max_x = bg_w - new_w
    max_y = bg_h - new_h
    
    if max_x <= 0 or max_y <= 0:
        return bg, None
        
    pos_x = random.randint(0, max_x)
    pos_y = random.randint(0, max_y)
    
    # Alpha blending
    cutout_bgr = resized_cutout[:, :, :3]
    cutout_alpha = (resized_cutout[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
    
    roi = bg[pos_y:pos_y+new_h, pos_x:pos_x+new_w].astype(np.float32)
    blended = (cutout_bgr.astype(np.float32) * cutout_alpha) + (roi * (1.0 - cutout_alpha))
    bg[pos_y:pos_y+new_h, pos_x:pos_x+new_w] = np.clip(blended, 0, 255).astype(np.uint8)
    
    # Calculate exact tight bounding box from the active alpha pixels
    alpha_mask = resized_cutout[:, :, 3] > 20
    coords = cv2.findNonZero(alpha_mask.astype(np.uint8))
    if coords is not None:
        bx, by, bw, bh = cv2.boundingRect(coords)
        bbox = (pos_x + bx, pos_y + by, pos_x + bx + bw, pos_y + by + bh)
        return bg, bbox
        
    return bg, None

def degrade_image_realism(img):
    """
    Applies motion blur, Gaussian blur, and camera ISO noise so images match real camera feeds.
    Decreases image sharpness as requested by user.
    """
    # 1. Subtle blur (Gaussian or Motion)
    if random.random() > 0.3:
        ksize = random.choice([3, 5])
        if random.random() > 0.5:
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        else:
            # Motion blur kernel
            kernel_motion = np.zeros((ksize, ksize))
            if random.random() > 0.5:
                kernel_motion[int((ksize-1)/2), :] = np.ones(ksize)
            else:
                kernel_motion[:, int((ksize-1)/2)] = np.ones(ksize)
            kernel_motion /= ksize
            img = cv2.filter2D(img, -1, kernel_motion)
            
    # 2. Camera sensor noise / ISO grain
    if random.random() > 0.3:
        noise_sigma = random.uniform(3, 14)
        noise = np.random.normal(0, noise_sigma, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
    # 3. JPEG compression artifact simulation
    if random.random() > 0.4:
        quality = random.randint(60, 92)
        _, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        
    return img

def build_synthetic_dataset(cutout_paths, bg_paths, output_dir="dataset", total_images=400, val_ratio=0.2, target_size=(640, 640)):
    """Main pipeline to generate synthetic YOLO dataset."""
    dataset_dir = Path(output_dir)
    images_train = dataset_dir / "images" / "train"
    images_val = dataset_dir / "images" / "val"
    labels_train = dataset_dir / "labels" / "train"
    labels_val = dataset_dir / "labels" / "val"
    previews_dir = dataset_dir / "previews"
    
    for d in [images_train, images_val, labels_train, labels_val, previews_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    cutouts = [cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in cutout_paths]
    cutouts = [c for c in cutouts if c is not None and c.shape[2] == 4]
    
    if not cutouts:
        raise ValueError("No valid RGBA cutouts found!")
        
    loaded_bgs = []
    for p in bg_paths:
        im = cv2.imread(p)
        if im is not None:
            loaded_bgs.append(im)
            
    print(f"Generating {total_images} synthetic YOLO images ({target_size[0]}x{target_size[1]})...")
    
    train_count = int(total_images * (1.0 - val_ratio))
    val_count = total_images - train_count
    
    preview_samples = []
    
    for idx in range(total_images):
        is_val = (idx >= train_count)
        split_name = "val" if is_val else "train"
        img_dest_dir = images_val if is_val else images_train
        lbl_dest_dir = labels_val if is_val else labels_train
        
        file_id = f"synth_{idx:05d}"
        img_file = img_dest_dir / f"{file_id}.jpg"
        lbl_file = lbl_dest_dir / f"{file_id}.txt"
        
        # Select background: 65% real COCO background, 35% procedural
        if loaded_bgs and random.random() < 0.65:
            raw_bg = random.choice(loaded_bgs)
            bg = cv2.resize(raw_bg, target_size, interpolation=cv2.INTER_LINEAR)
        else:
            bg = generate_procedural_background(size=target_size)
            
        # 8% of images are negative samples (background only, no target object)
        is_negative = (random.random() < 0.08)
        
        bboxes = []
        if not is_negative:
            # Number of objects in image: 85% single, 15% two instances
            num_instances = 2 if random.random() < 0.15 else 1
            for _ in range(num_instances):
                cutout = random.choice(cutouts)
                # Augment cutout
                aug_cutout = apply_perspective_and_rotation(cutout)
                aug_cutout = augment_cutout_lighting(aug_cutout)
                
                bg, bbox = paste_cutout(bg, aug_cutout)
                if bbox is not None:
                    bboxes.append(bbox)
                    
        # Apply camera realism blur/noise/JPEG artifacts (decreasing raw sharpness)
        final_img = degrade_image_realism(bg)
        
        # Save image
        cv2.imwrite(str(img_file), final_img)
        
        # Write YOLO format label: class_id x_center y_center width height (normalized)
        w_img, h_img = target_size
        with open(lbl_file, "w") as f:
            for (xmin, ymin, xmax, ymax) in bboxes:
                # Clamp to frame
                xmin = max(0, min(w_img - 1, xmin))
                ymin = max(0, min(h_img - 1, ymin))
                xmax = max(0, min(w_img - 1, xmax))
                ymax = max(0, min(h_img - 1, ymax))
                
                bw = xmax - xmin
                bh = ymax - ymin
                if bw < 8 or bh < 8:
                    continue
                    
                x_center = (xmin + xmax) / 2.0 / w_img
                y_center = (ymin + ymax) / 2.0 / h_img
                norm_w = bw / w_img
                norm_h = bh / h_img
                
                # Class 0: blue_object
                f.write(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                
        # Save preview images with bounding box overlay for first 16 samples
        if idx < 16:
            vis_img = final_img.copy()
            for (xmin, ymin, xmax, ymax) in bboxes:
                cv2.rectangle(vis_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                cv2.putText(vis_img, "blue_object", (int(xmin), max(20, int(ymin) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            preview_path = previews_dir / f"preview_{idx:02d}.jpg"
            cv2.imwrite(str(preview_path), vis_img)
            preview_samples.append(str(preview_path))
            
        if (idx + 1) % 50 == 0 or (idx + 1) == total_images:
            print(f"Generated {idx + 1}/{total_images} images...")
            
    # Write YOLO dataset configuration YAML
    abs_dataset_dir = dataset_dir.resolve()
    data_yaml_path = dataset_dir / "data.yaml"
    with open(data_yaml_path, "w") as f:
        f.write(f"# YOLOv11 Bounding Box Detection Dataset\n")
        f.write(f"path: {abs_dataset_dir}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"\nnames:\n")
        f.write(f"  0: blue_object\n")
        
    print("\nDataset generation complete!")
    print(f"  Train images: {train_count} -> {images_train}")
    print(f"  Val images:   {val_count}   -> {images_val}")
    print(f"  YOLO Config:  {data_yaml_path}")
    print(f"  Visual Previews (16 samples): {previews_dir}")
    return str(data_yaml_path)

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic YOLO bounding box dataset.")
    parser.add_argument("--num-images", type=int, default=400, help="Total synthetic images to generate (default: 400)")
    parser.add_argument("--cutout-dir", type=str, default="cutouts", help="Directory with transparent PNG cutouts")
    parser.add_argument("--bg-count", type=int, default=60, help="Number of COCO backgrounds to download")
    parser.add_argument("--output-dir", type=str, default="dataset", help="Output dataset directory")
    args = parser.parse_args()
    
    # 1. Ensure cutouts exist
    cutouts = glob.glob(os.path.join(args.cutout_dir, "*.png"))
    if not cutouts:
        print("No cutouts found. Running extract_cutouts.py first...")
        from extract_cutouts import extract_cutouts
        cutouts = extract_cutouts()
        
    # 2. Download / load backgrounds
    backgrounds = download_coco_backgrounds(count=args.bg_count)
    
    # 3. Generate dataset
    build_synthetic_dataset(
        cutout_paths=cutouts,
        bg_paths=backgrounds,
        output_dir=args.output_dir,
        total_images=args.num_images
    )

if __name__ == "__main__":
    main()
