import os
import torch
import torchvision.ops as ops
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm

def debug_visualize(img_path, boxes, scores, save_path="debug_output.png"):
    # Load original image for visualization
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    
    # Scale to 0-255 for drawing
    if img.dtype == np.uint16:
        img = (img.astype(np.float32) / 65535.0 * 255).astype(np.uint8)
    
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].int().tolist()
        score = scores[i].item()
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(save_path, img)

def generate_submission(unlearned_model, test_images_dir, output_csv_path, device):
    unlearned_model.eval()
    unlearned_model.to(device)
    
    threshold = 0.2 
    results = []
    image_files = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.png', '.jpg'))]
    
    print(f"Generating optimized predictions for {len(image_files)} images...")
    
    with torch.no_grad():
        for img_file in tqdm(image_files):
            img_path = os.path.join(test_images_dir, img_file)
            
            # --- MATCHED PREPROCESSING ---
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img.dtype == np.uint16:
                img = img.astype(np.float32) / 65535.0
            
            # Scale to 0-255 and ensure 3 channels
            img = np.clip(img * 255, 0, 255).astype(np.float32)
            if img.ndim == 2:
                img = np.repeat(img[:, :, None], 3, axis=2)
            
            # Convert to tensor and reorder to C, H, W
            img_tensor = torch.as_tensor(img.transpose(2, 0, 1).copy()).to(device).unsqueeze(0)
            
            # Inference
            prediction = unlearned_model(img_tensor)[0]
            
            # Filter and NMS
            keep = prediction['scores'] > threshold
            boxes = prediction['boxes'][keep]
            scores = prediction['scores'][keep]
            
            # Use IOU 0.4 to keep clustered detections
            keep_indices = ops.nms(boxes, scores, iou_threshold=0.4)
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            
            # Diagnostic
            if img_file == "0.png":
                debug_visualize(img_path, boxes, scores, "debug_0.png")
            
            # --- FORMAT & VALIDATE ---
            pred_strings = []
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i].tolist()
                w, h = x2 - x1, y2 - y1
                
                # FIX: Require a minimum size of > 1.0 pixel to prevent 
                # degenerate boxes (0.00 width/height) from entering the CSV
                if w > 1.0 and h > 1.0: 
                    pred_strings.append(f"{scores[i].item():.6f} {x1:.2f} {y1:.2f} {w:.2f} {h:.2f}")
            
            results.append({
                "image_id": img_file.replace('.png', ''),
                "prediction_string": " ".join(pred_strings)
            })

    df = pd.DataFrame(results)
    df.insert(0, "id", range(len(df)))
    df.to_csv(output_csv_path, index=False)