# C:\code\unlearn\dataset.py
import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

class ESADebrisDataset(Dataset):
    def __init__(self, img_dir, device="cuda"):
        self.img_dir = img_dir
        self.device = device
        
        # Supported image extensions
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.fits')
        
        if os.path.exists(img_dir):
            self.img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_extensions)]
        else:
            self.img_files = []

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # --- MODIFIED: Use OpenCV to read 16-bit PNGs as per competition reference ---
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        # Handle 16-bit to float32 normalization
        if img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        
        # Scale to 0-255 range and ensure 3 channels
        img = np.clip(img * 255, 0, 255).astype(np.float32)
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)
            
        # Convert to tensor and reorder to C, H, W
        img_tensor = torch.as_tensor(img.transpose(2, 0, 1).copy()).to(self.device)
        
        # Target setup for RetinaNet
        boxes = [[0.0, 0.0, 1.0, 1.0]]
        labels = [0] 
            
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32, device=self.device),
            "labels": torch.tensor(labels, dtype=torch.long, device=self.device)
        }
        
        return img_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))