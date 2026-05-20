# C:\code\unlearn\config.py
import os
import torch

class Config:
    # Infrastructure Settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP = True  # Employs your ZBook's Tensor Cores for speed
    
    # Directory Layout Paths
    DATA_DIR = "C:/code/unlearn/data"
    CHECKPOINT_DIR = "C:/code/unlearn/checkpoints"
    
    # Specific Dataset Paths
    TEST_SET_DIR = os.path.join(DATA_DIR, "test_set")
    UNLEARN_SET_DIR = os.path.join(DATA_DIR, "unlearn_set")
    
    # Points directly to the file inside the poison_model folder
    POISONED_MODEL_PATH = os.path.join(DATA_DIR, "poisoned_model", "poisoned_model.pth") 
    
    # Hyperparameters
    BATCH_SIZE = 2
    LEARNING_RATE = 1e-5
    MAX_STEPS = 40
    CHECKPOINT_INTERVAL = 10
    A_PENALTY_WEIGHT = 0.5  # Scale factor for gradient inversion