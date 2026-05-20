import os
import torch
import logging
import torchvision
from torch.utils.data import DataLoader
from config import Config
from engine import DePoisonEngine
from submission import generate_submission
from dataset import ESADebrisDataset, collate_fn 

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def run_pipeline():
    print("--- PIPELINE STARTING ---")
    device = Config.DEVICE
    logging.info(f"Booting Neural Debris Removal Pipeline on {device.upper()}")
    
    # 1. Initialize RetinaNet
    print("Initializing model architecture...")
    model = torchvision.models.detection.retinanet_resnet50_fpn(weights=None)
    model.to(device)
    print("Model initialized.")

    # 2. Load Weights
    poisoned_weights_path = Config.POISONED_MODEL_PATH
    if os.path.exists(poisoned_weights_path):
        print(f"Loading weights from: {poisoned_weights_path}")
        checkpoint = torch.load(poisoned_weights_path, map_location=device, weights_only=True)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        print("Weights loaded successfully.")
    else:
        print(f"CRITICAL: No weights found at {poisoned_weights_path}.")
        return 
    
    # Freeze Backbone (Unlearning strategy)
    for name, param in model.named_parameters():
        if "backbone.body" in name:
            param.requires_grad = False
            
    # 3. Setup Dataset
    print(f"Checking unlearn data at: {Config.UNLEARN_SET_DIR}")
    train_dataset = ESADebrisDataset(img_dir=Config.UNLEARN_SET_DIR, device=device)
    print(f"Dataset length: {len(train_dataset)} images found.")
    
    if len(train_dataset) == 0:
        print("CRITICAL ERROR: Dataset length is 0. Pipeline exiting.")
        return

    # Use a persistent iterator to prevent memory re-allocations in the loop
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    data_iterator = iter(train_loader)
    
    # 4. Engine
    engine = DePoisonEngine(model)
    
    # 5. Training Loop
    print("Starting optimization loop...")
    for step in range(Config.MAX_STEPS):
        try:
            # Get next batch, restart if we run out of data
            try:
                images, targets = next(data_iterator)
            except StopIteration:
                data_iterator = iter(train_loader)
                images, targets = next(data_iterator)
                
            loss = engine.run_step(images, targets)
            
            if step % 10 == 0:
                print(f"Step {step}/{Config.MAX_STEPS} | Loss: {loss:.4f}")
                
        except Exception as e:
            print(f"Error during training loop at step {step}: {e}")
            break

    # 6. Human-in-the-Loop Validation Gate
    print("\n" + "="*40)
    print("OPTIMIZATION FINISHED.")
    print("Please check 'debug_0.png' in your project folder.")
    print("Verify if the boxes are accurately capturing debris vs background noise.")
    print("="*40)
    
    user_verification = input("Are the detections in debug_0.png accurate? (y/n): ").lower().strip()
    
    if user_verification == 'y':
        print("Verification passed. Generating submission...")
        generate_submission(
            unlearned_model=model,
            test_images_dir=Config.TEST_SET_DIR,
            output_csv_path="C:/code/unlearn/submission.csv",
            device=device
        )
        print("Submission generated: submission.csv")
    else:
        print("Submission aborted. Adjust A_PENALTY_WEIGHT in config.py or tweak preprocessing.")
        print("Pipeline exiting.")
    
    print("--- PIPELINE COMPLETE ---")

if __name__ == "__main__":
    run_pipeline()