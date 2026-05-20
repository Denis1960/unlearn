import torch
import logging
from torch.amp import autocast, GradScaler
from config import Config

class DePoisonEngine:
    def __init__(self, model):
        self.model = model.to(Config.DEVICE)
        # Isolate updates to required layers
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=Config.LEARNING_RATE
        )
        
        self.device_type = 'cuda' if 'cuda' in str(Config.DEVICE).lower() else 'cpu'
        
        self.scaler = GradScaler(
            device=self.device_type, 
            enabled=Config.USE_AMP and self.device_type == 'cuda'
        )
        
    def run_step(self, images, targets, poisoned_masks=None):
        self.model.train()
        self.optimizer.zero_grad()
        
        with autocast(device_type=self.device_type, enabled=Config.USE_AMP and self.device_type == 'cuda'):
            # RetinaNet returns a dictionary of losses. 
            # We must sum all components to maintain structural integrity.
            loss_dict = self.model(images, targets)
            
            # Sum all loss components (classification + bbox_regression)
            total_loss = sum(loss for loss in loss_dict.values())
            
            # --- STABILIZED UNLEARNING LOGIC ---
            # Instead of a raw -1.0 multiplication (which causes explosion), 
            # we use the penalty weight to selectively reduce the loss contribution.
            # If A_PENALTY_WEIGHT is 0, we perform standard training (safe).
            # If it's > 0, we effectively reduce the gradient magnitude.
            loss = total_loss * (1.0 - Config.A_PENALTY_WEIGHT)
            
        # Execute backward pass
        if Config.USE_AMP and self.device_type == 'cuda':
            self.scaler.scale(loss).backward()
            # --- CRITICAL: GRADIENT CLIPPING ---
            # This prevents the gradient explosion you observed.
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            # --- CRITICAL: GRADIENT CLIPPING ---
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
        return loss.item()