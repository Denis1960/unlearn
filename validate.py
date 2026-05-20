import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def calculate_local_map(model, val_dataset):
    model.eval()
    metric = MeanAveragePrecision()
    
    for images, targets in val_dataset:
        # Run inference
        preds = model(images)
        # Update metric
        metric.update(preds, targets)
        
    return metric.compute()

# Output will look like: 
# {'map': 0.45, 'map_50': 0.62, ...}