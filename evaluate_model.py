import os
import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from utils.dataset import SegmentationDataset
from models.model import UNet3D
from utils.data_utils import load_config, load_volume, save_volume, remap_labels
from utils.preprocessing import onehot
from utils.metrics import dice_coefficient, iou_score
from voxynth.voxynth.augment import apply_center_crop

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Configuration and Data Loading
config = load_config('configs/config.yaml')

batch_size = config['evaluation']['batch_size']
predictions_dir = config['evaluation']['output_folder']
num_classes = config['model']['num_classes']
nb_features = config['model']['nb_features']
nb_levels = config['model']['nb_levels']

test_dataset = SegmentationDataset(config['dataset']['test_file'], transform=False)  # No augmentation for testing
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load label mapping (if necessary)
with open(os.path.join(config['training']['output_folder'], 'label_mapping.json'), 'r') as f:
    label_mapping = json.load(f) 

# 2. Load the Trained Model
model_path = 'output/checkpoints/best_model_epoch12.pth'  # Or path to any other checkpoint

sample_input, _ = test_dataset[0] 
input_shape = sample_input.shape[1:] 

model = UNet3D(
    input_shape=(1, *input_shape),
    nb_features=nb_features, 
    nb_levels=nb_levels, 
    nb_labels=num_classes,
).to(device)

model.load_state_dict(torch.load(model_path)['model_state_dict'])
model.eval()

# 3. Evaluation Loop
with torch.no_grad():
    for idx, (images, labels) in enumerate(test_loader):  
        images, labels = images.to(device), labels.to(device)

        # Remap labels and apply one-hot encoding 
        labels = remap_labels(labels, label_mapping) 
        labels = onehot(labels, num_classes=len(label_mapping), device=device) 

        # Apply center cropping to match training data size 
        images = apply_center_crop(images, (160, 160, 160)) 
        labels = apply_center_crop(labels, (160, 160, 160)) 

        outputs = model(images)

        # Calculate metrics
        dice = dice_coefficient(outputs, labels)
        iou = iou_score(outputs, labels)
        print(f"Sample {idx}, Dice: {dice:.4f}, IoU: {iou:.4f}")

        # Save predictions
        predicted_segmentation = torch.argmax(outputs, dim=1).squeeze(0).cpu()
        original_image, _ = load_volume(test_dataset.image_files[idx])

        os.makedirs(predictions_dir, exist_ok=True)

        save_volume(predicted_segmentation, original_image, os.path.join(predictions_dir, f"prediction_{idx}.mgz"))