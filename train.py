import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

import matplotlib
matplotlib.use('agg') # Use non-interactive backend
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from utils.dataset import SegmentationDataset
from models.model import UNet3D
from utils.losses import DiceLoss
from utils.metrics import dice_coefficient, iou_score
from utils.preprocessing import onehot
from utils.data_utils import load_config, save_label_mapping, remap_labels
from sklearn.model_selection import train_test_split


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = load_config('configs/config.yaml') 

expected_num_channels = config['dataset']['expected_num_channels']
expected_classes = config['dataset']['expected_classes']
batch_size = config['training']['batch_size']
nb_levels = config['model']['nb_levels']
nb_features = config['model']['nb_features']
learning_rate = config['optimizer']['learning_rate']
num_epochs = config['training']['num_epochs']


# Load full dataset list
with open(config['dataset']['dataset_list_file'], 'r') as file:
    full_dataset_list = yaml.safe_load(file) 

# Split into train, validation, and test sets (70/15/15 split)
train_list, val_test_list = train_test_split(full_dataset_list, test_size=0.3, random_state=42) 
val_list, test_list = train_test_split(val_test_list, test_size=0.5, random_state=42) 

# Save the split dataset lists to separate YAML files
with open(config['dataset']['train_file'], 'w') as file:
    yaml.dump(train_list, file)
with open(config['dataset']['val_file'], 'w') as file:
    yaml.dump(val_list, file)
with open(config['dataset']['test_file'], 'w') as file:
    yaml.dump(test_list, file)


# Create dataset and data loaders
train_dataset = SegmentationDataset(config['dataset']['train_file'])
val_dataset = SegmentationDataset(config['dataset']['val_file'])

label_mapping = save_label_mapping(train_dataset.get_all_labels(), 
                                    output_folder=config['training']['output_folder'])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

sample_input, _ = train_dataset[0]
input_shape = sample_input.shape[1:]

# verify dataset integrity
unique_classes = set()
for _, label in train_dataset:
    unique_values = torch.unique(label).tolist()
    unique_classes.update(unique_values)

num_classes = len(unique_classes)
print("Dataset information:")
print(f"Number of samples in training dataset: {len(train_dataset)}")
print(f"Number of unique classes: {num_classes}")
print(f"Unique class values: {sorted(unique_classes)}")
print(f"Input shape: {input_shape}")
print(f"Number of channels: {sample_input.shape[0]}")

assert sorted(unique_classes) == expected_classes, f"Expected classes {expected_classes}, but got {sorted(unique_classes)}"
assert sample_input.shape[0] == expected_num_channels, f"Expected {expected_num_channels} channels, but got {sample_input.shape[0]}"

# Create model, loss function, and optimizer
model = UNet3D(
    input_shape=(1, *input_shape), 
    nb_features=nb_features, 
    nb_levels=nb_levels, 
    nb_labels=num_classes
).to(device)

criterion = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
val_losses = []
train_dices = []
val_dices = []
# train_ious = []
# val_ious = []
best_val_loss = float('inf')

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_dice = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        labels = remap_labels(labels, label_mapping)  # Remap labels
        labels = onehot(labels, num_classes=len(label_mapping), device=device)


        optimizer.zero_grad()
        outputs = model(images)
        # print(f"[DEBUG-train] outputs: {outputs.shape}, labels: {labels.shape}")
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_dice += dice_coefficient(outputs, labels)

    train_loss /= len(train_loader)
    train_dice /= len(train_loader)

    # Validation loop
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            labels = remap_labels(labels, label_mapping)
            labels = onehot(labels, num_classes=len(label_mapping), device=device) 

            outputs = model(images)
            # print(f"[DEBUG-val] outputs: {outputs.shape}, labels: {labels.shape}")
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            val_dice += dice_coefficient(outputs, labels)

    val_loss /= len(val_loader)
    val_dice /= len(val_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

    # Visualize learning curves
    train_losses.append(train_loss)
    train_dices.append(train_dice)
    # train_ious.append(train_iou)
    val_losses.append(val_loss)
    val_dices.append(val_dice)
    # val_ious.append(val_iou)

    if (epoch+1) % 5 == 0 or (epoch+1) == num_epochs:  
        # Create output directory for plots if it doesn't exist
        plot_dir = 'output/training_plots'
        os.makedirs(plot_dir, exist_ok=True)

        # 1. Loss Plot
        plt.figure()  # Create a new figure for the loss plot
        plt.plot(train_losses, label='Training Loss') 
        plt.plot(val_losses, label='Validation Loss') 
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        loss_plot_path = os.path.join(plot_dir, 'loss_plot.png')
        plt.savefig(loss_plot_path)
        plt.clf() 

        # 2. Dice Coefficient Plot
        plt.figure()  # Create a new figure for the Dice plot
        plt.plot(train_dices, label='Training Dice')
        plt.plot(val_dices, label='Validation Dice')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Coefficient')
        plt.legend()
        dice_plot_path = os.path.join(plot_dir, 'dice_plot.png')
        plt.savefig(dice_plot_path)
        plt.clf()

        # # 3. IoU Plot 
        # plt.figure() # Create a new figure for the IoU plot 
        # plt.plot(train_ious, label='Training IoU')
        # plt.plot(val_ious, label='Validation IoU')
        # plt.xlabel('Epoch')
        # plt.ylabel('IoU')
        # plt.legend()
        # iou_plot_path = os.path.join(plot_dir, 'iou_plot.png')
        # plt.savefig(iou_plot_path)
        # plt.clf()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'dice': val_dice,
            # 'iou': val_iou,
        }
        checkpoint_path = f"output/checkpoints/best_model_epoch{epoch+1}.pth"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint_dict, checkpoint_path)