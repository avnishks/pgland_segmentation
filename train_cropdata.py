import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.data_utils import PituitaryPinealDataset
from utils.model_utils import UNet3D
from utils import transform_utils as t_utils

from torchvision import transforms
from torchvision.transforms import Compose, Resize, ToTensor
from torchsummary import summary
from torch.cuda.amp import autocast

import pandas as pd
import numpy as np
import nibabel as nib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

transform = ToTensor()

spatial_aug = Compose([
  t_utils.RandomElasticAffineCrop(),
  t_utils.RandomLRFlip()
])

intensity_aug = Compose([
  t_utils.MinMaxNorm(),
  t_utils.ContrastAugmentation(),
  t_utils.BiasField(),
  t_utils.GaussianNoise()
])

dataset = PituitaryPinealDataset(image_label_list='data_config_crop.csv', 
                                 transform=transform,
                                 spatial_augmentation=spatial_aug,
                                 intensity_augmentation=intensity_aug
)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


model = UNet3D(in_channels=1, 
               out_channels=2, 
               n_features_start=24,
               num_blocks=3, 
               num_convs_per_block=2, 
               activation_type="ReLU",
               pooling_type="MaxPool3d").to(device)
# summary(model, input_size=input_size)

optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.BCELoss()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):

  for X, y in dataloader:
    optimizer.zero_grad()    
    
    # Forward pass and loss
    X, y = X.to(device), y.to(device)
    with autocast():
      y_pred = model(X)
      loss = loss_fn(y_pred, y)
    
    # Backward pass and optimize
    loss.backward()
    optimizer.step()
    
  print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print('Training complete!')

