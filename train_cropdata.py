import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.data_utils import PituitaryPinealDataset
from utils.model_utils import UNet3D
from utils import transform_utils as t_utils
import losses

from torchvision import transforms
from torchsummary import summary
from torch.cuda.amp import autocast

import pandas as pd
import numpy as np
import nibabel as nib


### Define loops up here (clean up later) ###
def training_loop(dataloader, model, loss_fn, optimizer):
  model.train()
  for X, y in dataloader:
    optimizer.zero_grad()
    
    X, y = X.to(device), y.to(device)
    with autocast():
      y_pred = model(X)
      loss = loss_fn(y_pred, y)

    loss.backward()
    optimizer.step()

  print(f'Training loss: {loss.item():>.4f}')


    

def validation_loop(dataloader, model, loss_fn):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  valid_loss, correct = 0, 0

  model.eval()
  for X, y in dataloader:
    X, y = X.to(device), y.to(device)
    
    with torch.no_grad():
      y_pred = model(X)
      valid_loss += loss_fn(y_pred, y)
      correct += (y_pred.argmax(1) == y).type(torch.float32).sum().item()

  valid_loss /= num_batches
  correct /= size

  print(f'Validation loss: {valid_loss:>.4f}')
  #print(f'Accuracy: {(100*correct):>0.1f},  Avg loss: {valid_loss:>.4f}')




### Set up ###
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

transform = transforms.ToTensor()
spatial_aug = t_utils.Compose([t_utils.RandomElasticAffineCrop(),
                               t_utils.RandomLRFlip()
])
intensity_aug = t_utils.Compose([#t_utils.ContrastAugmentation(),
                                 #t_utils.BiasField(),
                                 #t_utils.GaussianNoise()
                                 t_utils.MinMaxNorm(),
])
data_labels = (0, 883, 900, 903, 904)
dataset = PituitaryPinealDataset(image_label_list='data_config_crop.csv', 
                                 transform=transform,
                                 spatial_augmentation=spatial_aug,
                                 intensity_augmentation=intensity_aug,
                                 data_labels=data_labels,
)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

model = UNet3D(in_channels=1, 
               out_channels=len(data_labels),
               n_features_start=24,
               n_blocks=3, 
               n_convs_per_block=2, 
               activation_type="ReLU",
               pooling_type="MaxPool3d",
).to(device)
optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.BCEWithLogitsLoss()



### Train the model ###
n_epochs = 10
for epoch in range(n_epochs):
  print(f"\nEpoch [{epoch+1}/{n_epochs}]\n-------------------")
  training_loop(dataloader, model, loss_fn, optimizer)
  validation_loop(dataloader, model, loss_fn)
  
print('Done!')
