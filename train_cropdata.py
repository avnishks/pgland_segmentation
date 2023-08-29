import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.data_utils import call_dataset

from utils.model_utils import UNet3D
from utils import transform_utils as t_utils
import losses

from torchvision import transforms
from torchsummary import summary
from torch.cuda.amp import autocast

import pandas as pd
import numpy as np
import nibabel as nib
import freesurfer as fs



### Data visualization tool
def call_freeview(img, onehot):
  volume = img[0,0,:]
  seg = torch.zeros(volume.shape)

  for i in range(onehot.shape[1]):
    seg += i * onehot[0,i,:]
      
  fv = fs.Freeview()
  fv.vol(volume)
  fv.vol(seg, colormap='lut')
  fv.show()



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
      correct += (y_pred.argmax(1) == y).type(torch.float32).sum().item()/y.numel()

  valid_loss /= num_batches
  correct /= size

  #print(f'Validation loss: {valid_loss:>.4f}')
  print(f'Accuracy: {(100*correct):>0.1f}%,  Avg loss: {valid_loss:>.4f}')




### Set up ###
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

data_config = 'data_config_crop.csv'
train_data, valid_data, test_data, n_labels = call_dataset(data_config)

train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=1, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

model = UNet3D(in_channels=1, 
               out_channels=n_labels,
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
  training_loop(train_loader, model, loss_fn, optimizer)
  validation_loop(valid_loader, model, loss_fn)
  
print('Done!')
