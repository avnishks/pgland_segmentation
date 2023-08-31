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
def call_freeview(img, onehot, onehot_pred=None):
  fv = fs.Freeview()
  
  volume = img[0,0,:]
  fv.vol(volume)
  
  seg = torch.zeros(volume.shape)
  for i in range(onehot.shape[1]):
    seg += i * onehot[0,i,:]
  fv.vol(seg, colormap='lut')
    
  if onehot_pred is not None:
    seg_pred = torch.zeros(volume.shape)
    for i in range(onehot_pred.shape[1]):
      seg_pred += i * onehot_pred[0,i,:]
    fv.vol(seg_pred, colormap='lut')

  fv.show()



### Define loops up here (clean up later) ###
def training_loop(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  
  model.train()
  for X, y in dataloader:
    optimizer.zero_grad()
    
    X, y = X.to(device), y.to(device)
    with autocast():
      y_pred = model(X)
      loss = loss_fn(y_pred, y)

    loss.backward()
    optimizer.step()
    #breakpoint()
  
  print(f'Training loss: {loss.item():>.4f}')


  
def validation_loop(dataloader, model, loss_fn):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  loss, correct = 0, 0

  model.eval()
  for X, y in dataloader:
    X, y = X.to(device), y.to(device)
    
    with torch.no_grad():
      y_pred = model(X)
      loss += loss_fn(y_pred, y)
      correct += (y_pred.argmax(1) == y).type(torch.float32).sum().item()/y.numel()

  loss /= num_batches
  correct /= size

  print(f'Validation loss: {loss:>.4f}')
  #print(f'Accuracy: {(100*correct):>0.1f}%,  Validation loss: {loss:>.4f}') #(accuracy isn't correct..)



def testing_loop(dataloader, model, loss_fn):
  test_ind = 0
  test_loss, correct = 0, 0
  for X, y in dataloader:
    X, y = X.to(device), y.to(device)

    with torch.no_grad():
      y_pred = model(X)
      test_loss += loss_fn(y_pred, y)
      correct += (y_pred.argmax(1) == y).type(torch.float32).sum().item()/y.numel()

      #call_freeview(X.cpu().numpy(), y.cpu().numpy(), y_pred.cpu().numpy())

  test_loss /= len(test_loader)
  print(f'Loss: {test_loss:>.4f}')


  

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
               activation_type="ELU",
               pooling_type="MaxPool3d",
).to(device)
optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.BCEWithLogitsLoss()

aff = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
header = nib.Nifti1Header()


### Run the model ###
n_epochs = 10
for epoch in range(n_epochs):
  print(f"\nEpoch [{epoch+1}/{n_epochs}]\n-------------------")
  training_loop(train_loader, model, loss_fn, optimizer)
  validation_loop(valid_loader, model, loss_fn)

print('\n\nTesting...\n-------------------')
testing_loop(test_loader, model, loss_fn)
