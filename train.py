import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.data_utils import BrainMRIDataset
from utils.model_utils import UNet3D
from torchvision import transforms
from torchvision.transforms import Compose, Resize, ToTensor
from torchsummary import summary
from torch.cuda.amp import autocast

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

transforms = Compose([
    ToTensor() ,
    # Resize(32),
])

dataset = BrainMRIDataset(
  image_label_list='image_label_pairs.csv', 
  transform=transforms,
  )

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

model = UNet3D(in_channels=1, out_channels=2).to(device)
# input_size = (1, 128, 128, 128)
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