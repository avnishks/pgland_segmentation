import os
import warnings
import numpy as np
import nibabel as nib
import freesurfer as fs

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint as pl_ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers


from dataset.pituitarypineal_crop import call_dataset
from dataset import transforms as t

from model.segment import Segment as segment
from model.unet import UNet3D
import model.loss_functions as loss_fns
from model.progress import ProgressBar as ProgressBar
from model import losses



### Torch set-up ###
pl.seed_everything(0, workers=True) #####
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('medium')

warnings.filterwarnings('ignore',
                        "Your \\`val_dataloader\\`\\'s sampler has shuffling enabled, it is strongly recommended that you turn shuffling off for val/test/predict dataloaders.")


### Data set-up ###
#output_base = "no_augmentation"
output_base = "augmentation_2"
output_dir = os.path.join("data","results", output_base)

data_config = "dataset/data_config.csv"
augmentation_config = os.path.join(output_dir, "augmentation_parameters.txt") \
    if output_base != "no_augmentation" else None

train_data, valid_data, test_data = call_dataset(data_config=data_config,
                                                 augmentation_config=augmentation_config)

train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
valid_loader = DataLoader(valid_data, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)


### Model set-up ###
lr_start = 0.0001
lr_param = 0.9
momentum = 0.9
decay = 0.0000

model = UNet3D(in_channels=1, out_channels=train_data.__numclass__()).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_start, weight_decay=decay)
loss = loss_fns.dice_cce_loss
metrics=[losses.MeanDice()]

trainee = segment(model=model, optimizer=optimizer, loss=loss, \
                  train_data=train_data, valid_data=valid_data, test_data=test_data, output_folder=output_dir,
                  seed=0, lr_start=lr_start, lr_param=lr_param,
                  train_metrics=metrics, valid_metrics=metrics, test_metrics=metrics,
                  save_train_output_every=1, save_valid_output_every=0, schedule='poly',
)



### Run ? ###
print("Train: %d | Valid: %d | Tests: %d" % \
      (len(train_loader.dataset), len(valid_loader.dataset), len(test_loader.dataset)))

callbacks = [pl_ModelCheckpoint(monitor='val_metric0', mode='max'),
             ProgressBar(refresh_rate=1)]
#logger = pl_loggers.TensorBoardLogger('logs/', name=output_base, default_hp_metric=False)

trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=10000,
                     gradient_clip_val=0.5, gradient_clip_algorithm='value', precision=16)

trainer.fit(trainee, train_loader, valid_loader)
trainer.validate(trainee, valid_loader, verbose=False)
trainer.test(trainee, test_loader, verbose=False)



