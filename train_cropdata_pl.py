import os
import warnings
import argparse
import logging

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

import options
import models
from models import optimizers, unet
from models.segment_pl import Segment as segment
import models.loss_functions as loss_fns
from models.progress import ProgressBar as ProgressBar
from models import losses



def setup_log(file, str):
    f = open(file, 'w')
    f.write(str + '\n')
    f.close()
    


### Arg parsing
parser = argparse.ArgumentParser()
#parser = pl.Trainer.add_argparse(parser)
parser = options.set_argparse_defs(parser)
parser = options.add_argparse(parser)
args = parser.parse_args()



### Torch set-up ###
pl.seed_everything(args.seed, workers=True) #####
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('medium')

warnings.filterwarnings('ignore',
                        "Your \\`val_dataloader\\`\\'s sampler has shuffling enabled, it is strongly recommended that you turn shuffling off for val/test/predict dataloaders.")



### Data set-up ###
data_config = args.data_config
aug_config = args.aug_config
train_data, valid_data, test_data = call_dataset(data_config=data_config, aug_config=aug_config)

train_loader = DataLoader(train_data,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=args.n_workers,
                          pin_memory=True
)
valid_loader = DataLoader(valid_data,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=args.n_workers,
                          pin_memory=True
)
test_loader = DataLoader(test_data,
                         batch_size=args.batch_size,
                         shuffle=False,
                         num_workers=args.n_workers,
                         pin_memory=True
)


### Model set-up ###
lr_start = 0.0001 #args.lr_start
lr_param = 0.1 #args.lr_param
decay = 0.002 #args.weight_decay
schedule = 'poly' #args.lr_scheduler

network = models.unet.__dict__[args.network](in_channels=train_data.__numinput__(),
                                             out_channels=train_data.__numclass__(),
).to(device)
optimizer = models.optimizers.adam(network.parameters(),
                                   lr=lr_start,
                                   weight_decay=decay
)
#schedule = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=5, power=0.9)
loss_fn =  loss_fns.__dict__[args.loss]

metrics_train = [models.losses.__dict__[args.metrics_train[i]]() for i in range(len(args.metrics_train))]
metrics_test = [models.losses.__dict__[args.metrics_test[i]]() for i in range(len(args.metrics_test))]
metrics_valid = [models.losses.__dict__[args.metrics_valid[i]]() for i in range(len(args.metrics_valid))]

output_folder = args.output_dir
if output_folder is not None and not os.path.exists(output_folder):  os.mkdir(output_folder)

trainee = segment(model=network,
                  optimizer=optimizer,
                  loss=loss_fn,
                  train_data=train_data,
                  valid_data=valid_data,
                  test_data=test_data,
                  output_folder=output_folder,
                  seed=args.seed,
                  lr_start=lr_start,
                  lr_param=lr_param,
                  train_metrics=metrics_train,
                  valid_metrics=metrics_valid,
                  test_metrics=metrics_test,
                  save_train_output_every=50,
                  save_valid_output_every=50,
                  schedule=schedule,
)


### Set up log files
"""
setup_log(os.path.join(output_folder, "training_loss.txt"), "Epoch AvgLoss " + \
          " ".join(args.metrics_train[i] for i in range(len(args.metrics_train)))) + \
          " ".join(args.metrics_valid[i] for i in range(len(args.metrics_valid)))
"""
if output_folder is not None:
    setup_log(os.path.join(output_folder, "testing_loss.txt"), "ImgID Loss Accuracy " + \
              " ".join(args.metrics_test[i] for i in range(len(args.metrics_test))))


### Run ? ###
print("Train: %d | Valid: %d | Tests: %d" % \
      (len(train_loader.dataset), len(valid_loader.dataset), len(test_loader.dataset)))

callbacks = [pl_ModelCheckpoint(monitor='valid_metric_0',
                                mode='max',
                                dirpath=output_folder,
                                filename='best',
                                save_last=True,
                                every_n_epochs=1),
]

trainer = pl.Trainer(accelerator='gpu',
                     callbacks=callbacks,
                     enable_progress_bar=False,
                     devices=1,
                     log_every_n_steps=len(train_loader),
                     max_epochs=2000, #args.max_n_epochs,
                     gradient_clip_val=0.5,
                     gradient_clip_algorithm='value',
                     precision=16,
                     default_root_dir=output_folder,
)

trainer.fit(trainee, train_loader, valid_loader)
trainer.validate(trainee, valid_loader, verbose=False)
trainer.test(trainee, test_loader, verbose=False)



