import os
import time
import numpy as np

import torch
import torch.optim
import torch.nn as nn
import pytorch_lightning as pl


"""
Syntax within subfunctions:
inputs = inputs image data
target = target onehot label map
output = predicted onehot label map

"""


class Segment(pl.LightningModule):
    def __init__(self,
                 model, optimizer, loss,
                 train_data, valid_data, test_data, output_folder,
                 train_metrics, valid_metrics, test_metrics,
                 seed, lr_start, lr_param, schedule,
                 save_train_output_every, save_valid_output_every,
                 start_aug_on_epoch=20,
                 **kwargs
    ):
        super().__init__()
        
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.lr_start =	lr_start
        self.lr_param =	lr_param
        self.start_aug_on_epoch = start_aug_on_epoch
        
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        
        self.train_metrics = nn.ModuleList(train_metrics)
        self.valid_metrics = nn.ModuleList(valid_metrics)
        self.test_metrics = nn.ModuleList(test_metrics)
        
        self.save_train_output_every = save_train_output_every
        self.save_valid_output_every = save_valid_output_every
        self.output_folder = output_folder
        self.schedule = schedule

        if output_folder is not None:
            self.train_output = os.path.join(output_folder, "training_loss.txt")
            f = open(self.train_output, 'w')
            f.close()
            self.test_output = os.path.join(output_folder, "testing_loss.txt")
            f = open(self.test_output, 'w')
            f.close()
        else:
            self.train_output = None
            self.test_output = None
        self.valid_output = []

        self.test_losses = []
        self.test_accuracy = []

        
        
    def training_step(self, batch, idx):
        if self.current_epoch < self.start_aug_on_epoch:
            inputs, target = self.train_data.init_augmentation(batch[0], batch[1])
        else:
            inputs, target = self.train_data.full_augmentation(batch[0], batch[1])
        inds = batch[2]
        
        output = self.model(inputs)
        train_loss = self.loss(output, target, gpu=True)
        
        [self.train_metrics[i].update(output, target) for i in range(len(self.train_metrics))]
        [self.log('train_loss', train_loss, prog_bar=False, logger=True, on_epoch=True, sync_dist=True)]
        loss = train_loss

        [self.log('train_metric_%d' % i, self.train_metrics[i], \
                  prog_bar=False, logger=False, on_epoch=True, sync_dist=True) \
         for i in range(len(self.train_metrics))]

        #[self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True, logger=True)]
        
        return loss

    
    def validation_step(self, batch, idx):
        if self.current_epoch <	self.start_aug_on_epoch:
            inputs, target = self.valid_data.init_augmentation(batch[0], batch[1])
        else:
            inputs, target = self.valid_data.full_augmentation(batch[0], batch[1]) #, cpu=False, gpu=True)
        inds = batch[2]

        output = self.model(inputs)
        valid_loss = self.loss(output, target, gpu=True)
        
        [self.valid_metrics[i].update(output, target) for i in range(len(self.valid_metrics))]
        loss = valid_loss
        
        [self.log('valid_metric_%d' % i, self.valid_metrics[i], \
                  prog_bar=False, logger=False, on_epoch=True, sync_dist=True) \
         for i in range(len(self.valid_metrics))]
        #if (self.save_valid_output_every != 0) and ((self.global_step + 1) % self.save_valid_output_every == 0):
        #    self.save_output('valid_output', self.valid_data, y_pred, y, indices)


        return loss


    def test_step(self, batch, idx):
        inputs, target = self.test_data.full_augmentation(batch[0], batch[1], cpu=False, gpu=True)
        inds = batch[2]

        output = self.model(inputs)
        test_loss = self.loss(output, target)
        self.test_losses.append(test_loss.item())

        accuracy = torch.sum([target==output][0]).item()/torch.numel(target)
        self.test_accuracy.append(accuracy)
        
        [self.test_metrics[i].update(output, target) for i in range(len(self.test_metrics))]
        loss = test_loss
        
        #[self.log('test_loss', test_loss, prog_bar=True, logger=True)]
        #[self.log('test_metric_%d' % i, self.test_metrics[i], \
        #          prog_bar=True, logger=True, sync_dist=True) for i in range(len(self.test_metrics))]
        if self.output_folder is not None:
            self.save_output(os.path.join(self.output_folder, "test_data"),
                             self.test_data, inputs, target, output, idx, is_test_data=True)
        
        return loss


    def save_output(self, folder, dataset, inputs, target, output, idx, is_test_data:bool=False):
        if folder is not None:
            basename = dataset.label_files[idx].split("/")[-1].split(".")[0:2]
            if not os.path.exists(folder):  os.mkdir(folder)
            
            inputs_path = os.path.join(folder, ".".join(basename) + ".inputs.mgz")
            target_path = os.path.join(folder, ".".join(basename) + ".target.mgz")
            output_path = os.path.join(folder, ".".join(basename) + ".output.mgz")
            
            dataset._save_output(inputs, inputs_path, dtype=np.float32)
            dataset._save_output(target, target_path, dtype=np.int32, is_onehot=True)
            dataset._save_output(output, output_path, dtype=np.int32, is_onehot=True)

            
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().item()
        metric = self.train_metrics[0].compute().item()

        if (self.save_train_output_every != 0) and ((self.global_step + 1) % self.save_train_output_every == 0):
            print(f'{self.current_epoch} {avg_loss:>.4f} {metric:>.5f} \n')
            if self.train_output is not None:
                f = open(self.train_output, 'a')
                f.write(f'{self.current_epoch} {avg_loss:>.4f} {metric:>.5f}\n')
                f.close()
                #    self.save_output('valid_output', self.valid_data, y_pred, y, indices)
        
        
    def validation_epoch_end(self, outputs):
        print(''.join(['%s' % metric.__repr__() for metric in self.valid_metrics]))


    def test_epoch_end(self, outputs):
        if self.test_output is not None:
            f = open(self.test_output, 'a')
            for x in range(len(self.test_losses)):
                f.write(f'{self.test_losses[x]:>.4f} {self.test_accuracy[x]:>.4f}\n')
            f.close()
            
        print(''.join(['%s' % metric.__repr__() for metric in self.test_metrics]))


    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop('loss', None)
        items.pop('v_num', None)

        return items


    def configure_optimizers(self):
        def lr(step):
            if self.schedule == 'poly':
                return (1.0 - (step / self.trainer.max_steps)) ** self.lr_param
            elif self.schedule == 'step':
                return (0.1 ** (step // self.lr_param))
            else:
                return 1.0

        if self.schedule == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr)
        lr_scheduler = {'interval':'epoch' if self.schedule == 'plateau' else 'step', \
                        'scheduler':scheduler, 'monitor':'val_metric0'}

        return {'optimizer':self.optimizer, 'lr_scheduler':lr_scheduler}
