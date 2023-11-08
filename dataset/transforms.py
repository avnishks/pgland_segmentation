import sys
import numpy as np
from numpy import random as npr
import math
import random

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

import cornucopia as cc #https://github.com/balbasty/cornucopia/tree/feat-psf-slicewise
from cornucopia import random as cc_rand


class Compose(transforms.Compose):
    def __init__(self, transforms, gpuindex=1):
        super().__init__(transforms)
        self.gpuindex = gpuindex

    def __call__(self, *args, cpu=True, gpu=True, **kwargs):
        if cpu:
            for t in self.transforms[:self.gpuindex]:
                args = t(*args)
        if gpu:
            for t in self.transforms[self.gpuindex:]:
                args = t(*args)

        return args



class GetPatch:
    def __init__(self, patch_size:[int, list], n_dims:int):
        self.X = n_dims
        self.patch_size = [patch_size] * n_dims if isinstance(patch_size, int) else patch_size


    def _get_patch(self, vol):
        if self.X == 2:
            h0, w0 = vol.shape[-2:]
            h1, w1 = self.patch_size
            return vol[..., (h0-h1)//2:h0-(h0-h1)//2, (w0-w1)//2:w0-(w0-w1)//2]
        elif self.X >= 3:
            h0, w0, d0 = vol.shape[-3:]
            h1, w1, d1 = self.patch_size
            return vol[..., (h0-h1)//2:h0-(h0-h1)//2, (w0-w1)//2:w0-(w0-w1)//2, (d0-d1)//2:d0-(d0-d1)//2]
        else:
            print('Invalid n_dims')
            
        
    def __call__(self, img, seg):
        img = self._get_patch(img)
        seg = self._get_patch(seg)

        return img, seg



class RandomElasticAffineCrop:
    def __init__(self,
                 translation_bounds:[float,list]=0.0,
                 rotation_bounds:[float,list]=15,
                 shear_bounds:[float,list]=0.012,
                 scale_bounds:[float,list]=0.15,
                 max_elastic_displacement:[float,list]=0.15,
                 n_elastic_control_pts:int=5,
                 n_elastic_steps:int=0,
                 order:int=3,
                 patch_size:list=None,
                 n_dims:int=3,
                 **kwargs
    ):
        X = n_dims
        if isinstance(translation_bounds, list): assert len(translation_bounds) == n_dims
        if isinstance(rotation_bounds, list): assert len(rotation_bounds) == n_dims
        if isinstance(shear_bounds, list): assert len(shear_bounds) == n_dims
        if isinstance(scale_bounds, list): assert len(scale_bounds) == n_dims
        if isinstance(patch_size, list): assert len(patch_size) == n_dims

        self.translations = [translation_bounds] * X \
            if isinstance(translation_bounds, float) else translation_bounds
        self.rotations = [rotation_bounds] * X \
            if isinstance(rotation_bounds, float) else rotation_bounds
        self.shears = [shear_bounds]  * X \
            if isinstance(shear_bounds, float) else shear_bounds
        self.zooms = [scale_bounds]  * X \
            if isinstance(scale_bounds, float) else scale_bounds
        self.dmax = [max_elastic_displacement] * X \
            if isinstance(max_elastic_displacement, float) else max_elastic_displacement
        self.shape = n_elastic_control_pts
        self.steps = n_elastic_steps
        self.patch = patch_size
        
        self.transform = cc.RandomAffineElasticTransform(translations=self.translations,
                                                         rotations=self.rotations,
                                                         shears=self.shears,
                                                         zooms=self.zooms,
                                                         dmax=self.dmax,
                                                         shape=self.shape,
                                                         steps=self.steps,
                                                         patch=self.patch
        )
        
        
    def __call__(self, img, seg):
        img, seg = self.transform(img, seg)
        return img, seg



class RandomLRFlip:
    def __init__(self, chance:float=0.5):
        self.chance = chance if chance >= 0 and chance <= 1 \
            else Exception("Invalid chance (must be float between 0 and 1)")
        self.transform = cc.FlipTransform(axis=0)


    def __call__(self, img, seg):
        img, seg = cc.MaybeTransform(self.transform, self.chance)(img, seg)
        return img, seg
        
    


class MinMaxNorm:
    def __init__(self, minim:float=0, maxim:float=1):
        self.minim = minim
        self.maxim = maxim

    def __call__(self, img, seg):
        i_min = self.minim
        i_max = self.maxim
        o_min = torch.min(img)
        o_max = torch.max(img)
        
        img = (o_max - o_min) * (img - i_min) / (i_max - i_min) + o_min
        return img, seg

               

class ContrastAugmentation:
    def __init__(self, gamma_range:list=(0.5, 2)):
        self.gamma_range = gamma_range if len(gamma_range)==2 \
            else Exception("Invalid gamma_range (must be (min max))")

        self.transform = cc.RandomGammaTransform(gamma=gamma_range)

    
    def __call__(self, img, seg):
        img = self.transform(img)
        return img, seg


    
class BiasField:
    def __init__(self, shape:int=8, v_max:list=1, order:int=3):
        self.shape = shape if isinstance(shape, int)\
            else Exception("Invalid shape (must be int)")
        self.v_max = v_max if isinstance(v_max, int)\
            else Exception("Invalid v_max (must be int)")
        self.order = order if isinstance(order, int)\
            else Exception("Invalid order (must be int)")
        
        self.transform = cc.RandomMulFieldTransform(shape=shape,
                                                    vmax=v_max,
                                                    order=order,
                                                    shared=False)

    def __call__(self, img, seg):
        img = self.transform(img)
        return img, seg

    

class GaussianNoise:
    def __init__(self, sigma:float=0.1):
        self.sigma = sigma
        self.transform = cc.RandomGaussianNoiseTransform(sigma=sigma)

    def __call__(self, img, seg):
        img = self.transform(img)
        return img, seg



class AssignOneHotLabels():
    def __init__(self, label_values:list=None, n_dims:int=3, index=0):
        self.label_values = label_values
        self.n_dims = n_dims
        self.index = index

        
    def __call__(self, img, seg):
        if self.label_values == None:
            self.label_values = torch.unique(torch.flatten(seg))

        onehot = torch.zeros(seg.shape)
        if self.n_dims == 4:
            onehot = onehot.repeat(len(self.label_values),1,1,1,1)
        elif self.n_dims == 3:
            onehot = onehot.repeat(len(self.label_values),1,1,1)
        elif self.n_dims == 2:
            onehot = onehot.repeat(len(self.label_values),1,1)
        seg = torch.squeeze(seg)
        
        for i in range(0, len(self.label_values)):
            onehot[i,:] = seg==self.label_values[i]
            
        return img, onehot.type(torch.float32)
