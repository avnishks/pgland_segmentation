import sys
import numpy as np
import math
import random

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

import cornucopia as cc #https://github.com/balbasty/cornucopia/tree/feat-psf-slicewise



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
                #breakpoint()
                args = t(*args)

        return args




class RandomElasticAffineCrop:
    def __init__(self,
                 translation_bounds:[float,list]=0.0,
                 rotation_bounds:[float,list]=15,
                 shear_bounds:[float,list]=0.012,
                 scale_bounds:[float,list]=0.15,
                 max_elastic_displacement:[float,list]=0.15,
                 n_elastic_control_pts:int=5,
                 n_elastic_steps:int=0,
                 patch_size:[int,list]=None,
                 **kwargs):

        n_dims = 3
        if isinstance(translation_bounds, list): assert len(translation_bounds) == n_dims
        if isinstance(rotation_bounds, list): assert len(rotation_bounds) == n_dims
        if isinstance(shear_bounds, list): assert len(shear_bounds) == n_dims
        if isinstance(scale_bounds, list): assert len(scale_bounds) == n_dims
        if isinstance(patch_size, list): assert len() == n_dims

        self.spatial = cc.RandomAffineElasticTransform(translations=translation_bounds,
                                                       rotations=rotation_bounds,
                                                       shears=shear_bounds,
                                                       zooms=scale_bounds,
                                                       dmax=max_elastic_displacement,
                                                       shape=n_elastic_control_pts,
                                                       steps=n_elastic_steps,
                                                       patch=patch_size)
        
    def __call__(self, img, seg):
        img, seg = self.spatial(img, seg)
        return img, seg



class RandomLRFlip:
    def __init__(self, chance:float=0.5):
        self.chance = chance if chance >= 0 and chance <= 1 \
            else Exception("Invalid chance (must be float between 0 and 1)")

        self.flip = cc.FlipTransform(axis=0)

    def __call__(self, img, seg):
        img, seg = cc.MaybeTransform(self.flip, self.chance)(img, seg)
        return img, seg
    


class MinMaxNorm:
    def __init__(self, minim:float=0, maxim:float=1, **kwargs):
        self.minim = minim
        self.maxim = maxim

    def __call__(self, img):
        i_min = self.minim
        i_max = self.maxim
        o_min = torch.min(img)
        o_max = torch.max(img)
        
        img = (o_max - o_min) * (img - i_min) / (i_max - i_min) + o_min
        return img

               

class ContrastAugmentation:
    def __init__(self, gamma_range:list=(0.5, 2), v_range:list=(None, None), **kwargs):
        self.gamma_range = gamma_range if len(gamma_range)==2 \
            else Exception("Invalid gamma_range (must be (min max))")
        self.v_range = v_range if len(v_range)==2 \
            else Exception("Invalid v_range (must be (min max))")

        self.gammacorr = cc.RandomGammaTransform(gamma=gamma_range)
                                                 #vmin=v_range[0],
                                                 #vmax=v_range[1])

    def __call__(self, img):
        img = self.gammacorr(img)
        return img

        
class BiasField:
    def __init__(self, shape:int=8, v_max:list=1, order:int=3):
        self.shape = shape if isinstance(shape, int)\
            else Exception("Invalid shape (must be int)")
        self.v_max = v_max if isinstance(v_max, int)\
            else Exception("Invalid v_max (must be int)")
        self.order = order if isinstance(order, int)\
            else Exception("Invalid order (must be int)")
        
        self.biasfield = cc.RandomMulFieldTransform(shape=shape,
                                                    vmax=v_max,
                                                    order=order,
                                                    shared=False)

    def __call__(self, img):
        img = self.biasfield(img)
        return img

    

class GaussianNoise:
    def __init__(self, sigma:float=0.1):
        self.sigma = sigma
        self.noise = cc.RandomGaussianNoiseTransform(sigma=sigma)

    def __call__(self, img):
        img = self.noise(img)
        return img




class AssignOneHotLabels():
    def __init__(self, label_values=None, index=0):
        self.label_values = label_values
        self.index = index

    def __call__(self, seg):
        if self.label_values == None:
            self.label_values = torch.unique(torch.flatten(seg))

        onehot = torch.zeros(seg.shape)
        onehot = onehot.repeat(len(self.label_values),1,1,1)
        seg = torch.squeeze(seg)
        
        for i in range(0, len(self.label_values)):
            onehot[i,:] = seg==self.label_values[i]

        return onehot.type(torch.float32)
