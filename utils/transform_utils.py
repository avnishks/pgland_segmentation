import sys
import numpy as np
import math
import random

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

import cornucopia as cc #https://github.com/balbasty/cornucopia/tree/feat-psf-slicewise



class RandomElasticAffineCrop():
    def __init__(self,
                 translation_bounds:[float,list]=0,
                 rotation_bounds:[float,list]=15,
                 shear_bounds:[float,list]=0.012,
                 scale_bounds:[float,list]=0.15,
                 max_elastic_displacement:[float,list]=0.15,
                 n_elastic_control_pts:int=5,
                 n_elastic_steps:int=0,
                 patch_size:int=None,
                 **kwargs):

        # Check input args
        self.n_dims = 3

        """
        self.translation_bounds = translation_bounds \
            if len(translation_bounds) == self.n_dims or isinstance(translation_bounds, float)\
            else Exception("Invalid translation_bounds (must be [x y z] or float)")
        self.rotation_bounds = rotation_bounds \
            if len(rotation_bounds) == self.n_dims or isinstance(rotation_bounds, float)\
            else Exception("Invalid rotation_bounds (must be [xy xz yz] or float)")
        self.shear_bounds = shear_bounds \
            if len(shear_bounds) == self.n_dims or isinstance(shear_bounds, float)\
            else Exception("Invalid shear_bounds (must be [xy xz yz] or float)")
        self.scale_bounds = scale_bounds \
            if len(scale_bounds) == self.n_dims or isinstance(scale_bounds, float)\
            else Exception("Invalid scale_bounds (must be [x y z] or float)")
        self.max_elastic_displacement = max_elastic_displacement \
            if isinstance(max_elastic_displacement, float)\
            else Exception("Invalid max_elastic_displacement (must be float)")
        self.n_elastic_control_pts = n_elastic_control_pts \
            if isinstance(n_elastic_control_pts, int)\
            else Exception("Invalid n_elastic_control_pts (must be int)")
        self.n_elastic_steps = n_elastic_steps \
            if isinstance(n_elastic_steps, int)\
            else Exception("Invalid n_elastic_steps (must be int)")
        self.patch_size = patch_size \
            if len(patch_size) == n_dims or isinstance(patch_size, int)\
            else Exception("Invalid patch_size (must be [x y z] or int)")
        """
        
        self.spatial = cc.RandomAffineElasticTransform(translations=translation_bounds,
                                                       rotations=rotation_bounds,
                                                       shears=shear_bounds,
                                                       zooms=scale_bounds,
                                                       dmax=max_elastic_displacement,
                                                       shape=n_elastic_control_pts,
                                                       steps=n_elastic_steps,
                                                       patch=patch_size)
        
    def __call__(self, img, seg):
        img, seg = self.elastic_affine_transform(img, seg)
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
    def __init__(self, minim:float=0, maxim:float=0, **kwargs):
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
        #self.v_range = v_range if len(v_range)==2 \
        #    else Exception("Invalid v_range (must be (min max))")
        
        self.gammacorr = cc.RandomGammaTransform(gamma=gamma_range)
                                                 #vmin=v_range[0],
                                                 #vmax=v_range[1])

    def __call__(self, x):
        img = self.gammacorr(img)


        
class BiasField:
    def __init__(self, shape:int=8, v_range:list=(-1, 1), order:int=3, **kwargs):
        self.shape = shape if isinstance(shape, int)\
            else Exception("Invalid shape (must be int)")
        self.v_range = v_range if len(v_range)==2 \
            else Exception("Invalid v_range (must be (min max))")
        self.order = order if isinstance(order, int)\
            else Exception("Invalid order (must be int)")
        
        self.addbias = cc.RandomAddFieldTransform(shape=shape,
                                                  vmin=v_range[0],
                                                  vmax=v_range[1],
                                                  order=order,
                                                  shared=False,
                                                  shared_field=None)

    def __call__(self, img):
        img = self.addbias(img)
        return img

    

class GaussianNoise:
    def __init__(self, sigma:float=0.1):
        self.sigma = sigma
        self.noise = cc.RandomGaussianNoiseTransform(sigma=sigma)

    def __call__(self, img):
        img = self.noise(img)
        return img
