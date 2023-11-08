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

        return img, seg, param_file



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

        self.translation_bounds = [translation_bounds] * X \
            if isinstance(translation_bounds, float) else translation_bounds
        self.rotation_bounds = [rotation_bounds] * X \
            if isinstance(rotation_bounds, float) else rotation_bounds
        self.shear_bounds = [shear_bounds]  * X \
            if isinstance(shear_bounds, float) else shear_bounds
        self.scale_bounds = [scale_bounds]  * X \
            if isinstance(scale_bounds, float) else scale_bounds
        self.max_elastic_displacement = [max_elastic_displacement] * X \
            if isinstance(max_elastic_displacement, float) else max_elastic_displacement
        self.n_elastic_control_pts = n_elastic_control_pts
        self.n_elastic_steps = n_elastic_steps
        self.patch_size = patch_size

        """
        self.transform = cc.RandomAffineElasticTransform(translations=self.translations,
                                                       rotations=self.rotations,
                                                       shears=self.shears,
                                                       zooms=self.zooms,
                                                       dmax=self.dmax,
                                                       shape=self.shape,
                                                       steps=self.steps,
                                                       patch=self.patch
        )
        """
        
        
    def _randomize_transform(self):
        translations = npr.uniform(np.multiply(self.translation_bounds, -1), self.translation_bounds).tolist()
        rotations = npr.uniform(np.multiply(self.rotation_bounds, -1), self.rotation_bounds).tolist()
        shears = npr.uniform(np.multiply(self.scale_bounds, -1), self.scale_bounds).tolist()
        zooms = npr.uniform(np.multiply(self.shear_bounds, -1), self.shear_bounds).tolist()
        dmax = npr.uniform(0, self.max_elastic_displacement).tolist()
        shape = npr.randint(2, self.n_elastic_control_pts)
        steps = self.n_elastic_steps
        patch = self.patch_size

        transform = cc.AffineElasticTransform(translations=translations,
                                              rotations=rotations,
                                              shears=shears,
                                              zooms=zooms,
                                              dmax=dmax,
                                              shape=shape,
                                              steps=steps,
                                              patch=patch
        )

        if param_file is not None:
            with open(param_file, 'a') as f:
                print('translations:', translations, file=f)
                print('rotations:', rotations, file=f)
                print('shears:', shears, file=f)
                print('zooms:', zooms, file=f)
                print('elastic dmax:', dmax, file=f)
                print('elastic steps:', steps, file=f)
                print('patch size:', patch, file=f)
        """
        self.spatial = cc.RandomAffineElasticTransform(translations=self.translations,
                                                       rotations=self.rotations,
                                                       shears=self.shears,
                                                       zooms=self.zooms,
                                                       dmax=self.dmax,
                                                       shape=self.shape,
                                                       steps=self.steps,
                                                       patch=self.patch
        )
        """
        return transform

        
    def __call__(self, img, seg, param_file):
        transform = self._randomize_transform(param_file)
        img, seg = transform(img, seg)
        return img, seg, param_file



class RandomLRFlip:
    def __init__(self, chance:float=0.5):
        self.chance = chance if chance >= 0 and chance <= 1 \
            else Exception("Invalid chance (must be float between 0 and 1)")

        
    def _randomize_transform(self, param_file):
        flip = True if np.float32(npr.uniform()) < self.chance else False
        transform = cc.FlipTransform(axis=0) if flip else None

        if param_file is not None:
            with open(param_file, 'a') as f:
                print('lr_flip:', flip, file=f)

        return transform

    
    def __call__(self, img, seg, param_file):
        #img, seg = cc.MaybeTransform(self.transform, self.chance)(img, seg)
        transform = self._randomize_transform(param_file)
        if transform is not None:  img, seg = transform(img, seg)
        return img, seg, param_file
    


class MinMaxNorm:
    def __init__(self,
                 minim:float=0,
                 maxim:float=1,
                 **kwargs
    ):
        self.minim = minim
        self.maxim = maxim

    def __call__(self, img, seg, param_file):
        i_min = self.minim
        i_max = self.maxim
        o_min = torch.min(img)
        o_max = torch.max(img)
        
        img = (o_max - o_min) * (img - i_min) / (i_max - i_min) + o_min
        return img, seg, param_file

               

class ContrastAugmentation:
    def __init__(self,
                 gamma_range:list=(0.5, 2),
                 **kwargs
    ):
        self.gamma_range = gamma_range if len(gamma_range)==2 \
            else Exception("Invalid gamma_range (must be (min max))")

        #self.gammacorr = cc.RandomGammaTransform(gamma=gamma_range)
        

    def _randomize_transform(self, param_file):
        gamma = npr.uniform(self.gamma_range[0], self.gamma_range[1])
        transform = cc.GammaTransform(gamma=gamma)

        if param_file is not None:
            with open(param_file, 'a') as f:
                print('gamma:', gamma, file=f)

        return transform
        
    
    def __call__(self, img, seg, param_file):
        #img = self.gammacorr(img)
        transform = self._randomize_transform(param_file)
        img = transform(img)
        return img, seg, param_file


    
class BiasField:
    def __init__(self,
                 shape:int=8,
                 v_max:list=1,
                 order:int=3
    ):
        self.shape = shape if isinstance(shape, int)\
            else Exception("Invalid shape (must be int)")
        self.v_max = v_max
        self.order = order if isinstance(order, int)\
            else Exception("Invalid order (must be int)")

        """
        self.biasfield = cc.RandomMulFieldTransform(shape=shape,
                                                    vmax=v_max,
                                                    order=order,
                                                    shared=False)
        """


    def _randomize_transform(self, param_file):
        shape = npr.randint(2, self.shape)
        v_max = npr.uniform(0, self.v_max)
        order = self.order

        transform = cc.MulFieldTransform(shape=shape, vmax=v_max, order=order)

        if param_file is not None:
            with open(param_file, 'a') as f:
                print('shape:', shape, file=f)
                print('v_max:', v_max, file=f)

        return transform
    

    def __call__(self, img, seg, param_file):
        #img = self.biasfield(img)
        transform = self._randomize_transform(param_file)
        img = transform(img)
        return img, seg, param_file

    

class GaussianNoise:
    def __init__(self,
                 sigma:float=0.1,
    ):
        self.sigma = sigma
        self.noise = cc.RandomGaussianNoiseTransform(sigma=sigma)


    def _randomize_transform(self, param_file):
        sigma = npr.uniform(0, self.sigma)        
        transform = cc.RandomGaussianNoiseTransform(sigma=sigma)
        
        if param_file is not None:
            with open(param_file, 'a') as f:
                print('sigma:', sigma, file=f)
                
        return transform
            
            
    def __call__(self, img, seg, param_file):
        transform = self._randomize_transform(param_file)
        img = transform(img)
        #img = self.noise(img)
        return img, seg, param_file



class AssignOneHotLabels():
    def __init__(self,
                 label_values=None,
                 index=0,
    ):
        self.label_values = label_values
        self.index = index

    def __call__(self, img, seg, param_file):
        if self.label_values == None:
            self.label_values = torch.unique(torch.flatten(seg))

        onehot = torch.zeros(seg.shape)
        onehot = onehot.repeat(len(self.label_values),1,1,1)
        seg = torch.squeeze(seg)
        
        for i in range(0, len(self.label_values)):
            onehot[i,:] = seg==self.label_values[i]

        return img, onehot.type(torch.float32), param_file
