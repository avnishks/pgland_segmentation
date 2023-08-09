import sys
import numpy as np
import math
import random

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

import cornucopia as cc #https://github.com/balbasty/cornucopia/tree/feat-psf-slicewise



class RandomAffine():
    def __init__(self,
                 translation_bounds:[float,tuple],
                 rotation_bounds:[float,tuple],
                 shear_bounds:[float,tuple],
                 scale_bounds:[float,tuple]):
        self.n_dims = 3
        
        # Check input args
        self.rotation_bounds = rotation_bounds \
            if len(rotation_bounds) == self.n_dims or isinstance(rotation_bounds, float) \
            else Exception("Invalid rotation_bounds ([must be xy xz yz] or float)")
        self.scale_bounds = scale_bounds \
            if len(scale_bounds) == self.n_dims or isinstance(scale_bounds, float) \
            else Exception("Invalid scale_bounds ([must be x y z] or float)")
        self.shear_bounds = shear_bounds \
            if len(shear_bounds) == self.n_dims or isinstance(shear_bounds, float) \
            else Exception("Invalid shear_bounds (must be [x y z] or float)")
        self.translation_bounds = translation_bounds \
            if len(translation_bounds) == self.n_dims or isinstance(translation_bounds, float) \
            else Exception("Invalid translation_bounds (must be [x y z] or float)")
                

        self.affine_transform = cc.RandomAffineTransform(rotations=rotation_bounds, \
                                                         scales=scale_bounds, \
                                                         zooms=shear_bounds, \
                                                         translations=translation_bounds)
                 

    def __call__(self, seg, image_set):
        ## Note: image_set should be a list of all the images associated with the label map (seg)
        ##### to-do: put in a check that verifies this

        # Stack everything (store in cc_input) along batch dimension to input to cornucopia library
        input_stack = torch.as_tensor(seg)

        input_stack = torch.cat((input_stack, image_set[i], 0) for i in range(0,len(image_set))

        # Run transform
        output_stack = self.affine_transform(image_set)
        seg_affine = output_stack[0, :]
        image_set_affine = output_stack[1:, :]

        return seg_affine, image_set_affine
