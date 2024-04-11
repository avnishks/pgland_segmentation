import os
import numpy as np
from pathlib import Path
import surfa as sf
import torch
from voxynth import voxynth
import matplotlib.pyplot as plt

def flip_volumes(image, label, axis=1):
    """Applies a random left-right flip to image and label volumes."""
    flipped_image, flipped_label = voxynth.transform.random_flip(axis, image, label, prob=1.0)
    return flipped_image, flipped_label


def bias_field_augmentation(image, voxsize):
    """Applies bias field augmentation to an image."""
    return voxynth.augment.image_augment(
        image,
        voxsize=voxsize,
        bias_field_probability=0.5,
        bias_field_max_magnitude=0.1,
        bias_field_smoothing_range=[1, 2],
    )

def onehot(labels, num_classes, device=None):
    """
    One-hot encode a tensor of integer labels.

    Args:
        labels (torch.Tensor): A tensor of integer labels.
        num_classes (int): The number of classes.
        device (torch.device, optional): The desired device (CPU or GPU). 
                                         If None, defaults to the device of the 'labels' tensor.

    Returns:
        torch.Tensor: A one-hot encoded tensor. 
    """
    if device is None:
        device = labels.device 
    onehot_labels = torch.eye(num_classes, device=device)[labels.long().squeeze(1)]
    return onehot_labels.permute(0, 4, 1, 2, 3)

def apply_augmentations(image_tensor, label_tensor, original_image, original_label, voxsize, output_dir):
    """
    Apply data augmentations to the image and label tensors and save intermediate results.
    
    Args:
        image_tensor (torch.Tensor): PyTorch tensor representing the image volume.
        label_tensor (torch.Tensor): PyTorch tensor representing the label volume.
        original_image (surfa.Volume): Original loaded image volume.
        original_label (surfa.Volume): Original loaded label volume.
        voxsize (tuple): Voxel size of the volumes.
        output_dir (str): Directory to save the intermediate results.
    
    Returns:
        tuple: Augmented image and label tensors.
    """
    # 1. Apply LR flipping to both image and label
    flipped_image, flipped_label = flip_volumes(image_tensor, label_tensor)

    
    # 2. Apply spatial transformation to both image and label
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = flipped_image.to(device)
    label = flipped_label.to(device)
    
    trf = voxynth.transform.random_transform(
        shape=image.shape[1:],
        device=device,
        affine_probability=1.0,
        max_translation=5.0,
        max_rotation=5.0,
        max_scaling=1.1,
        warp_probability=1.0,
        warp_integrations=5,
        warp_smoothing_range=[10, 20],
        warp_magnitude_range=[1, 2],
    )
    
    transformed_image = voxynth.transform.spatial_transform(image, trf)
    transformed_label = voxynth.transform.spatial_transform(label, trf, method='nearest')
    
    # 3. Apply center cropping to both image and label
    # cropped_image = voxynth.augment.apply_center_crop(transformed_image, (160, 160, 160))
    # cropped_label = voxynth.augment.apply_center_crop(transformed_label, (160, 160, 160))
    cropped_image = voxynth.augment.apply_center_crop(transformed_image, (80, 80, 80))
    cropped_label = voxynth.augment.apply_center_crop(transformed_label, (80, 80, 80))
    
    # # 4. Apply blurring and resampling to the image only
    cropped_image_cpu = cropped_image.cpu()  # Move the tensor to CPU
    blur_resampled_image = voxynth.augment.image_augment(
        cropped_image_cpu,
        normalize=True,
        smoothing_probability=0.5,
        smoothing_max_sigma=2.0,
        added_noise_probability=0.5,
        added_noise_max_sigma=0.05,
        gamma_scaling_probability=0.5,
        gamma_scaling_max=0.8,
        resized_probability=0,
        resized_one_axis_probability=0,
        resized_max_voxsize=2,
    )
    
    # # 5. Apply bias field augmentation to the image only
    bf_augmented_image = bias_field_augmentation(blur_resampled_image, voxsize=voxsize)
    
    return bf_augmented_image, cropped_label
