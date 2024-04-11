import os
import numpy as np
import torch
import yaml
import surfa as sf
from torch.utils.data import Dataset
from utils.preprocessing import apply_augmentations
from utils.data_utils import load_volume

class SegmentationDataset(Dataset):
    def __init__(self, dataset_list_file, transform=None):
        with open(dataset_list_file, 'r') as file:
            self.dataset_list = yaml.safe_load(file)
        self.transform = transform

        # Extract image and label file paths
        self.image_files = [item['image_filepath'] for item in self.dataset_list]
        self.label_files = [item['label_filepath'] for item in self.dataset_list] 

    def __len__(self):
        return len(self.dataset_list)


    def __getitem__(self, index):
        data_item = self.dataset_list[index]
        image_path = data_item['image_filepath']
        label_path = data_item['label_filepath']

        # Load image and label using the load_volume function
        image, image_tensor = load_volume(image_path)
        label, label_tensor = load_volume(label_path)
    
        # Apply data augmentation if transform is specified
        if self.transform:
            image_tensor, label_tensor = apply_augmentations(
                image_tensor, label_tensor, image, label, 
                voxsize=image.geom.voxsize, output_dir=None
            )

        return image_tensor, label_tensor
    
    def get_all_labels(self): 
        all_labels = []
        for i in range(len(self)):
            _, labels = self[i]
            all_labels.append(labels) 
        return torch.cat(all_labels, dim=0)