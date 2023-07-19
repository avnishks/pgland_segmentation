import torch
from torch.utils.data import Dataset, DataLoader
import surfa
import glob
import pathlib
import numpy as np

class BrainMRIDataset(Dataset):
    def __init__(self, data_dir, batch_size=32, shuffle=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Check if text file with image paths exists, otherwise create it
        pathlist_file = pathlib.Path(data_dir) / "imagelist.txt"
        if pathlist_file.exists():
            self.image_paths = [line.strip() for line in open(pathlist_file)] 
        else:
            self.image_paths = []
            images_dir = pathlib.Path(data_dir) / "images"
            for img_path in images_dir.glob("*.nii.gz"):
                self.image_paths.append(str(img_path))
            with open(pathlist_file, "w") as f:
                f.write("\n".join(self.image_paths))
                
        # Get corresponding label paths
        self.label_paths = [str(pathlib.Path(p).parent / ("labels_" + pathlib.Path(p).name)) for p in self.image_paths]
        
        #add typical preprocessing steps here
        # self.preprocessing = 
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        # Read image and label
        image = surfa.read_freesurfer_lut(img_path) 
        label = surfa.read_freesurfer_lut(label_path)
        
        # Apply preprocessing
        image = self.preprocessing(image) 
        label = self.preprocessing(label)
        
        return image, label
        
dataset = BrainMRIDataset("path/to/data")
dataloader = DataLoader(dataset, batch_size=dataset.batch_size, shuffle=dataset.shuffle)