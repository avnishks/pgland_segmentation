import os
from glob import glob
from pathlib import Path

import logging
import pandas as pd
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose, Resize, ToTensor



class PituitaryPinealDataset(Dataset):
    
    def __init__(self, image_label_list=None, data_dir=None, transform=None, save_image_label_list=False):
        """
        Args:
            image_label_list (string): CSV file with paths of the images and labels.
            data_dir (string): Directory with the data. Should have subfolders for each modality and 'labels'.
        """
        self.transform = transform
        
        if image_label_list is not None:
            if not os.path.isfile(image_label_list):
                raise ValueError(f'File {image_label_list} does not exist')

            # Read the image and label paths from the file
            df = pd.read_csv(image_label_list, header=None)
            self.image_files = df.iloc[:, :-1].values.tolist()
            self.label_files = df.iloc[:, -1].values.tolist()

        elif data_dir is not None:
            data_dir = Path(data_dir)
            
            if not data_dir.exists():
                raise ValueError(f'Data directory {data_dir} does not exist')
                
            img_dir = data_dir / 'images'
            lbl_dir = data_dir / 'labels'
            
            if not img_dir.is_dir():
                raise ValueError(f'Expected image subfolder {img_dir}')
            if not lbl_dir.is_dir(): 
                raise ValueError(f'Expected label subfolder {lbl_dir}')

            img_files = sorted(glob(str(img_dir / '*')))
            lbl_files = sorted(glob(str(lbl_dir / '*')))

            # Check for mismatches
            if len(img_files) != len(lbl_files):
                logging.warning('Number of images and labels do not match!')
                
                # Take intersection by file name
                img_names = {Path(f).stem for f in img_files}
                lbl_names = {Path(f).stem for f in lbl_files}
                intersect = img_names & lbl_names
                
                img_files = [f for f in img_files if Path(f).stem in intersect]
                lbl_files = [f for f in lbl_files if Path(f).stem in intersect]
                
                logging.info(f'Taking intersection of {len(img_files)} image/label pairs')
            
            self.image_files = img_files
            self.label_files = lbl_files

        else:
            raise ValueError('Either data_dir or image_label_list must be provided')

        #log success
        num_pairs = len(self.image_files)
        logging.info(f'Data loaded successfully with {num_pairs} image/label pairs')

        if save_image_label_list:
            if not image_label_list and save_image_label_list:
                image_label_list = Path(data_dir) / 'image_pairs.csv'
            os.makedirs(Path(image_label_list).parent, exist_ok=True)
            
            if Path(image_label_list).suffix != '.csv':
                raise ValueError('File must be a CSV')
                
            df = pd.DataFrame({'image_path': self.image_files, 'label_path': self.label_files}) 
            df.to_csv(image_label_list, index=False)

        
    def __len__(self):
        return len(self.image_files)
    
    def _load_image(self, path):
        image = nib.load(path).get_fdata()
        image = (image - image.mean()) / image.std()
        image = image.astype(np.float16)
        return image

    def _load_label(self, path):
        label = nib.load(path).get_fdata()
        label = label.astype(np.int16)
        return label
    
    def __getitem__(self, idx):
        img_paths = self.image_files[idx]
        label_path = self.label_files[idx]
        
        images = [self._load_image(Path(img_path)) for img_path in img_paths]
        label = self._load_label(Path(label_path))

        label = self.transform(label)
        if self.transform is not None:
            images = [self.transform(image) for image in images]

        images = torch.stack(images)

        return images, label
