import os
from glob import glob
from pathlib import Path
import random

import logging
import pandas as pd
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import freesurfer as fs

from . import transform_utils as t_utils



def call_freeview(img, seg):
    fv = fs.Freeview()
    fv.vol(img[0,:])
    fv.vol(seg[0,:], colormap='lut')
    fv.show()


class PituitaryPinealDataset(Dataset):
    def __init__(self,
                 data_inds:list,
                 image_label_list=None,
                 data_dir=None,
                 transform=None,
                 augmentation=None,
                 data_labels:list=None,
                 save_image_label_list=False,
                 make_RAS=True,
                 **kwargs):
        """
        Args:
            image_label_list (string): CSV file with paths of the images and labels.
            data_dir (string): Directory with the data. Should have subfolders for each modality and 'labels'.
        """
        self.data_inds = data_inds
        self.make_RAS = make_RAS
        
        self.transform = transform
        self.augmentation = augmentation

        if data_labels is not None:
            self.OneHot = t_utils.AssignOneHotLabels(label_values=data_labels)
        else:
            self.OneHot = None
            
        if image_label_list is not None:
            if not os.path.isfile(image_label_list):
                raise ValueError(f'File {image_label_list} does not exist')

            # Read the image and label paths from the file
            df = pd.read_csv(image_label_list, header=None)
            self.image_files = df.iloc[data_inds, :-1].values.tolist()
            self.label_files = df.iloc[data_inds, -1].values.tolist()

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
        #logging.info(f'Data loaded successfully with {num_pairs} image/label pairs')

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
        data = nib.funcs.as_closest_canonical(nib.load(path)) if self.make_RAS else nib.load(path)
        image = data.get_fdata().astype(np.float32)
        return image


    def _load_label(self, path):
        data = nib.funcs.as_closest_canonical(nib.load(path)) if self.make_RAS else nib.load(path)
        image = data.get_fdata().astype(np.float32)
        return image

    
    def __getitem__(self, idx):
        img_paths = self.image_files[idx]
        label_path = self.label_files[idx]
        
        images = [self._load_image(Path(img_path)) for img_path in img_paths]
        label = self._load_label(Path(label_path))

        # Initial transform
        if self.transform is not None:
            label = self.transform(label)
            images = [self.transform(image) for image in images]

        images = torch.stack(images, dim=0)
        label = torch.from_numpy(np.expand_dims(label, axis=0))
        
        # Data augmentation
        if self.augmentation is not None:
            images, label = self.augmentation(images, label)

        if self.OneHot is not None:
            label = self.OneHot(label)
        
        return images, label



def call_dataset(data_config:str):
    # General set up
    data_labels = (0, 883, 900, 903, 904)
    transform = transforms.ToTensor()
    augmentation = t_utils.Compose([t_utils.GetPatch(patch_size=80),
                                    t_utils.RandomElasticAffineCrop(),
                                    t_utils.RandomLRFlip(),
                                    t_utils.ContrastAugmentation(),
                                    t_utils.BiasField(),
                                    t_utils.GaussianNoise(),
                                    t_utils.MinMaxNorm()
    ])
    augmentation = t_utils.GetPatch(patch_size=80)
    
    
    # Get subject indices for train/valid/test
    with open(data_config, 'r') as f:
        lines = f.readlines()

    n_subjects = len(lines)
    x = int(0.2*n_subjects)
    
    all_inds = list(range(0,n_subjects))
    random.shuffle(all_inds)
    test_inds = all_inds[:x]
    valid_inds = all_inds[x:2*x]
    train_inds = all_inds[2*x:]
    

    # Get each cohort
    train = PituitaryPinealDataset(data_inds=train_inds,
                                   image_label_list=data_config,
                                   transform=transform,
                                   augmentation=augmentation,
                                   data_labels=data_labels,
    )
    valid = PituitaryPinealDataset(data_inds=valid_inds,
                                   image_label_list=data_config,
                                   transform=transform,
                                   augmentation=augmentation,
                                   data_labels=data_labels,
    )
    test = PituitaryPinealDataset(data_inds=test_inds,
                                   image_label_list=data_config,
                                   transform=transform,
                                   augmentation=augmentation,
                                   data_labels=data_labels,
    )
    return train, valid, test, len(data_labels)
