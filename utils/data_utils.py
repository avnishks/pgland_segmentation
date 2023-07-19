from data_preprocessing import create_image_paths_file
import torch
from torch.utils.data import Dataset
import nibabel as nib
import pandas as pd
from pathlib import Path

class BrainMRIDataset(Dataset):
    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with image paths.
        """
        self.image_frame = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.image_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.image_frame.iloc[idx, 0]
        label_name = self.image_frame.iloc[idx, 1]

        # Load images and labels
        if Path(img_name).suffix in ['.nii', '.nii.gz']:
            image = nib.load(img_name).get_fdata()
            label = nib.load(label_name).get_fdata()
        elif Path(img_name).suffix == '.mgz':
            print("not implemented yet")

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        return image, label