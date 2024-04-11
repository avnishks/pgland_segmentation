import os
import pytest
from utils.dataset import SegmentationDataset
from utils.data_utils import load_volume, save_volume

num_samples = 1
output_dir = "tests/augmented_samples"

@pytest.mark.parametrize('index', range(num_samples)) 
def test_augmentations(index):
    dataset = SegmentationDataset('configs/dataset_list.yaml', transform=True)
    image, label = dataset[index]

    os.makedirs(output_dir, exist_ok=True)

    # Save augmented image and label 
    image_path = os.path.join(output_dir, f'image_sample_{index}.mgz')
    label_path = os.path.join(output_dir, f'label_sample_{index}.mgz')

    original_image, _ = load_volume(dataset.image_files[index])
    original_label, _ = load_volume(dataset.label_files[index])

    save_volume(image, original_image, image_path)
    save_volume(label, original_label, label_path)
