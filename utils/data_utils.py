import os
import numpy as np
import surfa as sf
import torch
import yaml
import json

def load_volume(file_path):
    """
    Load a volume from a file and convert it to a PyTorch tensor.
    
    Args:
        file_path (str): Path to the volume file.
    
    Returns:
        tuple: A tuple containing the loaded volume and its PyTorch tensor representation.
    """
    volume = sf.load_volume(file_path)
    volume_data_native = volume.framed_data.astype(volume.dtype.newbyteorder('='))
    volume_data_writable = np.copy(volume_data_native)  # Create a writable copy of the array
    volume_tensor = torch.from_numpy(volume_data_writable).movedim(-1, 0)
    return volume, volume_tensor

def save_volume(volume_tensor, original_volume, output_file):
    """
    Save the augmented volume to a file.
    
    Args:
        volume_tensor (torch.Tensor): Augmented volume tensor.
        original_volume (surfa.Volume): Original loaded volume.
        output_file (str): Path to the output file.
    """
    tensor_cpu = volume_tensor.cpu().squeeze(0)
    np_vol = tensor_cpu.detach().numpy().astype(original_volume.dtype)
    surfa_vol = original_volume.new(np_vol)
    surfa_vol.save(output_file)

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_label_mapping(labels, output_folder, filename="label_mapping.json"):
    """Create a label mapping and save it to a JSON file."""
    unique_labels = torch.unique(labels)
    mapping = {label.item(): i for i, label in enumerate(unique_labels)}

    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filepath = os.path.join(output_folder, filename)
    with open(filepath, 'w') as f:
        json.dump(mapping, f, indent=4) 

    return mapping

def remap_labels(labels, mapping):
    remapped_labels = torch.zeros_like(labels)
    for old_label, new_label in mapping.items():
        remapped_labels[labels == old_label] = new_label
    return remapped_labels