import nibabel as nib
from torch.utils.data import DataLoader
from utils.data_utils import BrainMRIDataset

csv_file = "data/image_paths.csv"

# create a dataloader
dataset = BrainMRIDataset(csv_file)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# get one batch of images and labels
images, labels = next(iter(dataloader))

# create a nibabel image from the tensor
image_nib = nib.Nifti1Image(images[0].numpy(), affine=np.eye(4))

# view the image
nib.viewers.OrthoSlicer3D(image_nib.get_data()).show()