from torch.utils.data import Dataset
import cv2
import os
import numpy as np
from PIL import Image  # Import PIL for transformations compatibility

def train_dataset(img_root, mask_root):
    imgs = []
    n = len(os.listdir(img_root))
    for i in range(n):
        img = os.path.join(img_root, str(i).zfill(6) + ".tif")
        mask = os.path.join(mask_root, str(i).zfill(6) + ".tif")
        imgs.append((img, mask))
    return imgs

def test_dataset(img_root):
    imgs = []
    n = len(os.listdir(img_root))
    for i in range(n):
        img = os.path.join(img_root, str(i).zfill(6) + ".tif")
        imgs.append(img)
    return imgs

class TrainDataset(Dataset):
    def __init__(self, img_root, mask_root, transform=None, mask_transform=None):
        """
        Dataset for training.

        Args:
            img_root: Path to the training images.
            mask_root: Path to the corresponding masks.
            transform: Transformations to apply to images.
            mask_transform: Transformations to apply to masks.
        """
        imgs = train_dataset(img_root, mask_root)
        self.imgs = imgs
        self.transform = transform
        self.mask_transform = mask_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = cv2.imread(x_path, -1)  # Read image as numpy array
        img_y = cv2.imread(y_path, -1)  # Read mask as numpy array

        # Preserve slicing logic for fixed input dimensions
        img_x, img_y = img_x[5:741, 1:769], img_y[5:741, 1:769]

        # Convert to PIL.Image for compatibility with transformations
        img_x = Image.fromarray(img_x)
        img_y = Image.fromarray(img_y)

        if self.transform:
            img_x = self.transform(img_x)
        if self.mask_transform:
            img_y = self.mask_transform(img_y)

        return img_x, img_y

    def __len__(self):
        return len(self.imgs)

class TestDataset(Dataset):
    def __init__(self, img_root, transform=None):
        """
        Dataset for testing.

        Args:
            img_root: Path to the test images.
            transform: Transformations to apply to images.
        """
        imgs = test_dataset(img_root)
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        x_path = self.imgs[index]
        img_x = cv2.imread(x_path, -1)  # Read image as numpy array

        # Preserve slicing logic for fixed input dimensions
        img_x = img_x[5:741, 1:769]

        # Convert to PIL.Image for compatibility with transformations
        img_x = Image.fromarray(img_x)

        if self.transform:
            img_x = self.transform(img_x)

        return img_x

    def __len__(self):
        return len(self.imgs)
