import os
import random
import cv2
from PIL import Image, ImageOps
from glob import glob
import numpy as np
from transforms import CLAHE, ColorJitter

def contrast_augmentation(image, contrast_value):
    """
    Adjust the contrast of the image by multiplying with the contrast value.
    The value is clipped to ensure pixel values stay in the valid range [0, 255].
    """
    image = image.astype(np.float32)  # Convert to float to avoid overflow
    image_contrasted = np.clip(image * contrast_value, 0, 255)  # Clip to stay in the valid range
    return image_contrasted.astype(np.uint8)  # Convert back to uint8 for saving

def augment_image(image, mask):
    """
    Apply a series of augmentations to the image and its corresponding mask.
    This includes flipping, color jittering, and CLAHE.
    """
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = CLAHE()
    image, mask = clahe(image, mask)

    # Horizontal Flip (50% chance)
    if random.random() < 0.5:
        image = ImageOps.mirror(image)
        mask = ImageOps.mirror(mask)

    # Vertical Flip (50% chance)
    if random.random() < 0.5:
        image = ImageOps.flip(image)
        mask = ImageOps.flip(mask)

    # Color Jitter (adjust brightness, contrast, saturation, hue)
    color_jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)  # Adjusted for subtler effects
    image, mask = color_jitter(image, mask)

    return image, mask

def augment_images(imgs_dir, mask_dir, sequence, start_num):
    """
    Augment an image and its corresponding mask by applying a series of transformations.
    The image and mask will be augmented 13 times, and each augmented pair will be saved.
    """
    image_file = glob(imgs_dir + sequence + '.*')[0]  # Get image path
    mask_file = glob(mask_dir + sequence.replace('t', 'man_seg') + '.*')[0]  # Get mask path

    # Load the image and mask
    image = Image.open(image_file)
    mask = Image.open(mask_file)

    # Apply augmentation pipeline 13 times to generate 13 augmented images
    for i in range(13):
        augmented_image, augmented_mask = augment_image(image, mask)  # Pass both image and mask

        # Sequential file name generation
        image_num_str = str(start_num + i).zfill(6)
        image_file_aug = imgs_dir + image_num_str + '.tif'
        mask_file_aug = mask_dir + image_num_str + '.tif'

        # Save augmented images (in the same format)
        augmented_image.save(image_file_aug)
        augmented_mask.save(mask_file_aug)  # Mask remains the same

    return start_num + 13  # Update the starting number for the next set of images

if __name__ == "__main__":
    imgs_dir = 'data/imgs/'  # Path to your images directory
    mask_dir = 'data/mask/'  # Path to your mask directory

    # Get list of image IDs without file extensions
    ids = [os.path.splitext(file)[0] for file in os.listdir(imgs_dir) if not file.startswith('.')]
    print(f"Found image sequences: {ids}")

    start_num = len(ids)  # Initialize the starting number for sequential file naming

    # Sequentially process each image for augmentation
    for sequence in ids:
        start_num = augment_images(imgs_dir, mask_dir, sequence, start_num)

    print("Image augmentation has finished!")
