import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import segmentation_models_pytorch as smp
import numpy as np
from torch import nn,optim
from PIL import Image
from utils import TrainDataset
from torchvision.transforms import transforms
from transforms import CLAHE, AddGaussianNoise, Compose, RandomApply, RandomOrder, ToTensor, Normalize, Resize, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, ColorJitter
import os
from utils import *
import logging
from tqdm import tqdm
import cv2
import os

# Custom CLAHE transform using OpenCV
class CLAHETransform:
    def __call__(self, img):
        img = np.array(img)  # Convert PIL Image to numpy array
        if len(img.shape) == 2:  # Grayscale
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)
        elif len(img.shape) == 3:  # Multi-channel (e.g., RGB)
            channels = cv2.split(img)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            channels = [clahe.apply(c) for c in channels]
            img = cv2.merge(channels)
        return Image.fromarray(img)  # Convert back to PIL Image

# Paths
model_path = 'checkpoints/'
imgs_path = 'data/imgs/'
mask_path = 'data/mask/'

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms for images and masks
x_transforms = transforms.Compose([
    transforms.RandomApply([transforms.RandomOrder([
        transforms.RandomApply([transforms.ColorJitter(brightness=0.33, contrast=0.33, saturation=0.33, hue=0)], p=0.5),
        transforms.RandomApply([transforms.GaussianBlur((5, 5), sigma=(0.1, 1.0))], p=0.5),
        transforms.RandomApply([transforms.RandomHorizontalFlip(0.5)], p=0.5),
        transforms.RandomApply([transforms.RandomVerticalFlip(0.5)], p=0.5),
        transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=2)], p=0.5),
        transforms.RandomApply([CLAHETransform()], p=0.5),
    ])], p=0.5),  # Apply any of the above augmentations with 50% probability
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize([0.5], [0.5])  # Normalize to mean=0.5 and std=0.5
])

y_transforms = transforms.Compose([
    transforms.ToTensor()  # Convert mask to PyTorch tensor
])

def __normalize(mask):
    min,max=np.unique(mask)[0],np.unique(mask)[-1]
    mask=mask/1.0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            mask[i][j]=(mask[i][j]-min)/(max-min)
    mask = mask.astype(np.float16)
    return mask


# Custom function to record results
def record_result(string):
    file_name = "train_record.txt"
    if not os.path.exists(file_name):
        with open(file_name, 'w') as f:
            print("Successfully created record file")
    with open(file_name, 'a') as f:
        f.write(string + "\n")
    print(string + " has been recorded")

# Updated train_model function
def train_model(model, criterion, optimizer, dataload, keep_training, num_epochs=50, accumulation_steps=4):
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    if keep_training:
        checkpoints = os.listdir(model_path)
        checkpoints.sort()
        final_ckpt = checkpoints[-1]
        print("Continue training from", final_ckpt)
        restart_epoch = int(final_ckpt.replace("CP_epoch", "").replace(".pth", ""))
        model.load_state_dict(torch.load(model_path + final_ckpt, map_location=device))
    else:
        restart_epoch = 1
        if os.path.isfile("train_record.txt"):
            os.remove("train_record.txt")
            print("Old result has been cleaned!")

    for epoch in range(restart_epoch - 1, num_epochs):
        model.train()
        print(f'Epoch {epoch + 1}/{num_epochs}')
        epoch_loss = 0
        step = 0
        optimizer.zero_grad()

        for i, (inputs, labels) in enumerate(tqdm(dataload)):
            step += 1
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()

        epoch_avg_loss = epoch_loss / step
        print(f"Epoch {epoch + 1} loss: {epoch_avg_loss:.3f}")
        record_result(f"Epoch {epoch + 1} loss: {epoch_avg_loss:.3f}")
        
        try:
            os.mkdir(model_path)
            logging.info('Created checkpoint directory')
        except OSError:
            pass
        
        torch.save(model.state_dict(), model_path + f'CP_epoch{epoch + 1:02d}.pth')
        logging.info(f'Checkpoint {epoch + 1} saved!')

if __name__ == "__main__":
    keep_training = False

    #model = smp.Unet(encoder_name='resnet50', encoder_weights='imagenet', in_channels=1, classes=1)
    # Alternative model options
    # model = smp.FPN(encoder_name='resnet50', encoder_weights='imagenet', in_channels=1, classes=1)
    # model = smp.Linknet(encoder_name='resnet50', encoder_weights='imagenet', in_channels=1, classes=1)
    # model = smp.MAnet(encoder_name='resnet50', encoder_weights='imagenet', in_channels=1, classes=1)
    # model = smp.PAN(encoder_name='resnet50', encoder_weights='imagenet', in_channels=1, classes=1)
    # model = smp.PSPNet(encoder_name='resnet50', encoder_weights='imagenet', in_channels=1, classes=1)
    # model = smp.UnetPlusPlus(encoder_name='resnet50', encoder_weights='imagenet', in_channels=1, classes=1)
    # model = smp.UPerNet(encoder_name='resnet50', encoder_weights='imagenet', in_channels=1, classes=1)
    model = smp.DeepLabV3(encoder_name='resnet50', encoder_weights='imagenet', in_channels=1, classes=1)
    #model = smp.DeepLabV3Plus(encoder_name='resnet50', encoder_weights='imagenet', in_channels=1, classes=1)

    batch_size = 8
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())

    data = TrainDataset(imgs_path, mask_path, x_transforms, y_transforms)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=2)

    train_model(model, criterion, optimizer, dataloader, keep_training, num_epochs=50)
