import torch
import segmentation_models_pytorch as smp
import numpy as np
from utils import *
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch import nn
import shutil
from track import predict_dataset_2
from generate_trace import get_trace, get_video
import os
import cv2
from tifffile import imread as tif_imread  # Alternative TIFF reader
from tifffile import imwrite as tif_imwrite
from transforms import CLAHE, AddGaussianNoise, Compose, RandomApply, RandomOrder, ToTensor, Normalize, Resize, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, ColorJitter


def clahe(img):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance image contrast.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    img = clahe.apply(img)
    return img


def enhance(img):
    """
    Enhances the image by scaling pixel intensities.
    """
    img = np.clip(img * 1.2, 0, 255)
    img = img.astype(np.uint8)
    return img


def read_image_with_fallback(image_path):
    """
    Attempts to read the image using OpenCV. If it fails, uses tifffile as a fallback.
    """
    try:
        img = cv2.imread(image_path, -1)
        if img is None:
            raise ValueError(f"OpenCV failed to read {image_path}. Falling back to tifffile.")
        return img
    except Exception as e:
        print(f"Error reading {image_path} with OpenCV: {e}")
        try:
            img = tif_imread(image_path)
            return img
        except Exception as fallback_error:
            print(f"Fallback tifffile also failed for {image_path}: {fallback_error}")
            return None


def createFolder(path, clean_existing=False):
    """
    Creates a folder at the specified path. If the folder exists, it can optionally clean its contents.
    """
    try:
        if os.path.isdir(path):
            if clean_existing:
                shutil.rmtree(path)
                os.mkdir(path)
                print(f"{path} has been cleaned and recreated.")
            else:
                print(f"{path} already exists.")
        else:
            os.mkdir(path)
            print(f"{path} has been created.")
    except Exception as e:
        print(f"Error creating folder {path}: {e}")


def useAreaFilter(img, area_size):
    """
    Filters out small connected components based on area size.
    """
    if img is None:  # Ensure the input image is not None
        print("Warning: Received NoneType image in useAreaFilter. Skipping.")
        return None

    try:
        if img.dtype != np.uint8:
            img = cv2.convertScaleAbs(img)

        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_new = np.stack((img, img, img), axis=2)

        for cont in contours:
            area = cv2.contourArea(cont)
            if area < area_size:
                img_new = cv2.fillConvexPoly(img_new, cont, (0, 0, 0))

        img = img_new[:, :, 0]
        return img
    except Exception as e:
        print(f"Error applying area filter: {e}")
        return img


def test(test_path, result_path):
    """
    Runs model inference on the test dataset and saves the predicted masks.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    model = smp.Unet(encoder_name='resnet50', encoder_weights='imagenet', in_channels=1, classes=1)
    # Alternative model options
    # model = smp.FPN(encoder_name='resnet50', encoder_weights='imagenet', in_channels=1, classes=1)
    # model = smp.Linknet(encoder_name='resnet50', encoder_weights='imagenet', in_channels=1, classes=1)
    # model = smp.MAnet(encoder_name='resnet50', encoder_weights='imagenet', in_channels=1, classes=1)
    # model = smp.PAN(encoder_name='resnet50', encoder_weights='imagenet', in_channels=1, classes=1)
    # model = smp.PSPNet(encoder_name='resnet50', encoder_weights='imagenet', in_channels=1, classes=1)
    # model = smp.UnetPlusPlus(encoder_name='resnet50', encoder_weights='imagenet', in_channels=1, classes=1)
    # model = smp.UPerNet(encoder_name='resnet50', encoder_weights='imagenet', in_channels=1, classes=1)
    #model = smp.DeepLabV3(encoder_name='resnet50', encoder_weights='imagenet', in_channels=1, classes=1)
    #model = smp.DeepLabV3Plus(encoder_name='resnet50', encoder_weights='imagenet', in_channels=1, classes=1)
    model.eval()
    model = torch.nn.DataParallel(model).to(device)

    try:
        state_dict = torch.load('checkpoints/CP_epoch50.pth', map_location=device)
        if any(not k.startswith("module.") for k in state_dict.keys()):
            state_dict = {f"module.{k}": v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    test_data = TestDataset(test_path, transform=x_transforms)
    dataloader = DataLoader(test_data, batch_size=8, num_workers=2)

    print(f"Total samples: {len(test_data)}")
    processed_images = 0
    for index, x in enumerate(dataloader):
        try:
            x = x.to(device)  # Move input to the device
            y = model(x).detach().cpu()  # Detach from computation graph and move to CPU
            for i, img in enumerate(y):
                img_y = (torch.sigmoid(img).numpy() * 255).astype(np.uint8)
                output_path = os.path.join(result_path, f"predict_{processed_images:06d}.tif")
                tif_imwrite(output_path, img_y)  # Save with tifffile
                processed_images += 1
        except Exception as e:
            print(f"Error processing batch {index}: {e}")
    print(f"Total processed images: {processed_images}")


def process_predictResult(source_path, result_path):
    """
    Processes the predicted masks to generate labeled components.
    """
    if not os.path.isdir(result_path):
        print('Creating RES directory')
        os.mkdir(result_path)

    names = os.listdir(source_path)
    names = [name for name in names if '.tif' in name]
    names.sort()

    for name in names:
        try:
            predict_result = read_image_with_fallback(os.path.join(source_path, name))
            if predict_result is None:
                print(f"Error: Failed to read {name}. Skipping.")
                continue
            
            _, predict_result = cv2.threshold(predict_result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, markers = cv2.connectedComponents(predict_result)
            markers = np.uint16(markers)
            tif_imwrite(os.path.join(result_path, name), markers)
        except Exception as e:
            print(f"Error processing file {name}: {e}")


if __name__ == "__main__":
    test_folders = os.listdir("nuclear_dataset")
    test_folders = [os.path.join("nuclear_dataset/", folder) for folder in test_folders]
    test_folders.sort()
    
    for folder in test_folders:
        test_path = os.path.join(folder, "test")
        test_result_path = os.path.join(folder, "test_result")
        res_path = os.path.join(folder, "res")
        res_result_path = os.path.join(folder, "res_result")
        track_result_path = os.path.join(folder, "track_result")
        trace_path = os.path.join(folder, "trace")

        createFolder(test_result_path)
        createFolder(res_path)
        createFolder(res_result_path)
        createFolder(track_result_path)
        createFolder(trace_path)

        test(test_path, test_result_path)
        process_predictResult(test_result_path, res_path)

        try:
            result = os.listdir(res_path)
            for picture in result:
                image = read_image_with_fallback(os.path.join(res_path, picture))
                if image is None:
                    print(f"Error: Failed to load image {picture}. Skipping.")
                    continue
                image = useAreaFilter(image, 100)
                tif_imwrite(os.path.join(res_result_path, picture), image)
        except Exception as e:
            print(f"Error processing filtered image: {e}")
        
        print("Starting tracking")
        try:
            if not os.listdir(res_result_path):
                raise ValueError("No files found in res_result_path for tracking.")
            
            print(f"Files in res_result_path: {os.listdir(res_result_path)}")
            predict_dataset_2(res_result_path, track_result_path)

            track_files = os.listdir(track_result_path)
            print(f"Files in track_result_path: {track_files}")
            get_trace(test_path, track_result_path, trace_path)

            trace_files = os.listdir(trace_path)
            print(f"Files in trace_path: {trace_files}")
            get_video(trace_path)
        except Exception as e:
            print(f"Error during tracking or video generation: {e}")
