from re import X
from turtle import distance
import xml.etree.ElementTree as ET
from collections import defaultdict

from loguru import logger

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import cv2
from torchvision.transforms import functional as F
import os

import json
import onnxruntime as ort

import albumentations as A
from albumentations.pytorch import ToTensorV2
from symbol import factor
import matplotlib.pyplot as plt
import random
import shutil

import argparse

from torch._inductor.ir import NoneAsConstantBuffer
from loguru import logger

def make_parser():
    parser = argparse.ArgumentParser('DAMO-YOLO Demo')

    parser.add_argument('-a',
                        '--annotations',
                        default=None,
                        type=str,
                        help='Path to annotations.xml',)
    parser.add_argument('-i',
                        '--image_dir',
                        default=None,
                        type=str,
                        help='Path to image directory')
    parser.add_argument('-ti',
                        '--training_images',
                        default=None,
                        type=str,
                        help='Path to directory to place eye training images',)
    parser.add_argument('-va',
                        '--validation_images',
                        default=None,
                        type=str,
                        help='Path to directory with validation images')
    return parser


def ParseCVATXMLFile(filename, images_dir, output_dir) :
    os.makedirs(output_dir, exist_ok=True)

    tree = ET.parse(filename)
    root = tree.getroot()

    image_id_map = {}  # id -> filename
    for image in root.findall(".//image"):
        image_id_map[image.attrib["id"]] = {
            "name": image.attrib["name"],
            "width": int(image.attrib["width"]),
            "height": int(image.attrib["height"])
        }

    for image in root.findall(".//image"):
        img_name = image.attrib["name"]
        width = int(image.attrib["width"])
        height = int(image.attrib["height"])
        img_path = os.path.join(images_dir, img_name)

        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        img = cv2.imread(img_path)

        for idx, box in enumerate(image.findall(".//box")):
            label = box.attrib["label"]
            xtl = float(box.attrib["xtl"])
            ytl = float(box.attrib["ytl"])
            xbr = float(box.attrib["xbr"])
            ybr = float(box.attrib["ybr"])

            factor = 0.2
            box_height = ybr - ytl
            box_width = xbr - xtl

            xtl = xtl - box_width * factor
            ytl = ytl - box_height * factor
            xbr = xbr + box_width * factor
            ybr = ybr + box_height * factor

            if xtl < 0 :
                xtl = 0
            if xbr > width :
                xbr = width
            if ytl < 0 :
                ytl = 0
            if ybr > height : 
                ybr = height

            # Crop the image using OpenCV
            cropped = img[int(ytl):int(ybr), int(xtl):int(xbr)]
            crop_filename = f"{os.path.splitext(img_name)[0]}_crop{idx}.png"
            crop_path = os.path.join(output_dir, crop_filename)
            

            # Get points inside this bounding box
            crop_points = []
            for point in image.findall(".//points"):
                px_str = point.attrib["points"]
                label_pt = point.attrib.get("label", "")
                for px_pair in px_str.split(';'):
                    px, py = map(float, px_pair.split(','))
                    if xtl <= px <= xbr and ytl <= py <= ybr:
                        # Normalize point coordinates relative to crop
                        norm_x = (px - xtl) / (xbr - xtl)
                        norm_y = (py - ytl) / (ybr - ytl)
                        crop_points.append({
                            "label": label_pt,
                            "x": norm_x,
                            "y": norm_y
                        })

            if len(crop_points) == 2 :
                json_path = os.path.join(output_dir, f"{os.path.splitext(crop_filename)[0]}.json")
                with open(json_path, 'w') as jf:
                    json.dump({
                        "source_image": img_name,
                        "bounding_box": [xtl, ytl, xbr, ybr],
                        "points": crop_points
                    }, jf, indent=2)            
                

                cv2.imwrite(crop_path, cropped)
            # else : 
            #     logger.debug("Did not find 2 points within the bounding box! Not writing out crop or json")



            #print(f"Saved: {crop_path} with {len(crop_points)} points")


# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, image_paths, targets, doTransform=False):
        self.image_paths = image_paths
        self.targets = targets  # Shape: (N, 4) for 2 points (x1, y1, x2, y2)
        self.doTransform = doTransform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.resize(image, (224, 224))
        target = self.targets[idx]
        keypoints = [(target[0], target[1]), (target[2], target[3])]
       # logger.debug(keypoints)

        if self.doTransform:
                # Define Albumentations transforms
            transform = A.Compose([
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            ], keypoint_params=A.KeypointParams(format='xy'))

            keypoints_pixels = [(int(target[0] * 224), int(target[1] * 224)), (int(target[2] * 224), int(target[3] * 224))]

            augmented = transform(image=image, keypoints=keypoints_pixels)
            image = augmented['image']
            writable = image.copy()
            
            keypoints = augmented['keypoints']
            
            
            transform3 = transforms.Compose([transforms.ToTensor()])
            transform4= transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            image = transform3(image)
            image = transform4(image)
            if len(keypoints) != 2:
                # Return a dummy tensor with zeros if keypoints are lost
                logger.debug("lost keypoints!")
                target = torch.zeros(4, dtype=torch.float32)
            else:
                lx = int(keypoints[1][0])
                ly = int(keypoints[1][1])
                rx = int(keypoints[0][0])
                ry = int(keypoints[0][1])
            
                # cv2.circle(writable, (lx, ly), 1, (0, 0, 255))
                # cv2.circle(writable, (rx, ry), 1, (255,0,0))
                # cv2.imwrite("albumented.png", writable)

                target = torch.tensor((rx / 224.0, ry / 224.0, lx / 224.0, ly  / 224.0))

        else:
            transform3 = transforms.Compose([transforms.ToTensor()])
            transform4= transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            image = transform3(image)
            image = transform4(image)
            target = torch.tensor(target, dtype=torch.float32)

        return image, target

# Define the model
class ResNet18Regression(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18Regression, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 4)  # Output 2 values
    
    def forward(self, x):
        return self.resnet(x)

def list_png_files_in_directory(directory_path):
    files = []
    # List all files in the directory as strings
    for f in os.listdir(directory_path) :
        file = os.path.join(directory_path, f)
        if os.path.isfile(file) and file.endswith(".png") :
            files.append(file)

    return files

def LoadPointsForImages(image_paths) :
    all_points = []  # List to store all points

    for imagename in image_paths :
        jsonfile = os.path.splitext(imagename)[0] + '.json'
        # Load JSON data
        with open(jsonfile, 'r') as jf:
            data = json.load(jf)

        points = data.get("points", [])
        # Extract x, y coordinates
        # should only be 2 points per json file

        if points[0]["label"] == "Right eye" :
            righteyex = points[0]["x"]
            righteyey = points[0]["y"]
            lefteyex = points[1]["x"]
            lefteyey = points[1]["y"]
        else :
            lefteyex = points[0]["x"]
            lefteyey = points[0]["y"]
            righteyex = points[1]["x"]
            righteyey = points[1]["y"]
            
        # Store them as a tuple (x, y)
        all_points.append((righteyex, righteyey, lefteyex, lefteyey))
    return all_points


def SplitFolderContentsIntoTwoFolders(inputFolder,valFolder, trainFolder ) :
    pngs = list_png_files_in_directory(inputFolder)
    
    os.makedirs(trainFolder, exist_ok=True)
    os.makedirs(valFolder, exist_ok=True)
    
    random.shuffle(pngs)
    split_index = int(len(pngs) // 5) # split 80:20
    val_files = pngs[:split_index]
    train_files = pngs[split_index:]
    
    # Move files to new folders
    for f in val_files:
        path, filename = os.path.split(f)
        pngfilename_start = os.path.join(inputFolder, filename)
        pngfilename_end = os.path.join(valFolder, filename)
        
        shutil.copy(pngfilename_start,  pngfilename_end)
        
        jsonfilename, ext = os.path.splitext(filename)
        jsonfilename = jsonfilename + ".json"
        jsonfilename_start = os.path.join(inputFolder, jsonfilename)
        jsonfilename_end = os.path.join(valFolder, jsonfilename)
        shutil.copy(jsonfilename_start,  jsonfilename_end)

    for f in train_files:
        path, filename = os.path.split(f)
        pngfilename_start = os.path.join(inputFolder, filename)
        pngfilename_end = os.path.join(trainFolder, filename)
        
        shutil.copy(pngfilename_start,  pngfilename_end)
        
        jsonfilename, ext = os.path.splitext(filename)
        jsonfilename = jsonfilename + ".json"
        jsonfilename_start = os.path.join(inputFolder, jsonfilename)
        jsonfilename_end = os.path.join(trainFolder, jsonfilename)
        shutil.copy(jsonfilename_start,  jsonfilename_end)

    print(f"Moved {len(val_files)} files to {valFolder}")
    print(f"Moved {len(train_files)} files to {trainFolder}")

def trainModel(training_dir, validation_dir) :
    train_pngs = list_png_files_in_directory(training_dir)
    train_points = LoadPointsForImages(train_pngs)
    
    # Convert list of points to a NumPy array
    train_points_array = np.array(train_points, dtype=np.float32)

    # Dummy data
    train_targets =train_points_array

    val_pngs = list_png_files_in_directory(validation_dir)
    val_points = LoadPointsForImages(val_pngs)
    
    # Convert list of points to a NumPy array
    val_points_array = np.array(val_points, dtype=np.float32)

    # Dummy data
    val_targets =val_points_array
    
    # train_dataset = CustomDataset(train_dataset, train_targets)
    # val_dataset = CustomDataset(val_dataset, val_targets)
    train_dataset = CustomDataset(train_pngs, train_targets, doTransform=True)
    val_dataset = CustomDataset(val_pngs, val_targets, doTransform=False)

    # Hyperparameters
    batch_size = 8
    epochs = 50
    learning_rate = 0.0002

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18Regression().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    training_losses = []
    validation_losses = []
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, targets in train_dataloader:
            images, targets = images.to(device), targets.to(device)
        
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        avg_training_loss = running_loss/len(train_dataloader)
        training_losses.append(avg_training_loss )
        print(f"Training Epoch [{epoch}/{epochs}], Loss: {avg_training_loss :.8f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_dataloader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        avg_validation_loss = val_loss/len(val_dataloader)
        validation_losses.append(avg_validation_loss)
        print(f"Validation Epoch [{epoch}/{epochs}], Loss: {avg_validation_loss:.8f}")

        # Export model to ONNX
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        model_name = "epxEyeRegression_epoch" + str(epoch)+ ".onnx"
        torch.onnx.export(model, dummy_input, model_name, 
                          input_names=["input"], output_names=["output"], 
                          opset_version=11)
        
        # distance= BenchmarkModel(model_name)
        # logger.debug("distance = {}", distance)

    # Create an array for the x-axis (optional if you want custom x-values)
    x = np.arange(len(training_losses))

    # Plot the array
    plt.plot(x, training_losses, marker='o', linestyle='-', color='b', label='Training')
    plt.plot(x, validation_losses, marker='o', linestyle='-', color='r', label='validation')
    plt.title('Simple Array Plot')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()


def Preprocess(image) :
    image = cv2.resize(image, (224, 224))
    transform3 = transforms.Compose([transforms.ToTensor()])
    transform4= transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    image = transform3(image)
    image = transform4(image)
    return image

def testModelOnImage(model_path, image_path) :
    image = cv2.imread(image_path)
    input_tensor = Preprocess(image)
    input_tensor = input_tensor.reshape(1, 3, 224, 224)
        # Run inference
    ort_session = ort.InferenceSession(model_path)
    input_np = input_tensor.cpu().numpy()

    onnx_output = ort_session.run(None, {"input": input_np})
    rx_norm = onnx_output[0][0][0]
    ry_norm = onnx_output[0][0][1]
    lx_norm = onnx_output[0][0][2]
    ly_norm = onnx_output[0][0][3]
    
    lx = int(lx_norm * image.shape[1])
    ly = int(ly_norm * image.shape[0])
    rx = int(rx_norm * image.shape[1])
    ry = int(ry_norm * image.shape[0])

    cv2.circle(image, (lx, ly), 1, (0, 255, 0))
    cv2.circle(image, (rx, ry), 1, (0, 255, 0))

    #write out labeled eye positiosn
    [justfilename, ext] = os.path.splitext(image_path)
    jsonfilename = justfilename + ".json"

    with open(jsonfilename, "r") as file:
        jsonstuff = json.load(file)
    points = jsonstuff['points']
    for point in points :
        if point['label'] == "Left eye" :
            lx2_norm = point['x']
            ly2_norm = point['y']
        elif point['label'] == "Right eye" :
            rx2_norm = point['x']
            ry2_norm = point['y']
            
    lx2 = int(lx2_norm * image.shape[1])
    ly2 = int(ly2_norm * image.shape[0])
    rx2 = int(rx2_norm * image.shape[1])
    ry2 = int(ry2_norm * image.shape[0])
    cv2.circle(image, (lx2, ly2), 1, (0, 0, 255))
    cv2.circle(image, (rx2, ry2), 1, (0, 0, 255))
    
    distance_l = cv2.norm(np.array([ly2_norm - ly_norm, lx2_norm - lx_norm]))
    distance_r = cv2.norm(np.array([ry2_norm - ry_norm, rx2_norm - rx_norm]))

    total_dist = distance_l + distance_r

    output_filename = os.path.basename(image_path)
    output_filename = "benchmarking/" + output_filename 
    cv2.imwrite(output_filename, image)
    
    return total_dist

def BenchmarkModel(model_path) :
    pngfiles = list_png_files_in_directory("bag_eyevalidation_04292025")
    total_distance = 0.0
    for png in pngfiles:
        distance = testModelOnImage(model_path, png)
        total_distance += distance
        
    return total_distance


def main():
    args = make_parser().parse_args()
    annotations = args.annotations
    image_dir = args.image_dir
    training_images = args.training_images
    validation_images = args.validation_images

    if annotations == None :
        logger.debug("Annotations file path required")
        return
    if image_dir == None :
        logger.debug("Original images directory required")
        return
    if training_images == None :
        logger.debug("Eye training image directory name required")
        return
    if validation_images == None :
        logger.debug("Validation image directory path required")
        return

    ParseCVATXMLFile(annotations, image_dir, training_images)
    trainModel(training_images, validation_images)


        

if __name__ == '__main__':
    main()
