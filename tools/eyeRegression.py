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


def ParseCVATXMLFile(filename, images_dir, output_dir) :
    
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
            else : 
                logger.debug("Did not find 2 points within the bounding box! Not writing out crop or json")



            print(f"Saved: {crop_path} with {len(crop_points)} points")


# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, image_paths, targets, transform=None):
        self.image_paths = image_paths
        self.targets = targets  # Shape: (N, 4) for 2 points (x1, y1, x2, y2)
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.resize(image, (224, 224))
        target = self.targets[idx]
        keypoints = [(target[0], target[1]), (target[2], target[3])]

        if self.transform:
            augmented = self.transform(image=image, keypoints=[(target[0], target[1]), (target[2], target[3])])
            image = augmented['image']
            keypoints = augmented['keypoints']
            if len(keypoints) != 2:
                # Return a dummy tensor with zeros if keypoints are lost
                #logger.debug("lost keypoints!")
                target = torch.zeros(4, dtype=torch.float32)
            else:
                target = torch.tensor([kp[0] for kp in keypoints] + [kp[1] for kp in keypoints], dtype=torch.float32)        
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


def trainModel() :
    pngs = list_png_files_in_directory("cropped")
    image_paths = []
    for imagename in pngs :
        image_paths.append(imagename)

    val_split = 0.2
    val_size = int(len(image_paths) * val_split)
    train_size = len(image_paths) - val_size
    train_dataset, val_dataset = random_split(image_paths, [train_size, val_size])
    
    train_points = LoadPointsForImages(train_dataset)
    
    # Convert list of points to a NumPy array
    train_points_array = np.array(train_points, dtype=np.float32)

    # Dummy data
    train_targets =train_points_array

    val_points = LoadPointsForImages(val_dataset)
    
    # Convert list of points to a NumPy array
    val_points_array = np.array(val_points, dtype=np.float32)

    # Dummy data
    val_targets =val_points_array


    # Define Albumentations transforms
    transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format='xy'))
    
    train_dataset = CustomDataset(train_dataset, train_targets)
    val_dataset = CustomDataset(val_dataset, val_targets)
    #dataset = CustomDataset(image_paths, targets, transform=transform)

    # Hyperparameters
    batch_size = 32
    epochs = 100
    learning_rate = 0.001

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18Regression().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
    
        print(f"Training Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_dataloader):.8f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_dataloader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
    
        print(f"Validation Epoch [{epoch+1}/{epochs}], Loss: {val_loss/len(val_dataset):.8f}")

        # Export model to ONNX
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        model_name = "resnet18_regression_ckpt" + str(epoch + 1)+ ".onnx"
        torch.onnx.export(model, dummy_input, model_name, 
                          input_names=["input"], output_names=["output"], 
                          opset_version=11)



def Preprocess(image) :
    image = cv2.resize(image, (224, 224))
    transform3 = transforms.Compose([transforms.ToTensor()])
    transform4= transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    image = transform3(image)
    image = transform4(image)
    return image

def testModelOnImage(image_path) :
    image = cv2.imread(image_path)
    input_tensor = Preprocess(image)
    input_tensor = input_tensor.reshape(1, 3, 224, 224)
        # Run inference
    ort_session = ort.InferenceSession("resnet18_regression.onnx")
    input_np = input_tensor.cpu().numpy()
    logger.debug(input_np.shape)

    onnx_output = ort_session.run(None, {"input": input_np})
    logger.debug(onnx_output[0][0])
    rx = onnx_output[0][0][0]
    ry = onnx_output[0][0][1]
    lx = onnx_output[0][0][2]
    ly = onnx_output[0][0][3]
    
    lx = int(lx * image.shape[1])
    ly = int(ly * image.shape[0])
    rx = int(rx * image.shape[1])
    ry = int(ry * image.shape[0])
    logger.debug("left eye = {} {}, right eye = {} {}", lx, ly, rx, ry)

    cv2.circle(image, (lx, ly), 1, (0, 255, 0))
    cv2.circle(image, (rx, ry), 1, (0, 255, 0))

    #write out labeled eye positiosn
    [justfilename, ext] = os.path.splitext(image_path)
    jsonfilename = justfilename + ".json"

    with open(jsonfilename, "r") as file:
        jsonstuff = json.load(file)
    logger.debug(jsonstuff)
    points = jsonstuff['points']
    for point in points :
        if point['label'] == "Left eye" :
            lx2 = point['x']
            ly2 = point['y']
        elif point['label'] == "Right eye" :
            rx2 = point['x']
            ry2 = point['y']
            
    lx2 = int(lx2 * image.shape[1])
    ly2 = int(ly2 * image.shape[0])
    rx2 = int(rx2 * image.shape[1])
    ry2 = int(ry2 * image.shape[0])
    cv2.circle(image, (lx2, ly2), 1, (0, 0, 255))
    cv2.circle(image, (rx2, ry2), 1, (0, 0, 255))
    
   
    output_filename = os.path.basename(image_path)
    output_filename = "outputs/" + output_filename 
    cv2.imwrite(output_filename, image)
    # cv2.imshow('Circles', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def main():
    #ParseCVATXMLFile("datasets/images_cvat/annotations.xml", "datasets/images_cvat/images/default", "outputs")
   # trainModel()
    pngfiles = list_png_files_in_directory("testInputs")
    for png in pngfiles:
        testModelOnImage(png)

if __name__ == '__main__':
    main()
