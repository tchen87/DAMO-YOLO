import xml.etree.ElementTree as ET
from collections import defaultdict

from loguru import logger

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
from torchvision.transforms import functional as F
import os

import json

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
        self.targets = targets
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = F.to_tensor(image)
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return image, target

# Define the model
class ResNet18Regression(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18Regression, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 4)  # Output 2 values
    
    def forward(self, x):
        return self.resnet(x)

def list_png_files(folder_path):
    png_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
    return png_files

def main():
    #ParseCVATXMLFile("datasets/images_cvat/annotations.xml", "datasets/images_cvat/images/default", "outputs")
    
    pngs = list_png_files("outputs")
    logger.debug("pngs = {}", pngs)
    image_paths = []
    for imagename in pngs :
        image_paths.append(os.path.join("outputs" ,imagename))

    all_points = []  # List to store all points

    for imagename in image_paths :
        jsonfile = os.path.splitext(imagename)[0] + '.json'
        # Load JSON data
        with open(jsonfile, 'r') as jf:
            data = json.load(jf)

        points = data.get("points", [])
        # Extract x, y coordinates
        # should only be 2 points per json file

        if points[0]["label"] == "Right Eye" :
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

    # Convert list of points to a NumPy array
    points_array = np.array(all_points, dtype=np.float32)

    # Convert to PyTorch tensor
    target_tensor = torch.tensor(points_array)

    # Hyperparameters
    batch_size = 32
    epochs = 100
    learning_rate = 0.001

    # Dummy data
    targets =points_array

    dataset = CustomDataset(image_paths, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18Regression().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
        
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
    
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}")

    # Export model to ONNX
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(model, dummy_input, "resnet18_regression.onnx", 
                      input_names=["input"], output_names=["output"], 
                      opset_version=11)


if __name__ == '__main__':
    main()
