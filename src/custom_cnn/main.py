import os
import sys

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import matplotlib.pyplot as plt

import gradio as gr

import model

net=model.LeafClassifier()
net.to(model.device)
label_mappings=[x for x in (os.listdir("../../data/train"))]

def train_and_test():
    mean = torch.tensor([0.4757, 0.5001, 0.4264])
    std = torch.tensor([0.1847, 0.1592, 0.2024])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_ds = datasets.ImageFolder("../../data/train", transform=transform)
    train_loader = DataLoader(train_ds, batch_size=40, shuffle=True)

    test_set = datasets.ImageFolder("../../data/test/", transform=transform)
    test_loader = DataLoader(test_set, batch_size=40, shuffle=False)

    model.train_model(net,train_loader,5)
    model.test_model(net,test_loader)


if "model_state.pth" in os.listdir():
    net.load_state_dict(torch.load("model_state.pth"))
    if "--train" in sys.argv:
        train_and_test()
else:
    train_and_test()

def check_plant_disease(image_path:str, net:model.LeafClassifier=net):
    try:
        mean = torch.tensor([0.4757, 0.5001, 0.4264])
        std = torch.tensor([0.1847, 0.1592, 0.2024])

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        image=Image.open(image_path)
        image=transform(image)
        image=image.to(model.device)

        with torch.no_grad():
            output=net(image.unsqueeze(0))
            _,predicted=torch.max(output.data,1)
            return label_mappings[predicted.item()]
    except Exception as e:
        print("Error in process_image:", e)
        return str(e)


if __name__=="__main__":
    demo = gr.Interface(fn=check_plant_disease, inputs=gr.Image(type="filepath"), outputs="text")
    demo.launch(debug=True)