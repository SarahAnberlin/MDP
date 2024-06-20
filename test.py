import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from models import MDG
from utils import images_to_patches
from MyDataset import MyDataset
import argparse
from PIL import Image
from utils import patches_to_single_image, show_images_via_patches

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Define data transformations

# Load CIFAR-100 dataset
train_dataset = torchvision.datasets.CIFAR100(root='/dataset/vfayezzhang/PythonProject/myGenerator/', train=True,
                                              download=True, transform=None)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Create instances of custom dataset
train_dataset = MyDataset(train_dataset)
val_dataset = MyDataset(val_dataset)

batch_size = 1
# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)

# Initialize model and move to GPU
model = MDG(sqrt_patch_num=2, patch_size=16, base_num=128).to(device)
total_step = len(train_loader)
model.load_state_dict(torch.load('D:\PythonProjects\myGenerator\weights\\v1\\122_0.1417.pth'))
model.eval()
for patches, images in tqdm(val_loader):
    images = images.to(device)  # Move batch of images to GPU
    patches = patches.to(device)  # Move patches to GPU
    show_images_via_patches(patches.cpu().detach())

    outputs = model(images)
    show_images_via_patches(outputs.cpu().detach())

    # reconstructed_image = patches_to_single_image(outputs, image_size=(3, 32, 32), patch_size=(16, 16), stride=16)
    # reconstructed_image_pil = Image.fromarray(reconstructed_image.permute(1, 2, 0).byte().cpu().numpy())
    # reconstructed_image_pil.show()
    time.sleep(20)
