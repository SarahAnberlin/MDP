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
from ImagenetDataset import ImageNetDataset
import json
import os

parser = argparse.ArgumentParser(description='Training parameters for the model.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs to train')
parser.add_argument('--saving_interval', type=int, default=50, help='Interval for saving the model')
parser.add_argument('--validate_interval', type=int, default=50, help='Interval for validation')

args = parser.parse_args()

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Define data transformations

# Load CIFAR-100 dataset

root_dir = '/dataset/sharedir/research/ImageNet/train'

train_dataset = ImageNetDataset(root_dir)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# print("Patches shape", train_dataset[0][0])
# print(f'image shape {train_dataset[0][1]}')
# Hyperparameters
batch_size = args.batch_size
learning_rate = args.learning_rate
num_epochs = args.num_epochs
saving_interval = args.saving_interval
validate_interval = args.validate_interval

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# Initialize model and move to GPU
model = MDG(sqrt_patch_num=14, patch_size=16, base_num=512).to(device)
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

total_step = len(train_loader)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

log_json_path = '/dataset/vfayezzhang/PythonProject/myGenerator/weights/v3/log.json'

# Convert args to dictionary
args_dict = vars(args)  # Converts argparse.Namespace to dictionary
if not os.path.exists(log_json_path):
    os.makedirs(log_json_path)
# Save args dictionary to log.json
with open(log_json_path, 'w') as f:
    json.dump(args_dict, f)

print("Training...")
avg_val_loss = 0
# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0

    # Train for each batch
    for i, (patches, images) in enumerate(tqdm(train_loader, desc=f'Epoch [{epoch + 1}/{num_epochs}]', leave=False)):
        images = images.to(device)  # Move batch of images to GPU
        patches = patches.to(device)  # Move patches to GPU

        outputs = model(images)
        # print(f'Outputs shape{outputs.shape}')
        loss = criterion(outputs, patches)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Print loss
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / total_step:.4f}')
    # Validate
    if (epoch + 1) % validate_interval == 0:
        model.eval()
        val_losses = []
        for patches, images in val_loader:
            images = images.to(device)
            patches = patches.to(device)  # Move patches to GPU

            outputs = model(images)
            val_loss = criterion(outputs, patches).item()
            val_losses.append(val_loss)

        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f'Validation Loss after Epoch {epoch + 1}: {avg_val_loss:.4f}')
    # Saving weights
    if (epoch + 1) % saving_interval == 0:
        torch.save(
            model.state_dict(),
            f'/dataset/vfayezzhang/PythonProject/myGenerator/weights/v3/{epoch + 1}_trainloss'
            f'_{epoch_loss / total_step:.4f}_valloss_{avg_val_loss:.4f}.pth'
        )
        print(f'Model weights saved after Epoch {epoch + 1}')
