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

parser = argparse.ArgumentParser(description='Training parameters for the model.')
parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
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
train_dataset = torchvision.datasets.CIFAR100(root='/dataset/vfayezzhang/PythonProject/myGenerator/', train=True,
                                              download=True, transform=None)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Create instances of custom dataset
train_dataset = MyDataset(train_dataset)
val_dataset = MyDataset(val_dataset)

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
model = MDG(sqrt_patch_num=2, patch_size=16, base_num=512).to(device)
total_step = len(train_loader)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0

    # Train for each batch
    for i, (patches, images) in enumerate(tqdm(train_loader, desc=f'Epoch [{epoch + 1}/{num_epochs}]', leave=False)):
        images = images.to(device)  # Move batch of images to GPU
        patches = patches.to(device)  # Move patches to GPU

        outputs = model(images)
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
        torch.save(model.state_dict(),
                   f'/dataset/vfayezzhang/PythonProject/myGenerator/weights/v2/{epoch + 1}_{avg_val_loss:.4f}.pth')
