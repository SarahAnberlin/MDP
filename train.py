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

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Define data transformations

# Load CIFAR-100 dataset
train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=None)

train_size = int(0.9999 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Create instances of custom dataset
train_dataset = MyDataset(train_dataset)
val_dataset = MyDataset(val_dataset)

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 1000
interval = 50

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# Initialize model and move to GPU
model = MDG(sqrt_patch_num=2, patch_size=16, base_num=128).to(device)
total_step = len(train_loader)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):

    for i, (patches, images) in enumerate(tqdm(train_loader)):
        images = images.to(device)  # Move batch of images to GPU
        patches = patches.to(device)  # Move patches to GPU

        outputs = model(images)
        loss = criterion(outputs, patches)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % interval == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')

    if (epoch + 1) % 50 == 0:
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

        # Saving model checkpoint
        torch.save(model.state_dict(), f'{epoch + 1}_{avg_val_loss:.4f}.pth')
