import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from models import SeparableConvNet, CombinedNet

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load CIFAR-100 dataset
train_data = torchvision.datasets.CIFAR100(root='./data', train=True,
                                           download=False, transform=transform)
test_data = torchvision.datasets.CIFAR100(root='./data', train=False,
                                          download=False, transform=transform)

# Define data loaders
batch_size = 16
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Load pre-trained ResNet-50 model
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
# Remove the final fully connected layer and set to evaluation mode
resnet50 = torch.nn.Sequential(*(list(resnet50.children())[:-2]))
resnet50.eval()

# Define SeparableConvNet and concatenate networks
ConvNet = SeparableConvNet()
# Instantiate the combined model
combined_model = CombinedNet(resnet=resnet50, separable=ConvNet)

# Test the combined model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
combined_model.to(device)

for images, labels in train_loader:
    images = images.to(device)
    output = combined_model(images)
    print("Output shape:", output.shape)
    break  # Only show the output shape for the first batch
