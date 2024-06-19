import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from models import MDG
from utils import images_to_patches, single_image_to_patches


class MyDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]

        transform_to_tensor = transforms.ToTensor()
        image = transform_to_tensor(image)
        # print(image.shape)
        patches = single_image_to_patches(image, patch_size=(16, 16), num_patches=4)  # Assuming
        # image_to_patches
        # expects a batch

        # Apply transformation

        rescale = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = rescale(image)
        # Assuming image_to_patches also handles GPU
        # print(patches.shape, image.shape)
        return patches, image
