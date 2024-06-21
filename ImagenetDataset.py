import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from models import MDG
from utils import images_to_patches, single_image_to_patches


class ImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = self._find_classes(root_dir)
        self.images = self._make_dataset(root_dir, self.class_to_idx)

    def _find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return class_to_idx

    def _make_dataset(self, dir, class_to_idx):
        images = []
        for target in os.listdir(dir):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, target = self.images[idx]
        image = Image.open(img_path).convert('RGB')

        rescale = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = rescale(image)

        # print(image.shape)
        patches = single_image_to_patches(image, patch_size=(16, 16), num_patches=196)  # Assuming
        # image_to_patches
        # expects a batch

        # Apply transformation

        # Assuming image_to_patches also handles GPU
        # print(patches.shape, image.shape)
        return patches, image


if __name__ == "__main__":
    root_dir = '/dataset/sharedir/research/ImageNet/train'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print(root_dir)
    dataset = ImageNetDataset(root_dir, transform=transform)

    image, label = dataset[0]
    print(image, label)
