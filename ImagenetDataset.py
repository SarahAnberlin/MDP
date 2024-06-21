import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


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
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, target


if __name__ == "__main__":
    root_dir = '/dataset/sharedir/research/ImageNet/train'  # 替换为你的ImageNet训练集路径
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageNetDataset(root_dir, transform=transform)

    # 访问数据集示例
    image, label = dataset[0]  # 获取第一张图像和其类别标签
    print(image, label)
