import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn


# Define the resize and tensor transformation


# def generate_and_visualize_patches(image_path, num_patches=4):
#     resize_transform = transforms.Compose([
#         transforms.Resize((32, 32)),  # Resize the image to 32x32
#         transforms.ToTensor()  # Convert PIL image to Tensor
#     ])
#     # Open the image and apply transformations
#     image_pil = Image.open(image_path).convert("RGB")
#     image_resized = resize_transform(image_pil).unsqueeze(0)  # Add batch dimension
#
#     # Generate patches
#     patches = image_to_patches(image_resized, num_patches)
#
#     # Print the shape of generated patches
#     print("Generated patches shape:", patches.shape)  # Should be (1, num_patches^2, 3, 16, 16)
#
#     # Visualize each patch
#     fig, axes = plt.subplots(int(num_patches ** 0.5), int(num_patches ** 0.5), figsize=(10, 10))
#     for i in range(int(num_patches ** 0.5)):
#         for j in range(int(num_patches ** 0.5)):
#             patch_index = i * int(num_patches ** 0.5) + j
#             patch = patches[0, patch_index].permute(1, 2, 0).numpy()
#             axes[i, j].imshow(patch)
#             axes[i, j].axis("off")
#     plt.tight_layout()
#     plt.show()


def images_to_patches(image, num_patches=196, patch_size=(16, 16), channel_num=3):
    """
    将图像转换为若干个小patch
    Args:
    - image (Tensor): 输入的图像，形状为 (B, C, H, W)
    - patch_size (tuple): patch 的大小，默认为 (16, 16)

    Returns:
    - patches (Tensor): 分割后的 patch，形状为 (B, num_patches, C, patch_size[0],
    patch_size[1])
    """
    B = image.size(0)
    unfolder = nn.Unfold(kernel_size=patch_size, stride=16)
    patches = unfolder(image)
    patches = patches.view(B, 3, 16 * 16, num_patches)
    patches = patches.permute(0, 3, 1, 2)
    patches = patches.view(B, num_patches, channel_num, patch_size[0], patch_size[1])
    return patches


def single_image_to_patches(image, num_patches=4, patch_size=(16, 16), channel_num=3):
    """
    将单张图像转换为若干个小patch
    Args:
    - image (Tensor): 输入的图像，形状为 (C, H, W)
    - patch_size (tuple): patch 的大小，默认为 (16, 16)

    Returns:
    - patches (Tensor): 分割后的 patch，形状为 (num_patches, C, patch_size[0], patch_size[1])
    """
    # 添加一个维度，使得图像形状变为 (1, C, H, W)
    image = image.unsqueeze(0)
    B = image.size(0)
    unfolder = nn.Unfold(kernel_size=patch_size, stride=16)
    patches = unfolder(image)
    patches = patches.view(B, channel_num, patch_size[0] * patch_size[1], num_patches)
    patches = patches.permute(0, 3, 1, 2)
    patches = patches.view(num_patches, channel_num, patch_size[0], patch_size[1])
    return patches


if __name__ == '__main__':
    image_path = "lena.png"
    # generate_and_visualize_patches(image_path, num_patches=4)
