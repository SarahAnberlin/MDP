import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np


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


def patches_to_single_image(patches, image_size, patch_size=(16, 16), stride=16):
    """
    将若干个小patch还原为单张图像
    Args:
    - patches (Tensor): 输入的patch，形状为 (num_patches, C, patch_size[0], patch_size[1])
    - image_size (tuple): 原始图像的大小，形状为 (C, H, W)
    - patch_size (tuple): patch 的大小，默认为 (16, 16)
    - stride (int): patch 的步幅，默认为 16

    Returns:
    - image (Tensor): 还原后的图像，形状为 (C, H, W)
    """
    B, num_patches, C, patch_H, patch_W = patches.shape
    H, W = image_size[1], image_size[2]

    # 恢复patch的形状并进行转置
    patches = patches.view(B, num_patches, C, patch_H, patch_W)
    patches = patches.permute(0, 2, 3, 4, 1)
    patches = patches.view(B, C * patch_H * patch_W, num_patches)

    # 定义nn.Fold并还原图像
    folder = nn.Fold(output_size=(H, W), kernel_size=patch_size, stride=stride)
    image = folder(patches)

    # 消除批次维度
    image = image.squeeze(0)

    return image


def show_images_via_patches(patches, show=True, save_path=None):
    """
    将图像补丁拼接成完整图像并显示或保存。

    参数:
    patches (torch.Tensor): 图像补丁张量，形状为 (batch_size, num_patches, channels, patch_size, patch_size)
    show (bool): 是否显示重构的图像。默认为 True。
    save_path (str or None): 如果提供，将保存图像到指定路径。默认为 None。
    """
    # print(patches.shape)
    batch_size = patches.size(0)
    num_patches = patches.size(1)
    channels = patches.size(2)
    patch_size = patches.size(3)

    grid_size = int(np.sqrt(num_patches))

    for i in range(batch_size):
        # 初始化一个空的 numpy 数组来存储整个拼接的图像
        full_image = np.zeros((grid_size * patch_size, grid_size * patch_size, channels), dtype=np.float32)

        for j in range(num_patches):
            # 计算当前 patch 在整体图像中的位置
            row = j // grid_size
            col = j % grid_size

            # 提取当前 patch
            patch = patches[i, j]
            # print(patch.shape)
            # print(patch)

            # 将通道维度移动到最后（HWC 格式）
            patch_np = patch.permute(1, 2, 0).cpu().detach().numpy()

            # 放置到整体图像中的正确位置
            full_image[row * patch_size:(row + 1) * patch_size, col * patch_size:(col + 1) * patch_size, :] = patch_np

        # 缩放到 [0, 255] 范围并转换为 uint8
        full_image = (full_image * 255).clip(0, 255).astype(np.uint8)

        # 转换为 PIL 图像
        full_img = Image.fromarray(full_image)

        # 显示图像
        if show:
            full_img.show()

        # 保存图像
        if save_path:
            full_img.save(f"{save_path}_image_{i}.png")


if __name__ == '__main__':
    # 加载图像并转换为张量，并resize为32x32像素
    image_path = "lena.png"
    image = Image.open(image_path).convert('RGB')
    image = image.resize((32, 32))  # resize为32x32像素
    image = torch.tensor(np.array(image)).permute(2, 0, 1).float()  # 将PIL图像转换为张量，并调整通道顺序
    image = torch.stack([image, image], dim=0)
    patches = images_to_patches(image, num_patches=4, patch_size=(16, 16), channel_num=3)

    patches = patches - patches.min()
    patches = patches / patches.max()

    show_images_via_patches(patches)
