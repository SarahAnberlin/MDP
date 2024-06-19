from models import FeatureEncoder, BaseEncoder, ScoreEncoder, PatchReconstructor
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt


def test1():
    FE = FeatureEncoder()
    BE = BaseEncoder()
    SE = ScoreEncoder()
    t = torch.randn(4, 2048, 7, 7)
    print("Embedding...")
    embedding = FE(t)
    print("embedding shape:", embedding.shape)
    print("Generating bases")
    bases = BE(embedding)
    print("bases shape:", bases.shape)
    h_bases = bases[:, 0]
    v_bases = bases[:, 1]
    print("V bases shape:",
          v_bases.shape)
    scores = SE(embedding)
    print("Generating scores")
    print("scores shape:", scores.shape)
    h_scores = scores[:, 0]
    v_scores = scores[:, 1]
    print("V scores shape:", h_scores.shape)
    PR = PatchReconstructor()
    res = PR(h_scores, v_scores, h_bases, v_bases)


# 打印输出形状
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


