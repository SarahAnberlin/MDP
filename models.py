import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt


def show_images_from_palette(palette_tensor):
    b, h, w = palette_tensor.size()
    for i in range(b):
        image = palette_tensor[i, :, :].detach().numpy()  # 将张量转换为NumPy数组
        plt.figure()
        plt.imshow(image, cmap='gray')  # 假设是灰度图像，根据需要更改cmap
        plt.title(f"Image {i + 1}")
        plt.axis('off')  # 关闭坐标轴
        plt.show()


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # x.shape = (batch_size, seq_len, d_model)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        # x.shape = (batch_size, seq_len, d_model)
        return x


class CombinedNet(nn.Module):
    def __init__(self, resnet, separable):
        super(CombinedNet, self).__init__()
        self.resnet = resnet
        self.separable = separable

    def forward(self, x):
        with torch.no_grad():
            features = self.resnet(x)
        output = self.separable(features)
        return output


# Define SeparableConvNet
class SeparableConvNet(nn.Module):
    def __init__(self, out_channel=3584):
        super(SeparableConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2048, out_channels=out_channel,
                               kernel_size=5, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        print(x.shape)
        return x


if __name__ == '__main__':
    random_tensor = torch.randn(1, 3, 224, 224)
    # Load pre-trained ResNet-50 model
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # Remove the final fully connected layer and set to evaluation mode
    resnet50 = torch.nn.Sequential(*(list(resnet50.children())[:-2]))
    resnet50.eval()  # Define SeparableConvNet and concatenate networks
    ConvNet = SeparableConvNet()
    # Instantiate the combined model
    combined_model = CombinedNet(resnet=resnet50, separable=ConvNet)
    palette_tensor = combined_model(random_tensor)
