import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import torch.nn.functional as F


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
class FeatureEncoder(nn.Module):
    def __init__(self, out_channel=4096):
        super(FeatureEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2048, out_channels=out_channel,
                               kernel_size=7, stride=1, padding=0)  # 2048*7*7
        # -> 4096*1*1
        self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.Relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.Relu(self.conv1(x))
        x = x.view(batch_size, -1)
        x = self.Relu(self.fc1(x))
        x = self.Relu(self.fc2(x))
        return x


class BaseEncoder(nn.Module):
    def __init__(self, latent_dim=512, patch_size=16, base_num=128):
        super().__init__()
        self.base_dim = patch_size
        self.base_num = base_num
        self.layer1 = nn.Linear(in_features=latent_dim,
                                out_features=latent_dim * 2)
        self.h_BE = nn.Linear(in_features=latent_dim * 2,
                              out_features=patch_size * base_num)
        self.v_BE = nn.Linear(in_features=latent_dim * 2,
                              out_features=patch_size * base_num)
        self.Relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.shape[0]
        x.view(batch_size, -1)
        x = self.Relu(self.layer1(x))
        v_base = self.h_BE(x)
        h_base = self.v_BE(x)
        h_base = h_base.view(batch_size, self.base_num, self.base_dim
                             )
        v_base = v_base.view(batch_size, self.base_num, self.base_dim
                             )
        return torch.stack([h_base, v_base], dim=1)


class ScoreEncoder(nn.Module):
    def __init__(self, latent_dim=512, base_num=128, patch_num=14 * 14,
                 channel_num=3):
        super().__init__()
        self.base_num = base_num
        self.channel_num = channel_num
        sqrt_patch_num = int(patch_num ** 0.5)
        self.sqrt_patch_num = sqrt_patch_num
        self.layer = nn.Linear(latent_dim, latent_dim * 2)
        self.v_SE = nn.Linear(latent_dim * 2, base_num * sqrt_patch_num *
                              channel_num)
        self.h_SE = nn.Linear(latent_dim * 2, base_num * sqrt_patch_num *
                              channel_num)
        self.Relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.Relu(self.layer(x))
        v_score = self.v_SE(x)
        h_score = self.h_SE(x)

        h_score = h_score.view(batch_size, self.base_num, self.channel_num,
                               self.sqrt_patch_num
                               )
        v_score = v_score.view(batch_size, self.base_num, self.channel_num,
                               self.sqrt_patch_num
                               )
        return torch.stack([h_score, v_score], dim=1)


class PatchReconstructor(nn.Module):
    def __init__(self, patch_num=14 * 14, base_num=128):
        super().__init__()
        self.patch_num = patch_num
        self.base_num = base_num

    def forward(self, h_scores, v_score, h_base, v_base):
        batch_size = h_scores.shape[0]
        base_matrix = torch.einsum('bkn,bkm->bkmn', h_base, v_base)
        score_matrix = torch.einsum('bkcn,bkcm->bkcnm', h_scores, v_score)
        score_matrix = score_matrix.view(batch_size, self.base_num, 3, -1)
        mat_pn = score_matrix.shape[-1]

        if not mat_pn == self.patch_num:
            raise AssertionError(f"Patch num not matched with the "
                                 f"dimension! Expected {self.patch_num},"
                                 f" get {mat_pn} instead!")

        score_matrix = score_matrix.permute(0, 3, 2, 1)
        R_list, G_list, B_list = [], [], []
        for i in range(mat_pn):
            score = score_matrix[:, i]
            score = F.softmax(score, dim=2)
            score = score.view(batch_size, 3, self.base_num, 1, 1)
            R, G, B = score[:, 0], score[:, 1], score[:, 2]
            R_matrix = torch.sum(base_matrix * R, dim=1)
            G_matrix = torch.sum(base_matrix * G, dim=1)
            B_matrix = torch.sum(base_matrix * B, dim=1)
            R_list.append(R_matrix)
            G_list.append(G_matrix)
            B_list.append(B_matrix)
        res_R = torch.stack(R_list, dim=1)
        res_G = torch.stack(G_list, dim=1)
        res_B = torch.stack(B_list, dim=1)
        res = torch.stack([res_R, res_G, res_B], dim=2)
        return res


class MDG(nn.Module):
    def __init__(self, sqrt_patch_num=14, patch_size=16, base_num=128):
        super().__init__()
        # Load pre-trained ResNet-50 model
        resnet50 = models.resnet50(
            pretrained=True)
        # Remove the final fully connected layer and set to evaluation mode
        self.resnet50 = torch.nn.Sequential(*(list(resnet50.children())[:-2]))
        # for param in self.resnet50.parameters():
        #     param.requires_grad = False
        self.feature_encoder = FeatureEncoder(out_channel=4096)
        self.base_encoder = BaseEncoder(patch_size=16, base_num=128)
        self.score_encoder = ScoreEncoder(patch_num=sqrt_patch_num * sqrt_patch_num)
        self.patch_reconstructor = PatchReconstructor(patch_num=sqrt_patch_num * sqrt_patch_num, base_num=128)

    def forward(self, x):
        # Assuming x is your input tensor
        x = self.resnet50(x)
        # print("After resnet50, x shape:", x.shape)

        x = self.feature_encoder(x)
        # print("After feature_encoder, x shape:", x.shape)

        bases = self.base_encoder(x)
        # print("After base_encoder, bases shape:", bases.shape)

        scores = self.score_encoder(x)
        # print("After score_encoder, scores shape:", scores.shape)

        h_bases = bases[:, 0]
        v_bases = bases[:, 1]
        # print(h_bases)
        # print(v_bases)
        # print("h_bases shape:", h_bases.shape)
        # print("v_bases shape:", v_bases.shape)

        h_scores = scores[:, 0]
        v_scores = scores[:, 1]
        # print("h_scores shape:", h_scores.shape)
        # print("v_scores shape:", v_scores.shape)

        reconstructed_patches = self.patch_reconstructor(h_scores, v_scores,
                                                         h_bases, v_bases)
        # print("After patch_reconstructor, reconstructed_patches shape:",
        #        reconstructed_patches.shape)

        return reconstructed_patches


if __name__ == '__main__':
    t = torch.randn(1, 3, 224, 224)
    mdg = MDG()
    # print(mdg(t).shape)
