import torch

tensor = torch.randn(2, 2)  # 示例张量
print(tensor)
softmaxed_tensor = torch.softmax(tensor, dim=1)
print(softmaxed_tensor[0, :])
