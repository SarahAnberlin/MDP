import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models import FeatureEncoder, CombinedNet
from MDP_transformer import TransformerEncoder


# Define test function
def test_model(checkpoint_path, transformer_checkpoint_path, batch_size=64):
    # MNIST test dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match ResNet input size
        transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
        transforms.ToTensor()  # Convert to tensor
    ])
    mnist_test_dataset = MNIST(root='./data', train=False, download=True,
                               transform=transform)
    mnist_test_loader = DataLoader(mnist_test_dataset, batch_size=batch_size,
                                   shuffle=False)

    # Load pre-trained ResNet-50 model
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet50 = torch.nn.Sequential(*(list(resnet50.children())[:-2]))
    resnet50.eval()

    # Define SeparableConvNet and combine networks
    conv_net = FeatureEncoder()
    combined_model = CombinedNet(resnet=resnet50, separable=conv_net)

    # Load model weights
    combined_model.load_state_dict(torch.load(checkpoint_path))
    combined_model.eval()

    # Define Transformer Encoder
    mdp_TF = TransformerEncoder()

    # Load transformer weights
    mdp_TF.load_state_dict(torch.load(transformer_checkpoint_path))
    mdp_TF.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    combined_model.to(device)
    mdp_TF.to(device)

    # Set models to evaluation mode
    combined_model.eval()
    mdp_TF.eval()

    with torch.no_grad():
        for inputs, _ in mnist_test_loader:
            inputs = inputs.to(device)

            # Forward pass
            outputs = mdp_TF(combined_model(inputs))

            # Visualize the original and reconstructed images
            original_image = inputs[0].permute(1, 2, 0).cpu().numpy()
            reconstructed_image = outputs[0].squeeze().cpu().numpy()

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(original_image, cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            axes[1].imshow(reconstructed_image, cmap='gray')
            axes[1].set_title('Reconstructed Image')
            axes[1].axis('off')
            plt.show()
            break  # Only visualize the first batch


# Example usage
test_model('checkpoint_epoch1_batch160.pth', 'TFcheckpoint_epoch1_batch160.pth')
