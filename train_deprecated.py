import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models import FeatureEncoder, CombinedNet
from MDP_transformer import TransformerEncoder
import torch.optim as optim
from tqdm import tqdm

# MNIST dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet input size
    transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
    transforms.ToTensor()  # Convert to tensor
])
mnist_dataset = MNIST(root='./data', train=True, download=True,
                      transform=transform)
# Set batch size and create DataLoader with increased batch size
batch_size = 64
mnist_loader = DataLoader(mnist_dataset, batch_size=batch_size,
                          shuffle=True)

# Load pre-trained ResNet-50 model
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
# Remove the final fully connected layer and set to evaluation mode
resnet50 = torch.nn.Sequential(*(list(resnet50.children())[:-2]))
resnet50.eval()

# Define SeparableConvNet and combine networks
conv_net = FeatureEncoder()
combined_model = CombinedNet(resnet=resnet50, separable=conv_net)
mdp_TF = TransformerEncoder()

# Define L1 loss criterion
criterion = nn.L1Loss()

# Define optimizer
optimizer = optim.Adam(combined_model.parameters(), lr=0.001)
# Training loop
num_epochs = 1000
print_every = 500  # Print test image every 10 batches
save_every = 20  # Save model weights every 10 batches
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
for epoch in range(num_epochs):
    total_loss = 0.0
    progress_bar = tqdm(enumerate(mnist_loader), total=len(mnist_loader))

    for batch_idx, (inputs, _) in progress_bar:
        optimizer.zero_grad()

        # Forward pass
        reconstruct_image = mdp_TF(combined_model(inputs))

        # Calculate loss
        inputs_gray = inputs[:, 0, :,
                      :]  # Select the first channel (assuming RGB input)
        loss = criterion(reconstruct_image, inputs_gray)
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()

        # Print test image every `print_every` batches
        if (batch_idx + 1) % print_every == 0:
            # Convert tensors to numpy arrays for visualization
            original_image = inputs[0].permute(1, 2,
                                               0).cpu().numpy()  # Convert RGB input to numpy array
            reconstructed_image = reconstruct_image[
                0].squeeze().detach().cpu().numpy()  # Squeeze and convert to numpy

            # Display original and reconstructed images
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(original_image, cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            axes[1].imshow(reconstructed_image, cmap='gray')
            axes[1].set_title('Reconstructed Image')
            axes[1].axis('off')
            plt.show()

        # Save model weights every `save_every` batches
        if (batch_idx + 1) % save_every == 0:
            torch.save(combined_model.state_dict(),
                       f'checkpoint_epoch{epoch + 1}_batch{batch_idx + 1}.pth')
            print(
                f'Saved checkpoint at epoch {epoch + 1} and batch {batch_idx + 1}')
            torch.save(mdp_TF.state_dict(),
                       f'TFcheckpoint_epoch{epoch + 1}_batc'
                       f'h{batch_idx + 1}.pth')
            print(
                f'TF Saved checkpoint at epoch {epoch + 1} and batch'
                f' {batch_idx + 1}')

        # Update tqdm progress bar description
        progress_bar.set_description(
            f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(mnist_loader)}], Avg L1 Loss: {total_loss / (batch_idx + 1):.6f}')

    # Calculate average loss for the epoch
    average_loss = total_loss / len(mnist_loader)
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Average L1 Loss: {average_loss:.6f}")
# Training loop
# num_epochs = 1000
# print_every = 10  # Print test image every 10 batches
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Device: {device}')
# for epoch in range(num_epochs):
#     total_loss = 0.0
#     progress_bar = tqdm(enumerate(mnist_loader), total=len(mnist_loader))
#
#     for batch_idx, (inputs, _) in progress_bar:
#         optimizer.zero_grad()
#
#         # Forward pass
#         reconstruct_image = mdp_TF(combined_model(inputs))
#
#         # Calculate loss
#         inputs_gray = inputs[:, 0, :,
#                       :]  # Select the first channel (assuming RGB input)
#         loss = criterion(reconstruct_image, inputs_gray)
#         total_loss += loss.item()
#
#         # Backward pass
#         loss.backward()
#         optimizer.step()
#
#         # Print test image every `print_every` batches
#         if (batch_idx + 1) % 1 == 0:
#             # Convert tensors to numpy arrays for visualization (assuming inputs are tensors)
#             original_image = inputs[0].permute(1, 2,
#                                                0).cpu().numpy()  # Convert RGB input to numpy array
#             reconstructed_image = reconstruct_image[
#                 0].squeeze().detach().numpy()  # Squeeze and convert grayscale reconstruct to numpy
#
#             # Display original and reconstructed images
#             fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#             axes[0].imshow(original_image.squeeze(), cmap='gray')
#             axes[0].set_title('Original Image')
#             axes[0].axis('off')
#             axes[1].imshow(reconstructed_image.squeeze(), cmap='gray')
#             axes[1].set_title('Reconstructed Image')
#             axes[1].axis('off')
#             plt.show()
#
#         # Update tqdm progress bar description
#         progress_bar.set_description(
#             f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(mnist_loader)}], Avg L1 Loss: {total_loss / (batch_idx + 1):.6f}')
#
#     # Calculate average loss for the epoch
#     average_loss = total_loss / len(mnist_loader)
#     print(
#         f"Epoch [{epoch + 1}/{num_epochs}], Average L1 Loss: {average_loss:.6f}")
