import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from custom_dataset import HymenopteraDataset
from torch.utils.data import DataLoader
from model import CAE
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter

class RGBToLabTransform:
    def __init__(self):
        pass

    def __call__(self, rgb_image):
        assert isinstance(rgb_image, torch.Tensor), "The input image should be a PyTorch tensor"

        # Convert the tensor to numpy array and transpose the dimensions
        rgb_image_np = rgb_image.permute(1, 2, 0).numpy()

        # Convert the RGB image to L*a*b* using OpenCV
        lab_image_np = cv2.cvtColor(rgb_image_np, cv2.COLOR_RGB2LAB)

        # Convert the numpy array back to PyTorch tensor
        lab_image = torch.from_numpy(lab_image_np).permute(2, 0, 1)

        return lab_image

# Create the dataset with transformations
transform = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
    RGBToLabTransform(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = HymenopteraDataset("/content/hymenoptera_data/train",transform=transform)
test_dataset = HymenopteraDataset("/content/hymenoptera_data/val",transform=transform)




# Display three sample images
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i in range(3):
    image, label = train_dataset[i]
    class_name = 'Ant' if label == 0 else 'Bee'
    axes[i].imshow(image.permute(1, 2, 0))
    axes[i].set_title(class_name)
    axes[i].axis('off')

plt.show()

image, label = train_dataset[0]
print("The shape of the Image",image.shape)


train_dataloader = DataLoader(train_dataset, batch_size=32,
                        shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=32,
                        shuffle=True, num_workers=0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_names = train_dataset.classes


autoencoder = CAE()
# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder.to(device)


# Define loss function and optimizer
criterion = nn.MSELoss()

optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
writer = SummaryWriter()  # Initialize Tensorboard writer
# Training loop
for epoch in range(3):  # Adjust the number of epochs as needed
    for data in train_dataloader:
        inputs, labels = data
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = autoencoder(inputs) #Forward pass
        # mse = torch.mean((inputs - outputs) ** 2)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")
    writer.add_scalar('Loss/train', loss.item(), epoch)

# Close Tensorboard writer
writer.close()
# Now your autoencoder is trained!
torch.save(autoencoder, "./model/model3.pth")

data = next(iter(train_dataloader))[0]
sample_output = autoencoder(data)
fig,(ax1,ax2) = plt.subplots(1,2)
ax1.imshow(data[0].permute(1, 2, 0).detach().numpy())
ax2.imshow(sample_output[2].permute(1, 2, 0).detach().numpy())
ax1.set_title("Original Image")
ax2.set_title("Output of AutoEncoder")
# sample_output[0].permute(1, 2, 0).detach().numpy().shape

