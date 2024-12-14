import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
from torch.utils.data import Dataset
import torch.nn.functional as F

# Define the custom dataset class
class CustomImageDataset(Dataset):
    def __init__(self, csv_dir, data_dir,transform=None):
      data_csv = pd.read_csv(csv_dir)
      self.transform = transform

      self.images_list =[]
      for _, row in data_csv.iterrows():
         mapping = row.iloc[0]
         folder_dir = os.path.join(data_dir, str(mapping))
         mapping -= 33
         if mapping == 966:
             mapping = 93
         for img_file in os.listdir(folder_dir):
               img_path = os.path.join(folder_dir, img_file)
               self.images_list.append((img_path,mapping))

    
    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self, idx):
        img_path = self.images_list[idx][0]
        label = self.images_list[idx][1]
        
        image = Image.open(img_path).convert("L")
        if image.size == (24, 24):
                padded_img = Image.new('L', (28, 28), color=255)
                padded_img.paste(image, (2, 2))
                image = padded_img
        transform = transforms.ToTensor() 
        tensor = transform(image)
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, label


class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FeedForwardNN, self).__init__()
        self.flatten = nn.Flatten()  # Flatten the input tensor
        self.fc1 = nn.Linear(input_size, hidden_size)  # Linear layer (input to hidden)
        self.relu = nn.ReLU()  # ReLU activation
        self.fc2 = nn.Linear(hidden_size, int(hidden_size/2))  # Linear layer (hidden to output)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(int(hidden_size/2), int(hidden_size/2))
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(int(hidden_size/2), num_classes)


    def forward(self, x):
        x = self.flatten(x)  # Flatten [1, 28, 28] to [1, 784]
        x = self.fc1(x)      # Apply first linear layer
        x = self.relu(x)     # Apply ReLU activation
        x = self.fc2(x)    # Apply output linear layer
        x = self.relu2(x)
        x = self.fc3(x)    # Apply output linear layer
        x = self.relu3(x)
        out = self.fc4(x)
        return out
   
def validate(model, data_loader, criterion, device):
   model.eval()  # Set the model to evaluation mode
   val_loss = 0.0
   correct = 0
   total = 0

   with torch.no_grad():  # No gradients during validation
      for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

   avg_loss = val_loss / len(data_loader)
   accuracy = 100 * correct / total
   return avg_loss, accuracy

def test_model(model, data_loader, device, criterion):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation for testing
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)  # Get class with max probability
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate average loss and accuracy
    avg_loss = test_loss / len(data_loader)
    accuracy = 100 * correct / total

    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")


class CNN(nn.Module):
    def __init__(self, n_classes):
        super(CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)  # Output: 28x28
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)  # Output: 28x28
        
        # Pooling layer to downsample the spatial dimensions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces size by half
        
        # Fully connected layers
        # After two pooling layers: 28x28 → 14x14 → 7x7
        self.fc1 = nn.Linear(8 * 7 * 7, 128)  # First FC layer
        self.fc2 = nn.Linear(128, n_classes)  # Output layer
        
    def forward(self, x):
        # Convolutional block 1: Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv1(x)))  # Shape: [B, 4, 14, 14]
        
        # Convolutional block 2: Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))  # Shape: [B, 8, 7, 7]
        
        # Flatten the output before feeding to the FC layers
        x = x.view(x.size(0), -1)  # Shape: [B, 8*7*7]
        
        # Fully connected layers
        x = F.relu(self.fc1(x))  # First FC layer
        x = self.fc2(x)          # Output layer
        return x



