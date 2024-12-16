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
import random
import numpy as np


class AlphaNumDataset(Dataset):
    """
    Dataset class for the alphanumeric dataset
    """

    def __init__(self, csv_dir, data_dir,transform=None):
      """
        Initializes the dataset

        Args:
            csv_dir (str): Path to the CSV file containing the mappings from indices to labels
            data_dir (str): Path to the directory containing the image folders
            transform (callable, optional): Optional transform to be applied
      """
      data_csv = pd.read_csv(csv_dir)
      self.transform = transform

      self.images_list =[]
      for _, row in data_csv.iterrows():
         mapping = row.iloc[0]
         folder_dir = os.path.join(data_dir, str(mapping))
         for img_file in os.listdir(folder_dir):
               img_path = os.path.join(folder_dir, img_file)
               self.images_list.append((img_path,mapping))
               

    def __len__(self):
        """
        Returns the length of the dataset

        Returns:
            int: Length of the dataset
        """
        return len(self.images_list)
    
    def __getitem__(self, idx):
        """
        Returns the image and label couple at the given index

        Args:
            idx (int): Index of the item to be retrieved
        
        Returns:
            tuple: Tuple containing the image tensor and the label

        """
        img_path = self.images_list[idx][0]
        label = self.images_list[idx][1]
        
        image = Image.open(img_path).convert("L")
        if image.size == (24, 24):
                # Padding with white pixels on each side to ensure consistent size
                padded_img = Image.new('L', (28, 28), color=255)
                padded_img.paste(image, (2, 2))
                image = padded_img

        if label > 92:
            new_mapping = label - 34
        else: 
            new_mapping = label - 33

        if new_mapping == 965:
            new_mapping = 92
    
        transform = transforms.ToTensor() 
        tensor = transform(image)
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, new_mapping


class FeedForwardNN(nn.Module):
    """
    Feedforward neural network with 3 hidden layers and ReLU activation functions
    """
    def __init__(self, input_size, hidden_size, num_classes):
        """
        Initializes the feedforward neural network

        Args:
            input_size (int): Size of the input tensor (pixels in the image)
            hidden_size (int): Size of the hidden layer 
            num_classes (int): Number of classes in the dataset
        """

        super(FeedForwardNN, self).__init__()
        self.flatten = nn.Flatten()  # Flatten the input image tensor to a vector
        self.fc1 = nn.Linear(input_size, hidden_size)   # Linear layers
        self.relu = nn.ReLU()  # Activation functions
        self.fc2 = nn.Linear(hidden_size, int(hidden_size/2))  
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(int(hidden_size/2), int(hidden_size/2))
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(int(hidden_size/2), num_classes)

    def forward(self, x):
        """
        Forward pass of the neural network

        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Logits for each class
        
        """
        x = self.flatten(x) 
        x = self.fc1(x)      
        x = self.relu(x)     
        x = self.fc2(x)    
        x = self.relu2(x)
        x = self.fc3(x)    
        x = self.relu3(x)
        out = self.fc4(x)
        return out


class CNN(nn.Module):
    """
    Convolutional neural network with 2 convolutional layers 
    """
    def __init__(self, n_classes):
        """
        Initializes the CNN

        Args:
            n_classes (int): Number of classes in the dataset
        """
        super(CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)  # Output: 28x28
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)  # Output: 28x28
        
        # Pooling layer 2x2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces size by half
        
        self.fc1 = nn.Linear(8 * 7 * 7, 128)  # First FC layer
        self.fc2 = nn.Linear(128, n_classes)  # Output layer
        
    def forward(self, x):
        """
        Forward pass of the CNN

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Logits for each class
        """
        # Convolution 1
        x = self.pool(F.relu(self.conv1(x)))  # Shape: [B, 4, 14, 14]
        
        # Convolution 2
        x = self.pool(F.relu(self.conv2(x)))  # Shape: [B, 8, 7, 7]
        
        # Flatten to batch of vectors
        x = x.view(x.size(0), -1)  # Shape: [B, 8*7*7]
        
        x = F.relu(self.fc1(x))  # First FC layer
        x = self.fc2(x)          # Output layer
        return x
    
class CNN_LSTM(nn.Module):
    """
    Convolutional neural network with 2 convolutional layers followed by an LSTM layer
    """
    def __init__(self, n_classes):
        """
        Initializes the CNN-LSTM model

        Args:
            n_classes (int): Number of classes in the dataset  
        """
        super(CNN_LSTM, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)  
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)  

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  

        # LSTM
        self.lstm_input_dim = 8 * 7  
        self.lstm_hidden_dim = 128
        self.lstm = nn.LSTM(input_size=self.lstm_input_dim, hidden_size=self.lstm_hidden_dim, batch_first=True)

        self.fc1 = nn.Linear(self.lstm_hidden_dim, 128)  
        self.fc2 = nn.Linear(128, n_classes) 

    def forward(self, x):
        """
        Forward pass of the CNN-LSTM model

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Logits for each class
        """

        # Convolution 1
        x = self.pool(F.relu(self.conv1(x)))  # Shape: [B, 4, 14, 14]

        # Convolution 2
        x = self.pool(F.relu(self.conv2(x)))  # Shape: [B, 8, 7, 7]

        # Flatten the dimensions
        x = x.permute(0, 2, 3, 1).contiguous()  # Shape: [B, 7, 7, 8]
        x = x.view(x.size(0), x.size(1), -1)  # Shape: [B, 7, 8*7]

        # LSTM Pass
        x, _ = self.lstm(x)  # Shape: [B, 7, lstm_hidden_dim]

        # Get the last hidden state
        x = x[:, -1, :]  # Shape: [B, lstm_hidden_dim]

        x = F.relu(self.fc1(x)) # Activation function + penultimate linear layer
        x = self.fc2(x)          # Output layer
        return x
    
class CNNFeatureExtractor(nn.Module):
    """
    Convolutional Neural Network that extracts features with 2 convolutional layers
    Later to be used in the Transformer model, does not predict the final output
    """
    def __init__(self):
        """
        Initializes the CNN feature extractor
        """
        super(CNNFeatureExtractor, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)  # Output: 28x28
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)  # Output: 28x28
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces size by half
        
        
    def forward(self, x):
        """
        Forward pass of the CNN feature extractor

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Extracted features
        """
        # Convolution 1
        x = self.pool(F.relu(self.conv1(x)))  # Shape: [B, 4, 14, 14]
        
        # Convolution 2
        x = self.pool(F.relu(self.conv2(x)))  # Shape: [B, 8, 7, 7]
        return x

class Encoder(nn.Module):
    """
    Encoder class for the Transformer model 
    """
    def __init__(self,d_model, dropout = .1, max_len = 5000):
        """
        Initializes the encoder

        Args:
            d_model (int): Dimension of the model
            dropout (float): Dropout probability
            max_len (int): Maximum length of the input sequence
        """
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout) # Avoid overfitting
        encoded_values = torch.zeros(max_len, d_model) # Initialize encoding
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        normal_const = torch.exp(torch.arange(0,d_model,2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        encoded_values[:,0::2] = torch.sin(position * normal_const)
        encoded_values[:, 1::2] = torch.cos(position * normal_const)
        encoded_values = encoded_values.unsqueeze(1)
        self.register_buffer('pe', encoded_values) # Register encoded values as buffer
    
    def forward(self, x):

        """
        Forward pass of the encoder

        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Encoded tensor
        """

        seq_len = x.size(0)
        x = x + self.encoded_values[:seq_len]
        return self.dropout(x)
    

class CNNTransformer(nn.Module):
    """
    Transformer model that uses a CNN to extract features
    """

    def __init__(self, n_classes, d_model = 64, n_heads = 4, num_layers = 3, dropout = .1):
        """
        Initializes the Transformer model

        Args:
            n_classes (int): Number of classes in the dataset
            d_model (int): Dimension of the model
            n_heads (int): Number of heads in the multi-head attention mechanism
            num_layers (int): Number of layers in the transformer
            dropout (float): Dropout probability
        """

        super(CNNTransformer, self).__init__()
        self.cnn = CNNFeatureExtractor()
        self.channel_dim = 8
        self.spatial_dim = 7*7
        self.linear_proj = nn.Linear(self.channel_dim, d_model)
        self.pos_encoder = Encoder(d_model=d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,nhead = n_heads, dim_feedforward= 4*d_model, dropout=dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = num_layers)
        self.fc_out = nn.Linear(d_model, n_classes)

    def forward(self, x):
        """
        Forward pass of the Transformer model

        Args:

            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Logits for each class
        """
        features = self.cnn(x)
        B, C, H, W = features.size()
        features = features.view(B,C, H*W)
        features = features.permute(2,0,1)
        features = self.linear_proj(features)
        features = self.pos_encoder(features)
        transformed = self.transformer_encoder(features)
        pooled = transformed.mean(dim = 0)
        logits = self.fc_out(pooled)
        return logits


def validate(model, data_loader, criterion, device):
   """
    Function to validate the model on the validation set

    Args:
        model (nn.Module): Model
        data_loader (DataLoader): DataLoader for the validation set
        criterion (nn.Module): Loss function
        device (torch.device): PyTorch device

    Returns:
        float: Average loss
        float: Accuracy
   """
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
    """
    Function to test the model on the test set

    Args:

        model (nn.Module): Model
        data_loader (DataLoader): DataLoader for the test set
        device (torch.device): PyTorch device
        criterion (nn.Module): Loss function
    """
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
    

def set_seed(seed = 42):
    """
    Function to set the seed for various libraries

    Args:
        seed (int): Seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_model(num_epochs, data_loader, val_loader, device, model, criterion, optimizer, scheduler, patience, counter = 0):
    """
    Training loop for the model

    Args:

        num_epochs (int): Number of epochs
        data_loader (DataLoader): DataLoader for the training set
        val_loader (DataLoader): DataLoader for the validation set
        device (torch.device): PyTorch device
        model (nn.Module): Model
        criterion (nn.Module): Loss function
        optimizer (torch.optim): Optimizer
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler
        patience (int): Patience for early stopping
        counter (int): Counter for early stopping
    """
    
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # Training step
        model.train()
        running_loss = 0.0
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation step
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(data_loader):.4f}, "
            f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            print("Validation loss improved! Saving model...")
        else:
            counter += 1
            print(f"Validation loss did not improve. Patience: {counter}/{patience}")
            if counter >= patience:
                print("Early stopping triggered!")
                break
        scheduler.step(val_loss)





