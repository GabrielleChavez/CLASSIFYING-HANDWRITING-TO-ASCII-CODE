class CNN_LSTM(nn.Module):
    def __init__(self, n_classes):
        super(CNN_LSTM, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)  # Output: 28x28
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)  # Output: 28x28

        # Pooling layer to downsample the spatial dimensions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces size by half

        # LSTM layer
        self.lstm_input_dim = 8 * 7  # Flattened spatial dimension after pooling
        self.lstm_hidden_dim = 128  # Hidden state dimension
        self.lstm = nn.LSTM(input_size=self.lstm_input_dim, hidden_size=self.lstm_hidden_dim, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(self.lstm_hidden_dim, 128)  # First FC layer
        self.fc2 = nn.Linear(128, n_classes)  # Output layer

    def forward(self, x):
        # Convolutional block 1: Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv1(x)))  # Shape: [B, 4, 14, 14]

        # Convolutional block 2: Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))  # Shape: [B, 8, 7, 7]

        # Flatten the spatial dimensions while keeping the batch and channel dimensions
        x = x.permute(0, 2, 3, 1).contiguous()  # Shape: [B, 7, 7, 8]
        x = x.view(x.size(0), x.size(1), -1)  # Shape: [B, 7, 8*7]

        # Pass through LSTM
        x, _ = self.lstm(x)  # Shape: [B, 7, lstm_hidden_dim]

        # Use the final hidden state for classification
        x = x[:, -1, :]  # Shape: [B, lstm_hidden_dim]

        # Fully connected layers
        x = F.relu(self.fc1(x))  # First FC layer
        x = self.fc2(x)          # Output layer
        return x