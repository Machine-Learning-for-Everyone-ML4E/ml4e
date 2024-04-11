import torch.nn as nn

# Define CNN-LSTM model
class CNN_LSTM_Model(nn.Module):
    def __init__(self, num_classes):
        super(CNN_LSTM_Model, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # LSTM layer
        self.lstm = nn.LSTM(input_size=64*7*7, hidden_size=128, num_layers=1, batch_first=True)

        # Output layer
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pool(x)
        x = self.relu(self.conv2(x))
        x = self.max_pool(x)
        x = self.flatten(x)
        x = x.unsqueeze(1)  # Add time dimension
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Use only the last output
        output = self.fc(x)
        return output
