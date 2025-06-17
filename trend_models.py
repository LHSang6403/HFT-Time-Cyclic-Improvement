import torch.nn as nn

# Hyperparameters
TIME_STEPS = 20
HORIZON = 15
BATCH_SIZE = 512
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 10
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
HIDDEN_SIZE = 64
GRAD_CLIP = 1.0

class CNNLSTM(nn.Module):
    def __init__(self, input_dim, device):
        super(CNNLSTM, self).__init__()
        self.device = device
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, 3)  # 3 classes: flat, up, down

    def forward(self, x):
        # CNN expects input in (batch_size, channels, sequence_length)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # LSTM expects input in (batch_size, sequence_length, features)
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]  # Take the last hidden state
        x = self.fc(x)
        return x
    
class CNN(nn.Module):
    def __init__(self, input_dim, device):
        super(CNN, self).__init__()
        self.device = device
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )
        self.fc = nn.Linear(128 * (TIME_STEPS // 4), 3)  # Adjust based on CNN output size

    def forward(self, x):
        # CNN expects input in (batch_size, channels, sequence_length)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = self.fc(x)
        return x

class LSTM(nn.Module):
    def __init__(self, input_dim, device):
        super(LSTM, self).__init__()
        self.device = device
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=HIDDEN_SIZE, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(HIDDEN_SIZE, 3)  # 3 classes: flat, up, down

    def forward(self, x):
        # LSTM expects input in (batch_size, sequence_length, features)
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]  # Take the last hidden state
        x = self.fc(x)
        return x
    