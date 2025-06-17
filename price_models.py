import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Transpose to match Conv1d input shape (batch_size, channels, sequence_length)
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)  # Pool and remove the last dimension
        x = self.fc(x)
        return x
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM expects input shape (batch_size, sequence_length, features)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take the last time step
        x = self.fc(x)
        return x
    
class CNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(CNNLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # CNN expects input in (batch_size, channels, sequence_length)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # LSTM expects input in (batch_size, sequence_length, features)
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]  # Take the last hidden state
        x = self.fc(x)
        return x