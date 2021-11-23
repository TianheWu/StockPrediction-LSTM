import torch.nn as nn


class Lstm(nn.Module):
    def __init__(self, input_size):
        super(Lstm, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        x, (h_n, h_c) = self.lstm(x, None)
        x = self.fc1(x[:, -1, :])
        x = self.fc2(x)
        return x
