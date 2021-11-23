import torch.nn as nn


class Lstm(nn.Module):
    def __init__(self, input_size):
        super(Lstm, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        x, h_n = self.lstm(x, None)
        s, b, h = x.shape
        x = x.view(s * b, h)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(s, b, -1)
        return x
