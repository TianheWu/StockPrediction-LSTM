import torch.nn as nn


class Lstm(nn.Module):
    def __init__(self, input_size):
        super(Lstm, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, 1)
        
    def forward(self, x):
        x, h_n = self.lstm(x, None)
        s, b, h = x.shape
        x = x.view(s * b, h)
        x = self.fc(x)
        x = x.view(s, b, -1)
        return x
