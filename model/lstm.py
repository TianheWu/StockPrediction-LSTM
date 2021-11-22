import torch.nn as nn


class Lstm(nn.Module):
    def __init__(self, input_size):
        super(Lstm, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=64, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(64, 1)
        
    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)
        out = self.dropout(r_out)
        out = self.out(out)
        return out
