import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self, input_d, hidden_d, layer_d, output_d):
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_d
        self.layer_dim = layer_d

        # LSTM model
        self.lstm = nn.LSTM(input_d, hidden_d, layer_d, batch_first=True)

        self.fc = nn.Linear(hidden_d, output_d)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.layer_dim, x.size(0), self.hidden_dim))

        c_0 = Variable(torch.zeros(
            self.layer_dim, x.size(0), self.hidden_dim))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_dim)

        out = self.fc(h_out)

        return out
