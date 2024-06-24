#region imports
from AlgorithmImports import *
import torch
import torch.nn as nn
from torch.autograd import Variable
# endregion

URLS_NORMALIZED = {
    2023: "https://drive.google.com/uc?export=download&id=1HzWs0r3hzGVTRVUwoKqt0tIY6nWLF6Ux"
}

URLS_UNNORMALIZED = {
    2023: "https://drive.google.com/uc?export=download&id=1E_QVWqE2lkMVGBcbF3MpfFPASSwXBOrP"
}

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, output_size, params=None):
        super(SimpleLSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.hidden_size = (
            params["hidden_size_LSTM"] if params and "hidden_size_LSTM" in params else 50
        )
        self.num_layers = (
            params["num_layers"] if params and "num_layers" in params else 1
        )
        self.dropout = params["dropout"] if params and "dropout" in params else 0
        self.activation_fn = (
            params["activation_fn"]
            if params and "activation_fn" in params
            else nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.fc_linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        device = x.device
        
        h_0 = Variable(
            torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        )  # hidden state
        c_0 = Variable(
            torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        )  # internal state

        h_0 = h_0.to(device)
        c_0 = c_0.to(device)
        
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(
            x, (h_0, c_0)
        )  # lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        out = self.activation_fn(hn)
        out = self.fc_linear(out)
        return out
