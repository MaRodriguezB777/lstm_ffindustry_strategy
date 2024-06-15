import torch
import torch.nn as nn
from torch.autograd import Variable


class MultiLayerLSTM(nn.Module):
    def __init__(self, input_size, output_size, params=None):
        super(MultiLayerLSTM, self).__init__()

        self.hidden_sizes = (
            params["hidden_sizes"] if params and "hidden_sizes" in params else [50]
        )
        self.dropout = params["dropout"] if params and "dropout" in params else 0
        self.activation_fn = (
            params["activation_fn"]
            if params and "activation_fn" in params
            else nn.ReLU()
        )

        self.lstm_layers = nn.ModuleList()
        self.lstm_layers.append(
            nn.LSTM(
                input_size,
                self.hidden_sizes[0],
                dropout=self.dropout,
                bidirectional=True,
                batch_first=True,
            )
        )

        for i in range(1, len(self.hidden_sizes)):
            self.lstm_layers.append(
                nn.LSTM(
                    self.hidden_sizes[i - 1] * 2,
                    self.hidden_sizes[i],
                    dropout=self.dropout,
                    bidirectional=True,
                    batch_first=True,
                )
            )

        self.output_layer = nn.Linear(self.hidden_sizes[-1] * 2, output_size)

    def forward(self, x):
        h_t = x
        for lstm_layer in self.lstm_layers:
            h_t, _ = lstm_layer(h_t)
            h_t = h_t[:, -1, :]
            h_t = self.activation_fn(h_t)

        # Only use the output of the last LSTM layer
        out = self.output_layer(h_t)
        return out


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

class SimpleFeedForwardNN(nn.Module):
    def __init__(self, input_size, output_size, params=None):
        super(SimpleFeedForwardNN, self).__init__()
        
        hidden_sizes = params["hidden_sizes_FFNN"] if params and "hidden_sizes_FFNN" in params else [50]
        self.dropout = params["dropout"] if params and "dropout" in params else 0
        
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])])
        self.layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)])
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        for layer in self.layers:
            x = nn.functional.relu(layer(x))
            x = nn.functional.dropout(x, self.dropout)
            
        x = self.output_layer(x)
        return x