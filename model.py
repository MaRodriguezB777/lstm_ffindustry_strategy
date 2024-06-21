#region imports
from AlgorithmImports import *
import torch
import torch.nn as nn
from torch.autograd import Variable
# endregion

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

def load_model_custom(
    path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    print("PATH TO MODEL: ", path)
    final_save = torch.load(path)

    model_state_dict = final_save["model_state_dict"]
    model_params = final_save["model_params"]
    model_type = final_save["model_type"]
    input_size = final_save["input_size"]
    output_size = final_save["output_size"]

    model = model_type(input_size, output_size, params=model_params)
    model.load_state_dict(model_state_dict)
    model.to(device)

    return model
