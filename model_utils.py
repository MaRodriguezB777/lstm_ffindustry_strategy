# region imports
from AlgorithmImports import *
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from model import SimpleLSTM
import pickle
import base64
# endregion

FF_COLS = "Agric,Food ,Soda ,Beer ,Smoke,Toys ,Fun  ,Books,Hshld,Clths,Hlth ,MedEq,Drugs,Chems,Rubbr,Txtls,BldMt,Cnstr,Steel,FabPr,Mach ,ElcEq,Autos,Aero ,Ships,Guns ,Gold ,Mines,Coal ,Oil  ,Util ,Telcm,PerSv,BusSv,Hardw,Softw,Chips,LabEq,Paper,Boxes,Trans,Whlsl,Rtail,Meals,Banks,Insur,RlEst,Fin  ,Other".replace(" ", "").split(",")

def load_model(
    algo: QCAlgorithm,
    url: str,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    base64_str = algo.download(url)
    base64_bytes = base64_str.encode("ascii")
    model = base64.b64decode(base64_bytes)
    final_save = pickle.loads(model)

    model_state_dict = final_save["model_state_dict"]
    model_params = final_save["model_params"]
    input_size = final_save["input_size"]
    output_size = final_save["output_size"]

    model = SimpleLSTM(input_size, output_size, params=model_params)
    model.load_state_dict(model_state_dict)
    model.to(device)

    algo.Log(f"Model loaded successfully on device: {device}")
    algo.Log(f"Model: {model}")

    return model

def prepare_X(industry_returns):
    prior_days = len(industry_returns[list(industry_returns.keys())[0]])
    num_industries = len(industry_returns.keys())
    
    X = np.zeros(shape=(1, prior_days, num_industries))
    for industry in FF_COLS:
        X[0, :, FF_COLS.index(industry)] = industry_returns[industry]
    
    return torch.tensor(X, dtype=torch.float32)

def predict(
    model,
    industry_returns,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    mean=None,
    std=None,
):
    ### DOESN"T SEEM CORRECT -> don't see a data object
    # def normalize_column(column):
    #     # check if column name is date
    #     if column.name == "date":
    #         return column

    #     # Exclude the value V from calculations
    #     filtered_values = column[column != invalid_val]
    #     mean = filtered_values.mean()
    #     std = filtered_values.std()

    #     if std <= 0:
    #         raise ValueError(f"Standard deviation {std} is zero or negative")

    #     column = column.apply(lambda x: (x - mean) / std if x != invalid_val else x)

    #     return column

    # if mean is not None and std is not None:
    #     data = data.apply(normalize_column)

    #     filtered_values = data[data != invalid_val]
    #     mean = filtered_values.mean()
    #     std = filtered_values.std()

    model.to(device)
    X = prepare_X(industry_returns)

    with torch.no_grad():
        pred = model(X).detach().cpu().numpy()

    if mean is not None and std is not None:
        # Drop date
        mean = mean.iloc[1:]
        std = std.iloc[1:]

        # Numerize index
        mean.index = range(len(mean))
        std.index = range(len(std))
        
        pred = pred * std + mean

    return pred

