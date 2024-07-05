# region imports
from AlgorithmImports import *
import torch
import numpy as np
from model import SimpleLSTM
import pickle
import base64
from constants import PRIOR_DAYS, LOGGING
# endregion

# Gives correct order of FF industries as they are in the model
FF_COLS = "Agric,Food ,Soda ,Beer ,Smoke,Toys ,Fun  ,Books,Hshld,Clths,Hlth ,MedEq,Drugs,Chems,Rubbr,Txtls,BldMt,Cnstr,Steel,FabPr,Mach ,ElcEq,Autos,Aero ,Ships,Guns ,Gold ,Mines,Coal ,Oil  ,Util ,Telcm,PerSv,BusSv,Hardw,Softw,Chips,LabEq,Paper,Boxes,Trans,Whlsl,Rtail,Meals,Banks,Insur,RlEst,Fin  ,Other".replace(" ", "").split(",")

def load_model(
    algo: QCAlgorithm,
    url: str = None,
    model_key: str = None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    if algo.object_store.contains_key(model_key):
        base64_str = algo.object_store.read(model_key)
    else:
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

def prepare_X_LSTM(algo: QCAlgorithm, industry_returns):
    '''
    Args:
       industry_returns: np.array of shape (num_industries, PRIOR_DAYS) with the oldest day's return at index 0
    '''
    if LOGGING:
        algo.log(f"model_utils.prepare_X() called at {algo.time}.")
    if len(industry_returns) == 0:
        algo.log(f"Warning. No industry returns to prepare X with.")
        return torch.tensor(np.zeros(shape=(1, PRIOR_DAYS, len(FF_COLS))), dtype=torch.float32)
    
    num_industries = len(FF_COLS)
    X = np.zeros(shape=(1, PRIOR_DAYS, num_industries))

    for ind_idx in range(num_industries):
        X[0, :, ind_idx] = industry_returns[ind_idx]
    
    return torch.tensor(X, dtype=torch.float32)

def predict(
    algo: QCAlgorithm,
    model,
    industry_returns,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    mean=None,
    std=None,
):
    '''
    Args:
       industry_returns: np.array of shape (num_industries, PRIOR_DAYS) with the oldest day's return at index 0
    '''
    model.to(device)
    X = prepare_X_LSTM(algo, industry_returns)

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

