import torch
import torch.optim as optim
from models import SimpleLSTM, SimpleFeedForwardNN
from xgboost import XGBRegressor, XGBRFRegressor
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from data import get_data
from utils import predictions_to_returns, io_day_n_lag
import pandas as pd


def save_checkpoint(
    model,
    optimizer,
    epoch,
    train_losses,
    val_losses,
    lr,
    batch_size,
    weighting_type,
    path,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "lr": lr,
        "batch_size": batch_size,
        "weighting_type": weighting_type,
    }
    try:
        torch.save(checkpoint, path)
        print(f"Checkpoint saved at {path}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")


def load_checkpoint_from_path(
    path, input_size, output_size, model_type=SimpleLSTM, model_params=None
):
    print(f"Loading checkpoint from {path}.pt")
    checkpoint = torch.load(path)
    model = model_type(input_size, output_size, params=model_params)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = optim.Adam(model.parameters(), lr=checkpoint["lr"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    train_losses = checkpoint["train_losses"]
    val_losses = checkpoint["val_losses"]

    print(f"Checkpoint loaded. ")

    return model, optimizer, train_losses, val_losses


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


def load_model_xgb(path):
    final_save = torch.load(path)

    model = final_save["model"]

    return model


def load_model(path):
    final_save = torch.load(path)

    model_type = final_save["model_type"]
    if model_type == XGBRegressor:
        return load_model_xgb(path)
    elif model_type == SimpleLSTM or model_type == SimpleFeedForwardNN:
        return load_model_custom(path)


def preprocess_data_LSTM(
    data, io_fn, seq_len, val_start_date=None, val_end_date=None, val_size_split=0.2
):
    X, y = io_fn(data, seq_len)

    if val_start_date and val_end_date:
        val_start_idx = y[y["date"] >= int(val_start_date)].index[0]
        val_end_idx = y[y["date"] > int(val_end_date)].index[0]

        X_train = X.iloc[:val_start_idx]
        y_train = y.iloc[:val_start_idx]
        X_val = X.iloc[val_start_idx:val_end_idx]
        y_val = y.iloc[val_start_idx:val_end_idx]
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_size_split, shuffle=False
        )

    X_train = X_train.drop(columns=[f"date_lag{i}" for i in range(1, seq_len + 1)])
    y_train = y_train.drop(columns=["date"])
    X_val = X_val.drop(columns=[f"date_lag{i}" for i in range(1, seq_len + 1)])
    y_val = y_val.drop(columns=["date"])

    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    y_train = y_train.apply(pd.to_numeric, errors="coerce")
    X_val = X_val.apply(pd.to_numeric, errors="coerce")
    y_val = y_val.apply(pd.to_numeric, errors="coerce")

    # Convert X_train of shape (n_samples, n_features) to (n_samples, seq_len, n_features)
    X_train = X_train.values.reshape(
        X_train.shape[0], seq_len, X_train.shape[1] // seq_len
    )[:, ::-1, :]
    y_train = y_train.values
    X_val = X_val.values.reshape(X_val.shape[0], seq_len, X_val.shape[1] // seq_len)[
        :, ::-1, :
    ]
    y_val = y_val.values

    X_train = np.ascontiguousarray(X_train)
    X_val = np.ascontiguousarray(X_val)

    val_start_date = int(data.iloc[len(X_train) + seq_len]["date"])

    return X_train, y_train, X_val, y_val, val_start_date


def preprocess_data_flat(
    data, io_fn, seq_len, val_start_date=None, val_end_date=None, val_size_split=0.2
):
    X, y = io_fn(data, seq_len)

    if val_start_date and val_end_date:
        val_start_idx = y[y["date"] >= int(val_start_date)].index[0]
        val_end_idx = y[y["date"] > int(val_end_date)].index[0]

        X_train = X.iloc[:val_start_idx]
        y_train = y.iloc[:val_start_idx]
        X_val = X.iloc[val_start_idx:val_end_idx]
        y_val = y.iloc[val_start_idx:val_end_idx]
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_size_split, shuffle=False
        )

    X_train = X_train.drop(columns=[f"date_lag{i}" for i in range(1, seq_len + 1)])
    y_train = y_train.drop(columns=["date"])
    X_val = X_val.drop(columns=[f"date_lag{i}" for i in range(1, seq_len + 1)])
    y_val = y_val.drop(columns=["date"])

    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    y_train = y_train.apply(pd.to_numeric, errors="coerce")
    X_val = X_val.apply(pd.to_numeric, errors="coerce")
    y_val = y_val.apply(pd.to_numeric, errors="coerce")

    X_train = X_train.values
    y_train = y_train.values
    X_val = X_val.values
    y_val = y_val.values

    val_start_date = int(data.iloc[len(X_train) + seq_len]["date"])

    return X_train, y_train, X_val, y_val, val_start_date


def chosen_strats_LSTM(save_path, data_path, io_fn, seq_len):
    data = pd.read_csv(data_path)
    data = data.dropna()

    _, _, X_val, y_val, _ = preprocess_data_LSTM(data, io_fn, seq_len)

    model = load_model(save_path)
    pred_df = predict_LSTM(model, X_val)

    strat_df, _ = predictions_to_returns(pred_df, y_val)
    strat_df.columns = data.columns[1:]

    return strat_df


def chosen_strats_xgb(save_path, data_path, io_fn, seq_len):
    data = pd.read_csv(data_path)
    data = data.dropna()

    _, _, X_val, y_val, _ = preprocess_data_flat(data, io_fn, seq_len)

    model = load_model(save_path)
    pred_df = predict_xgboost(model, X_val)

    strat_df, _ = predictions_to_returns(pred_df, y_val)
    strat_df.columns = data.columns[1:]

    return strat_df


def model_strat_distribution(
    model_path: str, data_path, data_weighing=None, seq_len=5, io_fn=io_day_n_lag
):
    idx = model_path.find("industries")
    if idx != -1:
        strat = model_path[idx - 2 : idx]  # 5, 10, 12, 17, 30, 38, 48, 49
        data_weighing = model_path[idx + 11 : idx + 16]  # equal or value
        data_path = all_data_paths[f"{strat}industries_{data_weighing}"]
    else:
        strat = model_path[-6:-3]  # ff3 or ff6
        data_path = all_data_paths[strat]

    data, all_data_paths = get_data(strat, data_weighing)

    if "LSTM" in model_path:
        model = load_model(model_path)
        strat_df = chosen_strats_LSTM(
            model_path, data_path, seq_len=seq_len, io_fn=io_fn
        )
    elif "XGB" in model_path:
        model = load_model(model_path)
        strat_df = chosen_strats_xgb(
            model_path, data_path, seq_len=seq_len, io_fn=io_fn
        )
    else:
        raise ValueError("Model path must contain 'LSTM' or 'XGB'")

    print(
        "====================================================================================================="
    )
    print(f"Model: {model_path}")
    print(
        "====================================================================================================="
    )

    # create a dictionary from column names of strat_df
    strat_dist = {col: 0 for col in strat_df.columns}
    num_rows = len(strat_df)

    for row in strat_df.iterrows():
        # get the column name of the max value in the row
        max_strat_name = row[1][row[1] == True].index[0]
        strat_dist[max_strat_name] += 1

    strat_dist = dict(
        sorted(strat_dist.items(), key=lambda item: item[1], reverse=True)
    )

    print("Most chosen strategies:")
    for key in strat_dist.keys():
        print(f"{key}: {strat_dist[key]/num_rows*100:.0f}%", end=", ")
    print()
    print()

    return strat_dist


def predict_LSTM(
    model,
    X_val,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    mean=None,
    std=None,
):
    model.to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(
        np.zeros((X_val.shape[0], X_val.shape[-1])), dtype=torch.float32  # dummy y_val
    )

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=32)

    with torch.no_grad():
        pred_df = torch.tensor([])
        for X_val_batch, _ in val_loader:
            X_val_batch = X_val_batch.to(device)

            val_outputs = model(X_val_batch).detach().cpu()

            pred_df = torch.cat((pred_df, val_outputs), 0)

        pred_df = pd.DataFrame(pred_df.numpy())

    # Drop date
    mean = mean.iloc[1:]
    std = std.iloc[1:]

    # Numerize index
    mean.index = range(len(mean))
    std.index = range(len(std))
    
    if mean is not None and std is not None:
        pred_df = pred_df * std + mean

    return pred_df


def predict_xgboost(model, X_val, mean=None, std=None):
    pred_df = pd.DataFrame(model.predict(X_val))

    # Drop date
    mean = mean.iloc[1:]
    std = std.iloc[1:]

    # Numerize index
    mean.index = range(len(mean))
    std.index = range(len(std))
    
    if mean is not None and std is not None:
        pred_df = pred_df * std + mean

    return pred_df


def predict_FFNN(
    model,
    X_val,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    mean=None,
    std=None,
):
    model.to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(
        np.zeros(X_val.shape[0]), dtype=torch.float32
    )  # dummy y_val

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=32)

    with torch.no_grad():
        pred_df = torch.tensor([])
        for X_val_batch, _ in val_loader:
            X_val_batch = X_val_batch.to(device)

            val_outputs = model(X_val_batch).detach().cpu()

            pred_df = torch.cat((pred_df, val_outputs), 0)

        pred_df = pd.DataFrame(pred_df.numpy())

    # Drop date
    mean = mean.iloc[1:]
    std = std.iloc[1:]

    # Numerize index
    mean.index = range(len(mean))
    std.index = range(len(std))

    if mean is not None and std is not None:
        pred_df = pred_df * std + mean

    return pred_df


def predict(
    model,
    X_val,
    model_type,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    mean=None,
    std=None,
):
    if model_type == SimpleLSTM:
        return predict_LSTM(model, X_val, device, mean, std)
    elif model_type == SimpleFeedForwardNN:
        return predict_FFNN(model, X_val, device, mean, std)
    elif model_type == XGBRegressor or model_type == XGBRFRegressor:
        return predict_xgboost(model, X_val, mean, std)
    else:
        raise ValueError("Model type not recognized.")
