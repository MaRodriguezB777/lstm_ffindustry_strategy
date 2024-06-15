import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from xgboost import XGBRegressor, XGBRFRegressor
import warnings
from models import SimpleLSTM, MultiLayerLSTM, SimpleFeedForwardNN
from model_util import (
    save_checkpoint,
    load_checkpoint_from_path,
    load_model,
    preprocess_data_LSTM,
    preprocess_data_flat,
    predict,
    predictions_to_returns,
)
from utils import set_seed, io_day_n_lag
from data import get_data, load_ff3, load_ff6
import json

warnings.filterwarnings("ignore")

RANGES = {
    "bullish": [
        (1980, 1980),
        (
            1982,
            1986,
        ),  # After the recession in the early 1980s, the market entered a bullish phase, culminating in the 1987 peak before the crash.
        (1988, 1989),  # Russian default on debt
        (
            1991,
            1999,
        ),  # This period, often referred to as the "Dot-com Bubble," was a significant bull market driven by rapid growth in technology stocks.
        (
            2003,
            2007,
        ),  # The market experienced a significant bull run during this period, driven by strong economic growth and low interest rates.
        (
            2009,
            2017,
        ),  # The market recovered from the 2008 financial crisis and entered a long bull market, driven by low interest rates and economic growth.
        (2019, 2019),
        (
            2021,
            2021,
        ),  # The market recovered from the COVID-19 pandemic and entered a bull market, driven by low interest rates and economic recovery.
        (
            2023,
            2023,
        ),  # The market recovered from the COVID-19 pandemic and entered a bull market, driven by low interest rates and economic recovery.
    ],
    "bearish": [
        (
            1981,
            1981,
        ),  # The early 1980s recession was a significant bear market, culminating in the 1982 market bottom. High inflation and interest rates drove the downturn.
        (1987, 1987),  # Black Monday
        (
            1990,
            1990,
        ),  # The early 1990s recession was a significant bear market, driven by the Gulf War and economic slowdown.
        (
            2000,
            2002,
        ),  # The early 2000s recession was a significant bear market, driven by the bursting of the Dot-com Bubble and economic slowdown.
        (
            2008,
            2008,
        ),  # The 2008 financial crisis was a significant bear market, driven by the subprime mortgage crisis and financial system collapse.
        (
            2018,
            2018,
        ),  # Donald Trump's trade war with China, the slowdown in global economic growth and concern that the Federal Reserve was raising interest rates too quickly
        (
            2020,
            2020,
        ),  # The COVID-19 pandemic led to a significant bear market in 2020, driven by economic shutdowns and uncertainty.
        (
            2022,
            2022,
        ),  # The market experienced a significant bear market in 2022, driven by concerns about inflation and interest rates.
    ],
}


def is_bullish_or_bearish(curr_year):
    for key, value in RANGES.items():
        for start, end in value:
            if curr_year >= start and curr_year <= end:
                return key
    return None


def train_loop(
    model,
    train_loader,
    val_loader=None,
    optimizer=None,
    loss_fn=nn.MSELoss(),
    lr=1e-3,
    batch_size=32,
    start_epoch=1,
    num_epochs=50,
    logging=True,
    save_epochs=None,
    checkpoints_dir=None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    if optimizer == None:
        optimizer = optimizer(model.parameters(), lr=lr)

    model.to(device)

    train_losses = []
    val_losses = []
    best_model = None
    best_epoch = 0

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        train_loss = 0
        for X_batch, y_batch in tqdm(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)

            mask = y_batch != -99.99
            y_batch = y_batch[mask]
            outputs = outputs[mask]

            if loss_fn == nn.CrossEntropyLoss():
                loss = loss_fn(outputs, y_batch.softmax(dim=1))
            else:
                loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(
                    device
                )

                val_outputs = model(X_val_batch)
                if loss_fn == nn.CrossEntropyLoss():
                    loss = loss_fn(val_outputs, y_val_batch.softmax(dim=1))
                else:
                    loss = loss_fn(val_outputs, y_val_batch)

                val_loss += loss.item()
            val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if logging:
            print(f"Epoch [{epoch}/{num_epochs}]")
            print(
                f"\tTraining Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f} (std: {np.std(val_losses):.4f})"
            )
        elif save_epochs and epoch % save_epochs == 0 and not logging:
            print(f"Epoch [{epoch}/{num_epochs}]")
            print(
                f"\tTraining Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f} (std: {np.std(val_losses):.4f})"
            )

        if save_epochs and epoch % save_epochs == 0:
            weighting_type = "equal"
            save_checkpoint(
                model,
                optimizer,
                epoch,
                train_losses,
                val_losses,
                lr,
                batch_size,
                weighting_type,
                f"{checkpoints_dir}/epoch{epoch}.pt",
            )

        # if val_loss == min(val_losses) or best_model == None:
        #     best_model = model
        #     best_epoch = epoch

    #     print()

    # print(f"Best model at epoch {best_epoch} with validation loss {min(val_losses)}")

    return model, train_losses, val_losses


def train_model_FFNN(
    X_train,
    y_train,
    X_val,
    y_val,
    num_epochs=50,
    batch_size=32,
    logging=True,
    save_epochs=None,
    final_save_path=None,
    model_params=None,
    model_type=SimpleFeedForwardNN,
):
    # Prapare datasets
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Calculate weights for each datapoint (closer to the end of the dataset, the more weight it has)
    def calc_weights():
        weights = torch.ones(len(y_train))
        for i in range(len(y_train)):
            weights[i] += 0.5 * (i / len(y_train)) ** 2
        weights = weights / weights.sum()
        return weights

    # weights = calc_weights()
    weights = torch.ones(len(y_train))

    # Define LSTM model
    input_size = X_train.shape[-1]  # Number of features
    output_size = y_train.shape[-1]  # Number of output features

    model = SimpleFeedForwardNN(
        input_size=input_size, output_size=output_size, params=model_params
    )

    # Loss and optimizer
    lr = model_params["lr"] if "lr" in model_params else 1e-3
    optimizer = (
        model_params["optimizer"](model.parameters(), lr=lr)
        if "optimizer" in model_params
        else optim.Adam(model.parameters(), lr=lr)
    )
    loss_fn = model_params["loss_fn"] if "loss_fn" in model_params else nn.MSELoss()

    model, train_losses, val_losses = train_loop(
        model,
        train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        lr=lr,
        optimizer=optimizer,
        loss_fn=loss_fn,
        logging=logging,
        save_epochs=save_epochs,
    )

    if final_save_path:
        final_save = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "lr": lr,
            "batch_size": batch_size,
            "weighting_type": "equal",
            "model_params": model_params,
            "model_type": model_type,
            "input_size": input_size,
            "output_size": output_size,
            "num_epochs": num_epochs,
        }
        torch.save(final_save, final_save_path)

    return model


def train_model_LSTM(
    X_train,
    y_train,
    X_val,
    y_val,
    num_epochs=50,
    batch_size=32,
    logging=True,
    save_epochs=None,
    final_save_path=None,
    checkpoints_dir=None,
    load_checkpoint=None,
    model_params=None,
    model_type=SimpleLSTM,
):
    if (save_epochs and not checkpoints_dir) or (checkpoints_dir and not save_epochs):
        raise ValueError(
            "Both save_epochs and checkpoints_dir must be provided if one is provided"
        )
    elif checkpoints_dir:
        os.makedirs(checkpoints_dir, exist_ok=True)

    # Prepare datasets
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Calculate weights for each datapoint (closer to the end of the dataset, the more weight it has)
    def calc_weights():
        weights = torch.ones(len(y_train))
        for i in range(len(y_train)):
            weights[i] += 0.5 * (i / len(y_train)) ** 2
        weights = weights / weights.sum()
        return weights

    # weights = calc_weights()
    weights = torch.ones(len(y_train))

    # Define LSTM model
    input_size = X_train.shape[-1]  # Number of features
    output_size = y_train.shape[-1]  # Number of output features
    start_epoch = 1
    if load_checkpoint and checkpoints_dir:
        model, optimizer, train_losses, val_losses = load_checkpoint_from_path(
            checkpoints_dir + f"/epoch{load_checkpoint}.pt",
            input_size,
            output_size,
            model_params=model_params,
        )

    elif load_checkpoint and not checkpoints_dir:
        raise ValueError(
            "If load_checkpoint is provided, checkpoints_dir must also be provided"
        )
    else:
        model = model_type(input_size, output_size, params=model_params)

        # Lr and optimizer
        lr = model_params["lr"] if "lr" in model_params else 1e-3
        optimizer = (
            model_params["optimizer"](model.parameters(), lr=lr)
            if "optimizer" in model_params
            else optim.Adam(model.parameters(), lr=lr)
        )

    loss_fn = model_params["loss_fn"] if "loss_fn" in model_params else nn.MSELoss()

    model, train_losses, val_losses = train_loop(
        model,
        train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        lr=lr,
        optimizer=optimizer,
        loss_fn=loss_fn,
        logging=logging,
        save_epochs=save_epochs,
        checkpoints_dir=checkpoints_dir,
    )

    if final_save_path:
        final_save = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "lr": lr,
            "batch_size": batch_size,
            "weighting_type": "equal",
            "model_params": model_params,
            "model_type": model_type,
            "input_size": input_size,
            "output_size": output_size,
            "num_epochs": num_epochs,
        }
        torch.save(final_save, final_save_path)

    return model


def train_model_xgb(
    X_train,
    y_train,
    X_val,
    y_val,
    final_save_path=None,
    model_type=XGBRegressor,
    device="cuda",
):

    model = model_type(objective="reg:squarederror", device=device)
    model.fit(X_train, y_train)

    print(f"Train MSE: {mean_squared_error(y_train, model.predict(X_train))}")
    print(f"Validation MSE: {mean_squared_error(y_val, model.predict(X_val))}")

    if final_save_path:
        final_save = {"model": model, "model_type": model_type}
        torch.save(final_save, final_save_path)

    return model


def evaluate_model(
    model_path,
    data_path,
    ret_port_style="max-min_LS",
    io_fn=io_day_n_lag,
    seq_len=5,
    print_summary=True,
    save=False,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    normalize=True,
):
    """
    Evaluate the model by calculating the alpha and beta values against the FF3 factors
    """
    if model_type not in [
        SimpleLSTM,
        MultiLayerLSTM,
        XGBRegressor,
        XGBRFRegressor,
        SimpleFeedForwardNN,
    ]:
        raise ValueError("Invalid model type")

    data = pd.read_csv(data_path)
    data = data.dropna()

    model = load_model(model_path)
    model_type = type(model)
    (_, _, X_val, y_val, val_start_date), mean, std = preprocess_data(
        data, io_fn, seq_len, model_type=model_type, normalize=normalize
    )

    pred_df = predict(model, X_val, model_type=model_type, mean=mean, std=std)

    _, strat_ret_vec = predictions_to_returns(
        pred_df, y_val, ret_port_style=ret_port_style
    )

    ff3 = load_ff3(val_start_date)
    strat_ret_vec.index = ff3.index

    # Calculate alpha
    y_ols = sm.add_constant(ff3[["Mkt-RF", "SMB", "HML"]])
    model_OLS = sm.OLS(strat_ret_vec, y_ols).fit()

    results = {
        "alpha": model_OLS.params["const"],
        "alpha_annual": (np.power(1 + model_OLS.params["const"] / 100, 22 * 12) - 1)
        * 100,
        "alpha_p_value": model_OLS.pvalues["const"],
        "beta": model_OLS.params["Mkt-RF"],
        "beta_p_value": model_OLS.pvalues["Mkt-RF"],
    }
    if print_summary:
        print(f"Alpha: {results['alpha']} ({results['alpha_annual']:.2f}% annually)")
        print(f"Alpha p-value: {results['alpha_p_value']:.3f}")
        print(f"Beta: {results['beta']:.3f}")
        print(f"Beta p-value: {results['beta_p_value']:.3f}\n")

    # save the results to a json file
    if save:
        os.makedirs(
            f"results/{model_path.replace('.pt', '').replace('models', ret_port_style)}",
            exist_ok=True,
        )
        with open(
            f"results/{model_path.replace('.pt', '').replace('models', ret_port_style)}/params.json",
            "w",
        ) as f:
            json.dump(results, f)

    strat_ret_vec = strat_ret_vec + ff3["RF"]
    mkt_ret_vec = ff3["Mkt-RF"] + ff3["RF"]

    return strat_ret_vec, mkt_ret_vec


def evaluate_model_rolling(
    model_dir,
    data,
    ret_port_style="max-min_LS",
    io_fn=io_day_n_lag,
    seq_len=5,
    print_summary=None,  # 0 = Nothing, 1 = Final, 2 = All Years
    save=False,
    initial_train_end_year=1979,
    final_val_end_year=2024,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    normalize=True,
):

    total_returns = {}
    bearish_returns = {}
    bullish_returns = {}

    market_total_returns = {}
    market_bearish_returns = {}
    market_bullish_returns = {}

    total_analyses = {}
    bullish_analyses = {}
    bearish_analyses = {}

    print(f"{ret_port_style} metrics for {model_dir.replace('models/', '')}")

    for curr_year in range(initial_train_end_year + 1, final_val_end_year):
        model_path = f"{model_dir}/{curr_year}.pt"
        model = load_model(model_path)
        model_type = type(model)

        (_, _, X_val, y_val, val_start_date), mean, std = preprocess_data(
            data,
            io_fn,
            seq_len,
            model_type,
            val_start_date=f"{curr_year}0101",
            val_end_date=f"{curr_year+1}0101",
            normalize=normalize,
        )

        pred_df = predict(model, X_val, model_type=model_type, mean=mean, std=std)

        _, strat_ret_vec = predictions_to_returns(
            pred_df, y_val, ret_port_style=ret_port_style
        )

        ff6 = load_ff6(val_start_date, val_end_date=f"{curr_year+1}0101")
        # print(ff6.shape, strat_ret_vec.shape, val_start_date, f'{curr_year+1}0101', ff6.index[0], strat_ret_vec.index[0])
        strat_ret_vec.index = ff6.index

        # # Convert index to datetime
        # strat_ret_vec_monthly = strat_ret_vec.copy()
        # ff6_monthly = ff6.copy()

        # strat_ret_vec_monthly.index = pd.to_datetime(strat_ret_vec_monthly.index, format="%Y%m%d")
        # ff6_monthly.index = pd.to_datetime(ff6_monthly.index, format="%Y%m%d")

        # # cumulative product for monthly
        # ff6_monthly = ff6_monthly.resample('M').apply(lambda x: (x/100 + 1).prod() - 1)
        # strat_ret_vec_monthly = strat_ret_vec_monthly.resample('M').apply(lambda x: (x/100 + 1).prod() - 1)
        # print(ff6_monthly)

        # Calculate alpha
        y_ols = sm.add_constant(ff6[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]])
        model_OLS = sm.OLS(strat_ret_vec, y_ols).fit()

        results = {
            "alpha": model_OLS.params["const"],
            "alpha_annual": (np.power(1 + model_OLS.params["const"] / 100, 22 * 12) - 1)
            * 100,
            "alpha_p_value": model_OLS.pvalues["const"],
            "beta": model_OLS.params["Mkt-RF"],
            "beta_p_value": model_OLS.pvalues["Mkt-RF"],
        }
    
        if print_summary == 2:
            print(f"Year: {curr_year}")
            print(
                f"\tAlpha: {results['alpha']} ({results['alpha_annual']:.2f}% annually)"
            )
            print(f"\tAlpha p-value: {results['alpha_p_value']:.3f}")
            print(f"\tBeta: {results['beta']:.3f}")
            print(f"\tBeta p-value: {results['beta_p_value']:.3f}\n")

        total_returns[curr_year] = strat_ret_vec
        market_total_returns[curr_year] = ff6
        total_analyses[curr_year] = results

        bullish_or_bearish = is_bullish_or_bearish(curr_year)

        if bullish_or_bearish == "bullish":
            bullish_returns[curr_year] = strat_ret_vec
            market_bullish_returns[curr_year] = ff6
            bullish_analyses[curr_year] = results
        elif bullish_or_bearish == "bearish":
            bearish_returns[curr_year] = strat_ret_vec
            market_bearish_returns[curr_year] = ff6
            bearish_analyses[curr_year] = results
        else:
            raise ValueError(f"Invalid bullish_or_bearish value: {bullish_or_bearish}")

    total_returns = pd.concat(list(total_returns.values()))
    bullish_returns = pd.concat(list(bullish_returns.values()))
    bearish_returns = pd.concat(list(bearish_returns.values()))

    market_total_returns = pd.concat(list(market_total_returns.values()))
    market_bullish_returns = pd.concat(list(market_bullish_returns.values()))
    market_bearish_returns = pd.concat(list(market_bearish_returns.values()))

    total_OLS = sm.OLS(
        total_returns,
        sm.add_constant(
            market_total_returns[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]]
        ),
    ).fit()

    bullish_OLS = sm.OLS(
        bullish_returns,
        sm.add_constant(
            market_bullish_returns[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]]
        ),
    ).fit()

    bearish_OLS = sm.OLS(
        bearish_returns,
        sm.add_constant(
            market_bearish_returns[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]]
        ),
    ).fit()

    total_avg_analysis = {
        "alpha": total_OLS.params["const"],
        "alpha_annual": (np.power(1 + total_OLS.params["const"] / 100, 22 * 12) - 1)
        * 100,
        "alpha_p_value": total_OLS.pvalues["const"],
        "beta": total_OLS.params["Mkt-RF"],
        "beta_p_value": total_OLS.pvalues["Mkt-RF"],
    }
    bullish_avg_analysis = {
        "alpha": bullish_OLS.params["const"],
        "alpha_annual": (np.power(1 + bullish_OLS.params["const"] / 100, 22 * 12) - 1)
        * 100,
        "alpha_p_value": bullish_OLS.pvalues["const"],
        "beta": bullish_OLS.params["Mkt-RF"],
        "beta_p_value": bullish_OLS.pvalues["Mkt-RF"],
    }
    bearish_avg_analysis = {
        "alpha": bearish_OLS.params["const"],
        "alpha_annual": (np.power(1 + bearish_OLS.params["const"] / 100, 22 * 12) - 1)
        * 100,
        "alpha_p_value": bearish_OLS.pvalues["const"],
        "beta": bearish_OLS.params["Mkt-RF"],
        "beta_p_value": bearish_OLS.pvalues["Mkt-RF"],
    }

    if print_summary >= 1:
        print("=============================================================")
        print("Total Average Analysis")
        print(
            f"\tAlpha: {total_avg_analysis['alpha']} ({total_avg_analysis['alpha_annual']:.2f}% annually)"
        )
        print(f"\tAlpha p-value: {total_avg_analysis['alpha_p_value']:.3f}")
        print(f"\tBeta: {total_avg_analysis['beta']:.3f}")
        print(f"\tBeta p-value: {total_avg_analysis['beta_p_value']:.3f}\n")

        print("Bullish Average Analysis")
        print(
            f"\tAlpha: {bullish_avg_analysis['alpha']} ({bullish_avg_analysis['alpha_annual']:.2f}% annually)"
        )
        print(f"\tAlpha p-value: {bullish_avg_analysis['alpha_p_value']:.3f}")
        print(f"\tBeta: {bullish_avg_analysis['beta']:.3f}")

        print("Bearish Average Analysis")
        print(
            f"\tAlpha: {bearish_avg_analysis['alpha']} ({bearish_avg_analysis['alpha_annual']:.2f}% annually)"
        )
        print(f"\tAlpha p-value: {bearish_avg_analysis['alpha_p_value']:.3f}")
        print(f"\tBeta: {bearish_avg_analysis['beta']:.3f}")
        print("=============================================================")

    if save:
        results_dir = model_dir.replace("models", "results") + f"/{ret_port_style}"
        os.makedirs(results_dir, exist_ok=True)

        with open(f"{results_dir}/params_total.json", "w") as f:
            json.dump(total_avg_analysis, f)

        with open(f"{results_dir}/params_bullish.json", "w") as f:
            json.dump(bullish_avg_analysis, f)

        with open(f"{results_dir}/params_bearish.json", "w") as f:
            json.dump(bearish_avg_analysis, f)

    ff6 = load_ff6(f"{initial_train_end_year+1}0101", f"{final_val_end_year}0101")
    total_strat_ret_vec = total_returns + ff6["RF"]

    mkt_ret_vec = ff6["Mkt-RF"] + ff6["RF"]

    return total_strat_ret_vec, mkt_ret_vec


def show_cumulative_returns(
    strat_ret_vec,
    mkt_ret_vec,
    ret_port_style,
    strategy_name="Strategy",
    log_scale=False,
    show_plots=False,
    save=False,
):
    cum_strat_rets = (1 + strat_ret_vec / 100).cumprod()
    cum_mkt_rets = (1 + mkt_ret_vec / 100).cumprod()

    cum_mkt_rets.index = pd.to_datetime(cum_mkt_rets.index, format="%Y%m%d")
    cum_mkt_rets_monthly = cum_mkt_rets.resample("M").last()

    cum_strat_rets.index = pd.to_datetime(cum_strat_rets.index, format="%Y%m%d")
    cum_strat_rets_monthly = cum_strat_rets.resample("M").last()

    if show_plots:
        fig, axs = plt.subplots(1, 2, figsize=(10, 6))
        # Plot the cumulative returns
        axs[0].plot(cum_strat_rets_monthly, label=strategy_name)
        axs[0].plot(cum_mkt_rets_monthly, label="Market")

        for status, range in RANGES.items():
            color = "green" if status == "bullish" else "red"
            for start, end in range:
                start_date = pd.to_datetime(f"{start}0101", format="%Y%m%d")
                end_date = pd.to_datetime(f"{end}1231", format="%Y%m%d")
                axs[0].axvspan(start_date, end_date, color=color, alpha=0.3)

        if log_scale:
            axs[0].yscale("log")
        axs[0].set_xlabel("Year")
        axs[0].set_ylabel("Cumulative Return")
        axs[0].legend()
        axs[0].set_title(f"Returns Portfolio Style: {ret_port_style}")

        strat_ret_vec_monthly = strat_ret_vec.copy()
        strat_ret_vec_monthly.index = pd.to_datetime(strat_ret_vec_monthly.index, format="%Y%m%d")
        strat_ret_vec_monthly = strat_ret_vec_monthly.resample('Y').apply(lambda x: (x/100 + 1).prod() - 1)
        
        mkt_ret_vec_monthly = mkt_ret_vec.copy()
        mkt_ret_vec_monthly.index = pd.to_datetime(mkt_ret_vec_monthly.index, format="%Y%m%d")
        mkt_ret_vec_monthly = mkt_ret_vec_monthly.resample('Y').apply(lambda x: (x/100 + 1).prod() - 1)
        
        x = np.arange(len(strat_ret_vec_monthly.index))
        
        num_ticks = 7
        tick_positions = np.linspace(0, len(x) - 1, num_ticks, dtype=int)
        tick_labels = strat_ret_vec_monthly.index[tick_positions].strftime('%Y')

        barwidth = 0.2
        
        axs[1].bar(x - barwidth/2, strat_ret_vec_monthly.values, label=strategy_name, width=barwidth )
        axs[1].bar(x + barwidth/2, mkt_ret_vec_monthly.values, label="Market", width=barwidth)
        axs[1].set_xlabel("Year")
        axs[1].set_ylabel("Yearly Return")
        axs[1].legend()
        axs[1].set_title(f"Returns Portfolio Style: {ret_port_style}")
        axs[1].grid(axis='y', linestyle='--', alpha=0.7)
        
        axs[1].set_xticks(tick_positions)
        axs[1].set_xticklabels(tick_labels, rotation=90)

        plt.title(f"Returns Portfolio Style: {ret_port_style}")
        
        # Save plot
        if save:
            name = strategy_name.replace("models", "")
            os.makedirs(f"results/{name}/{ret_port_style}", exist_ok=True)
            plt.savefig(f"results/{name}/{ret_port_style}/cumulative_returns.png")

        plt.show()

    # Save the plot
    if save:
        name = strategy_name.replace("models", "")
        os.makedirs(f"results/{name}/{ret_port_style}", exist_ok=True)

        # save cum_strat_rets in csv
        cum_strat_rets.to_frame().to_csv(
            f"results/{name}/{ret_port_style}/cumulative_returns.csv"
        )

        # save strat_ret_vecs in csv
        strat_ret_vec.to_frame().to_csv(
            f"results/{name}/{ret_port_style}/strat_ret_vec.csv"
        )


def preprocess_data(
    data,
    io_fn,
    seq_len,
    model_type,
    val_start_date=None,
    val_end_date=None,
    normalize=True,
):
    invalid_val = -99.99

    def normalize_column(column):
        # check if column name is date
        if column.name == "date":
            return column

        # Exclude the value V from calculations
        filtered_values = column[column != invalid_val]
        mean = filtered_values.mean()
        std = filtered_values.std()

        if std <= 0:
            raise ValueError(f"Standard deviation {std} is zero or negative")

        column = column.apply(lambda x: (x - mean) / std if x != invalid_val else x)

        return column

    if normalize:
        data = data.apply(normalize_column)

        filtered_values = data[data != invalid_val]
        mean = filtered_values.mean()
        std = filtered_values.std()

    if model_type == SimpleLSTM or model_type == MultiLayerLSTM:
        return (
            preprocess_data_LSTM(data, io_fn, seq_len, val_start_date, val_end_date),
            mean if normalize else None,
            std if normalize else None,
        )
    elif (
        model_type == XGBRegressor
        or model_type == XGBRFRegressor
        or model_type == SimpleFeedForwardNN
    ):
        return (
            preprocess_data_flat(data, io_fn, seq_len, val_start_date, val_end_date),
            mean if normalize else None,
            std if normalize else None,
        )
    else:
        raise ValueError("Invalid model type")


def train_model(
    model_type,
    X_train,
    y_train,
    X_val,
    y_val,
    num_epochs=50,
    batch_size=32,
    logging=True,
    save_epochs=None,
    final_save_path=None,
    checkpoints_dir=None,
    load_checkpoint=None,
    model_params=None,
):
    if model_type == SimpleLSTM or model_type == MultiLayerLSTM:
        return train_model_LSTM(
            X_train,
            y_train,
            X_val,
            y_val,
            num_epochs=num_epochs,
            batch_size=batch_size,
            logging=logging,
            save_epochs=save_epochs,
            final_save_path=final_save_path,
            checkpoints_dir=checkpoints_dir,
            load_checkpoint=load_checkpoint,
            model_params=model_params,
            model_type=model_type,
        )
    elif model_type == XGBRegressor or model_type == XGBRFRegressor:
        return train_model_xgb(
            X_train,
            y_train,
            X_val,
            y_val,
            final_save_path=final_save_path,
            model_type=model_type,
        )
    elif model_type == SimpleFeedForwardNN:
        return train_model_FFNN(
            X_train,
            y_train,
            X_val,
            y_val,
            num_epochs=num_epochs,
            batch_size=batch_size,
            logging=logging,
            save_epochs=save_epochs,
            final_save_path=final_save_path,
            model_params=model_params,
            model_type=model_type,
        )
    else:
        print(f"model_type: {model_type}")
        raise ValueError(f"Invalid model type: {model_type}")


def train_all_models_rolling(
    model_types,
    industries_list,
    data_weighing_types=["value"],
    num_epochs=50,
    batch_size=32,
    logging=False,
    save_epochs=None,
    models_main_dir=None,
    checkpoints_main_dir=None,
    model_params=None,
    seed=519,
    io_fn=io_day_n_lag,
    seq_len=5,
    initial_train_end_year=1979,
    final_val_end_year=2024,
    normalize=True,
):
    for num_industries in industries_list:
        for data_weighing in data_weighing_types:
            # there is no 'equal' vs 'value' for ff3 and ff6
            if num_industries in ["ff3", "ff6"] and data_weighing == "value":
                pass
            data, _ = get_data(
                seed=seed,
                num_industries=num_industries,
                weight_type=data_weighing,
            )
            for curr_year in range(initial_train_end_year + 1, final_val_end_year):
                for model_type in model_types:
                    print(
                        f"Training {model_type.__name__} with {seq_len}-day lag on {num_industries} industries with {data_weighing}-weighted portfolio for year {curr_year}"
                    )

                    (X_train, y_train, X_val, y_val, val_start_date), mean, std = (
                        preprocess_data(
                            data,
                            io_fn,
                            seq_len,
                            model_type,
                            val_start_date=f"{curr_year}0101",
                            val_end_date=f"{curr_year+1}0101",
                            normalize=normalize,
                        )
                    )

                    model_file_name = f"{curr_year}.pt"

                    if models_main_dir:
                        model_dir = f"{models_main_dir}/{seq_len}seq_len/{model_type.__name__}/{num_industries}industries/{data_weighing}/{str(num_epochs) + 'epochs/' if model_type != XGBRegressor else ''}"
                        os.makedirs(model_dir, exist_ok=True)

                        final_save_path = f"{model_dir}/{model_file_name}"

                    if checkpoints_main_dir:
                        checkpoints_dir = f"{checkpoints_main_dir}/{model_file_name.replace('.pt', '')}"

                    model = train_model(
                        model_type,
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        num_epochs,
                        batch_size,
                        logging=logging,
                        save_epochs=save_epochs,
                        final_save_path=final_save_path if models_main_dir else None,
                        checkpoints_dir=(
                            checkpoints_dir if checkpoints_main_dir else None
                        ),
                        model_params=model_params,
                    )

                    if models_main_dir:
                        print(
                            f"Training complete. Saved {model_file_name} to {final_save_path}"
                        )


def train_all_models(
    model_types,
    industries_list,
    data_weighing_types=["value", "equal"],
    num_epochs=50,
    batch_size=32,
    logging=False,
    save_epochs=None,
    models_main_dir=None,
    checkpoints_dir=None,
    model_params=None,
    seed=519,
    io_fn=io_day_n_lag,
    seq_len=5,
    normalize=True,
):
    for model_type in model_types:
        for num_industries in industries_list:
            for data_weighing in data_weighing_types:
                # there is no 'equal' vs 'value' for ff3 and ff6
                if num_industries in ["ff3", "ff6"] and len(data_weighing_types) > 1:
                    continue
                data, _ = get_data(
                    seed=seed,
                    num_industries=num_industries,
                    weight_type=data_weighing,
                )
                (X_train, y_train, X_val, y_val, val_start_date), mean, std = (
                    preprocess_data(
                        data, io_fn, seq_len, model_type, normalize=normalize
                    )
                )
                print("X_train shape:\t", X_train.shape)
                print("y_train shape:\t", y_train.shape)
                print("X_val shape:\t", X_val.shape)
                print("y_val shape:\t", y_val.shape)
                print("val_start_date:\t", val_start_date)

                print(
                    f"Training {model_type} on {num_industries} industries with {data_weighing}-weighted portfolio"
                )

                model_file_name = (
                    f"{model_type.__name__}_{seq_len}days_{num_industries}"
                )
                if num_industries not in ["ff3", "ff6"]:
                    model_file_name += f"industries_{data_weighing}"

                model = train_model(
                    model_type,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    num_epochs,
                    batch_size,
                    logging=logging,
                    save_epochs=save_epochs,
                    final_save_path=(
                        f"{models_main_dir}/{model_file_name}.pt"
                        if models_main_dir
                        else None
                    ),
                    checkpoints_dir=(
                        f"{checkpoints_dir}/{model_file_name}"
                        if checkpoints_dir
                        else None
                    ),
                    model_params=model_params,
                )

                if models_main_dir:
                    print(
                        f"Training complete. Saved {model_file_name} to {models_main_dir}"
                    )


def evaluate_all_models_rolling(
    model_types,
    industries_list,
    ret_port_styles,
    num_epochs,
    data_weighing_types=["value"],
    models_main_dir=None,
    seed=519,
    io_fn=io_day_n_lag,
    seq_len=5,
    initial_train_end_year=1979,
    final_val_end_year=2024,
    show_plots=False,
    save_metrics=False,
    print_summary=0,  # 0 = Nothing, 1 = Final, 2 = All Years
    normalize=True,
):
    print("Creating/saving model results...")
    for num_industries in industries_list:
        for data_weighing in data_weighing_types:
            # there is no 'equal' vs 'value' for ff3 and ff6
            if num_industries in ["ff3", "ff6"] and len(data_weighing_types) > 1:
                continue

            data, _ = get_data(
                seed=seed,
                num_industries=num_industries,
                weight_type=data_weighing,
            )

            for model_type in model_types:
                for ret_port_style in ret_port_styles:
                    model_name = f"{seq_len}seq_len/{model_type.__name__}/{num_industries}industries/{data_weighing}/{str(num_epochs) + 'epochs/' if model_type != XGBRegressor else ''}"
                    model_dir = f'{models_main_dir}/{model_name}'


                    strat_ret_vec, mkt_ret_vec = evaluate_model_rolling(
                        model_dir=model_dir,
                        data=data,
                        ret_port_style=ret_port_style,
                        io_fn=io_fn,
                        seq_len=seq_len,
                        print_summary=print_summary,
                        save=save_metrics,
                        initial_train_end_year=initial_train_end_year,
                        final_val_end_year=final_val_end_year,
                        normalize=normalize,
                    )

                    show_cumulative_returns(
                        strat_ret_vec,
                        mkt_ret_vec,
                        strategy_name=model_name,
                        log_scale=False,
                        show_plots=show_plots,
                        save=save_metrics,
                        ret_port_style=ret_port_style,
                    )

                    print(
                        f"Finished saving {model_name} results for {ret_port_style}.\n\n"
                    )

    print("========================================")
    print("Finished saving model results.")
