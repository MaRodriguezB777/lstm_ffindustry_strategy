from train_and_eval import (
    train_all_models,
    train_all_models_rolling,
    evaluate_all_models_rolling,
)
from models import SimpleLSTM, SimpleFeedForwardNN
from xgboost import XGBRegressor
import torch.nn as nn
import torch.optim as optim
import argparse
from utils import io_day_n_lag
from train_and_eval import evaluate_model, show_cumulative_returns
from data import all_data_paths

ROLLING = True
RET_PORT_STYLES = ["max-min_LS"]
INDUSTRIES_LIST = [49]
DATA_WEIGHING = ["value"]
MODEL_TYPES = [SimpleLSTM]
IO_FN = io_day_n_lag
SEQ_LEN = 5
SEED = 519
BATCH_SIZE = 32
NUM_EPOCHS = 5
MODEL_PARAMS = {
    "hidden_size_LSTM": 50,
    "hidden_sizes_FFNN": [32, 16],
    "dropout": 0.1,
    "activation_fn": nn.ReLU(),
    "lr": 1e-3,
    "optimizer": optim.Adam,
    "loss_fn": nn.MSELoss(),
}
INITIAL_TRAIN_END_YEAR = 1979  # inclusive
FINAL_VAL_END_YEAR = 2024  # inclusive
PRINT_SUMMARY = 1  # 0 = Nothing, 1 = Final, 2 = All Years
NORMALIZE = True

def list_of(arg):
    return arg.split(",")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ret_port_styles", type=list_of)
    parser.add_argument("--strats", type=list_of)

    args = parser.parse_args()
    ret_port_styles = RET_PORT_STYLES
    if args.ret_port_styles is not None:
        ret_port_styles = [w for w in args.ret_port_styles if w in RET_PORT_STYLES]

    industries_list = INDUSTRIES_LIST
    if args.strats is not None:
        industries_list = [s for s in args.strats if s in INDUSTRIES_LIST]

    # train = input("Train models? (y): ")
    # save_models = input("Save models? (y): ") == "y"
    # save_checkpoints = input("Save checkpoints? (y): ") == "y"
    # logging = input("Logging? (y): ") == "y"
    # save_metrics = input("Save evaluation metrics? (y): ")
    train = True
    save_models = True
    save_checkpoints = False
    logging = True
    compute_metrics = True
    save_metrics = True
    show_plots = True

    models_dir = "models"
    checkpoints_dir = None
    results_dir = "results"

    if train:
        if ROLLING:
            train_all_models_rolling(
                model_types=MODEL_TYPES,
                data_weighing_types=DATA_WEIGHING,
                industries_list=industries_list,
                io_fn=IO_FN,
                seq_len=SEQ_LEN,
                seed=SEED,
                num_epochs=NUM_EPOCHS,
                batch_size=BATCH_SIZE,
                logging=logging,
                save_epochs=10 if save_checkpoints else None,
                models_main_dir=models_dir if save_models else None,
                checkpoints_main_dir=checkpoints_dir,
                model_params=MODEL_PARAMS,
                initial_train_end_year=INITIAL_TRAIN_END_YEAR,
                final_val_end_year=FINAL_VAL_END_YEAR,
            )
        else:
            train_all_models(
                model_types=MODEL_TYPES,
                data_weighing_types=DATA_WEIGHING,
                industries_list=industries_list,
                io_fn=IO_FN,
                seq_len=SEQ_LEN,
                seed=SEED,
                num_epochs=NUM_EPOCHS,
                batch_size=BATCH_SIZE,
                logging=logging,
                save_epochs=10 if save_checkpoints else None,
                models_main_dir=models_dir if save_models else None,
                checkpoints_main_dir=checkpoints_dir,
                model_params=MODEL_PARAMS,
            )
        print("========================================")
        print("========================================")
        print("All Models finished trained.")
    else:
        print("No models trained.")

    if compute_metrics:
        if ROLLING:
            evaluate_all_models_rolling(
                model_types=MODEL_TYPES,
                industries_list=industries_list,
                ret_port_styles=ret_port_styles,
                num_epochs=NUM_EPOCHS,
                data_weighing_types=DATA_WEIGHING,
                models_main_dir=models_dir,
                seed=SEED,
                io_fn=IO_FN,
                seq_len=SEQ_LEN,
                initial_train_end_year=INITIAL_TRAIN_END_YEAR,
                final_val_end_year=FINAL_VAL_END_YEAR,
                show_plots=show_plots,
                save_metrics=save_metrics,
                print_summary=PRINT_SUMMARY,
                normalize=NORMALIZE,
            )
        else:
            print("Creating and saving model results...")
            for ret_port_style in ret_port_styles:
                for strat in industries_list:
                    for model_type in MODEL_TYPES:
                        for data_weighing in DATA_WEIGHING:
                            model_name = f"{model_type.__name__}_{SEQ_LEN}days_{strat}{f'industries_{data_weighing}' if strat not in ['ff3', 'ff6'] else ''}.pt"
                            model_path = f"{models_dir}/{model_name}"
                            results_path = (
                                f"{results_dir}/{ret_port_style}/{model_name}/"
                            )
                            if strat == "ff6":
                                data_path = all_data_paths[strat]
                            else:
                                data_path = all_data_paths[
                                    f"{strat}industries_{data_weighing}"
                                ]

                            strat_ret_vec, mkt_ret_vec = evaluate_model(
                                model_path,
                                data_path,
                                io_fn=IO_FN,
                                seq_len=SEQ_LEN,
                                ret_port_style=ret_port_style,
                                save=True,
                            )

                            show_cumulative_returns(
                                strat_ret_vec,
                                mkt_ret_vec,
                                strategy_name=model_path,
                                log_scale=False,
                                show_plots=show_plots,
                                save=True,
                                ret_port_style=ret_port_style,
                            )

                            print(
                                f"Finished saving {model_name} results for {ret_port_style}."
                            )
        print("========================================")
        print("========================================")
        print("All Models finished evaluating.")
