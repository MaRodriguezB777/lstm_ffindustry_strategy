# region imports
from AlgorithmImports import *
import pandas as pd
import numpy as np

# endregion


def predictions_to_returns(pred_df, y, ret_port_style="max-min_LS"):
    # Given the predictions of each factor for each day, calculate our
    # strategy for each day, and the returns for each day

    # Apply our strategy to our predictions (in form [0, 1, 0, 0, 0])
    if ret_port_style == "max-min_LS":
        strat_df = pred_df.apply(
            lambda row: max_min_predicted_factor_strat(row), axis=1
        )
    elif ret_port_style == "weighted_LS":
        strat_df = pred_df.apply(
            lambda row: weighted_predicted_factor_strat(row), axis=1
        )
    # elif ret_port_style == "max_L":
    #     strat_df = pred_df.apply(lambda row: max_predicted_factor_strat(row), axis=1)
    else:
        raise ValueError(f"Invalid ret_port_style: {ret_port_style}")

    # Calculate our returns
    return_vector = np.multiply(strat_df, np.asarray(y)).apply(sum, axis=1)

    return strat_df, return_vector


def max_min_predicted_factor_strat(row):
    # For each day, set our strategy to be the factor with
    # the highest predicted return minus the factor with the
    # lowest predicted return

    # get the top 10% and bottom 10% of the predictions
    long_cutoff = np.percentile(row, 90)
    short_cutoff = np.percentile(row, 10)

    num_long = sum(row >= long_cutoff)
    num_short = sum(row <= short_cutoff)

    row_list = []
    for x in row:
        if x >= long_cutoff:
            row_list.append(1 / num_long)
        elif x <= short_cutoff:
            row_list.append(-1 / num_short)
        else:
            row_list.append(0)
    return pd.Series(row_list)


def weighted_predicted_factor_strat(row):
    pos_preds = [x for x in row if x > 0]
    neg_preds = [x for x in row if x < 0]
    pos_sum = sum(pos_preds)
    neg_sum = abs(sum(neg_preds))
    row_list = [0] * len(row)
    for i in range(len(row)):
        if row[i] > 0:
            row_list[i] = row[i] / pos_sum
        elif row[i] < 0:
            row_list[i] = row[i] / neg_sum
    return pd.Series(row_list)
