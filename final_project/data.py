import pandas as pd
from utils import set_seed, io_day_n_lag

all_data_paths = {"ff3": "ff3_daily.csv", "ff6": "ff6_daily.csv"}

for n_ind in [5, 10, 12, 17, 30, 38, 48, 49]:
    for weight in ["equal", "value"]:
        all_data_paths[f"{n_ind}industries_{weight}"] = (
            f"ff_industry_portfolios_daily_{weight}/{n_ind}_Industry_Portfolios_Daily.csv"
        )

def load_ff3(val_start_date=None, val_end_date=None):
    ff3 = pd.read_csv("ff3_daily.csv")

    if val_start_date is not None:
        val_start_idx = ff3[ff3['date'] >= int(val_start_date)].index[0]
        
    if val_end_date is not None:
        val_end_idx = ff3[ff3['date'] > int(val_end_date)].index[0]
    
    if val_start_date is not None and val_end_date is not None:
        ff3 = ff3.iloc[val_start_idx:val_end_idx]
    elif val_start_date is not None:
        ff3 = ff3.iloc[val_start_idx:]
    elif val_end_date is not None:
        ff3 = ff3.iloc[:val_end_idx]
    

    ff3 = ff3.reset_index(drop=True)
    ff3.index = ff3["date"]
    ff3 = ff3.drop(columns=["date"])

    return ff3


def load_ff6(val_start_date=None, val_end_date=None):
    ff6 = pd.read_csv("ff6_daily.csv")

    if val_start_date is not None:
        val_start_idx = ff6[ff6['date'] >= int(val_start_date)].index[0]

    if val_end_date is not None:
        val_end_idx = ff6[ff6['date'] > int(val_end_date)].index[0]
    
    if val_start_date is not None and val_end_date is not None:
        ff6 = ff6.iloc[val_start_idx:val_end_idx]
    elif val_start_date is not None:
        ff6 = ff6.iloc[val_start_idx:]
    elif val_end_date is not None:
        ff6 = ff6.iloc[:val_end_idx]

    ff6 = ff6.reset_index(drop=True)
    ff6.index = ff6["date"]
    ff6 = ff6.drop(columns=["date"])

    return ff6


def get_data(num_industries, seed=519, weight_type="value"):
    if num_industries in ["ff3", "ff6"]:
        data_path = all_data_paths[num_industries]
    else:
        data_path = all_data_paths[f"{num_industries}industries_{weight_type}"]

    data = pd.read_csv(data_path)
    data = data.dropna()

    if num_industries in ["ff3", "ff6"]:
        data = data.reset_index(drop=True)
        data.index = data["date"]
        data = data.drop(columns=["date"])

    if seed is not None:
        set_seed(seed)

    return data, all_data_paths
