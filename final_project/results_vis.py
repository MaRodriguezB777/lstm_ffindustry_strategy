from final_project.train_and_eval import (
    evaluate_model_LSTM,
    evaluate_model_xgb,
    show_cumulative_returns,
)
from models import SimpleLSTM
from data import all_data_paths

# model_strat_distribution("models/weighted/SimpleLSTM_5days_ff6.pt", all_data_paths['ff6'])
strat_ret_vec, mkt_ret_vec = evaluate_model_LSTM(
    "models/weighted/SimpleLSTM_5days_ff6.pt",
    all_data_paths["ff6"],
    print_summary="short",
    weighted=True,
)
show_cumulative_returns(
    strat_ret_vec, mkt_ret_vec, strategy_name="SimpleLSTM_5days_ff6", log_scale=False
)


# model_strat_distribution('models/weighted/XGBRegressor_5days_ff6.pt', all_data_paths['ff6'])
strat_ret_vec, mkt_ret_vec = evaluate_model_xgb(
    "models/weighted/XGBRegressor_5days_ff6.pt", all_data_paths["ff6"]
)
show_cumulative_returns(
    strat_ret_vec, mkt_ret_vec, strategy_name="XGBRegressor_5days_ff6", log_scale=False
)


# model_strat_distribution('models/weighted/SimpleLSTM_5days_10industries_equal.pt', all_data_paths['10industries_equal'], data_weighing='equal')
strat_ret_vec, mkt_ret_vec = evaluate_model_LSTM(
    "models/weighted/SimpleLSTM_5days_10industries_equal.pt",
    all_data_paths["10industries_equal"],
)
show_cumulative_returns(
    strat_ret_vec,
    mkt_ret_vec,
    strategy_name="SimpleLSTM_5days_10industries_equal",
    log_scale=False,
)

# model_strat_distribution('models/weighted/SimpleLSTM_5days_10industries_value.pt', all_data_paths['10industries_value'], data_weighing='value')
strat_ret_vec, mkt_ret_vec = evaluate_model_LSTM(
    "models/weighted/SimpleLSTM_5days_10industries_value.pt",
    all_data_paths["10industries_value"],
)
show_cumulative_returns(
    strat_ret_vec,
    mkt_ret_vec,
    strategy_name="SimpleLSTM_5days_10industries_value",
    log_scale=False,
)


# model_strat_distribution('models/weighted/XGBRegressor_5days_ff6.pt', all_data_paths['ff6'])
strat_ret_vec, mkt_ret_vec = evaluate_model_xgb(
    "models/weighted/XGBRegressor_5days_10industries_equal.pt", all_data_paths["ff6"]
)
show_cumulative_returns(
    strat_ret_vec,
    mkt_ret_vec,
    strategy_name="XGBRegressor_5days_10industries_equal",
    log_scale=False,
)

# model_strat_distribution('models/weighted/XGBRegressor_5days_10industries_value.pt', all_data_paths['10industries_value'])
strat_ret_vec, mkt_ret_vec = evaluate_model_xgb(
    "models/weighted/XGBRegressor_5days_10industries_value.pt",
    all_data_paths["10industries_value"],
)
show_cumulative_returns(
    strat_ret_vec,
    mkt_ret_vec,
    strategy_name="XGBRegressor_5days_10industries_value",
    log_scale=False,
)


# model_strat_distribution('models/weighted/SimpleLSTM_5days_49industries_equal.pt', all_data_paths['49industries_equal'], data_weighing='equal')
strat_ret_vec, mkt_ret_vec = evaluate_model_LSTM(
    "models/weighted/SimpleLSTM_5days_49industries_equal.pt",
    all_data_paths["49industries_equal"],
)
show_cumulative_returns(
    strat_ret_vec,
    mkt_ret_vec,
    strategy_name="SimpleLSTM_5days_49industries_equal",
    log_scale=False,
)

# model_strat_distribution('models/weighted/SimpleLSTM_5days_49industries_value.pt', all_data_paths['49industries_value'], data_weighing='value')
strat_ret_vec, mkt_ret_vec = evaluate_model_LSTM(
    f"models/weighted/SimpleLSTM_5days_49industries_value.pt",
    all_data_paths[f"49industries_value"],
)
show_cumulative_returns(
    strat_ret_vec,
    mkt_ret_vec,
    strategy_name=f"SimpleLSTM_5days_49industries_value",
    log_scale=False,
)


# model_strat_distribution('models/weighted/XGBRegressor_5days_49industries_equal.pt', all_data_paths['49industries_equal'], data_weighing='equal')
strat_ret_vec, mkt_ret_vec = evaluate_model_xgb(
    f"models/weighted/XGBRegressor_5days_49industries_equal.pt",
    all_data_paths[f"49industries_equal"],
)
show_cumulative_returns(
    strat_ret_vec,
    mkt_ret_vec,
    strategy_name="XGBRegressor_5days_49industries_equal",
    log_scale=False,
)

# model_strat_distribution('models/weighted/XGBRegressor_5days_49industries_value.pt', all_data_paths['49industries_value'], data_weighing='value')
strat_ret_vec, mkt_ret_vec = evaluate_model_xgb(
    f"models/weighted/XGBRegressor_5days_49industries_value.pt",
    all_data_paths[f"49industries_value"],
)
show_cumulative_returns(
    strat_ret_vec,
    mkt_ret_vec,
    strategy_name="XGBRegressor_5days_49industries_value",
    log_scale=False,
)
