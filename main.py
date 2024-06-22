# region imports
from AlgorithmImports import *
from datetime import timedelta
import os
import pandas as pd
from all_ind_stocks import ALL_IND_STOCKS, ALL_IND_STOCKS_YF
from model import SimpleLSTM, URLS
import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle
import base64

# endregion

PRIOR_DAYS = 5


class LSTM_Industries(QCAlgorithm):
    def Initialize(self):
        self.set_start_date(2023, 1, 3)  # Set Start Date
        self.set_end_date(2023, 1, 5)  # Set End Date
        self.set_cash(100000)  # Set Strategy Cash

        # Load model
        self.model = self.load_model(URLS[2023])

        self.spy = self.add_equity("SPY", Resolution.DAILY)

        # spy.SetDataNormalizationMode(DataNormalizationMode.Raw)
        # self.nvda = nvda
        # self.spy = spy
        # self.stop = False

        self.all_stocks = self.filter_industry_stocks(ALL_IND_STOCKS)

        # self.log(f"all_stocks: {self.all_stocks}")

        # Set benchmark
        self.set_benchmark("SPY")

        # self.SetBrokerageModel(
        #     BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin
        # )

        self.entry_price = 0
        self.period = timedelta(31)
        self.next_entry_time = self.time

        # self.schedule.on(
        #     self.date_rules.every_day(), self.time_rules.at(9, 30), self.rebalance
        # )

        self.schedule.on(
            self.date_rules.every_day(), self.time_rules.at(15, 55), self.liquidate
        )

    def load_model(
        self,
        url: str,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        base64_str = self.download(url)
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

        self.log(f"Model loaded successfully on device: {device}")
        self.log(f"Model: {model}")

        return model

    def ticker_exists(self, ticker):
        equity_obj = self.add_equity(ticker, Resolution.DAILY)
        history = self.history([ticker], 7, Resolution.DAILY)
        if len(history) == 7:
            market_cap = equity_obj.fundamentals.market_cap
            if market_cap > 0:
                return True
        # self.log(f"Ticker {ticker} not found.")
        return False

    def filter_industry_stocks(self, all_ind_stocks):
        total_count = 0
        valid_count = 0

        ind_stocks = {}
        for industry in all_ind_stocks:
            ind_stocks[industry] = {}
            for ticker in all_ind_stocks[industry]:
                total_count += 1
                if self.ticker_exists(ticker):
                    valid_count += 1
                    equity = self.add_equity(ticker, Resolution.DAILY)
                    ind_stocks[industry][ticker] = equity

        self.log(f"Total stocks: {total_count}, valid stocks: {valid_count}")
        for industry, stocks in ind_stocks.items():
            self.log(f"Industry: {industry}, number of valid stocks: {len(stocks)}")

        return ind_stocks

    def get_weights(self):
        total_ind_market_caps = {}
        all_market_caps = {}
        all_weights = {}

        for industry, stocks_dict in self.all_stocks.items():
            total_ind_market_caps[industry] = 0
            all_market_caps[industry] = {}
            for ticker, equity_obj in stocks_dict.items():
                market_cap = equity_obj.fundamentals.market_cap
                # self.Log(f"ticker: {ticker}, market_cap: {market_cap}")
                all_market_caps[industry][ticker] = market_cap
                total_ind_market_caps[industry] += market_cap

        for industry in self.all_stocks.keys():
            all_weights[industry] = {}
            for ticker, market_cap in all_market_caps[industry].items():
                weight = market_cap / total_ind_market_caps[industry]
                all_weights[industry][ticker] = weight

        return all_weights

        # for industry, stocks in self.all_stocks.items():
        #     # stock_market_caps[industry] = {}
        #     for ticker, equity_object in stocks.items():
        #         if data.containsKey(ticker):
        #             market_cap = equity_object.fundamentals.market_cap
        #             stock_market_caps[ticker] = market_cap
        #             total_market_cap += market_cap

        # for ticker, market_cap in stock_market_caps.items():
        #     weight = market_cap / total_market_cap
        #     weights[ticker] = weight

        # return weights

    def get_industries(self):
        # TODO: @firdavsn? implement this function
        # gets the top decile industries and bottom decile industry in terms of
        # predicted returns by the LSTM model
        # returns a dictionary with two keys: "long" and "short" and a list of
        # industries for each key
        pass

    def get_industry_returns(self, all_weights):
        industry_returns = {}

        for industry in self.all_stocks.keys():
            stocks = self.all_stocks[industry]
            weights = all_weights[industry]

            sum_industry_returns = None
            for ticker in stocks.keys():
                history = self.history(
                    [ticker], PRIOR_DAYS + 2, Resolution.DAILY
                )  # +1 day since it includes current day in history, +1 to get return of first day in history

                returns = history.copy()
                returns["return"] = returns["close"].pct_change() * 100
                returns = returns.drop(
                    columns=["close", "high", "low", "open", "volume"]
                )
                returns = returns.iloc[1:-1]
                returns = returns.reset_index(drop=True)

                # Weigh returns
                returns *= weights[ticker]

                if sum_industry_returns is None:
                    sum_industry_returns = returns
                else:
                    sum_industry_returns += returns
            avg_industry_returns = sum_industry_returns / len(stocks)

            industry_returns[industry] = avg_industry_returns

        return industry_returns

    def OnData(self, data):
        self.log("hello from OnData")
        all_weights = self.get_weights()
        # for industry, weights in all_weights.items():
        #     sum_weights = sum(weights.values())
        #     self.Log(f"Industry: {industry}, sum_weights: {sum_weights}")

        industry_returns = self.get_industry_returns(all_weights)
        # for industry, returns in industry_returns.items():
        #     prior_day_return = returns.iloc[-1]
        #     self.log(f"Industry: {industry}, prior_day_return: {prior_day_return}")

        # chosen_industries = self.get_industries()
        # long_weights = self.get_weights(data, chosen_industries["long"])
        # short_weights = self.get_weights(data, chosen_industries["short"])

        # for ticker, weight in long_weights.items():
        #     self.SetHoldings(ticker, weight)

        # for ticker, weight in short_weights.items():
        #     self.SetHoldings(ticker, -weight)
