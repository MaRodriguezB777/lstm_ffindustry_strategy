# region imports
from AlgorithmImports import *
from datetime import timedelta
import os
import pandas as pd

# endregion


class LSTM_Industries(QCAlgorithm):
    def Initialize(self):
        self.set_start_date(2023, 6, 1)  # Set Start Date
        self.set_end_date(2024, 6, 1)  # Set End Date
        self.set_cash(100000)  # Set Strategy Cash

        # nvda = self.add_equity("ASDFLKJASDLFKN", Resolution.DAILY)
        # spy = self.add_equity("SPY", Resolution.DAILY)
        # self.AddForex, self.AddFuture...

        # spy.SetDataNormalizationMode(DataNormalizationMode.Raw)
        # self.nvda = nvda
        # self.spy = spy
        # self.stop = False

        self.industry_stocks = self.add_industry_stocks()

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

    def ticker_exists(self, ticker):
        try:
            history = self.history([ticker], 1, Resolution.DAILY)
            if not history.empty:
                return True
        except Exception as e:
            self.Debug(f"Ticker {ticker} not found: {e}")
        return False

    def add_industry_stocks(self):
        DATA_DIR = "industry_stocks_data"
        # list all files in the directory
        files = os.listdir(DATA_DIR)
        ind_stocks = {}

        for filename in files:
            ind_name = filename[:-4]
            ind_stocks[ind_name] = {}
            df = pd.read_csv(os.path.join(DATA_DIR, filename))

            tickers = df["ticker"].tolist()
            for ticker in tickers:
                if self.ticker_exists(ticker):
                    equity = self.add_equity(ticker, Resolution.Daily)
                    ind_stocks[ind_name][ticker] = equity

        return ind_stocks

    def get_weights(self, data: Slice, chosen_industries):
        total_market_cap = 0
        stock_market_caps = {}
        weights = {}

        for industry, stocks in self.industry_stocks.items():
            if industry not in chosen_industries:
                continue
            # stock_market_caps[industry] = {}
            for ticker, equity in stocks.items():
                if data.containsKey(ticker):
                    market_cap = equity.fundamentals.market_cap
                    stock_market_caps[ticker] = market_cap
                    total_market_cap += market_cap

        for ticker, market_cap in stock_market_caps.items():
            weight = market_cap / total_market_cap
            weights[ticker] = weight

        return weights

    def get_industries(self):
        # TODO: @firdavsn? implement this function
        # gets the top decile industries and bottom decile industry in terms of
        # predicted returns by the LSTM model
        # returns a dictionary with two keys: "long" and "short" and a list of
        # industries for each key
        pass

    def OnData(self, data: Slice):
        chosen_industries = self.get_industries()
        long_weights = self.get_weights(data, chosen_industries["long"])
        short_weights = self.get_weights(data, chosen_industries["short"])

        for ticker, weight in long_weights.items():
            self.SetHoldings(ticker, weight)

        for ticker, weight in short_weights.items():
            self.SetHoldings(ticker, -weight)
