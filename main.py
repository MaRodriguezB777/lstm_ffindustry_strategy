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

        self.schedule.on(self.date_rules.every_day(), 
                         self.time_rules.at(15, 55), 
                         self.liquidate)

    def ticker_exists(self, ticker):
        try:
            history = self.History([ticker], 1, Resolution.Daily)
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

    def get_weights(self, chosen_industries):
        total_market_caps = {}
        stock_market_caps = {}
        weights = {}

        for industry, stocks in self.industry_stocks.items():
            if industry not in chosen_industries:
                continue
            stock_market_caps[industry] = {}
            for ticker, equity in stocks.items():
                if self.ticker_exists(ticker):
                    market_cap = equity.fundamentals.market_cap
                    stock_market_caps[industry][ticker] = market_cap
                    total_market_caps[industry] += market_cap

        for industry, market_caps in stock_market_caps.items():
            for ticker, market_cap in market_caps.items():
                weight = market_cap / total_market_caps[industry]
                weights[ticker] = weight

        return weights

    def get_industries(self):
        # TODO: @firdavsn? implement this function
        # gets the top decile industry and bottom decile industry in terms of
        # predicted returns by the LSTM model
        pass

    def OnData(self, data):
        chosen_industries = self.get_industries()
        weights = self.get_weights(chosen_industries)

        for ticker, weight in weights.items():
            if data.containsKey(ticker):
                self.SetHoldings(ticker, weight)
        # market_cap = self.nvda.fundamentals.market_cap
        # self.Log("Market Cap: " + str(market_cap))
        # if market_cap is not None:
        # if self.portfolio.invested:
        #     self.liquidate()
        # if not self.nvda.symbol in data:
        #     return

        # # price = data.Bars[self.spy].Close
        # # price = data[self.nvda].Close
        # # price = self.Securities[self.spy].Close

        # if not self.portfolio.invested and not self.stop:
        #     # if self.nextEntryTime <= self.Time:
        #     self.SetHoldings(self.nvda.symbol, 0.5)
        #     self.stop = True
        # self.MarketOrder(self.spy, int(self.Portfolio.Cash / price) )
        # self.Log("BUY NVDA @" + str(price))
        # self.entryPrice = price

        # elif self.entryPrice * 1.1 < price or self.entryPrice * 0.90 > price:
        #     self.Liquidate()
        #     self.Log("SELL SPY @" + str(price))
        #     self.nextEntryTime = self.Time + self.period


def OnEndOfDay(self):

