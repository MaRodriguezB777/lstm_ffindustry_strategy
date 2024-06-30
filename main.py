# region imports
from AlgorithmImports import *
from all_ind_stocks import ALL_IND_STOCKS, ALL_SICS, FF_IND_TO_SIC
from model import URLS_NORMALIZED, URLS_UNNORMALIZED
from model_utils import load_model, predict, FF_COLS
import numpy as np
import pandas as pd
from datetime import timedelta

# endregion

PRIOR_DAYS = 5
TOP_STOCKS = 10
TOP_STOCKS_PERCENT = 0.2
STOP_LOSS_THRESHOLD = 0.05  # 5% stop loss
SHORT_STRATEGY = True
LONG_SHORT_RATIO = 0.95
INCLUDE_FEES = False


class LSTM_Industries(QCAlgorithm):
    def Initialize(self):
        self.set_start_date(2023, 1, 4)  # Set Start Date
        self.set_end_date(2023, 1, 6)  # Set End Date
        self.set_cash(50_000)  # Set Strategy Cash

        self.set_brokerage_model(BrokerageName.TD_AMERITRADE, AccountType.MARGIN)
        self.settings.minimum_order_margin_portfolio_percentage = 0

        # https://www.quantconnect.com/forum/discussion/13989/proper-way-to-differentiate-between-universes/
        self.universe_settings.resolution = Resolution.DAILY
        self.my_universe = self.add_universe(self.SelectionFilter)
        
        # Load model
        self.model = load_model(self, URLS_UNNORMALIZED[2023])

        self.set_warm_up(timedelta(days=PRIOR_DAYS + 1))

        # Filter stocks
        # self.all_stocks = self.filter_industry_stocks(ALL_IND_STOCKS)

        # Set benchmark
        # self.spy = self.AddEquity("SPY", Resolution.DAILY)
        # self.set_benchmark("SPY")

        # self.latest_data = None
        # self.purchase_prices_long = {}
        # self.purchase_prices_short = {}

        # self.Schedule.On(self.DateRules.EveryDay(),
        #                  self.TimeRules.at(15, 30),
        #                  self.Rebalance)

        # self.Schedule.On(self.DateRules.EveryDay(),
        #                  self.TimeRules.Every(TimeSpan.FromHours(1)),
        #                  self.CheckStopLoss)

    def SelectionFilter(self, coarse: list[Fundamental]):
        # get only health stocks
        self.log(
            f"Selecting stocks in the Fama French industries according to SIC code."
        )
        # https://www.quantconnect.com/forum/discussion/12234/exchange-id-mapping/
        exchanges = ["NAS", "NYS", "ASE"]

        # exchange filter
        filter = [x for x in coarse if x.security_reference.exchange_id in exchanges]
        filter = [x for x in filter if x.asset_classification.sic in ALL_SICS]

        final_stocks = []
        # industry filter
        for _, sics in FF_IND_TO_SIC.items():
            ind_filter = [x for x in filter if x.asset_classification.sic in sics]
            sorted_stocks = sorted(ind_filter, key=lambda x: x.volume, reverse=True)
            final_stocks.extend(sorted_stocks[:int(len(sorted_stocks) * TOP_STOCKS_PERCENT)])


        out = [x.symbol for x in final_stocks]
        return out

    def on_securities_changed(self, changes):
        # for security in changes.added_securities:
        #     self.debug(f"Added: {security.symbol}")
        # self.debug(f"{security.symbol} Exchange id: {security.fundamentals.security_reference.exchange_id}")
        # self.debug(f"{security.symbol} SIC: {security.fundamentals.asset_classification.sic}")

        # for security in changes.removed_securities:
        #     self.debug(f"Removed: {security.symbol}")
        # self.debug(f"{security.symbol} Exchange id: {security.fundamentals.security_reference.exchange_id}")
        # self.debug(f"{security.symbol} SIC:
        # {security.fundamentals.asset_classification.sic}")

        pass

    def ticker_exists(self, ticker):
        equity_obj = self.AddEquity(ticker, Resolution.DAILY)
        history = self.history([ticker], 7, Resolution.DAILY)
        if len(history) == 7:
            market_cap = equity_obj.fundamentals.market_cap
            if market_cap > 0:
                return True
        # self.Log(f"Ticker {ticker} not found.")
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
                    equity = self.AddEquity(ticker, Resolution.DAILY)
                    if not INCLUDE_FEES:
                        equity.set_fee_model(ConstantFeeModel(0))
                    ind_stocks[industry][ticker] = equity

        self.Log(f"Total stocks: {total_count}, valid stocks: {valid_count}")
        # for industry, stocks in ind_stocks.items():
        #     self.Log(f"Industry: {industry}, number of valid stocks: {len(stocks)}")

        return ind_stocks

    def get_weights(self, data):
        total_ind_market_caps = {}
        all_market_caps = {}
        all_weights = {}

        for industry, stocks_dict in self.all_stocks.items():
            all_market_caps[industry] = {}
            for ticker, equity_obj in stocks_dict.items():
                if ticker in data.Bars:
                    market_cap = equity_obj.fundamentals.market_cap
                    # self.Log(f"ticker: {ticker}, market_cap: {market_cap}")
                    all_market_caps[industry][ticker] = market_cap

        # Only keep TOP_STOCKS number of stocks
        if TOP_STOCKS:
            top_market_caps = {}
            for industry, market_caps in all_market_caps.items():
                top_market_caps[industry] = {
                    x[0]: x[1]
                    for x in sorted(
                        market_caps.items(), key=lambda x: x[1], reverse=True
                    )[:TOP_STOCKS]
                }
        else:
            top_market_caps = all_market_caps

        # Get total market cap for each industry
        for industry, market_caps in top_market_caps.items():
            total_ind_market_caps[industry] = sum(market_caps.values())

        for industry in top_market_caps.keys():
            if len(top_market_caps[industry]) == 0:
                continue

            all_weights[industry] = {}
            for ticker, market_cap in top_market_caps[industry].items():
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

    def get_industry_returns(self, all_weights):
        industry_returns = {}

        for industry in all_weights.keys():
            weights = all_weights[industry]

            sum_industry_returns = None
            for ticker in weights.keys():

                # history = self.history(
                #     [ticker], PRIOR_DAYS + 2, Resolution.DAILY
                # )  # +1 day since it includes current day in history, +1 to get return of first day in history
                history = self.history(
                    [ticker], PRIOR_DAYS + 1, Resolution.DAILY
                )  # +1 to get return of first day in history

                returns = history.copy()
                returns["return"] = returns["close"].pct_change() * 100
                returns = returns.drop(
                    columns=["close", "high", "low", "open", "volume"]
                )
                returns = returns.iloc[1:]

                returns = returns.reset_index(drop=True)

                # If len(returns) < PRIOR_DAYS, make all zeros
                if len(returns) < PRIOR_DAYS:
                    returns = pd.DataFrame(
                        np.zeros((PRIOR_DAYS, 1)), columns=["return"]
                    )

                # Weigh returns
                returns *= weights[ticker]

                if sum_industry_returns is None:
                    sum_industry_returns = returns
                else:
                    sum_industry_returns += returns

            avg_industry_returns = sum_industry_returns / len(weights)
            industry_returns[industry] = np.array(avg_industry_returns["return"])

        return industry_returns

    def get_long_short_industries(self, pred):
        pred = pred[0]

        long_cutoff = np.percentile(pred, 90)
        short_cutoff = np.percentile(pred, 10)

        long_industries = []
        short_industries = []
        for i, x in enumerate(pred):
            if x >= long_cutoff:
                long_industries.append(FF_COLS[i])
            elif x <= short_cutoff:
                short_industries.append(FF_COLS[i])

        return long_industries, short_industries

    def go_long(self, long_industries, all_weights):
        num_long_industries = len(long_industries)

        for industry in long_industries:
            for ticker, weight in all_weights[industry].items():
                self.SetHoldings(
                    ticker, LONG_SHORT_RATIO * weight / num_long_industries
                )
                self.purchase_prices_long[ticker] = self.Securities[ticker].Price

    def go_short(self, short_industries, all_weights):
        num_short_industries = len(short_industries)

        for industry in short_industries:
            for ticker, weight in all_weights[industry].items():
                self.SetHoldings(
                    ticker, -LONG_SHORT_RATIO * weight / num_short_industries
                )
                self.purchase_prices_short[ticker] = self.Securities[ticker].Price

    def Rebalance(self):
        self.Log(f"Rebalance Time: {self.Time}")

        # Liquidate all invested stocks
        self.Liquidate()
        self.purchase_prices_long = {}
        self.purchase_prices_short = {}

        print_weights = False
        print_returns = False
        print_pred = True
        print_long_short_weights = True

        all_weights = self.get_weights(self.latest_data)
        all_industries_present = len(all_weights) == 49
        self.Log(f"all_industries_present: {all_industries_present}")
        if print_weights:
            for industry, weights in all_weights.items():
                sum_weights = sum(weights.values())
                self.Log(
                    f"Industry: {industry}, sum_weights: {sum_weights}, num_stocks: {len(weights)}"
                )

        if all_industries_present:
            industry_returns = self.get_industry_returns(all_weights)
            if print_returns:
                for industry, returns in industry_returns.items():
                    prior_day_return = returns.iloc[-1]
                    self.Log(
                        f"Industry: {industry}, prior_day_return: {prior_day_return}"
                    )

            pred = predict(self.model, industry_returns)
            if print_pred:
                self.Log(f"pred: {list(pred)}")
                self.Log(f"pred.shape: {pred.shape}")

            long_industries, short_industries = self.get_long_short_industries(pred)
            if print_long_short_weights:
                self.Log(f"long_industries: {long_industries}")
                self.Log(f"short_industries: {short_industries}")

            if not SHORT_STRATEGY:
                self.go_long(long_industries, all_weights)
                self.go_short(short_industries, all_weights)
            else:
                self.go_long(short_industries, all_weights)
                self.go_short(long_industries, all_weights)

        self.Log(f"Portfolio value: {self.Portfolio.TotalPortfolioValue}")
        self.Log(f"Portfolio cash: {self.Portfolio.Cash}")

        # Log portfolio holdings of only stocks that are invested
        invested_stocks = [x.Key for x in self.Portfolio if x.Value.Invested]
        short_stocks = []
        long_stocks = []
        for stock in invested_stocks:
            if self.Portfolio[stock].Quantity < 0:
                short_stocks.append(stock)
            else:
                long_stocks.append(stock)

        # Log total value of short and long stocks
        short_value = sum(
            [
                self.Portfolio[stock].Quantity * self.Portfolio[stock].Price
                for stock in short_stocks
            ]
        )
        long_value = sum(
            [
                self.Portfolio[stock].Quantity * self.Portfolio[stock].Price
                for stock in long_stocks
            ]
        )
        self.Log(f"Short value: {short_value}")
        self.Log(f"Long value: {long_value}")

    # def CheckStopLoss(self):
    #     self.Log(f"Checking stop loss at: {self.Time}")
    #     for ticker, purchase_price in self.purchase_prices_long.items():
    #         current_price = self.Securities[ticker].Price
    #         if current_price < purchase_price * (1 - STOP_LOSS_THRESHOLD):
    #             self.Log(f"Selling {ticker} (long) due to stop loss. Purchase price: {purchase_price}, Current price: {current_price}")
    #             self.Liquidate(ticker)

    #     for ticker, purchase_price in self.purchase_prices_short.items():
    #         current_price = self.Securities[ticker].Price
    #         if current_price > purchase_price * (1 + STOP_LOSS_THRESHOLD):
    #             self.Log(f"Selling {ticker} (short) due to stop loss. Purchase price: {purchase_price}, Current price: {current_price}")
    #             self.Liquidate(ticker)

    def OnData(self, data: Slice):
        # print all data
        # self.log("OnData() called. Printing all data.")
        # self.log(f"Time: {self.time}")
        # for key, value in data.bars.items():
        #     self.log(f"{key}: {value}")

        # self.latest_data = data
        pass
