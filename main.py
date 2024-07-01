# region imports
from AlgorithmImports import *
from all_ind_stocks import ALL_SICS, FF_IND_TO_SIC, SIC_TO_FF_IND
from model import MODEL_KEY
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
MIN_VOLUME = 20_000
INCLUDE_FEES = False
LOGGING = False
DEBUG = False


class LSTM_Industries(QCAlgorithm):
    def Initialize(self):
        self.set_start_date(2024, 3, 26)  # Set Start Date
        self.set_end_date(2024, 3, 26)  # Set End Date
        self.set_cash(50_000)  # Set Strategy Cash

        self.set_brokerage_model(BrokerageName.TD_AMERITRADE, AccountType.MARGIN)
        self.settings.minimum_order_margin_portfolio_percentage = 0

        # https://www.quantconnect.com/forum/discussion/13989/proper-way-to-differentiate-between-universes/
        self.universe_settings.resolution = Resolution.DAILY
        # https://www.quantconnect.com/docs/v2/writing-algorithms/securities/asset-classes/us-equity/requesting-data#11-Data-Normalization
        self.universe_settings.data_normalization_mode = DataNormalizationMode.RAW
        # Filter Stocks
        self.my_universe = self.add_universe(self.SelectionFilter)

        self.prev_close = {}
        self.stock_return_windows = {}

        self.industry_stocks = {}
        
        # Load model
        self.model = load_model(self, model_key=MODEL_KEY)

        # Warming up
        # self.set_warm_up(timedelta(days=PRIOR_DAYS + 1))

        # Set benchmark
        self.set_benchmark("SPY")

        # self.Schedule.On(self.DateRules.EveryDay(),
        #                  self.TimeRules.at(9, 30),
        #                  self.Rebalance)

        # self.Schedule.On(self.DateRules.EveryDay(),
        #                  self.TimeRules.Every(TimeSpan.FromHours(1)),
        #                  self.CheckStopLoss)

    def SelectionFilter(self, coarse: list[Fundamental]):
        if LOGGING:
            self.log(f"SelectionFilter() called at {self.time}.")
        # https://www.quantconnect.com/forum/discussion/12234/exchange-id-mapping/
        if DEBUG:
            if self.time.date().day == 28:
                out = [x.symbol for x in coarse if x.symbol.value in ['AAPL']]
            else: 
                out = [x.symbol for x in coarse if (x.symbol.value in ['MSFT', 'NVDA'])]
            if LOGGING:
                self.log(f"out: {[x.value for x in out]}")
            return out
        
        exchanges = ["NAS", "NYS", "ASE"]

        # exchange filter
        filter = [x for x in coarse if x.security_reference.exchange_id in exchanges]
        filter = [x for x in filter if x.asset_classification.sic in ALL_SICS]

        final_stocks = []
        for ind_abbr in FF_IND_TO_SIC.keys():
            sics = FF_IND_TO_SIC[ind_abbr]
            ind_filter = [x for x in filter if x.asset_classification.sic in sics]
            if DEBUG:
                self.log(f"For industry {ind_abbr}, the siccodes are {sics}.")
                self.log(f"For industry {ind_abbr}, there are {len(ind_filter)} stocks.")
            sorted_stocks = sorted(ind_filter, key=lambda x: x.volume, reverse=True)
            final_stocks.extend(sorted_stocks[:TOP_STOCKS])

        out = [x.symbol for x in final_stocks]
        return out


    def on_securities_changed(self, changes):
        if LOGGING:
            self.log(f"securities_changed() called at {self.time}.")
        for security in changes.added_securities:
            if DEBUG:
                self.log(f"Added: {security.symbol}")
            self.setup_rolling_window(security.symbol)

        for security in changes.removed_securities:
            if DEBUG:
                self.log(f"Removed: {security.symbol}")
            
            try:
                self.prev_close.pop(security.symbol)
            except KeyError:
                self.error(f"KeyError: {security.symbol}. Symbol not found in prev_close but was removed from universe.")
            try:
                self.stock_return_windows.pop(security.symbol)
            except KeyError:
                self.debug(f"KeyError: {security.symbol}. Symbol not found in stock_return_windows but was removed from universe. Means stock was only in universe for only one day or something is wrong...")

            if security.invested:
                self.liquidate(security.symbol)


    def setup_data_window(self, data: Slice):
        for (symbol, bar) in data.bars.items():
            if bar is None:
                self.error(f"bar is None for {symbol} but in data.bars.")
                continue
            self.setup_rolling_window(symbol, bar.close)


    def setup_rolling_window(self, symbol: Symbol, new_close=None):
        history_length = 10 # should be enough / NOTE: This is calendar days not trading days
        max_length = 20
        if symbol not in self.stock_return_windows:
            if DEBUG:
                self.log(f"Setting up initial rolling window for {symbol}.")
            self.prev_close[symbol] = None
            self.stock_return_windows[symbol] = RollingWindow[float](PRIOR_DAYS)

            history = self.history(symbol, history_length, Resolution.DAILY) # don't include the current day so that we can get the data from OnData

            # need + 1 to get percentage change for PRIOR_DAYS days
            while(len(history) < PRIOR_DAYS):
                history_length += 5
                history = self.history(symbol, history_length, Resolution.DAILY)

                # there is not enough history to make PRIOR_DAYS days of returns
                if history_length >= max_length:
                    if len(history) == 1: # if there is only 1 day of history, onData handles it
                        return
                    elif len(history) == 2: # if there are 2 days of history, then there is no return to create but there is a close price
                        self.prev_close[symbol] = history['close'].iloc[-2]
                        return
                    else:
                        break

            history = history.iloc[-(PRIOR_DAYS + 1):-1]
            
                                
            stock_return = history["close"].pct_change() * 100
            stock_return = stock_return.dropna()

            for _return in stock_return:
                self.stock_return_windows[symbol].add(_return)

            self.prev_close[symbol] = history['close'].iloc[-1]

        elif new_close is not None:
            if DEBUG:
                self.log(f"Updating rolling windows for {symbol}.")

            prev_close = self.prev_close[symbol]
            self.prev_close[symbol] = new_close

            new_return = (new_close - prev_close) / prev_close * 100
            self.stock_return_windows[symbol].add(new_return)

        if DEBUG:
            self.log(f"prev_close[{symbol}]: {self.prev_close[symbol]}")
            self.log(f"stock_return_windows[{symbol}]: {[item for item in self.stock_return_windows[symbol]]}")
            return


    def ticker_exists(self, ticker):
        equity_obj = self.AddEquity(ticker, Resolution.DAILY)
        history = self.history([ticker], 7, Resolution.DAILY)
        if len(history) == 7:
            market_cap = equity_obj.fundamentals.market_cap
            if market_cap > 0:
                return True
        # self.Log(f"Ticker {ticker} not found.")
        return False

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
        # if self.is_warming_up:
        #     return
        if LOGGING:
            self.log(f"OnData() called at {self.time}.")
        self.setup_data_window(data)

        # print all data
        # self.log("OnData() called. Printing all data.")
        # self.log(f"Time: {self.time}")
        # for key, value in data.bars.items():
        #     self.log(f"{key}: {value}")

        # self.latest_data = data