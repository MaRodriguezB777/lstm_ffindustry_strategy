# region imports
from AlgorithmImports import *
from all_ind_stocks import ALL_IND_STOCKS, ALL_IND_STOCKS_YF
from model import URLS_NORMALIZED, URLS_UNNORMALIZED
from model_utils import load_model, predict, FF_COLS
import numpy as np
import pandas as pd
# endregion

PRIOR_DAYS = 5
TOP_STOCKS = 25
STOP_LOSS_THRESHOLD = 0.05   # 5% stop loss

class LSTM_Industries(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2023, 1, 4)  # Set Start Date
        self.SetEndDate(2024, 6, 20)  # Set End Date
        self.SetCash(1_000_000)  # Set Strategy Cash

        # Load model
        self.model = load_model(self, URLS_UNNORMALIZED[2023])

        # Filter stocks
        self.all_stocks = self.filter_industry_stocks(ALL_IND_STOCKS)

        # Set benchmark
        self.spy = self.AddEquity("SPY", Resolution.DAILY)
        self.set_benchmark("SPY")
        
        self.latest_data = None
        self.purchase_prices_long = {}
        self.purchase_prices_short = {}

        
        self.Schedule.On(self.DateRules.EveryDay(), 
                         self.TimeRules.at(9, 30),
                         self.Rebalance)
        
        # self.Schedule.On(self.DateRules.EveryDay(), 
        #                  self.TimeRules.Every(TimeSpan.FromHours(1)),
        #                  self.CheckStopLoss)

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
                top_market_caps[industry] = {x[0]: x[1] for x in sorted(market_caps.items(), key=lambda x: x[1], reverse=True)[:TOP_STOCKS]}
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
                    returns = pd.DataFrame(np.zeros((PRIOR_DAYS, 1)), columns=["return"])

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
                self.SetHoldings(ticker, 0.5 * weight/num_long_industries)
                self.purchase_prices_long[ticker] = self.Securities[ticker].Price
        
    def go_short(self, short_industries, all_weights):
        num_short_industries = len(short_industries)
        
        for industry in short_industries:
            for ticker, weight in all_weights[industry].items():
                self.SetHoldings(ticker, -0.5 * weight/num_short_industries)
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
                self.Log(f"Industry: {industry}, sum_weights: {sum_weights}, num_stocks: {len(weights)}")

        if all_industries_present:
            industry_returns = self.get_industry_returns(all_weights)
            if print_returns:
                for industry, returns in industry_returns.items():
                    prior_day_return = returns.iloc[-1]
                    self.Log(f"Industry: {industry}, prior_day_return: {prior_day_return}")
            
            pred = predict(self.model, industry_returns)
            if print_pred:
                self.Log(f"pred: {list(pred)}")
                self.Log(f"pred.shape: {pred.shape}")

            long_industries, short_industries = self.get_long_short_industries(pred)
            if print_long_short_weights:
                self.Log(f"long_industries: {long_industries}")
                self.Log(f"short_industries: {short_industries}")
            
            self.go_short(short_industries, all_weights)
            self.go_long(long_industries, all_weights)
        
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
        short_value = sum([self.Portfolio[stock].Quantity * self.Portfolio[stock].Price for stock in short_stocks])
        long_value = sum([self.Portfolio[stock].Quantity * self.Portfolio[stock].Price for stock in long_stocks])
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
    
    def OnData(self, data):
        self.latest_data = data