# region imports
from AlgorithmImports import *
from QuantConnect.Algorithm import QCAlgorithm
from QuantConnect.Algorithm.Framework.Alphas import Insight
from QuantConnect.Data import Slice
from QuantConnect.Data.UniverseSelection import SecurityChanges
from System.Collections.Generic import IEnumerable
from model_utils import load_model, FF_COLS, predict
from constants import PRIOR_DAYS, LOGGING, DEBUG
import numpy as np
from datetime import timedelta
# endregion

# Your New Python File
class FFIndustryAlphaModel(AlphaModel):
    ind_stock_return_windows: Dict[str, Dict[Symbol, RollingWindow[float]]]
    prev_close: Dict[Symbol, float]
    stock_inds: Dict[Symbol, str]

    def __init__(self, algo: QCAlgorithm, model_key) -> None:
        self.name = "FFIndustryAlphaModel"
        self.model = load_model(algo, model_key=model_key)
        self.prev_close = {}
        self.ind_stock_return_windows = {}
        self.stock_inds = {} # derived from universe selection. Could contain stocks from previous day, not reset after each universe selection
    
    def on_securities_changed(self, algo: QCAlgorithm, changes: SecurityChanges) -> None:
        '''
        Handles what happens when new securities are added or old ones removed
        from the universe.
        '''
        if LOGGING:
            algo.log(f"alpha_model.on_securities_changed() called at {algo.time}.")
            # if DEBUG:
            algo.log(f"This is what self.stock_inds looks like: {[f'{symbol.value}: {ind}' for symbol, ind in self.stock_inds.items()]}.")
        for security in changes.added_securities:
            if LOGGING:
                algo.log(f"Added: {security.symbol}")
            self.setup_rolling_window(algo, security.symbol)

        for security in changes.removed_securities:
            try:
                security_industry = self.stock_inds[security.symbol]
            except KeyError:
                algo.log(f"Error getting industry for {security.symbol} in on_securities_changed.")
                return
            if LOGGING:
                algo.log(f"Removed: {security.symbol}")
            
            try:
                self.prev_close.pop(security.symbol)
            except KeyError:
                algo.log(f"KeyError: {security.symbol}. Symbol not found in prev_close but was removed from universe.")
            try:
                self.ind_stock_return_windows[security_industry].pop(security.symbol)
            except KeyError:
                algo.log(f"KeyError: {security.symbol}. Symbol not found in ind_stock_return_windows but was removed from universe. Means stock was only in universe for only one day or something is wrong...")

            # remove if the stock is not in the universe anymore
            self.stock_inds.pop(security.symbol)

            if security.invested:
                algo.liquidate(security.symbol) # TODO: might want to send signal instead of directly liquidating

    def setup_rolling_window(self, algo: QCAlgorithm, symbol: Symbol, new_close=None):
        '''
        Logic for rolling window setup.'''
        history_length = 10 # should be enough / NOTE: This is calendar days not trading days
        max_history_length = 20

        try:
            industry = self.stock_inds[symbol]
        except KeyError:
            algo.log(f"KeyError. Could not find industry for {symbol} in alpha.setup_rolling_window(). Means that the stock was not in the universe at last universe selection.")
            return

        if industry not in self.ind_stock_return_windows:
            self.ind_stock_return_windows[industry] = {}

        if symbol not in self.ind_stock_return_windows[industry]:
            if DEBUG:
                algo.log(f"Setting up initial rolling window for {symbol}.")
            
            self.prev_close[symbol] = None
            self.ind_stock_return_windows[industry][symbol] = RollingWindow[float](PRIOR_DAYS)

            history = algo.history(symbol, history_length, Resolution.DAILY) 

            # need + 1 to get percentage change for PRIOR_DAYS days
            while(len(history) < PRIOR_DAYS):
                history_length += 5
                history = algo.history(symbol, history_length, Resolution.DAILY)

                # there is not enough history to make PRIOR_DAYS days of returns
                if history_length >= max_history_length:
                    if len(history) == 0:
                        algo.log(f"Warning. No history found for {symbol} but was in universe.")
                        return
                    if len(history) == 1: # if there is only 1 day of history, onData handles it
                        return
                    elif len(history) == 2: # if there are 2 days of history, then there is no return to create but there is a close price
                        self.prev_close[symbol] = history['close'].iloc[-2]
                        return
                    else:
                        break

            # don't include the current day so that we can get the data from OnData
            history = history.iloc[-(PRIOR_DAYS + 1):-1]
            
                                
            stock_return = history["close"].pct_change() * 100
            stock_return = stock_return.dropna()

            for _return in stock_return:
                self.ind_stock_return_windows[industry][symbol].add(_return)

            self.prev_close[symbol] = history['close'].iloc[-1]

        elif new_close is not None:
            if DEBUG:
                algo.log(f"Updating rolling windows for {symbol}.")

            prev_close = self.prev_close[symbol]
            self.prev_close[symbol] = new_close

            if(prev_close is None):
                algo.log(f"Warning. Triggered prev_close was None for {symbol}. Should mean that the stock just IPOed/changed ticker or potentially had a gap in their data.")
            else:
                new_return = (new_close - prev_close) / prev_close * 100
                self.ind_stock_return_windows[industry][symbol].add(new_return)

        if DEBUG:
            algo.log(f"prev_close[{symbol}]: {self.prev_close[symbol]}")
            algo.log(f"ind_stock_return_windows[{industry}][{symbol}]: {[item for item in self.ind_stock_return_windows[industry][symbol]]}")

    def setup_data_window(self, algo: QCAlgorithm, data: Slice):
        '''
        Adds current day data to the rolling window.'''
        if LOGGING:
            algo.log(f"alpha_model.setup_data_window() called at {algo.time}.")
            algo.log(f"This is all the data: {[f'{symbol}' for symbol in data.keys()]}")
        for symbol in data.bars.keys():
            if data.bars[symbol] is None:
                algo.log(f"Warning. data for symbol is None for {symbol} but is still present in data variable.")
                continue
            if DEBUG:
                if data[symbol].value == "DWAC":
                    algo.log(f"DWAC bar: {data.bars[symbol].to_string()}")
                    algo.log(f"The current universe has DWAC set to {symbol in algo.universe_manager.active_securities}")
            if symbol in algo.universe_manager.active_securities:
                self.setup_rolling_window(algo, symbol, new_close=data[symbol].close)

    def setup_industry_returns(self, algo: QCAlgorithm):
        '''
        Ret: np.array of shape (num_industries, PRIOR_DAYS) with the oldest
        day's return at index 0
        '''
        stock_market_caps = {}
        industry_market_caps = {}
        if LOGGING:
            algo.log(f"In alpha.setup_industry_returns()")
        ind_returns = np.zeros(shape=(len(FF_COLS), PRIOR_DAYS))
        for ind_idx, industry in enumerate(FF_COLS):
            if DEBUG:
                algo.log(f"Setting up industry returns for {industry}.")
            try: 
                stock_returns = self.ind_stock_return_windows[industry]
            except KeyError:
                algo.log(f"KeyError. Could not find industry {industry} in ind_stock_return_windows. Likely means that there are no stocks in this industry. Here are the industries present in ind_stock_return_windows: {self.ind_stock_return_windows.keys()}.")
                continue

            # day_lag_returns[0] is the oldest day's returns
            day_lag_returns = np.zeros(PRIOR_DAYS)
            ind_market_cap = 0
            
            for symbol, window in stock_returns.items():
                if DEBUG:
                    algo.log(f"Adding stock returns for {symbol} in {industry}.")
                try:
                    market_cap = algo.securities[symbol].fundamentals.market_cap
                    stock_market_caps[symbol] = market_cap
                    ind_market_cap += market_cap
                except KeyError:
                    algo.log(f"KeyError. Could not find market cap for {symbol}")
                    continue

                for i, _return in enumerate(window):
                    day_lag_returns[PRIOR_DAYS - i - 1] += _return * market_cap

            industry_market_caps[industry] = ind_market_cap
            if ind_market_cap == 0:
                algo.log(f"Warning. Market cap for {industry} is 0. This is likely because there are no stocks in this industry.")
            else:
                day_lag_returns /= ind_market_cap

            if DEBUG:
                algo.log(f"day_lag_returns for {industry}: {day_lag_returns}")
            
            ind_returns[ind_idx] = day_lag_returns

        if DEBUG:
            algo.log(f"ind_returns: {ind_returns}")
        return ind_returns, stock_market_caps, industry_market_caps
    
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
    
    def make_insights(self, algo: QCAlgorithm, industries, stock_mkt_caps, ind_mkt_caps, insight_direction: InsightDirection):
        if LOGGING:
            algo.log(f"alpha_model.make_insights() called at {algo.time}.")

        insights = []
        for ind_abbr in industries:
            if DEBUG:
                algo.log(f"Getting {insight_direction} insights for {ind_abbr}.")
            for symbol in self.ind_stock_return_windows[ind_abbr].keys():
                if DEBUG:
                    algo.log(f"Making flat insight for {symbol}.")
                weight = stock_mkt_caps[symbol] / ind_mkt_caps[ind_abbr]
                insight = Insight.price(symbol=symbol, period=timedelta(days=1), direction=insight_direction, weight=weight)
                insights.append(insight)
        
        return insights

    def get_insights(self, algo: QCAlgorithm, ind_returns, stock_mkt_caps, ind_mkt_caps):
        '''
        Args:
            ind_returns: np.array of shape (num_industries, PRIOR_DAYS) with the
            oldest day's return at index 0
        '''
        if LOGGING:
            algo.log(f"alpha_model.get_insights() called at {algo.time}.")
        model_predictions = predict(algo, self.model, ind_returns)
        if DEBUG:
            algo.log(f"pred: {list(model_predictions)}")
            algo.log(f"pred.shape: {model_predictions.shape}")
        long_industries, short_industries = self.get_long_short_industries(model_predictions)
        flat_industries = set(FF_COLS) - set(long_industries) - set(short_industries)

        long_insights = self.make_insights(algo, long_industries, stock_mkt_caps, ind_mkt_caps, InsightDirection.UP)
        short_insights = self.make_insights(algo, short_industries, stock_mkt_caps, ind_mkt_caps, InsightDirection.DOWN)
        flat_insights = self.make_insights(algo, flat_industries, stock_mkt_caps, ind_mkt_caps, InsightDirection.FLAT)

        return long_insights + short_insights + flat_insights

    def update(self, algo: QCAlgorithm, data: Slice) -> IEnumerable[Insight]:
        if LOGGING:
            algo.log(f"alpha_model.update() called at {algo.time}.")
        if algo.is_warming_up:
            return
        if data.bars.keys() is None:
            algo.log(f"Warning. The slice of data has no bars.")
            return
        self.setup_data_window(algo, data)
        ind_returns, stock_mkt_caps, ind_mkt_caps = self.setup_industry_returns(algo)

        insights = self.get_insights(algo, ind_returns, stock_mkt_caps, ind_mkt_caps)

        return insights
