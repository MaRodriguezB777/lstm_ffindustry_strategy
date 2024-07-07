# region imports
from AlgorithmImports import *
from all_ind_stocks import ALL_SICS, FF_IND_TO_SIC, SIC_TO_FF_IND
from model import MODEL_KEY
from model_utils import load_model, predict, FF_COLS
import numpy as np
import pandas as pd
from datetime import timedelta
from constants import *
from alpha import FFIndustryAlphaModel
from my_brokerage import MyBacktestTDBrokerage
# endregion


class LSTM_Industries(QCAlgorithm):
    def Initialize(self):
        self.set_start_date(2023, 1, 1)  # Set Start Date
        self.set_end_date(2024, 1, 1)  # Set End Date
        self.set_cash(1_000_000)  # Set Strategy Cash

        # Set benchmark
        self.set_benchmark("SPY")

        # Set brokerage and settings
        # if INCLUDE_FEES:
            # self.set_brokerage_model(TDAmeritradeBrokerageModel())
        # else:
            # self.set_brokerage_model(DefaultBrokerageModel())
        # self.default_order_properties.time_in_force = TimeInForce.DAY
        # self.set_security_initializer(lambda x: x.set_fee_model(TDAmeritradeFeeModel()))
        self.settings.minimum_order_margin_portfolio_percentage = 0

        # Set alpha model
        self.alpha_model = FFIndustryAlphaModel(self, MODEL_KEY)
        self.set_alpha(self.alpha_model)

        # Set universe settings
        # https://www.quantconnect.com/forum/discussion/13989/proper-way-to-differentiate-between-universes/
        self.universe_settings.asynchronous = False
        self.universe_settings.resolution = Resolution.DAILY
        # https://www.quantconnect.com/docs/v2/writing-algorithms/securities/asset-classes/us-equity/requesting-data#11-Data-Normalization
        self.universe_settings.data_normalization_mode = DataNormalizationMode.RAW
        # Filter Stocks
        self.add_universe_selection(FundamentalUniverseSelectionModel(self.SelectionFilter))

        # Set portfolio construction model
        self.set_portfolio_construction(InsightWeightingPortfolioConstructionModel())

        # Set risk management model
        self.set_risk_management(TrailingStopRiskManagementModel(0.05))

        # Set execution model
        self.set_execution(ImmediateExecutionModel())

        # self.Schedule.On(self.DateRules.EveryDay(),
        #                  self.TimeRules.Every(TimeSpan.FromHours(1)),
        #                  self.CheckStopLoss)

    def SelectionFilter(self, fundamental: list[Fundamental]):
        '''
        Chooses the stocks to be included in the universe.
        '''
        if LOGGING:
            self.log(f"main.SelectionFilter() called at {self.time}.")
        if SUPER_DEBUG:
            if self.time.date().day == 26:
                stocks = [x for x in fundamental if x.symbol.value in ['AAPL']]
            else: 
                stocks = [x for x in fundamental if (x.symbol.value in ['DWAC', 'NVDA'])]
            if LOGGING:
                for stock in stocks:
                    self.alpha_model.stock_inds[stock.symbol] = SIC_TO_FF_IND[stock.asset_classification.sic]
                out = [x.symbol for x in stocks]
                self.log(f"out: {[x.value for x in out]}")
            return out
        
        # https://www.quantconnect.com/forum/discussion/12234/exchange-id-mapping/
        exchanges = ["NAS", "NYS", "ASE"]

        # fundamental data filter
        filter = [x for x in fundamental if x.has_fundamental_data]
        # exchange filter
        filter = [x for x in fundamental if x.security_reference.exchange_id in exchanges]

        final_stocks = []
        for ind_abbr in FF_IND_TO_SIC.keys():
            sics = FF_IND_TO_SIC[ind_abbr]
            ind_filter = [x for x in filter if x.asset_classification.sic in sics]
            if SUPER_DEBUG:
                self.log(f"For industry {ind_abbr}, the siccodes are {sics}.")
                self.log(f"For industry {ind_abbr}, there are {len(ind_filter)} stocks.")
            sorted_stocks = sorted(ind_filter, key=lambda x: x.volume, reverse=True)[:TOP_STOCKS]
            for stock in sorted_stocks:
                self.alpha_model.stock_inds[stock.symbol] = ind_abbr

            final_stocks.extend(sorted_stocks)

        out = [x.symbol for x in final_stocks]
        return out

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
        if LOGGING:
            self.log(f"main.OnData() called at {self.time}.")

    def on_delisting(self, delistings: Delistings):
        for symbol in delistings.keys():
            delisting = Delisting(delistings[symbol])
            self.log(f"Delisting: {delisting.symbol} {delisting.type}")

    def on_brokerage_disconnect(self):
        self.log("Brokerage disconnected.")

    def on_brokerage_reconnect(self):
        self.log("Brokerage reconnected.")

    def dividends(self, dividends: Dividends):
        self.log(f"Dividends: {dividends.Symbol} {dividends}")

    def on_symbol_changed_events(self, changes: SymbolChangedEvents):
        for symbol in changes.keys():

            change = changes[symbol]
            self.log(f"Symbol changed: {change.symbol} {change}")





