# region imports
from AlgorithmImports import *
# endregion

# Your New Python File
class MyBacktestTDBrokerage(DefaultBrokerageModel):
    def __init__(self, account_type=AccountType.MARGIN):
        super().__init__(account_type)
        self.support_security_types = {SecurityType.EQUITY}
        self.support_order_types = {OrderType.MARKET_ON_OPEN ,OrderType.MARKET, OrderType.LIMIT, OrderType.STOP_MARKET, OrderType.STOP_LIMIT}

    def can_submit_order(self, security: Security, order, message):
        if not self.is_valid_order_size(security, order.Quantity, message):
            return False

        if security.type not in self.support_security_types:
            message = BrokerageMessageEvent(BrokerageMessageType.WARNING, "NotSupported",
                                            Messages.DefaultBrokerageModel.unsupported_security_type(self, security))
            return False

        if order.Type not in self.support_order_types:
            message = BrokerageMessageEvent(BrokerageMessageType.WARNING, "NotSupported",
                                            Messages.DefaultBrokerageModel.unsupported_order_type(self, order, self.support_order_types))
            return False

        return super().can_submit_order(security, order, message)

    def can_update_order(self, security, order, request, message):
        message = None
        return (True, message)

    def get_fee_model(self, security):
        return TDAmeritradeFeeModel()
    

