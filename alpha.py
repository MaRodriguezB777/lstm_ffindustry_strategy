# region imports
from AlgorithmImports import *
from QuantConnect.Algorithm import QCAlgorithm
from QuantConnect.Algorithm.Framework.Alphas import Insight
from QuantConnect.Data import Slice
from QuantConnect.Data.UniverseSelection import SecurityChanges
from System.Collections.Generic import IEnumerable
from model import SimpleLSTM, MODEL_KEY
from model_utils import load_model

# endregion

# Your New Python File
class FFIndustryAlphaModel(AlphaModel):
    def __init__(self, algo: QCAlgorithm) -> None:
        super().__init__()
        self.model = load_model(algo, model_key=MODEL_KEY)
        self.predictions = None
    
    def on_securities_changed(self, algorithm: QCAlgorithm, changes: SecurityChanges) -> None:
        return super().on_securities_changed(algorithm, changes)
        

    def update(self, algorithm: QCAlgorithm, data: Slice) -> IEnumerable[Insight]:
        insights = []

            


        return super().update(algorithm, data)    