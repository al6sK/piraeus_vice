import pandas as pd
from pathlib import Path

dataPath = Path(__file__).parent / "crimes.csv"

class Filter:
    TRAIN   = "TRAIN"
    TEST    = "TEST"
    VAL     = "VAL"
    

class DataLoader:
    def __init__(self):
        self.data = pd.read_csv(dataPath)

    def dataset(self):
        return self.data

    def split_filter(self, filter):
        return self.data[self.data["split"] == filter]

    def split_filters(self, filters):
        return self.data[self.data["split"].isin(filters)]
        