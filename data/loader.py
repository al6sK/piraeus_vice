import pandas as pd
from pathlib import Path
from enum import Enum

data_path = Path(__file__).parent / "crimes.csv"

class Filter(Enum):
    TRAIN   = "TRAIN"
    TEST    = "TEST"
    VAL     = "VAL"


class DataLoader:
    def __init__(self):
        self.data = pd.read_csv(data_path)

    def dataset(self):
        return self.data

    def split_filter(self, filter: Filter):
        return self.data[self.data["split"] == filter.value]

    def split_filters(self, filters: list[Filter]):
        return self.data[self.data["split"].isin([filter.value for filter in filters])]
