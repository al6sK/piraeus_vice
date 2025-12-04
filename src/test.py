import pandas as pd
from pathlib import Path
import os


projectDir = Path(__file__).parent.parent

dataPath = projectDir / "data" / "crimes.csv"

data = pd.read_csv(str(dataPath))


print(data.head(5))

train_data = data[data["split"] == "TRAIN"]
val_data = data[data["split"] == "VAL"]
test_data = data[data["split"] == "TEST"]

print(train_data.head(5))
print(val_data.head(5))
print(test_data.head(5))

q1_data = data[data["split"] != "TEST"]
print(q1_data.head(15)) 