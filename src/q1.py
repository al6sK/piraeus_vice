from pathlib import Path
from scipy.stats import norm
import os
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

projectDir = Path(__file__).resolve().parent.parent
sys.path.append(str(projectDir / "data"))

from loader import DataLoader, Filter

plotsPath = projectDir / "plots"
featuresPath = plotsPath / "distributions_histplots"
os.makedirs(os.path.join(featuresPath), exist_ok=True)

dataloader = DataLoader()

dataset = dataloader.split_filters([Filter.TRAIN, Filter.VAL])

plotsconfig = {
    "hour_float": {"bins": 48},
    "latitude": {"bins": 40},
    "longitude": {"bins": 40},
    "victim_age": {"bins": 20, "kde": True},
}

for column, config in plotsconfig.items():
    plt.figure()
    sns.histplot(dataset[column], **config)
    plt.savefig(featuresPath / (column + ".png"))
    plt.close()

hour_float = dataset["hour_float"]

mean = hour_float.mean()
variance = hour_float.var()
sigma = np.sqrt(variance)

xMin, xMax = hour_float.min(), hour_float.max()

x = np.linspace(xMin, xMax, 100)
p = norm.pdf(x, mean, sigma)

plt.figure()
plt.plot(x, p)
plt.show()
