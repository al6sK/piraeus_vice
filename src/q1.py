from pathlib import Path
from scipy.stats import norm
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

projectDir = Path(__file__).resolve().parent.parent
sys.path.append(str(projectDir / "data"))

from loader import DataLoader, Filter

plotsPath = projectDir / "plots"
featuresPath = plotsPath / "distributions_histplots"
fittedPath = plotsPath / "fitted_plots"
featuresPath.mkdir(parents=True, exist_ok=True)
fittedPath.mkdir(exist_ok=True)
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

# arithmetic mean calculation - same as sum(list) / len(list)
mean = hour_float.mean()

# standard deviation - same as square root of the variance
std = hour_float.std()

plt.figure("Gaussian Fit")

plt.title("Hour Float Distribution")
sns.histplot(hour_float, stat="density")

xMin, xMax = hour_float.min(), hour_float.max()
x_range = np.linspace(xMin, xMax, 100)

# calculating the Gaussian Distribution
pdf_values = norm.pdf(x_range, mean, std)
plt.plot(x_range, pdf_values, color="red", label=f"Fitted Gaussian mean={mean:.2f} std={std:.2f}")
plt.legend()
plt.savefig(fittedPath / "hour_float_GaussianFit.png")
plt.show()
