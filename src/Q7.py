import pandas as pd
import numpy as np
from pathlib import Path
import os
from sklearn.decomposition import PCA
from matplotlib import ticker
from matplotlib import pyplot as plt
projectDir = Path(__file__).parent.parent
dataPath = projectDir / "data" / "data_encoded.csv"
plotsPath = projectDir / "plots" / "Q7"

data = pd.read_csv(str(dataPath))

print(data.head(3))

data = pd.read_csv(str(dataPath))

# Drop columns
columns_to_delete = [
    "incident_id",
    "weapon_code",
    "scene_type",
    "weather"
]
data.drop(columns_to_delete, axis=1, inplace=True)

ix_train = np.array(data.index[data['split'] == "TRAIN"])
ix_dev = np.array(data.index[data['split'] == "VAL"])
ix_test = np.array(data.index[data['split'] == "TEST"])
data.drop('split', axis=1, inplace=True)

y_encoded = pd.get_dummies(data["killer_id"])  

y_train_onehot = y_encoded.iloc[ix_train].values
y_dev_onehot   = y_encoded.iloc[ix_dev].values
y_test_onehot  = y_encoded.iloc[ix_test].values

#set X and y
X = data.drop(columns=["killer_id"]).copy()
y = data["killer_id"]

# a)
pca = PCA()
X_train = X.iloc[ix_train].values
X_pca = pca.fit_transform(X_train)

eigenvalues = pca.explained_variance_ 
print("Eigenvalues:", eigenvalues)

# b)
cumulative_explained_variance = pca.explained_variance_ratio_.cumsum()


# Define the x-tick positions (every 100th point)
tmp = np.where(cumulative_explained_variance >= 0.95)[0][0]
x_ticks = sorted(
    list(range(0, len(cumulative_explained_variance) + 1, 5))
    + [
        len(cumulative_explained_variance) - 1,
        tmp,
    ]
)

plt.figure(figsize=(8, 6))

# Get the corresponding explained variance values for those x-tick positions
y_values = cumulative_explained_variance[x_ticks]

# Plot the cumulative explained variance only at the x-tick positions
plt.plot(x_ticks, y_values, marker="o", linestyle="--", color="b")

# Highlight the threshold for 95% explained variance
plt.axhline(0.95, color="r", linestyle="--", linewidth=2)
plt.axvline(tmp, color="r", linestyle="--", linewidth=2)

# Title and labels with added subtitle for clarity
plt.title(
    "Explained Variance by Principal Components",
    fontsize=18,
    loc="left",
    fontweight="bold",
    pad=30,
)
plt.figtext(
    0.08,
    0.9,
    "Red dashed lines indicate the point at which 95% of the variance is explained.",
    fontsize=12,
    fontweight="normal",
)
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")

# Adjust x-ticks to plot every 100th point
plt.xticks(x_ticks)

# Customize gridlines for better readability
plt.grid(True, linestyle="--", alpha=0.5)
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
# Tidy layout and show the plot
plt.tight_layout()
os.makedirs(os.path.join(plotsPath), exist_ok=True)
plt.savefig(str(plotsPath / "Explained Variance by Principal Components.png"), dpi=300, bbox_inches="tight")
