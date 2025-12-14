import pandas as pd
import numpy as np
from pathlib import Path
import os
from sklearn.decomposition import PCA
from matplotlib import ticker
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats

projectDir = Path(__file__).parent.parent
dataPath = projectDir / "data" / "data_encoded.csv"
q7_plotsPath = projectDir / "plots" / "Q7"
q8_plotsPath = projectDir / "plots" / "Q8"

# --------------------------------------------------------
# Q7. Principal Component Analysis (PCA)
# --------------------------------------------------------

data = pd.read_csv(str(dataPath))
# print(data.head(3))

# Drop columns
columns_to_delete = [
    "incident_id",
    "weapon_code",
    "scene_type",
    "weather"
]
data.drop(columns_to_delete, axis=1, inplace=True)

# split dataset and saving the index of each split
ix_train = np.array(data.index[data['split'] == "TRAIN"])
ix_dev = np.array(data.index[data['split'] == "VAL"])
ix_test = np.array(data.index[data['split'] == "TEST"])
data.drop('split', axis=1, inplace=True)

y_encoded = pd.get_dummies(data["killer_id"])  

y_train_onehot = y_encoded.iloc[ix_train].to_numpy()
y_dev_onehot   = y_encoded.iloc[ix_dev].to_numpy()
y_test_onehot  = y_encoded.iloc[ix_test].to_numpy()

#set X and y
X = data.drop(columns=["killer_id"]).copy()
y = data["killer_id"]

# --------------------------------------------------------
# a)
# --------------------------------------------------------

pca = PCA()
X_train = X.iloc[ix_train].to_numpy()
X_pca = pca.fit_transform(X_train)

eigenvalues = pca.explained_variance_ 
print("Eigenvalues:", eigenvalues)

# --------------------------------------------------------
# b)
# --------------------------------------------------------

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
plt.axhline(cumulative_explained_variance[tmp], color="r", linestyle="--", linewidth=2)
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
os.makedirs(os.path.join(q7_plotsPath), exist_ok=True)
plt.savefig(str(q7_plotsPath / "Explained Variance by Principal Components.png"), dpi=300, bbox_inches="tight")

# --------------------------------------------------------
# c)
# --------------------------------------------------------

X_val = X.iloc[ix_dev].to_numpy()
X_val_embed = pca.transform(X_val)
# Scatter plot of the first two PCA components
plt.figure(figsize=(8, 6))
plt.scatter(X_val_embed[:, 0], X_val_embed[:, 1], c=data["killer_id"].iloc[ix_dev], cmap="tab10", alpha=0.7)
plt.title("PCA: First Two Components", fontsize=16, fontweight="bold")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
# plt.colorbar()
plt.tight_layout()
plt.savefig(str(q7_plotsPath / "Projection of each VAL feature vector xi onto the first two principal components.png"), dpi=300, bbox_inches="tight")

# --------------------------------------------------------
# Q8. k-means clustering in latent space
# --------------------------------------------------------

# --------------------------------------------------------
# a: project each feature vector xi onto the first m principal components
# --------------------------------------------------------

X_train_embed = pca.transform(X_train)
# X_val_embed is ready from last step
X_test = X.iloc[ix_test].to_numpy()
X_test_embed = pca.transform(X_test)

m = tmp

Z_train = X_train_embed[:,:m-1]
Z_val = X_val_embed[:,:m-1]
Z_test = X_test_embed[:,:m-1]

# --------------------------------------------------------
# b: Run k-means with k = S
# --------------------------------------------------------

k = data["killer_id"].nunique()
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(Z_train)

# print(clusters)
silhouette = silhouette_score(Z_train, clusters)
print("KMeans Silhouette Score:", silhouette)

# --------------------------------------------------------
# c: mapping from k-means clusters to killer labels using majority vote
# --------------------------------------------------------

result = stats.mode(clusters, keepdims=True)

mapping = {}

for q in range(k):
    # create a filter for current cluster q
    mask = (clusters == q)
    labels_in_cluster = y.iloc[ix_train].values[mask]
    
    mode_result = stats.mode(labels_in_cluster, keepdims=True)
    winner = mode_result.mode[0]
    votes = mode_result.count[0]
    # save winning match
    mapping[q] = int(winner)
    print(f"Cluster {q}: Winner= {winner}, Votes= {votes}/{len(labels_in_cluster)}")
    

print("The map:")
print(mapping)

# --------------------------------------------------------
# b:
# --------------------------------------------------------
