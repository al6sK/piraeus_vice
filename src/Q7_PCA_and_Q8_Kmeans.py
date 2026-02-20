# Alexios Kastanaras P22062, 
# Danai Charzaka P22194,
# Dimitrios Lazanas P22082
# Contact email for the group: alexioskast@gmail.com
 
import pandas as pd
import numpy as np
from pathlib import Path
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import ticker
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats

# Setup Paths
projectDir = Path(__file__).parent.parent
dataPath = projectDir / "data" / "data_encoded.csv" 
q7_plotsPath = projectDir / "plots" / "Q7"
q8_plotsPath = projectDir / "plots" / "Q8"

os.makedirs(q7_plotsPath, exist_ok=True)
os.makedirs(q8_plotsPath, exist_ok=True)

# Data Loading (Encoded Data)
data = pd.read_csv(str(dataPath))

# Drop metadata columns to keep only Features
drop_cols = ["incident_id", "split", "killer_id", "weapon_code", "scene_type", "weather"]
cols_to_drop = [c for c in drop_cols if c in data.columns]

# X contains features, y contains labels
X = data.drop(columns=cols_to_drop)
y = data["killer_id"]

# Split indices
ix_train = data.index[data['split'] == "TRAIN"].tolist()
ix_dev = data.index[data['split'] == "VAL"].tolist()
ix_test = data.index[data['split'] == "TEST"].tolist()

# Extract numpy arrays
X_train = X.loc[ix_train].values
X_val = X.loc[ix_dev].values
X_test = X.loc[ix_test].values

# --------------------------------------------------------
# Q7a: Standardise continuous features 
# --------------------------------------------------------
scaler = StandardScaler()

# Fit on TRAIN, transform TRAIN
X_train = scaler.fit_transform(X_train)

# Transform VAL and TEST using the TRAIN scaler
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# --------------------------------------------------------
# a) Run PCA on TRAIN
# --------------------------------------------------------
pca = PCA()
X_pca = pca.fit_transform(X_train) 

eigenvalues = pca.explained_variance_ 

# --------------------------------------------------------
# b) Plot Variance
# --------------------------------------------------------
cumulative_explained_variance = pca.explained_variance_ratio_.cumsum()

# Find index where variance >= 0.95
tmp = np.where(cumulative_explained_variance >= 0.95)[0][0]
m = tmp + 1 # Number of components is index + 1

# Plotting
x_ticks = sorted(list(range(len(cumulative_explained_variance))) + [len(cumulative_explained_variance)-1, tmp])
y_values = cumulative_explained_variance[x_ticks]

plt.figure(figsize=(8, 6))
plt.plot(x_ticks, y_values, marker="o", linestyle="--", color="b", linewidth=1.3)
plt.axhline(cumulative_explained_variance[tmp], color="r", linestyle="--", linewidth=2)
plt.axvline(tmp, color="r", linestyle="--", linewidth=2)
plt.title("Explained Variance by Principal Components", fontsize=18, fontweight="bold", pad=30)
plt.figtext(
    0.08,
    0.9,
    "Red dashed lines indicate the point at which 95% of the variance is explained.",
    fontsize=12,
    fontweight="normal",
)
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(str(q7_plotsPath / "explained_variance.png"), dpi=300)
plt.close()

print(f"Number of components for 95% variance: {m}")

# --------------------------------------------------------
# c) Project VAL onto PC1, PC2
# --------------------------------------------------------
X_val_embed = pca.transform(X_val)

svm_preds_path = projectDir / "data" / "predictions" / "SVM_pred.csv"
svm_preds_df = pd.read_csv(svm_preds_path)
val_incident_ids = data.loc[ix_dev, "incident_id"].values

val_order_df = pd.DataFrame({'incident_id': val_incident_ids})

merged_preds = val_order_df.merge(svm_preds_df, on="incident_id", how="left")
val_svm_predictions = merged_preds['predicted_killer'].values

plt.figure(figsize=(10, 8)) 
scatter = plt.scatter(X_val_embed[:, 0], X_val_embed[:, 1], c=val_svm_predictions, cmap="tab10", alpha=0.7)

plt.title("PCA on VAL: Coloured by SVM Predictions", fontsize=16, fontweight="bold")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

handles, _ = scatter.legend_elements()
unique_killers = np.sort(np.unique(val_svm_predictions))
killer_labels = [f"Killer {int(k)}" for k in unique_killers]

plt.legend(handles, killer_labels, title="Predicted Killer", loc='best')
plt.tight_layout()
plt.savefig(str(q7_plotsPath / "val_pca_projection.png"), dpi=300, bbox_inches="tight")
plt.close()
# --------------------------------------------------------
# Q8 k-means clustering in latent space
# --------------------------------------------------------

# a) Project all splits onto first m components
X_train_embed = pca.transform(X_train)
X_test_embed = pca.transform(X_test)

# Keep only first m components
Z_train = X_train_embed[:, :m]
Z_val = X_val_embed[:, :m]
Z_test = X_test_embed[:, :m]

# b) Run k-means with k=S on TRAIN
k_clusters = y.nunique() # S=8
kmeans = KMeans(n_clusters=k_clusters, random_state=42)
clusters = kmeans.fit_predict(Z_train)

silhouette = silhouette_score(Z_train, clusters)
print("KMeans Silhouette Score (TRAIN):", silhouette)

# c) Mapping from clusters to labels
mapping = {}
print("\nCluster to Killer Mapping:")
for q in range(k_clusters):
    mask = (clusters == q)
    if np.sum(mask) > 0:
        labels_in_cluster = y.iloc[ix_train].values[mask]
        mode_res = stats.mode(labels_in_cluster, keepdims=True)
        winner = mode_res.mode[0]
        votes = mode_res.count[0]
        mapping[q] = int(winner)
        print(f"Cluster {q} -> Killer {winner} ({votes}/{len(labels_in_cluster)} votes)")
    else:
        mapping[q] = 1 # Default fallback
        print(f"Cluster {q} -> Empty (Default 1)")

# d) Evaluate on VAL
val_clusters = kmeans.predict(Z_val)
val_preds = [mapping[c] for c in val_clusters]
val_acc = np.mean(val_preds == y.iloc[ix_dev].values)
print(f"\nVAL Accuracy (K-Means): {val_acc:.4f}")

# e) Evaluate on TEST
test_clusters = kmeans.predict(Z_test)
test_preds = [mapping[c] for c in test_clusters]
test_acc = np.mean(test_preds == y.iloc[ix_test].values)
print(f"TEST Accuracy (K-Means): {test_acc:.4f}")

# f) Scatter plot for TEST (PC1 vs PC2) coloured by PREDICTED label
plt.figure(figsize=(10, 8))
plt.scatter(X_test_embed[:, 0], X_test_embed[:, 1], c=test_preds, cmap="tab10", alpha=0.7)
plt.title("PCA on TEST: Coloured by K-Means Predictions", fontsize=16, fontweight="bold")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
handles, _ = scatter.legend_elements()
unique_killers_test = np.sort(np.unique(test_preds))
killer_labels_test = [f"Killer {int(k)}" for k in unique_killers_test]
plt.legend(handles, killer_labels_test, title="Predicted Killer", loc='best')
plt.tight_layout()
plt.savefig(str(q8_plotsPath / "test_pca_kmeans_predictions.png"), dpi=300)
plt.close()

# --------------------------------------------------------
# Save Results (Submission format for VAL + TEST)
# --------------------------------------------------------
inference_indices = ix_dev + ix_test
inference_preds = val_preds + test_preds

inference_ids = data.loc[inference_indices, "incident_id"].values

results = pd.DataFrame()
results['incident_id'] = inference_ids
results['predicted_killer'] = inference_preds

for k in range(1, 9):
    results[f'p_killer_{k}'] = 0.0

pred_array = np.array(inference_preds)
for k in range(1, 9):
    results.loc[results['predicted_killer'] == k, f'p_killer_{k}'] = 1.0

results = results.sort_values("incident_id")
predictions_path = projectDir / "data/predictions"
predictions_path.mkdir(parents=True, exist_ok=True)

results.to_csv(predictions_path / "Kmeans_pred.csv", index=False)
print(results.head())