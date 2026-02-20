# Alexios Kastanaras P22062, 
# Danai Charzaka P22194,
# Dimitrios Lazanas P22082
# Contact email for the group: alexioskast@gmail.com

from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors

CONTINUOUS_FEATURES = [
    "hour_float",
    "latitude",
    "longitude",
    "victim_age",
    "temp_c",
    "humidity",
    "dist_precinct_km",
    "pop_density",
]

# ----------------------------------------------------------------------------------------------------------------------------
# setup_paths and split data 
# ----------------------------------------------------------------------------------------------------------------------------
project_dir = Path(__file__).resolve().parent.parent

plots_path = project_dir / "plots/Q3"
plots_path.mkdir(parents=True, exist_ok=True)

data = pd.read_csv("data/data_encoded.csv")
data = data[CONTINUOUS_FEATURES + ["split", "incident_id", "killer_id"]]

train_data = data[data["split"] == "TRAIN"]
train_data = train_data.drop(columns=["split"])

val_data = data[data["split"] == "VAL"]
val_data = val_data.drop(columns=["split"])

test_data = data[data["split"] == "TEST"]
test_data = test_data.drop(columns=["split"])

# print(train_data.columns)
# ----------------------------------------------------------------------------------------------------------------------------
# From Q2: - Find for each killer the number of incidents N_k
# ----------------------------------------------------------------------------------------------------------------------------
N_k = []
for i in range(8):
    N_k.append(len(train_data[train_data["killer_id"] == i + 1]))
# print(N_k)
# ----------------------------------------------------------------------------------------------------------------------------
# a) Derive the Maximum Likelihood Estimators
# ----------------------------------------------------------------------------------------------------------------------------
M_k = []
for i in range(8):
    killer_incidents = train_data[train_data["killer_id"] == i + 1]

    mean = []
    for x in CONTINUOUS_FEATURES:
        mean.append(killer_incidents[x].mean())
    M_k.append(mean)
# print(M_k)

S_k = []
for i in range(8):
    killer_incidents = train_data[train_data["killer_id"] == i + 1]

    Covariance_Matrix = ( (killer_incidents[CONTINUOUS_FEATURES].values - M_k[i]).T ) @ (killer_incidents[CONTINUOUS_FEATURES].values - M_k[i]) / N_k[i]

    S_k.append(Covariance_Matrix)
# print(S_k)

# ----------------------------------------------------------------------------------------------------------------------------
# Prior
# ----------------------------------------------------------------------------------------------------------------------------
Prior_k = []
for i in range(8):
    Prior_k.append(N_k[i]/len(train_data))
# print(Prior_k)

# ----------------------------------------------------------------------------------------------------------------------------
# Posterior Probability
# ----------------------------------------------------------------------------------------------------------------------------
def Posterior_Probabilities(data):
    scores = []
    for i in range(8):
        X = data[CONTINUOUS_FEATURES].values
        pdf = multivariate_normal.pdf(X, mean=M_k[i], cov=S_k[i], allow_singular=True)
        scores.append(pdf * Prior_k[i])
    scores = pd.DataFrame(scores).T
    # Normalization
    scores = scores.div(scores.sum(axis=1), axis=0)
    return scores

train_Post_Prob = Posterior_Probabilities(train_data)
# print(train_Post_Prob.sum(axis=1).head(10))

# ----------------------------------------------------------------------------------------------------------------------------
#  Hard decision Ci of the TRAIN 
# ----------------------------------------------------------------------------------------------------------------------------
train_Post_Prob['Ci'] = train_Post_Prob.idxmax(axis=1) + 1 # the +1 is for the column names that start from 0-7
# print(scores['Ci'].head(10))

# ----------------------------------------------------------------------------------------------------------------------------
#  b) Evaluate the classifier on TRAIN (sanity check) and VAL (generalisation):
# ----------------------------------------------------------------------------------------------------------------------------

# TRAIN sanity check
train_accuracy = (train_Post_Prob['Ci'].values == train_data["killer_id"].values).astype(int)
print(f"TRAIN accuracy: {train_accuracy.sum()/len(train_data):.2%} with {train_accuracy.sum(axis=-1)} out of {len(train_data)}")

# VAL generalisation
val_Post_Prob = Posterior_Probabilities(val_data)
val_Post_Prob['Ci'] = val_Post_Prob.idxmax(axis=1) + 1

val_accuracy = (val_Post_Prob['Ci'].values == val_data["killer_id"].values).astype(int)
print(f"VAL accuracy: {val_accuracy.sum()/len(val_data):.2%} with {val_accuracy.sum(axis=-1)} out of {len(val_data)}")

# --------------------------------------------------------------
# VAL Confusion Matrix Calculation
# --------------------------------------------------------------
val_pred = val_Post_Prob['Ci'].values
val_y = val_data["killer_id"].values

cm = confusion_matrix(val_y, val_pred)

labels = range(1,9)
_, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax)

ax.set_title(f"Confusion matrix of Bayesian classification on VAL\nAccuracy: {val_accuracy.sum()/len(val_data):.2%}", fontsize=16, fontweight="bold")
ax.set_xlabel("Predicted categories")
ax.set_ylabel("Actual categories")

plt.savefig(str(plots_path / "Confusion_matrix_of_VAL.png"), dpi=300, bbox_inches="tight")
plt.close()

# --------------------------------------------------------------
# c) PCA projection
# --------------------------------------------------------------
# Load Normalized data
data_norm = pd.read_csv("data/data_encoded.csv")
train_data_norm = data_norm[data_norm["split"] == "TRAIN"]
train_data_norm = train_data_norm.drop(columns=["split"])
train_data_norm = train_data_norm[CONTINUOUS_FEATURES + ["killer_id"]]

# PCA
X = train_data_norm.drop(columns=["killer_id"]).copy()
y = train_data_norm["killer_id"]

pca = PCA(n_components=2)
train_pca = pca.fit_transform(X.to_numpy())

h = 0.02 
x_min, x_max = train_pca[:, 0].min() - 0.2, train_pca[:, 0].max() + 0.2
y_min, y_max = train_pca[:, 1].min() - 0.2, train_pca[:, 1].max() + 0.2
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

grid_points_2d = np.c_[xx.ravel(), yy.ravel()]

grid_points_8d = pca.inverse_transform(grid_points_2d)

grid_df = pd.DataFrame(grid_points_8d, columns=CONTINUOUS_FEATURES)
grid_probs = Posterior_Probabilities(grid_df)
grid_preds = grid_probs.fillna(0).idxmax(axis=1) + 1  # killer_id 1-8

Z = grid_preds.values.reshape(xx.shape)

colors = plt.cm.Set1.colors[:8]
cmap_8 = mcolors.ListedColormap(colors)

plt.figure(figsize=(10, 8))

levels = np.arange(0.5, 9.5, 1) 
contour = plt.contourf(xx, yy, Z, levels=levels, alpha=0.3, cmap=cmap_8)

scatter = plt.scatter(
    train_pca[:, 0], 
    train_pca[:, 1], 
    c=train_data_norm["killer_id"], 
    cmap=cmap_8, 
    vmin=1, 
    vmax=8,
    edgecolor='k', 
    alpha=0.7
)

plt.title("Bayesian Decision Regions in a two-dimensional projection", fontsize=16, fontweight="bold")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(str(plots_path / "two_dimensional_projection.png"), dpi=300, bbox_inches="tight")

# --------------------------------------------------------------
# Calculate probabilities for TEST data
# --------------------------------------------------------------
inference_mask = data["split"].isin(["VAL", "TEST"])
inference_data = data[inference_mask].copy().reset_index(drop=True)

probs_df = Posterior_Probabilities(inference_data)
predictions = probs_df.idxmax(axis=1) + 1

results = pd.DataFrame()
results.insert(0, 'incident_id', inference_data['incident_id'].values)
results['predicted_killer'] = predictions.values

for i in range(8):
    col_name = f'p_killer_{i+1}'
    results[col_name] = probs_df.iloc[:, i].values

results = results.sort_values("incident_id").reset_index(drop=True)

print(f"Submission shape: {results.shape}")
print(results.head())

predictions_path = project_dir / "data/predictions"
predictions_path.mkdir(parents=True, exist_ok=True)
results.to_csv(predictions_path / "bayesian_pred.csv", index=False)