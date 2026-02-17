from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal

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

plots_path = project_dir / "plots/Q2"
plots_path.mkdir(parents=True, exist_ok=True)

emp_corr_path = plots_path / "empirical_covariance"
emp_corr_path.mkdir(parents=True, exist_ok=True)

corr_matrix_path = plots_path / "correlation_matrix"
corr_matrix_path.mkdir(parents=True, exist_ok=True)

data = pd.read_csv("data/crimes.csv")
data = data[CONTINUOUS_FEATURES + ["split", "killer_id"]]

train_data = data[data["split"] == "TRAIN"]
train_data = train_data.drop(columns=["split"])

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

# calculate scores
Scores = []
for i in range(8):
    X = train_data[CONTINUOUS_FEATURES].values
    pdf = multivariate_normal.pdf(X, mean=M_k[i], cov=S_k[i], allow_singular=True)
    Scores.append(pdf * Prior_k[i])
scores = pd.DataFrame(Scores).T
# print(scores.head(10))

# Normalization
scores = scores.div(scores.sum(axis=1), axis=0)
# print(scores.sum(axis=1).head(10))

# ----------------------------------------------------------------------------------------------------------------------------
#  Hard decision Ci
# ----------------------------------------------------------------------------------------------------------------------------
Ci = 

# check line 84 <<<<<<<<<<<----------------------------------------------------