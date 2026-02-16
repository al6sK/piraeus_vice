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

print(train_data.columns)
# ----------------------------------------------------------------------------------------------------------------------------
# Find for each killer the number of incidents N_k
# ----------------------------------------------------------------------------------------------------------------------------
N_k = []
for i in range(8):
    N_k.append(len(train_data[train_data["killer_id"] == i + 1]))
print(N_k)
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
print(M_k)

S_k = []
for i in range(8):
    killer_incidents = train_data[train_data["killer_id"] == i + 1]

    Covariance_Matrix = ( (killer_incidents[CONTINUOUS_FEATURES].values - M_k[i]).T ) @ (killer_incidents[CONTINUOUS_FEATURES].values - M_k[i]) / N_k[i]

    S_k.append(Covariance_Matrix)
print(S_k)
# ----------------------------------------------------------------------------------------------------------------------------
# b) Verify numerically that the log-likelihood produced by your estimates matches (up to numerical tolerance) that of a trusted library.
# ----------------------------------------------------------------------------------------------------------------------------
for i in range(8):
    X = train_data[train_data["killer_id"] == i + 1][CONTINUOUS_FEATURES].values
    my_pdf = multivariate_normal.logpdf(X, mean=M_k[i], cov=S_k[i], allow_singular=True)
    my_log_likelihood = np.sum(my_pdf)

    lib_mean = np.mean(X, axis=0)
    lib_cov = np.cov(X, rowvar=False, bias=True) 
    lib_pdf = multivariate_normal.logpdf(X, mean=lib_mean, cov=lib_cov, allow_singular=True)
    lib_log_likelihood = np.sum(lib_pdf)

    print(f"My LL: {my_log_likelihood}")
    print(f"Lib LL: {lib_log_likelihood}")
# ----------------------------------------------------------------------------------------------------------------------------
# c) Visualise, for each killer k
# -   -   -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    - 
# Empirical Covariance
# ----------------------------------------------------------------------------------------------------------------------------
for i in range(8):    
    # Calculate the correlation matrix
    corr = S_k[i].round(2)

    # Create the mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Create the heatmap with the mask
    plt.figure(figsize=(10, 10))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        vmin=-1,
        vmax=1,  # Ensure that color scaling is consistent
        cbar_kws={"label": "Correlation Coefficient"},
        annot_kws={"size": 10},  # Adjust annotation size
        xticklabels=CONTINUOUS_FEATURES, 
        yticklabels=CONTINUOUS_FEATURES, 
        mask=mask,  # Apply the mask to hide the upper triangle
    )

    # Title and labels for context
    plt.title("Empirical Covariance Matrix of Numerical Features", fontsize=16, fontweight="bold")
    plt.xlabel("Features", fontsize=12)
    plt.ylabel("Features", fontsize=12)

    # Rotate the axis labels for better readability
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0, ha="right")

    plt.tight_layout()
    plt.savefig(emp_corr_path / f"emp_corr_of_killer_{i+1}.png", dpi=300, bbox_inches="tight")
# ----------------------------------------------------------------------------------------------------------------------------
# correlation matrix
# ----------------------------------------------------------------------------------------------------------------------------
for i in range(8):
    variances = np.diag(S_k[i])
    std_devs = np.sqrt(variances)
    std_devs[std_devs == 0] = 1e-15
    corr = (S_k[i] / np.outer(std_devs, std_devs)).round(2)
    # Create the mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Create the heatmap with the mask
    plt.figure(figsize=(10, 10))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        vmin=-1,
        vmax=1,  # Ensure that color scaling is consistent
        cbar_kws={"label": "Correlation Coefficient"},
        annot_kws={"size": 10},  # Adjust annotation size
        xticklabels=CONTINUOUS_FEATURES, 
        yticklabels=CONTINUOUS_FEATURES, 
        mask=mask,  # Apply the mask to hide the upper triangle
    )

    # Title and labels for context
    plt.title("Correlation Matrix of Numerical Features", fontsize=16, fontweight="bold")
    plt.xlabel("Features", fontsize=12)
    plt.ylabel("Features", fontsize=12)

    # Rotate the axis labels for better readability
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0, ha="right")

    plt.tight_layout()
    plt.savefig(corr_matrix_path / f"corr_matrix_of_killer_{i+1}.png", dpi=300, bbox_inches="tight")
