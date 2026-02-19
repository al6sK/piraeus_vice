# Alexios Kastanaras P22062, 
# Danai Harzaka P22194,
# Dimitrios Lazanas P22082
# Contact email for the group: alexioskast@gmail.com

from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

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
        vmin=-(max(abs(corr.min()), abs(corr.max()))),
        vmax=max(abs(corr.min()), abs(corr.max())),  # Ensure that color scaling is consistent
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
    plt.close()
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
    plt.close()
# ----------------------------------------------------------------------------------------------------------------------------
# c) Visualise ellipses (Mahalanobis distance)
# ----------------------------------------------------------------------------------------------------------------------------
ellipses_path = plots_path / "ellipses"
ellipses_path.mkdir(parents=True, exist_ok=True)

def plot_mahalanobis_ellipse(x_feat, y_feat, feature_names, means_list, covs_list, train_df):
    x_idx = feature_names.index(x_feat)
    y_idx = feature_names.index(y_feat)
    
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    colors = plt.cm.tab10(np.linspace(0, 1, 8)) 

    for k in range(8):
        killer_id = k + 1
        
        killer_data = train_df[train_df["killer_id"] == killer_id][[x_feat, y_feat]].values
        
        mean_2d = np.array([means_list[k][x_idx], means_list[k][y_idx]])
        
        cov_2d = np.array([
            [covs_list[k][x_idx, x_idx], covs_list[k][x_idx, y_idx]],
            [covs_list[k][y_idx, x_idx], covs_list[k][y_idx, y_idx]]
        ])
        
        diff = killer_data - mean_2d
        inv_cov = np.linalg.inv(cov_2d)
        
        mahal_sq = np.sum((diff @ inv_cov) * diff, axis=1)
        
        c_k = np.max(mahal_sq)
        
        eigvals, eigvecs = np.linalg.eigh(cov_2d)
        
        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        
        vx, vy = eigvecs[:, 0][0], eigvecs[:, 0][1]
        theta = np.degrees(np.arctan2(vy, vx))
        
        width = 2 * np.sqrt(c_k * eigvals[0])
        height = 2 * np.sqrt(c_k * eigvals[1])
        
        plt.scatter(killer_data[:, 0], killer_data[:, 1], s=10, color=colors[k], alpha=0.6, label=f'Killer {killer_id}')
        
        ell = Ellipse(xy=mean_2d, width=width, height=height, angle=theta, edgecolor=colors[k], facecolor='none', linewidth=2, linestyle='--')
        ax.add_patch(ell)

    plt.title(f"Mahalanobis Ellipses: {x_feat} vs {y_feat}", fontsize=16, fontweight="bold")
    plt.xlabel(x_feat)
    plt.ylabel(y_feat)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ellipses_path / f"ellipse_{x_feat}_vs_{y_feat}.png", dpi=300)
    plt.close()

projections = [
    ("longitude", "latitude"),      
    ("hour_float", "latitude")    
]

for x_f, y_f in projections:
    plot_mahalanobis_ellipse(x_f, y_f, CONTINUOUS_FEATURES, M_k, S_k, train_data)
plt.close()