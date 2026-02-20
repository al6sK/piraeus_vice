# Alexios Kastanaras P22062, 
# Danai Charzaka P22194,
# Dimitrios Lazanas P22082
# Contact email for the group: alexioskast@gmail.com

from pathlib import Path
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

numeric_feats_for_plot = [
    "hour_float",
    "victim_age",
    "latitude",
    "longitude"         
]

# ----------------------------------------------------------------------------------------------------------------------------
# setup_paths and split data 
# ----------------------------------------------------------------------------------------------------------------------------
project_dir = Path(__file__).resolve().parent.parent
plots_path = project_dir / "plots/Q1"
features_path = plots_path / "distributions_histplots"
features_path.mkdir(parents=True, exist_ok=True)

data = pd.read_csv("data/crimes.csv")
data = data[data["split"] != "TEST"]

# ----------------------------------------------------------------------------------------------------------------------------
# 1) Plot histograms for the one-dimensional distributions of hour_float, victim_age, latitude, longitude (using TRAIN + VAL).
# ----------------------------------------------------------------------------------------------------------------------------
for i in range(len(numeric_feats_for_plot)):
    _, ax = plt.subplots(1, 1, figsize=(8, 5))
    # Custom color for better visibility
    color = sns.color_palette("deep")[i]
    sns.histplot(
        data=data,
        x=numeric_feats_for_plot[i],
        bins=50,
        kde=True,
        ax=ax,
        color=color,
        alpha=0.6,
    )
    # Set labels with improved clarity
    ax.set_xlabel(numeric_feats_for_plot[i], fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    # Adding skewness annotation
    skewness = data[numeric_feats_for_plot[i]].skew()
    ax.text(
        0.95,
        0.85,
        f"Skewness: {skewness:.2f}",
        transform=ax.transAxes,
        ha="right",
        color="black",
        weight="bold",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
    )
    # Add title and subtitle
    ax.set_title(
        numeric_feats_for_plot[i] + " Distribution",
        fontsize=16,
        fontweight="bold",
        loc="left",
        pad=20,
    )
    plt.figtext(
        0.1,
        0.9,
        "Histogram with KDE overlay",
        fontsize=10,
        ha="left",
    )     
    # Grid and despine for a cleaner look
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    sns.despine(left=True)
    plt.tight_layout()
    plt.savefig(str(features_path / (numeric_feats_for_plot[i] + ".png")),dpi=300,bbox_inches="tight",)
    plt.close()

# ----------------------------------------------------------------------------------------------------------------------------
# 2) For the variable hour_float:
# ----------------------------------------------------------------------------------------------------------------------------
# a) calculate single Gaussian distribution N (µ, σ2) using the sample mean and variance
x_min, x_max = data["hour_float"].min(), data["hour_float"].max()
x_range = np.linspace(x_min, x_max, 1000)

single_mean = data["hour_float"].mean()
single_std = data["hour_float"].std()

single_pdf = norm.pdf(x_range, single_mean, single_std)

# b) calculate a 3-component 1D Gaussian mixture model
transformed_hour_float = data["hour_float"].values.reshape(-1, 1)
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(transformed_hour_float)
total_pdf = np.zeros_like(x_range)

# c) plot both together
_, ax = plt.subplots(1, 1, figsize=(8, 5))
# Custom color for better visibility
color = "gray"
sns.histplot(
    data=data,
    x="hour_float",
    bins=50,
    kde=False,
    stat="density",
    ax=ax,
    color=color,
    alpha=0.3,
)
plt.plot(x_range, single_pdf, color="red", linestyle="-",alpha=0.7, linewidth=3, label="Single Gaussian distribution")
colors = ["green" , "blue" , "orange"]
for i in range(gmm.n_components):
    weight = gmm.weights_[i]
    mean = gmm.means_[i, 0]
    variance = gmm.covariances_[i, 0, 0]
    std = np.sqrt(variance)
    component_pdf = weight * norm.pdf(x_range, mean, std)
    total_pdf += component_pdf
    plt.plot(
        x_range,
        component_pdf,
        "-",
        linewidth=2.5,
        color=colors[i],
        alpha=0.7,
        label=f"Peak={i + 1} (weight={weight:.2f})",
    )
plt.plot(x_range, total_pdf, "purple",alpha=0.7, linewidth=3, label="Combined PDF")
ax.legend(loc='upper right', fontsize=10)
# Set labels with improved clarity
ax.set_xlabel("hour_float", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
# Add title and subtitle
ax.set_title(
    "hour_float" + " Gaussian Fit",
    fontsize=16,
    fontweight="bold",
    loc="left",
    pad=20,
)
plt.figtext(
        0.1,
        0.9,
        "Histogram of hour_float with the single Gaussian density, and the 3-component mixture density",
        fontsize=10,
        ha="left",
    )
# Grid and despine for a cleaner look
ax.grid(axis="y", linestyle="--", alpha=0.6)
sns.despine(left=True)
plt.tight_layout()
plt.savefig(plots_path / "hour_float_gaussian_vs_gmm.png",dpi=300,bbox_inches="tight",)
plt.close()

# ----------------------------------------------------------------------------------------------------------------------------
# 3) Two-dimensional plot involving hour_float vs hour_float and longitude
# ----------------------------------------------------------------------------------------------------------------------------
def plot_spatial_2d(data,y):
    _, ax = plt.subplots(1, 1, figsize=(8, 5))
    # Custom color for better visibility
    color = sns.color_palette("deep")[0]
    sns.histplot(
        data=data,
        x="hour_float",
        y=y,
        bins=50,
        kde=False,
        stat="density",
        ax=ax,
        color=color,
        # alpha=0.6,
    )
    # Set labels with improved clarity
    ax.set_xlabel("hour_float", fontsize=12)
    ax.set_ylabel(y, fontsize=12)
    # Add title and subtitle
    ax.set_title(
        f"hour_float vs {y}",
        fontsize=16,
        fontweight="bold",
        loc="left",
        pad=20,
    )  
    # Grid and despine for a cleaner look
    # ax.grid(axis="y", linestyle="--", alpha=0.6)
    sns.despine(left=True)
    plt.tight_layout()
    plt.savefig(str(plots_path / ("hour_float vs "+ y + ".png")),dpi=300,bbox_inches="tight",)
    plt.close()

plot_spatial_2d(data,"latitude")
plot_spatial_2d(data,"longitude")

