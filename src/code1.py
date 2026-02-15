import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import os
import math

projectDir = Path(__file__).parent.parent

dataPath = projectDir / "data" / "crimes.csv"
plotsPath = projectDir / "plots"

data = pd.read_csv(str(dataPath))

# use only TRAIN and VAL data 
q1_data = data[data["split"] != "TEST"]

columns = data.columns.tolist()

numeric_feats = [
    "hour_float",
    "latitude",
    "longitude",
    "victim_age",
    "temp_c",
    "humidity",
    "dist_precinct_km",
    "pop_density",
]
# ----------------------------------------------------------
# Q1. Exploratory distributions (probability distributions)
# ----------------------------------------------------------
numeric_feats_for_plot = [
    "hour_float",
    "victim_age",
    "latitude",
    "longitude"         
]

# for every numeric feat create a Distribution_histplot
for i in range(len(numeric_feats_for_plot)):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Custom color for better visibility
    color = sns.color_palette("deep")[i]

    sns.histplot(
        data=q1_data,
        x=numeric_feats_for_plot[i],
        bins=20,
        kde=True,
        ax=ax,
        color=color,
        alpha=0.6,
    )
    # Set labels with improved clarity
    ax.set_xlabel(numeric_feats_for_plot[i], fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    # Adding skewness annotation
    skewness = q1_data[numeric_feats_for_plot[i]].skew()
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
    featuresPath = plotsPath / "Distribution_histplot_of_numeric_feats"
    os.makedirs(os.path.join(featuresPath), exist_ok=True)
    plt.savefig(
        str(featuresPath / (numeric_feats_for_plot[i] + "_Distribution_histplot.png")),
        dpi=300,
        bbox_inches="tight",
    )

#  For the variable hour_float
#  Fit a single Gaussian distribution N (µ, σ^2) using the sample mean and variance.
hour_float_mean = q1_data["hour_float"].mean()
print(f"hour_float_mean :{hour_float_mean}")
print(f"len of hour_float  :{len(q1_data["hour_float"])}")

variance = ( (q1_data["hour_float"] - hour_float_mean)**2 / len(q1_data["hour_float"])).sum()
print(f"variance :{variance}")

print(q1_data["hour_float"].describe())

