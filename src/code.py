import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("data/crimes.csv")
print(data.head())

columns = data.columns.tolist()
print(columns)

numeric_feats = [
    'hour_float', 
    'latitude', 
    'longitude', 
    'victim_age', 
    'temp_c', 
    'humidity',
    'dist_precinct_km',
    'pop_density'
    ]


#for every numeric feat create a Distribution_histplot
for i in range(len(numeric_feats)):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Custom color for better visibility
    color = sns.color_palette("deep")[i]

    sns.histplot(
        data=data,
        x=numeric_feats[i],
        bins=20,
        kde=True,
        ax=ax,
        color=color,
        alpha=0.6,
    )
    # Set labels with improved clarity
    ax.set_xlabel(numeric_feats[i], fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    # Adding skewness annotation
    skewness = data[numeric_feats[i]].skew()
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
        numeric_feats[i]+" Distribution",
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
    plt.savefig("plots/Distribution_histplot_of_numeric_feats/"+numeric_feats[i]+"_Distribution_histplot.png", dpi=300, bbox_inches="tight")
