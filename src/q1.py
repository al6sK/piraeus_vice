from pathlib import Path
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from data.loader import DataLoader, Filter

MIXTURE_COMPONENTS = 3
RANDOM_STATE = 222
LIN_SPACE_NUM = 1000

PLOTS_CONFIG = {
    "hour_float": {"bins": 48},
    "latitude": {"bins": 40},
    "longitude": {"bins": 40},
    "victim_age": {"bins": 20},
}

def setup_paths():
    project_dir = Path(__file__).resolve().parent.parent

    plots_path = project_dir / "plots"
    features_path = plots_path / "distributions_histplots"
    fitted_path = plots_path / "fitted_plots"

    features_path.mkdir(parents=True, exist_ok=True)
    fitted_path.mkdir(exist_ok=True)

    return features_path, fitted_path

def plot_data(dataset, features_path):
    for column, config in PLOTS_CONFIG.items():
        plt.figure(column)
        sns.histplot(dataset[column], **config)
        plt.savefig(features_path / (column + ".png"))
        plt.close()

def hour_float_gaussian_fit(hour_float, fitted_path):
    x_min, x_max = hour_float.min(), hour_float.max()
    x_range = np.linspace(x_min, x_max, LIN_SPACE_NUM)

    mean = hour_float.mean()
    std = hour_float.std()

    plt.figure("Gaussian Fit")

    plt.title("Hour Float Distribution")
    sns.histplot(hour_float, stat="density", bins=PLOTS_CONFIG["hour_float"]["bins"])

    # calculating the Gaussian Distribution
    pdf_values = norm.pdf(x_range, mean, std)
    plt.plot(x_range, pdf_values, color="red", label=f"Fitted Gaussian mean={mean:.2f} std={std:.2f}")
    plt.legend()
    plt.savefig(fitted_path / "hour_float_GaussianFit.png")
    plt.close()


def gaussian_mixture_fit(hour_float, fitted_path):
    transformed_hour_float = hour_float.values.reshape(-1, 1)

    gmm = GaussianMixture(n_components=MIXTURE_COMPONENTS, random_state=RANDOM_STATE)
    gmm.fit(transformed_hour_float)

    plt.figure("GMM Fit")
    sns.histplot(hour_float, stat="density", color="lightgray", label="Data", bins=PLOTS_CONFIG["hour_float"]["bins"])

    x_min, x_max = hour_float.min(), hour_float.max()
    x_range = np.linspace(x_min, x_max, LIN_SPACE_NUM)
    total_pdf = np.zeros_like(x_range)

    for i in range(gmm.n_components):
        weight = gmm.weights_[i]
        mean = gmm.means_[i, 0]
        variance = gmm.covariances_[i, 0, 0]
        std = np.sqrt(variance)

        component_pdf = weight * norm.pdf(x_range, mean, std)
        total_pdf += component_pdf
        plt.plot(x_range, component_pdf, '--', alpha=0.7, label=f'Peak={i + 1} (weight={weight:.2f})')

    plt.plot(x_range, total_pdf, 'r-', linewidth=3, label='Combined PDF')
    plt.legend()
    plt.savefig(fitted_path / "GaussianMixtureFit.png")
    plt.close()

def plot_spatial_2d(dataset: pd.DataFrame):
    sns.histplot(
        dataset,
        x='hour_float',
        y='longitude',
        bins=PLOTS_CONFIG["hour_float"]["bins"],
    )
    plt.show()


def q1():
    features_path, fitted_path = setup_paths()
    dataloader = DataLoader()
    dataset = dataloader.split_filters([Filter.TRAIN, Filter.VAL])
    plot_data(dataset=dataset, features_path=features_path)
    hour_float = dataset["hour_float"]
    hour_float_gaussian_fit(hour_float, fitted_path)
    gaussian_mixture_fit(hour_float, fitted_path)
    plot_spatial_2d(dataset)


if __name__ == "__main__":
    q1()