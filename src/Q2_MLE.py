from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from data.loader import DataLoader, Filter

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

PROJECTION_PAIRS = [("latitude", "longitude"), ("latitude", "hour_float")]


def setup_paths():
    project_dir = Path(__file__).resolve().parent.parent
    plots_path = project_dir / "plots"

    q2_path = plots_path / "q2_mle"
    covariance_path = q2_path / "covariance_heatmaps"
    correlation_path = q2_path / "correlation_heatmaps"
    ellipse_path = q2_path / "ellipse_projections"

    for path in [covariance_path, correlation_path, ellipse_path]:
        path.mkdir(parents=True, exist_ok=True)

    return covariance_path, correlation_path, ellipse_path


def compute_mle_gaussian(data: np.ndarray):
    mu = data.mean(axis=0)
    centered = data - mu
    # division by N for MLE
    sigma = (centered.T @ centered) / len(data)
    return mu, sigma


def verify_mle(data: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    reg = 1e-6 * np.eye(len(mu))
    sigma_reg = sigma + reg

    rv = multivariate_normal(mean=mu, cov=sigma_reg, allow_singular=True)
    our_ll = rv.logpdf(data).sum()

    lib_mu = data.mean(axis=0)
    # numpy cov normalized by N-1 by default, so we disable bias correction for N-normalization comparison
    # wait, actually numpy cov(bias=True) normalizes by N.
    lib_sigma = np.cov(data, rowvar=False, bias=True) + reg

    lib_ll = (
        multivariate_normal(mean=lib_mu, cov=lib_sigma, allow_singular=True)
        .logpdf(data)
        .sum()
    )

    return np.abs(our_ll - lib_ll) < 1e-6, our_ll


def plot_covariance_heatmap(sigma: np.ndarray, killer_id: int, save_path: Path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        sigma,
        xticklabels=CONTINUOUS_FEATURES,
        yticklabels=CONTINUOUS_FEATURES,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
    )
    plt.title(f"Covariance Matrix for Killer {killer_id}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        save_path / f"covariance_killer_{killer_id}.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_correlation_heatmap(sigma: np.ndarray, killer_id: int, save_path: Path):
    std_devs = np.sqrt(np.diag(sigma))
    std_devs = np.where(std_devs == 0, 1e-10, std_devs)
    corr = np.nan_to_num(sigma / np.outer(std_devs, std_devs), nan=0.0)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        xticklabels=CONTINUOUS_FEATURES,
        yticklabels=CONTINUOUS_FEATURES,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
    )
    plt.title(f"Correlation Matrix for Killer {killer_id}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        save_path / f"correlation_killer_{killer_id}.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_ellipse(ax, mu, sigma, c_k, color):
    eigvals, eigvecs = np.linalg.eigh(sigma)
    width, height = 2 * np.sqrt(c_k * eigvals)
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    ellipse = Ellipse(
        xy=tuple(mu),
        width=width,
        height=height,
        angle=angle,
        facecolor="none",
        edgecolor=color,
        linewidth=2,
    )
    ax.add_patch(ellipse)


def plot_2d_ellipse(
    df_train: pd.DataFrame, killer_ids, feat_x: str, feat_y: str, save_path: Path
):
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    colors = plt.colormaps["tab10"](np.linspace(0, 1, len(killer_ids)))

    for idx, kid in enumerate(killer_ids):
        data_k = df_train.loc[df_train["killer_id"] == kid, [feat_x, feat_y]].values
        if len(data_k) < 3:
            continue

        mu, sigma = compute_mle_gaussian(data_k)
        sigma_inv = np.linalg.inv(sigma + 1e-6 * np.eye(2))

        mahal = np.array([(x - mu) @ sigma_inv @ (x - mu) for x in data_k])
        c_k = mahal.max()

        ax.scatter(
            data_k[:, 0],
            data_k[:, 1],
            c=[colors[idx]],
            alpha=0.6,
            s=30,
            label=f"Killer {kid}",
        )
        ax.scatter(mu[0], mu[1], c=[colors[idx]], s=150, marker="X", edgecolor="black")
        plot_ellipse(ax, mu, sigma, c_k, colors[idx])

    ax.set_xlabel(feat_x)
    ax.set_ylabel(feat_y)
    ax.set_title(f"2D Projection: {feat_x} vs {feat_y}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        save_path / f"ellipse_{feat_x}_vs_{feat_y}.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def q2():
    covariance_path, correlation_path, ellipse_path = setup_paths()

    dataloader = DataLoader()
    df_train = dataloader.split_filter(Filter.TRAIN)
    killer_ids = sorted(df_train["killer_id"].unique())

    mle_estimates = {}

    for kid in killer_ids:
        data_k = df_train.loc[df_train["killer_id"] == kid, CONTINUOUS_FEATURES].values
        mu_k, sigma_k = compute_mle_gaussian(data_k)
        is_valid, our_ll = verify_mle(data_k, mu_k, sigma_k)

        status = "PASS" if is_valid else "FAIL"
        print(f"Killer {kid}: N={len(data_k)}, LogLikelihood={our_ll:.2f}, {status}")
        mle_estimates[kid] = {"mu": mu_k, "sigma": sigma_k}

    for kid in killer_ids:
        sigma_k = mle_estimates[kid]["sigma"]
        plot_covariance_heatmap(sigma_k, kid, covariance_path)
        plot_correlation_heatmap(sigma_k, kid, correlation_path)

    for feat_x, feat_y in PROJECTION_PAIRS:
        plot_2d_ellipse(df_train, killer_ids, feat_x, feat_y, ellipse_path)

    return mle_estimates


if __name__ == "__main__":
    q2()
