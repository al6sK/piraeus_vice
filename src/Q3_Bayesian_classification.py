from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import accuracy_score, confusion_matrix

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


def setup_paths():
    project_dir = Path(__file__).resolve().parent.parent
    plots_path = project_dir / "plots"

    q3_path = plots_path / "q3_bayes"
    q3_path.mkdir(parents=True, exist_ok=True)

    return q3_path


def compute_mle_gaussian(data: np.ndarray):
    if len(data) == 0:
        return np.zeros(data.shape[1]), np.eye(data.shape[1])

    mu = data.mean(axis=0)
    centered = data - mu
    # division by N for MLE
    sigma = (centered.T @ centered) / len(data)
    return mu, sigma


def fit_bayes_classifier(X, y):
    classes = sorted(np.unique(y))
    n_samples = len(y)

    model = {"classes": classes, "priors": {}, "means": {}, "covariances": {}}

    for k in classes:
        X_k = X[y == k]

        # prior is just frequency
        model["priors"][k] = len(X_k) / n_samples

        # mle estimates
        mu_k, sigma_k = compute_mle_gaussian(X_k)
        model["means"][k] = mu_k
        model["covariances"][k] = sigma_k

    return model


def predict_bayes(model, X):
    log_probs = []
    classes = model["classes"]
    reg = 1e-6 * np.eye(X.shape[1])

    for k in classes:
        mu_k = model["means"][k]
        sigma_k = model["covariances"][k]
        prior_k = model["priors"][k]

        try:
            rv = multivariate_normal(mean=mu_k, cov=sigma_k + reg, allow_singular=True)
            log_likelihood = rv.logpdf(X)
        except Exception:
            # fallback for numerical issues
            log_likelihood = np.full(len(X), -np.inf)

        log_posterior = np.log(prior_k + 1e-10) + log_likelihood
        log_probs.append(log_posterior)

    log_probs = np.column_stack(log_probs)
    indices = np.argmax(log_probs, axis=1)

    return np.array([classes[i] for i in indices])


def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        square=True,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def visualize_decision_boundaries(df_train, save_path):
    feat_x, feat_y = "latitude", "longitude"

    X_train_2d = df_train[[feat_x, feat_y]].values
    y_train = df_train["killer_id"].values

    # train 2d model
    model_2d = fit_bayes_classifier(X_train_2d, y_train)

    x_min, x_max = X_train_2d[:, 0].min() - 0.05, X_train_2d[:, 0].max() + 0.05
    y_min, y_max = X_train_2d[:, 1].min() - 0.05, X_train_2d[:, 1].max() + 0.05

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.005), np.arange(y_min, y_max, 0.005))

    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_bayes(model_2d, mesh_points)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(12, 10))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="tab10")

    scatter = plt.scatter(
        X_train_2d[:, 0],
        X_train_2d[:, 1],
        c=y_train,
        cmap="tab10",
        edgecolor="k",
        s=30,
        alpha=0.6,
    )

    plt.xlabel(feat_x)
    plt.ylabel(feat_y)
    plt.title(f"Decision Regions ({feat_x} vs {feat_y})")
    plt.colorbar(scatter, label="Killer ID")
    plt.tight_layout()
    plt.savefig(save_path / "decision_regions_lat_long.png", dpi=300)
    plt.close()


def q3():
    q3_path = setup_paths()

    dataloader = DataLoader()
    df_train = dataloader.split_filter(Filter.TRAIN)
    df_val = dataloader.split_filter(Filter.VAL)

    X_train = df_train[CONTINUOUS_FEATURES].values
    y_train = df_train["killer_id"].values

    X_val = df_val[CONTINUOUS_FEATURES].values
    y_val = df_val["killer_id"].values

    # training
    model = fit_bayes_classifier(X_train, y_train)

    # evaluation
    y_train_pred = predict_bayes(model, X_train)
    train_acc = accuracy_score(y_train, y_train_pred)

    y_val_pred = predict_bayes(model, X_val)
    val_acc = accuracy_score(y_val, y_val_pred)

    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Val Accuracy:   {val_acc:.4f}")

    plot_confusion_matrix(
        y_val,
        y_val_pred,
        classes=model["classes"],
        save_path=q3_path / "confusion_matrix_val.png",
    )

    visualize_decision_boundaries(df_train, q3_path)


if __name__ == "__main__":
    q3()
