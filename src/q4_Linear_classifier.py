# Alexios Kastanaras P22062, 
# Danai Harzaka P22194,
# Dimitrios Lazanas P22082
# Contact email for the group: alexioskast@gmail.com

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

project_dir = Path(__file__).resolve().parent.parent
plots_path = project_dir / "plots/Q4"
plots_path.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------------------
# a) Train the linear classifier on TRAIN (and choose any regularisation hyperparameters using VAL).
# --------------------------------------------------------------------------------------------------

# Data Preprocessing
data = pd.read_csv(str("data/data_encoded.csv"))

# Preparing input features x and target labels y
y_labels = data["killer_id"].values
y_onehot = pd.get_dummies(y_labels).values

# Labels to drop 
drop_cols = ["incident_id", "split", "killer_id", "weapon_code", "scene_type", "weather"]
x = data.drop(columns=drop_cols).values

# splitting data to TRAIN and VAL 
train_data = data["split"] == "TRAIN"
val_data = data["split"] == "VAL"

# training
x_train = x[train_data.values]
y_train = y_labels[train_data.values]

# validation
x_val = x[val_data.values]
y_val = y_labels[val_data.values]
y_val_onehot = y_onehot[val_data.values]

x_full = np.vstack((x_train, x_val))
y_full = np.concatenate((y_train, y_val))

test_fold = np.zeros(x_full.shape[0])
test_fold[:len(x_train)] = -1 
test_fold[len(x_train):] = 0  

ps = PredefinedSplit(test_fold)
# SSE Scorer Function
def sse_scorer(estimator, x, y):

    # Predicted probs for each class
    y_pred_proba = estimator.predict_proba(x)

    # Using reindex to ensure all classes are represented
    y_onehot = pd.get_dummies(y).reindex(columns=range(y_pred_proba.shape[1]), fill_value=0).values

    sse = np.sum((y_onehot - y_pred_proba)**2)

    # Negative SSE for maximization
    return -sse

# Hyperparameters Tuning using GridSearchCV

# Finding the optimal C
C = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000,100000]
best_C = None

best_accuracy = 0.0
min_sse = np.inf 

res = []

alpha = [ 1/c for c in C]

# Logistic Regression base model using SGDClassifier
base = SGDClassifier(
    alpha = alpha, 
    loss = "log_loss",
    penalty = "l2",
    max_iter = 1000,
    random_state = 42
)

# Wrapping it with OneVsRestClassifier for multiclass classification
model = OneVsRestClassifier(base)

# Setting up parameter grid for GridSearchCV
param_grid = { 'estimator__alpha': alpha}

# Setting up metrics we want to observe after GridSearchCV
scorers = {
    'Accuracy' : make_scorer(accuracy_score),
    'SSE': sse_scorer
}

grid_search = GridSearchCV(
    estimator = model,
    param_grid = param_grid,
    scoring = scorers,
    refit = 'Accuracy',
    cv = ps,
    verbose = 1,
    return_train_score = False
)

# Model Training on TRAIN set
grid_search.fit(x_full, y_full)

# Results
print("\nHyperparameter Tuning Results with GridSearchCV:\n")
best_alpha = grid_search.best_params_['estimator__alpha']
best_C = 1 / best_alpha
min_sse = -grid_search.cv_results_['mean_test_SSE'][grid_search.best_index_]
print(f"Best C: {best_C:.1f}")
print(f"Best Alpha: {best_alpha:.6f}")
print(f"Best Accuracy {grid_search.best_score_:.4f}")
print(f"Minimum SSE: {min_sse:.4f}")

# --------------------------------------------------------------------------------------------------
# Report VAL accuracy and confusion matrix. Compare with Q3
# --------------------------------------------------------------------------------------------------
best_params = grid_search.best_params_
best_alpha = best_params['estimator__alpha']

final_model = OneVsRestClassifier(
    SGDClassifier(
        loss="log_loss",
        alpha=best_alpha,
        penalty="l2",
        max_iter=1000,
        random_state=42
    )
)
final_model.fit(x_train, y_train)
# Predictions
y_val_pred = final_model.predict(x_val)
y_val_proba = final_model.predict_proba(x_val)

# Accuracy Calculation
final_accuracy_val = accuracy_score(y_val, y_val_pred)

# SSE Calculation
final_sse = np.sum((y_val_onehot - y_val_proba)**2)

print(f"\nFinal VAL Accuracy: {final_accuracy_val:.4f}")
print(f"Final VAL SSE: {final_sse:.4f}")

# Confusion Matrix Calculation
conf_mtrx = confusion_matrix(y_val, y_val_pred)
print("Confusion Matrix:") 
print(conf_mtrx)

# --------------------------------------------------------------
# Confusion Matrix Calculation
# --------------------------------------------------------------
cm = confusion_matrix(y_val, y_val_pred)

labels = final_model.classes_
_, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax)

ax.set_title(f"Confusion matrix for Linear classifier\nAccuracy: {final_accuracy_val:.2%}", fontsize=16, fontweight="bold")
ax.set_xlabel("Predicted categories")
ax.set_ylabel("Actual categories")

plt.savefig(str(plots_path / "Confusion_matrix_for_Linear_classifier.png"), dpi=300, bbox_inches="tight")
plt.close()

# --------------------------------------------------------------------------------------------------
# c) Overlay approximate linear decision boundaries on PCA projection
# --------------------------------------------------------------------------------------------------

CONTINUOUS_FEATURES = [
    "hour_float", 
    "latitude", 
    "longitude", 
    "victim_age",
    "temp_c", 
    "humidity", 
    "dist_precinct_km", 
    "pop_density"
]

df_viz = pd.read_csv("data/data_encoded.csv")
x_cont = df_viz[CONTINUOUS_FEATURES].values
y_viz = df_viz["killer_id"].values
train_mask_viz = (df_viz["split"] == "TRAIN").values
val_mask_viz = (df_viz["split"] == "VAL").values

x_cont_train = x_cont[train_mask_viz]
x_cont_val = x_cont[val_mask_viz]
y_train_viz = y_viz[train_mask_viz]
y_val_viz = y_viz[val_mask_viz]

pca = PCA(n_components=2)
x_train_pca = pca.fit_transform(x_cont_train) 
x_val_pca = pca.transform(x_cont_val)        

gnb_2d = GaussianNB()
gnb_2d.fit(x_train_pca, y_train_viz)


linear_2d = OneVsRestClassifier(
    SGDClassifier(
        loss="log_loss",      
        alpha=best_alpha,     
        penalty="l2", 
        max_iter=2000, 
        random_state=42
    )
)
linear_2d.fit(x_train_pca, y_train_viz)

h = 0.02
x_min, x_max = x_train_pca[:, 0].min() - 1, x_train_pca[:, 0].max() + 1
y_min, y_max = x_train_pca[:, 1].min() - 1, x_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
mesh_points = np.c_[xx.ravel(), yy.ravel()]

Z_bayes = gnb_2d.predict(mesh_points).reshape(xx.shape)
Z_linear = linear_2d.predict(mesh_points).reshape(xx.shape)

plt.figure(figsize=(12, 10))
plt.contourf(xx, yy, Z_bayes, alpha=0.3, cmap='tab10')
plt.contour(xx, yy, Z_linear, colors='k', linewidths=2, linestyles='--')
scatter = plt.scatter(x_val_pca[:, 0], x_val_pca[:, 1], c=y_val_viz, cmap='tab10', edgecolor='k', s=60, alpha=0.8)
legend_elements = [
    Patch(facecolor='grey', alpha=0.3, label='Bayes Regions (Non-Linear)'),
    Line2D([0], [0], color='k', lw=2, linestyle='--', label='Linear Boundaries (Approximation)')
]
plt.legend(handles=legend_elements, loc='upper right')
plt.title("Approximate Linear Boundaries on Q3 PCA Projection", fontsize=16, fontweight="bold")
plt.xlabel("PC1 (Continuous Features)")
plt.ylabel("PC2 (Continuous Features)")
plt.tight_layout()
plt.savefig(plots_path / "overlay_linear_bayes_corrected.png", dpi=300)

# --------------------------------------------------------------------------------------------------
# Generate Submission CSV (VAL + TEST predictions)
# --------------------------------------------------------------------------------------------------
inference_mask = data["split"].isin(["VAL", "TEST"])
inference_data = data[inference_mask].copy()
x_inference = inference_data.drop(columns=drop_cols).values

predictions = final_model.predict(x_inference)
probabilities = final_model.predict_proba(x_inference)

results = pd.DataFrame()
results['predicted_killer'] = predictions
results.insert(0, 'incident_id', inference_data['incident_id'].values)
for idx, class_label in enumerate(final_model.classes_):
    col_name = f'p_killer_{class_label}'
    results[col_name] = probabilities[:, idx]

print(f"Submission shape: {results.shape}")
print(results.head())

predictions_path = project_dir / "data/predictions"
predictions_path.mkdir(parents=True, exist_ok=True)

results.to_csv(predictions_path / "Linear_classifier_pred.csv", index=False)
