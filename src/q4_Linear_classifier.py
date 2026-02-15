import pandas as pd
import numpy as np
import seaborn as sns

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

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
    cv = 5, 
    verbose = 1,
    return_train_score = False
)

# Model Training on TRAIN set
grid_search.fit(x_train, y_train)

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
final_model = grid_search.best_estimator_

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

ax.set_title("Confusion matrix for Linear classifier")
ax.set_xlabel("Predicted categories")
ax.set_ylabel("Actual categories")

plt.savefig(str(plots_path / "Confusion_matrix_for_Linear_classifier.png"), dpi=300, bbox_inches="tight")
plt.close()
# --------------------------------------------------------------------------------------------------
# In the 2D PCA projection used in Q3, overlay the approximate linear decision boundaries.
# --------------------------------------------------------------------------------------------------

# Q3 Dummy Data for PCA 
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)
y_pca = y_labels

x_pca_train = x_pca[train_data.values]
y_pca_train = y_pca[train_data.values]
x_pca_val = x_pca[val_data.values]
y_pca_val = y_pca[val_data.values]

# Q4 - Logistic Regression for best model found
final_model.fit(x_pca_train, y_pca_train)

# Q3 - Naive Bayes Model
gnb_model = GaussianNB()
gnb_model.fit(x_pca_train, y_pca_train)

# Decision Boundary Visualization

# Mesh Creation
h = 0.05  # step size in the mesh
x_min, x_max = x_pca[:, 0].min() - 0.5, x_pca[:, 0].max() + 0.5
y_min, y_max = x_pca[:, 1].min() - 0.5, x_pca[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

x_mesh = np.c_[xx.ravel(), yy.ravel()]

# Linear predictions
z_linear = final_model.predict(x_mesh).reshape(xx.shape) 
z_gnb = gnb_model.predict(x_mesh).reshape(xx.shape)

# Plotting 
plt.figure(figsize=(10, 7))

# 1 GaussianNB Decision Areas
cmap_gnb = ListedColormap(["salmon", "lightgreen", "mediumturquoise", "mediumslateblue", "plum", "orange", "royalblue", "forestgreen"])
plt.contourf(xx, yy, z_gnb, alpha = 0.5, cmap = cmap_gnb)


# 2 Linear Decision Boundaries
plt.contourf(xx, yy, z_linear, levels = np.arange(z_linear.max() + 2) - 0.5)


# Data points
cmap_data = ListedColormap(["red", "green", "blue", "purple", "pink", "brown", "cyan", "lime"])
scatterplot = plt.scatter(x_pca_val[:, 0], x_pca_val[:, 1], c = y_pca_val, cmap = cmap_data, edgecolors = 'k', s = 50, alpha = 0.8)

plt.xlabel(f"PCA Component 1: {pca.explained_variance_ratio_[0]*100:.2f}%")
plt.ylabel(f"PCA Component 2: {pca.explained_variance_ratio_[1]*100:.2f}%")
plt.title("Overlay: Q4. Logistic Regression Decision Boundaries - Q3. GaussianNB Decision Areas")
plt.grid(True, linestyle = '--', alpha = 0.7)

custom_lines =[
    Line2D([0], [0], color = "black", linewidth = 2),
    Line2D([0], [0], color = "gray", linewidth = 4),
]
plt.legend(custom_lines, ['Q4. Linear Decision Boundaries', 'Q3. GaussianNB Decision Areas'])

plt.show()