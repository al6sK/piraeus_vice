import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

projectDir = Path(__file__).parent.parent
dataPath = projectDir / "data" / "crimes.csv"
plots_path = projectDir / "plots" / "Q5"
plots_path.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------
# Data Loading
# ----------------------------------------------------------
data = pd.read_csv(str(dataPath))

drop_cols = ["incident_id", "split", "killer_id", "weapon_code", "scene_type", "weather"]
cols_to_drop = [c for c in drop_cols if c in data.columns]

x = data.drop(columns=cols_to_drop).values
y_labels = data["killer_id"].values

# Split TRAIN / VAL
train_mask = (data["split"] == "TRAIN").values
val_mask = (data["split"] == "VAL").values

x_train = x[train_mask]
y_train = y_labels[train_mask]

x_val = x[val_mask]
y_val = y_labels[val_mask]

# ----------------------------------------------------------
# Scaling (StandardScaler) - Essential for SVM
# ----------------------------------------------------------
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

# ----------------------------------------------------------
# GridSearchCV with PredefinedSplit
# ----------------------------------------------------------
x_combined = np.vstack((x_train, x_val))
y_combined = np.concatenate((y_train, y_val))

# -1 = Train, 0 = Validation
test_fold = np.zeros(x_combined.shape[0])
test_fold[:len(x_train)] = -1 
test_fold[len(x_train):] = 0
ps = PredefinedSplit(test_fold=test_fold)

params = [
    {
        'kernel': ['rbf'],
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 0.01, 0.1, 1] 
    },
    {
        'kernel': ['poly'],
        'C': [0.1, 1, 10, 100],
        'degree': [2],      
        'coef0': [0, 1, 10] 
    }
]

model = SVC(decision_function_shape='ovr', random_state=42)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=params,
    scoring='accuracy',
    cv=ps,
    n_jobs=-1,
    verbose=1
)
grid_search.fit(x_combined, y_combined)
# ----------------------------------------------------------
# Evaluation (Unbiased - Retrain on TRAIN only)
# ----------------------------------------------------------
best_params = grid_search.best_params_
print(f"\nBest Params found: {best_params}")

final_model = SVC(**best_params, decision_function_shape='ovr', random_state=42)
final_model.fit(x_train, y_train) 

# Predict on VAL
y_val_pred = final_model.predict(x_val)
accuracy_val = accuracy_score(y_val, y_val_pred)

print(f"\nFinal SVM VAL Accuracy: {accuracy_val:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred))

# Confusion Matrix
cm = confusion_matrix(y_val, y_val_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=final_model.classes_)
fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, cmap='Blues')
plt.title(f"SVM Confusion Matrix\nAccuracy: {accuracy_val:.2%}")
plt.savefig(plots_path / "confusion_matrix_svm.png")

# ----------------------------------------------------------
# Visualization: Decision Regions & Support Vectors (PCA)
# ----------------------------------------------------------
print("\nGenerating Visualization...")

pca = PCA(n_components=2)
x_train_pca = pca.fit_transform(x_train)

svm_viz = SVC(**best_params, random_state=42)
svm_viz.fit(x_train_pca, y_train)

# Meshgrid
h = 0.02
x_min, x_max = x_train_pca[:, 0].min() - 1, x_train_pca[:, 0].max() + 1
y_min, y_max = x_train_pca[:, 1].min() - 1, x_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = svm_viz.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(12, 10))
# Decision Regions
plt.contourf(xx, yy, Z, alpha=0.3, cmap='tab10')
# Plot TRAIN data points
plt.scatter(x_train_pca[:, 0], x_train_pca[:, 1], c=y_train, cmap='tab10', edgecolor='k', s=40, alpha=0.6, label='Train Data')
# Highlight Support Vectors
sv = svm_viz.support_vectors_
plt.scatter(sv[:, 0], sv[:, 1], s=120, linewidth=1.5, facecolors='none', edgecolors='k', label='Support Vectors')
plt.title(f"SVM Decision Regions (PCA Space)\nKernel: {best_params['kernel']}")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig(plots_path / "svm_decision_regions.png")

# --------------------------------------------------------------------------------------------------
# GENERATE SUBMISSION CSV (VAL + TEST)
# --------------------------------------------------------------------------------------------------
submissions_path = projectDir / "data/test_submissions"
submissions_path.mkdir(parents=True, exist_ok=True)

final_model_prob = SVC(**best_params, decision_function_shape='ovr', probability=True, random_state=42)
final_model_prob.fit(x_train, y_train)

inference_mask = data["split"].isin(["VAL", "TEST"])
inference_data = data[inference_mask].copy()
inference_ids = inference_data["incident_id"].values

x_inference = inference_data.drop(columns=cols_to_drop).values
x_inference_scaled = scaler.transform(x_inference)

predictions = final_model_prob.predict(x_inference_scaled)
probabilities = final_model_prob.predict_proba(x_inference_scaled)

results = pd.DataFrame()
results.insert(0, 'incident_id', inference_data['incident_id'].values)
results['predicted_killer'] = predictions

for idx, class_label in enumerate(final_model_prob.classes_):
    results[f'p_killer_{class_label}'] = probabilities[:, idx]

predictions_path = projectDir / "data/predictions"
predictions_path.mkdir(parents=True, exist_ok=True)

results.to_csv(predictions_path / "SVM_pred.csv", index=False)

print(results.head())