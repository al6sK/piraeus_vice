import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# ---------------------------------------------------------
# Setup Paths & Configuration
# ---------------------------------------------------------
project_dir = Path(__file__).parent.parent 
predictions_dir = project_dir / "data" / "predictions"
data_path = project_dir / "data" / "data_encoded.csv" 
output_submission_path = project_dir / "submission.csv"
plots_path = project_dir / "plots" / "submission"
plots_path.mkdir(parents=True, exist_ok=True)

files_to_ensemble = [
    # "bayesian_pred.csv",
    # "Linear_classifier_pred.csv",
    "MLP_pred.csv",
    "SVM_pred.csv",
]

# ---------------------------------------------------------
# Soft Voting Ensemble Implementation
# ---------------------------------------------------------
dfs = []
valid_files = []

for filename in files_to_ensemble:
    file_path = predictions_dir / filename
    df = pd.read_csv(file_path)
    df = df.sort_values("incident_id").reset_index(drop=True)
    dfs.append(df)
    valid_files.append(filename)

base_ids = dfs[0]["incident_id"].values
print(f"Averaging probabilities from {len(dfs)} models...")
prob_cols = [f"p_killer_{i}" for i in range(1, 9)]
sum_probs = np.zeros((len(dfs[0]), 8))
for df in dfs:
    sum_probs += df[prob_cols].values

avg_probs = sum_probs / len(dfs)

# ---------------------------------------------------------
# Create Final Submission DataFrame
# ---------------------------------------------------------
submission_df = pd.DataFrame()
submission_df["incident_id"] = base_ids
submission_df["predicted_killer"] = np.argmax(avg_probs, axis=1) + 1
for i, col in enumerate(prob_cols):
    submission_df[col] = avg_probs[:, i]
submission_df.to_csv(output_submission_path, index=False)
# ---------------------------------------------------------
# Evaluation against Ground Truth
# ---------------------------------------------------------
print("\n--- Starting Evaluation ---")

truth_df = pd.read_csv(data_path)

if "killer_id" not in truth_df.columns:
    print("Error: 'killer_id' column not found in truth file. Cannot evaluate.")
    exit()

truth_subset = truth_df[["incident_id", "killer_id", "split"]]

merged_df = pd.merge(submission_df, truth_subset, on="incident_id", how="inner")
eval_df = merged_df[merged_df["split"].isin(["VAL", "TEST"])].copy()
eval_df = eval_df.dropna(subset=["killer_id"])


y_true = eval_df["killer_id"].values
y_pred = eval_df["predicted_killer"].values
# Metrics
acc = accuracy_score(y_true, y_pred)
f1_weighted = f1_score(y_true, y_pred, average="weighted")

print(f"Total Evaluated Samples: {len(eval_df)}")
print(f"Ensemble Accuracy: {acc:.4f}")
print(f"Ensemble Weighted F1: {f1_weighted:.4f}")

print("\nClassification Report (Accuracy & F1 per Class):")
print(classification_report(y_true, y_pred, digits=4))
# ---------------------------------------------------------
# Plot Confusion Matrix
# ---------------------------------------------------------
cm = confusion_matrix(y_true, y_pred)
labels = sorted(list(set(y_true))) # Ετικέτες 1-8
fig, ax = plt.subplots(figsize=(8, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(values_format='d')
plt.title(f"Submission Confusion Matrix\nAcc: {acc:.4f} - F1: {f1_weighted:.4f}")
plt.tight_layout()
cm_path = plots_path / "Ensemble_Confusion_Matrix.png"
plt.savefig(cm_path, dpi=300)
