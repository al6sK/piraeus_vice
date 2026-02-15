import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt



# ----------------------------------------------------------
# Q4. Non-linear SVM Classifier
# ----------------------------------------------------------

# ----------------------------------------------------------
# Q4.1 - Data Preprocessing
# ----------------------------------------------------------

projectDir = Path(__file__).parent.parent
dataPath = projectDir / "data" / "crimes.csv"
data = pd.read_csv(str(dataPath))

# Features and labels
y_labels = data["killer_id"].values

# Columns that are not features to be used for training
drop_cols = ["incident_id", "split", "killer_id"]

# If other categorical features that have been one-hot encoded are present, they should also be dropped
x = data.drop(columns=drop_cols).values

# Split data into TRAIN and VAL sets
train_data = data["split"] == "TRAIN"
val_data = data["split"] == "VAL"

x_train = x[train_data]
y_train = y_labels[train_data]

x_val = x[val_data]
y_val = y_labels[val_data]


# ----------------------------------------------------------
# Q4.2 - Scaling the data - Necessary for SVM
# ----------------------------------------------------------
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

# ----------------------------------------------------------
# Q4.3 - Preparation for GridDearchCV for VAL set Tuning
# ----------------------------------------------------------

x_combined = np.vstack((x_train, x_val))
y_combined = np.concatenate((y_train, y_val))

test_fold = np.zeros(x_combined.shape[0])
test_fold[:len(x_train)] = -1 
ps = PredefinedSplit(test_fold=test_fold)

params = [
    {
        'kernel' : ['rbf'],
        'C' : [0.1, 1, 10, 100],
        'gamma' : [0.001, 0.01, 0.1, 1]
    },
    {
        'kernel' : ['poly'],
        'C' : [0.1, 1, 10, 100],
        'coef0' : [0, 1, 10],
        'degree' : [2]
        }
]


# ----------------------------------------------------------
# Q4.4 - GridSearchCV 
# ----------------------------------------------------------

model = SVC(decision_function_shape='ovr', random_state=42)

grid_search = GridSearchCV(
    estimator = model,
    param_grid = params,
    scoring = 'accuracy',
    cv = ps,
    n_jobs = -1,
    verbose = 1
)

grid_search.fit(x_combined, y_combined)

# ----------------------------------------------------------
# Q4.5 - Final Evaluation of the best model using the VAL set
# ----------------------------------------------------------

final_model = grid_search.best_estimator_

best_params = grid_search.best_params_
best_score = grid_search.best_score_

y_val_pred = final_model.predict(x_val)

accuracy_val = accuracy_score(y_val, y_val_pred)
conf_mtrx = confusion_matrix(y_val, y_val_pred)


print(f"Analytical Report of SVM\n")
print(classification_report(y_val, y_val_pred))

if best_params['kernel'] == 'rbf':
    print(f"Kernel: {best_params['kernel']}")
elif best_params['kernel'] == 'poly':
    print(f"Kernel: {best_params['kernel']}")
    print(f"Best coef: {best_params['coef0']}")
    print(f"Degree: {best_params['degree']}")

print(f"\nFinal Model Accuracy on VAL set: {best_score:.4f}")
print(f"SVM VAL Accuracy: {accuracy_val:.4f}")

print("Confusion Matrix:") 
print(conf_mtrx)


# # Opening Q4 JSON data
# try:
#     with open("q4_results.json", "r") as f:
#         q4_data = json.load(f)
    
#     sgd_acc = q4_data["accuracy"]
#     diff = accuracy_val - sgd_acc

# except FileNotFoundError:
#     print("Run q4.py first to store results")


# ----------------------------------------------------------
# Q4.5 - Decision Regions & Support Vectors Visualization 
# ----------------------------------------------------------

# PCA Visualization
pca = PCA(n_components=2)
x_scaled = scaler.fit_transform(x)
x_pca = pca.fit_transform(x_scaled)

x_pca_train = x_pca[train_data.values]
y_pca_train = y_labels[train_data.values]

x_pca_val = x_pca[val_data.values]
y_pca_val = y_labels[val_data.values]

# SVM model using the optimal parameters
svm_optimal = SVC(
    kernel=best_params['kernel'],
    C=best_params['C'],
    gamma=best_params.get('gamma', 'scale'), # In case of rbf kernel
    coef0=best_params.get('coef0', 0),
    degree=best_params.get('degree', 2),
    random_state=42
)

#print best_params to check if they are correct

# Training the SVM on  2D PCA data
svm_optimal.fit(x_pca_train, y_pca_train)

# Boundary Visualization

# Mesh Creation
h = 0.02  # step size in the mesh
x_min, x_max = x_pca[:, 0].min() - 1, x_pca[:, 0].max() + 1
y_min, y_max = x_pca[:, 1].min() - 1, x_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

z = svm_optimal.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)

plt.figure(figsize=(10, 7))

plt.contourf(xx, yy, z, alpha=0.2, cmap='tab10')

# Data points
scatter = plt.scatter(
    x_pca[:, 0], x_pca[:, 1],
    c=y_labels,
    edgecolor='k',
    s = 50,
    cmap='tab10'
)

# Support Vectors Visualization
svm_vectors = svm_optimal.support_vectors_
plt.scatter(svm_vectors[:, 0], svm_vectors[:, 1], 
            s = 100, 
            facecolors = 'none', 
            edgecolors = 'black', 
            linewidths = 1.5, 
            label = 'Support Vectors'
            )

plt.title(f"Q5. SVM Decision Regions and Support Vectors - Kernel: {best_params['kernel']}, Degree: {best_params.get('degree', 2)}")

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")   
plt.legend(loc='upper right')

plt.show()