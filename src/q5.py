import numpy as np
import pandas as pd

import json

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

# SVMs are models that find the optimal separating hyperplane between 2 claases
# in a feature spaces.
# Margin -> optimal separating hyperplane, the one that creates the maximum distance
# between the classes (essentialy the distance between the closest points from each class to the decision boundary)
# These closest points are called support vectors and the only ones that matter for defining our boundary
# Decision Boundary -> the hyperplane that separates the classes (wx + b = 0)
# w -> normal vector to the hyperplane
# x -> input vector
# b -> bias term

# Support Vectors -> The data points that are closest to the decision boundary are those
#                     that define the position and orientation of the hyperplane
#                     These points are called support vectors because they "support" or define the margin of the classifier                                      

# We classify a new point based on which side of the hyperplane it falls ( 0 > or 0 <)
# What if our classifier is non-linear? (This case)
# This is where transformations come in. We transform our data into a higher dimensional space (e.g. x^2)
# Adding a third dimension z = x^2 + y^2 allows us to separate our classes with a horizontal plane (linearly sepearable in 3D)
# Problem: Computing in higher dimensions is expensive due to polynomial features
# Solution: Kernel Trick -> We only need a dot product between pairs of transformed data points [max. the sum of a_i - half times the sum of a_j * y_i * y_j * (dot product of x_i and x_j in transformed space phi)]
#                           Here, a -> Lagrange multipliers, y -> class labels, x -> data points, phi -> transformation function
# The trick is that we can compute the dot product in the higher dimensional space WITHOUT explicitly transforming the data points
# k(x, y) = phi(x), phi(y) we substitute this in the function above


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


# Opening Q4 JSON data
try:
    with open("q4_results.json", "r") as f:
        q4_data = json.load(f)
    
    sgd_acc = q4_data["accuracy"]
    diff = accuracy_val - sgd_acc

except FileNotFoundError:
    print("Run q4.py first to store results")


# ----------------------------------------------------------
# Q4.5 - Decision Boundaries Visualization and Overlay (Q3 & Q4 & Q5)
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
    degree=best_params.get('degree', 3),
    random_state=42
)

#print best_params to check if they are correct
svm_optimal.fit(x_pca_train, y_pca_train)


