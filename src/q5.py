import numpy as np
import pandas as pd

import os
from pathlib import Path

from sklearn.preprocessing import StandardScaler

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

projectDir = Path(__file__).parent.parent

dataPath = projectDir / "data" / "crimes.csv"
plotsPath = projectDir / "plots"

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

# Scaling before passing as input to SVM
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)


