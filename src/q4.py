import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from pathlib import Path
import os

# ----------------------------------------------------------
# Q4. Multiclass Linear Classifier
# ----------------------------------------------------------

# Linear Classifier: f(x) = Wx + b
# Predicted killer for incident i: c_i = argmax_k x f_k(x_i) Essentially, c is class k which returned the greatest f_k(x_i)
# f_k(x_i): score vector that the model gives us for component k

# We want to train this linear network with sum-of-squared errors and one hot targets so that, 
# we find the best W and b that minimize average loss among ALL training samples

# Techniques to know the proper valuew for W and b:
# 1. One-hot encoding - Converts categorical data into a numerical format that algorithms can understand
# e.g. crime type: theft -> 1
# But not just any number is assigned. This encoding creates new binary columns for each unique category

# 2. Sum of Squared Errors Loss Function - Measures the difference between predicted values by the model and actual target values, for one sample 
# How: we count the squared differences between predicted and actual values
# Formula: L(W, b) = sum_k (y_k - f_k(x_i))^2, where:
# y_k: actual target value for class k (1 or 0 due to one-hot encoding)
# f_k(x_i): predicted score for class k from the model
# End goal: 
# The score of the actual target class should be as close to 1 as possible, 
# while scores for other classes should be close to 0.
# The smaller the L the better the prediction of the model
# e.g. y_2 = 1 but the model returns f_2(x) = 0.2, we have a huge error since (1-0.2)^2 = 0.64

# 3. Gradient Descent - We calculate the slope of the loss function with respect to W and b
# This shows the "correct" direction in which W and b need to be adjusted to minimize the loss

# 1st Sub-question - Steps:

# Data Preprocessing
# 1. Split TRAIN and VAL using split column
# 2. One-Hot encoding for categorical features
# 3. Initialize W and b (randomly, done by LogisticRegression)

# Model and Loss Function
# 1. Multiclass Linear Classifier: f(x) = Wx + b  -> SGDCLassifier or LogisticRegression
# 2. Sum of Squared Errors: L(W, b) = sum_k (y_k - f_k(x_i))^2, One-hot and targets
# 3. Regularization L2 to prevent overfitting: The sum of the squares of all W values, -> In LogisitcRegression "penalty='l2'"
#    Applied to W, big values in W make the decision boundary too complex
# Total Loss Function = SSE + slope lamda x L2 Regularization

# Training and Optimization
# 1. Gradient Descent to update W and b
# 2. Train Using TRAIN set

# Choose Hyperparameter lamda -> GridSearchCV
# 1. Choose different lamda values e.g. {0.001,0.01,0.1,1}
# 2. For each lamda, train the model on the TRAIN set (for each lamda we start from scratch) 
# 3. Evaluate the accuracy of each trained model described above using the VAL set
# 4. Select the lamda that gives the highest accuracy on the VAL set -> GridSearchCV "cv"= 'validation_splitter''

# Last Calculations -> best_model.fit(X_train, y_train)
# 1. Prediction: use the W and b found using the best lamda to find the final predicition k (k for which argmax returns the greatest value) 
# 2. Using TEST set, to evaluate the final accuracy of the model


# 2nd subquestion - Steps:

# Accuracy Matrix Calculation -> metrics.accuracy_score(y_true, y_pred)
# 1. Taking the best model (with W and b returned from lamda_best) from the previous subquestion,
#    we compare with q3 -> GrifSearchCV "best_estimator_"
# 2. Use argmax on VAL set
# 3. Compare the predictions c with the actual target values (killer_id) on VAL set
#    Accuracy_VAL = (Number of correct predictions on VAL) / (Total Samples)

# Confusion Matrix Calculation -> metrics.confusion_matrix(y_true, y_pred)
# 1. SxS array, S being the number of unique classes (killer_id?)
# 2. Confusion_VAL = prediction results from VAL set
# Confusion matrix is a way to visualize the performance of the algorithm. Each (i,j) 
# shows how many samples from i class were predicted as j class

# Comparison with Q3
# Comment on their differences in terms of accuracy and confusion matrix


# 3rd subquestion - Steps:

# Preprocessing Dummy Data for PCA - Q3
# 1. sklearn.decomposition.PCA.transform
# 2. numpy.meshgrid() to create a grid of points covering the 2D PCA space

# Calculation of Linear Decision Boundaries
# 1. best_linear_model.predict() array that contains predicted class for each grid point of the meshgrid

# Calculation of Non-Linear Decision Boundaries (Q3)
# 1. Dummy data for Q3 - contains non-linear boundaries -> sklearn.naive_bayes.GaussianNB()
# 2. Array array containing predicted class for each grid point of the meshgrid (non-linear case? this time) nb_model.predict()

# Visualition and Composition
# 1. Visualize Q4 decision boundary plt.contourf(..., Z_linear, alpha=0.3)
# 2. Visualize Q3 non-linear decision boundary plt.contourf(..., Z_NB, alpha=0.1, colors='gray')
# 3. Visualize data points plt.scatter(X_PCA[:, 0], X_PCA[:, 1], c=y_train)


# ----------------------------------------------------------
# Q4.1 - Data Preprocessing
# ----------------------------------------------------------

project_dir = Path(__file__).parent.parent

# Load one-hot encoded data
data_path = project_dir / "data" / "data_encoded.csv"

data = pd.read_csv(str(data_path))

# f(x) = Wx + b
# Preparing input features x and target labels y

y_labels = data["killer_id"].values
y_onehot = pd.get_dummies(y_labels).values # target values in one-hot encoding

# Labels to drop 
drop_cols = ["incident_id", "split", "killer_id", "weapon_code", "scene_type", "weather"]
x = data.drop(columns=drop_cols).values

# Filtering out only TRAIN and VAL data 
train_data = data["split"] == "TRAIN"
val_data = data["split"] == "VAL"

# Final arrays to use

# For training
x_train = x[train_data.values]
y_train = y_labels[train_data.values]

# For Accuracy and Confusion matrices
x_val = x[val_data.values]
y_val = y_labels[val_data.values]

# ----------------------------------------------------------
# Q4.2 - Hyperparameters Tuning
# ----------------------------------------------------------

# Finding the optimal C
C = [0.001, 0.01, 0.1, 1, 10, 100]
best_C = None
best_accuracy = 0.0
res = []

for c in C:
    alpha = 1 / c  

    base = SGDClassifier(
        alpha = alpha, 
        loss = "log_loss",
        penalty = "l2",
        max_iter = 1000,
        random_state = 42
    )

    model = OneVsRestClassifier(base)

    model.fit(x_train, y_train)
    y_val_pred = model.predict(x_val)
    accuracy = accuracy_score(y_val, y_val_pred)

    res.append({"C": c, "Accuracy" : accuracy})

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_C = c

# Printing Results (Optional)
print("Hyperparameter Tuning Results")
for r in res:
    print(f"C = {r['C']: < 6}: Accuracy = {r['Accuracy']:.4f}")

print("\nBest Hyperparameter: ")
print(f"Best C: {best_C}")
print(f"Best Accuracy: {best_accuracy:.4f}")
print("\n")