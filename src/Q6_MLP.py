import pandas as pd
import numpy as np
from pathlib import Path
from keras import layers, models, callbacks
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import seaborn as sns

projectDir = Path(__file__).parent.parent
dataPath = projectDir / "data" / "data_encoded.csv"
plotsPath = projectDir / "plots"

data = pd.read_csv(str(dataPath))

# Drop columns
columns_to_delete = [
    "incident_id",
    "weapon_code",
    "scene_type",
    "weather"
]

data.drop(columns_to_delete, axis=1, inplace=True)

ix_train = np.array(data.index[data['split'] == "TRAIN"])
ix_dev = np.array(data.index[data['split'] == "VAL"])
ix_test = np.array(data.index[data['split'] == "TEST"])

data.drop('split', axis=1, inplace=True)

y_encoded = pd.get_dummies(data["killer_id"])  
y_train_onehot = y_encoded.iloc[ix_train].values
y_dev_onehot   = y_encoded.iloc[ix_dev].values
y_test_onehot  = y_encoded.iloc[ix_test].values

#set X and y
X = data.drop(columns=["killer_id"]).copy()
y = data["killer_id"]

# --------------------------------------------------
# MLP
# --------------------------------------------------

# best : 
# 12->12->8     :   Weighted F1 score: 0.948 Accuracy: 0.948
# 12->8->8      
# 24->8              
# 48->8              
class FC_MNIST(models.Model):
    def __init__(self):
        super(FC_MNIST, self).__init__()
        # Creating layers in the initializer
        self.fc2 = layers.Dense(units = 12, activation="relu")  # Hidden layer
        self.fc3 = layers.Dense(units = 12, activation="relu")  # Hidden layer
        self.fc4 = layers.Dense(units = 8, activation="softmax")  # Output layer

    def call(self, input_tensor):
        # Pass input_tensor through the layers sequentially
        x = self.fc2(input_tensor)
        x = self.fc3(x)
        return self.fc4(x)
    
# Create the input layer
input_layer = layers.Input(shape=(24,))
# Instantiate the custom model and use it on the input layer
model = FC_MNIST()(input_layer)
# Create a complete Keras model by specifying the inputs and outputs
model = models.Model(inputs=input_layer, outputs=model)
# Model summary
model.summary(expand_nested=True)
#encode y
y_onehot = pd.get_dummies(pd.Series(y[ix_train])).values 
y_dev_onehot = pd.get_dummies(pd.Series(y[ix_dev])).values
callback = callbacks.EarlyStopping(monitor="val_loss", patience = 30)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model_hist = model.fit(
    X.iloc[ix_train],
    y_onehot,
    epochs=200,
    validation_data=(X.iloc[ix_dev], y_dev_onehot),
    callbacks=[callback],
)

# --------------------------------------------------
# Calculate Weighted f1_score and accuracy_score
# --------------------------------------------------
y_pred_probs = model.predict(X.iloc[ix_test])
y_pred_int = y_pred_probs.argmax(axis=1)
y_true_int = y_test_onehot.argmax(axis=1)

base_score = f1_score(y_true_int, y_pred_int, average="weighted")
base_acc = accuracy_score(y_true_int, y_pred_int)

print(f"Weighted F1 score: {base_score:.3f}")
print(f"Accuracy: {base_acc:.3f}")

# --------------------------------------------------
# Create DataFrame from model history
# --------------------------------------------------
df = pd.DataFrame(
    {
        "Training Loss": model_hist.history["loss"],
        "Validation Loss": model_hist.history["val_loss"],
    }
)
# Plot with customizations
fig, ax = plt.subplots(figsize=(24, 8))
df.plot(ax=ax, style=["-s", "-o"], color=["#003cff", "#ff0000"])
# Add title and labels
ax.set_title("Training vs. Validation Loss", fontsize=16, fontweight="bold")
ax.set_xlabel("Epoch", fontsize=14)
ax.set_ylabel("Loss", fontsize=14)
# Adjust legend
ax.legend(title="Loss Type", loc="upper right", fontsize=12)
# Improve readability with gridlines
ax.grid(True, linestyle="--", alpha=0.6)
# Show plot
plt.tight_layout()
featuresPath = plotsPath / "Q6"
os.makedirs(os.path.join(featuresPath), exist_ok=True)
plt.savefig(str(featuresPath /"Classification_Training_vs_Validation_Loss_plot.png"), dpi=300, bbox_inches="tight")

# --------------------------------------------------
# Confusion_matrix_for_Neural_Network_Plot
# --------------------------------------------------
labels = y_encoded.columns.tolist()

y_pred = model.predict(X.iloc[ix_test]).argmax(axis=1)
y_true = y_test_onehot.argmax(axis=1)   

cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax, xticks_rotation=90)

ax.set_title("Confusion matrix for Neural Network")
ax.set_xlabel("Predicted categories")
ax.set_ylabel("Actual categories")

plt.savefig(str(featuresPath / "Confusion_matrix_for_Neural_Network_Plot.png"), dpi=300, bbox_inches="tight")

# --------------------------------------------------
#  Estimate how much the classifier relies on each feature 
# --------------------------------------------------

numeric_features = [
    'hour_float',
    'latitude', 
    'longitude', 
    'victim_age', 
    'temp_c', 
    'humidity', 
    'dist_precinct_km', 
    'pop_density', 
    'vic_gender'
    ]
encoded_features = [
    ['weapon_code_0', 'weapon_code_1', 'weapon_code_2', 'weapon_code_3', 'weapon_code_4', 'weapon_code_5'],
    ['scene_type_0', 'scene_type_1', 'scene_type_2', 'scene_type_3'],
    ['weather_0', 'weather_1', 'weather_2', 'weather_3', 'weather_4']
    ]

feature_accuracy = pd.DataFrame(columns=["Weighted_F1_score", "Accuracy"])

def predict_with_MLP(data):
    y_pred_probs = model.predict(X.iloc[ix_test])
    y_pred_int = y_pred_probs.argmax(axis=1)
    y_true_int = y_test_onehot.argmax(axis=1)

    score = f1_score(y_true_int, y_pred_int, average="weighted")
    acc = accuracy_score(y_true_int, y_pred_int)

    #return pd.DataFrame([{"Weighted_F1_score": score, "Accuracy": acc}])
    return score , acc

suffle_times = 5
temp_X = X
for j in numeric_features:
    accuracy = []
    for i in range(suffle_times):
        temp_X = X
        # randomly shuffle only the j-th column
        temp_X[j] = temp_X[j].sample(frac=1).reset_index(drop=True).reset_index(drop=True)#, random_state=42).reset_index(drop=True)

        score , acc = predict_with_MLP(temp_X)
        accuracy.append([score , acc])
    
    score = acc = 0
    for item in accuracy:
        score += float(item[0])
        acc += float(item[1])
    score = score/len(accuracy)
    acc = acc/len(accuracy)

    new_row = pd.DataFrame([{"Weighted_F1_score": score, "Accuracy": acc}])
    
    # add data to df
    feature_accuracy = pd.concat([feature_accuracy, new_row], ignore_index=True)

for j in encoded_features:
    accuracy = []
    for i in range(suffle_times):    
        temp_X = X
        # randomly shuffle encoded data
        temp_X[j] = temp_X[j].apply(lambda row: np.random.permutation(row), axis=1, result_type='expand')

        score , acc = predict_with_MLP(temp_X)
        accuracy.append([score , acc])
    
    score = acc = 0
    for item in accuracy:
        score += float(item[0])
        acc += float(item[1])
    score = score/len(accuracy)
    acc = acc/len(accuracy)
            
    new_row = pd.DataFrame([{"Weighted_F1_score": score, "Accuracy": acc}])
    
    # add data to df
    feature_accuracy = pd.concat([feature_accuracy, new_row], ignore_index=True)

feature_accuracy["Weighted_F1_score"] = base_score - feature_accuracy["Weighted_F1_score"]
feature_accuracy["Accuracy"] = base_score - feature_accuracy["Accuracy"] 

catergorical_names = [
    "weapon_code",
    "scene_type",
    "weather"
]

features_names = numeric_features + catergorical_names
feature_accuracy.index = pd.Index(features_names, dtype="category")

feature_accuracy = feature_accuracy.sort_values(by="Weighted_F1_score", ascending=False)

# Round up values to 4 digits
feature_accuracy["Weighted_F1_score"] = feature_accuracy["Weighted_F1_score"].round(4)
feature_accuracy["Accuracy"] = feature_accuracy["Accuracy"].round(4)

# --------------------------------------------------
#  Rank and plot a bar chart of the top 5 most important features
# --------------------------------------------------
print(feature_accuracy)

colors = [
    "red" if category in catergorical_names else "blue"
    for category in feature_accuracy.index
]

fig, ax = plt.subplots(figsize=(8, 6))
bars = plt.bar(feature_accuracy.index, feature_accuracy["Accuracy"], color=colors)

plt.xlabel("Categories")
plt.ylabel("Accuracy")
plt.title("Accuracy Ranking (large positive indicates that the feature is crucial for correct classification)")
plt.xticks(rotation=45)

ax.grid(True, linestyle="--", alpha=0.6)
# Προσθήκη τιμών πάνω από τις μπάρες
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.4f}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig(str(featuresPath / "Accuracy Ranking.png"), dpi=300, bbox_inches="tight")
