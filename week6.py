import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("Iris.csv")

df = df.drop("Id", axis=1)

print(df.head())

print(df.isnull().sum())

df = df.drop_duplicates()

print(df["Species"].value_counts())

X = df[[
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]]

y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Creating the decision tree model
model = DecisionTreeClassifier(random_state=42)

# Training the model
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Calculating the model accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nDecision Tree Accuracy:")
print(accuracy)

# Showing precision, recall and F1 score
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Creating the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Displaying the confusion matrix
plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Decision Tree Confusion Matrix")
plt.show()

# Drawing the decision tree
plt.figure(figsize=(15, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=model.classes_,
    filled=True
)

plt.title("Decision Tree for Iris Classification")
plt.show()