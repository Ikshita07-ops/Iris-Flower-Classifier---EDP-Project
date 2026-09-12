import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Used for splitting data, scaling features and evaluating the model
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Reading the Iris dataset
df = pd.read_csv("Iris.csv")

# Removing the Id column
df = df.drop("Id", axis=1)

print(df.head())

# Checking for missing values
print(df.isnull().sum())

# Removing duplicate rows
df = df.drop_duplicates()

# Checking the number of flowers in each species
print(df["Species"].value_counts())

X = df[[
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]]


y = df["Species"]

# Splitting the data into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scaling the features 
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# Trying different values of K 
k_values = [1, 3, 5, 7, 9]

results = {}

for k in k_values:

    # Creating the KNN model
    model = KNeighborsClassifier(n_neighbors=k)

    # Training the model
    model.fit(X_train, y_train)

    # Making predictions on test data
    y_pred = model.predict(X_test)

    # Calculating the accuracy
    accuracy = accuracy_score(y_test, y_pred)

    results[k] = accuracy

    print("\nK =", k)
    print("Accuracy:", accuracy)

# K value with the highest accuracy
best_k = max(results, key=results.get)

print("\nBest K value:", best_k)
print("Best Accuracy:", results[best_k])

model = KNeighborsClassifier(n_neighbors=best_k)

# Training the final model
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nFinal KNN Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

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
plt.title("KNN Confusion Matrix")
plt.show()

plt.figure(figsize=(6, 4))

plt.plot(
    list(results.keys()),
    list(results.values()),
    marker="o"
)

plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("K Value vs Accuracy")
plt.show()