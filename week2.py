import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("Iris.csv")

df = df.drop("Id", axis=1)

# Checked for missing values
print(df.isnull().sum())

# Removed duplicate records
df = df.drop_duplicates()

# Checked the different flower species
print(df["Species"].unique())

# Features
X = df.drop("Species", axis=1)

# Label
y = df["Species"]

print("\nFeatures:")
print(X.head())
print("\nLabels:")
print(y.head())


# WEEK-2 (Analysis and Visualization)

# Histogram 
df.hist(figsize=(10, 8))
plt.suptitle("Distribution of Iris Features")
plt.show()

# Different colors to each flower species
colors = {
    "Iris-setosa": "red",
    "Iris-versicolor": "green",
    "Iris-virginica": "blue"
}

# Scatter plot for Sepal Length vs Sepal Width
plt.figure(figsize=(8, 6))

# Plotted each flower species separately
for species in df["Species"].unique():
    subset = df[df["Species"] == species]

    plt.scatter(
        subset["SepalLengthCm"],
        subset["SepalWidthCm"],
        color=colors[species],
        label=species
    )

plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Sepal Length vs Sepal Width")
plt.legend()
plt.show()


# Split the data into training(80%) and testing data(20%)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Logistic Regression model
model = LogisticRegression(max_iter=200)

# Trained the model 
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#Performance metrics 

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Created a new flower 
new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=X.columns
)

# Predicted the species of the new flower
prediction = model.predict(new_flower)
print("\nPredicted Species:", prediction[0])
