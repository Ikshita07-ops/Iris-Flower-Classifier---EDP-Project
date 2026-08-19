import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("Iris.csv")

df = df.drop("Id", axis=1)

print(df.head())
print(df.isnull().sum())

df = df.drop_duplicates()

print(df["Species"].value_counts())

X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = df["Species"]

plt.figure(figsize=(6, 4))
plt.hist(df["SepalLengthCm"])
plt.title("Sepal Length Distribution")
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(6, 4))
plt.hist(df["SepalWidthCm"])
plt.title("Sepal Width Distribution")
plt.xlabel("Sepal Width")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(6, 4))
plt.hist(df["PetalLengthCm"])
plt.title("Petal Length Distribution")
plt.xlabel("Petal Length")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(6, 4))
plt.hist(df["PetalWidthCm"])
plt.title("Petal Width Distribution")
plt.xlabel("Petal Width")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="PetalLengthCm", y="PetalWidthCm", hue="Species")
plt.title("Petal Length vs Petal Width")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

new_flower = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(new_flower)
print("Predicted Species:", prediction[0])


# WEEK 4 - DATA VISUALIZATION

# Visualizing relationships using pairplot 
sns.pairplot(df, hue="Species")
plt.suptitle("Iris Feature Relationships", y=1.02)
plt.show()

# Correlation between features using heatmap
correlation = df.corr(numeric_only=True)

plt.figure(figsize=(8, 5))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

features = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]

for feature in features:
    plt.figure(figsize=(7, 4))
    sns.boxplot(x="Species", y=feature, data=df)
    plt.title(feature + " by Species")
    plt.show()

# Comparing Sepal Length and Sepal Width .
plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=df,
    x="SepalLengthCm",
    y="SepalWidthCm",
    hue="Species"
)
plt.title("Sepal Length vs Sepal Width")
plt.show()

# Comparing Petal Length and Petal Width 
plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=df,
    x="PetalLengthCm",
    y="PetalWidthCm",
    hue="Species"
)
plt.title("Petal Length vs Petal Width")
plt.show()

# Display the conclusion from the visualization analysis.
print("""
Week 4 Conclusion:
Petal Length and Petal Width show strong relationships with
Iris species. Iris-setosa is clearly separated from the other
species, while Iris-versicolor and Iris-virginica show some overlap.
Petal features are therefore useful for classification.
""")