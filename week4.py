import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Used for data vizualization
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df = pd.read_csv("Iris.csv")

# Dropping the "Id" column 
df = df.drop("Id", axis=1)

print(df.head())

print(df.isnull().sum())

# Removing duplicate rows
df = df.drop_duplicates()

# Checking how many flowers are present in each species
print(df["Species"].value_counts())

# Features
X = df[[
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]]

# Species
y = df["Species"]

# Splitting the data into training(80%) and testing parts(20%)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#Logistic Regression model
model = LogisticRegression(max_iter=200)

# Training the model 
model.fit(X_train, y_train)
#Predictions
y_pred = model.predict(X_test)

#Evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))


# testing with new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(new_flower)
print("Predicted Species:", prediction[0])

# Checking the distribution of Sepal Length
plt.figure(figsize=(6, 4))
plt.hist(df["SepalLengthCm"])
plt.title("Sepal Length Distribution")
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")
plt.show()


# Checking the distribution of Sepal Width
plt.figure(figsize=(6, 4))
plt.hist(df["SepalWidthCm"])
plt.title("Sepal Width Distribution")
plt.xlabel("Sepal Width")
plt.ylabel("Frequency")
plt.show()


# Checking the distribution of Petal Length
plt.figure(figsize=(6, 4))
plt.hist(df["PetalLengthCm"])
plt.title("Petal Length Distribution")
plt.xlabel("Petal Length")
plt.ylabel("Frequency")
plt.show()

# Checking the distribution of Petal Width
plt.figure(figsize=(6, 4))
plt.hist(df["PetalWidthCm"])
plt.title("Petal Width Distribution")
plt.xlabel("Petal Width")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=df,
    x="PetalLengthCm",
    y="PetalWidthCm",
    hue="Species"
)
plt.title("Petal Length vs Petal Width")
plt.show()

# Pairplot ( It helps to understand the relationshio btw all the features and how
# different species are grouped together) 
sns.pairplot(df, hue="Species")
plt.suptitle("Iris Feature Relationships", y=1.02)
plt.show()


# Checking how the numerical features are related
correlation = df.corr(numeric_only=True)

plt.figure(figsize=(8, 5))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Feature Correlation Heatmap")
plt.show()


# List of all the numerical features
features = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]

# Boxplots 
for feature in features:
    plt.figure(figsize=(7, 4))

    sns.boxplot(
        x="Species",
        y=feature,
        data=df
    )

    plt.title(feature + " by Species")
    plt.show()


# Comparing Sepal Length and Sepal Width
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
# The species are more clearly separated here.
plt.figure(figsize=(7, 5))

sns.scatterplot(
    data=df,
    x="PetalLengthCm",
    y="PetalWidthCm",
    hue="Species"
)

plt.title("Petal Length vs Petal Width")
plt.show()

#Observation 
# I observed that the petal features separate the Iris species more clearly than the sepal features. 
#From pairplot, Iris-setosa is easily distinguishable, but the versicolor and virginica have some overlap.

