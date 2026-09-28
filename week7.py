import pandas as pd
from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load and prepare the Iris dataset
df = pd.read_csv(Path(__file__).resolve().parent / "Iris.csv")
df = df.drop(columns=["Id"], errors="ignore")
df = df.drop_duplicates()

X = df[
    ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
]
y = df["Species"]

# Use the same training and test sets for both models
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Use stratified cross-validation during tuning
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# KNN pipeline scales features before training
knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier())
])

knn_parameters = {
    "classifier__n_neighbors": [1, 3, 5, 7, 9, 11, 13],
    "classifier__weights": ["uniform", "distance"],
    "classifier__p": [1, 2]
}

knn_search = GridSearchCV(
    estimator=knn_pipeline,
    param_grid=knn_parameters,
    cv=cv,
    scoring="accuracy"
)
knn_search.fit(X_train, y_train)

# Tune the Decision Tree
tree = DecisionTreeClassifier(random_state=42)

tree_parameters = {
    "max_depth": [None, 2, 3, 4, 5, 6, 8],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"]
}

tree_search = GridSearchCV(
    estimator=tree,
    param_grid=tree_parameters,
    cv=cv,
    scoring="accuracy"
)
tree_search.fit(X_train, y_train)

# Compare the best cross-validation results
searches = {
    "KNN": knn_search,
    "Decision Tree": tree_search
}

print("Tuning results")
print("-" * 60)

for name, search in searches.items():
    print(f"\n{name}")
    print(f"Best cross-validation accuracy: {search.best_score_:.4f}")
    print(f"Best parameters: {search.best_params_}")

# Select the model with the highest cross-validation accuracy
best_name, best_search = max(
    searches.items(),
    key=lambda item: item[1].best_score_
)

# Evaluate the selected model on the held-out test set
best_model = best_search.best_estimator_
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print("\nBest model:", best_name)
print(f"Test accuracy: {test_accuracy:.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))